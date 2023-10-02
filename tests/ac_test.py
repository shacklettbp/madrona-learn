import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from jax.experimental import checkify
import optax

import math
from functools import partial
from time import time

from madrona_learn.models import MLP
from madrona_learn.rnn import LSTM
from madrona_learn.moving_avg import EMANormalizer
from madrona_learn.actor_critic import *
from madrona_learn.models import *
from madrona_learn.train_state import PolicyTrainState, HyperParams

num_worlds = 16384
num_iters = 1000
num_policies = 32

def assert_valid_input(tensor):
    checkify.check(jnp.isnan(tensor).any() == False, "NaN!")
    checkify.check(jnp.isinf(tensor).any() == False, "Inf!")

class ProcessObsCommon(nn.Module):
    num_lidar_samples: int

    @nn.compact
    def __call__(self,
                 self_obs,
                 teammate_obs,
                 opponent_obs,
                 lidar,
                 alive,
                 opponent_masks,
                 train):
        assert_valid_input(self_obs)
        assert_valid_input(teammate_obs)
        assert_valid_input(opponent_obs)
        assert_valid_input(lidar)
        assert_valid_input(alive)
        assert_valid_input(opponent_masks)

        lidar_normalized = EMANormalizer(0.99999)('normalize', train, lidar)

        #lidar_processed = nn.Conv(
        #        features=1,
        #        kernel_size=(self.num_lidar_samples,),
        #        padding='CIRCULAR',
        #    )(jnp.expand_dims(lidar_normalized, axis=-1))
        

        #lidar_processed = lax.stop_gradient(lidar_processed.squeeze(axis=-1))
        lidar_processed = lidar_normalized

        return (self_obs, teammate_obs, opponent_obs, lidar_processed,
                alive, opponent_masks)


class ActorNet(nn.Module):
    num_mlp_channels: int

    @nn.compact
    def __call__(self, obs_tensors, train):
        self_obs, teammate_obs, opponent_obs, lidar_processed, alive, opponent_masks = \
            obs_tensors
        
        opponent_obs_masked = opponent_obs * opponent_masks

        flattened = jnp.concatenate([
            self_obs.reshape(self_obs.shape[0], -1),
            teammate_obs.reshape(teammate_obs.shape[0], -1),
            opponent_obs_masked.reshape(opponent_obs_masked.shape[0], -1),
            alive.reshape(alive.shape[0], -1),
        ], axis=-1)

        normalized = EMANormalizer(0.99999)('normalize', train, flattened)

        features = jnp.concatenate([normalized, lidar_processed], axis=1)
        features = lax.stop_gradient(features)

        return MLP(
                num_channels = self.num_mlp_channels,
                num_layers = 4,
            )(features)


class CriticNet(nn.Module):
    num_mlp_channels : int

    @nn.compact
    def __call__(self, obs_tensors, train):
        self_obs, teammate_obs, opponent_obs, lidar_processed, alive, opponent_masks = \
            obs_tensors
        
        flattened = jnp.concatenate([
            self_obs.reshape(self_obs.shape[0], -1),
            teammate_obs.reshape(teammate_obs.shape[0], -1),
            opponent_obs.reshape(opponent_obs.shape[0], -1),
            alive.reshape(alive.shape[0], -1),
        ], axis=1)

        normalized = EMANormalizer(0.99999)('normalize', train, flattened)

        features = jnp.concatenate([normalized, lidar_processed], axis=1)
        features = lax.stop_gradient(features)

        return MLP(
                num_channels = self.num_mlp_channels,
                num_layers = 4,
            )(features)

def make_policy(num_obs_features,
                num_channels,
                num_hidden_channels,
                num_lidar_samples):
    obs_common = ProcessObsCommon(num_lidar_samples)

    actor_encoder = RecurrentBackboneEncoder(
        net = ActorNet(num_channels),
        rnn = LSTM(
            hidden_channels = num_hidden_channels,
            num_layers = 1,
        ),
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = CriticNet(num_channels),
        rnn = LSTM(
            hidden_channels = num_hidden_channels,
            num_layers = 1,
        ),
    )

    backbone = BackboneSeparate(
        prefix = obs_common,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder,
    )

    return ActorCritic(
        backbone = backbone,
        actor = DenseLayerDiscreteActor(
            [4, 8, 5, 5, 2, 2],
        ),
        critic = DenseLayerCritic(),
    )

def policy_infer_rollout(state, prng_key, rnn_states, obs):
    _, _, _, rnn_states = state.apply_fn(
        {
            'params': state.params,
            'batch_stats': state.batch_stats,
        },
        prng_key,
        rnn_states,
        *obs,
        method='rollout',
    )

    return rnn_states

def policy_infer_indirect(states, assignment, input):
    prng_key, rnn_state, ob = input

    state = jax.tree_map(lambda x: x[assignment], states)

    rnn_state = jax.tree_map(lambda x: x[jnp.newaxis, ...], rnn_state)
    ob = jax.tree_map(lambda x: x[jnp.newaxis, ...], ob)

    rnn_state = policy_infer_rollout(state, prng_key, rnn_state, ob)

    rnn_state = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), rnn_state)

    return rnn_state

def policy_infer_sort(
        train_states, policy_assignments, prng_key, rnn_states, obs):
    batch_size = policy_assignments.shape[0]
    rebatch = lambda x: x.reshape(
        num_policies, batch_size // num_policies, *x.shape[1:])

    sort_idxs = jnp.argsort(policy_assignments)

    sorted_rnn_states = jax.tree_map(
        lambda x : rebatch(jnp.take(x, sort_idxs, 0)), rnn_states)
    sorted_obs = jax.tree_map(
        lambda x: rebatch(jnp.take(x, sort_idxs, 0)), obs)

    rollout_vec = jax.vmap(policy_infer_rollout, in_axes=(0, 0, 0, 0))
    sample_keys = random.split(prng_key, num_policies)

    sorted_rnn_states = rollout_vec(
        train_states, sample_keys, sorted_rnn_states, sorted_obs)

    # unbatch
    sorted_rnn_states = jax.tree_map(
        lambda x: x.reshape(batch_size, *x.shape[2:]), sorted_rnn_states)

    # unsort
    unsort_idxs = jnp.arange(batch_size)[sort_idxs]

    rnn_states = jax.tree_map(
        lambda x: jnp.take(x, unsort_idxs, 0), sorted_rnn_states)

    return rnn_states

def fake_rollout_iter(train_states, i, inputs):
    prng_key, rnn_states, obs = inputs

    prng_key, assign_key, sample_key = random.split(prng_key, 3)

    batch_size = obs[0].shape[0]
    policy_assignments = random.randint(
        assign_key, shape=(batch_size,), dtype=jnp.int32,
        minval=0, maxval=num_policies)

    #rollout_vec = jax.vmap(
    #    policy_infer_indirect, in_axes=(None, 0, 0))

    #sample_keys = random.split(sample_key, batch_size)
    #rnn_states = rollout_vec(train_states, policy_assignments,
    #    (sample_keys, rnn_states, obs))


    rnn_states = policy_infer_sort(
        train_states, policy_assignments, sample_key, rnn_states, obs)

    return prng_key, rnn_states, obs

def fake_rollout_loop(train_states, prng_key, rnn_states, *obs):
    prng_key, rnn_states, _ = lax.fori_loop(
        0, num_iters,
        partial(fake_rollout_iter, train_states),
        (prng_key, rnn_states, obs))

    return rnn_states

def setup_new_policy(policy, prng_key, fake_inputs):
    variables = policy.init(prng_key, *fake_inputs, method='rollout')

    params = variables['params']
    batch_stats = variables['batch_stats']

    return PolicyTrainState.create(
        apply_fn = policy.apply,
        params = params,
        tx = optax.adam(0.01),
        hyper_params = HyperParams(0, 0, 0),
        batch_stats = batch_stats,
        scheduler = None,
        scaler = None,
        value_normalize_fn = None,
        value_normalize_stats = {},
    )

def test():
    policy = make_policy(5, 256, 512, 32)

    prng_key = random.PRNGKey(5)

    obs = [
        jnp.zeros((num_worlds, 5), dtype=jnp.float32),
        jnp.zeros((num_worlds, 2), dtype=jnp.float32),
        jnp.zeros((num_worlds, 2), dtype=jnp.float32),
        jnp.zeros((num_worlds, 32), dtype=jnp.float32),
        jnp.zeros((num_worlds, 1), dtype=jnp.float32),
        jnp.zeros((num_worlds, 1), dtype=jnp.float32),
    ]

    dev = jax.devices()[0]
    print(dev)

    cur_rnn_states = policy.init_recurrent_state(
        num_worlds, dev, jnp.float32)

    prng_key, init_key = random.split(prng_key)

    init_keys = random.split(init_key, num_policies)

    setup_new_policies = jax.vmap(partial(setup_new_policy, policy),
                                  in_axes=(0, None))
    setup_new_policies = jax.jit(checkify.checkify(setup_new_policies))

    err, train_states = setup_new_policies(init_keys,
        (random.PRNGKey(0), cur_rnn_states, *obs))
    err.throw()

    print(jax.tree_util.tree_map(jnp.shape, train_states.params))
    print(jax.tree_util.tree_map(jnp.shape, train_states.batch_stats))

    rollout_loop_fn = jax.jit(checkify.checkify(fake_rollout_loop))

    rollout_loop_fn = rollout_loop_fn.lower(
        train_states, prng_key, cur_rnn_states, *obs)
    rollout_loop_fn = rollout_loop_fn.compile()

    start = time()

    err, rnn_states = rollout_loop_fn(
        train_states, prng_key, cur_rnn_states, *obs)
    err.throw()

    end = time()

    print(num_worlds * num_iters / (end - start))

if __name__ == "__main__":
    test()

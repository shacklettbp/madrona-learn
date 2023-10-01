import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

import math


from madrona_learn.models import MLP
from madrona_learn.rnn import LSTM
from madrona_learn.moving_avg import EMANormalizer
from madrona_learn.actor_critic import *
from madrona_learn.models import *

def assert_valid_input(tensor):
    assert(not jnp.isnan(tensor).any())
    assert(not jnp.isinf(tensor).any())

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

        lidar_processed = nn.Conv(
                features=1,
                kernel_size=(self.num_lidar_samples,),
                padding='CIRCULAR',
            )(jnp.expand_dims(lidar_normalized, axis=-1))

        lidar_processed = lax.stop_gradient(lidar_processed.squeeze(axis=-1))

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
        ], axis=1)

        normalized = EMANormalizer(0.99999)('normalize', train, flattened)

        features = jnp.concatenate([normalized, lidar_processed], axis=1)
        features = lax.stop_gradient(features)

        return MLP(
                num_channels = self.num_mlp_channels,
                num_layers = 2,
            )(features)


class CriticNet(nn.Module):
    num_mlp_channels : int

    @nn.compact
    def __call__(self, obs_tensors, train):
        self_obs, teammate_obs, opponent_obs, lidar_processed, alive, opponent_masks = \
            obs_tensors
        
        flattened = torch.cat([
            self_obs.reshape(self_obs.shape[0], -1),
            teammate_obs.reshape(teammate_obs.shape[0], -1),
            opponent_obs.reshape(opponent_obs.shape[0], -1),
            alive.reshape(alive.shape[0], -1),
        ], dim=1)

        normalized = EMANormalizer(0.99999)('normalize', train, flattened)

        features = jnp.concatenate([normalized, lidar_processed], axis=1)
        features = lax.stop_gradient(features)

        return MLP(
                num_channels = self.num_mlp_channels,
                num_layers = 2,
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

def fake_rollout_iter(policy, variables, step_key, rnn_states, obs):
    actions, log_probs, values, rnn_states = policy.apply(variables,
        step_key, rnn_states, *obs,
        method='rollout')

    return rnn_states

def fake_rollout_loop(policy, variables, prng_key, rnn_states, *obs):
    def iter(i, v):
        nonlocal prng_key, rnn_states

        prng_key, step_key = random.split(prng_key)
        rnn_states = fake_rollout_iter(policy, variables, step_key,
                                       rnn_states, obs)

    lax.fori_loop(0, 100, iter, None)

def test():
    policy = make_policy(5, 128, 256, 32)

    num_worlds = 1024

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

    cur_rnn_states = policy.init_recurrent_state(
        num_worlds, dev, jnp.float32)

    prng_key, init_key = random.split(prng_key)
    param_init_key, init_func_key = random.split(init_key)

    variables = policy.init(
        { 'params': param_init_key },
        init_func_key,
        cur_rnn_states,
        *obs,
        method='rollout')
        
    fake_rollout_loop(policy, variables, prng_key, cur_rnn_states, *obs)

if __name__ == "__main__":
    test()

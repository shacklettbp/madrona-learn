import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass

from madrona_learn import (
    ActorCritic, RecurrentBackboneEncoder, BackboneShared
)

from madrona_learn.train_state import (
    PolicyState,
)

from madrona_learn.rollouts import (
    RolloutConfig,
    RolloutState,
    rollout_loop,
    _init_matchmake_assignments,
    _compute_reorder_chunks,
    _compute_reorder_state,
)

def check_reorder_chunks(arr, P, C):
    B = arr.size // C + P - 1

    @jax.jit
    def reorder(arr):
        return _compute_reorder_chunks(arr, P, C, B)

    to_policy_idxs, to_sim_idxs = reorder(arr)

    policy_batches = jnp.take(
        arr, to_policy_idxs, mode='fill', fill_value=-1)

    assert jnp.sum(jnp.where(policy_batches != -1, 1, 0))
    arr_reconstructed = jnp.take(
        policy_batches.reshape(-1), to_sim_idxs, 0)

    #print(arr)
    #print(arr_reconstructed)
    #print(policy_batches)

    assert jnp.all(jnp.equal(arr, arr_reconstructed))

def test_reorder_chunks1():
    P = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 5, 3, 2, 1, 0, 3, 3])
    check_reorder_chunks(arr, P, C)

def test_reorder_chunks2():
    P = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 4, 5, 2, 1, 0, 3])
    check_reorder_chunks(arr, P, C)

def test_reorder_chunks3():
    P = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 4, 3, 2, 1, 0, 3])
    check_reorder_chunks(arr, P, C)

def test_reorder_chunks4():
    P = 6
    C = 4
    arr = jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5])
    arr = random.permutation(random.PRNGKey(5), arr, independent=True)
    check_reorder_chunks(arr, P, C)


def check_reorder(
    arr,
    num_current_policies,
    num_past_policies,
    policy_chunk_size_override,
):
    rollout_cfg = RolloutConfig.setup(
        num_current_policies = num_current_policies,
        num_past_policies = num_past_policies,
        num_teams = 2,
        team_size = 1,
        sim_batch_size = arr.size,
        self_play_portion = 0.0,
        cross_play_portion = 1.0,
        past_play_portion = 0.0,
        float_dtype = jnp.float16,
        policy_chunk_size_override = policy_chunk_size_override,
    )

    @jax.jit
    def reorder(arr):
        return _compute_reorder_state(arr, rollout_cfg)

    reorder_state = reorder(arr)

    policy_batches = jnp.take(
        arr, reorder_state.to_policy_idxs, mode='fill', fill_value=-1)

    assert jnp.sum(jnp.where(policy_batches != -1, 1, 0))
    arr_reconstructed = jnp.take(
        policy_batches.reshape(-1), reorder_state.to_sim_idxs, 0)

    #print(arr)
    #print(arr_reconstructed)
    #print(policy_batches)

    assert jnp.all(jnp.equal(arr, arr_reconstructed))

def test_reorder1():
    P = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 5, 3, 2, 1, 0, 3])
    check_reorder(arr, P, 0, C)


def setup_init_matchmake(
    num_current_policies,
    num_past_policies,
    num_teams,
    team_size,
    batch_size,
    self_play,
    cross_play,
    past_play,
    policy_chunk_size_override = 0
): 
    rollout_cfg = RolloutConfig.setup(
        num_current_policies = num_current_policies,
        num_past_policies = num_past_policies,
        num_teams = num_teams,
        team_size = team_size,
        sim_batch_size = batch_size,
        self_play_portion = self_play,
        cross_play_portion = cross_play,
        past_play_portion = past_play,
        float_dtype = jnp.float16,
        policy_chunk_size_override = policy_chunk_size_override,
    )

    @jax.jit
    def init_matchmake(rnd):
        return _init_matchmake_assignments(rnd, rollout_cfg)

    return init_matchmake(random.PRNGKey(7)).reshape(-1, num_teams, team_size)

def test_init_matchmake1():
    matchmake = setup_init_matchmake(
        num_current_policies = 4,
        num_past_policies = 0,
        num_teams = 1,
        team_size = 4,
        batch_size = 512,
        self_play = 1.0,
        cross_play = 0.0,
        past_play = 0.0,
    )

    print(matchmake)

def test_init_matchmake2():
    matchmake = setup_init_matchmake(
        num_current_policies = 4,
        num_past_policies = 3,
        num_teams = 2,
        team_size = 2,
        batch_size = 32,
        self_play = 0.0,
        cross_play = 0.5,
        past_play = 0.5,
    )

    print(matchmake[:4])
    print(matchmake[4:])


@dataclass(frozen=True)
class FakeActionDist:
    action: jax.Array

    def best(self):
        return self.action


class FakeNet(nn.Module):
    @nn.compact
    def __call__(self, inputs, train):
        bias = self.param('bias',
            jax.nn.initializers.constant(0), (), jnp.int32)

        return jnp.concatenate([
            inputs + bias,
            jnp.broadcast_to(bias[None, None], inputs.shape),
        ], axis=-1)


class FakeRNN(nn.Module):
    @nn.nowrap
    def init_recurrent_state(self, N):
        return jnp.zeros((N, 1), dtype=jnp.int32)

    @nn.nowrap
    def clear_recurrent_state(self, rnn_states, should_clear):
        return jnp.where(
            should_clear, jnp.zeros((), dtype=jnp.int32), rnn_states)

    def setup(self):
        pass

    def __call__(self, cur_hiddens, in_features, train):
        y = in_features[..., 0:1] + cur_hiddens
        y = jnp.concatenate([y, in_features[..., 1:2], cur_hiddens], axis=-1)

        new_hiddens = cur_hiddens + 1

        return y, new_hiddens


def check_rollout_loop(
    rnd,
    num_steps,
    episode_len,
    num_current_policies,
    num_past_policies,
    num_teams,
    team_size,
    batch_size,
    self_play,
    cross_play,
    past_play,
    policy_chunk_size_override = 0,
):
    rollout_cfg = RolloutConfig.setup(
        num_current_policies = num_current_policies,
        num_past_policies = num_past_policies,
        num_teams = num_teams,
        team_size = team_size,
        sim_batch_size = batch_size,
        self_play_portion = self_play,
        cross_play_portion = cross_play,
        past_play_portion = past_play,
        float_dtype = jnp.float16,
        policy_chunk_size_override = policy_chunk_size_override,
    )

    init_obs = jnp.arange(batch_size).reshape((batch_size, 1)) + 1

    init_sim_data = frozen_dict.freeze({
        'obs': init_obs,
        'rewards': jnp.zeros((batch_size, 1), dtype=jnp.float16),
        'dones': jnp.zeros((batch_size, 1), dtype=jnp.bool_),
        'actions': jnp.zeros((batch_size, 2), dtype=jnp.int32),
        'counter': jnp.zeros((batch_size, 1), dtype=jnp.int32),
    })

    def fake_sim(sim_data):
        sim_data = jax.tree_map(jnp.copy, sim_data)

        new_counter = sim_data['counter'] + 1
        new_dones = new_counter == episode_len

        new_counter %= episode_len

        return sim_data.copy({
            'obs': sim_data['actions'][..., 0:1] + 1,
            'rewards': sim_data['actions'][..., 0:1].astype(jnp.float16),
            'counter': new_counter,
            'dones': new_dones,
        })

    fake_backbone = BackboneShared(
        prefix = lambda x, train: x,
        encoder = RecurrentBackboneEncoder(
            net = FakeNet(),
            rnn = FakeRNN(),
        ),
    )

    def fake_actor(features, train):
        return FakeActionDist(features[..., 0:2])

    def fake_critic(features, train):
        return features[..., 2:3] + 1

    policy = ActorCritic(
        backbone = fake_backbone,
        actor = fake_actor,
        critic = fake_critic,
    )

    rnd, rnd_rollout = random.split(rnd)

    @jax.jit
    def init_rollout_state():
        rnn_states = policy.init_recurrent_state(batch_size)

        rnn_states = jax.tree_map(
            lambda x: jnp.arange(x.shape[0]).reshape(-1, 1), rnn_states)

        return RolloutState.create(
            rollout_cfg = rollout_cfg,
            step_fn = fake_sim,
            prng_key = rnd_rollout,
            rnn_states = rnn_states,
            init_sim_data = init_sim_data,
        )

    rollout_state = init_rollout_state()
    
    def make_policy(policy_idx, init_rnd):
        variables = policy.init(
            init_rnd, random.PRNGKey(0), rollout_state.rnn_states,
            rollout_state.sim_data['obs'], 
            sample_actions = False,
            method='rollout')

        params = variables['params']
        params['backbone']['encoder']['net']['bias'] = jnp.array(
            policy_idx, dtype=jnp.int32)

        return PolicyState(
            apply_fn = policy.apply,
            rnn_reset_fn = policy.clear_recurrent_state,
            params = params,
            batch_stats = {}
        )

    rnd, rnd_init = random.split(rnd)

    make_policies = jax.jit(jax.vmap(make_policy))

    policy_states = make_policies(
        jnp.arange(num_current_policies),
        random.split(rnd_init, num_current_policies))

    def post_inference_cb(step_idx, policy_obs, policy_out,
                          reorder_state, rollout_store):
        obs, actions, values = reorder_state.to_sim(
            (policy_obs, policy_out['actions'], policy_out['values']))
        
        return rollout_store.copy({
            'obs': rollout_store['obs'].at[step_idx].set(obs),
            'actions': rollout_store['actions'].at[step_idx].set(actions),
            'values': rollout_store['values'].at[step_idx].set(values),
        })

    def post_step_cb(step_idx, dones, rewards, reorder_state, rollout_store):
        return rollout_store.copy({
            'rewards': rollout_store['rewards'].at[step_idx].set(rewards),
        })

    rollout_store = frozen_dict.freeze({
        'obs': jnp.zeros((num_steps, batch_size, 1), dtype=jnp.int32),
        'values': jnp.zeros((num_steps, batch_size, 1), dtype=jnp.int32),
        'actions': jnp.zeros((num_steps, batch_size, 2), dtype=jnp.int32),
        'rewards': jnp.zeros((num_steps, batch_size, 1), dtype=jnp.float16),
    })

    def rollout_loop_wrapper(rollout_state):
        return rollout_loop(
            rollout_state = rollout_state,
            policy_states = policy_states,
            rollout_cfg = rollout_cfg,
            num_steps = num_steps,
            post_inference_cb = post_inference_cb,
            post_step_cb = post_step_cb,
            cb_state = rollout_store,
            sample_actions = False,
            return_debug = False,
        )

    rollout_loop_wrapper = jax.jit(
        checkify.checkify(rollout_loop_wrapper),
        donate_argnums=0)

    err, (rollout_state, rollout_store) = rollout_loop_wrapper(rollout_state)
    err.throw()

    assert jnp.all(rollout_store['obs'][0] == init_obs)

    def compute_gt_rnn(step_idx):
        return jnp.arange(batch_size) + step_idx + 1

    compute_gt_rnn = jax.jit(jax.vmap(compute_gt_rnn))
    gt_rnn = compute_gt_rnn(jnp.arange(num_steps))

    assert jnp.all(gt_rnn == rollout_store['values'][..., 0])

    def compute_gt_out(step_idx, policy_idx):
        return (jnp.arange(batch_size) +
                (step_idx + 1) *
                (step_idx + 2 * (jnp.arange(batch_size) + 1) +
                 2 * policy_idx) // 2)

    compute_gt_out = jax.jit(jax.vmap(compute_gt_out))

    gt_out = compute_gt_out(
        jnp.arange(num_steps), rollout_store['actions'][..., 1])
    actual_out = rollout_store['actions'][..., 0]
    assert jnp.all(gt_out == actual_out)

    return rollout_state, rollout_store
    

def test_rollout_loop1():
    rollout_state, rollout_store = check_rollout_loop(
        random.PRNGKey(5),
        num_steps = 10,
        episode_len = 11,
        num_current_policies = 4,
        num_past_policies = 0,
        num_teams = 2,
        team_size = 2,
        batch_size = 32,
        self_play = 0.0,
        cross_play = 1.0,
        past_play = 0.0,
    )

    final_values = rollout_store['values'][-1, ...]
    final_rnn_states = rollout_state.rnn_states

    assert jnp.all(final_values == final_rnn_states)

def test_rollout_loop2():
    rollout_state, rollout_store = check_rollout_loop(
        random.PRNGKey(5),
        num_steps = 20,
        episode_len = 10,
        num_current_policies = 4,
        num_past_policies = 0,
        num_teams = 2,
        team_size = 2,
        batch_size = 32,
        self_play = 0.0,
        cross_play = 1.0,
        past_play = 0.0,
    )

#test_reorder_chunks1()
#test_reorder_chunks2()
#test_reorder_chunks3()
#test_reorder_chunks4()

#test_init_matchmake1()
#test_init_matchmake2()

test_rollout_loop1()
test_rollout_loop2()

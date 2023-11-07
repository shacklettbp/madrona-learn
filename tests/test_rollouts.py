import jax
import jax.numpy as jnp
from jax import lax, random
from jax.experimental import checkify
import flax

from madrona_learn.rollouts import (
    ActorCritic,
    RolloutConfig,
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
        total_batch_size = arr.size,
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
        total_batch_size = batch_size,
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


def check_end_to_end(
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
        total_batch_size = batch_size,
        self_play_portion = self_play,
        cross_play_portion = cross_play,
        past_play_portion = past_play,
        float_dtype = jnp.float16,
        policy_chunk_size_override = policy_chunk_size_override,
    )

    def fake_backbone():
        pass

    def fake_actor():
        pass

    def fake_critic():
        pass

    fake_sim_data = 0

    def fake_sim(sim_data):
        return sim_data

    fake_ac = ActorCritic(
        backbone = fake_backbone(),
        actor = fake_actor(),
        critic = fake_critic(),
    )

    @jax.jit
    def init_rollout_state():
        fake_rnn_states = fake_ac.init_recurrent_state(
            rollout_cfg.sim_batch_size)

        return RolloutState.create(
            rollout_cfg = rollout_cfg,
            step_fn = fake_sim,
            prng_key = random.PRNGKey(0),
            rnn_states = fake_rnn_states,
            init_sim_data = fake_sim_data,
        )

    rollout_state = init_rollout_state()


def test_end_to_end1():
    check_end_to_end(
        num_current_policies = 4,
        num_past_policies = 3,
        num_teams = 2,
        team_size = 2,
        batch_size = 32,
        self_play = 0.0,
        cross_play = 0.5,
        past_play = 0.5,
    )

test_reorder_chunks1()
test_reorder_chunks2()
test_reorder_chunks3()
test_reorder_chunks4()

#test_init_matchmake1()
test_init_matchmake2()

test_end_to_end1()

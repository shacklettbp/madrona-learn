import jax
import jax.numpy as jnp
from jax import lax, random
from jax.experimental import checkify

from madrona_learn.rollouts import (
    RolloutConfig,
    _init_matchmake_assignments,
    _compute_reorder_state,
)

def check_reorder(arr, M, C):
    num_chunks = arr.size // C + M - 1

    @jax.jit
    def reorder(arr):
        return _compute_reorder_state(arr, M, C, num_chunks)

    reorder_state = reorder(arr)

    policy_batches = jnp.take(
        arr, reorder_state.to_policy_idxs, mode='fill', fill_value=-1)

    assert jnp.sum(jnp.where(policy_batches != -1, 1, 0))
    arr_reconstructed = jnp.take(
        policy_batches.reshape(-1), reorder_state.to_sim_idxs, 0)

    #print(arr)
    #print(policy_batches)

    assert jnp.all(jnp.equal(arr, arr_reconstructed))


def test_reorder1():
    M = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 5, 3, 3])
    check_reorder(arr, M, C)

def test_reorder2():
    M = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 4, 5, 3, 3])
    check_reorder(arr, M, C)

def test_reorder3():
    M = 6
    C = 4
    arr = jnp.array([1, 1, 0, 0, 2, 2, 4, 3, 3, 3])
    check_reorder(arr, M, C)

def test_reorder4():
    M = 6
    C = 4
    arr = jnp.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    arr = random.permutation(random.PRNGKey(5), arr, independent=True)
    check_reorder(arr, M, C)


def setup_init_matchmake(
    num_current_policies,
    num_past_policies,
    num_teams,
    team_size,
    batch_size,
    self_play,
    cross_play,
    past_play,
    policy_batch_size_override = 0
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
        policy_batch_size_override = policy_batch_size_override,
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

#test_reorder1()
#test_reorder2()
#test_reorder3()
#test_reorder4()

#test_init_matchmake1()
test_init_matchmake2()

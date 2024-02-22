import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
import numpy as np
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional

from .cfg import TrainConfig, PBTConfig
from .train_state import PolicyState, PolicyTrainState, TrainStateManager

@dataclass(frozen=True)
class PBTMatchmakeConfig:
    num_current_policies: int
    num_past_policies: int
    total_num_policies: int
    num_teams: int
    team_size: int
    self_play_batch_size: int
    cross_play_batch_size: int
    past_play_batch_size: int
    static_play_batch_size: int
    num_cross_play_matches: int
    num_past_play_matches: int
    num_static_play_matches: int
    num_total_matches: int
    complex_matchmaking: bool 
    
    @staticmethod
    def setup(
        num_current_policies: int,
        num_past_policies: int,
        num_teams: int,
        team_size: int,
        sim_batch_size: int,
        self_play_portion: float,
        cross_play_portion: float,
        past_play_portion: float,
        static_play_portion: float,
    ):
        total_num_policies = num_current_policies + num_past_policies

        assert (self_play_portion + cross_play_portion +
            past_play_portion + static_play_portion == 1.0)

        self_play_batch_size = int(sim_batch_size * self_play_portion)
        cross_play_batch_size = int(sim_batch_size * cross_play_portion)
        past_play_batch_size = int(sim_batch_size * past_play_portion)
        static_play_batch_size = int(sim_batch_size * static_play_portion)

        assert (self_play_batch_size +
                cross_play_batch_size +
                past_play_batch_size +
                static_play_batch_size == sim_batch_size)

        agents_per_world = num_teams * team_size

        assert cross_play_batch_size % agents_per_world == 0
        assert past_play_batch_size % agents_per_world == 0
        assert static_play_batch_size % agents_per_world == 0

        num_cross_play_matches = cross_play_batch_size // agents_per_world
        num_past_play_matches = past_play_batch_size // agents_per_world
        num_static_play_matches = static_play_batch_size // agents_per_world

        num_total_matches = sim_batch_size // agents_per_world

        assert num_cross_play_matches % num_current_policies == 0
        assert num_past_play_matches % num_current_policies == 0
        assert num_static_play_matches % num_current_policies == 0
        
        complex_matchmaking = self_play_portion != 1.0

        return PBTMatchmakeConfig(
            num_current_policies = num_current_policies,
            num_past_policies = num_past_policies,
            total_num_policies = total_num_policies,
            num_teams = num_teams,
            team_size = team_size,
            self_play_batch_size = self_play_batch_size,
            cross_play_batch_size = cross_play_batch_size,
            past_play_batch_size = past_play_batch_size,
            static_play_batch_size = static_play_batch_size,
            num_cross_play_matches = num_cross_play_matches,
            num_past_play_matches = num_past_play_matches,
            num_static_play_matches = num_static_play_matches,
            num_total_matches = num_total_matches,
            complex_matchmaking = complex_matchmaking,
        )


class PBTLog:
    pass


def pbt_init_matchmaking(
    assign_rnd: random.PRNGKey,
    mm_cfg: PBTMatchmakeConfig,
    static_play_assignments: Optional[jax.Array],
):
    def self_play_assignments(batch_size):
        return jnp.repeat(
            jnp.arange(mm_cfg.num_current_policies),
            batch_size // mm_cfg.num_current_policies)

    def cross_play_opponents(rnd, base_assignments):
        num_matches = mm_cfg.num_cross_play_matches
        num_teams = mm_cfg.num_teams

        base_assignments = base_assignments.reshape(
            (num_matches, num_teams, mm_cfg.team_size))

        opponents = random.randint(rnd, (num_matches, num_teams - 1),
            0, mm_cfg.num_current_policies - 1)[..., None]

        opponents = jnp.where(opponents >= base_assignments[:, 0:1, 0:1],
            opponents + 1, opponents)

        return opponents
    
    def past_play_opponents(rnd):
        num_matches = mm_cfg.num_past_play_matches
        num_teams = mm_cfg.num_teams
        
        return random.randint(rnd, (num_matches, num_teams - 1),
            mm_cfg.num_current_policies,
            mm_cfg.num_current_policies + mm_cfg.num_past_policies)[
                ..., None]


    self_play_batch_size = mm_cfg.self_play_batch_size
    cross_play_batch_size = mm_cfg.cross_play_batch_size
    past_play_batch_size = mm_cfg.past_play_batch_size
    static_play_batch_size = mm_cfg.static_play_batch_size

    sub_assignments = []

    if self_play_batch_size > 0:
        self_assignments = self_play_assignments(self_play_batch_size)
        assert self_assignments.shape[0] == self_play_batch_size
        sub_assignments.append(self_assignments)

    if cross_play_batch_size > 0:
        assign_rnd, cross_rnd = random.split(assign_rnd)

        cross_assignments = self_play_assignments(cross_play_batch_size)
        assert cross_assignments.shape[0] == cross_play_batch_size

        cross_assignments = cross_assignments.reshape(
            mm_cfg.num_cross_play_matches, mm_cfg.num_teams,
            mm_cfg.team_size)

        cross_opponent_assignments = cross_play_opponents(
            cross_rnd, cross_assignments)

        cross_assignments = cross_assignments.at[:, 1:, :].set(
            cross_opponent_assignments)

        sub_assignments.append(cross_assignments.reshape(-1))

    if past_play_batch_size > 0:
        past_assignments = self_play_assignments(past_play_batch_size)
        assert past_assignments.shape[0] == past_play_batch_size
        past_assignments = past_assignments.reshape(
            mm_cfg.num_past_play_matches, mm_cfg.num_teams,
            mm_cfg.team_size)

        past_opponent_assignments = past_play_opponents(assign_rnd)

        past_assignments = past_assignments.at[:, 1:, :].set(
            past_opponent_assignments)

        sub_assignments.append(past_assignments.reshape(-1))

    if static_play_batch_size > 0:
        sub_assignments.append(static_play_assignments)

    policy_assignments = jnp.concatenate(sub_assignments, axis=0)
    return policy_assignments

def _cross_play_matchmake(
    assignments,
    policy_states,
    dones,
    assign_rnd,
    mm_cfg,
):
    num_matches = mm_cfg.num_cross_play_matches
    num_teams = mm_cfg.num_teams
    team_size = mm_cfg.team_size

    assignments = assignments.reshape(num_matches, num_teams, team_size)
    dones = dones.reshape(num_matches, num_teams, team_size)

    new_assignments = random.randint(assign_rnd, (num_matches, num_teams - 1),
        0, mm_cfg.num_current_policies - 1)

    new_assignments = jnp.where(new_assignments >= assignments[:, 0:1, 0],
        new_assignments + 1, new_assignments)

    new_assignments = jnp.where(
        dones[:, 1:, :], new_assignments[:, :, None], assignments[:, 1:, :])

    return (
        assignments.at[:, 1:, :].set(new_assignments).reshape(-1),
        policy_states)


def _past_play_matchmake(
    assignments,
    policy_states,
    dones,
    assign_rnd,
    mm_cfg,
):
    num_matches = mm_cfg.num_past_play_matches
    num_teams = mm_cfg.num_teams
    team_size = mm_cfg.team_size
    
    assignments = assignments.reshape(num_matches, num_teams, team_size)
    dones = dones.reshape(num_matches, num_teams, team_size)

    new_assignments = random.randint(assign_rnd, (num_matches, num_teams - 1),
        mm_cfg.num_current_policies,
        mm_cfg.num_current_policies + mm_cfg.num_past_policies)

    new_assignments = jnp.where(
        dones[:, 1:, :], new_assignments[:, :, None], assignments[:, 1:, :])

    return (
        assignments.at[:, 1:, :].set(new_assignments).reshape(-1),
        policy_states)


def _elo_expected_result(
    my_elo,
    opponent_elo,
):
    return 1 / (1 + 10 ** ((opponent_elo - my_elo) / 400))


def _update_fitness(
    assignments,
    policy_states,
    dones,
    match_results,
    mm_cfg,
):
    assert mm_cfg.num_teams > 1

    # FIXME
    if mm_cfg.num_teams != 2:
        return policy_states

    assignments = assignments.reshape(
        mm_cfg.num_total_matches, mm_cfg.num_teams,
        mm_cfg.team_size, 1)
    dones = dones.reshape(
        mm_cfg.num_total_matches, mm_cfg.num_teams,
        mm_cfg.team_size, 1)

    a_assignments = assignments[:, 0, 0, 0]
    b_assignments = assignments[:, 1, 0, 0]
    dones = dones[:, 0, 0, :]

    def update_elo(policy_idx, cur_fitness_score):
        @jax.vmap
        def compute_differences(raw_match_result, a_assignment, b_assignment, done):
            is_a = a_assignment == policy_idx
            is_b = b_assignment == policy_idx

            valid = jnp.logical_and(
                jnp.logical_and(
                    jnp.logical_or(is_a, is_b),
                    done,
                ),
                a_assignment != b_assignment,
            ).squeeze(axis=0)

            def compute_diff():
                a_score = policy_states.get_team_a_score_fn(
                    raw_match_result)

                a_elo = policy_states.fitness_score[a_assignment, 0]
                b_elo = policy_states.fitness_score[b_assignment, 0]

                my_score = jnp.where(is_b, 1 - a_score, a_score)

                my_elo = jnp.where(is_a, a_elo, b_elo)
                opponent_elo = jnp.where(is_a, b_elo, a_elo)

                expected_score = _elo_expected_result(my_elo, opponent_elo)
                diff = my_score - expected_score

                return diff

            def skip_diff():
                return jnp.zeros((), dtype=jnp.float32)

            return lax.cond(valid, compute_diff, skip_diff)

        diffs = compute_differences(match_results, a_assignments,
                                    b_assignments, dones)

        #expected_current_policy_matches = (
        #    (mm_cfg.num_cross_play_matches * 2 +
        #        mm_cfg.num_past_play_matches) /
        #    mm_cfg.num_current_policies)

        #if mm_cfg.num_past_policies > 0:
        #    expected_past_policy_matches = (
        #        mm_cfg.num_past_play_matches / mm_cfg.num_past_policies)

        #    current_reweight = (expected_past_policy_matches /
        #                        expected_current_policy_matches)
        #else:
        #    current_reweight = 1
        #K = jnp.where(policy_idx < mm_cfg.num_current_policies,
        #              current_reweight * K, K)

        K = jnp.array([8], dtype=jnp.float32)

        return cur_fitness_score + K * diffs.sum()
        #return jnp.clip(cur_fitness_score + K * diffs.sum(), a_min=100)

    new_fitness_scores = jax.vmap(update_elo)(
        jnp.arange(policy_states.fitness_score.shape[0]),
        policy_states.fitness_score)

    policy_states = policy_states.update(fitness_score=new_fitness_scores)

    return policy_states


def pbt_update_matchmaking(
    assignments,
    policy_states,
    dones,
    match_results,
    assign_rnd,
    mm_cfg,
):
    policy_states = _update_fitness(
        assignments, policy_states, dones, match_results, mm_cfg)

    cross_start = mm_cfg.self_play_batch_size
    cross_end = cross_start + mm_cfg.cross_play_batch_size
    
    past_start = cross_end
    past_end = past_start + mm_cfg.past_play_batch_size

    if mm_cfg.cross_play_batch_size > 0:
        assign_rnd, cross_rnd = random.split(assign_rnd)

        new_cross_assignments, policy_states = _cross_play_matchmake(
            assignments[cross_start:cross_end], policy_states,
            dones[cross_start:cross_end], cross_rnd, mm_cfg)

        assignments = assignments.at[cross_start:cross_end].set(
            new_cross_assignments)

    if mm_cfg.past_play_batch_size > 0:
        assign_rnd, past_rnd = random.split(assign_rnd)

        new_past_assignments, policy_states = _past_play_matchmake(
            assignments[past_start:past_end], policy_states,
            dones[past_start:past_end], past_rnd, mm_cfg)

        assignments = assignments.at[past_start:past_end].set(
            new_past_assignments)

    return assignments, policy_states, assign_rnd


def _rebase_elos(policy_states):
    # Rebase elo so average is 1500
    elos = policy_states.fitness_score[..., 0]
    elo_correction = jnp.mean(elos) - 1500

    return policy_states.update(
        fitness_score = policy_states.fitness_score - elo_correction,
    )


def _check_overwrite(cfg, policy_states, src_idx, dst_idx):
    src_elo = policy_states.fitness_score[src_idx, 0]
    dst_elo = policy_states.fitness_score[dst_idx, 0]

    src_expected_winrate = _elo_expected_result(src_elo, dst_elo)
    return src_expected_winrate >= cfg.pbt.policy_overwrite_threshold


def _pbt_cull_update(
    cfg: TrainConfig,
    train_state_mgr: TrainStateManager,
):
    policy_states = train_state_mgr.policy_states
    train_states = train_state_mgr.train_states
    pbt_rng = train_state_mgr.pbt_rng

    assert 2 * cfg.pbt.num_cull_policies < cfg.pbt.num_train_policies

    sort_idxs = jnp.argsort(policy_states.fitness_score[
        0:cfg.pbt.num_train_policies, 0])
    
    bottom_idxs = sort_idxs[:cfg.pbt.num_cull_policies]
    top_idxs = sort_idxs[-cfg.pbt.num_cull_policies:]

    @partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
    def cull_train_policy(policy_states, train_states,
                          mutate_rng, dst_idx, src_idx):
        def get_overwrite_policy():
            src_policy_state = jax.tree_map(
                lambda x: x[src_idx], policy_states)

            src_train_state = jax.tree_map(
                lambda x: x[src_idx], train_states)

            # Preserve original update_prng_key
            src_train_state = src_train_state.update(
                update_prng_key = train_states.update_prng_key[dst_idx],
            )

            if src_policy_state.reward_hyper_params != None:
                reward_hyper_params = src_policy_state.mutate_reward_hyper_params(
                    mutate_rng, src_policy_state.reward_hyper_params)

                src_policy_state = src_policy_state.update(
                    reward_hyper_params = reward_hyper_params,
                )

            return src_policy_state, src_train_state

        def noop():
            # Return copy of self (dst)
            dst_policy_state = jax.tree_map(
                lambda x: x[dst_idx], policy_states)

            dst_train_state = jax.tree_map(
                lambda x: x[dst_idx], train_states)
            
            return dst_policy_state, dst_train_state

        should_overwrite = _check_overwrite(cfg, policy_states, src_idx, dst_idx)
        jax.debug.print("Train {} {} {}", dst_idx, src_idx, should_overwrite, ordered=True)

        return lax.cond(should_overwrite, get_overwrite_policy, noop)

    pbt_rng, mutate_base_rng = random.split(pbt_rng, 2)

    overwrite_policy_states, overwrite_train_states = cull_train_policy(
        policy_states, train_states,
        random.split(mutate_base_rng, cfg.pbt.num_cull_policies),
        bottom_idxs, top_idxs)

    def overwrite_param(param, srcs):
        return param.at[bottom_idxs].set(srcs)

    policy_states = jax.tree_map(
        overwrite_param, policy_states, overwrite_policy_states)

    train_states = jax.tree_map(
        overwrite_param, train_states, overwrite_train_states)

    policy_states = _rebase_elos(policy_states)

    return train_state_mgr.replace(
        policy_states = policy_states,
        train_states = train_states,
        pbt_rng = pbt_rng,
    )


def _pbt_past_update(
    cfg: TrainConfig,
    train_state_mgr: TrainStateManager,
):
    if cfg.pbt.num_past_policies == 0:
        return train_state_mgr

    policy_states = train_state_mgr.policy_states
    pbt_rng = train_state_mgr.pbt_rng
    pbt_rng, src_idx_rng = random.split(pbt_rng, 2)

    src_idx = random.randint(src_idx_rng, (), 0, cfg.pbt.num_train_policies)
    dst_idx = jnp.argmin(policy_states.fitness_score[cfg.pbt.num_train_policies:])
    dst_idx = dst_idx + cfg.pbt.num_train_policies

    def overwrite_past_policy(policy_states):
        def save_param(x):
            return x.at[dst_idx].set(x[src_idx])

        policy_states = jax.tree_map(save_param, policy_states)
        policy_states = _rebase_elos(policy_states)

        jax.debug.print("Past {} {}", src_idx, dst_idx, ordered=True)

        return policy_states

    def noop(policy_states):
        return policy_states

    should_overwrite = _check_overwrite(cfg, policy_states, src_idx, dst_idx)

    policy_states = lax.cond(should_overwrite,
        overwrite_past_policy, noop, policy_states)

    return train_state_mgr.replace(
        policy_states = policy_states,
        pbt_rng = pbt_rng,
    )


def pbt_update(
    cfg: TrainConfig,
    train_state_mgr: TrainStateManager,
    update_idx: jax.Array,
):
    if cfg.pbt == None:
        return

    def pbt_update_noop(train_state_mgr):
        return train_state_mgr

    def pbt_past_update(train_state_mgr):
        return _pbt_past_update(cfg, train_state_mgr)

    def pbt_cull_update(train_state_mgr):
        return _pbt_cull_update(cfg, train_state_mgr)

    should_update_past = jnp.logical_and(update_idx != 0,
        update_idx % cfg.pbt.past_policy_update_interval == 0)
    train_state_mgr = lax.cond(should_update_past,
        pbt_past_update, pbt_update_noop, train_state_mgr)

    should_cull_policy = jnp.logical_and(update_idx != 0,
        update_idx % cfg.pbt.train_policy_cull_interval == 0)
    train_state_mgr = lax.cond(should_cull_policy,
        pbt_cull_update, pbt_update_noop, train_state_mgr)

    return train_state_mgr



import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
import numpy as np
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from functools import partial
import math
from typing import Callable, List, Optional

from .algo_common import HyperParams
from .cfg import TrainConfig, PBTConfig, ParamExplore
from .train_state import (
    PolicyState, PolicyTrainState, TrainStateManager,
    MMR, MovingEpisodeScore,
)
from .profile import profile

@dataclass(frozen=True)
class PBTMatchmakeConfig:
    num_current_policies: int
    num_past_policies: int
    total_num_policies: int
    num_teams: int
    team_size: int

    self_play_portion: float
    cross_play_portion: float
    past_play_portion: float
    static_play_portion: float

    self_play_batch_size: int
    cross_play_batch_size: int
    past_play_batch_size: int
    static_play_batch_size: int

    num_cross_play_matches: int
    num_past_play_matches: int
    num_static_play_matches: int
    num_total_matches: int

    complex_matchmaking: bool 

    custom_policy_ids: List[int]
    
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
        custom_policy_ids: List[int],
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
        #assert num_static_play_matches % num_current_policies == 0
        
        complex_matchmaking = self_play_portion != 1.0

        return PBTMatchmakeConfig(
            num_current_policies = num_current_policies,
            num_past_policies = num_past_policies,
            total_num_policies = total_num_policies,
            num_teams = num_teams,
            team_size = team_size,

            self_play_portion = self_play_portion,
            cross_play_portion = cross_play_portion,
            past_play_portion = past_play_portion,
            static_play_portion = static_play_portion,

            self_play_batch_size = self_play_batch_size,
            cross_play_batch_size = cross_play_batch_size,
            past_play_batch_size = past_play_batch_size,
            static_play_batch_size = static_play_batch_size,

            num_cross_play_matches = num_cross_play_matches,
            num_past_play_matches = num_past_play_matches,
            num_static_play_matches = num_static_play_matches,
            num_total_matches = num_total_matches,

            complex_matchmaking = complex_matchmaking,
            custom_policy_ids = custom_policy_ids,
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

    return assignments.at[:, 1:, :].set(new_assignments).reshape(-1)


def _past_play_matchmake(
    assignments,
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

    return assignments.at[:, 1:, :].set(new_assignments).reshape(-1)


def _elo_expected_result(
    my_elo,
    opponent_elo,
):
    return 1 / (1 + 10 ** ((opponent_elo - my_elo) / 400))

def _convert_custom_policy_ids(assignments, mm_cfg):
    for i, custom_id in enumerate(mm_cfg.custom_policy_ids):
        assignments = jnp.where(
            assignments == custom_id,
            i + mm_cfg.total_num_policies,
            assignments)

    return assignments

def pbt_update_elo(
    get_episode_scores_fn,
    assignments,
    dones,
    episode_results,
    policy_elos,
    mm_cfg,
):
    assert mm_cfg.num_teams == 2

    assignments = _convert_custom_policy_ids(assignments, mm_cfg)

    assignments = assignments.reshape(
        mm_cfg.num_total_matches, mm_cfg.num_teams,
        mm_cfg.team_size, 1)
    dones = dones.reshape(
        mm_cfg.num_total_matches, mm_cfg.num_teams,
        mm_cfg.team_size, 1)

    a_assignments = assignments[:, 0, 0, 0]
    b_assignments = assignments[:, 1, 0, 0]
    dones = dones[:, 0, 0, :]

    def update_mmr(policy_idx, cur_elo):
        @jax.vmap
        def compute_differences(episode_result, a_assignment, b_assignment, done):
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
                a_score, b_score = get_episode_scores_fn(
                    episode_result)

                a_elo = policy_elos[a_assignment]
                b_elo = policy_elos[b_assignment]

                my_score = jnp.where(is_a, a_score, b_score)

                my_elo = jnp.where(is_a, a_elo, b_elo)
                opponent_elo = jnp.where(is_a, b_elo, a_elo)

                expected_score = _elo_expected_result(my_elo, opponent_elo)
                diff = my_score - expected_score

                return diff

            def skip_diff():
                return jnp.zeros((), dtype=jnp.float32)

            return lax.cond(valid, compute_diff, skip_diff)

        diffs = compute_differences(episode_results, a_assignments,
                                    b_assignments, dones)

        K = 1.0
        new_elo = cur_elo + K * diffs.sum()

        return new_elo

    new_elos = jax.vmap(update_mmr)(
        jnp.arange(policy_elos.shape[0]), policy_elos)

    return new_elos


def pbt_update_matchmaking(
    assignments,
    policy_states,
    dones,
    episode_results,
    assign_rnd,
    mm_cfg,
):
    cross_start = mm_cfg.self_play_batch_size
    cross_end = cross_start + mm_cfg.cross_play_batch_size
    
    past_start = cross_end
    past_end = past_start + mm_cfg.past_play_batch_size

    if mm_cfg.cross_play_batch_size > 0:
        assign_rnd, cross_rnd = random.split(assign_rnd)
        new_cross_assignments = _cross_play_matchmake(
            assignments[cross_start:cross_end],
            dones[cross_start:cross_end], cross_rnd, mm_cfg)

        assignments = assignments.at[cross_start:cross_end].set(
            new_cross_assignments)

    if mm_cfg.past_play_batch_size > 0:
        assign_rnd, past_rnd = random.split(assign_rnd)

        new_past_assignments = _past_play_matchmake(
            assignments[past_start:past_end],
            dones[past_start:past_end], past_rnd, mm_cfg)

        assignments = assignments.at[past_start:past_end].set(
            new_past_assignments)

    return assignments, assign_rnd


def pbt_update_fitness(
    assignments,
    policy_states,
    dones,
    episode_results,
    mm_cfg,
):
    assert mm_cfg.num_teams == 1
    assert policy_states.mmr == None and policy_states.episode_score != None

    assignments = assignments.reshape(
        mm_cfg.num_total_matches, mm_cfg.team_size)
    dones = dones.reshape(
        mm_cfg.num_total_matches, mm_cfg.team_size)

    assignments = assignments[:, 0]
    dones = dones[:, 0]

    # Note this ema is biased but equally so it seems like it shouldn't matter
    ema_decay = 0.9999

    def update_policy_episode_score(policy_idx, cur_episode_score):
        @jax.vmap
        def get_scores(episode_result, assignment, done):
            def valid(episode_result):
                return policy_states.get_episode_scores_fn(episode_result), True

            def invalid(episode_result):
                return 0.0, False

            is_valid = jnp.logical_and(done, assignment == policy_idx)

            return lax.cond(is_valid, valid, invalid, episode_result)

        x_scores, valids = get_scores(
            episode_results, assignments, dones)

        x_N = valids.sum()

        def update_moving_avg(cur_episode_score):
            x_mean = jnp.mean(x_scores, where=valids)

            x_var = lax.cond(x_N > 1,
                lambda scores, valids: jnp.var(scores, where=valids, ddof=1),
                lambda scores, valids: jnp.float32(0),
                x_scores, valids)

            mean_delta = x_mean - cur_episode_score.mean

            cur_weight = jnp.expm1(
                x_N.astype(jnp.float32) * jnp.log(ema_decay)) + 1
            x_weight = 1 - cur_weight

            N_max = jnp.iinfo(cur_episode_score.N.dtype).max

            # Saturate new_N. At this point the scaling factor between cur_N 
            # and new_N would be very very close to 1 anyway.
            cur_N = cur_episode_score.N
            new_N = jnp.where(x_N > N_max - cur_N, N_max, cur_N + x_N)

            def mean_delta_var():
                scale = (cur_N.astype(jnp.float32) /
                         ((new_N - 1).astype(jnp.float32)))

                return scale * (cur_weight * x_weight) * jnp.square(mean_delta)

            new_mean = cur_weight * cur_episode_score.mean + x_weight * x_mean
            new_var = (
                cur_weight * cur_episode_score.var + x_weight * x_var +
                lax.cond(cur_N > 0, mean_delta_var, lambda: 0.0)
            )

            return cur_episode_score.replace(
                mean = new_mean,
                var = new_var,
                N = new_N,
            )

        def skip(cur_episode_score):
            return cur_episode_score

        return lax.cond(x_N > 0, update_moving_avg, skip, cur_episode_score)

    new_episode_scores = jax.vmap(update_policy_episode_score)(
        jnp.arange(policy_states.episode_score.mean.shape[0]),
        policy_states.episode_score)

    policy_states = policy_states.update(episode_score=new_episode_scores)

    return policy_states

def pbt_explore_hyperparams(
    cfg: TrainConfig,
    explore_rng: random.PRNGKey,
    policy_state: PolicyState,
    train_state: PolicyTrainState,
    resample_chance: float,
):
    def explore_param(rnd, param, param_explore):
        lo = param_explore.base * param_explore.min_scale
        hi = param_explore.base * param_explore.max_scale

        def resample_param(param_rnd, param):
            if param_explore.log10_scale:
                lo_sample = math.log10(lo)
                hi_sample = math.log10(hi)
            elif param_explore.ln_scale:
                lo_sample = math.log(lo)
                hi_sample = math.log(hi)
            else:
                lo_sample = lo
                hi_sample = hi

            sampled = random.uniform(param_rnd, (), dtype=jnp.float32,
                minval=lo_sample, maxval=hi_sample)

            if param_explore.log10_scale:
                sampled = 10 ** sampled
            elif param_explore.ln_scale:
                sampled = jnp.exp(sampled)

            return sampled

        def perturb_param(param_rnd, param):
            perturbed = param * random.uniform(
                param_rnd, (), dtype=jnp.float32,
                minval=param_explore.perturb_rnd_min,
                maxval=param_explore.perturb_rnd_max)

            if param_explore.clip_perturb:
                perturbed = jnp.clip(a=perturbed, a_min=lo, a_max=hi)

            return perturbed

        resample_rnd, param_rnd = random.split(rnd, 2)
        should_resample = random.uniform(resample_rnd, (), dtype=jnp.float32,
            minval=0, maxval=1) < resample_chance

        return lax.cond(should_resample, resample_param, perturb_param,
                        param_rnd, param)

    lr_rnd, entropy_rnd, reward_hyper_params_rnd = random.split(explore_rng, 3)

    if policy_state.reward_hyper_params != None:
        reward_hyper_params = policy_state.reward_hyper_params
        assert reward_hyper_params.ndim == 1

        reward_hyper_params_rnd = random.split(
            reward_hyper_params_rnd, reward_hyper_params.shape[0])

        for i, (name, param_explore) in enumerate(
                cfg.pbt.reward_hyper_params_explore.items()):
            new_param = explore_param(reward_hyper_params_rnd[i],
                                      reward_hyper_params[i], param_explore)

            reward_hyper_params = reward_hyper_params.at[i].set(new_param)

        policy_state = policy_state.update(
            reward_hyper_params = reward_hyper_params,
        )

    train_hyper_params = train_state.hyper_params

    if isinstance(cfg.lr, ParamExplore):
        train_hyper_params = train_hyper_params.replace(
            lr = explore_param(lr_rnd, train_hyper_params.lr, cfg.lr),
        )

    # FIXME, entropy is PPO specific, should have an algo permutation function
    if isinstance(cfg.algo.entropy_coef, ParamExplore):
        train_hyper_params = train_hyper_params.replace(
            entropy_coef = explore_param(
                entropy_rnd, train_hyper_params.entropy_coef,
                cfg.algo.entropy_coef),
        )

    train_state = train_state.update(
        hyper_params = train_hyper_params,
    )

    return policy_state, train_state


def _check_overwrite(cfg, policy_states, src_idx, dst_idx):
    if policy_states.mmr != None:
        src_elo = policy_states.mmr.elo[src_idx]
        dst_elo = policy_states.mmr.elo[dst_idx]

        src_expected_winrate = _elo_expected_result(src_elo, dst_elo)
        return src_expected_winrate >= cfg.pbt.policy_overwrite_threshold
    else:
        src_episode_score = jax.tree_map(
            lambda x: x[src_idx], policy_states.episode_score)
        dst_episode_score = jax.tree_map(
            lambda x: x[dst_idx], policy_states.episode_score)

        src_mean = policy_states.episode_score.mean[src_idx]
        src_var = policy_states.episode_score.var[src_idx]
        src_N = policy_states.episode_score.N[src_idx]

        dst_mean = policy_states.episode_score.mean[dst_idx]
        dst_var = policy_states.episode_score.var[dst_idx]
        dst_N = policy_states.episode_score.N[dst_idx]

        src_s2 = src_var / src_N.astype(jnp.float32)
        dst_s2 = dst_var / dst_N.astype(jnp.float32)

        t = (src_mean - dst_mean) / jnp.sqrt(src_s2 + dst_s2)

        p = 1 - jax.scipy.stats.norm.cdf(t)

        jax.debug.print("{} {}, {} {}, {} {} {} {}", t, p,
                        src_N, dst_N,
                        src_mean, dst_mean,
                        src_var,
                        dst_var)

        return p < 0.20


def _get_fitness_scores(policy_states):
    if policy_states.mmr != None:
        return policy_states.mmr.elo
    else:
        return policy_states.episode_score.mean


def pbt_cull_update(
    cfg: TrainConfig,
    train_state_mgr: TrainStateManager,
    num_cull_policies: int,
):
    policy_states = train_state_mgr.policy_states
    train_states = train_state_mgr.train_states
    pbt_rng = train_state_mgr.pbt_rng

    assert 2 * num_cull_policies <= cfg.pbt.num_train_policies

    fitness_scores = _get_fitness_scores(policy_states)
    sort_idxs = jnp.argsort(fitness_scores[0:cfg.pbt.num_train_policies])
    
    bottom_idxs = sort_idxs[:num_cull_policies]
    top_idxs = sort_idxs[-num_cull_policies:]

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

            src_policy_state, src_train_state = pbt_explore_hyperparams(
                cfg, mutate_rng, src_policy_state, src_train_state, 0.2)

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
        random.split(mutate_base_rng, num_cull_policies),
        bottom_idxs, top_idxs)

    def overwrite_param(param, srcs):
        return param.at[bottom_idxs].set(srcs)

    policy_states = jax.tree_map(
        overwrite_param, policy_states, overwrite_policy_states)

    train_states = jax.tree_map(
        overwrite_param, train_states, overwrite_train_states)

    return train_state_mgr.replace(
        policy_states = policy_states,
        train_states = train_states,
        pbt_rng = pbt_rng,
    )


def pbt_past_update(
    cfg: TrainConfig,
    train_state_mgr: TrainStateManager,
):
    if cfg.pbt.num_past_policies == 0:
        return train_state_mgr

    policy_states = train_state_mgr.policy_states
    pbt_rng = train_state_mgr.pbt_rng
    pbt_rng, src_idx_rng = random.split(pbt_rng, 2)

    fitness_scores = _get_fitness_scores(policy_states)

    src_idx = random.randint(src_idx_rng, (), 0, cfg.pbt.num_train_policies)
    dst_idx = jnp.argmin(fitness_scores[cfg.pbt.num_train_policies:])
    dst_idx = dst_idx + cfg.pbt.num_train_policies

    def overwrite_past_policy(policy_states):
        def save_param(x):
            return x.at[dst_idx].set(x[src_idx])

        policy_states = jax.tree_map(save_param, policy_states)

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

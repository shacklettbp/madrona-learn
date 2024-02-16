import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict
import optax

from os import environ as env_vars
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Dict, Callable

from .cfg import TrainConfig
from .rollouts import RolloutConfig, RolloutManager, RolloutState
from .actor_critic import ActorCritic
from .algo_common import AlgoBase
from .metrics import CustomMetricConfig, TrainingMetrics
from .moving_avg import EMANormalizer
from .train_state import PolicyTrainState, TrainStateManager
from .policy import Policy
from .profile import profile

def train(
    dev: jax.Device,
    cfg: TrainConfig,
    sim_init: Callable,
    sim_step: Callable,
    policy: Policy,
    iter_cb: Callable,
    metrics_cfg: CustomMetricConfig,
    restore_ckpt: str = None,
    profile_port: int = None,
):
    print(cfg)

    with jax.default_device(dev):
        return _train_impl(dev.platform, cfg, sim_init, sim_step,
            policy, iter_cb, metrics_cfg,
            restore_ckpt, profile_port)

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

    def save_param(x):
        return x.at[bottom_idxs].set(x[top_idxs])

    policy_states = jax.tree_map(save_param, policy_states)
    train_states = jax.tree_map(save_param, train_states)

    # Need to split the update keys for the copied policies
    update_prng_keys = train_states.update_prng_key
    new_prng_keys = jax.vmap(random.split, in_axes=(0, None), out_axes=1)(
        update_prng_keys[top_idxs], 2)

    update_prng_keys = update_prng_keys.at[top_idxs].set(
        new_prng_keys[0])
    update_prng_keys = update_prng_keys.at[bottom_idxs].set(
        new_prng_keys[1])

    train_states = train_states.update(update_prng_key=update_prng_keys)

    if policy_states.reward_hyper_params != None:
        pbt_rng, mutate_rng = random.split(pbt_rng, 2)
        reward_hyper_params = policy_states.reward_hyper_params

        reward_hyper_params = reward_hyper_params.at[bottom_idxs].set(
            jax.vmap(policy_states.mutate_reward_hyper_params)(
                random.split(mutate_rng, cfg.pbt.num_cull_policies),
                reward_hyper_params[bottom_idxs]))

        policy_states = policy_states.update(
            reward_hyper_params = reward_hyper_params,
        )

    # FIXME: elo grows unbounded, I think because we copy high elo policies
    # Rebase elo so average is 1500
    elos = policy_states.fitness_score[..., 0]
    elo_correction = jnp.mean(elos) - 1500
    policy_states = policy_states.update(
        fitness_score = policy_states.fitness_score - elo_correction,
    )

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

    current_policies = jax.tree_map(
        lambda x: x[:cfg.pbt.num_train_policies], policy_states)

    past_policies = jax.tree_map(
        lambda x: x[cfg.pbt.num_train_policies:], policy_states)

    assert (current_policies.fitness_score.shape[0] ==
            cfg.pbt.num_train_policies)
    assert past_policies.fitness_score.shape[0] == cfg.pbt.num_past_policies

    src_idx = jnp.argmax(current_policies.fitness_score)
    dst_idx = jnp.argmin(past_policies.fitness_score)
    dst_idx = dst_idx + cfg.pbt.num_train_policies

    def save_param(x):
        return x.at[dst_idx].set(x[src_idx])

    policy_states = jax.tree_map(save_param, policy_states)

    return train_state_mgr.replace(
        policy_states = policy_states,
    )

    #pbt_rng = train_state_mgr.pbt_rng
    #pbt_rng, save_idx_rng, store_idx_rng = random.split(pbt_rng, 3)

    #save_idx = random.randint(
    #    save_idx_rng, (), 0, cfg.pbt.num_train_policies)

    #store_idx = random.randint(
    #    store_idx_rng, (), cfg.pbt.num_train_policies,
    #    cfg.pbt.num_train_policies + cfg.pbt.num_past_policies)

    #def save_param(x):
    #    return x.at[store_idx].set(x[save_idx])

    #policy_states = jax.tree_map(save_param, policy_states)

    #return train_state_mgr.replace(
    #    policy_states = policy_states,
    #    pbt_rng = pbt_rng,
    #)

def _update_loop(
    algo: AlgoBase,
    iter_cb: Callable,
    cfg: TrainConfig,
    metrics_cfg: CustomMetricConfig,
    rollout_state: RolloutState,
    rollout_mgr: RolloutManager,
    train_state_mgr: TrainStateManager,
    start_update_idx: int,
):
    num_updates_remaining = cfg.num_updates - start_update_idx
    if cfg.pbt != None:
        pbt_update_interval = min(
            cfg.pbt.past_policy_update_interval,
            cfg.pbt.policy_cull_interval)

        num_train_policies = cfg.pbt.num_train_policies
    else:
        pbt_update_interval = num_updates_remaining
        num_train_policies = 1

    @jax.vmap
    def algo_wrapper(policy_state, train_state, rollout_data, metrics):
        return algo.update(
            cfg, 
            policy_state,
            train_state,
            rollout_data,
            metrics_cfg.update_cb,
            metrics,
        )

    def inner_update_iter(update_idx, inputs):
        rollout_state, train_state_mgr, metrics = inputs

        with profile("Update Iter"):
            with profile('Collect Rollouts'):
                (train_state_mgr, rollout_state, rollout_data,
                 obs_stats, metrics) = rollout_mgr.collect(
                    train_state_mgr, rollout_state, metrics)

            train_policy_states = jax.tree_map(
                lambda x: x[0:num_train_policies],
                train_state_mgr.policy_states)

            with profile('Update Observations Stats'):
                # Policy optimization only uses preprocessed observations,
                # so it is safe to update the preprocess state immediately,
                # because this will only affect the next set of rollouts
                train_policy_states = \
                    train_policy_states.update(obs_preprocess_state = \
                        train_policy_states.obs_preprocess.update_state(
                            train_policy_states.obs_preprocess_state,
                            obs_stats,
                            True,
                        )
                    )

            with profile('Learn'):
                (train_policy_states, updated_train_states,
                 metrics) = algo_wrapper(
                    train_policy_states, train_state_mgr.train_states,
                    rollout_data, metrics)

            # Copy new params into the full policy_state array
            with profile('Set New Policy States'):
                policy_states = jax.tree_map(
                    lambda full, new: full.at[0:num_train_policies].set(new),
                    train_state_mgr.policy_states, train_policy_states)

            train_state_mgr = train_state_mgr.replace(
                policy_states = policy_states,
                train_states = updated_train_states,
            )
            
        reset_metrics = iter_cb(update_idx, metrics, train_state_mgr)

        metrics = lax.cond(
            reset_metrics, lambda: metrics.reset(), lambda: metrics)

        return rollout_state, train_state_mgr, metrics

    def outer_update_iter(outer_update_idx, inputs):
        rollout_state, train_state_mgr, metrics = inputs

        inner_begin_idx = (
            start_update_idx + outer_update_idx * pbt_update_interval
        )
        
        if cfg.pbt != None:
            def pbt_update_noop(train_state_mgr):
                return train_state_mgr

            def pbt_cull_update(train_state_mgr):
                return _pbt_cull_update(cfg, train_state_mgr)

            should_cull_policy = jnp.logical_and(inner_begin_idx != 0,
                inner_begin_idx % cfg.pbt.policy_cull_interval == 0)
            train_state_mgr = lax.cond(should_cull_policy,
                pbt_cull_update, pbt_update_noop, train_state_mgr)

            def pbt_past_update(train_state_mgr):
                return _pbt_past_update(cfg, train_state_mgr)

            should_update_past = jnp.logical_and(inner_begin_idx != 0,
                inner_begin_idx % cfg.pbt.past_policy_update_interval == 0)
            train_state_mgr = lax.cond(should_update_past,
                pbt_past_update, pbt_update_noop, train_state_mgr)

        inner_end_idx = inner_begin_idx + pbt_update_interval
        inner_end_idx = jnp.minimum(inner_end_idx, cfg.num_updates)

        rollout_state, train_state_mgr, metrics = lax.fori_loop(
            inner_begin_idx, inner_end_idx,
            inner_update_iter, (rollout_state, train_state_mgr, metrics))
        
        return rollout_state, train_state_mgr, metrics
    
    metrics = algo.add_metrics(cfg, FrozenDict())
    metrics = rollout_mgr.add_metrics(cfg, metrics)
    metrics = metrics_cfg.add_metrics(metrics)
    metrics = TrainingMetrics.create(cfg, metrics)

    num_outer_iters = num_updates_remaining // pbt_update_interval
    if num_outer_iters * pbt_update_interval < num_updates_remaining:
        num_outer_iters += 1

    rollout_state, train_state_mgr, metrics = lax.fori_loop(
        0, num_outer_iters, outer_update_iter,
        (rollout_state, train_state_mgr, metrics))

    return rollout_state, train_state_mgr


def _setup_rollout_cfg(dev_type, cfg):
    sim_batch_size = cfg.num_agents_per_world * cfg.num_worlds

    if cfg.pbt != None:
        assert (cfg.pbt.num_teams * cfg.pbt.team_size ==
                cfg.num_agents_per_world)

        return RolloutConfig.setup(
            num_current_policies = cfg.pbt.num_train_policies,
            num_past_policies = cfg.pbt.num_past_policies,
            num_teams = cfg.pbt.num_teams,
            team_size = cfg.pbt.team_size,
            sim_batch_size = sim_batch_size,
            self_play_portion = cfg.pbt.self_play_portion,
            cross_play_portion = cfg.pbt.cross_play_portion,
            past_play_portion = cfg.pbt.past_play_portion,
            static_play_portion = 0.0,
            policy_dtype = cfg.compute_dtype,
            policy_chunk_size_override = \
                cfg.pbt.rollout_policy_chunk_size_override,
        )
    else:
        return RolloutConfig.setup(
            num_current_policies = 1,
            num_past_policies = 0,
            num_teams = 1,
            team_size = cfg.num_agents_per_world,
            sim_batch_size = sim_batch_size,
            self_play_portion = 1.0,
            cross_play_portion = 0.0,
            past_play_portion = 0.0,
            static_play_portion = 0.0,
            policy_dtype = cfg.compute_dtype,
        )


def _train_impl(
    dev_type,
    cfg,
    sim_init,
    sim_step,
    policy,
    iter_cb,
    metrics_cfg,
    restore_ckpt,
    profile_port,
):
    if profile_port != None:
        jax.profiler.start_server(profile_port)
        env_vars['TF_GPU_CUPTI_FORCE_CONCURRENT_KERNEL'] = '1'

    checkify_errors = checkify.user_checks
    if 'MADRONA_LEARN_FULL_CHECKIFY' in env_vars and \
            env_vars['MADRONA_LEARN_FULL_CHECKIFY'] == '1':
        checkify_errors |= (
            checkify.float_checks |
            checkify.nan_checks |
            checkify.div_checks |
            checkify.index_checks
        )

    algo = cfg.algo.setup()

    rollout_rng, init_rng = random.split(random.PRNGKey(cfg.seed))

    rollout_cfg = _setup_rollout_cfg(dev_type, cfg)

    @jax.jit
    def init_rollout_state():
        rnn_states = policy.actor_critic.init_recurrent_state(
                rollout_cfg.sim_batch_size)

        return RolloutState.create(
            rollout_cfg = rollout_cfg,
            init_fn = sim_init,
            step_fn = sim_step,
            prng_key = rollout_rng,
            rnn_states = rnn_states,
            static_play_assignments = None,
        )

    rollout_state = init_rollout_state()

    train_state_mgr = TrainStateManager.create(
        policy = policy, 
        cfg = cfg,
        algo = algo,
        base_rng = init_rng,
        example_obs = rollout_state.cur_obs,
        example_rnn_states = rollout_state.rnn_states,
        track_policy_fitness = rollout_cfg.has_matchmaking,
        checkify_errors = checkify_errors,
    )

    if restore_ckpt != None:
        train_state_mgr, start_update_idx = train_state_mgr.load(restore_ckpt)
    else:
        start_update_idx = 0

    rollout_mgr = RolloutManager(
        train_cfg = cfg,
        rollout_cfg = rollout_cfg,
        init_rollout_state = rollout_state,
        example_policy_states = train_state_mgr.policy_states,
    )

    def update_loop_wrapper(rollout_state, train_state_mgr):
        return _update_loop(
            algo = algo,
            iter_cb = iter_cb,
            cfg = cfg,
            metrics_cfg = metrics_cfg,
            rollout_state = rollout_state,
            rollout_mgr = rollout_mgr,
            train_state_mgr = train_state_mgr,
            start_update_idx = start_update_idx,
        )

    update_loop_wrapper = jax.jit(
        checkify.checkify(update_loop_wrapper, errors=checkify_errors),
        donate_argnums=[0, 1])

    lowered_update_loop = update_loop_wrapper.lower(
        rollout_state, train_state_mgr)

    if 'MADRONA_LEARN_DUMP_LOWERED' in env_vars:
        with open(env_vars['MADRONA_LEARN_DUMP_LOWERED'], 'w') as f:
            print(lowered_update_loop.as_text(), file=f)

    compiled_update_loop = lowered_update_loop.compile()

    if 'MADRONA_LEARN_DUMP_IR' in env_vars:
        with open(env_vars['MADRONA_LEARN_DUMP_IR'], 'w') as f:
            print(compiled_update_loop.as_text(), file=f)

    err, (rollout_state, train_state_mgr) = compiled_update_loop(
        rollout_state, train_state_mgr)
    err.throw()

    if profile_port != None:
        train_state_mgr.train_states.update_prng_key.block_until_ready()
        jax.profiler.stop_server()

    return train_state_mgr

import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict
import optax

import math
from os import environ as env_vars
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Dict, Callable, Any

from .cfg import TrainConfig
from .rollouts import RolloutConfig, RolloutManager, RolloutState
from .actor_critic import ActorCritic
from .algo_common import AlgoBase
from .metrics import TrainingMetrics, Metric
from .moving_avg import EMANormalizer
from .train_state import PolicyState, PolicyTrainState, TrainStateManager
from .pbt import pbt_update, pbt_explore_hyperparams
from .policy import Policy
from .profile import profile

# Inherit from this class and override any methods you see fit to modify / add
# functionality in the training loop. Note that this class itself (and any
# classes inherited from it) MUST be stateless. Any dataclass variables are
# effectively compile time constants (anything accessed through self.)
# Custom state can put in the pytree returned by init_user_state().
# This user state will be kept in TrainStateManager and passed back to all the
# methods of this class. Additionally it will get saved to disk in checkpoints
# along with model parameters etc.
@dataclass(frozen=True)
class TrainHooks:
    # Can override to return any pytree of jax arrays you want to store
    # custom state.
    def init_user_state(self):
        return None

    # You need to implement this function in your class. This is where (based
    # on update_idx) you save train_state_mgr to disk, write metrics, etc.
    # Return true if metrics should be reset (for example after writing them
    # to disk). If you return false, metrics will continue to be averaged
    # until the next call to this function.
    # Whatever is returned by user_state above will be in
    # train_state_mgr.user_state
    def post_update(
        self,
        update_idx: int,
        metrics: FrozenDict[str, Metric],
        train_state_mgr: TrainStateManager,
    ) -> bool:
        raise NotImplementedError

    # Called right before rollout collection loop starts
    def start_rollouts(
        self,
        rollout_state: RolloutState,
        user_state: Any,
    ):
        return rollout_state, user_state

    # Called right after rollout collection loop completes, but before
    # returns / advantages are calculated
    def finish_rollouts(
        self,
        rollouts: Dict[str, Any],
        bootstrap_values: jax.Array,
        unnormalized_values: jax.Array,
        unnormalized_bootstrap_values: jax.Array,
        user_state: Any,
    ):
        return rollouts, user_state

    def add_metrics(
        self,
        metrics: FrozenDict[str, Metric],
    ):
        # Check out RolloutManager.add_metrics for an example of how to 
        # use this function to add new custom metrics.
        return metrics
    
    # Update metrics after rollouts
    def rollout_metrics(
        self,
        metrics: FrozenDict[str, Metric],
        rollouts: Dict[str, Any],
        user_state: Any,
    ):
        return metrics

    # Update metrics after each minibatch
    def optimize_metrics(
        self,
        metrics: FrozenDict[str, Metric],
        epoch_idx: int,
        minibatch: Dict[str, Any],
        policy_state: PolicyState,
        train_state: PolicyTrainState,
    ):
        return metrics

def train(
    dev: jax.Device,
    cfg: TrainConfig,
    sim_fns: Dict[str, Callable],
    policy: Policy,
    user_hooks: TrainHooks,
    restore_ckpt: str = None,
    profile_port: int = None,
):
    print(cfg)
    print()

    with jax.default_device(dev):
        return _train_impl(dev.platform, cfg, sim_fns,
            policy, user_hooks, restore_ckpt, profile_port)


def _update_loop(
    algo: AlgoBase,
    cfg: TrainConfig,
    user_hooks: TrainHooks,
    rollout_state: RolloutState,
    rollout_mgr: RolloutManager,
    train_state_mgr: TrainStateManager,
    start_update_idx: int,
):
    num_updates_remaining = cfg.num_updates - start_update_idx
    if cfg.pbt != None:
        outer_loop_interval = math.gcd(
            cfg.pbt.past_policy_update_interval,
            cfg.pbt.train_policy_cull_interval)

        num_train_policies = cfg.pbt.num_train_policies
    else:
        outer_loop_interval = num_updates_remaining
        num_train_policies = 1

    if outer_loop_interval == 0:
        outer_loop_interval = num_updates_remaining

    @jax.vmap
    def algo_wrapper(policy_state, train_state, rollout_data, metrics):
        return algo.update(
            cfg, 
            policy_state,
            train_state,
            rollout_data,
            user_hooks.optimize_metrics,
            metrics,
        )

    def inner_update_iter(update_idx, inputs):
        rollout_state, train_state_mgr, metrics = inputs

        with profile("Update Iter"):
            with profile('Collect Rollouts'):
                (train_state_mgr, rollout_state, rollout_data,
                 obs_stats, metrics) = rollout_mgr.collect(
                    train_state_mgr, rollout_state, metrics,
                    user_hooks.start_rollouts, user_hooks.finish_rollouts,
                    user_hooks.rollout_metrics)

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
            
        reset_metrics = user_hooks.post_update(update_idx, metrics, train_state_mgr)

        metrics = lax.cond(
            reset_metrics, lambda: metrics.reset(), lambda: metrics)

        return rollout_state, train_state_mgr, metrics

    def outer_update_iter(outer_update_idx, inputs):
        rollout_state, train_state_mgr, metrics = inputs

        inner_begin_idx = (
            start_update_idx + outer_update_idx * outer_loop_interval
        )
        
        train_state_mgr = pbt_update(cfg, train_state_mgr, inner_begin_idx)

        inner_end_idx = inner_begin_idx + outer_loop_interval
        inner_end_idx = jnp.minimum(inner_end_idx, cfg.num_updates)

        rollout_state, train_state_mgr, metrics = lax.fori_loop(
            inner_begin_idx, inner_end_idx,
            inner_update_iter, (rollout_state, train_state_mgr, metrics))
        
        return rollout_state, train_state_mgr, metrics
    
    metrics = algo.add_metrics(cfg, FrozenDict())
    metrics = rollout_mgr.add_metrics(cfg, metrics)
    metrics = user_hooks.add_metrics(metrics)
    metrics = TrainingMetrics.create(cfg, metrics)

    num_outer_iters = num_updates_remaining // outer_loop_interval
    if num_outer_iters * outer_loop_interval < num_updates_remaining:
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
    sim_fns,
    policy,
    user_hooks,
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

    if isinstance(cfg.seed, int):
        seed = random.key(cfg.seed)
    else:
        seed = cfg.seed

    rollout_rng, init_rng = random.split(seed)

    rollout_cfg = _setup_rollout_cfg(dev_type, cfg)

    @jax.jit
    def init_rollout_state():
        rnn_states = policy.actor_critic.init_recurrent_state(
                rollout_cfg.sim_batch_size)

        return RolloutState.create(
            rollout_cfg = rollout_cfg,
            sim_fns = sim_fns,
            prng_key = rollout_rng,
            rnn_states = rnn_states,
            static_play_assignments = None,
        )

    rollout_state = init_rollout_state()

    train_state_mgr = TrainStateManager.create(
        policy = policy, 
        cfg = cfg,
        algo = algo,
        init_user_state_cb = user_hooks.init_user_state,
        base_rng = init_rng,
        example_obs = rollout_state.cur_obs,
        example_rnn_states = rollout_state.rnn_states,
        use_competitive_mmr = rollout_cfg.pbt.complex_matchmaking,
        checkify_errors = checkify_errors,
    )

    @partial(jax.jit, donate_argnums=0)
    def sample_hyperparams(train_state_mgr):
        policy_states = train_state_mgr.policy_states
        train_states = train_state_mgr.train_states
        pbt_rng = train_state_mgr.pbt_rng

        explore_hyperparams = jax.vmap(
            pbt_explore_hyperparams, in_axes=(None, 0, 0, 0, None))

        rngs = random.split(pbt_rng, cfg.pbt.num_train_policies + 1)
        pbt_rng = rngs[0]
        explore_rngs = rngs[1:]

        train_policy_states = jax.tree_map(
            lambda x: x[0:cfg.pbt.num_train_policies], policy_states)

        train_policy_states, train_states = explore_hyperparams(
            cfg, explore_rngs, train_policy_states, train_states, 1.0,
        )

        policy_states = jax.tree_map(
            lambda x, y: x.at[0:cfg.pbt.num_train_policies].set(y),
            policy_states, train_policy_states)

        return train_state_mgr.replace(
            policy_states = policy_states,
            train_states = train_states,
            pbt_rng = pbt_rng,
        )

    if cfg.pbt:
        train_state_mgr = sample_hyperparams(train_state_mgr)

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
            cfg = cfg,
            user_hooks = user_hooks,
            rollout_state = rollout_state,
            rollout_mgr = rollout_mgr,
            train_state_mgr = train_state_mgr,
            start_update_idx = start_update_idx,
        )

    update_loop_wrapper = jax.jit(
        checkify.checkify(update_loop_wrapper, errors=checkify_errors),
        #donate_argnums=[0, 1])
        donate_argnums=[1]) # rollout_state has a token which breaks donation

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

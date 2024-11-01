import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict
import optax
import numpy as np

import math
import os
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
from .utils import aot_compile, get_checkify_errors

class TrainingManager(flax.struct.PyTreeNode):
    state: TrainStateManager
    rollout: RolloutState
    metrics: TrainingMetrics
    update_idx: int
    update_fn: Callable = flax.struct.field(pytree_node=False)
    profile_port: int = flax.struct.field(pytree_node=False)

    def save_ckpt(self, path):
        update_idx = int(self.update_idx)
        self.state.save(update_idx, os.path.join(path, str(update_idx)))

    def load_ckpt(self, path):
        return self.replace(state=self.state.load(path))

    def update_iter(self):
        new_state, new_rollout, new_metrics = self.update_fn(
            self.state, self.rollout, self.metrics, self.update_idx)

        return self.replace(
            state=new_state,
            rollout=new_rollout,
            metrics=new_metrics,
            update_idx=self.update_idx + 1,
        )

    def log_metrics_tensorboard(self, tb_writer):
        cpu_metrics = jax.tree.map(np.asarray, self.metrics)
        cpu_metrics.tensorboard_log(self.update_idx - 1, tb_writer)


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


def init_training(
    dev: jax.Device,
    cfg: TrainConfig,
    sim_fns: Dict[str, Callable],
    policy: Policy,
    user_hooks: TrainHooks = TrainHooks(),
    restore_ckpt: str = None,
    profile_port: int = None,
) -> TrainingManager:
    print(cfg)
    print()

    with jax.default_device(dev):
        return _init_training(dev.platform, cfg, sim_fns,
            policy, user_hooks, restore_ckpt, profile_port)

def stop_training(
    training_mgr: TrainingManager, 
):
    if training_mgr.profile_port != None:
        training_mgr.state.train_states.update_prng_key.block_until_ready()
        jax.profiler.stop_server()

def _update_impl(
    algo: AlgoBase,
    cfg: TrainConfig,
    user_hooks: TrainHooks,
    rollout_state: RolloutState,
    rollout_mgr: RolloutManager,
    train_state_mgr: TrainStateManager,
    metrics: TrainingMetrics,
    update_idx: int,
):
    metrics_vmap = jax.tree_util.tree_map_with_path(
        lambda kp, x: 1 if kp[0].name == 'metrics' else None, metrics)

    @partial(jax.vmap,
             in_axes=(0, 0, 0, metrics_vmap),
             out_axes=(0, 0, metrics_vmap))
    def algo_wrapper(policy_state, train_state, rollout_data, metrics):
        return algo.update(
            cfg, 
            policy_state,
            train_state,
            rollout_data,
            user_hooks.optimize_metrics,
            metrics,
        )

    if cfg.pbt != None:
        num_train_policies = cfg.pbt.num_train_policies
    else:
        num_train_policies = 1

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
        
    metrics = metrics.advance()

    return train_state_mgr, rollout_state, metrics

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
            reward_gamma = cfg.gamma,
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
            reward_gamma = cfg.gamma,
            policy_dtype = cfg.compute_dtype,
        )

def _init_training(
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
        checkify_errors = get_checkify_errors(),
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

    metrics = algo.add_metrics(cfg, FrozenDict())
    metrics = rollout_mgr.add_metrics(cfg, metrics)
    metrics = user_hooks.add_metrics(metrics)
    metrics = TrainingMetrics.create(cfg, metrics, start_update_idx)

    def update_wrapper(train_state_mgr, rollout_state, metrics, update_idx):
        return _update_impl(
            algo = algo,
            cfg = cfg,
            user_hooks = user_hooks,
            rollout_state = rollout_state,
            rollout_mgr = rollout_mgr,
            train_state_mgr = train_state_mgr,
            metrics = metrics,
            update_idx = update_idx,
        )

    return TrainingManager(
        state=train_state_mgr,
        rollout=rollout_state,
        metrics=metrics,
        update_idx=jnp.asarray(start_update_idx, jnp.int32),
        update_fn=update_wrapper,
        profile_port=profile_port,
    )

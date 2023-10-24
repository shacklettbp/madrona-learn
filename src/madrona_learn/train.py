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
from time import time

from .cfg import TrainConfig
from .rollouts import RolloutExecutor, RolloutState
from .actor_critic import ActorCritic
from .algo_common import AlgoBase, InternalConfig
from .metrics import CustomMetricConfig, TrainingMetrics
from .moving_avg import EMANormalizer
from .train_state import HyperParams, PolicyTrainState, TrainStateManager
from .profile import profile
from .utils import init_recurrent_states

def train(
    dev: jax.Device,
    cfg: TrainConfig,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    iter_cb: Callable,
    metrics_cfg: CustomMetricConfig,
    restore_ckpt: str = None,
):
    print(cfg)
    icfg = InternalConfig(dev, cfg)

    with jax.default_device(dev):
        _train_impl(cfg, icfg, sim_step, init_sim_data,
                    policy, iter_cb, metrics_cfg, restore_ckpt)

def _update_loop(
    algo: AlgoBase,
    iter_cb: Callable,
    cfg: TrainConfig,
    metrics_cfg: CustomMetricConfig,
    icfg: InternalConfig,
    rollout_state: RolloutState,
    rollout_exec: RolloutExecutor,
    train_state_mgr: TrainStateManager,
    start_update_idx: int,
):
    @jax.vmap
    def algo_wrapper(train_state, rollout_data, metrics):
        return algo.update(
            cfg, 
            icfg,
            train_state,
            rollout_data,
            metrics_cfg.update_cb,
            metrics,
        )

    @partial(jax.vmap, axis_size = cfg.pbt_ensemble_size)
    def metric_init_wrapper(train_state):
        return TrainingMetrics.create(
            cfg.algo.metrics() + metrics_cfg.custom_metrics)

    def update_iter(update_idx, inputs):
        rollout_state, train_state_mgr = inputs

        #update_start_time = time()
        update_start_time = 0

        metrics = algo.add_metrics(cfg, FrozenDict())
        metrics = rollout_exec.add_metrics(cfg, metrics)
        metrics = TrainingMetrics.create(cfg, metrics)

        with profile("Update Iter"):
            with profile('Collect Rollouts'):
                rollout_state, rollout_data, metrics = rollout_exec.collect(
                    train_state_mgr, rollout_state, metrics)

            with profile('Optimize'):
                updated_train_states, metrics = algo_wrapper(
                    train_state_mgr.train_states, rollout_data, metrics)

        train_state_mgr = TrainStateManager(
            train_states = updated_train_states)
            
        #update_end_time = time()
        update_end_time = 0
        update_time = update_end_time - update_start_time
        iter_cb(update_idx, update_time, metrics, train_state_mgr)

        return rollout_state, train_state_mgr

    return lax.fori_loop(start_update_idx, cfg.num_updates, update_iter,
        (rollout_state, train_state_mgr))

def _train_impl(cfg, icfg, sim_step, init_sim_data,
                policy, iter_cb, metrics_cfg, restore_ckpt):
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

    init_sim_data = frozen_dict.freeze(init_sim_data)

    rollout_rnd, init_rnd = random.split(random.PRNGKey(cfg.seed))

    rnn_states = init_recurrent_states(
        policy, icfg.rollout_agents_per_policy,
        cfg.pbt_ensemble_size * cfg.pbt_history_len)

    rollout_state = RolloutState.create(
        step_fn = sim_step,
        prng_key = rollout_rnd,
        rnn_states = rnn_states,
        init_sim_data = init_sim_data,
    )

    hyper_params = algo.init_hyperparams(cfg)
    optimizer = algo.make_optimizer(hyper_params)

    train_state_mgr = TrainStateManager.create(
        policy = policy, 
        optimizer = optimizer,
        hyper_params = hyper_params,
        mixed_precision = cfg.mixed_precision,
        num_policies = cfg.pbt_ensemble_size * cfg.pbt_history_len,
        batch_size_per_policy = icfg.rollout_agents_per_policy,
        base_init_rng = init_rnd,
        example_obs = init_sim_data['obs'],
        example_rnn_states = rnn_states,
        checkify_errors = checkify_errors,
    )

    if restore_ckpt != None:
        train_state_mgr, start_update_idx = train_state_mgr.load(restore_ckpt)
    else:
        start_update_idx = 0

    rollout_exec = RolloutExecutor(
        cfg,
        icfg,
        train_state_mgr,
        rollout_state,
    )

    def update_loop_wrapper(rollout_state, train_state_mgr):
        return _update_loop(
            algo = algo,
            iter_cb = iter_cb,
            cfg = cfg,
            metrics_cfg = metrics_cfg,
            icfg = icfg,
            rollout_state = rollout_state,
            rollout_exec = rollout_exec,
            train_state_mgr = train_state_mgr,
            start_update_idx = start_update_idx,
        )

    update_loop_wrapper = jax.jit(
        checkify.checkify(update_loop_wrapper, errors=checkify_errors),
        donate_argnums=[0, 1])

    lowered_update_loop = update_loop_wrapper.lower(
        rollout_state, train_state_mgr)

    if 'MADRONA_LEARN_PRINT_LOWERED' in env_vars and \
            env_vars['MADRONA_LEARN_PRINT_LOWERED'] == '1':
        print(lowered_update_loop.as_text())

    compiled_update_loop = lowered_update_loop.compile()

    err, (rollout_state, train_state_mgr) = compiled_update_loop(
        rollout_state, train_state_mgr)
    err.throw()

import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.training.dynamic_scale import DynamicScale
from flax.core import FrozenDict
import optax

from os import environ as env_vars
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Dict, Callable
from time import time

from .cfg import TrainConfig, CustomMetricConfig
from .rollouts import RolloutExecutor, RolloutState
from .actor_critic import ActorCritic
from .algo_common import InternalConfig, TrainingMetrics
from .moving_avg import EMANormalizer
from .train_state import HyperParams, PolicyTrainState, TrainStateManager
from .profile import profile

def train(
    dev: jax.Device,
    cfg: TrainConfig,
    sim_step: Callable,
    init_sim_data: FrozenDict,
    policy: ActorCritic,
    iter_cb: Callable,
    metrics_cfg: CustomMetricConfig,
    restore_ckpt: str =None,
):
    print(cfg)
    icfg = InternalConfig(dev, cfg)

    with jax.default_device(dev):
        _train_impl(cfg, icfg, sim_step, init_sim_data,
                    policy, iter_cb, metrics_cfg, restore_ckpt)

def _update_loop(
    algo_update_fn: Callable,
    iter_cb: Callable,
    cfg: TrainConfig,
    metrics_cfg: CustomMetricConfig,
    icfg: InternalConfig,
    rollout_state: RolloutState,
    rollout_exec: RolloutExecutor,
    train_state_mgr: TrainStateManager,
    start_update_idx: int,
):
    def algo_wrapper(train_state, rollout_data):
        metrics = TrainingMetrics.create(
            cfg.algo.metrics() + metrics_cfg.custom_metrics)

        return algo_update_fn(
            cfg, 
            icfg,
            train_state,
            rollout_data,
            metrics_cfg.cb,
            metrics,
        )

    algo_wrapper = jax.vmap(algo_wrapper, in_axes=(0, 0))

    def update_iter(update_idx, inputs):
        rollout_state, train_state_mgr = inputs

        #update_start_time = time()
        update_start_time = 0

        with profile("Update Iter"):
            with profile('Collect Rollouts'):
                rollout_state, rollout_data = rollout_exec.collect(
                    rollout_state, train_state_mgr)

            with profile('Optimize'):
                updated_train_states, metrics = algo_wrapper(
                    train_state_mgr.train_states, rollout_data)

        train_state_mgr = TrainStateManager(
            train_states = updated_train_states)
            
        #update_end_time = time()
        update_end_time = 0
        update_time = update_end_time - update_start_time
        iter_cb(update_idx, update_time, metrics, train_state_mgr)

        return rollout_state, train_state_mgr

    return lax.fori_loop(start_update_idx, cfg.num_updates, update_iter,
        (rollout_state, train_state_mgr))

def _setup_value_normalizer(cfg, rng_key, fake_values):
    value_norm_decay = \
        cfg.value_normalizer_decay if cfg.normalize_values else 1.0

    value_normalizer = EMANormalizer(
        value_norm_decay, disable=not cfg.normalize_values)

    value_normalizer_vars = value_normalizer.init(
        rng_key, 'normalize', False, fake_values)

    return value_normalizer.apply, value_normalizer_vars['batch_stats']

def _setup_new_policy(policy, cfg, prng_key, rnn_states, obs):
    model_init_rng, value_norm_rng, update_rng = random.split(prng_key, 3)
    fake_outs, variables = policy.init_with_output(
        model_init_rng, random.PRNGKey(0), rnn_states, obs,
        method='rollout')

    params = variables['params']
    batch_stats = variables['batch_stats']

    value_norm_fn, value_norm_stats = _setup_value_normalizer(
        cfg, value_norm_rng, fake_outs[2])

    hyper_params = HyperParams(
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    optimizer = optax.adam(learning_rate=hyper_params.lr)
    opt_state = optimizer.init(params)

    if cfg.mixed_precision:
        scaler = DynamicScale()
    else:
        scaler = None

    return PolicyTrainState(
        apply_fn = policy.apply,
        params = params,
        batch_stats = batch_stats,
        value_normalize_fn = value_norm_fn,
        value_normalize_stats = value_norm_stats,
        hyper_params = hyper_params,
        tx = optimizer,
        opt_state = opt_state,
        scheduler = None,
        scaler = scaler,
        update_prng_key = update_rng,
    )

def _setup_train_states(policy, cfg, icfg, base_init_rng,
                        rollout_state, checkify_errors):
    setup_new_policies = jax.vmap(
        partial(_setup_new_policy, policy, cfg), in_axes=(0, 0, 0))

    obs = jax.tree_map(lambda x: x.reshape(
            cfg.pbt_ensemble_size * cfg.pbt_history_len,
            icfg.rollout_agents_per_policy,
            *x.shape[1:],
        ), rollout_state.sim_data['obs'])

    setup_new_policies = jax.jit(
        checkify.checkify(setup_new_policies, errors=checkify_errors))

    init_rngs = random.split(base_init_rng, cfg.pbt_ensemble_size)

    err, train_states = setup_new_policies(
        init_rngs, rollout_state.rnn_states, obs)
    err.throw()

    return TrainStateManager(train_states=train_states)

def init(mem_fraction):
    env_vars["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_fraction:.2f}"
    #jax.config.update("jax_numpy_rank_promotion", "raise")
    jax.config.update("jax_numpy_dtype_promotion", "strict")

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

    rollout_rnd, init_rnd = random.split(random.PRNGKey(cfg.seed))

    def init_rnn_states():
        def init(arg):
            return policy.init_recurrent_state(icfg.rollout_agents_per_policy)

        return jax.vmap(init)(jnp.zeros(
            cfg.pbt_ensemble_size * cfg.pbt_history_len))

    rnn_states = jax.jit(init_rnn_states)()

    rollout_state = RolloutState.create(
        step_fn = sim_step,
        prng_key = rollout_rnd,
        rnn_states = rnn_states,
        init_sim_data = init_sim_data,
    )

    train_state_mgr = _setup_train_states(
        policy, cfg, icfg, init_rnd, rollout_state, checkify_errors)

    if restore_ckpt != None:
        start_update_idx = train_state_mgr.load(restore_ckpt)
    else:
        start_update_idx = 0

    rollout_exec = RolloutExecutor(
        cfg,
        icfg,
        policy,
        rollout_state,
    )

    algo_update_fn = cfg.algo.update_fn()

    def update_loop_wrapper(rollout_state, train_state_mgr):
        return _update_loop(
            algo_update_fn = algo_update_fn,
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

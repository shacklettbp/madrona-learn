import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.training.dynamic_scale import DynamicScale
import optax

from os import environ as env_vars
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Dict, Callable
from time import time

from .cfg import TrainConfig
from .rollouts import RolloutExecutor
from .actor_critic import ActorCritic
from .algo_common import InternalConfig
from .moving_avg import EMANormalizer
from .train_state import HyperParams, PolicyTrainState, TrainStateManager
from .profile import profile


def _update_loop(update_iter_fn : Callable,
                 gpu_sync_fn : Callable,
                 user_cb : Callable,
                 cfg : TrainConfig,
                 icfg : InternalConfig,
                 sim : dict,
                 rollout_exec : RolloutExecutor,
                 ac_functional : Callable,
                 train_state_mgr : TrainStateManager,
                 start_update_idx : int):
    for update_idx in range(start_update_idx, cfg.num_updates):
        update_start_time  = time()

        with profile("Update Iter Timing"):
            update_result = update_iter_fn(
                cfg,
                icfg,
                sim,
                rollout_exec,
                ac_functional,
                train_state_mgr.policy_states,
            )

            gpu_sync_fn()

        profile.gpu_measure()
        profile.commit()

        update_end_time = time()
        update_time = update_end_time - update_start_time
        user_cb(update_idx, update_time, update_result, train_state_mgr)

def _setup_value_normalizer(cfg, rng_key, fake_values):
    value_norm_decay = \
        cfg.value_normalizer_decay if cfg.normalize_values else 1.0

    value_normalizer = EMANormalizer(
        value_norm_decay, disable=not cfg.normalize_values)

    value_normalizer_vars = value_normalizer.init(
        rng_key, 'normalize', False, fake_values)

    return value_normalizer.apply, value_normalizer_vars

def _setup_new_policy(policy, cfg, prng_key, rnn_states, obs):
    model_init_rng, value_norm_rng = random.split(prng_key, 2)
    fake_outs, variables = policy.init_with_output(
        model_init_rng, random.PRNGKey(0), rnn_states, obs,
        method='rollout')

    params = variables['params']
    batch_stats = variables['batch_stats']

    value_norm_fn, value_norm_vars = _setup_value_normalizer(
        cfg, value_norm_rng, fake_outs[2])

    hyper_params = HyperParams(
        lr=cfg.lr,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    optimizer = optax.adam(learning_rate=hyper_params.lr)

    if cfg.mixed_precision:
        scaler = DynamicScale()
    else:
        scaler = None

    return PolicyTrainState.create(
        apply_fn = policy.apply,
        params = params,
        tx = optimizer,
        hyper_params = hyper_params,
        batch_stats = batch_stats,
        scheduler = None,
        scaler = scaler,
        value_normalize_fn = value_norm_fn,
        value_normalize_vars = value_norm_vars,
    )

def _setup_train_states(policy, cfg, icfg, base_init_rng, rollout_state):
    setup_new_policies = jax.vmap(
        partial(_setup_new_policy, policy, cfg), in_axes=(0, 0, 0))

    obs = jax.tree_map(lambda x: x.reshape(
            cfg.pbt_ensemble_size * cfg.pbt_history_len,
            icfg.rollout_agents_per_policy,
            *x.shape[1:],
        ), rollout_state.sim_data['obs'])

    setup_new_policies = jax.jit(checkify.checkify(setup_new_policies))

    init_rngs = random.split(base_init_rng, cfg.pbt_ensemble_size)

    err, train_states = setup_new_policies(
        init_rngs, rollout_state.rnn_states, obs)
    err.throw()

    return TrainStateManager(train_states=train_states)

def init(mem_fraction):
    env_vars["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_fraction:.2f}"
    #jax.config.update("jax_numpy_rank_promotion", "raise")
    jax.config.update("jax_numpy_dtype_promotion", "strict")

def train(dev, sim, cfg, policy_constructor, update_cb, restore_ckpt=None):
    print(cfg)

    icfg = InternalConfig(dev, cfg)

    meta_policy = policy_constructor().to('meta')

    train_state_mgr = TrainStateManager([
            _setup_new_policy(
                dev,
                policy_constructor,
                cfg.lr,
                cfg.value_normalizer_decay if cfg.normalize_values else None,
            ) for _ in range(cfg.pbt_ensemble_size)
        ]
    )

    if restore_ckpt != None:
        start_update_idx = train_state_mgr.load(restore_ckpt)
    else:
        start_update_idx = 0

    policy_recurrent_cfg = train_state_mgr.policy_states[0].policy.recurrent_cfg

    rollout_exec = RolloutExecutor(dev, sim, cfg, icfg, policy_recurrent_cfg)

    update_iter_fn = cfg.algo.setup(dev, cfg, icfg)

    if 'MADRONA_LEARN_COMPILE' in env_vars and \
            env_vars['MADRONA_LEARN_COMPILE'] == '1':
        if 'MADRONA_LEARN_COMPILE_DEBUG' in env_vars and \
                env_vars['MADRONA_LEARN_COMPILE_DEBUG'] == '1':
            torch._dynamo.config.verbose=True

        if 'MADRONA_LEARN_COMPILE_CXX' in env_vars:
            from torch._inductor import config as inductor_cfg
            inductor_cfg.cpp.cxx = env_vars['MADRONA_LEARN_COMPILE_CXX']

        update_iter_fn = torch.compile(update_iter_fn, dynamic=False)

    if dev.type == 'cuda':
        def gpu_sync_fn():
            torch.cuda.synchronize()
    else:
        def gpu_sync_fn():
            pass

    _update_loop(
        update_iter_fn=update_iter_fn,
        gpu_sync_fn=gpu_sync_fn,
        user_cb=update_cb,
        cfg=cfg,
        icfg=icfg,
        sim=sim,
        rollout_exec=rollout_exec,
        ac_functional=meta_policy,
        train_state_mgr=train_state_mgr,
        start_update_idx=start_update_idx,
    )

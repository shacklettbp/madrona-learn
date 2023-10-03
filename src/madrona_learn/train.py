import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

from os import environ as env_vars
from typing import Callable
from dataclasses import dataclass
from typing import List, Optional, Dict
from .profile import profile
from time import time
from pathlib import Path

from .cfg import TrainConfig, SimInterface
from .rollouts import RolloutManager, Rollouts
from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer
from .train_state import PolicyTrainState, TrainStateManager
from .utils import InternalConfig


def _update_loop(update_iter_fn : Callable,
                 gpu_sync_fn : Callable,
                 user_cb : Callable,
                 cfg : TrainConfig,
                 icfg : InternalConfig,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
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
                rollout_mgr,
                ac_functional,
                train_state_mgr.policy_states,
            )

            gpu_sync_fn()

        profile.gpu_measure()
        profile.commit()

        update_end_time = time()
        update_time = update_end_time - update_start_time
        user_cb(update_idx, update_time, update_result, train_state_mgr)


def _setup_new_policy(dev, policy_constructor, base_lr, value_norm_decay):
    policy = policy_constructor().to(dev)
    optimizer = optim.Adam(policy.parameters(), lr=base_lr)

    if amp.enabled and dev.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if value_norm_decay == None:
        value_norm_decay = 1.0
        value_norm_disable = True
    else:
        value_norm_disable = False

    value_normalizer = EMANormalizer(
        value_norm_decay, disable=value_norm_disable)
    value_normalizer = value_normalizer.to(dev)

    return PolicyTrainState(
        policy = policy,
        optimizer = optimizer,
        scheduler = None,
        scaler = scaler,
        value_normalizer = value_normalizer,
    )


def train(dev, sim, cfg, policy_constructor, update_cb, restore_ckpt=None):
    print(cfg)

    icfg = InternalConfig(dev, cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp.init(dev, cfg.mixed_precision)

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

    rollout_mgr = RolloutManager(dev, sim, cfg, icfg, policy_recurrent_cfg)

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
        rollout_mgr=rollout_mgr,
        ac_functional=meta_policy,
        train_state_mgr=train_state_mgr,
        start_update_idx=start_update_idx,
    )

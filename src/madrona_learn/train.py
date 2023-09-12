import torch
from torch import nn
import torch.nn.functional as F
import torch._dynamo
from torch import optim
from torch.func import vmap
from os import environ as env_vars
from typing import Callable
from dataclasses import dataclass
from typing import List, Optional, Dict
from .profile import profile
from time import time
from pathlib import Path

from .cfg import TrainConfig, SimInterface
from .rollouts import RolloutManager, Rollouts
from .amp import amp 
from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer
from .training_state import PolicyLearningState, TrainingState


def _update_loop(update_iter_fn : Callable,
                 gpu_sync_fn : Callable,
                 user_cb : Callable,
                 cfg : TrainConfig,
                 icfg : InternalConfig,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 training_state : TrainingState,
                 start_update_idx : int):
    for update_idx in range(start_update_idx, cfg.num_updates):
        update_start_time  = time()

        with profile("Update Iter Timing"):
            update_result = update_iter_fn(
                cfg,
                icfg,
                sim,
                rollout_mgr,
                advantages,
                training_state.policy_states,
            )

            gpu_sync_fn()

        profile.gpu_measure()
        profile.commit()

        update_end_time = time()
        update_time = update_end_time - update_start_time
        user_cb(update_idx, update_time, update_result, training_state)


def _setup_new_policy(dev, policy_constructor, base_lr, value_norm_decay):
    policy = policy_constructor().to(dev)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr)

    if amp.enabled and dev.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    value_normalizer = EMANormalizer(cfg.value_normalizer_decay,
                                     disable=not cfg.normalize_values)
    value_normalizer = value_normalizer.to(dev)

    return PolicyLearningState(
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

    training_state = TrainingState([
            _setup_new_policy(dev,
                              policy_constructor,
                              cfg.lr,
                              cfg.value_normalizer_decay,
                             )
            for _ in range(cfg.pbt_ensemble_size)
        ]
    )

    if restore_ckpt != None:
        start_update_idx = training_state.load(restore_ckpt)
    else:
        start_update_idx = 0

    policy_recurrent_cfg = training_state.policy_states[0].policy.recurrent_cfg

    rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
        cfg.num_bptt_chunks, policy_recurrent_cfg)

    update_iter_fn = cfg.algo.setup(dev, cfg)

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
        training_state=training_state,
        start_update_idx=start_update_idx,
    )

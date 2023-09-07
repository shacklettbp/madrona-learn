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
from .learning_state import LearningState


def _update_loop(update_iter_fn : Callable,
                 gpu_sync_fn : Callable,
                 user_cb : Callable,
                 cfg : TrainConfig,
                 num_agents: int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 learning_state : LearningState,
                 start_update_idx : int):
    num_train_seqs = num_agents * cfg.num_bptt_chunks

    advantages = torch.zeros_like(rollout_mgr.rewards)

    for update_idx in range(start_update_idx, cfg.num_updates):
        update_start_time  = time()

        with profile("Update Iter Timing"):
            update_result = update_iter_fn(
                cfg,
                num_train_seqs,
                sim,
                rollout_mgr,
                advantages,
                learning_state.policy,
                learning_state.optimizer,
                learning_state.scheduler,
                learning_state.value_normalizer,
            )

            gpu_sync_fn()

        profile.gpu_measure()
        profile.commit()

        update_end_time = time()
        update_time = update_end_time - update_start_time
        user_cb(update_idx, update_time, update_result, learning_state)

def train(dev, sim, cfg, actor_critic, update_cb, restore_ckpt=None):
    print(cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp.init(dev, cfg.mixed_precision)

    num_agents = sim.actions.shape[0]

    actor_critic = actor_critic.to(dev)

    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr)

    value_normalizer = EMANormalizer(cfg.value_normalizer_decay,
                                     disable=not cfg.normalize_values)
    value_normalizer = value_normalizer.to(dev)

    learning_state = LearningState(
        policy = actor_critic,
        optimizer = optimizer,
        scheduler = None,
        value_normalizer = value_normalizer,
    )

    if restore_ckpt != None:
        start_update_idx = learning_state.load(restore_ckpt)
    else:
        start_update_idx = 0

    rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
        cfg.num_bptt_chunks, actor_critic.recurrent_cfg)

    if 'MADRONA_LEARN_COMPILE' in env_vars and \
            env_vars['MADRONA_LEARN_COMPILE'] == '1':
        if 'MADRONA_LEARN_COMPILE_DEBUG' in env_vars and \
                env_vars['MADRONA_LEARN_COMPILE_DEBUG'] == '1':
            torch._dynamo.config.verbose=True

        if 'MADRONA_LEARN_COMPILE_CXX' in env_vars:
            from torch._inductor import config as inductor_cfg
            inductor_cfg.cpp.cxx = env_vars['MADRONA_LEARN_COMPILE_CXX']

        update_iter_fn = torch.compile(cfg.algo.update_iter_fn, dynamic=False)
    else:
        update_iter_fn = cfg.algo.update_iter_fn

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
        num_agents=num_agents,
        sim=sim,
        rollout_mgr=rollout_mgr,
        learning_state=learning_state,
        start_update_idx=start_update_idx,
    )

    return actor_critic.cpu()

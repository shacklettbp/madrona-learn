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

from .cfg import TrainConfig, SimInterface
from .rollouts import RolloutManager, Rollouts
from .amp import AMPInfo
from .actor_critic import ActorCritic

@dataclass(frozen = True)
class MiniBatch:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    rnn_start_states: tuple[torch.Tensor, ...]

def _mb_slice(tensor, inds):
    # Tensors come from the rollout manager as (C, T, N, ...)
    # Want to select mb from C * N and keep sequences of length T

    return tensor.transpose(0, 1).reshape(
        tensor.shape[1], tensor.shape[0] * tensor.shape[2], *tensor.shape[3:])[:, inds, ...]

def _mb_slice_rnn(rnn_state, inds):
    # RNN state comes from the rollout manager as (C, :, :, N, :)
    # Want to select minibatch from C * N and keep sequences of length T

    reshaped = rnn_state.permute(1, 2, 0, 3, 4).reshape(
        rnn_state.shape[1], rnn_state.shape[2], -1, rnn_state.shape[4])

    return reshaped[:, :, inds, :] 

def _gather_minibatch(rollouts : Rollouts,
                      advantages : torch.Tensor,
                      inds : torch.Tensor,
                      amp : AMPInfo):
    obs_slice = tuple(_mb_slice(obs, inds) for obs in rollouts.obs)
    
    actions_slice = _mb_slice(rollouts.actions, inds)
    log_probs_slice = _mb_slice(rollouts.log_probs, inds).to(
        dtype=amp.compute_dtype)
    dones_slice = _mb_slice(rollouts.dones, inds)
    rewards_slice = _mb_slice(rollouts.rewards, inds).to(
        dtype=amp.compute_dtype)
    values_slice = _mb_slice(rollouts.values, inds).to(
        dtype=amp.compute_dtype)
    advantages_slice = _mb_slice(advantages, inds).to(
        dtype=amp.compute_dtype)

    rnn_starts_slice = tuple(
        _mb_slice_rnn(state, inds) for state in rollouts.rnn_start_states)

    return MiniBatch(
        obs=obs_slice,
        actions=actions_slice,
        log_probs=log_probs_slice,
        dones=dones_slice,
        rewards=rewards_slice,
        values=values_slice,
        advantages=advantages_slice,
        rnn_start_states=rnn_starts_slice,
    )

def _compute_advantages(cfg : TrainConfig,
                        amp : AMPInfo,
                        advantages_out : torch.Tensor,
                        rollouts : Rollouts):
    # This function is going to be operating in fp16 mode completely
    # when mixed precision is enabled since amp.compute_dtype is fp16
    # even though there is no autocast here. Unclear if this is desirable or
    # even beneficial for performance.

    num_chunks, steps_per_chunk, N = rollouts.dones.shape[0:3]
    T = num_chunks * steps_per_chunk

    seq_dones = rollouts.dones.view(T, N, 1)
    seq_rewards = rollouts.rewards.view(T, N, 1)
    seq_values = rollouts.values.view(T, N, 1)
    seq_advantages_out = advantages_out.view(T, N, 1)

    next_advantage = 0.0
    next_values = rollouts.bootstrap_values
    for i in reversed(range(cfg.steps_per_update)):
        cur_dones = seq_dones[i].to(dtype=amp.compute_dtype)
        cur_rewards = seq_rewards[i].to(dtype=amp.compute_dtype)
        cur_values = seq_values[i].to(dtype=amp.compute_dtype)

        next_valid = 1.0 - cur_dones

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        td_err = (cur_rewards + 
            cfg.gamma * next_valid * next_values - cur_values)

        # A_t = sum (gamma * lambda)^(l - 1) * delta_l (EQ 16 GAE)
        #     = delta_t + gamma * lambda * A_t+1
        cur_advantage = (td_err +
            cfg.gamma * cfg.gae_lambda * next_valid * next_advantage)

        seq_advantages_out[i] = cur_advantage

        next_advantage = cur_advantage
        next_values = cur_values

def _compute_action_scores(cfg, amp, advantages):
    if not cfg.normalize_advantages:
        return advantages
    else:
        # Unclear from docs if var_mean is safe under autocast
        with amp.disable():
            var, mean = torch.var_mean(advantages.to(dtype=torch.float32))
            action_scores = advantages - mean
            action_scores.mul_(torch.rsqrt(var + 1e-5))

            return action_scores.to(dtype=amp.compute_dtype)

def _ppo_update(cfg : TrainConfig,
                amp : AMPInfo,
                mb : MiniBatch,
                actor_critic : ActorCritic,
                optimizer : torch.optim.Optimizer):
    with amp.enable():
        with profile('AC Forward', gpu=True):
            new_log_probs, entropies, new_values = actor_critic.train(
                mb.rnn_start_states, mb.dones, mb.actions, *mb.obs)

        with torch.no_grad():
            action_scores = _compute_action_scores(cfg, amp, mb.advantages)

        ratio = torch.exp(new_log_probs - mb.log_probs)
        surr1 = action_scores * ratio
        surr2 = action_scores * (
            torch.clamp(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef))

        action_obj = torch.min(surr1, surr2)

        returns = mb.advantages + mb.values

        if cfg.ppo.clip_value_loss:
            with torch.no_grad():
                low = mb.values - cfg.ppo.clip_coef
                high = mb.values + cfg.ppo.clip_coef

            new_values = torch.clamp(new_values, low, high)

        value_loss = 0.5 * F.mse_loss(new_values, returns, reduction='none')

        action_obj = torch.mean(action_obj)
        value_loss = torch.mean(value_loss)
        entropies = torch.mean(entropies)

        loss = (
            - action_obj # Maximize the action objective function
            + cfg.ppo.value_loss_coef * value_loss
            - cfg.ppo.entropy_coef * entropies # Maximize entropy
        )

    with profile('Optimize'):
        if amp.scaler is None:
            loss.backward()
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            optimizer.step()
        else:
            amp.scaler.scale(loss).backward()
            amp.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            amp.scaler.step(optimizer)
            amp.scaler.update()

        optimizer.zero_grad()

    with torch.no_grad():
        print(f"    Loss: {loss.cpu().float().item()} {-action_obj.cpu().float().item()} {value_loss.cpu().float().item()} {-entropies.cpu().float().item()}")

def _update_iter(cfg : TrainConfig,
                 amp : AMPInfo,
                 num_train_seqs : int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 actor_critic : ActorCritic,
                 advantages : torch.Tensor,
                 optimizer,
                 scheduler):
    with torch.no_grad():
        with profile('Collect Rollouts'):
            rollouts = rollout_mgr.collect(amp, sim, actor_critic)
    
        # Engstrom et al suggest recomputing advantages after every epoch
        # but that's pretty annoying for a recurrent policy since values
        # need to be recomputed. https://arxiv.org/abs/2005.12729
        with profile('Compute Advantages'):
            _compute_advantages(cfg, amp, advantages, rollouts)
    
    with profile('PPO'):
        for epoch in range(cfg.ppo.num_epochs):
            for inds in torch.randperm(num_train_seqs).chunk(
                    cfg.ppo.num_mini_batches):
                with torch.no_grad(), profile('Gather Minibatch', gpu=True):
                    mb = _gather_minibatch(rollouts, advantages, inds, amp)
                _ppo_update(cfg, amp, mb, actor_critic, optimizer)

def _update_loop(update_iter_fn : Callable,
                 gpu_sync_fn : Callable,
                 cfg : TrainConfig,
                 amp : AMPInfo,
                 num_agents: int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 actor_critic : ActorCritic,
                 optimizer,
                 scheduler):
    num_train_seqs = num_agents * cfg.num_bptt_chunks
    assert(num_train_seqs % cfg.ppo.num_mini_batches == 0)

    advantages = torch.zeros_like(rollout_mgr.rewards)

    outer_timing_mean = 0
    for update_idx in range(cfg.num_updates):
        start_outer = time()

        if update_idx % 1 == 0:
            print(f'\nUpdate: {update_idx}')

        with profile("Update Iter Timing"):
            update_iter_fn(cfg,
                           amp,
                           num_train_seqs,
                           sim,
                           rollout_mgr,
                           actor_critic,
                           advantages,
                           optimizer,
                           scheduler)

            gpu_sync_fn()

        profile.gpu_measure()
        profile.commit()
        profile.report()

        end_outer = time()
        outer_diff = end_outer - start_outer
        outer_timing_mean += (outer_diff - outer_timing_mean) / (update_idx + 1)
        print(f"    Outer Loop Timing: {outer_timing_mean:.3f}")

def train(dev, sim, cfg, actor_critic, update_cb):
    print(cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    amp = AMPInfo(dev, cfg.mixed_precision)

    num_agents = sim.actions.shape[0]

    actor_critic = actor_critic.to(dev)

    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr)

    rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
        cfg.num_bptt_chunks, amp, actor_critic.recurrent_cfg)

    if 'MADRONA_LEARN_COMPILE' in env_vars and \
            env_vars['MADRONA_LEARN_COMPILE'] == '1':
        if 'MADRONA_LEARN_COMPILE_DEBUG' in env_vars and \
                env_vars['MADRONA_LEARN_COMPILE_DEBUG'] == '1':
            torch._dynamo.config.verbose=True

        if 'MADRONA_LEARN_COMPILE_CXX' in env_vars:
            from torch._inductor import config as inductor_cfg
            inductor_cfg.cpp.cxx = env_vars['MADRONA_LEARN_COMPILE_CXX']

        update_iter_fn = torch.compile(_update_iter, dynamic=False)
    else:
        update_iter_fn = _update_iter

    if dev.type == 'cuda':
        def gpu_sync_fn():
            torch.cuda.synchronize()
    else:
        def gpu_sync_fn():
            pass

    _update_loop(
        update_iter_fn=update_iter_fn,
        gpu_sync_fn=gpu_sync_fn,
        cfg=cfg,
        amp=amp,
        num_agents=num_agents,
        sim=sim,
        rollout_mgr=rollout_mgr,
        actor_critic=actor_critic,
        optimizer=optimizer,
        scheduler=None,
    )

    return actor_critic.cpu()

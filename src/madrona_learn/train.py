import torch
from torch import nn
import torch.nn.functional as F
import torch._dynamo
from torch import optim
from torch.func import vmap
from time import time
from os import environ as env_vars
from typing import Callable
from dataclasses import dataclass
from typing import List, Optional, Dict

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
    rnn_hidden_starts: Optional[torch.Tensor]

def _gather_minibatch(rollouts : Rollouts,
                      advantages : torch.Tensor,
                      inds : torch.Tensor,
                      amp : AMPInfo):
    obs_slice = [obs[:, inds, ...] for obs in rollouts.obs]
    
    actions_slice = rollouts.actions[:, inds, ...]
    log_probs_slice = rollouts.log_probs[:, inds, ...].to(dtype=amp.compute_dtype)
    dones_slice = rollouts.dones[:, inds, ...]
    rewards_slice = rollouts.rewards[:, inds, ...].to(dtype=amp.compute_dtype)
    values_slice = rollouts.values[:, inds, ...].to(dtype=amp.compute_dtype)
    advantages_slice = advantages[:, inds, ...].to(dtype=amp.compute_dtype)

    if rollouts.rnn_hidden_starts != None:
        rnn_hidden_starts_slice = rollouts.rnn_hidden_starts[:, :, inds, ...]
    else:
        rnn_hidden_starts_slice = None
    
    return MiniBatch(
        obs=obs_slice,
        actions=actions_slice,
        log_probs=log_probs_slice,
        dones=dones_slice,
        rewards=rewards_slice,
        values=values_slice,
        advantages=advantages_slice,
        rnn_hidden_starts=rnn_hidden_starts_slice,
    )

def _compute_advantages(cfg : TrainConfig,
                        amp : AMPInfo,
                        advantages_out : torch.Tensor,
                        rollouts : Rollouts):
    # This function is going to be operating in fp16 mode completely
    # when mixed precision is enabled since amp.compute_dtype is fp16
    # even though there is no autocast here. Unclear if this is desirable or
    # even beneficial for performance.

    next_advantage = 0.0
    next_values = rollouts.bootstrap_values
    for i in reversed(range(cfg.steps_per_update)):
        cur_dones = rollouts.dones[i].to(dtype=amp.compute_dtype)
        cur_rewards = rollouts.rewards[i].to(dtype=amp.compute_dtype)
        cur_values = rollouts.values[i].to(dtype=amp.compute_dtype)

        next_valid = 1.0 - cur_dones

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        td_err = (cur_rewards + 
            cfg.gamma * next_valid * next_values - cur_values)

        # A_t = sum (gamma * lambda)^(l - 1) * delta_l (EQ 16 GAE)
        #     = delta_t + gamma * lambda * A_t+1
        cur_advantage = (td_err +
            cfg.gamma * cfg.gae_lambda * next_valid * next_advantage)

        advantages_out[i] = cur_advantage

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
        if mb.rnn_hidden_starts is not None:
            new_log_probs, entropies, new_values = actor_critic.train(
                mb.rnn_hidden_starts, mb.dones, mb.actions, *mb.obs)
        else:
            new_log_probs, entropies, new_values = actor_critic.train(
                mb.actions, *mb.obs)

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

def _update_loop(cfg : TrainConfig,
                 amp : AMPInfo,
                 num_agents: int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 actor_critic : ActorCritic,
                 optimizer,
                 scheduler):
    assert(num_agents % cfg.ppo.num_mini_batches == 0)

    advantages = torch.zeros_like(rollout_mgr.rewards)

    for update_idx in range(cfg.num_updates):
        update_start_time = time()

        if update_idx % 1 == 0:
            print(f'Update: {update_idx}')

        with torch.no_grad():
            rollouts = rollout_mgr.collect(amp, sim, actor_critic)

            # Engstrom et al suggest recomputing advantages after every epoch
            # but that's pretty annoying for a recurrent policy since values
            # need to be recomputed. https://arxiv.org/abs/2005.12729
            _compute_advantages(cfg, amp, advantages, rollouts)

        for epoch in range(cfg.ppo.num_epochs):
            for inds in torch.randperm(num_agents).chunk(
                    cfg.ppo.num_mini_batches):
                mb = _gather_minibatch(rollouts, advantages, inds, amp)
                _ppo_update(cfg, amp, mb, actor_critic, optimizer)

        update_end_time = time()

        print(f"    Time: {update_end_time - update_start_time}\n")

def train(sim, cfg, actor_critic, dev):
    print(cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    amp = AMPInfo(dev, cfg.mixed_precision)

    num_agents = sim.actions.shape[0]

    actor_critic = actor_critic.to(dev)

    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr)

    rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
        amp, actor_critic.rnn_hidden_shape)

    if 'MADRONA_LEARN_COMPILE' in env_vars and \
            env_vars['MADRONA_LEARN_COMPILE'] == '1':
        if 'MADRONA_LEARN_COMPILE_DEBUG' in env_vars and \
                env_vars['MADRONA_LEARN_COMPILE_DEBUG'] == '1':
            torch._dynamo.config.verbose=True

        if 'MADRONA_LEARN_COMPILE_CXX' in env_vars:
            from torch._inductor import config as inductor_cfg
            inductor_cfg.cpp.cxx = env_vars['MADRONA_LEARN_COMPILE_CXX']

        update_loop = torch.compile(_update_loop, dynamic=False)
    else:
        update_loop = _update_loop

    update_loop(
        cfg=cfg,
        amp=amp,
        num_agents=num_agents,
        sim=sim,
        rollout_mgr=rollout_mgr,
        actor_critic=actor_critic,
        optimizer=optimizer,
        scheduler=None)

    return actor_critic.cpu()

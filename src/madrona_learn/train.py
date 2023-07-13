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
from typing import List, Optional

from .cfg import TrainConfig, SimInterface
from .rollouts import RolloutManager, Rollouts

@dataclass(frozen = True)
class PolicyInterface:
    actor_critic: nn.Module
    train_fwd: Callable
    rollout_infer: Callable
    rollout_infer_values: Callable

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
                      float_compute_type : torch.dtype):
    obs_slice = [obs[:, inds, ...] for obs in rollouts.obs]
    
    actions_slice = rollouts.actions[:, inds, ...]
    log_probs_slice = rollouts.log_probs[:, inds, ...].to(dtype=float_compute_type)
    dones_slice = rollouts.dones[:, inds, ...]
    rewards_slice = rollouts.rewards[:, inds, ...].to(dtype=float_compute_type)
    values_slice = rollouts.values[:, inds, ...].to(dtype=float_compute_type)
    advantages_slice = advantages[:, inds, ...].to(dtype=float_compute_type)

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
                        advantages_out : torch.Tensor,
                        rollouts : Rollouts,
                        float_compute_type : torch.dtype):
    next_advantage = 0.0
    next_values = rollouts.bootstrap_values
    for i in reversed(range(cfg.steps_per_update)):
        cur_dones = rollouts.dones[i].to(dtype=float_compute_type)
        cur_rewards = rollouts.rewards[i].to(dtype=float_compute_type)
        cur_values = rollouts.values[i].to(dtype=float_compute_type)

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

def _compute_action_scores(cfg, advantages):
    with torch.no_grad():
        if not cfg.normalize_advantages:
            return advantages
        else:
            var, mean = torch.var_mean(advantages)

            action_scores = advantages - mean
            action_scores.mul_(torch.rsqrt(var + 1e-5))

            return action_scores

def _ppo_update(cfg : TrainConfig,
                policy : PolicyInterface,
                mb : MiniBatch,
                optimizer,
                scaler):
    if mb.rnn_hidden_starts is not None:
        new_log_probs, entropies, new_values = policy.train_fwd(
            mb.rnn_hidden_starts, mb.dones, mb.actions, *mb.obs)
    else:
        new_log_probs, entropies, new_values = policy.train_fwd(
            mb.actions, *mb.obs)

    action_scores = _compute_action_scores(cfg, mb.advantages)

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

    with torch.no_grad():
        print(f"    Loss: {loss.cpu().float().item()} {-action_obj.cpu().float().item()} {value_loss.cpu().float().item()} {-entropies.cpu().float().item()}")

    if scaler is None:
        loss.backward()
        nn.utils.clip_grad_norm_(policy.actor_critic.parameters(), cfg.ppo.max_grad_norm)
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(policy.actor_critic.parameters(), cfg.ppo.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

    optimizer.zero_grad()

def _update_loop(cfg : TrainConfig,
                 num_agents: int,
                 float_compute_type: torch.dtype,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 policy : PolicyInterface,
                 optimizer,
                 scaler,
                 scheduler):
    assert(num_agents % cfg.ppo.num_mini_batches == 0)

    advantages = torch.zeros_like(rollout_mgr.rewards)

    for update_idx in range(cfg.num_updates):
        update_start_time = time()

        if update_idx % 1 == 0:
            print(f'Update: {update_idx}')

        with torch.no_grad():
            rollouts = rollout_mgr.collect(sim, policy.rollout_infer,
                policy.rollout_infer_values)

            # Engstrom et al suggest recomputing advantages after every epoch
            # but that's pretty annoying for a recurrent policy since values
            # need to be recomputed. https://arxiv.org/abs/2005.12729
            _compute_advantages(cfg, advantages, rollouts, float_compute_type)

        for epoch in range(cfg.ppo.num_epochs):
            for inds in torch.randperm(num_agents).chunk(
                    cfg.ppo.num_mini_batches):
                mb = _gather_minibatch(rollouts, advantages, inds,
                                       float_compute_type)
                _ppo_update(cfg, policy, mb, optimizer, scaler)

        update_end_time = time()

        print(f"    Time: {update_end_time - update_start_time}\n")

def train(sim, cfg, actor_critic, dev):
    print(cfg)

    num_agents = sim.actions.shape[0]

    actor_critic = actor_critic.to(dev)

    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr)

    enable_mixed_precision = dev.type == 'cuda' and cfg.mixed_precision

    if enable_mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if enable_mixed_precision:
        def autocast_wrapper(fn):
            def autocast_wrapped_fn(*args, **kwargs):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    return fn(*args, **kwargs)

            return autocast_wrapped_fn
    else:
        def autocast_wrapper(fn):
            return fn

    def policy_train_fwd(*args, **kwargs):
        return actor_critic.train(*args, **kwargs)

    def policy_rollout_infer(*args, **kwargs):
        return actor_critic.rollout_infer(*args, **kwargs)

    def policy_rollout_infer_values(*args, **kwargs):
        return actor_critic.rollout_infer_values(*args, **kwargs)

    policy_train_fwd = autocast_wrapper(policy_train_fwd)
    policy_rollout_infer = autocast_wrapper(policy_rollout_infer)
    policy_rollout_infer_values = autocast_wrapper(policy_rollout_infer_values)

    policy_interface = PolicyInterface(
        actor_critic = actor_critic,
        train_fwd = policy_train_fwd,
        rollout_infer = policy_rollout_infer,
        rollout_infer_values = policy_rollout_infer_values,
    )

    compute_dtype = torch.float16 if enable_mixed_precision else torch.float32
    rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
        compute_dtype, actor_critic.rnn_hidden_shape)

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
        num_agents=num_agents,
        float_compute_type=compute_dtype,
        sim=sim,
        rollout_mgr=rollout_mgr,
        policy=policy_interface,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=None)

    return actor_critic.cpu()

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

from .cfg import TrainConfig, SimInterface
from .rollouts import RolloutManager, RolloutMiniBatch

@dataclass(frozen = True)
class PolicyInterface:
    actor_critic: nn.Module
    train_fwd: Callable
    rollout_infer: Callable
    rollout_infer_values: Callable

def _ppo_update(cfg : TrainConfig,
                policy : PolicyInterface,
                mb : RolloutMiniBatch,
                optimizer,
                scaler):

    if mb.rnn_hidden_starts is not None:
        new_log_probs, entropies, new_values = policy.train_fwd(
            mb.rnn_hidden_starts, mb.dones, mb.actions, *mb.obs)
    else:
        new_log_probs, entropies, new_values = policy.train_fwd(
            mb.actions, *mb.obs)

    ratio = torch.exp(new_log_probs - mb.log_probs)

    surr1 = mb.advantages * ratio
    surr2 = mb.advantages * (
        torch.clamp(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef))

    action_loss = -torch.min(surr1, surr2)

    returns = mb.advantages + mb.values

    if cfg.ppo.clip_value_loss:
        with torch.no_grad():
            low = mb.values - cfg.ppo.clip_coef
            high = mb.values + cfg.ppo.clip_coef

        new_values = torch.clamp(low, high)

    value_loss = 0.5 * F.mse_loss(new_values, returns, reduction='none')

    action_loss = torch.mean(action_loss)
    value_loss = torch.mean(value_loss)
    entropies = torch.mean(entropies)

    loss = (
        action_loss +
        cfg.ppo.value_loss_coef * value_loss +
        cfg.ppo.entropy_coef * entropies
    )

    with torch.no_grad():
        print(f"    Loss: {loss.cpu().float().item()} {action_loss.cpu().float().item()} {value_loss.cpu().float().item()}")

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
                 sim : SimInterface,
                 rollouts : RolloutManager,
                 policy : PolicyInterface,
                 optimizer,
                 scaler,
                 scheduler):
    num_agents = rollouts.actions.shape[1]

    assert(num_agents % cfg.ppo.num_mini_batches == 0)

    for update_idx in range(cfg.num_updates):
        update_start_time = time()

        if update_idx % 1 == 0:
            print(f'Update: {update_idx}')

        with torch.no_grad():
            rollouts.collect(sim, policy.rollout_infer,
                             policy.rollout_infer_values)

        for epoch in range(cfg.ppo.num_epochs):
            for inds in torch.randperm(num_agents).chunk(
                    cfg.ppo.num_mini_batches):
                mb = rollouts.gather_minibatch(inds)
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
    rollouts = RolloutManager(dev, sim, cfg.steps_per_update, cfg.gamma,
        cfg.gae_lambda, compute_dtype, actor_critic.rnn_hidden_shape)

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
        sim=sim,
        rollouts=rollouts,
        policy=policy_interface,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=None)

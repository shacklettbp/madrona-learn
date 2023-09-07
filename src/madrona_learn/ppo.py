from dataclasses import dataclass
import torch

from .cfg import Algorithm, TrainConfig
from .rollouts import RolloutManager, Rollouts

from .train_common import (
        MiniBatch, UpdateResult, 
        compute_advantages, compute_action_scores, gather_minibatch
    )

__all__ = [ "PPOConfig", "cfg_standard_ppo", "cfg_competitive_ppo" ]

@dataclass(frozen=True)
class PPOConfig:
    num_mini_batches: int
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    num_epochs: int = 1
    clip_value_loss: bool = False
    adaptive_entropy: bool = True


@dataclass
class PPOStats:
    loss : float = 0
    action_loss : float = 0
    value_loss : float = 0
    entropy_loss : float = 0
    returns_mean : float = 0
    returns_stddev : float = 0


def _ppo_update(cfg : TrainConfig,
                mb : MiniBatch,
                actor_critic : ActorCritic,
                optimizer : torch.optim.Optimizer,
                value_normalizer : EMANormalizer,
            ):
    with amp.enable():
        with profile('AC Forward', gpu=True):
            new_log_probs, entropies, new_values = actor_critic.fwd_update(
                mb.rnn_start_states, mb.dones, mb.actions, *mb.obs)

        with torch.no_grad():
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

        normalized_returns = value_normalizer(returns)
        value_loss = 0.5 * F.mse_loss(
            new_values, normalized_returns, reduction='none')

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
        returns_var, returns_mean = torch.var_mean(normalized_returns)
        returns_stddev = torch.sqrt(returns_var)

        stats = PPOStats(
            loss = loss.cpu().float().item(),
            action_loss = -(action_obj.cpu().float().item()),
            value_loss = value_loss.cpu().float().item(),
            entropy_loss = -(entropies.cpu().float().item()),
            returns_mean = returns_mean.cpu().float().item(),
            returns_stddev = returns_stddev.cpu().float().item(),
        )

    return stats


def ppo(cfg: TrainConfig,
        num_train_seqs : int,
        sim : SimInterface,
        rollout_mgr : RolloutManager,
        advantages : torch.Tensor,
        actor_critic : ActorCritic,
        optimizer : torch.optim.Optimizer,
        scheduler : torch.optim.lr_scheduler.LRScheduler,
        value_normalizer : EMANormalizer
        ):
    with torch.no_grad():
        actor_critic.eval()
        value_normalizer.eval()

        with profile('Collect Rollouts'):
            rollouts = rollout_mgr.collect(sim, actor_critic)
    
        # Engstrom et al suggest recomputing advantages after every epoch
        # but that's pretty annoying for a recurrent policy since values
        # need to be recomputed. https://arxiv.org/abs/2005.12729
        with profile('Compute Advantages'):
            _compute_advantages(cfg,
                                value_normalizer,
                                advantages,
                                rollouts)
    
    actor_critic.train()
    value_normalizer.train()

    with profile('PPO'):
        aggregate_stats = PPOStats()
        num_stats = 0

        for epoch in range(cfg.ppo.num_epochs):
            for inds in torch.randperm(num_train_seqs).chunk(
                    cfg.ppo.num_mini_batches):
                with torch.no_grad(), profile('Gather Minibatch', gpu=True):
                    mb = _gather_minibatch(rollouts, advantages, inds)
                cur_stats = _ppo_update(cfg,
                                        mb,
                                        actor_critic,
                                        optimizer,
                                        value_normalizer)

                with torch.no_grad():
                    num_stats += 1
                    aggregate_stats.loss += (cur_stats.loss - aggregate_stats.loss) / num_stats
                    aggregate_stats.action_loss += (
                        cur_stats.action_loss - aggregate_stats.action_loss) / num_stats
                    aggregate_stats.value_loss += (
                        cur_stats.value_loss - aggregate_stats.value_loss) / num_stats
                    aggregate_stats.entropy_loss += (
                        cur_stats.entropy_loss - aggregate_stats.entropy_loss) / num_stats
                    aggregate_stats.returns_mean += (
                        cur_stats.returns_mean - aggregate_stats.returns_mean) / num_stats
                    # FIXME
                    aggregate_stats.returns_stddev += (
                        cur_stats.returns_stddev - aggregate_stats.returns_stddev) / num_stats

    return UpdateResult(
        actions = rollouts.actions.view(-1, *rollouts.actions.shape[2:]),
        rewards = rollouts.rewards.view(-1, *rollouts.rewards.shape[2:]),
        values = rollouts.values.view(-1, *rollouts.values.shape[2:]),
        advantages = advantages.view(-1, *advantages.shape[2:]),
        bootstrap_values = rollouts.bootstrap_values,
        ppo_stats = aggregate_stats,
    )


def cfg_standard_ppo(cfg: PPOConfig):
    return Algorithm(name='ppo', cfg=cfg, update_iter_fn=ppo)


def cfg_competitive_ppo(cfg: PPOConfig):
    return Algorithm(name='ppo', cfg=cfg, update_iter_fn=ppo)

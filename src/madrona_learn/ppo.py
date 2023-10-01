import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import optax

from dataclasses import dataclass
from typing import List, Callable

from .actor_critic import ActorCritic
from .amp import amp
from .cfg import AlgoConfig, TrainConfig, SimInterface
from .moving_avg import EMANormalizer
from .rollouts import RolloutManager, Rollouts
from .profile import profile
from .training_state import HyperParams, PolicyLearningState

from .algo_common import (
        MiniBatch, UpdateResult,
        compute_advantages, compute_action_scores, gather_minibatch
    )
from .utils import InternalConfig

__all__ = [ "PPOConfig" ]

@dataclass(frozen=True)
class PPOConfig(AlgoConfig):
    num_mini_batches: int
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    num_epochs: int = 1
    clip_value_loss: bool = False
    adaptive_entropy: bool = True

    def name(self):
        return "ppo"

    def setup(self,
              dev: jax.Device,
              cfg: TrainConfig,
              icfg: InternalConfig):
        return PPO(dev, cfg, icfg)


@dataclass
class PPOHyperParams(HyperParams):
    clip_coef: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float


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
                optimizer : optax.GradientTransformation,
                value_normalizer : EMANormalizer,
            ):
    with amp.enable():
        with profile('AC Forward', gpu=True):
            new_log_probs, entropies, new_values = actor_critic.fwd_update(
                mb.rnn_start_states, mb.dones, mb.actions, *mb.obs)

        with torch.no_grad():
            action_scores = compute_action_scores(cfg, mb.advantages)

        ratio = torch.exp(new_log_probs - mb.log_probs)
        surr1 = action_scores * ratio
        surr2 = action_scores * (
            torch.clamp(ratio, 1.0 - cfg.algo.clip_coef, 1.0 + cfg.algo.clip_coef))

        action_obj = torch.min(surr1, surr2)

        returns = mb.advantages + mb.values

        if cfg.algo.clip_value_loss:
            with torch.no_grad():
                low = mb.values - cfg.algo.clip_coef
                high = mb.values + cfg.algo.clip_coef

            new_values = torch.clamp(new_values, low, high)

        normalized_returns = value_normalizer(returns)
        value_loss = 0.5 * F.mse_loss(
            new_values, normalized_returns, reduction='none')

        action_obj = torch.mean(action_obj)
        value_loss = torch.mean(value_loss)
        entropies = torch.mean(entropies)

        loss = (
            - action_obj # Maximize the action objective function
            + cfg.algo.value_loss_coef * value_loss
            - cfg.algo.entropy_coef * entropies # Maximize entropy
        )

    with profile('Optimize'):
        if amp.scaler is None:
            loss.backward()
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.algo.max_grad_norm)
            optimizer.step()
        else:
            amp.scaler.scale(loss).backward()
            amp.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.algo.max_grad_norm)
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


def _ppo(
        cfg : TrainConfig,
        icfg : InternalConfig,
        sim : SimInterface,
        rollout_mgr : RolloutManager,
        advantages : jax.Array,
        ac_functional : Callable,
        policy_states : List[PolicyLearningState],
    ):
    with torch.no_grad():
        for state in policy_states:
            state.policy.eval()
            state.value_normalizer.eval()

        with profile('Collect Rollouts'):
            rollouts = rollout_mgr.collect(sim, ac_functional, policy_states)
    
        # Engstrom et al suggest recomputing advantages after every epoch
        # but that's pretty annoying for a recurrent policy since values
        # need to be recomputed. https://arxiv.org/abs/2005.12729
        with profile('Compute Advantages'):
            compute_advantages(cfg,
                               value_normalizer,
                               advantages,
                               rollouts)
    
    for state in policy_states:
        state.policy.train()
        state.value_normalizer.train()

    with profile('PPO'):
        aggregate_stats = PPOStats()
        num_stats = 0

        for epoch in range(cfg.algo.num_epochs):
            for inds in torch.randperm(icfg.num_train_seqs).chunk(
                    cfg.algo.num_mini_batches):
                with torch.no_grad(), profile('Gather Minibatch', gpu=True):
                    mb = gather_minibatch(rollouts, advantages, inds)
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
        algo_stats = aggregate_stats,
    )


def _ensemble_ppo(cfg : TrainConfig,
                  icfg : InternalConfig,
                  sim : SimInterface,
                  rollout_mgr : RolloutManager,
                  advantages : jax.Array,
                  policy_states : List[PolicyLearningState],
                 ):
    pass


class PPO:
    def __init__(self, dev, cfg, icfg):
        assert(icfg.num_train_seqs % cfg.algo.num_mini_batches == 0)

        self.advantages = torch.zeros(
            (cfg.num_bptt_chunks,
             icfg.num_bptt_steps,
             icfg.num_train_agents,
             1),
            dtype=icfg.float_storage_type, device=dev)

    def __call__(
            self,
            cfg: TrainConfig,
            icfg: InternalConfig,
            sim : SimInterface,
            rollout_mgr : RolloutManager,
            ac_functional : Callable,
            policy_states : List[PolicyLearningState],
        ):
        return _ppo(cfg,
                    icfg,
                    sim,
                    rollout_mgr,
                    self.advantages,
                    ac_functional,
                    policy_states)

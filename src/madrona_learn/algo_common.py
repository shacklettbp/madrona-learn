import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

from dataclasses import dataclass
from typing import List, Dict

from .cfg import TrainConfig
from .moving_avg import EMANormalizer
from .utils import DataclassProtocol

@dataclass
class InternalConfig:
    rollout_batch_size: int
    rollout_agents_per_policy: int
    rollout_batch_size_per_policy: int
    num_train_agents: int
    train_agents_per_policy: int
    num_train_seqs: int
    num_bptt_steps: int
    float_storage_type : jnp.dtype

    def __init__(self, dev, cfg):
        self.rollout_batch_size = \
            cfg.num_teams * cfg.team_size * cfg.num_worlds

        assert(cfg.num_worlds % cfg.pbt_ensemble_size == 0)

        self.rollout_agents_per_policy = self.rollout_batch_size // (
            cfg.pbt_ensemble_size * cfg.pbt_history_len)

        self.num_train_agents = cfg.team_size * cfg.num_worlds
        self.train_agents_per_policy = \
            self.num_train_agents // cfg.pbt_ensemble_size

        assert(cfg.steps_per_update % cfg.num_bptt_chunks == 0)
        self.num_train_seqs = self.num_train_agents * cfg.num_bptt_chunks
        self.num_bptt_steps = cfg.steps_per_update // cfg.num_bptt_chunks

        if dev.platform == 'cpu' and cfg.mixed_precision:
            self.float_storage_type = jnp.bfloat16
        else:
            self.float_storage_type = jnp.float16


@dataclass(frozen = True)
class MiniBatch:
    obs: List[jax.Array]
    actions: jax.Array
    log_probs: jax.Array
    dones: jax.Array
    rewards: jax.Array
    values: jax.Array
    advantages: jax.Array
    rnn_start_states: tuple[jax.Array, ...]


@dataclass(frozen = True)
class UpdateResult:
    actions : jax.Array
    rewards : jax.Array
    values : jax.Array
    advantages : jax.Array
    bootstrap_values : jax.Array
    algo_stats : DataclassProtocol


def compute_advantages(cfg : TrainConfig,
                       value_normalizer : EMANormalizer,
                       advantages_out : jax.Array,
                       rollouts : Dict):
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
    next_values = value_normalizer.invert(rollouts.bootstrap_values)
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


def compute_action_scores(cfg, advantages):
    if not cfg.normalize_advantages:
        return advantages
    else:
        # Unclear from docs if var_mean is safe under autocast
        with amp.disable():
            var, mean = torch.var_mean(advantages.to(dtype=torch.float32))
            action_scores = advantages - mean
            action_scores.mul_(torch.rsqrt(var.clamp(min=1e-5)))

            return action_scores.to(dtype=amp.compute_dtype)


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

def gather_minibatch(
    rollouts : Dict,
    advantages : jax.Array,
    inds : jax.Array
):
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

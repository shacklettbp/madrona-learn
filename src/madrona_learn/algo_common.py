import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict

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

        if cfg.mixed_precision:
            if dev.platform == 'gpu':
                self.float_storage_type = jnp.float16
            else:
                self.float_storage_type = jnp.bfloat16
        else:
            self.float_storage_type = jnp.float32


@dataclass(frozen = True)
class UpdateStats:
    actions : jax.Array
    rewards : jax.Array
    values : jax.Array
    advantages : jax.Array
    bootstrap_values : jax.Array
    algo_stats : DataclassProtocol


def compute_returns(cfg: TrainConfig,
                    rollouts: FrozenDict):
    num_chunks, steps_per_chunk, P, B = rollouts['dones'].shape[0:4]

    T = num_chunks * steps_per_chunk
    N = P * B

    seq_dones, seq_rewards = jax.tree_map(
        lambda x: x.shape(T, N, 1), (rollouts['dones'], rollouts['rewards']))

    bootstrap_values = rollouts['bootstrap_values']
    bootstrap_values = bootstrap_values.reshape(-1, 1)

    returns = jnp.empty_like(seq_rewards)
    zero = jnp.zeros((), dtype=seq_rewards.dtype)

    def return_step(i_fwd, inputs):
        i = T - 1 - i_fwd
        next_return, returns = inputs

        cur_dones = seq_dones[i]
        cur_rewards = seq_rewards[i]

        next_return = jnp.where(cur_dones, zero, next_return)

        cur_return = cur_rewards + cfg.gamma * next_return

        returns = returns.at[i].set(cur_return)

        return cur_return, returns

    next_return, returns = lax.fori_loop(
        0, T, return_step, (bootstrap_values, returns))

    returns = returns.reshape(num_chunks, steps_per_chunk, P, B)

    return rollouts.copy({
        'returns': returns
    })

def compute_advantages(cfg: TrainConfig,
                       rollouts: FrozenDict):
    num_chunks, steps_per_chunk, P, B = rollouts['dones'].shape[0:4]

    T = num_chunks * steps_per_chunk
    N = P * B

    seq_dones, seq_rewards, seq_values = jax.tree_map(
        lambda x: x.reshape(T, N, 1),
        (rollouts['dones'], rollouts['rewards'], rollouts['values']))

    bootstrap_values = rollouts['bootstrap_values']
    bootstrap_values = bootstrap_values.reshape(-1, 1)

    advantages = jnp.empty_like(seq_rewards)

    zero = jnp.zeros((), dtype=seq_rewards.dtype)

    def advantage_step(i_fwd, inputs):
        i = T - 1 - i_fwd
        next_advantage, next_values, advantages = inputs

        cur_dones = seq_dones[i]
        cur_rewards = seq_rewards[i]
        cur_values = seq_values[i]

        next_values = jnp.where(cur_dones, zero, next_values)
        next_advantage = jnp.where(cur_dones, zero, next_advantage)

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        td_err = cur_rewards + cfg.gamma * next_values - cur_values

        # A_t = sum (gamma * lambda)^(l - 1) * delta_l (EQ 16 GAE)
        #     = delta_t + gamma * lambda * A_t+1
        cur_advantage = td_err + cfg.gamma * cfg.gae_lambda * next_advantage

        advantages = advantages.at[i].set(cur_advantage)

        return cur_advantage, cur_values, advantages

    next_advantage, next_values, advantages = lax.fori_loop(
        0, T, advantage_step,
        (jnp.zeros_like(bootstrap_values), bootstrap_values, advantages))

    advantages = advantages.reshape(num_chunks, steps_per_chunk, P, B, 1)

    return rollouts.copy({
        'advantages': advantages
    })

def normalize_advantages(cfg, advantages):
    if not cfg.normalize_advantages:
        return advantages
    else:
       mean = jnp.mean(advantages, dtype=jnp.float32)
       var = jnp.var(advantages, dtype=jnp.float32)

       mean = jnp.asarray(mean, dtype=advantages.dtype)
       var = jnp.asarray(var, dtype=advantages.dtype)

       return (advantages - mean) * jnp.rsqrt(jnp.clip(var, min=1e-5))


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

import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from typing import List, Dict

from .cfg import TrainConfig
from .metrics import Metric
from .moving_avg import EMANormalizer
from .utils import DataclassProtocol

class HyperParams(flax.struct.PyTreeNode):
    lr: float
    gamma: float
    gae_lambda: float
    normalize_values: bool
    value_normalizer_decay: float


class AlgoBase:
    def init_hyperparams(self, cfg: TrainConfig):
        raise NotImplementedError

    def make_optimizer(
        self,
        hyper_params: HyperParams,
    ):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def add_metrics(
        self,
        cfg: TrainConfig, 
        metrics: FrozenDict[str, Metric],
    ):
        raise NotImplementedError


def compute_returns(
    cfg: TrainConfig,
    rewards: jax.Array,
    dones: jax.Array,
    bootstrap_values: jax.Array,
):
    num_chunks, steps_per_chunk, P, B = dones.shape[0:4]

    T = num_chunks * steps_per_chunk
    N = P * B

    seq_dones, seq_rewards = jax.tree_map(
        lambda x: x.reshape(T, N, 1), (dones, rewards))

    bootstrap_values = bootstrap_values.reshape(-1, 1)

    returns = jnp.empty_like(seq_rewards)

    def return_step(i_fwd, inputs):
        i = T - 1 - i_fwd
        next_return, returns = inputs

        cur_dones = seq_dones[i]
        cur_rewards = seq_rewards[i]

        next_return = jnp.where(cur_dones, 0, next_return)

        cur_return = cur_rewards + cfg.gamma * next_return

        returns = returns.at[i].set(cur_return)

        return cur_return, returns

    next_return, returns = lax.fori_loop(
        0, T, return_step, (bootstrap_values, returns))

    return returns.reshape(num_chunks, steps_per_chunk, P, B, 1)


def compute_advantages(
    cfg: TrainConfig,
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    bootstrap_values: jax.Array,
):
    num_chunks, steps_per_chunk, P, B = dones.shape[0:4]

    T = num_chunks * steps_per_chunk
    N = P * B

    seq_dones, seq_rewards, seq_values = jax.tree_map(
        lambda x: x.reshape(T, N, 1),
        (dones, rewards, values))

    bootstrap_values = bootstrap_values.reshape(-1, 1)

    advantages = jnp.empty_like(seq_rewards)

    def advantage_step(i_fwd, inputs):
        i = T - 1 - i_fwd
        next_advantage, next_values, advantages = inputs

        cur_dones = seq_dones[i]
        cur_rewards = seq_rewards[i]
        cur_values = seq_values[i]

        next_values = jnp.where(cur_dones, 0, next_values)
        next_advantage = jnp.where(cur_dones, 0, next_advantage)

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

    return advantages.reshape(num_chunks, steps_per_chunk, P, B, 1)


def zscore_data(data):
    mean = jnp.mean(data, dtype=jnp.float32)
    var = jnp.var(data, dtype=jnp.float32)

    mean = jnp.asarray(mean, dtype=data.dtype)
    var = jnp.asarray(var, dtype=data.dtype)

    return (data - mean) * lax.rsqrt(jnp.clip(var, a_min=1e-5))

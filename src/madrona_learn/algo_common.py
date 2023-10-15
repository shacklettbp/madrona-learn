import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

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
    num_train_seqs_per_policy: int
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
        self.num_train_seqs_per_policy = \
            self.train_agents_per_policy * cfg.num_bptt_chunks

        self.num_bptt_steps = cfg.steps_per_update // cfg.num_bptt_chunks

        if cfg.mixed_precision:
            if dev.platform == 'gpu':
                self.float_storage_type = jnp.float16
            else:
                self.float_storage_type = jnp.bfloat16
        else:
            self.float_storage_type = jnp.float32

class Metric(flax.struct.PyTreeNode):
    mean: jnp.float32
    stddev: jnp.float32
    min: jnp.float32
    max: jnp.float32


class TrainingMetrics(flax.struct.PyTreeNode):
    metrics: FrozenDict[str, Metric]
    count: jnp.int32

    @staticmethod
    def create(metric_names):
        init_metrics = {}
        for name in metric_names:
            init_metrics[name] = Metric(
                mean = jnp.float32(0),
                stddev = jnp.float32(0),
                min = jnp.float32(jnp.finfo(jnp.float32).max),
                max = jnp.float32(jnp.finfo(jnp.float32).min),
            )

        return TrainingMetrics(
            metrics = frozen_dict.freeze(init_metrics),
            count = 0,
        )

    def record(self, data):
        def compute_metric(x):
            mean = jnp.mean(x, dtype=jnp.float32)
            stddev = jnp.std(x, dtype=jnp.float32)
            min = jnp.asarray(jnp.min(x), dtype=jnp.float32)
            max = jnp.asarray(jnp.max(x), dtype=jnp.float32)

            return Metric(mean, stddev, min, max)

        merged_metrics = {}
        for k in data.keys():
            old_metric = self.metrics[k]
            new_metric = compute_metric(data[k])

            merged_metrics[k] = Metric(
                mean = (old_metric.mean +
                    (new_metric.mean - old_metric.mean) / self.count),
                stddev = (old_metric.stddev +
                    (new_metric.stddev - old_metric.stddev) / self.count),
                min = jnp.minimum(old_metric.min, new_metric.min),
                max = jnp.maximum(old_metric.max, new_metric.max),
            )

        return TrainingMetrics(
            metrics = self.metrics.copy(merged_metrics),
            count = self.count,
        )

    def increment_count(self):
        return TrainingMetrics(
            metrics = self.metrics,
            count = self.count + 1,
        )

    def __repr__(self):
        rep = "TrainingMetrics:\n"

        def comma_separate(v):
            r = []

            for i in range(v.shape[0]):
                r.append(f"{float(v[i]): .3e}")

            return ", ".join(r)

        for k, v in self.metrics.items():
            rep += f"    {k}:\n"
            rep += f"        Mean: "
            rep += comma_separate(v.mean) + "\n"
            rep += f"        Std:  "
            rep += comma_separate(v.stddev) + "\n"
            rep += f"        Min:  "
            rep += comma_separate(v.min) + "\n"
            rep += f"        Max:  "
            rep += comma_separate(v.max) + "\n"

        return rep

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

       return (advantages - mean) * lax.rsqrt(jnp.clip(var, a_min=1e-5))

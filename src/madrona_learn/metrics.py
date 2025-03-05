import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
import numpy as np
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from functools import partial
from typing import Callable, List

class Metric(flax.struct.PyTreeNode):
    per_policy: bool = flax.struct.field(pytree_node=False)
    mean: jnp.float32
    m2: jnp.float32
    min: jnp.float32
    max: jnp.float32
    count: jnp.int32

    @staticmethod
    def init(per_policy):
        return Metric(
            per_policy = per_policy,
            mean = jnp.float32(0),
            m2 = jnp.float32(0),
            min = jnp.float32(jnp.finfo(jnp.float32).max),
            max = jnp.float32(jnp.finfo(jnp.float32).min),
            count = jnp.int32(0),
        )

    @staticmethod
    def init_from_data(per_policy, data):
        num_new_elems = jnp.int32(data.size)
        mean = jnp.mean(data, dtype=jnp.float32)
        min = jnp.min(data).astype(jnp.float32)
        max = jnp.max(data).astype(jnp.float32)

        deltas = data.astype(jnp.float32) - mean
        m2 = jnp.sum(deltas * deltas, dtype=jnp.float32)
        
        return Metric(
            per_policy = per_policy,
            mean = mean,
            m2 = m2,
            min = min,
            max = max,
            count = num_new_elems,
        )

    @staticmethod
    def init_from_data_masked(per_policy, data, mask):
        num_new_elems = jnp.int32(data.size)
        mean = jnp.mean(data, dtype=jnp.float32)
        min = jnp.min(data).astype(jnp.float32)
        max = jnp.max(data).astype(jnp.float32)

        deltas = data.astype(jnp.float32) - mean
        m2 = jnp.sum(deltas * deltas, dtype=jnp.float32)
        
        return Metric(
            per_policy = per_policy,
            mean = mean,
            m2 = m2,
            min = min,
            max = max,
            count = num_new_elems,
        )

    def reset(self):
        return Metric(
            per_policy = self.per_policy,
            mean = jnp.zeros_like(self.mean),
            m2 = jnp.zeros_like(self.m2),
            min = jnp.full_like(self.min, jnp.finfo(jnp.float32).max),
            max = jnp.full_like(self.max, jnp.finfo(jnp.float32).min),
            count = jnp.zeros_like(self.count),
        )

    def merge(self, new_metric):
        new_count = self.count + new_metric.count 

        delta = new_metric.mean - self.mean

        safe_denom = 1 / jnp.maximum(new_count.astype(jnp.float32), 1)

        mean = (self.mean + delta * 
                new_metric.count.astype(jnp.float32) * safe_denom)
        m2 = (self.m2 + new_metric.m2 + delta * delta *
              self.count.astype(jnp.float32) *
              new_metric.count.astype(jnp.float32) * safe_denom)

        return self.replace(
            mean = mean,
            m2 = m2,
            min = jnp.minimum(self.min, new_metric.min),
            max = jnp.maximum(self.max, new_metric.max),
            count = new_count,
        )


class TrainingMetrics(flax.struct.PyTreeNode):
    metrics: FrozenDict[str, Metric]
    update_idx: jax.Array
    cur_buffer_offset: jax.Array
    update_buffer_size: jax.Array
    print_names: FrozenDict[str, str] = flax.struct.field(pytree_node=False)

    @staticmethod
    def create(cfg, metrics: FrozenDict[str, Metric], start_update_idx: int,
               num_policies: int):
        max_keylen = 0
        for name in metrics.keys():
            max_keylen = max(max_keylen, len(name))

        print_names = {}
        for name in metrics.keys():
            print_names[name] = name# + ' ' * (max_keylen - len(name))

        num_policies = cfg.pbt.num_train_policies if cfg.pbt else 1

        def expand_metric(x):
            @partial(jax.vmap, in_axes=None, out_axes=0,
                     axis_size=num_policies)
            def expand_policy_dim(x):
                return x

            @partial(jax.vmap, in_axes=None, out_axes=0,
                     axis_size=cfg.metrics_buffer_size)
            def expand_time_dim(x):
                return x

            x = expand_time_dim(x)

            if x.per_policy:
                x = expand_policy_dim(x)

            return x

        metrics = FrozenDict({k: expand_metric(v) for k,v in metrics.items()})

        return TrainingMetrics(
            metrics = metrics,
            update_idx = jnp.full(
                (num_policies,), start_update_idx, dtype=jnp.int32),
            cur_buffer_offset = jnp.full(
                (num_policies,), 0, dtype=jnp.int32),
            update_buffer_size = jnp.full(
                (num_policies,), cfg.metrics_buffer_size, dtype=jnp.int32),
            print_names = print_names,
        )

    def update_metrics(self, metrics):
        updated_metrics = {}
        for k in metrics.keys():
            updated_metrics[k] = jax.tree.map(
                lambda x, y: x.at[:, self.cur_buffer_offset].set(y),
                self.metrics[k], metrics[k])

        return self.replace(metrics = self.metrics.copy(updated_metrics))

    def record(self, data):
        updated_metrics = {}
        for k in data.keys():
            per_policy = self.metrics[k].per_policy

            def init_metric(data):
                return Metric.init_from_data(per_policy, data)

            # If this is a per-policy metric and record isn't being
            # called in a vmap'd region, update_metric needs to be vmapped
            if per_policy and self.metrics[k].mean.ndim > 1:
                init_metric = jax.vmap(init_metric)

                update_metric = lambda x, y: x.at[:, self.cur_buffer_offset].set(y)
            else:
                update_metric = lambda x, y: x.at[self.cur_buffer_offset].set(y)

            updated_metrics[k] = jax.tree.map(
                update_metric, self.metrics[k], init_metric(data[k]))

        return self.replace(metrics = self.metrics.copy(updated_metrics))
    
    def advance(self):
        return self.replace(
            update_idx = self.update_idx + 1,
            cur_buffer_offset =
                (self.cur_buffer_offset + 1) % self.update_buffer_size,
        )

    def pretty_print(self, tab=2):
        tab = ' ' * tab

        formatted = [tab + "TrainingMetrics"]
        for k, name in self.print_names.items():
            m = self.metrics[k]

            def fmt(x):
                if not m.per_policy:
                    return f"{float(x): .3e}"

                r = []

                for i in range(x.shape[0]):
                    r.append(f"{float(x[i]): .3e}")

                return ", ".join(r)

            stddev = np.sqrt(m.m2 / m.count)

            formatted.append(tab * 2 + f"{name}:")
            formatted.append(tab * 3 + f"Avg: {fmt(m.mean)}")
            formatted.append(tab * 3 + f"Min: {fmt(m.min)}")
            formatted.append(tab * 3 + f"Max: {fmt(m.max)}")
            formatted.append(tab * 3 + f"σ:   {fmt(stddev)}")
        
        print("\n".join(formatted))

    def tensorboard_log(self, base_update_idx, writer):
        for buf_idx in range(self.update_buffer_size[0]):
            out_idx = base_update_idx + buf_idx

            for name, metric in self.metrics.items():
                if not metric.per_policy:
                    stddev = np.sqrt(
                        metric.m2[buf_idx] / metric.count[buf_idx])

                    writer.scalar(f"{name} Mean", metric.mean[buf_idx], out_idx)
                    writer.scalar(f"{name} σ", stddev, out_idx)
                    writer.scalar(f"{name} Min", metric.min[buf_idx], out_idx)
                    writer.scalar(f"{name} Max", metric.max[buf_idx], out_idx)
                else:
                    num_policies = metric.mean.shape[0]

                    for i in range(num_policies):
                        stddev = np.sqrt(
                            metric.m2[i, buf_idx] / metric.count[i, buf_idx])

                        writer.scalar(f"p{i}/{name} Mean",
                                      metric.mean[i, buf_idx], out_idx)
                        writer.scalar(f"p{i}/{name} σ", stddev, out_idx)
                        writer.scalar(f"p{i}/{name} Min",
                                      metric.min[i, buf_idx], out_idx)
                        writer.scalar(f"p{i}/{name} Max",
                                      metric.max[i, buf_idx], out_idx)

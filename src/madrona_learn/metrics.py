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

    def reset(self):
        return Metric(
            per_policy = self.per_policy,
            mean = jnp.zeros_like(self.mean),
            m2 = jnp.zeros_like(self.m2),
            min = jnp.full_like(self.min, jnp.finfo(jnp.float32).max),
            max = jnp.full_like(self.max, jnp.finfo(jnp.float32).min),
            count = jnp.zeros_like(self.count),
        )


class TrainingMetrics(flax.struct.PyTreeNode):
    metrics: FrozenDict[str, Metric]
    print_names: FrozenDict[str, str] = flax.struct.field(pytree_node=False)

    @staticmethod
    def create(cfg, metrics: FrozenDict[str, Metric]):
        max_keylen = 0
        for name in metrics.keys():
            max_keylen = max(max_keylen, len(name))

        print_names = {}
        for name in metrics.keys():
            print_names[name] = name# + ' ' * (max_keylen - len(name))

        num_policies = cfg.pbt.num_train_policies if cfg.pbt else 1

        @partial(jax.vmap, in_axes=None, out_axes=0,
                 axis_size=num_policies)
        def expand_policy_dim(x):
            return x

        def expand_metric(x):
            if x.per_policy:
                return expand_policy_dim(x)
            else:
                return x

        metrics = FrozenDict({k: expand_metric(v) for k,v in metrics.items()})

        return TrainingMetrics(
            metrics = metrics,
            print_names = print_names,
        )

    def _update_metric(self, cur_metric, new_data):
        num_new_elems = new_data.size
        new_data_mean = jnp.mean(new_data, dtype=jnp.float32)
        new_data_min = jnp.asarray(jnp.min(new_data), dtype=jnp.float32)
        new_data_max = jnp.asarray(jnp.max(new_data), dtype=jnp.float32)

        new_data_deltas = new_data - jnp.asarray(
            new_data_mean, dtype=new_data.dtype)
        new_data_m2 = jnp.sum(new_data_deltas * new_data_deltas,
                              dtype=jnp.float32)

        new_count = cur_metric.count + num_new_elems

        delta = new_data_mean - cur_metric.mean

        mean = (cur_metric.mean + delta * 
            jnp.asarray(num_new_elems, dtype=jnp.float32) /
            jnp.asarray(new_count, dtype=jnp.float32))
        m2 = (cur_metric.m2 + new_data_m2  + delta * delta *
              jnp.asarray(cur_metric.count, dtype=jnp.float32) *
              jnp.asarray(num_new_elems, dtype=jnp.float32) /
              jnp.asarray(new_count, dtype=jnp.float32))

        return cur_metric.replace(
            mean = mean,
            m2 = m2,
            min = jnp.minimum(cur_metric.min, new_data_min),
            max = jnp.maximum(cur_metric.max, new_data_max),
            count = new_count,
        )

    def record(self, data):
        merged_metrics = {}
        for k in data.keys():
            old_metric = self.metrics[k]

            # If this is a per-policy metric and record isn't being
            # called in a vmap'd region, _update_metric needs to be vmapped
            if old_metric.per_policy and old_metric.mean.ndim > 0:
                update_metric = jax.vmap(self._update_metric)
            else:
                update_metric = self._update_metric

            merged_metrics[k] = update_metric(old_metric, data[k])

        return self.replace(metrics = self.metrics.copy(merged_metrics))

    def reset(self):
        reset_metrics = FrozenDict({
            k: m.reset() for k, m in self.metrics.items()
        })

        return self.replace(metrics = reset_metrics)

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

    def tensorboard_log(self, writer, update):
        for name, metric in self.metrics.items():
            if not metric.per_policy:
                stddev = np.sqrt(metric.m2 / metric.count)

                writer.scalar(f"{name} Mean", metric.mean, update)
                writer.scalar(f"{name} σ", stddev, update)
                writer.scalar(f"{name} Min", metric.min, update)
                writer.scalar(f"{name} Max", metric.max, update)
            else:
                num_policies = metric.mean.shape[0]

                for i in range(num_policies):
                    stddev = np.sqrt(metric.m2[i] / metric.count[i])

                    writer.scalar(f"p{i}/{name} Mean", metric.mean[i], update)
                    writer.scalar(f"p{i}/{name} σ", stddev, update)
                    writer.scalar(f"p{i}/{name} Min", metric.min[i], update)
                    writer.scalar(f"p{i}/{name} Max", metric.max[i], update)

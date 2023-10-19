import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from typing import Callable, List

@dataclass(frozen = True)
class CustomMetricConfig:
    cb: Callable
    custom_metrics: List[str]


class Metric(flax.struct.PyTreeNode):
    mean: jnp.float32
    stddev: jnp.float32
    min: jnp.float32
    max: jnp.float32


class TrainingMetrics(flax.struct.PyTreeNode):
    metrics: FrozenDict[str, Metric]
    count: jnp.int32
    print_names: FrozenDict[str, str] = flax.struct.field(pytree_node=False)

    @staticmethod
    def create(metric_names):
        init_metrics = {}
        max_keylen = 0
        for name in metric_names:
            init_metrics[name] = Metric(
                mean = jnp.float32(0),
                stddev = jnp.float32(0),
                min = jnp.float32(jnp.finfo(jnp.float32).max),
                max = jnp.float32(jnp.finfo(jnp.float32).min),
            )

            max_keylen = max(max_keylen, len(name))

        print_names = {}
        for name in metric_names:
            print_names[name] = name + ' ' * (max_keylen - len(name))

        return TrainingMetrics(
            metrics = frozen_dict.freeze(init_metrics),
            count = 0,
            print_names = print_names,
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
            print_names = self.print_names,
        )

    def increment_count(self):
        return TrainingMetrics(
            metrics = self.metrics,
            count = self.count + 1,
            print_names = self.print_names,
        )

    def __repr__(self):
        rep = "TrainingMetrics:\n"

        def comma_separate(v):
            r = []

            for i in range(v.shape[0]):
                r.append(f"{float(v[i]): .3e}")

            return ", ".join(r)

        for k, name in self.print_names.items():
            v = self.metrics[k]
            rep += f"    {name} => Avg: {comma_separate(v.mean)}, Min: {comma_separate(v.min)}, Max: {comma_separate(v.max)}, σ: {comma_separate(v.stddev)}\n"

        return rep

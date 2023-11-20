import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict

from dataclasses import dataclass
from typing import List, Callable, Any

from .moving_avg import EMANormalizer

@dataclass(frozen=True)
class ObservationsPreprocess:
    preprocessors: FrozenDict[str, Any]
    init_state_fn: Callable
    update_state_fn: Callable
    init_obs_stats_fn: Callable
    update_obs_stats_fn: Callable
    preprocess_fn: Callable

    def init_state(self, obs):
        return jax.tree_map(self.init_state_fn, self.preprocessors, obs)

    def update_state(self, states, o_stats):
        return jax.tree_map(
            self.update_state_fn, self.preprocessors, states, o_stats)

    def init_obs_stats(self, states):
        return jax.tree_map(self.init_obs_stats_fn, self.preprocessors, states)

    def update_obs_stats(self, state, cur_obs_stats, num_prev_updates, obs):
        return jax.tree_map(
            lambda preproc, state, o_stats, o: self.update_obs_stats_fn(
                preproc, state, o_stats, num_prev_updates, o),
            self.preprocessors, states, cur_obs_stats, obs)

    def preprocess(self, states, obs):
        return jax.tree_map(
            self.preprocess_fn, self.preprocessors, states, obs)


@dataclass(frozen=True)
class ObservationsNormalizer:
    decay: float
    dtype: jnp.dtype
    eps: float = 1e-5
    normalizer_cls: type = EMANormalizer

    def init(self, obs):
        preprocessors = jax.tree_map(lambda _: self.normalizer_cls(
                decay = self.decay,
                out_dtype = self.dtype,
                eps = self.eps,
            ), obs)

        return ObservationsPreprocess(
            preprocessors = preprocessors,
            init_state_fn = lambda norm, o: norm.init_estimates(o),
            update_state_fn = lambda norm, est, o_stats: (
                norm.update_estimates(est, o_stats)),
            init_obs_stats_fn = lambda norm, est: (
                norm.init_input_stats(est)),
            update_obs_stats_fn = lambda norm, _, o_stats, prev_updates, o: (
                norm.update_init_stats(o_states, prev_updates, o)),
            preprocess_fn = lambda norm, est, o: norm.normalize(est, o),
        )

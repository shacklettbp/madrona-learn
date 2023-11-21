import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict

from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Callable, Any

from .moving_avg import EMANormalizer

@dataclass(frozen=True)
class ObservationsPreprocess:
    preprocessors: FrozenDict[str, Any]
    preprocess_fn: Callable
    is_stateful: bool 
    init_state_fn: Optional[Callable] = None
    update_state_fn: Optional[Callable] = None
    init_obs_stats_fn: Optional[Callable] = None
    update_obs_stats_fn: Optional[Callable] = None

    def preprocess(self, states, obs, vmap):
        if not self.is_stateful:
            if vmap:
                preprocess = jax.vmap(self.preprocess_fn, in_axes=(None, 0))
            else:
                preprocess = self.preprocess_fn
            return jax.tree_map(preprocess, self.preprocessors, obs)
        else:
            if vmap:
                preprocess = jax.vmap(self.preprocess_fn, in_axes=(None, 0, 0))
            else:
                preprocess = self.preprocess_fn
            return jax.tree_map(preprocess, self.preprocessors, states, obs)

    def init_state(self, obs, vmap):
        if not self.is_stateful:
            return None

        if vmap:
            init_state = jax.vmap(self.init_state_fn, in_axes=(None, 0))
        else:
            init_state = self.init_state_fn

        return jax.tree_map(init_state, self.preprocessors, obs)

    def update_state(self, states, o_stats, vmap):
        if not self.is_stateful:
            return None

        if vmap:
            update_state = jax.vmap(self.update_state_fn, in_axes=(None, 0, 0))
        else:
            update_state = self.update_state_fn

        return jax.tree_map(update_state, self.preprocessors, states, o_stats)

    def init_obs_stats(self, states, vmap):
        if not self.is_stateful:
            return None

        if vmap:
            init_stats = jax.vmap(
                self.init_obs_stats_fn, in_axes=(None, 0))
        else:
            init_stats = self.init_obs_stats_fn

        return jax.tree_map(init_stats, self.preprocessors, states)

    def update_obs_stats(self, states, cur_obs_stats,
                         num_prev_updates, obs, vmap):
        if not self.is_stateful:
            return None
        
        if vmap:
            def update_stats(preproc, state, stats, obs):
                return self.update_obs_stats_fn(
                    preproc, state, stats, num_prev_updates, obs)

            update_stats = jax.vmap(update_stats, in_axes=(None, 0, 0, 0))
        else:
            update_stats = self.update_obs_stats_fn

        return jax.tree_map(update_stats,
            self.preprocessors, states, cur_obs_stats, obs)


@dataclass(frozen=True)
class ObservationsNormalizer:
    decay: float
    dtype: jnp.dtype
    eps: float = 1e-5
    normalizer_cls: type = EMANormalizer

    def init(self, obs):
        normalizers = jax.tree_map(lambda _: self.normalizer_cls(
                decay = self.decay,
                out_dtype = self.dtype,
                eps = self.eps,
            ), obs)

        return ObservationsPreprocess(
            preprocessors = normalizers,
            preprocess_fn = lambda norm, est, o: norm.normalize(est, o),
            is_stateful = True,
            init_state_fn = lambda norm, o: norm.init_estimates(o),
            update_state_fn = lambda norm, est, o_stats: (
                norm.update_estimates(est, o_stats)),
            init_obs_stats_fn = lambda norm, est: (
                norm.init_input_stats(est)),
            update_obs_stats_fn = lambda norm, _, o_stats, prev_updates, o: (
                norm.update_init_stats(o_states, prev_updates, o)),
        )

@dataclass(frozen=True)
class ObservationsCaster:
    dtype: jnp.dtype

    def init(self, obs):
        return ObservationsPreprocess(
            preprocessors = jax.tree_map(lambda o: self, obs),
            preprocess_fn = lambda caster, ob: ob.astype(caster.dtype),
            is_stateful = False,
        )

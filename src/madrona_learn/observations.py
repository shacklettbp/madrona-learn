import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict

from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Callable, Any

from .moving_avg import EMANormalizer

@dataclass(frozen=True)
class ObservationsPreprocess:
    is_stateful: bool 
    overrides: Dict[str, Callable]

    def preprocess(self, states, obs, vmap):
        if vmap:
            preprocess = jax.vmap(self._preprocess)
        else:
            preprocess = self._preprocess

        args = [states, obs]

        if not self.is_stateful:
            args = [obs]
        else:
            args = [states, obs]

        r = {}
        for k, ob in obs.items():
            if k in self.overrides:
                r[k] = self.overrides[k](ob)
            elif not self.is_stateful:
                r[k] = preprocess(ob)
            else:
                r[k] = preprocess(states[k], ob)

        return FrozenDict(r)

    def init_state(self, obs, vmap):
        if not self.is_stateful:
            return None

        if vmap:
            init_state = jax.vmap(self._init_state)
        else:
            init_state = self._init_state

        return self._iter_non_overrides(init_state, obs)

    def update_state(self, states, o_stats, vmap):
        if not self.is_stateful:
            return None

        if vmap:
            update_state = jax.vmap(self._update_state)
        else:
            update_state = self._update_state

        return self._iter_non_overrides(update_state, states, o_stats)

    def init_obs_stats(self, states, vmap):
        if not self.is_stateful:
            return None

        if vmap:
            init_stats = jax.vmap(self._init_obs_stats)
        else:
            init_stats = self._init_obs_stats

        return self._iter_non_overrides(init_stats, states)

    def update_obs_stats(self, states, cur_obs_stats,
                         num_prev_updates, obs, vmap):
        if not self.is_stateful:
            return None

        def update_stats(state, stats, obs):
            return self._update_obs_stats(
                state, stats, num_prev_updates, obs)
        
        if vmap:
            update_stats = jax.vmap(update_stats)

        return self._iter_non_overrides(
            update_stats, states, cur_obs_stats, obs)

    def _iter_non_overrides(self, f, *args):
        keys = args[0].keys()

        r = {}

        for k in keys:
            if k in self.overrides:
                continue
            r[k] = f(*[a[k] for a in args])

        return FrozenDict(r)

@dataclass(frozen=True)
class ObservationsEMANormalizer(ObservationsPreprocess):
    normalizer: EMANormalizer

    @staticmethod
    def create(
        decay: float,
        dtype: jnp.dtype,
        eps: float = 1e-5,
        overrides: Dict[str, Callable] = {},
    ):
        return ObservationsEMANormalizer(
            is_stateful = True,
            overrides = overrides,
            normalizer = EMANormalizer(
                decay = decay,
                out_dtype = dtype,
                eps = eps,
            ),
        )

    def _preprocess(self, est, ob):
        return self.normalizer.normalize(est, ob)

    def _init_state(self, ob):
        return self.normalizer.init_estimates(ob)

    def _update_state(self, est, ob_stats):
        return self.normalizer.update_estimates(est, ob_stats)

    def _init_obs_stats(self, est):
        return self.normalizer.init_input_stats(est)

    def _update_obs_stats(self, est, ob_stats, num_prev_updates, ob):
        return self.normalizer.update_input_stats(
            ob_stats, num_prev_updates, ob)


@dataclass(frozen=True)
class ObservationsCaster(ObservationsPreprocess):
    dtype: jnp.dtype

    @staticmethod
    def create(
        dtype: jnp.dtype,
    ):
        return ObservationsCaster(
            is_stateful = False,
            overrides = {},
            dtype = dtype,
        )

    def _preprocess(self, ob):
        return ob.astype(self.dtype)


@dataclass(frozen=True)
class ObservationsPreprocessNoop(ObservationsPreprocess):
    @staticmethod
    def create():
        return ObservationsPreprocessNoop(
            is_stateful = False,
            overrides = {},
        )

    def _preprocess(self, ob):
        return ob



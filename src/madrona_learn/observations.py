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
        args = [states, obs]

        if not self.is_stateful:
            args = [obs]
        else:
            args = [states, obs]

        if vmap:
            preprocess = jax.vmap(self._preprocess,
                                  in_axes=(None, *((0,) * len(args))))
        else:
            preprocess = self._preprocess

        r = {}
        for k, ob in obs.items():
            if k in self.overrides:
                r[k] = self.overrides[k](ob)
            elif not self.is_stateful:
                r[k] = preprocess(k, ob)
            else:
                r[k] = preprocess(k, states[k], ob)

        return FrozenDict(r)

    def init_state(self, obs, vmap):
        if not self.is_stateful:
            return None

        if vmap:
            init_state = jax.vmap(self._init_state, in_axes=(None, 0))
        else:
            init_state = self._init_state

        return self._iter_non_overrides(init_state, obs)

    def update_state(self, states, o_stats, vmap):
        if not self.is_stateful:
            return None

        if vmap:
            update_state = jax.vmap(self._update_state, in_axes=(None, 0, 0))
        else:
            update_state = self._update_state

        return self._iter_non_overrides(update_state, states, o_stats)

    def init_obs_stats(self, states, vmap):
        if not self.is_stateful:
            return None

        if vmap:
            init_stats = jax.vmap(self._init_obs_stats, in_axes=(None, 0))
        else:
            init_stats = self._init_obs_stats

        return self._iter_non_overrides(init_stats, states)

    def update_obs_stats(self, states, cur_obs_stats,
                         num_prev_updates, obs, vmap):
        if not self.is_stateful:
            return None

        def update_stats(ob_name, state, stats, obs):
            return self._update_obs_stats(
                ob_name, state, stats, num_prev_updates, obs)
        
        if vmap:
            update_stats = jax.vmap(update_stats, in_axes=(None, 0, 0, 0))

        return self._iter_non_overrides(
            update_stats, states, cur_obs_stats, obs)

    def _iter_non_overrides(self, f, *args):
        keys = args[0].keys()

        r = {}

        for k in keys:
            if k in self.overrides:
                continue
            r[k] = f(k, *[a[k] for a in args])

        return FrozenDict(r)

@dataclass(frozen=True)
class ObservationsEMANormalizer(ObservationsPreprocess):
    normalizer: EMANormalizer
    prep_fns: Dict[str, Callable]

    @staticmethod
    def create(
        decay: float,
        dtype: jnp.dtype,
        eps: float = 1e-5,
        prep_fns: Dict[str, Callable] = {},
        overrides: Dict[str, Callable] = {},
    ):
        return ObservationsEMANormalizer(
            is_stateful = True,
            overrides = overrides,
            normalizer = EMANormalizer(
                decay = decay,
                norm_dtype = dtype,
                inv_dtype = dtype,
                eps = eps,
            ),
            prep_fns = prep_fns,
        )

    def _prep_ob(self, ob_name, ob):
        prep_fn = self.prep_fns.get(ob_name, lambda x: x)
        return prep_fn(ob)

    def _preprocess(self, ob_name, est, ob):
        ob = self._prep_ob(ob_name, ob)
        return self.normalizer.normalize(est, ob)

    def _init_state(self, ob_name, ob):
        ob = self._prep_ob(ob_name, ob)
        return self.normalizer.init_estimates(ob)

    def _update_state(self, ob_name, est, ob_stats):
        return self.normalizer.update_estimates(est, ob_stats)

    def _init_obs_stats(self, ob_name, est):
        return self.normalizer.init_input_stats(est)

    def _update_obs_stats(self, ob_name, est, ob_stats, num_prev_updates, ob):
        ob = self._prep_ob(ob_name, ob)
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



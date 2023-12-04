import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict

from dataclasses import dataclass
from functools import partial
from typing import Dict, Set, Optional, Callable, Any

from .moving_avg import EMANormalizer

@dataclass(frozen=True)
class ObservationsPreprocess:
    def preprocess(self, states, obs, vmap):
        return self._map_obs(self._preprocess, vmap, states, obs)

    def init_state(self, obs, vmap):
        return self._map_obs(self._init_state, vmap, obs)

    def update_state(self, states, o_stats, vmap):
        return self._map_obs(self._update_state, vmap, states, o_stats)

    def init_obs_stats(self, states, vmap):
        return self._map_obs(self._init_obs_stats, vmap, states)

    def update_obs_stats(self, states, cur_obs_stats,
                         num_prev_updates, obs, vmap):
        def update_stats(ob_name, state, stats, obs):
            return self._update_obs_stats(
                ob_name, state, stats, num_prev_updates, obs)

        return self._map_obs(
            update_stats, vmap, states, cur_obs_stats, obs)

    def _map_obs(self, cb, vmap, *args):
        keys = args[0].keys()

        r = {}
        for ob_name in keys:
            ob_args = [a[ob_name] for a in args]

            if vmap:
                num_args = len(args)
                vmap_axes = [None] + [0 if a != None else None for a in ob_args]

                if all(axis is None for axis in vmap_axes):
                    f = cb
                else:
                    f = jax.vmap(cb, in_axes=vmap_axes)
            else:
                f = cb

            r[ob_name] = f(ob_name, *ob_args)

        return FrozenDict(r)

    def _init_state(self, ob_name, ob):
        return None

    def _update_state(self, ob_name, est, ob_stats):
        return None

    def _init_obs_stats(self, ob_name, est):
        return None

    def _update_obs_stats(self, ob_name, est, ob_stats, num_prev_updates, ob):
        return None

@dataclass(frozen=True)
class ObservationsEMANormalizer(ObservationsPreprocess):
    normalizer: EMANormalizer
    prep_fns: Dict[str, Callable]
    skip_normalization: Set[str]

    @staticmethod
    def create(
        decay: float,
        dtype: jnp.dtype,
        eps: float = 1e-5,
        prep_fns: Dict[str, Callable] = {},
        skip_normalization: Set[str] = set(),
    ):
        return ObservationsEMANormalizer(
            normalizer = EMANormalizer(
                decay = decay,
                norm_dtype = dtype,
                inv_dtype = dtype,
                eps = eps,
            ),
            prep_fns = prep_fns,
            skip_normalization = skip_normalization,
        )

    def _prep_ob(self, ob_name, ob):
        prep_fn = self.prep_fns.get(ob_name, lambda x: x)
        return prep_fn(ob)

    def _preprocess(self, ob_name, est, ob):
        ob = self._prep_ob(ob_name, ob)

        if ob_name in self.skip_normalization:
            return ob
        else:
            return self.normalizer.normalize(est, ob)

    def _init_state(self, ob_name, ob):
        if ob_name in self.skip_normalization:
            return None

        ob = self._prep_ob(ob_name, ob)
        return self.normalizer.init_estimates(ob)

    def _update_state(self, ob_name, est, ob_stats):
        if ob_name in self.skip_normalization:
            return None

        return self.normalizer.update_estimates(est, ob_stats)

    def _init_obs_stats(self, ob_name, est):
        if ob_name in self.skip_normalization:
            return None

        return self.normalizer.init_input_stats(est)

    def _update_obs_stats(self, ob_name, est, ob_stats, num_prev_updates, ob):
        if ob_name in self.skip_normalization:
            return None

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
            dtype = dtype,
        )

    def _preprocess(self, ob_name, _, ob):
        return ob.astype(self.dtype)


@dataclass(frozen=True)
class ObservationsPreprocessNoop(ObservationsPreprocess):
    @staticmethod
    def create():
        return ObservationsPreprocessNoop()

    def _preprocess(self, ob_name, _, ob):
        return ob



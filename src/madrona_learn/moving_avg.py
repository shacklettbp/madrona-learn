import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import FrozenDict
from dataclasses import dataclass

# Exponential Moving Average mean and variance estimator for
# values and observations
@dataclass(frozen=True)
class EMANormalizer:
    decay: float
    out_dtype: jnp.dtype
    eps: float = 1e-5
    disable: bool = False

    def init_estimates(self, x):
        if self.disable:
            return {}

        dim = x.shape[-1]

        # Parameter estimates. Initialized to mu 0, sigma 1 as a
        # noop. Note that these values do not bias the overall estimate,
        # they will be immediately overwritten by values derived from
        # mu_biased and sigma_sq_biased when update_stats is called.

        return FrozenDict(
            mu = jnp.zeros((dim,), jnp.float32),
            inv_sigma = jnp.ones((dim,), jnp.float32),
            sigma = jnp.ones((dim,), jnp.float32),
            # Intermediate values used to compute the moving average
            mu_biased = jnp.zeros((dim,), jnp.float32),
            sigma_sq_biased = jnp.zeros((dim,), jnp.float32),
            N = jnp.zeros((), jnp.int32),
        )

    def normalize(self, est, x):
        x = self._convert_nonfloat(x)
        return ((x - est['mu'].astype(x.dtype)) *
            est['inv_sigma'].astype(x.dtype)).astype(self.out_dtype)

    def invert(self, est, x):
        x = self._convert_nonfloat(x)
        return (x * est['sigma'].astype(x.dtype) +
            est['mu'].astype(x.dtype)).astype(self.out_dtype)

    def init_input_stats(self, est):
        return jnp.zeros_like(est['mu']), jnp.zeros_like(est['mu'])

    def update_input_stats(self, cur_stats, num_prev_updates, x):
        a_mean, a_var = cur_stats

        reduce_axes = tuple(range(len(x.shape) - 1))

        x = self._convert_nonfloat(x)

        b_mean = jnp.mean(x, axis=reduce_axes, dtype=jnp.float32)
        b_var = jnp.mean(jnp.square(x - b_mean),
            axis=reduce_axes, dtype=jnp.float32)

        delta = b_mean - a_mean

        n_a = num_prev_updates
        n_ab = n_a + 1

        b_weight = jnp.reciprocal(jnp.float32(n_ab))
        a_weight = jnp.float32(1) - b_weight

        ab_mean = a_mean + delta * b_weight
        ab_var = (a_weight * a_var + b_weight * b_var +
                  jnp.square(delta) * a_weight * b_weight)

        return ab_mean, ab_var

    def update_estimates(self, est, input_stats):
        # This code is derived from the generalized formulas provided in
        # Numerically Stable Parallel Computation of Co-Variance,
        # Schubert & Gertz, 2018.
        # Basically this paper is a generalization of Chan's algorithm to
        # arbitrary weights on each item, which makes it easy to combine
        # Chan's algorithm + EMA. One non-obvious thing from the paper is
        # that VW_a (sum of squared differences from the mean) can simply be
        # rescaled by (1 - alpha) from the prior iteration because the
        # weight changes in the mean are cancelled out.
        x_mean, x_var = input_stats

        mean_delta = x_mean - est['mu']

        one_minus_alpha = jnp.float32(self.decay)
        alpha = jnp.float32(1) - one_minus_alpha

        new_N = est['N'] + 1

        new_mu_biased = (
            one_minus_alpha * est['mu_biased'] + alpha * x_mean
        )

        new_sigma_sq_biased = (
            one_minus_alpha * est['sigma_sq_biased'] + alpha * x_var +
            (est['N'].astype(jnp.float32) / new_N.astype(jnp.float32)) *
                (one_minus_alpha * alpha) * jnp.square(mean_delta)
        )
        
        bias_correction = -1 / jnp.expm1(
            new_N.astype(jnp.float32) * jnp.log(one_minus_alpha))

        new_mu = new_mu_biased * bias_correction
        new_sigma_sq = new_sigma_sq_biased * bias_correction

        # Write out new unbiased params
        new_inv_sigma = lax.rsqrt(lax.max(new_sigma_sq, jnp.float32(self.eps)))
        new_sigma = jnp.reciprocal(new_inv_sigma)

        return FrozenDict(
            mu = new_mu,
            inv_sigma = new_inv_sigma,
            sigma = new_sigma,
            # Intermediate values used to compute the moving average
            mu_biased = new_mu_biased,
            sigma_sq_biased = new_sigma_sq_biased,
            N = new_N,
        )

    def _convert_nonfloat(self, x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return x
        else:
            return x.astype(jnp.float32)

    # Generalization of chan's algorithm to compute variance of N sets
    #def merge_means_vars(self, inputs_means_vars):
    #    x_means, x_vars = inputs_means_vars

    #    merged_mean = jnp.mean(x_means, axis=0, dtype=jnp.float32)

    #    num_merge = x_means.shape[0]
    #    merged_var = jnp.float32(1) / jnp.float32(num_merge) * jnp.sum(
    #        x_vars + jnp.square(x_means - merged_mean[None, :]),
    #        axis=0, dtype=jnp.float32)

    #    return merged_mean, merged_var


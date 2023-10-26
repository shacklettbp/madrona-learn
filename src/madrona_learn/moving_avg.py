import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

# Exponential Moving Average mean and variance estimator for
# values and observations
class EMANormalizer(nn.Module):
    decay: jnp.float32
    eps: jnp.float32 = 1e-5
    disable: bool = False

    def _update_stats(
        self,
        x,
        mu,
        inv_sigma,
        sigma,
        mu_biased,
        sigma_sq_biased,
        N,
    ):
        one_minus_decay = jnp.float32(1) - self.decay

        reduce_axes = tuple(range(len(x.shape) - 1))
        x_mean = jnp.mean(x, axis=reduce_axes, dtype=jnp.float32)

        assert(x_mean.shape == mu.value.shape)

        N.value = N.value + 1
        bias_correction = -jnp.expm1(N.value * jnp.log(self.decay))

        mu_biased.value = (mu_biased.value * self.decay +
            x_mean * one_minus_decay)

        new_mu = mu_biased.value / bias_correction

        # prev_mu needs to be unbiased (bias_correction only accounts
        # for the initial EMA with 0), since otherwise variance would
        # be off by a squared factor.
        # On the first iteration, simply treat x's variance as the 
        # full estimate of variance
        prev_mu = jnp.where(N.value == 0, new_mu, mu.value)

        sigma_sq_new = jnp.mean(
            (x - jnp.asarray(prev_mu, dtype=x.dtype)) *
            (x - jnp.asarray(new_mu, dtype=x.dtype)),
            axis=reduce_axes, dtype=jnp.float32)

        assert(sigma_sq_new.shape == sigma_sq_biased.value.shape)

        sigma_sq_biased.value = (sigma_sq_biased.value * self.decay +
            sigma_sq_new * one_minus_decay)

        sigma_sq = sigma_sq_biased.value / bias_correction

        # Write out new unbiased params
        mu.value = new_mu
        inv_sigma.value = lax.rsqrt(lax.max(sigma_sq, self.eps))
        sigma.value = jnp.reciprocal(inv_sigma.value)

    @nn.compact
    def __call__(self, mode, update_stats, x):
        if self.disable:
            return x 

        input_dtype = jnp.result_type(x)
        dim = x.shape[-1]

        # Current parameter estimates
        mu = self.variable("batch_stats", "mu",
            lambda: jnp.zeros((dim,), jnp.float32))
        inv_sigma = self.variable("batch_stats", "inv_sigma", 
            lambda: jnp.ones((dim,), jnp.float32))
        sigma = self.variable("batch_stats", "sigma",
            lambda: jnp.ones((dim,), jnp.float32))

        # Intermediate values used to compute the moving average
        mu_biased = self.variable("batch_stats", "mu_biased",
            lambda: jnp.zeros((dim,), jnp.float32))
        sigma_sq_biased = self.variable("batch_stats", "sigma_sq_biased",
            lambda: jnp.zeros((dim,), jnp.float32))

        N = self.variable("batch_stats", "N",
            lambda: jnp.zeros((), jnp.float32))

        if mode == 'normalize':
            if update_stats:
                self._update_stats(x, mu, inv_sigma,
                    sigma, mu_biased, sigma_sq_biased, N)
            return ((x - jnp.asarray(mu.value, input_dtype)) *
                    jnp.asarray(inv_sigma.value, input_dtype))
        elif mode == 'invert':
            return (x * jnp.asarray(sigma.value, input_dtype) +
                    jnp.asarray(mu.value, input_dtype))
        else:
            raise Exception("Invalid mode")

import jax
from jax import random, lax, numpy as jnp
from madrona_learn.moving_avg import EMANormalizer

decay = 0.999
batch_size = 1024
num_subchunks = 32
num_iters = 2
num_dims = 2

jax.config.update("jax_enable_x64", True)

rnd = random.PRNGKey(5)
rnd, mean_rnd, stddev_rnd = random.split(rnd, 3)

means = random.uniform(mean_rnd, (num_iters, num_dims)) * 10 - 5
stddevs = random.uniform(stddev_rnd, (num_iters, num_dims)) * 2 + 2

means = means.at[-1].set(-20)
stddevs = stddevs.at[-1].set(0.01)

#means = means.at[:].set(0)

@jax.vmap
def gen_values(rnd, mean, stddev):
    return random.normal(rnd, (batch_size, num_dims)) * stddev + mean

all_values = gen_values(
    random.split(rnd, num_iters), means, stddevs)

@jax.jit
def f(all_values):
    normalizer = EMANormalizer(
        decay = decay,
        out_dtype = jnp.float32,
    )

    naive_decay = jnp.array(decay, dtype=jnp.float64)
    naive_one_minus_decay = 1 - naive_decay

    def iter(i, carry):
        norm_est, naive_x_mean, naive_xx_mean = carry
    
        values = all_values[i]

        def subiter(j, values_mean_var):
            subvalues = values.reshape(
                num_subchunks,  batch_size // num_subchunks, num_dims)[j]

            return normalizer.update_input_stats(values_mean_var, j, subvalues)

        values_mean_var = normalizer.init_input_stats(norm_est)
        values_mean_var = lax.fori_loop(0, num_subchunks, subiter, values_mean_var)

        norm_est = normalizer.update_estimates(norm_est, values_mean_var)
    
        batch_x_mean = jnp.mean(values.astype(jnp.float64), axis=0)
        batch_xx_mean = jnp.mean(jnp.square(values.astype(jnp.float64)), axis=0)
        naive_x_mean = naive_decay * naive_x_mean + naive_one_minus_decay * batch_x_mean
        naive_xx_mean = naive_decay * naive_xx_mean + naive_one_minus_decay * batch_xx_mean

        return norm_est, naive_x_mean, naive_xx_mean

    norm_est = normalizer.init_estimates(all_values[0])
    naive_x_mean = jnp.zeros(all_values.shape[-1], dtype=jnp.float64)
    naive_xx_mean = jnp.zeros_like(naive_x_mean)

    return lax.fori_loop(0, all_values.shape[0], iter,
                         (norm_est, naive_x_mean, naive_xx_mean))


norm_est, naive_x_mean, naive_xx_mean = f(all_values)

naive_bias_correction = (
    -jnp.expm1(jnp.array(num_iters, dtype=jnp.float64) *
        jnp.log(jnp.array(decay, dtype=jnp.float64)))
)

naive_x_mean /= naive_bias_correction
naive_xx_mean /= naive_bias_correction

print(norm_est['mu'])
print(naive_x_mean)
print()

print(norm_est['sigma'])
print(jnp.sqrt(naive_xx_mean - naive_x_mean * naive_x_mean))
print()

print(jnp.mean(all_values, axis=(0, 1)),
      jnp.sqrt(jnp.var(all_values, axis=(0, 1))))

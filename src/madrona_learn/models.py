import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from typing import List

from .action import DiscreteActionDistributions
from .amp import amp

class MLP(nn.Module):
    num_channels: int
    num_layers: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        for i in range(self.num_layers):
            x = nn.Dense(
                    self.num_channels,
                    use_bias=True,
                    kernel_init=jax.nn.initializers.he_normal(),
                    bias_init=jax.nn.initializers.constant(0),
                    dtype=self.dtype,
                )(x)
            x = nn.LayerNorm(dtype=self.dtype)(x)
            x = nn.relu(x)

        return x

class DenseLayerDiscreteActor(nn.Module):
    actions_num_buckets: List[int]
    dtype: jnp.dtype

    def setup(self):
        total_action_dim = sum(self.actions_num_buckets)
        self.impl = nn.Dense(
                total_action_dim,
                use_bias=True,
                kernel_init=jax.nn.initializers.orthogonal(scale=0.01),
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )

    def __call__(self, features):
        logits = self.impl(features)
        return DiscreteActionDistributions(self.actions_num_buckets, logits)

class DenseLayerCritic(nn.Module):
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, features):
        return nn.Dense(
                1,
                use_bias=True,
                kernel_init=jax.nn.initializers.orthogonal(),
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )(features)

import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from typing import List

from .action import DiscreteActionDistributions
from .amp import amp

class MLP(nn.Module):
    num_channels : int
    num_layers : int

    #    self.net = nn.Sequential(*layers)

    #    for layer in self.net:
    #        if isinstance(layer, nn.Linear):
    #            nn.init.kaiming_normal_(
    #                layer.weight, nn.init.calculate_gain("relu"))
    #            if layer.bias is not None:
    #                nn.init.constant_(layer.bias, val=0)

    #def forward(self, inputs):
    #    return self.net(inputs)

    @nn.compact
    def __call__(self, inputs):
        x = inputs

        for i in range(self.num_layers):
            x = nn.Dense(self.num_channels, use_bias=True,
                kernel_init=jax.nn.initializers.he_normal())(x)
            x = nn.LayerNorm()(x)
            x = nn.relu(x)

        return x

class DenseLayerDiscreteActor(nn.Module):
    actions_num_buckets : List[int]

    def setup(self):
        total_action_dim = sum(self.actions_num_buckets)
        self.impl = nn.Dense(total_action_dim, use_bias=True,
            kernel_init=jax.nn.initializers.orthogonal(scale=0.01),
            bias_init=jax.nn.initializers.constant(0))

    def __call__(self, features):
        logits = self.impl(features)
        return DiscreteActionDistributions(self.actions_num_buckets, logits)

class DenseLayerCritic(nn.Module):
    @nn.compact
    def __call__(self, features):
        return nn.Dense(1, use_bias=True,
            kernel_init=jax.nn.initializers.orthogonal(),
            bias_init=jax.nn.initialzers.constant(0))(features)

import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

from .action import DiscreteActionDistributions
from .actor_critic import ActorCritic, DiscreteActor, Critic
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

class LinearLayerDiscreteActor(DiscreteActor):
    def __init__(self, actions_num_buckets, in_channels):
        total_action_dim = sum(actions_num_buckets)
        impl = nn.Linear(in_channels, total_action_dim)

        super().__init__(actions_num_buckets, impl)

        nn.init.orthogonal_(self.impl.weight, gain=0.01)
        nn.init.constant_(self.impl.bias, 0)

class LinearLayerCritic(Critic):
    def __init__(self, in_channels):
        super().__init__(nn.Linear(in_channels, 1))

        nn.init.orthogonal_(self.impl.weight)
        nn.init.constant_(self.impl.bias, 0)

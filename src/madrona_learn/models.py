import torch
import torch.nn as nn
import torch.nn.functional as F

from .action import DiscreteActionDistributions
from .recurrent_policy import LSTMRecurrentPolicy
from .actor_critic import ActorCritic

class SmallMLPBackbone(nn.Module):
    def __init__(self, process_obs_fn, input_dim, num_channels):
        super().__init__()

        self.process_obs = process_obs_fn

        layers = [
                nn.Linear(input_dim, num_channels),
                nn.ReLU(),
                nn.Linear(num_channels, num_channels),
                nn.ReLU()
            ]

        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, *in_obs):
        with torch.no_grad():
            processed_obs = self.process_obs(*in_obs)

        return self.net(processed_obs)

class LinearLayerDiscreteActor(ActorCritic.DiscreteActor):
    def __init__(self, actions_num_buckets, in_channels):
        total_action_dim = sum(actions_num_buckets)
        impl = nn.Linear(in_channels, total_action_dim)

        super().__init__(actions_num_buckets, impl)

        nn.init.orthogonal_(self.impl.weight, gain=0.01)
        nn.init.constant_(self.impl.bias, 0)

class LinearLayerCritic(ActorCritic.Critic):
    def __init__(self, in_channels):
        super().__init__(nn.Linear(in_channels, 1))

        nn.init.orthogonal_(self.impl.weight)
        nn.init.constant_(self.impl.bias, 0)

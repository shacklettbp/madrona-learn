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

    def forward(self, *in_obs):
        with torch.no_grad():
            processed_obs = self.process_obs(*in_obs)

        return self.net(processed_obs)

class LinearLayerDiscreteActor(ActorCritic.DiscreteActor):
    def __init__(self, actions_num_buckets, in_channels):
        total_action_dim = sum(actions_num_buckets)
        impl = nn.Linear(in_channels, total_action_dim, bias=False)

        super().__init__(actions_num_buckets, impl)

class LinearLayerCritic(ActorCritic.Critic):
    def __init__(self, in_channels):
        super().__init__(nn.Linear(in_channels, 1))

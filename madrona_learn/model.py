import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class SmallMLP(nn.Module):
    def __init__(self, input_dim, num_channels):
        super().__init__()

        layers = [
                nn.Linear(input_dim, num_channels),
                nn.ReLU(),
                nn.Linear(num_channels, num_channels),
                nn.ReLU()
            ]

        self.net = nn.Sequential(*layers)

    def forward(self, in_features):
        return self.net(in_features)

class SharedActorCritic(nn.Module):
    class DefaultDiscreteActor(nn.Module):
        def __init__(self, in_channels, actions_num_buckets):
            super().__init__()

            total_action_dim = math.sum(actions_num_buckets)

            self.net = nn.Linear(in_channels, total_action_dim, bias=False)

            self.dists = []

            for num_buckets 
    
        def forward(self, in_features):
            out_feats = self.net(in_features)
            return F.softmax(out_feats, dim=1)

        def act(self, in_features, out):
            out_feats = self.net(in_features)
            probs = torch.softmax(out_feats, dim=1)
            indices = torch.argmax(probs)
    
    class DefaultCritic(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.net = nn.Linear(in_channels, 1)
    
        def forward(self, in_features):
            return self.net(in_features)

    def __init__(self, process_obs_fn, core, actor, critic):
        super().__init__()

        self.process_obs = process_obs_fn
        self.core = core
        self.actor = actor
        self.critic = critic

    def _common(self, *obs):
        processed_obs = self.process_obs(*obs)
        return self.core(processed_obs)

    def forward(self, *obs):
        features = self._common(*obs)
        actions = self.actor(features)
        values = self.critic(features)

        return actions, values

    def act(self, actions_out, *obs):
        features = self._common(*obs)
        self.actor.act(features, out=actions_out)

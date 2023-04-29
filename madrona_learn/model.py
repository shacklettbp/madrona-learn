import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        core_layers = [
                nn.Linear(obs_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
                nn.ReLU()
            ]

        self.core = nn.Sequential(*layers)

        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

    def forward(self, obs_in):
        features = self.core(obs_in)
        actions = self.actor(features)
        values = self.critic(features)

        return actions, values

    def forward_inplace(self, obs_in, actions_out, values_out):
        features = self.core(obs_in)
        self.actor(features, out=actions_out)
        self.critic(features, out=values_out)

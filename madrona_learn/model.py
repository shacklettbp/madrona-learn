import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

def _sample_categorical_dists(dists, out):
    actions = [dist.sample() for dist in dists]
    torch.stack(actions, dim=1, out=out)

def _top_action_categorical_dists(dists, out):
    actions = [dist.mode() for dist in dists]
    torch.stack(actions, dim=1, out=out)

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

class LSTMRecurrentPolicy(nn.Module):
    def __init__(self, in_channels, num_hidden, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=num_hidden
            num_layers=num_layers,
            batch_first=True)

    def forward(self, in_features, hidden_state):
        return self.lstm(

class SharedActorCritic(nn.Module):
    class DefaultDiscreteActor(nn.Module):
        def __init__(self, in_channels, actions_num_buckets):
            super().__init__()

            total_action_dim = sum(actions_num_buckets)

            self.net = nn.Linear(in_channels, total_action_dim, bias=False)

            self.actions_num_buckets = actions_num_buckets

        def forward(self, in_features):
            out_feats = self.net(in_features)
            return F.softmax(out_feats, dim=1)

        def _setup_dists(self, all_logits):
            cur_bucket_offset = 0

            dists = []
            for num_buckets in self.actions_num_buckets:
                dists.append(Categorical(logits = all_logits[
                    :, cur_bucket_offset:cur_bucket_offset + num_buckets],
                    validate_args=False))
                cur_bucket_offset += num_buckets

            return dists

        def infer(self, in_features, actions_out, stochastic=True):
            all_logits = self.net(in_features)

            dists = self._setup_dists(all_logits)

            if stochastic:
                _sample_categorical_dists(dists, out=actions_out)
            else:
                _top_action_categorical_dists(dists, out=actions_out)
    
    class DefaultCritic(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.net = nn.Linear(in_channels, 1)
    
        def forward(self, in_features):
            return self.net(in_features)

    def __init__(self, process_obs_fn, backbone, rnn, actor, critic):
        super().__init__()

        self.process_obs = process_obs_fn
        self.backbone = backbone 
        self.rnn = rnn
        self.actor = actor
        self.critic = critic

    def _common(self, *obs, rnn_cur_hidden):
        processed_obs = self.process_obs(*obs)
        features = self.backbone(processed_obs)

        return self.rnn(features, rnn_cur_hidden)

    def infer(self, actions_out, rnn_cur_hidden, *obs):
        features, rnn_next_hidden = self._common(*obs, rnn_cur_hidden)
        self.actor.infer(features, actions_out=actions_out)

        return rnn_next_hidden

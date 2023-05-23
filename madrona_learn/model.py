import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from .recurrent_policy import LSTMRecurrentPolicy

def _sample_categorical_dists(dists, actions_out, log_probs_out):
    actions = [dist.sample() for dist in dists]
    log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]

    torch.stack(actions, dim=1, out=actions_out)
    torch.stack(log_probs, dim=1, out=log_probs_out)

def _top_action_categorical_dists(dists, out):
    actions = [dist.probs.argmax(dim=-1, keepdim=True) for dist in dists]
    torch.stack(actions, dim=1, out=out)

def _flatten_obs_sequence(obs):
    return [o.view(-1, *o.shape[2:]) for o in obs]

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

class ActorCritic(nn.Module):
    class DefaultDiscreteActor(nn.Module):
        def __init__(self, in_channels, actions_num_buckets):
            super().__init__()

            total_action_dim = sum(actions_num_buckets)

            self.net = nn.Linear(in_channels, total_action_dim, bias=False)

            self.actions_num_buckets = actions_num_buckets

        def _setup_dists(self, all_logits):
            cur_bucket_offset = 0

            dists = []
            for num_buckets in self.actions_num_buckets:
                dists.append(Categorical(logits = all_logits[
                    :, cur_bucket_offset:cur_bucket_offset + num_buckets],
                    validate_args=False))
                cur_bucket_offset += num_buckets

            return dists

        def infer(self, in_features, actions_out):
            logits = self.net(in_features)
            dists = self._setup_dists(logits)
            _top_action_categorical_dists(dists, out=actions_out)

        def rollout_infer(self, in_features, actions_out, log_probs_out):
            logits = self.net(in_features)
            dists = self._setup_dists(logits)

            _sample_categorical_dists(dists, actions_out=actions_out,
                                      log_probs_out=log_probs_out)

        def train(self, rollout_actions, in_features):
            logits = self.net(in_features)
            dists = self._setup_dists(logits)

            log_probs = []
            entropies = []
            for i, dist in enumerate(dists):
                log_probs.append(dist.log_prob(rollout_actions[:, i]))
                entropies.append(dist.entropy())

            return torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)
    
    class DefaultCritic(nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.net = nn.Linear(in_channels, 1)
    
        def infer(self, in_features, values_out):
            val_tmp = self.net(in_features)
            values_out[...] = val_tmp

        def train(self, in_features):
            return self.net(in_features)

    def __init__(self, backbone, actor, critic):
        super().__init__()

        self.backbone = backbone 
        self.actor = actor
        self.critic = critic
        self.rnn_hidden_shape = None

    def rollout_infer(self, actions_out, log_probs_out, values_out, *obs):
        features = self.backbone(*obs)
        self.actor.rollout_infer(features, actions_out=actions_out,
                                 log_probs_out=log_probs_out)
        self.critic.infer(features, values_out=values_out)

    def rollout_infer_values(self, values_out, *obs):
        features = self.backbone(*obs)
        self.critic.infer(features, values_out=values_out)

    def _train_backbone(self, *obs):
        obs_flattened = _flatten_obs_sequence(obs)
        return self.backbone(*obs_flattened)

    def _train_actor_critic(self, rollout_actions, in_features):
        T, N = rollout_actions.shape[0:2]
        flattened_actions = rollout_actions.view(T * N,
                                                 *rollout_actions.shape[2:])

        log_probs, entropies = self.actor.train(flattened_actions, in_features)
        values = self.critic.train(in_features)

        log_probs = log_probs.view(T, N, *log_probs.shape[1:])
        entropies = entropies.view(T, N, *entropies.shape[1:])
        values = values.view(T, N, *values.shape[1:])

        return log_probs, entropies, values

    def train(self, rollout_actions, *obs):
        features = self._train_backbone(*obs)
        return self._train_actor_critic(rollout_actions, features)

class RecurrentActorCritic(ActorCritic):
    def __init__(self, backbone, rnn, actor, critic):
        super().__init__(backbone, actor, critic)

        self.rnn = rnn
        self.rnn_hidden_shape = self.rnn.hidden_shape

    def rollout_infer(self, actions_out, log_probs_out, values_out,
                      rnn_hidden_out, rnn_hidden_in, *obs):
        features = self.backbone(*obs)
        rnn_out, rnn_next_hidden = self.rnn.infer(features, rnn_hidden_in)
        self.actor.rollout_infer(rnn_out, actions_out=actions_out,
                                 log_probs_out=log_probs_out)
        self.critic.infer(rnn_out, values_out=values_out)

        # FIXME, actual in place:
        rnn_hidden_out[...] = rnn_next_hidden

    def rollout_infer_values(self, values_out, rnn_hidden_in, *obs):
        features = self.backbone(*obs)
        rnn_out, _ = self.rnn.infer(features, rnn_hidden_in)
        self.critic.infer(rnn_out, values_out=values_out)

    def train(self, rnn_hidden_starts, dones, rollout_actions, *obs):
        features = self._train_backbone(*obs)
        features_seq = features.view(*dones.shape[0:2], *features.shape[1:])

        rnn_out_seq = self.rnn.eval_sequence(
            features_seq, rnn_hidden_starts, dones)

        rnn_out_flattened = rnn_out_seq.view(-1, *rnn_out_seq.shape[2:])

        return self._train_actor_critic(rollout_actions, rnn_out_flattened)

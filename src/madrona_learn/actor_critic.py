import torch
import torch.nn as nn
import torch.nn.functional as F
from .action import DiscreteActionDistributions

def _flatten_obs_sequence(obs):
    return [o.view(-1, *o.shape[2:]) for o in obs]

class ActorCritic(nn.Module):
    class DiscreteActor(nn.Module):
        def __init__(self, actions_num_buckets, impl):
            super().__init__()
    
            self.actions_num_buckets = actions_num_buckets
            self.impl = impl

        def forward(self, features_in):
            logits = self.impl(features_in)
    
            return DiscreteActionDistributions(
                    self.actions_num_buckets, logits=logits)

    class Critic(nn.Module):
        def __init__(self, impl):
            super().__init__()
            self.impl = impl 
    
        def forward(self, features_in):
            return self.impl(features_in)

    def __init__(self, backbone, actor, critic):
        super().__init__()

        self.backbone = backbone 
        self.actor = actor
        self.critic = critic
        self.rnn_hidden_shape = None

    # Direct call intended for debugging only, should use below
    # specialized functions
    def forward(self, *obs):
        features = self.backbone(*obs)

        action_dists = self.actor(features)
        values = self.critic(features)

        return action_dists, values

    def eval_infer(self, actions_out, *obs):
        features = self.backbone(*obs)
        action_dists = self.actor(features)
        action_dists.best(out=actions_out)

    def rollout_infer(self, actions_out, log_probs_out, values_out, *obs):
        action_dists, values = self(*obs)

        action_dists.sample(actions_out, log_probs_out)
        values_out[...] = values

    def rollout_infer_values(self, values_out, *obs):
        features = self.backbone(*obs)
        values_out[...] = self.critic(features)

    def _train_backbone(self, *obs):
        obs_flattened = _flatten_obs_sequence(obs)
        return self.backbone(*obs_flattened)

    def _train_actor_critic(self, rollout_actions, features_in):
        T, N = rollout_actions.shape[0:2]
        flattened_actions = rollout_actions.view(T * N,
                                                 *rollout_actions.shape[2:])

        action_dists = self.actor(features_in)

        log_probs, entropies = action_dists.action_stats(flattened_actions)
        values = self.critic(features_in)

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

    # Only call directly for debugging
    def forward(self, *obs):
        features = self.backbone(*obs)
        rnn_out, rnn_new_hidden = self.rnn.infer(features, rnn_hidden_in)

        action_dists = self.actor(rnn_out)
        values = self.critic(rnn_out)

        return action_dists, values, rnn_new_hidden

    def rollout_infer(self, actions_out, log_probs_out, values_out,
                      rnn_hidden_out, rnn_hidden_in, *obs):
        action_dists, values, rnn_new_hidden = self(*obs)

        action_dists.sample(actions_out, log_probs_out)

        # FIXME, actual in place:
        values_out[...] = values
        rnn_hidden_out[...] = rnn_new_hidden

    def rollout_infer_values(self, values_out, rnn_hidden_in, *obs):
        features = self.backbone(*obs)
        rnn_out, _ = self.rnn.infer(features, rnn_hidden_in)
        values_out[...] = self.critic(rnn_out)

    def train(self, rnn_hidden_starts, dones, rollout_actions, *obs):
        features = self._train_backbone(*obs)
        features_seq = features.view(*dones.shape[0:2], *features.shape[1:])

        rnn_out_seq = self.rnn.eval_sequence(
            features_seq, rnn_hidden_starts, dones)

        rnn_out_flattened = rnn_out_seq.view(-1, *rnn_out_seq.shape[2:])

        return self._train_actor_critic(rollout_actions, rnn_out_flattened)

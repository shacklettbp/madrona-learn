import torch
import torch.nn as nn
import torch.nn.functional as F
from .action import DiscreteActionDistributions

def _flatten_obs_sequence(obs):
    return [o.view(-1, *o.shape[2:]) for o in obs]

class ActorCritic(nn.Module):
    class DiscreteActor(nn.Module):
        def __init__(self, net, actions_num_buckets):
            super().__init__()
    
            self.net = net
            self.actions_num_buckets = actions_num_buckets
    
        def eval_infer(self, actions_out, features_in):
            logits = self.net(features_in)
    
            dists = DiscreteActionDistributions(self.actions_num_buckets, logits=logits)
            dists.best(out=actions_out)
    
        def rollout_infer(self, actions_out, log_probs_out, features_in):
            logits = self.net(features_in)
    
            dists = DiscreteActionDistributions(self.actions_num_buckets, logits=logits)
            dists.sample(actions_out, log_probs_out)
    
        def train(self, rollout_actions, features_in):
            logits = self.net(features_in)
            dists = DiscreteActionDistributions(self.actions_num_buckets, logits=logits)
    
            log_probs, entropies = dists.action_stats(rollout_actions)
            return log_probs, entropies
    
    class Critic(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
    
        def forward_inplace(self, values_out, features_in):
            val_tmp = self.net(features_in)
            values_out[...] = val_tmp

        def forward(self, features_in):
            return self.net(features_in)

    def __init__(self, backbone, actor, critic):
        super().__init__()

        self.backbone = backbone 
        self.actor = actor
        self.critic = critic
        self.rnn_hidden_shape = None

    def eval_infer(self, actions_out, *obs):
        features = self.backbone(*obs)
        self.actor.eval_infer(actions_out=actions_out, features_in=features)

    def rollout_infer(self, actions_out, log_probs_out, values_out, *obs):
        features = self.backbone(*obs)
        self.actor.rollout_infer(actions_out=actions_out,
                                 log_probs_out=log_probs_out,
                                 features_in=features)
        self.critic.forward_inplace(values_out=values_out, features_in=features)

    def rollout_infer_values(self, values_out, *obs):
        features = self.backbone(*obs)
        self.critic.forward_inplace(values_out=values_out, features_in=features)

    def _train_backbone(self, *obs):
        obs_flattened = _flatten_obs_sequence(obs)
        return self.backbone(*obs_flattened)

    def _train_actor_critic(self, rollout_actions, features_in):
        T, N = rollout_actions.shape[0:2]
        flattened_actions = rollout_actions.view(T * N,
                                                 *rollout_actions.shape[2:])

        log_probs, entropies = self.actor.train(flattened_actions, features_in)
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

    def rollout_infer(self, actions_out, log_probs_out, values_out,
                      rnn_hidden_out, rnn_hidden_in, *obs):
        features = self.backbone(*obs)
        rnn_out, rnn_next_hidden = self.rnn.infer(features, rnn_hidden_in)
        self.actor.rollout_infer(actions_out=actions_out,
                                 log_probs_out=log_probs_out,
                                 features_in=rnn_out)
        self.critic.infer(values_out=values_out,
                          features_in=rnn_out)

        # FIXME, actual in place:
        rnn_hidden_out[...] = rnn_next_hidden

    def rollout_infer_values(self, values_out, rnn_hidden_in, *obs):
        features = self.backbone(*obs)
        rnn_out, _ = self.rnn.infer(features, rnn_hidden_in)
        self.critic.infer(values_out=values_out,
                          features_in=rnn_out)

    def train(self, rnn_hidden_starts, dones, rollout_actions, *obs):
        features = self._train_backbone(*obs)
        features_seq = features.view(*dones.shape[0:2], *features.shape[1:])

        rnn_out_seq = self.rnn.eval_sequence(
            features_seq, rnn_hidden_starts, dones)

        rnn_out_flattened = rnn_out_seq.view(-1, *rnn_out_seq.shape[2:])

        return self._train_actor_critic(rollout_actions, rnn_out_flattened)

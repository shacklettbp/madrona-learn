import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .action import DiscreteActionDistributions

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

    def _flatten_obs_sequence(obs):
        return [o.view(-1, *o.shape[2:]) for o in obs]

    def forward(self, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        raise NotImplementedError

    def fwd_sequence(self, rnn_start_states, dones, *obs_in):
        raise NotImplementedError


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

class RecurrentStateConfig:
    def __init__(self, shapes):
        self.shapes = shapes

    def copy(storage, new_states):
        for o, i in zip(storage, new_states):
            o.copy_(i)
    
class ActorCritic(nn.Module):
    def __init__(self, backbone, actor, critic):
        super().__init__()

        self.backbone = backbone 
        self.recurrent_cfg = backbone.recurrent_cfg
        self.actor = actor
        self.critic = critic

    # Direct call intended for debugging only, should use below
    # specialized functions
    def forward(self, rnn_states, *obs):
        actor_features, critic_features, new_rnn_states = self.backbone(
            rnn_states, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        return action_dists, values, new_rnn_states

    def actor_infer(self, actions_out, rnn_states_out, rnn_states_in, *obs_in):
        actor_features = self.backbone.fwd_actor_only(
            rnn_states_out, rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        action_dists.best(out=actions_out)

    def critic_infer(self, values_out, rnn_states_out, rnn_states_in, *obs_in):
        features = self.backbone.fwd_critic_only(
            rnn_states_out, rnn_states_in, *obs_in)
        values_out[...] = self.critic(features)

    def rollout_infer(self, actions_out, log_probs_out, values_out,
                      rnn_states_out, rnn_states_in, *obs_in):
        actor_features, critic_features = self.backbone.fwd_rollout(
            rnn_states_out, rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        action_dists.sample(actions_out, log_probs_out)
        values_out[...] = values

    def train(self, rnn_states, dones, rollout_actions, *obs):
        actor_features, critic_features = self.backbone.fwd_sequence(
            rnn_states, dones, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        T, N = rollout_actions.shape[0:2]
        flattened_actions = rollout_actions.view(
            T * N, *rollout_actions.shape[2:])

        log_probs, entropies = action_dists.action_stats(flattened_actions)

        log_probs = log_probs.view(T, N, *log_probs.shape[1:])
        entropies = entropies.view(T, N, *entropies.shape[1:])
        values = values.view(T, N, *values.shape[1:])

        return log_probs, entropies, values

class BackboneEncoder(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.rnn_state_shape = None

    def forward(self, rnn_states, *inputs):
        features = self.net(*inputs)
        return features, ()

    def fwd_inplace(self, rnn_states_out, rnn_states_in, *inputs):
        return self.net(*inputs)

    # *inputs come in pre-flattened
    def fwd_sequence(self, rnn_start_states, dones, *flattened_inputs):
        return self.net(*inputs)

class RecurrentBackboneEncoder(nn.Module):
    def __init__(self, net, rnn):
        super().__init__()
        self.net = net
        self.rnn = rnn
        self.rnn_state_shape = rnn.hidden_shape

    def forward(self, rnn_states, *inputs):
        features = self.net(*inputs)
        rnn_out, new_rnn_states = self.rnn(features)

        return rnn_out, (new_rnn_states,)

    def fwd_inplace(self, rnn_states_out, rnn_states_in, *inputs):
        features = self.net(*inputs)
        rnn_out, new_rnn_states = self.rnn(features)

        # FIXME: proper inplace
        if rnn_states_out:
            rnn_states_out[...] = rnn_states_in

        return rnn_out

    # *inputs come in pre-flattened
    def fwd_sequence(self, rnn_start_states, dones, *flattened_inputs):
        features = self.net(*flattened_inputs)
        features_seq = features.view(*dones.shape[0:2], *features.shape[1:])

        rnn_out_seq = self.rnn.fwd_sequence(
            features_seq, rnn_start_states, dones)

        rnn_out_flattened = rnn_out_seq.view(-1, *rnn_out_seq.shape[2:])
        return rnn_out_flattened

class BackboneShared(Backbone):
    def __init__(self, process_obs, encoder):
        super().__init__()
        self.process_obs = process_obs
        self.encoder = encoder 
        if encoder.rnn_state_shape:
            self.recurrent_cfg = RecurrentStateConfig([encoder.rnn_state_shape])
        else:
            self.recurrent_cfg = RecurrentStateConfig([])

    def forward(self, rnn_states, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        features, new_rnn_states = self.encoder(processed_obs)
        return features, features, new_rnn_states

    def _rollout_common(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.encoder.fwd_inplace(
            rnn_states_out, rnn_states_in, processed_obs)

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(
            self, rnn_states_out, rnn_states_in, *obs_in)

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(
            self, rnn_states_out, rnn_states_in, *obs_in)

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        features = self._rollout_common(
            self, rnn_states_out, rnn_states_in, *obs_in)

        return features, features

    def fwd_sequence(self, rnn_start_states, dones, *obs_in):
        with torch.no_grad():
            flattened_obs = self._flatten_obs_sequence(obs_in)
            processed_obs = self.process_obs(*flattened_obs)
        
        features = self.encoder.fwd_sequence(
            rnn_start_states, dones, processed_obs)

        return features, features

class BackboneSeparate(Backbone):
    def __init__(self, process_obs, actor_encoder, critic_encoder):
        self.process_obs = process_obs
        self.actor_encoder = actor_encoder
        self.critic_encoder = critic_encoder
        self.recurrent_cfg = 


class BackboneSeparate(Backbone):
    def __init__(self, common_process_obs, actor_backbone, critic_backbone):
        super().__init__()
        self.common_process_obs = common_process_obs
        self.actor_backbone = actor_backbone
        self.critic_backbone = critic_backbone
        self.recurrent_cfg = RecurrentStateConfig([])

    def forward(self, rnn_states, *obs_in):
        with torch.no_grad():
            processed_obs = self.common_process_obs(*obs_in)

        actor_features = self.actor_backbone(processed_obs)
        critic_features = self.critic_backbone(processed_obs)
        return actor_features, critic_features, ()

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.common_process_obs(*obs_in)

        return self.actor_backbone(processed_obs)

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.common_process_obs(*obs_in)

        return self.critic_backbone(processed_obs)

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        actor_features, critic_features, _ = self(rnn_states_in, *obs_in)
        return actor_features, critic_features

    def fwd_sequence(self, rnn_start_states, dones, *obs_in):
        flattened_obs = self._flatten_obs_sequence(obs_in)

        with torch.no_grad():
            processed_obs = self.common_process_obs(*flattened_obs)

        actor_features = self.actor_backbone(processed_obs)
        critic_features = self.critic_backbone(processed_obs)

        return actor_features, critic_features


class RecurrentBackboneShared(Backbone):
    def __init__(self, process_obs, encoder, rnn):
        super().__init__()

        self.process_obs = process_obs
        self.encoder = encoder
        self.rnn = rnn
        self.recurrent_cfg = RecurrentStateConfig([rnn.hidden_shape])

    def _encode(self, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        return self.encoder(processed_obs)

    def forward(self, rnn_states_in, *obs_in):
        features = self._encode(*obs_in)
        rnn_out, new_rnn_states = self.rnn(features, rnn_states_in)

        return features, features, (new_rnn_states,)

    def _rollout_common(self, rnn_states_out, rnn_states_in, *obs_in):
        features = self._encode(*obs_in)
        rnn_out, new_rnn_states = self.rnn(features, rnn_states_in)

        # FIXME inplace
        if rnn_states_out:
            rnn_states_out[...] = new_rnn_states

        return rnn_out

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(rnn_states_out, rnn_states_in, *obs_in)

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        return self._rollout_common(rnn_states_out, rnn_states_in, *obs_in)

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        features = self.rollout_common(rnn_states_out, rnn_states_in, *obs_in)
        return features, featuers

    def fwd_sequence(self, rnn_start_states, dones, *obs_in):
        flattened_obs = self._flatten_obs_sequence(obs_in)
        features = self._encode(*flattened_obs)

        features_seq = features.view(*dones.shape[0:2], *features.shape[1:])

        rnn_out_seq = self.rnn.fwd_sequence(
            features_seq, rnn_start_states, dones)

        rnn_out_flattened = rnn_out_seq.view(-1, *rnn_out_seq.shape[2:])

        return rnn_out_flattened, rnn_out_flattened


class RecurrentBackboneSeparate(Backbone):
    def __init__(self, common_process_obs,
                 actor_encoder, actor_rnn,
                 critic_encoder, critic_rnn):
        super().__init__()

        self.common_process_obs = common_process_obs
        self.actor_encoder = actor_encoder
        self.actor_rnn = actor_rnn
        self.critic_encoder = critic_encoder
        self.critic_rnn = critic_rnn
        self.recurrent_cfg = RecurrentStateConfig([
            actor_rnn.hidden_shape, critic_rnn.hidden_shape])

    def forward(self, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_features = self.actor_encoder(processed_obs)
        critic_features = self.critic_encoder(processed_obs)

        actor_rnn_out, new_actor_rnn_states = self.actor_rnn(
            actor_features, rnn_states_in[0])

        critic_rnn_out, new_critic_rnn_states = self.critic_rnn(
            critic_features, rnn_states_in[1])

        return actor_rnn_out, critic_rnn_out,
            (new_actor_rnn_states, new_critic_rnn_states)

    def fwd_actor_only(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_features = self.actor_encoder(processed_obs)
        actor_rnn_out, new_actor_rnn_states = self.actor_rnn(
                actor_features, rnn_states_in[0])

        if rnn_states_out:
            rnn_states_out[0][...] = new_actor_rnn_states

        return actor_rnn_out

    def fwd_critic_only(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        critic_features = self.critic_encoder(processed_obs)
        critic_rnn_out, new_critic_rnn_states = self.critic_rnn(
                critic_features, rnn_states_in[1])

        if rnn_states_out:
            rnn_states_out[1][...] = new_critic_rnn_states

        return critic_rnn_out

    def fwd_rollout(self, rnn_states_out, rnn_states_in, *obs_in):
        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_features = self.actor_encoder(processed_obs)
        critic_features = self.critic_encoder(processed_obs)

        actor_rnn_out, new_actor_rnn_states = self.actor_rnn(
            actor_features, rnn_states_in[0])

        critic_rnn_out, new_critic_rnn_states = self.critic_rnn(
            critic_features, rnn_states_in[1])

        rnn_states_out[0][...] = actor_rnn_out
        rnn_states_out[1][...] = critic_rnn_out

        return actor_rnn_out, critic_rnn_out

    def fwd_sequence(self, rnn_start_states, dones, *obs_in):
        flattened_obs = self._flatten_obs_sequence(obs_in)

        with torch.no_grad():
            processed_obs = self.process_obs(*obs_in)

        actor_features = self.actor_encoder(processed_obs)
        critic_features = self.critic_encoder(processed_obs)

        actor_features_seq = actor_features.view(
                *dones.shape[0:2], *actor_features.shape[1:])
        critic_features_seq = critic_features.view(
                *dones.shape[0:2], *critic_features.shape[1:])

        rnn_out_seq = self.rnn.fwd_sequence(
            features_seq, rnn_start_states, dones)

        rnn_out_flattened = rnn_out_seq.view(-1, *rnn_out_seq.shape[2:])

        return rnn_out_flattened, rnn_out_flattened

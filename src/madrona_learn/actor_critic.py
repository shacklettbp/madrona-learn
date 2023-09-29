import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from typing import Optional, List, Union, Callable

from dataclasses import dataclass
from .action import DiscreteActionDistributions
from .profile import profile

class Backbone(nn.Module):
    def _flatten_obs_sequence(self, obs):
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


class RecurrentStateConfig:
    def __init__(self, shapes):
        self.shapes = shapes


class ActorCritic(nn.Module):
    backbone : nn.Module
    actor : nn.Module
    critic : nn.Module

    def setup(self):
        self.recurrent_cfg = self.backbone.recurrent_cfg

    def debug(self, rnn_states, *obs):
        actor_features, critic_features, new_rnn_states = self.backbone(
            rnn_states, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        return action_dists, values, new_rnn_states

    def actor_only(self, rnn_states_in, *obs_in):
        actor_features, rnn_states_out = self.backbone.actor_only(
                rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        return action_dists.best(), rnn_states_out

    def critic_only(self, rnn_states_in, *obs_in):
        critic_features, rnn_states_out = self.backbone.critic_only(
            rnn_states_in, *obs_in)
        values = self.critic(critic_features)
        return values, rnn_states_out

    def rollout(self, prng_key, rnn_states_in, *obs_in):
        actor_features, critic_features, rnn_states_out = self.backbone(
            rnn_states_in, *obs_in)

        action_dists = self.actor(actor_features)
        actions, log_probs = action_dists.sample(prng_key)

        values = self.critic(critic_features)
        return actions, log_probs, values, rnn_states_out

    def update(self, rnn_states, sequence_breaks, rollout_actions, *obs):
        actor_features, critic_features = self.backbone.sequence(
            rnn_states, sequence_breaks, *obs)

        action_dists = self.actor(actor_features)
        values = self.critic(critic_features)

        T, N = rollout_actions.shape[0:2]
        flattened_actions = rollout_actions.view(
            T * N, *rollout_actions.shape[2:])

        log_probs, entropies = action_dists.action_stats(flattened_actions)

        log_probs = log_probs.reshape(T, N, *log_probs.shape[1:])
        entropies = entropies.reshape(T, N, *entropies.shape[1:])
        values = values.reshape(T, N, *values.shape[1:])

        return log_probs, entropies, values

class BackboneEncoder(nn.Module):
    net : nn.Module

    def setup(self):
        self.rnn_state_shape = None

    def __call__(self, rnn_states, *inputs):
        features = self.net(*inputs)
        return features, None

    def sequence(self, rnn_start_states, sequence_breaks, *flattened_inputs):
        return self.net(*flattened_inputs)

class RecurrentBackboneEncoder(nn.Module):
    net : nn.Module
    rnn : nn.Module

    def setup(self):
        self.rnn_state_shape = self.rnn.hidden_shape

    def __call__(self, rnn_states_in, *inputs):
        features = self.net(*inputs)
        rnn_out, new_rnn_states = self.rnn(features, rnn_states_in)

        return rnn_out, new_rnn_states

    # *inputs come in pre-flattened
    def sequence(self, rnn_start_states, sequence_breaks,
                 *flattened_inputs):
        features = self.net(*flattened_inputs)
        features_seq = features.reshape(
            *sequence_breaks.shape[0:2], *features.shape[1:])

        with profile('rnn.fwd_sequence'):
            rnn_out_seq = self.rnn.sequence(
                features_seq, rnn_start_states, sequence_breaks)

        rnn_out_flattened = rnn_out_seq.reshape(-1, *rnn_out_seq.shape[2:])
        return rnn_out_flattened


class BackboneShared(Backbone):
    prefix : Union[nn.Module, Callable]
    encoder : nn.Module

    def setup(self):
        if encoder.rnn_state_shape:
            self.recurrent_cfg = RecurrentStateConfig([encoder.rnn_state_shape])
            self.extract_rnn_state = lambda x: x[0] if x != None else None
            self.package_rnn_state = lambda x: (x,)
        else:
            self.recurrent_cfg = RecurrentStateConfig([])
            self.extract_rnn_state = lambda x: None
            self.package_rnn_state = lambda x: ()

    def _rollout_common(self, rnn_states_in, *obs_in):
        processed = self.prefix(*obs_in)

        features, rnn_states_out = self.encoder(
            self.extract_rnn_state(rnn_states_in),
            processed_obs,
        )

        return features, self.package_rnn_state(rnn_states_out)

    def __call__(self, rnn_states_in, *obs_in):
        features, rnn_states_out = self._rollout_common(rnn_states_in, *obs_in)
        return features, features, rnn_states_out

    def actor_only(self, rnn_states_in, *obs_in):
        return self._rollout_common(rnn_states_in, *obs_in)

    def critic_only(self, rnn_states_in, *obs_in):
        return self._rollout_common(rnn_states_in, *obs_in)

    def sequence(self, rnn_start_states, sequence_breaks, *obs_in):
        flattened_obs = self._flatten_obs_sequence(obs_in)
        processed_obs = self.process_obs(*flattened_obs)
        
        features = self.encoder.sequence(
            self.extract_rnn_state(rnn_start_states),
            sequence_breaks, processed_obs)

        return features, features


class BackboneSeparate(Backbone):
    prefix : Union[nn.Module, Callable]
    actor_encoder : nn.Module
    critic_encoder : nn.Module

    def setup(self):
        rnn_state_shapes = []

        if self.actor_encoder.rnn_state_shape == None:
            self.extract_actor_rnn_state = lambda rnn_states: None
        else:
            actor_rnn_idx = len(rnn_state_shapes)
            rnn_state_shapes.append(self.actor_encoder.rnn_state_shape)
            self.extract_actor_rnn_state = \
                lambda rnn_states: rnn_states[actor_rnn_idx]

        if self.critic_encoder.rnn_state_shape == None:
            self.extract_critic_rnn_state = lambda rnn_states: None
        else:
            critic_rnn_idx = len(rnn_state_shapes)
            rnn_state_shapes.append(self.critic_encoder.rnn_state_shape)
            self.extract_critic_rnn_state = \
                lambda rnn_states: rnn_states[critic_rnn_idx]

        if (self.actor_encoder.rnn_state_shape and
                self.critic_encoder.rnn_state_shape):
            self.package_rnn_states = lambda a, c: (a, c)
        elif self.actor_encoder.rnn_state_shape:
            self.package_rnn_states = lambda a, c: (a,)
        elif self.critic_encoder.rnn_state_shape:
            self.package_rnn_states = lambda a, c: (c,)
        else:
            self.package_rnn_states = lambda a, c: ()

        self.recurrent_cfg = RecurrentStateConfig(rnn_state_shapes)

    def __call__(self, rnn_states_in, *obs_in):
        processed = self.prefix(*obs_in)

        actor_features, actor_rnn_out = self.actor_encoder(
            self.extract_actor_rnn_state(rnn_states_in), processed)
        critic_features, critic_rnn_out = self.critic_encoder(
            self.extract_critic_rnn_state(rnn_states_in), processed)

        return actor_features, critic_features, self.package_rnn_states(
            actor_rnn_out, critic_rnn_out)

    def actor_only(self, rnn_states_in, *obs_in):
        processed = self.prefix(*obs_in)

        features, rnn_states_out = self.actor_encoder(
            self.extract_actor_rnn_state(rnn_states_in), processed)

        return features, self.package_rnn_states(
                rnn_states_out, self.extract_critic_rnn_state(rnn_states_in))

    def critic_only(self, rnn_states_in, *obs_in):
        processed = self.prefix(*obs_in)

        features, rnn_states_out = self.critic_encoder(
            self.extract_critic_rnn_state(rnn_states_in),
            processed_obs)

        return features, self.package_rnn_states(
            self.extract_actor_rnn_state(rnn_states_in), rnn_states_out)

    def sequence(self, rnn_start_states, sequence_breaks, *obs_in):
        flattened_obs = self._flatten_obs_sequence(obs_in)
        processed = self.prefix(*flattened_obs)
        
        actor_features = self.actor_encoder.sequence(
            self.extract_actor_rnn_state(rnn_start_states),
            sequence_breaks, processed)

        critic_features = self.critic_encoder.sequence(
            self.extract_critic_rnn_state(rnn_start_states),
            sequence_breaks, processed)

        return actor_features, critic_features

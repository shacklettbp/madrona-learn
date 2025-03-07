import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import frozen_dict, FrozenDict

from dataclasses import dataclass
from typing import Optional, List, Union, Callable

from .dists import DiscreteActionDistributions
from .profile import profile

class Backbone(nn.Module):
    def _flatten_obs_sequence(self, obs):
        return jax.tree_map(lambda o: o.reshape(-1, *o.shape[2:]), obs)

    @nn.nowrap
    def init_recurrent_state(self, N):
        raise NotImplementedError

    @nn.nowrap
    def clear_recurrent_state(self, recurrent_states, should_clear):
        raise NotImplementedError

    def __call__(self, rnn_states, inputs, train):
        raise NotImplementedError

    def sequence(
        self,
        rnn_start_states,
        sequence_ends,
        flattened_inputs,
        train,
    ):
        raise NotImplementedError


class ActorCritic(nn.Module):
    backbone: Backbone
    actor: nn.Module
    critic: nn.Module

    @nn.nowrap
    def init_recurrent_state(self, N):
        return self.backbone.init_recurrent_state(N)

    @nn.nowrap
    def clear_recurrent_state(self, recurrent_states, should_clear):
        return self.backbone.clear_recurrent_state(
            recurrent_states, should_clear)

    def setup(self):
        pass

    def actor_only(self, rnn_states_in, obs_in, train=False):
        actor_features, rnn_states_out = self.backbone.actor_only(
                rnn_states_in, obs_in, train=train)

        action_dists = self.actor(actor_features, train=train)

        return FrozenDict({
            'actions': action_dists.best(),
        }), rnn_states_out

    def critic_only(self, rnn_states_in, obs_in, train=False):
        critic_features, rnn_states_out = self.backbone.critic_only(
            rnn_states_in, obs_in, train=train)
        critic_out = self.critic(critic_features, train=train)

        return FrozenDict({
            'critic': critic_out,
        }), rnn_states_out

    def rollout(self, prng_key, rnn_states_in, obs_in, train=False,
                sample_actions=True, return_debug=False):
        actor_features, critic_features, rnn_states_out = self.backbone(
            rnn_states_in, obs_in, train=train)

        action_dists = self.actor(actor_features, train=train)
        results = {}

        if sample_actions:
            actions, log_probs = action_dists.sample(prng_key)
            results['log_probs'] = log_probs
        else:
            actions = action_dists.best()

        results['actions'] = actions

        results['critic'] = self.critic(critic_features, train=train)

        #if return_debug:
        #    results['action_probs'] = action_dists.probs()
        #    results['action_logits'] = action_dists.logits()

        return frozen_dict.freeze(results), rnn_states_out

    def update(
            self,
            rnn_states,
            sequence_breaks,
            rollout_actions,
            obs,
            train=True,
        ):
        actor_features, critic_features = self.backbone.sequence(
            rnn_states, sequence_breaks, obs, train=train)

        action_dists = self.actor(actor_features, train=train)
        critic_out = self.critic(critic_features, train=train)

        T, N = sequence_breaks.shape[0:2]
        flattened_actions = jax.tree.map(
            lambda a:  a.reshape(T * N, *a.shape[2:]),
            rollout_actions)

        log_probs, entropies = action_dists.action_stats(flattened_actions)

        log_probs = jax.tree.map(lambda x: x.reshape(T, N, *x.shape[1:]), log_probs)
        entropies = jax.tree.map(lambda x: x.reshape(T, N, *x.shape[1:]), entropies)
        critic_out = jax.tree.map(
            lambda x: x.reshape(T, N, *x.shape[1:]), critic_out)

        return FrozenDict({
            'log_probs': log_probs,
            'entropies': entropies,
            'critic': critic_out,
        })


class BackboneEncoder(nn.Module):
    net: nn.Module

    @nn.nowrap
    def init_recurrent_state(self, N):
        return ()

    @nn.nowrap
    def clear_recurrent_state(self, recurrent_states, should_clear):
        return ()

    def __call__(self, rnn_states, inputs, train):
        features = self.net(inputs, train=train)
        return features, ()

    def sequence(
        self,
        rnn_start_states,
        sequence_ends,
        flattened_inputs,
        train,
    ):
        return self.net(flattened_inputs, train=train)


class RecurrentBackboneEncoder(nn.Module):
    net: nn.Module
    rnn: nn.Module

    @nn.nowrap
    def init_recurrent_state(self, N):
        return self.rnn.init_recurrent_state(N)

    @nn.nowrap
    def clear_recurrent_state(self, recurrent_states, should_clear):
        return self.rnn.clear_recurrent_state(
            recurrent_states, should_clear)

    def setup(self):
        pass

    def __call__(self, rnn_states_in, *inputs, train):
        features = self.net(*inputs, train=train)
        rnn_out, new_rnn_states = self.rnn(rnn_states_in, features, train)

        return rnn_out, new_rnn_states

    # *inputs come in pre-flattened
    def sequence(
        self,
        rnn_start_states,
        sequence_ends,
        flattened_inputs,
        train
    ):
        features = self.net(flattened_inputs, train=train)

        features_seq = jax.tree.map(
            lambda x: x.reshape(*sequence_ends.shape[0:2], *x.shape[1:]),
            features)

        with profile('rnn.fwd_sequence'):
            rnn_out_seq = self.rnn.sequence(
                rnn_start_states, sequence_ends, features_seq, train=train)

        rnn_out_flattened = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            rnn_out_seq)
        return rnn_out_flattened


class BackboneShared(Backbone):
    prefix: Union[nn.Module, Callable]
    encoder: nn.Module

    @nn.nowrap
    def init_recurrent_state(self, N):
        return self.encoder.init_recurrent_state(N)

    @nn.nowrap
    def clear_recurrent_state(self, recurrent_states, should_clear):
        return self.encoder.clear_recurrent_state(
            recurrent_states, should_clear)

    def setup(self):
        pass

    def _rollout_common(self, rnn_states_in, obs_in, train):
        processed = self.prefix(obs_in, train=train)

        features, rnn_states_out = self.encoder(
            rnn_states_in, processed, train=train)

        return features, rnn_states_out

    def __call__(self, rnn_states_in, obs_in, train):
        features, rnn_states_out = self._rollout_common(
            rnn_states_in, obs_in, train)
        return features, features, rnn_states_out

    def actor_only(self, rnn_states_in, obs_in, train):
        return self._rollout_common(rnn_states_in, obs_in, train)

    def critic_only(self, rnn_states_in, obs_in, train):
        return self._rollout_common(rnn_states_in, obs_in, train)

    def sequence(self, rnn_start_states, sequence_ends, obs_in, train):
        flattened_obs = self._flatten_obs_sequence(obs_in)
        processed_obs = self.prefix(flattened_obs, train=train)
        
        features = self.encoder.sequence(
            rnn_start_states, sequence_ends, processed_obs, train=train)

        return features, features


class BackboneSeparate(Backbone):
    prefix: Union[nn.Module, Callable]
    actor_encoder: nn.Module
    critic_encoder: nn.Module

    @nn.nowrap
    def init_recurrent_state(self, N):
        return (self.actor_encoder.init_recurrent_state(N),
                self.critic_encoder.init_recurrent_state(N))

    @nn.nowrap
    def clear_recurrent_state(self, recurrent_states, should_clear):
        return (self.actor_encoder.clear_recurrent_state(recurrent_states[0],
                                                         should_clear),
                self.critic_encoder.clear_recurrent_state(recurrent_states[1],
                                                         should_clear))

    def setup(self):
        pass

    def __call__(self, rnn_states_in, obs_in, train):
        processed = self.prefix(obs_in, train=train)

        actor_features, actor_rnn_out = self.actor_encoder(
            rnn_states_in[0], processed, train=train)
        critic_features, critic_rnn_out = self.critic_encoder(
            rnn_states_in[1], processed, train=train)

        return actor_features, critic_features, (actor_rnn_out, critic_rnn_out)

    def actor_only(self, rnn_states_in, obs_in, train):
        processed = self.prefix(obs_in, train=train)

        features, rnn_states_out = self.actor_encoder(
            rnn_states_in[0], processed, train=train)

        return features, (rnn_states_out, rnn_states_in[1])

    def critic_only(self, rnn_states_in, obs_in, train):
        processed = self.prefix(obs_in, train=train)

        features, rnn_states_out = self.critic_encoder(
            rnn_states_in[1], processed, train=train)

        return features, (rnn_states_in[0], rnn_states_out)

    def sequence(self, rnn_start_states, sequence_ends, obs_in, train):
        flattened_obs = self._flatten_obs_sequence(obs_in)
        processed = self.prefix(flattened_obs, train=train)
        
        actor_features = self.actor_encoder.sequence(
            rnn_start_states[0], sequence_ends, processed, train=train)

        critic_features = self.critic_encoder.sequence(
            rnn_start_states[1], sequence_ends, processed, train=train)

        return actor_features, critic_features

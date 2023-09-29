import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

from madrona_learn.models import MLP
from madrona_learn.rnn import LSTM, FastLSTM
from madrona_learn.moving_avg import EMANormalizer
from madrona_learn.actor_critic import *
from madrona_learn.models import *

import math

def assert_valid_input(tensor):
    assert(not jnp.isnan(tensor).any())
    assert(not jnp.isinf(tensor).any())

class ProcessObsCommon(nn.Module):
    num_lidar_samples: int

    @nn.compact
    def __call__(self,
                 self_obs,
                 teammate_obs,
                 opponent_obs,
                 lidar,
                 alive,
                 opponent_masks):
        assert_valid_input(self_obs)
        assert_valid_input(teammate_obs)
        assert_valid_input(opponent_obs)
        assert_valid_input(lidar)
        assert_valid_input(alive)
        assert_valid_input(opponent_masks)

        lidar_normalized = EMANormalizer(0.99999)(lidar)

        lidar_processed = nn.Conv(
                features=1,
                kernel_size=self.num_lidar_samples,
                padding='circular',
            )(lidar_normalized.unsqueeze(axis=-1))

        lidar_processed = lax.stop_gradient(lidar_processed.squeeze(axis=-1))

        return (self_obs, teammate_obs, opponent_obs, lidar_processed,
                alive, opponent_masks)


class ActorNet(nn.Module):
    num_mlp_channels: int

    @nn.compact
    def __call__(self, obs_tensors):
        self_obs, teammate_obs, opponent_obs, lidar_processed, alive, opponent_masks = \
            obs_tensors
        
        opponent_obs_masked = opponent_obs * opponent_masks

        flattened = jnp.concatenate([
            self_obs.reshape(self_obs.shape[0], -1),
            teammate_obs.reshape(teammate_obs.shape[0], -1),
            opponent_obs_masked.reshape(opponent_obs_masked.shape[0], -1),
            alive.reshape(alive.shape[0], -1),
        ], axis=1)

        normalized = EMANormalizer(0.99999)(flattened)

        features = jnp.concatenate([normalized, lidar_processed], axis=1)
        features = lax.stop_gradient(features)

        return MLP(
                num_channels = self.num_mlp_channels,
                num_layers = 2,
            )(features)


class CriticNet(nn.Module):
    num_mlp_channels : int

    @nn.compact
    def __call__(self, obs_tensors):
        self_obs, teammate_obs, opponent_obs, lidar_processed, alive, opponent_masks = \
            obs_tensors
        
        with torch.no_grad():
            flattened = torch.cat([
                self_obs.reshape(self_obs.shape[0], -1),
                teammate_obs.reshape(teammate_obs.shape[0], -1),
                opponent_obs.reshape(opponent_obs.shape[0], -1),
                alive.reshape(alive.shape[0], -1),
            ], dim=1)

            normalized = self.normalizer(flattened)

            features = jnp.concatenate([normalized, lidar_processed], axis=1)
            features = lax.stop_gradient(features)

        return MLP(
                num_channels = self.num_mlp_channels,
                num_layers = 2,
            )(features)

def make_tdm_policy(num_obs_features, num_channels, num_lidar_samples):
    obs_common = ProcessObsCommon(num_lidar_samples)

    actor_encoder = RecurrentBackboneEncoder(
        net = ActorNet(num_channels),
        rnn = LSTM(
            in_channels = num_channels,
            hidden_channels = num_channels,
            num_layers = 1,
        ),
    )

    critic_encoder = RecurrentBackboneEncoder(
        net = CriticNet(num_channels),
        rnn = LSTM(
            in_channels = num_channels,
            hidden_channels = num_channels,
            num_layers = 1,
        ),
    )

    backbone = BackboneSeparate(
        prefix = obs_common,
        actor_encoder = actor_encoder,
        critic_encoder = critic_encoder,
    )

    return ActorCritic(
        backbone = backbone,
        actor = DenseLayerDiscreteActor(
            [4, 8, 5, 5, 2, 2],
        ),
        critic = DenseLayerCritic(),
    )

def fake_rollout_iter(policy, obs, rnn_states, step_key):
    actions, log_probs, values, rnn_states = policy.rollout(
        step_key, rnn_states, *obs)

    return rnn_states

def fake_rollout_loop(policy, obs, rnn_states, prng_key):
    def iter(i, v):
        prng_key, step_key = random.split(prng_key)
        rnn_states = fake_rollout_iter(policy, obs, rnn_states, step_key)

    lax.fori_loop(0, 100, iter, None)

def test():
    policy = make_tdm_policy(5, 128, 32)

    obs_input = jnp.zeros(1024, 5, dtype=jnp.float32)

    cur_rnn_states = []
    
    for shape in policy.recurrent_cfg.shapes:
        cur_rnn_states.append(jnp.zeros(
            *shape[0:2], actions.shape[0], shape[2], dtype=jnp.float32))

    prng_key = random.PRNGKey(5)

    fake_rollout_loop(policy, obs, cur_rnn_states, prng_key)

if __name__ == "__main__":
    test()

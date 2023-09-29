import jax
import jax.nn
from jax import lax, random, numpy as jnp

from dataclasses import dataclass
from typing import List

@dataclass
class DiscreteActionDistributions:
    actions_num_buckets : List[int]
    all_logits: jax.Array

    def _iter_dists(self, cb):
        cur_bucket_offset = 0
        for num_buckets in self.actions_num_buckets:
            cb(self.all_logits[
                :, cur_bucket_offset:cur_bucket_offset + num_buckets])

    def _compute_log_probs(self, logits, actions):
        action_logits = logits[..., actions]
        return (
            action_logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True))

    def sample(self, prng_key):
        all_actions = []
        all_log_probs = []

        def sample_actions(logits):
            prng_key, rnd = random.split(prng_key)

            actions = random.categorical(prng_key, logits)
            action_log_probs = self._compute_log_probs(logits, actions)

            all_actions.append(actions)
            all_log_probs.append(action_log_probs)

        self.iter_dists(sample_actions)

        return (jnp.stack(all_actions, axis=1),
                jnp.stack(all_log_probs, axis=1))

    def best(self):
        all_actions = []

        def best_action(logits):
            all_actions.append(jnp.argmax(logits))

        self.iter_dists(best_action)

        return jnp.stack(all_actions, axis=1)

    def action_stats(self, actions):
        all_log_probs = []
        all_entropies = []

        def compute_stats(logits):
            log_probs = self._compute_log_probs(logits, actions)
            p_logp = jnp.exp(log_probs) * log_probs
            entropies = -p_logp.sum(axis=-1)

            all_log_probs.append(log_probs)
            all_entropies.append(entropies)

        self.iter_dists(compute_stats)

        return (jnp.stack(all_log_probs, axis=1),
                jnp.stack(all_entropies, axis=1))

    def probs(self):
        all_probs = []

        def compute_probs(logits):
            log_probs = self._compute_log_probs(logits, actions)
            probs = jnp.exp(log_probs)
            all_probs.append(probs)

        self.iter_dists(compute_probs)

        return all_probs

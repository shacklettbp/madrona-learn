import jax
import jax.nn
from jax import lax, random, numpy as jnp
import flax

from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class DiscreteActionDistributions:
    actions_num_buckets : List[int]
    all_logits: jax.Array

    def _iter_dists(self, cb, *args):
        cur_bucket_offset = 0
        for i, num_buckets in enumerate(self.actions_num_buckets):
            logits_slice = self.all_logits[
                :, cur_bucket_offset:cur_bucket_offset + num_buckets]

            args_sliced = tuple(jnp.expand_dims(a[..., i], -1) for a in args)
            cb(logits_slice, *args_sliced)

    def sample(self, prng_key):
        all_actions = []
        all_log_probs = []

        def sample_actions(logits):
            nonlocal prng_key

            prng_key, rnd = random.split(prng_key)

            actions = random.categorical(prng_key, logits)
            actions = jnp.expand_dims(actions, axis=-1)

            action_logits = jnp.take_along_axis(logits, actions, axis=-1)
            action_log_probs = action_logits - jax.nn.logsumexp(
                logits, axis=-1, keepdims=True)

            all_actions.append(actions)
            all_log_probs.append(action_log_probs)

        self._iter_dists(sample_actions)

        return (jnp.concatenate(all_actions, axis=-1),
                jnp.concatenate(all_log_probs, axis=-1))

    def best(self):
        all_actions = []

        def best_action(logits):
            all_actions.append(jnp.argmax(logits, keepdims=True, axis=-1))

        self._iter_dists(best_action)

        return jnp.concatenate(all_actions, axis=-1)

    def action_stats(self, all_actions):
        all_log_probs = []
        all_entropies = []

        def compute_stats(logits, actions):
            log_probs = \
                logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
            p_logp = jnp.exp(log_probs) * log_probs
            entropies = -p_logp.sum(axis=-1, keepdims=True)

            action_log_probs = jnp.take_along_axis(log_probs, actions, axis=-1)

            all_log_probs.append(action_log_probs)
            all_entropies.append(entropies)

        self._iter_dists(compute_stats, all_actions)

        return (jnp.concatenate(all_log_probs, axis=-1),
                jnp.concatenate(all_entropies, axis=-1))

    def probs(self):
        all_probs = []

        def compute_probs(logits):
            log_probs = logits - jax.nn.logsumexp(
                logits, axis=-1, keepdims=True)
            probs = jnp.exp(log_probs)
            all_probs.append(probs)

        self._iter_dists(compute_probs)

        return all_probs

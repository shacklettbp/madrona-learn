import jax
import jax.nn
from jax import lax, random, numpy as jnp
import flax

from dataclasses import dataclass
from typing import List

class DiscreteActionDistributions(flax.struct.PyTreeNode):
    actions_num_buckets : List[int] = flax.struct.field(pytree_node=False)
    all_logits: jax.Array

    def _iter_logits(self):
        cur_bucket_offset = 0
        for num_buckets in self.actions_num_buckets:
            logits_slice = self.all_logits[
                ..., cur_bucket_offset:cur_bucket_offset + num_buckets]

            yield logits_slice

            cur_bucket_offset += num_buckets

    def sample(self, prng_key):
        all_actions = []
        all_log_probs = []

        sample_keys = random.split(prng_key, len(self.actions_num_buckets))

        for sample_key, logits in zip(sample_keys, self._iter_logits()):
            actions = random.categorical(sample_key, logits)
            actions = jnp.expand_dims(actions, axis=-1)

            action_logits = jnp.take_along_axis(logits, actions, axis=-1)
            action_log_probs = action_logits - jax.nn.logsumexp(
                logits, axis=-1, keepdims=True)

            all_actions.append(actions)
            all_log_probs.append(action_log_probs)

        return (jnp.concatenate(all_actions, axis=-1),
                jnp.concatenate(all_log_probs, axis=-1))

    def best(self):
        all_actions = []

        for logits in self._iter_logits():
            all_actions.append(jnp.argmax(logits, keepdims=True, axis=-1))

        return jnp.concatenate(all_actions, axis=-1)

    def action_stats(self, all_actions):
        all_log_probs = []
        all_entropies = []

        for i, logits in enumerate(self._iter_logits()):
            actions = jnp.expand_dims(all_actions[..., i], axis=-1)
            log_probs = \
                logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
            p_logp = jnp.exp(log_probs) * log_probs
            entropies = -p_logp.sum(axis=-1, keepdims=True)

            action_log_probs = jnp.take_along_axis(log_probs, actions, axis=-1)

            all_log_probs.append(action_log_probs)
            all_entropies.append(entropies)

        return (jnp.concatenate(all_log_probs, axis=-1),
                jnp.concatenate(all_entropies, axis=-1))

    def probs(self):
        all_probs = []

        for logits in self._iter_logits():
            log_probs = logits - jax.nn.logsumexp(
                logits, axis=-1, keepdims=True)
            probs = jnp.exp(log_probs)
            all_probs.append(probs)

        return all_probs

    def logits(self):
        all_logits = []

        for logits in self._iter_logits():
            all_logits.append(logits)

        return all_logits

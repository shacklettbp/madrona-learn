import jax
from jax import lax, random, numpy as jnp

from dataclasses import dataclass

from typing import List

class DiscreteActionDistributions:
    def __init__(self, actions_num_buckets, logits = None):
        self.actions_num_buckets = actions_num_buckets

        self.dists = []
        cur_bucket_offset = 0

        for num_buckets in self.actions_num_buckets:
            self.dists.append(Categorical(logits = logits[
                :, cur_bucket_offset:cur_bucket_offset + num_buckets],
                validate_args=False))
            cur_bucket_offset += num_buckets

    def best(self, out):
        actions = [dist.probs.argmax(dim=-1) for dist in self.dists]
        torch.stack(actions, dim=1, out=out)

    def sample(self, actions_out, log_probs_out):
        actions = [dist.sample() for dist in self.dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(self.dists, actions)]

        torch.stack(actions, dim=1, out=actions_out)
        torch.stack(log_probs, dim=1, out=log_probs_out)

    def action_stats(self, actions):
        log_probs = []
        entropies = []
        for i, dist in enumerate(self.dists):
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())

        return torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)

    def probs(self):
        return [dist.probs for dist in self.dists]

@dataclass
class DiscreteActionDistributions:
    actions_num_buckets : List[int]

    def __init__(self, actions_num_buckets, logits = None):
        self.actions_num_buckets = actions_num_buckets

        self.dists = []
        cur_bucket_offset = 0

        for num_buckets in self.actions_num_buckets:
            self.dists.append(Categorical(logits = logits[
                :, cur_bucket_offset:cur_bucket_offset + num_buckets],
                validate_args=False))
            cur_bucket_offset += num_buckets

    def _iter_dists(self, logits, cb):
        cur_bucket_offset = 0
        for num_buckets in self.actions_num_buckets:
            cb(logits[:, cur_bucket_offset:cur_bucket_offset + num_buckets])

    def sample(self, prng_key, all_logits):
        actions = []
        log_probs = []


        def sample_action(logits):
            prng_key, rnd = random.split(prng_key)

            action = random.categorical(prng_key, logits)

            actions.append(action)
            log_probs.append(

        self.iter_dists(self, all_logits, sample_action)

        actions = jnp.stack(actions, axis=1)
        log_probs = jnp.stack(log_probs, axis=1)

        return actions, log_probs

    def best(self, all_logits):
        actions = []

        def best_action(logits):
            actions.append(jnp.argmax(logits))

        return jnp.stack(actions, axis=1)

    def sample(self, actions_out, log_probs_out):
        actions = [dist.sample() for dist in self.dists]
        log_probs = [dist.log_prob(action) for dist, action in zip(self.dists, actions)]

        torch.stack(actions, dim=1, out=actions_out)
        torch.stack(log_probs, dim=1, out=log_probs_out)

    def action_stats(self, actions):
        log_probs = []
        entropies = []
        for i, dist in enumerate(self.dists):
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())

        return torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)

    def probs(self):
        return [dist.probs for dist in self.dists]

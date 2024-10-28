import jax
import jax.nn
from jax import lax, random, numpy as jnp
import flax

from dataclasses import dataclass
from typing import List

from .utils import symlog, symexp

class DiscreteActionDistributions(flax.struct.PyTreeNode):
    actions_num_buckets : List[int] = flax.struct.field(pytree_node=False)
    all_logits: jax.Array

    def _iter_logits(self):
        cur_bucket_offset = 0
        for num_buckets in self.actions_num_buckets:
            logits_slice = self.all_logits[
                ..., cur_bucket_offset:cur_bucket_offset + num_buckets]

            yield logits_slice.astype(jnp.float32)

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
            # This is equivalent to jax.nn.log_softmax. This formulation is
            # slightly more efficient for action sampling, so keep it the 
            # same here. Worth noting that log_softmax doesn't have a custom
            # jvp in jax (unlike jax.nn.softmax)
            log_probs = \
                logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
            # Using jax.nn.softmax here rather than jnp.exp(log_probs).
            # jax has a custom jvp for jax.nn.softmax for one.
            p_logp = jax.nn.softmax(logits) * log_probs
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

# From DreamerV3 paper.
# Modified from https://github.com/danijar/dreamerv3
# MIT License:
# Copyright (c) 2023 Danijar Hafner
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class SymExpTwoHotDistribution(flax.struct.PyTreeNode):
    logits: jax.Array

    @staticmethod
    def create(logits):
        return SymExpTwoHotDistribution(
            logits=logits.astype(jnp.float32),
        )

    def _compute_bins(self):
        num_bins = self.logits.shape[-1]
        assert num_bins % 2 == 1 and num_bins > 1
        
        # Default bin spacing is e^-20 to e^20 in dreamerv3
        # Reduce the range as we're trying to use fewer bins / smaller models
        half = jnp.linspace(
            -14, 0, num_bins // 2 + 1, dtype=jnp.float32)
        half = symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]], axis=0)

        assert bins.shape[-1] == num_bins

        return bins

    def mean(self):
        bins = self._compute_bins()
        num_bins = bins.shape[-1]

        midpoint = (num_bins - 1) // 2

        probs = jax.nn.softmax(self.logits)

        # JAX summations are not very accurate. Use symmetric sum for top
        # and bottom of distribution to ensure they sum to 0 at initialization
        # See DreamerV3 paper

        p1 = probs[..., :midpoint]
        p2 = probs[..., midpoint:midpoint + 1]
        p3 = probs[..., midpoint + 1:]

        b1 = bins[..., :midpoint]
        b2 = bins[..., midpoint:midpoint + 1]
        b3 = bins[..., midpoint + 1:]

        weighted_avg = (
            (p2 * b2).sum(axis=-1, keepdims=True) +
            ((p1 * b1)[..., ::-1] +
             (p3 * b3)).sum(axis=-1, keepdims=True)
        )

        return symexp(weighted_avg)

    def two_hot_cross_entropy_loss(self, targets):
        assert targets.dtype == jnp.float32

        bins = self._compute_bins()
        num_bins = bins.shape[-1]

        targets = symlog(targets)
        lower_bin_idx = (
            (bins <= targets).astype(jnp.int32).sum(axis=-1) - 1
        )
        upper_bin_idx = num_bins - (
            bins > targets).astype(jnp.int32).sum(axis=-1)

        lower_bin_idx = jnp.clip(lower_bin_idx, 0, num_bins - 1)
        upper_bin_idx = jnp.clip(upper_bin_idx, 0, num_bins - 1)

        is_same_bin = (lower_bin_idx == upper_bin_idx)

        dist_to_lower = jnp.where(is_same_bin[..., None],
            1, jnp.abs(bins[lower_bin_idx, None] - targets))
        dist_to_upper = jnp.where(is_same_bin[..., None],
            1, jnp.abs(bins[upper_bin_idx, None] - targets))

        total_dist = dist_to_lower + dist_to_upper
        lower_bin_weight = dist_to_lower / total_dist
        upper_bin_weight = dist_to_upper / total_dist

        lower_bin_one_hot = jax.nn.one_hot(lower_bin_idx, num_bins)
        upper_bin_one_hot = jax.nn.one_hot(upper_bin_idx, num_bins)

        targets_two_hot = (
            lower_bin_one_hot * lower_bin_weight +
            upper_bin_one_hot * upper_bin_weight
        )

        log_probs = self.logits - jax.nn.logsumexp(
            self.logits, axis=-1, keepdims=True)

        return (targets_two_hot * log_probs).sum(-1, keepdims=True)



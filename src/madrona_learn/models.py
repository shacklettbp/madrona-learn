import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from typing import List, Callable

from .action import DiscreteActionDistributions
from .moving_avg import EMANormalizer

class MLP(nn.Module):
    num_channels: int
    num_layers: int
    dtype: jnp.dtype
    weight_initializer: Callable = jax.nn.initializers.he_normal()

    @nn.compact
    def __call__(self, inputs, train):
        x = inputs

        for i in range(self.num_layers):
            x = nn.Dense(
                    self.num_channels,
                    use_bias=True,
                    kernel_init=self.weight_initializer,
                    bias_init=jax.nn.initializers.constant(0),
                    dtype=self.dtype,
                )(x)
            with jax.numpy_dtype_promotion('standard'):
                x = nn.LayerNorm(dtype=self.dtype)(x)
            x = nn.relu(x)

        return x


class DenseLayerDiscreteActor(nn.Module):
    actions_num_buckets: List[int]
    dtype: jnp.dtype
    weight_initializer: Callable = jax.nn.initializers.orthogonal(scale=0.01)

    def setup(self):
        total_action_dim = sum(self.actions_num_buckets)
        self.impl = nn.Dense(
                total_action_dim,
                use_bias=True,
                kernel_init=self.weight_initializer,
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )

    def __call__(self, features, train=False):
        logits = self.impl(features)
        return DiscreteActionDistributions(self.actions_num_buckets, logits)


class DenseLayerCritic(nn.Module):
    dtype: jnp.dtype
    weight_initializer: Callable = jax.nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, features, train=False):
        return nn.Dense(
                1,
                use_bias=True,
                kernel_init=self.weight_initializer,
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )(features)


class EMANormalizeTree(nn.Module):
    decay: jnp.float32

    @nn.compact
    def __call__(self, tree, train):
        orig_shapes = jax.tree_map(jnp.shape, tree)
        flattened, treedef = jax.tree_util.tree_flatten(tree)

        combined = jnp.concatenate(flattened, axis=-1)

        normalized_combined = EMANormalizer(self.decay)(
            'normalize', train, combined)

        normalized = []
        cur_offset = 0
        for x in flattened:
            num_features = x.shape[-1]
            normalized.append(normalized_combined[
                ..., cur_offset:cur_offset + num_features])
            cur_offset += num_features

        normalized_tree = jax.tree_util.tree_unflatten(
                treedef, normalized)

        return jax.tree_map(
            lambda x, y: x.reshape(y), normalized_tree, orig_shapes)


class EgocentricSelfAttentionNet(nn.Module):
    num_embed_channels: int
    num_heads: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, input_tree, train):
        inputs, treedef = jax.tree_util.tree_flatten_with_path(input_tree)

        embedded = []
        for keypath, x in inputs:
            embedding = nn.Dense(
                    self.num_embed_channels,
                    use_bias=True,
                    kernel_init=jax.nn.initializers.orthogonal(),
                    bias_init=jax.nn.initializers.constant(0),
                    dtype=self.dtype,
                )(x)

            embedded.append(embedding)

        embedded = jnp.stack(embedded, axis=1)

        attended = nn.SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.num_embed_channels,
                out_features=self.num_embed_channels,
                dtype=self.dtype,
            )(embedded)

        attended = nn.Dense(
                self.num_embed_channels,
                dtype=self.dtype,
            )(attended)

        out = attended + embedded
        out = out.mean(axis=1)

        return out

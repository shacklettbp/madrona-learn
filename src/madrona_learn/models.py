import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

from typing import List, Callable
from functools import partial

from .action import DiscreteActionDistributions
from .moving_avg import EMANormalizer

from .pallas import monkeypatch as _pl_patch
from .pallas import layer_norm as pl_layer_norm
from .pallas import attention as pl_attention

class PallasLayerNorm(nn.Module):
    dtype: jnp.dtype
    use_ref: bool = False

    @nn.compact
    def __call__(self, x):
        orig_shape = x.shape
        dim = orig_shape[-1]

        # Pallas layernorm wants (batch, seq, features) but
        # seq is just treated as another batch dim anyway
        x = x.reshape(-1, 1, dim)

        scale = self.param('scale',
            jax.nn.initializers.constant(1), (dim,), jnp.float32)

        bias = self.param('bias',
            jax.nn.initializers.constant(0), (dim,), jnp.float32)

        normalized = pl_layer_norm.layer_norm(x, scale, bias)

        return normalized.reshape(orig_shape)


class LayerNorm(nn.Module):
    dtype: jnp.dtype
    use_ref: bool = True # Current pallas layernorm is not faster

    @nn.compact
    def __call__(self, x):
        if self.use_ref:
            with jax.numpy_dtype_promotion('standard'):
                return nn.LayerNorm(name='impl', dtype=self.dtype)(x)
        else:
            return PallasLayerNorm(name='impl', dtype=self.dtype)(x)


class SelfAttention(nn.Module):
    num_heads: int
    qkv_features: int
    out_features: int
    dtype: jnp.dtype
    use_ref: bool = True

    @nn.compact
    def __call__(self, x):
        if self.use_ref or self.dtype != jnp.float16:
            attention_fn = nn.attention.dot_product_attention
        else:
            def attention_fn(q, k, v, mask, dropout_rng, dropout_rate,
                             broadcast_dropout, deterministic,
                             dtype, precision):
                seq_len = q.shape[1]
                pad_amount = max(0, 16 - seq_len)

                q = jnp.pad(q, ((0, 0), (0, pad_amount), (0, 0), (0, 0)),
                            constant_values = 0)

                k = jnp.pad(k, ((0, 0), (0, pad_amount), (0, 0), (0, 0)),
                            constant_values = 0)

                v = jnp.pad(v, ((0, 0), (0, pad_amount), (0, 0), (0, 0)),
                            constant_values = 0)

                with jax.numpy_dtype_promotion('standard'):
                    out = pl_attention.mha(q, k, v, segment_ids=None)

                return out[:, 0:seq_len, :, :]

        return nn.SelfAttention(
            num_heads = self.num_heads,
            qkv_features = self.qkv_features,
            out_features = self.out_features,
            dtype = self.dtype,
            attention_fn = attention_fn
        )(x)

class MLP(nn.Module):
    num_channels: int
    num_layers: int
    dtype: jnp.dtype
    weight_init: Callable = jax.nn.initializers.he_normal()

    @nn.compact
    def __call__(self, inputs, train):
        x = inputs

        for i in range(self.num_layers):
            x = nn.Dense(
                    self.num_channels,
                    use_bias=True,
                    kernel_init=self.weight_init,
                    bias_init=jax.nn.initializers.constant(0),
                    dtype=self.dtype,
                )(x)
            x = LayerNorm(dtype=self.dtype)(x)
            x = nn.relu(x)

        return x


class DenseLayerDiscreteActor(nn.Module):
    actions_num_buckets: List[int]
    dtype: jnp.dtype
    weight_init: Callable = jax.nn.initializers.orthogonal(scale=0.01)

    def setup(self):
        total_action_dim = sum(self.actions_num_buckets)
        self.impl = nn.Dense(
                total_action_dim,
                use_bias=True,
                kernel_init=self.weight_init,
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )

    def __call__(self, features, train=False):
        logits = self.impl(features)
        return DiscreteActionDistributions(self.actions_num_buckets, logits)


class DenseLayerCritic(nn.Module):
    dtype: jnp.dtype
    weight_init: Callable = jax.nn.initializers.orthogonal()

    @nn.compact
    def __call__(self, features, train=False):
        return nn.Dense(
                1,
                use_bias=True,
                kernel_init=self.weight_init,
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )(features)


class EMANormalizeTree(nn.Module):
    decay: jnp.float32
    out_dtype: jnp.dtype

    @nn.compact
    def __call__(self, tree, train):
        return jax.tree_map(
            lambda x: EMANormalizer(self.decay, self.out_dtype)(
                'normalize', train, x), tree)


# Based on the Emergent Tool Use policy paper
class EntitySelfAttentionNet(nn.Module):
    num_embed_channels: int
    num_heads: int
    dtype: jnp.dtype
    dense_init: Callable = jax.nn.initializers.orthogonal()
    # To follow the paper, this should be true. If the observations are
    # already egocentric, this seems redundant.
    embed_concat_self: bool = False

    @nn.compact
    def __call__(self, x_tree, train):
        def make_embed():
            return nn.Dense(
                self.num_embed_channels,
                use_bias = True,
                kernel_init = self.dense_init,
                bias_init = jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )

        x_tree, x_self = x_tree.pop('self')

        x_self = jnp.expand_dims(x_self, axis=-2)

        embed_self = make_embed()(x_self)

        x_flat, treedef = jax.tree_util.tree_flatten_with_path(x_tree)

        embedded_entities = [embed_self]
        for keypath, x_entities in x_flat:
            if self.embed_concat_self:
                x_entities = jnp.concatenate([x_entities, x_self], axis=-1)

            embedding = make_embed()(x_entities)

            embedded_entities.append(embedding)

        embedded_entities = jnp.concatenate(embedded_entities, axis=-2)

        attended_entities = SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.num_embed_channels,
                out_features=self.num_embed_channels,
                dtype=self.dtype,
            )(embedded_entities)

        attended_entities = nn.Dense(
                self.num_embed_channels,
                dtype=self.dtype,
            )(attended_entities)

        attended_entities = attended_entities + embedded_entities
        attended_entities = LayerNorm(dtype=self.dtype)(attended_entities)
        attended_entities = attended_entities.mean(axis=-2)

        out = jnp.concatenate(
            [jnp.squeeze(embed_self, axis=-2), attended_entities], axis=-1)
        out = LayerNorm(dtype=self.dtype)(out)

        out = nn.Dense(
                self.num_embed_channels * 2,
                use_bias = True,
                kernel_init = self.dense_init,
                bias_init = jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )(out)

        out = LayerNorm(dtype=self.dtype)(out)

        return out

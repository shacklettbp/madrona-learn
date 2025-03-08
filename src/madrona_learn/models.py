import jax
import numpy as np
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
from flax.core import FrozenDict

from typing import List, Callable, Any

from .cfg import DiscreteActionsConfig
from .dists import (
    DiscreteActionDistributions,
    SymExpTwoHotDistribution,
)

from .utils import symexp

#from .pallas import monkeypatch as _pl_patch
#from .pallas import layer_norm as pl_layer_norm
#from .pallas import attention as pl_attention

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
    weight_init: Callable = jax.nn.initializers.orthogonal(scale=np.sqrt(2))

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
    cfg: DiscreteActionsConfig
    dtype: jnp.dtype
    weight_init: Callable = jax.nn.initializers.orthogonal(scale=0.01)

    def setup(self):
        total_action_dim = sum(self.cfg.actions_num_buckets)
        self.impl = nn.Dense(
                total_action_dim,
                use_bias=True,
                kernel_init=self.weight_init,
             bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )

    def __call__(self, features, train=False):
        logits = self.impl(features)
        return DiscreteActionDistributions(self.cfg.actions_num_buckets, logits)


class DenseLayerCritic(nn.Module):
    dtype: jnp.dtype
    weight_init: Callable = jax.nn.initializers.orthogonal(scale=1.0)

    @nn.compact
    def __call__(self, features, train=False):
        return nn.Dense(
                1,
                use_bias=True,
                kernel_init=self.weight_init,
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )(features).astype(jnp.float32)


class DreamerV3Critic(nn.Module):
    dtype: jnp.dtype
    weight_init: Callable = jax.nn.initializers.constant(0)

    # Note that default num_bins in dreamerv3 codebase is 255
    num_bins: int = 63

    @nn.compact
    def __call__(self, features, train=False):
        logits = nn.Dense(
                self.num_bins,
                use_bias=True,
                kernel_init=self.weight_init,
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )(features)

        return SymExpTwoHotDistribution.create(logits)

# Based on M3 / Stop Regressing papers
class HLGaussDist(flax.struct.PyTreeNode):
    logits: jax.Array

    smoothness: float = flax.struct.field(pytree_node=False)

    centers: jax.Array = flax.struct.field(pytree_node=False)
    bounds: jax.Array = flax.struct.field(pytree_node=False)

    def mean(self):
        def categorical_pred(logits, centers):
            midpoint = (centers.size - 1) // 2

            probs = jax.nn.softmax(logits)
            
            # JAX summations are not very accurate. Use symmetric sum for top
            # and bottom of distribution to ensure they sum to 0 at initialization
            # See DreamerV3 paper
            p1 = probs[..., :midpoint]
            p2 = probs[..., midpoint:midpoint + 1]
            p3 = probs[..., midpoint + 1:]

            c1 = centers[..., :midpoint]
            c2 = centers[..., midpoint:midpoint + 1]
            c3 = centers[..., midpoint + 1:]

            weighted_avg = (
                (p2 * c2).sum(axis=-1, keepdims=True) +
                ((p1 * c1)[..., ::-1] +
                 (p3 * c3)).sum(axis=-1, keepdims=True)
            )

            return weighted_avg

        return categorical_pred(self.logits, self.centers)

    def loss(self, targets):
        targets = jnp.clip(
            targets, self.centers[0], self.centers[-1])

        erf = jax.scipy.special.erf

        def compute_sigma(bounds, tgts):
            lower_bin_idx = (
                (bounds <= tgts).astype(jnp.int32).sum(axis=-1) - 1
            )

            upper_bin_idx = lower_bin_idx + 1

            lower_bin_idx = jnp.clip(lower_bin_idx, 0, bounds.size - 2)
            upper_bin_idx = jnp.clip(upper_bin_idx, 1, bounds.size - 1)
            
            width = bounds[upper_bin_idx] - bounds[lower_bin_idx]

            return self.smoothness * width[..., None]


        def hl_gauss(logits, bounds, tgts):
            sigmas = compute_sigma(bounds, tgts)

            a = bounds[0]
            b = bounds[1]

            cdfs = erf((bounds - tgts) / (jnp.sqrt(2) * sigmas))

            z = cdfs[..., -1] - cdfs[..., 0]
            z = z[..., None]

            c = 1 / z * (cdfs[..., 1:] - cdfs[..., :-1])

            log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)

            return -(c * log_probs).sum(-1, keepdims=True)

        return hl_gauss(self.logits, self.bounds, targets)


class HLGaussCritic(nn.Module):
    dtype: jnp.dtype

    centers: jax.Array
    bounds: jax.Array

    smoothness: float = 0.75

    weight_init: Callable = jax.nn.initializers.constant(0)

    @staticmethod
    def create(
        dtype: jnp.dtype,
        num_bins: int = 127,
        min_bound = -100,
        max_bound = 100,
        smoothness: float = 0.75,
    ):
        def gen_bins():
            half = np.linspace(min_bound, 0, num_bins // 2 + 1)

            bins = np.concatenate([half, -half[:-1][::-1]], axis=0)

            width = bins[1] - bins[0]

            bounds = bins - 0.5 * width
            bounds = np.concatenate([bounds, np.asarray([bounds[-1] + width])], axis=0)

            return jnp.asarray(bins, dtype=jnp.float32), jnp.asarray(bounds, dtype=jnp.float32)

        bins, bounds = gen_bins()

        return HLGaussCritic(
            dtype=dtype,
            centers=bins,
            bounds=bounds,
            smoothness=smoothness,
        )

    @nn.compact
    def __call__(self, features, train=False):
        logits = nn.Dense(
                self.centers.shape[0],
                use_bias=True,
                kernel_init=self.weight_init,
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )(features)

        return HLGaussDist(
            centers=self.centers,
            bounds=self.bounds,
            logits=logits.astype(jnp.float32),
            smoothness=self.smoothness,
        )

class HLGaussTwoPartDist(flax.struct.PyTreeNode):
    small_dist: HLGaussDist
    large_dist: HLGaussDist

    def mean(self):
        return self.small_dist.mean() + self.large_dist.mean()

    def loss(self, targets):
        small_large_boundary = 2
        small_tgt = targets % (jnp.where(targets >= 0, 1, -1) * 2)
        large_tgt = targets - small_tgt

        return self.small_dist.loss(small_tgt) + self.large_dist.loss(large_tgt)


class HLGaussTwoPartCritic(nn.Module):
    dtype: jnp.dtype

    small_centers: jax.Array
    small_bounds: jax.Array

    large_centers: jax.Array
    large_bounds: jax.Array

    smoothness: float = 0.75

    weight_init: Callable = jax.nn.initializers.constant(0)

    @staticmethod
    def create(
        dtype: jnp.dtype,
        num_small_bins: int = 127,
        num_large_bins: int = 127,
        smoothness: float = 0.75,
    ):
        def gen_floating_point_bins(
            num_mantissa_bits: int,
            num_exp_bits: int,
            bias: int,
            denorm: bool,
        ):
            half = []
            widths = []
            for exp in range(2 ** num_exp_bits):
                if denorm and exp == 0:
                    scale = (2 ** (1 - bias))
                else:
                    scale = (2 ** (exp - bias))

                width = scale / (2 ** num_mantissa_bits)
                for mantissa in range(2 ** num_mantissa_bits):
                    frac = mantissa / (2 ** num_mantissa_bits)
                    if denorm and exp == 0:
                        half.append(frac * scale)
                    elif exp == 0 and mantissa == 0:
                        half.append(0)
                    else:
                        half.append((1 + frac) * scale)
                    widths.append(width)

            half = np.asarray(half, dtype=np.float32)
            bins = np.concatenate([-half[:0:-1], half])

            widths = np.asarray(widths, dtype=np.float32)
            widths = np.concatenate([widths[:0:-1], widths])

            bounds = bins - 0.5 * widths
            bounds = np.concatenate([bounds, np.asarray([bounds[-1] + widths[-1]])])

            return jnp.asarray(bins, dtype=jnp.float32), jnp.asarray(bounds, dtype=jnp.float32)

        def gen_small_bins():
            num_mantissa_bits = 3
            num_exp_bits = 3
            bias = 2**3 - 1

            bins, bounds = gen_floating_point_bins(
                num_mantissa_bits, num_exp_bits, bias, True)

            assert bins.shape[0] == num_small_bins

            return bins, bounds

        def gen_large_bins():
            num_mantissa_bits = 3
            num_exp_bits = 3
            bias = -3

            bins, bounds = gen_floating_point_bins(
                num_mantissa_bits, num_exp_bits, bias, True)

            assert bins.shape[0] == num_large_bins

            return bins, bounds

        small_bins, small_bounds = gen_small_bins()
        large_bins, large_bounds = gen_large_bins()

        return HLGaussTwoPartCritic(
            dtype=dtype,
            small_centers=small_bins,
            small_bounds=small_bounds,
            large_centers=large_bins,
            large_bounds=large_bounds,
            smoothness=smoothness,
        )

    @nn.compact
    def __call__(self, features, train=False):
        small_logits = nn.Dense(
                self.small_centers.shape[0],
                use_bias=True,
                kernel_init=self.weight_init,
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )(features)

        large_logits = nn.Dense(
                self.large_centers.shape[0],
                use_bias=True,
                kernel_init=self.weight_init,
                bias_init=jax.nn.initializers.constant(0),
                dtype=self.dtype,
            )(features)

        return HLGaussTwoPartDist(
            small_dist = HLGaussDist(
                centers=self.small_centers,
                bounds=self.small_bounds,
                logits=small_logits.astype(jnp.float32),
                smoothness=self.smoothness,
            ),
            large_dist = HLGaussDist(
                centers=self.large_centers,
                bounds=self.large_bounds,
                logits=large_logits.astype(jnp.float32),
                smoothness=self.smoothness,
            ),
        )


# Based on the Emergent Tool Use policy paper
class EntitySelfAttentionNet(nn.Module):
    num_embed_channels: int
    num_out_channels: int
    num_heads: int
    dtype: jnp.dtype
    dense_init: Callable = jax.nn.initializers.orthogonal(scale=np.sqrt(2))
    # To follow the paper, this should be true. If the observations are
    # already egocentric, this seems redundant.
    embed_concat_self: bool = False

    @nn.compact
    def __call__(self, x_tree, train):
        def make_embed(name):
            def embed(x):
                o = nn.Dense(
                    self.num_embed_channels,
                    use_bias = True,
                    kernel_init = self.dense_init,
                    bias_init = jax.nn.initializers.constant(0),
                    dtype=self.dtype,
                    name=name,
                )(x)

                o = LayerNorm(dtype=self.dtype)(o)
                o = nn.leaky_relu(o)

                return o

            return embed

        x_tree, x_self = x_tree.pop('self')

        x_self = jnp.expand_dims(x_self, axis=-2)

        embed_self = make_embed('self_embed')(x_self)

        x_flat, treedef = jax.tree_util.tree_flatten_with_path(x_tree)

        embedded_entities = [embed_self]
        for keypath, x_entities in x_flat:
            if self.embed_concat_self:
                x_entities = jnp.concatenate([x_entities, x_self], axis=-1)

            embed_name = keypath[-1].key + '_embed'
            embedding = make_embed(embed_name)(x_entities)
            embedded_entities.append(embedding)

        embedded_entities = jnp.concatenate(embedded_entities, axis=-2)

        attended_out = SelfAttention(
                num_heads=self.num_heads,
                qkv_features=self.num_embed_channels,
                out_features=self.num_out_channels,
                dtype=self.dtype,
            )(embedded_entities)

        if self.num_embed_channels != self.num_out_channels:
            attended_out = attended_out + jnp.tile(
                embedded_entities, self.num_out_channels // self.num_embed_channels)
        else:
            attended_out = attended_out + embedded_entities

        attended_out = attended_out.mean(axis=-2)
        attended_out = LayerNorm(dtype=self.dtype)(attended_out)

        # Feedforward

        ff_out = nn.Dense(
                self.num_out_channels,
                use_bias = True,
                dtype=self.dtype,
                kernel_init = self.dense_init,
                bias_init = jax.nn.initializers.constant(0),
                name='ff_0',
            )(attended_out)

        ff_out = LayerNorm(dtype=self.dtype)(ff_out)
        ff_out = nn.leaky_relu(ff_out)
        ff_out = nn.Dense(
                self.num_out_channels,
                use_bias = True,
                dtype=self.dtype,
                kernel_init = self.dense_init,
                bias_init = jax.nn.initializers.constant(0),
                name='ff_1',
            )(ff_out)

        # Transformer block wouldn't have this activation
        ff_out = nn.leaky_relu(ff_out)
        ff_out = attended_out + ff_out
        ff_out = LayerNorm(dtype=self.dtype)(ff_out)

        return ff_out

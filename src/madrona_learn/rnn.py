import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

from typing import Tuple

__all__ = ["LSTM"]

class MultiLayerLSTMCell(nn.RNNCellBase):
    num_hidden_channels: int
    num_layers: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(
        self,
        carries: Tuple[jax.Array, jax.Array],
        inputs: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        x = inputs

        all_c = []
        all_h = []
        all_out = []

        in_c, in_h = carries

        for i in range(self.num_layers):
            (new_c, new_h), out = nn.OptimizedLSTMCell(
                    features=self.num_hidden_channels,
                    kernel_init=jax.nn.initializers.orthogonal(),
                    recurrent_kernel_init=jax.nn.initializers.orthogonal(),
                    bias_init=jax.nn.initializers.constant(0),
                    dtype=self.dtype,
                )((in_c[i], in_h[i]), x)
            x = new_h

            all_c.append(new_c)
            all_h.append(new_h)
            all_out.append(out)

        all_out = jnp.concatenate(all_out, axis=-1)

        return (all_c, all_h), all_out

class LSTM(nn.Module):
    num_hidden_channels: int
    num_layers: int
    dtype: jnp.dtype

    @nn.nowrap
    def init_recurrent_state(self, N):
        c_states = []
        h_states = []

        init_zeros = jnp.zeros((N, self.num_hidden_channels), self.dtype)

        for i in range(self.num_layers):
            c_states.append(init_zeros)
            h_states.append(init_zeros)

        return c_states, h_states

    @nn.nowrap
    def clear_recurrent_state(self, rnn_states, should_clear):
        new_c_states = []
        new_h_states = []

        c_states, h_states = rnn_states

        for i in range(self.num_layers):
            layer_masked_c = jnp.where(should_clear,
                jnp.zeros((), dtype=c_states[i].dtype), c_states[i])
            layer_masked_h = jnp.where(should_clear,
                jnp.zeros((), dtype=h_states[i].dtype), h_states[i])

            new_c_states.append(layer_masked_c)
            new_h_states.append(layer_masked_h)

        return new_c_states, new_h_states

    def setup(self):
        self.cell = MultiLayerLSTMCell(
            self.num_hidden_channels, self.num_layers, self.dtype)

    def __call__(self, cur_hiddens, in_features, train):
        new_hiddens, out = self.cell(cur_hiddens, in_features)
        return out, new_hiddens

    def sequence(self, start_hiddens, seq_ends, seq_x, train):
        def process_step(cell, carry, x, end):
            carry, y = cell(carry, x)
            carry = self.clear_recurrent_state(carry, end)

            return carry, y

        scan_txfm = nn.scan(
            process_step,
            in_axes=0,
            out_axes=0,
            unroll=1,
            variable_axes={},
            variable_broadcast='params',
            variable_carry=False,
            split_rngs={ 'params': False },
        )

        _, outputs = scan_txfm(self.cell, start_hiddens, seq_x, seq_ends)

        return outputs

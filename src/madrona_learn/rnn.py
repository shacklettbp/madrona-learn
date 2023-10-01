import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn

from typing import Tuple

__all__ = ["LSTM"]

class MaskableLSTMCell(nn.RNNCellBase):
    hidden_channels: int
    num_layers: int = 1

    @nn.compact
    def __call__(
        self,
        carries: Tuple[jax.Array, jax.Array],
        inputs: jax.Array,
        breaks: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        x = inputs

        all_c = []
        all_h = []
        all_out = []

        in_c, in_h = carries

        for i in range(self.num_layers):
            layer_masked_c = jnp.where(breaks, jnp.zeros(()), in_c[i])
            layer_masked_h = jnp.where(breaks, jnp.zeros(()), in_h[i])

            (new_c, new_h), out = nn.OptimizedLSTMCell(
                    features=hidden_channels,
                    kernel_init=jax.nn.initializers.orthogonal(),
                    recurrent_kernel_init=jax.nn.initializers.orthogonal(),
                    bias_init=jax.nn.initializers.constant(0),
                )((layer_masked_c, layer_masked_h), x)
            x = new_h

            all_c.append(new_c)
            all_h.append(new_h)
            all_out.append(out)

        all_out = jnp.concatenate(all_out, axis=-1)

        return (all_c, all_h), all_out

class LSTM(nn.Module):
    hidden_channels: int
    num_layers: int = 1

    def setup(self):
        self.cell = MaskableLSTMCell(self.hidden_channels, self.num_layers)

    def init_recurrent_state(self, N, dev, dtype):
        c_states = []
        h_states = []

        with jax.default_device(dev):
            init_zeros = jnp.zeros((N, self.hidden_channels), dtype)

        for i in range(self.num_layers):
            c_states.append(init_zeros)
            h_states.append(init_zeros)

        return (c_states, h_states)

    def __call__(self, cur_hiddens, in_features, breaks, train):
        new_hiddens, out = self.cell(cur_hiddens, in_features, breaks)
        return out, new_hiddens

    def sequence(self, start_hiddens, seq_x , seq_breaks, train):
        def process_step(cell, carry, x, breaks):
            carry, y = cell(carry, x, breaks)

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

        _, outputs = scan_txfm(self.cell, start_hiddens, seq_x, seq_breaks)

        return outputs

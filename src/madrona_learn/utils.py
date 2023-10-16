import jax
from jax import lax, random, numpy as jnp

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from os import environ as env_vars

@runtime_checkable
@dataclass
class DataclassProtocol(Protocol):
    pass


@dataclass
class TypedShape:
    shape: jnp.shape
    dtype: jnp.dtype


def init(mem_fraction):
    env_vars["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_fraction:.2f}"
    #jax.config.update("jax_numpy_rank_promotion", "raise")
    jax.config.update("jax_numpy_dtype_promotion", "strict")

def init_recurrent_states(policy, batch_size_per_policy, num_policies):
    def init_rnn_states():
        def init(arg):
            return policy.init_recurrent_state(batch_size_per_policy)

        return jax.vmap(init)(jnp.empty(num_policies))

    return jax.jit(init_rnn_states)()

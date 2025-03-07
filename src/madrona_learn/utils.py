import jax
from jax import lax, random, numpy as jnp
import numpy as np

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


def cfg_jax_mem(mem_fraction):
    env_vars["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_fraction:.2f}"
    #jax.config.update("jax_numpy_rank_promotion", "raise")
    jax.config.update("jax_numpy_dtype_promotion", "strict")


def convert_float_leaves(data, desired_dtype):
    def convert(x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.asarray(x, dtype=desired_dtype)
        else:
            return x

    return jax.tree_map(convert, data)

def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))

def symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))

def aot_compile(func, *args):
    func = jax.jit(func, donate_argnums=range(len(args)))

    lowered = func.lower(*args)

    if 'MADRONA_LEARN_DUMP_LOWERED' in env_vars:
        with open(env_vars['MADRONA_LEARN_DUMP_LOWERED'], 'w') as f:
            print(lowered.as_text(), file=f)

    compiled = lowered.compile()

    if 'MADRONA_LEARN_DUMP_IR' in env_vars:
        with open(env_vars['MADRONA_LEARN_DUMP_IR'], 'w') as f:
            print(compiled.as_text(), file=f)

    return compiled

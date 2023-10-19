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
        return policy.init_recurrent_state(batch_size_per_policy)

    init_rnn_states = jax.jit(
        jax.vmap(init_rnn_states, axis_size=num_policies))

    return init_rnn_states()

def make_pbt_reorder_funcs(dyn_assignment, num_policies):
    def group_into_policy_batches(args):
        return jax.tree_map(
            lambda x: x.reshape(num_policies, -1, *x.shape[1:]), args)

    if dyn_assignment:
        def prep_for_policy(args, sort_idxs):
            reordered = jax.tree_map(lambda x: jnp.take(x, sort_idxs, 0), args)
            return group_into_policy_batches(reordered)

        def prep_for_sim(args, sort_idxs):
            args = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), args)

            unsort_idxs = jnp.arange(sort_idxs.shape[0])[sort_idxs]
            return jax.tree_map(lambda x: jnp.take(x, unsort_idxs, 0), args)
    else:
        def prep_for_policy(args, sort_idxs):
            assert(sort_idxs == None)
            return group_into_policy_batches(args)

        def prep_for_sim(args, sort_idxs):
            assert(sort_idxs == None)
            return jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), args)

    return prep_for_policy, prep_for_sim

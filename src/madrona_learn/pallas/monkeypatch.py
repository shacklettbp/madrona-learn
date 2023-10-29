import jax_triton
import jax
from jax import numpy as jnp

from jax._src.pallas.triton import lowering
from jax._src.pallas import primitives
from jax.experimental import pallas
from .lowering_hack import (
    lower_jaxpr_to_triton_module,
    compile_jaxpr,
)

lowering.lower_jaxpr_to_triton_module = lower_jaxpr_to_triton_module 
lowering.compile_jaxpr = compile_jaxpr

def dot_outdtype(a, b, trans_a: bool = False, trans_b: bool = False,
                 allow_tf32: bool | None = None, precision=None,
                 out_dtype=None):
  if (a.ndim != 2) or (b.ndim != 2):
    raise ValueError("`a` and `b` must be 2D arrays.")
  lhs_contract_dim = 0 if trans_a else 1
  rhs_contract_dim = 0 if not trans_b else 1
  if allow_tf32 is not None:
    if precision is not None:
      raise ValueError("Only one of allow_tf32 and precision can be specified")
    precision = lax.Precision.HIGH if allow_tf32 else lax.Precision.HIGHEST
  return jax.lax.dot_general(
      a,
      b,
      dimension_numbers=(((lhs_contract_dim,), (rhs_contract_dim,)), ((), ())),
      precision=precision,
      preferred_element_type=(out_dtype or jnp.float32),
  )

primitives.dot = dot_outdtype
pallas.dot = dot_outdtype

import jax_triton
from jax._src.pallas.triton import lowering
from .lowering_hack import (
    lower_jaxpr_to_triton_module,
    compile_jaxpr,
)

lowering.lower_jaxpr_to_triton_module = lower_jaxpr_to_triton_module 
lowering.compile_jaxpr = compile_jaxpr

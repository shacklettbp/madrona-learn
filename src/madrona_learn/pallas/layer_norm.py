# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# shacklettbp: Modified to work with mixed-precision

"""Module containing fused layer norm forward and backward pass."""

import functools

from typing import Optional

import jax
from jax import lax
import jax.numpy as jnp
from jax._src.lax.control_flow.for_loop import for_loop

from jax.experimental import pallas as pl

def layer_norm_forward_kernel(
    x_ref, weight_ref, bias_ref, # Input arrays
    o_ref, mean_ref=None, rstd_ref=None, # Output arrays
    *, eps: float, block_size: int):
  n_col = x_ref.shape[0]

  def mean_body(i, acc_ref):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < n_col
    a = pl.load(x_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last")
    acc_ref[:] += a.astype(jnp.float32)
  mean = for_loop(pl.cdiv(n_col, block_size), mean_body,
                  jnp.zeros(block_size)).sum() / n_col

  mean = mean.astype(x_ref.dtype)

  def var_body(i, acc_ref):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < n_col
    a = pl.load(x_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last")
    a = jnp.where(mask, a - mean, 0.)
    acc_ref[:] += (a * a).astype(jnp.float32)
  var = for_loop(pl.cdiv(n_col, block_size), var_body,
                 jnp.zeros(block_size)).sum() / n_col
  rstd = lax.rsqrt(var + eps)
  rstd = rstd.astype(x_ref.dtype)

  if mean_ref is not None:
    mean_ref[...] = mean.astype(mean_ref.dtype)
  if rstd_ref is not None:
    rstd_ref[...] = rstd.astype(rstd_ref.dtype)

  def body(i, _):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < n_col
    weight = pl.load(weight_ref, (col_idx,), mask=mask).astype(x_ref.dtype)
    bias = pl.load(bias_ref, (col_idx,), mask=mask).astype(x_ref.dtype)
    x = pl.load(x_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_first")
    out = (x - mean.astype(x_ref.dtype)) * rstd.astype(x_ref.dtype) * weight + bias
    pl.store(o_ref, (col_idx,), out.astype(o_ref.dtype), mask=mask)
  for_loop(pl.cdiv(n_col, block_size), body, ())


def layer_norm_forward(
    x, weight, bias,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = 3,
    eps: float = 1e-5,
    backward_pass_impl: str = 'triton',
    interpret: bool = False):
  del num_stages
  del backward_pass_impl
  n = x.shape[-1]
  # Triton heuristics
  # Less than 64KB per feature: enqueue fused kernel
  max_fused_size = 65536 // x.dtype.itemsize
  block_size = min(max_fused_size, pl.next_power_of_2(n))
  block_size = min(max(block_size, 128), 4096)
  num_warps = min(max(block_size // 256, 1), 8)

  kernel = functools.partial(layer_norm_forward_kernel, eps=eps,
                             block_size=block_size)
  out_shape = [
          jax.ShapeDtypeStruct(shape=(n,), dtype=x.dtype),
          jax.ShapeDtypeStruct(shape=(), dtype=x.dtype),
          jax.ShapeDtypeStruct(shape=(), dtype=x.dtype)
  ]
  method = pl.pallas_call(kernel, num_warps=num_warps,
                          grid=(), out_shape=out_shape, debug=False,
                          interpret=interpret, name='ln_forward')

  method = jax.vmap(jax.vmap(method, in_axes=(0, None, None)), in_axes=(0, None, None))
  out, mean, rstd = method(x, weight, bias)
  return out, (x, weight, bias, mean, rstd)


def layer_norm_backward_kernel_dx(
    # Inputs
    x_ref, weight_ref, bias_ref, do_ref,
    mean_ref, rstd_ref,
    # Outputs
    dx_ref,
    *, eps: float, block_size: int):
  n_col = x_ref.shape[0]

  def mean_body(i, acc_ref):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < n_col
    a = pl.load(x_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last")
    dout = pl.load(do_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last").astype(a.dtype)
    weight = pl.load(weight_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last").astype(a.dtype)
    a_hat = (a - mean_ref[...]) * rstd_ref[...]
    wdout = weight * dout
    mean1_acc_ref, mean2_acc_ref = acc_ref
    mean1_acc_ref[:] += (a_hat * wdout).astype(jnp.float32)
    mean2_acc_ref[:] += wdout.astype(jnp.float32)
  mean = for_loop(pl.cdiv(n_col, block_size), mean_body,
                  (jnp.zeros(block_size), jnp.zeros(block_size)))
  mean1, mean2 = mean
  mean1 = mean1.sum() / n_col
  mean2 = mean2.sum() / n_col

  mean1 = mean1.astype(x_ref.dtype)
  mean2 = mean2.astype(x_ref.dtype)

  def dx_body(i, acc_ref):
    col_idx = i * block_size + jnp.arange(block_size)
    mask = col_idx < n_col
    a = pl.load(x_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last")
    dout = pl.load(do_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last").astype(a.dtype)
    weight = pl.load(weight_ref, (col_idx,), mask=mask, other=0.,
                eviction_policy="evict_last").astype(a.dtype)
    a_hat = (a - mean_ref[...]) * rstd_ref[...]
    wdout = weight * dout
    da = (wdout - (a_hat * mean1 + mean2)) * rstd_ref[...]
    pl.store(dx_ref, (col_idx,), da.astype(dx_ref.dtype), mask=mask)
  for_loop(pl.cdiv(n_col, block_size), dx_body, ())


def layer_norm_backward_kernel_dw_db(
    # Inputs
    x_ref, weight_ref, bias_ref, do_ref,
    mean_ref, rstd_ref,
    # Outputs
    dw_ref, db_ref,
    *, eps: float, block_m: int, block_n: int):
  m, n_col = x_ref.shape
  j = pl.program_id(0)
  col_idx = j * block_n + jnp.arange(block_n)
  col_mask = col_idx < n_col

  def body(i, acc_ref):
    row_idx = i * block_m + jnp.arange(block_m)
    row_mask = row_idx < m
    mask = row_mask[:, None] & col_mask[None, :]
    a = pl.load(
        x_ref, (row_idx[:, None], col_idx[None]), mask=mask, other=0.0
    )
    dout = pl.load(
        do_ref, (row_idx[:, None], col_idx[None]), mask=mask, other=0.0
    ).astype(a.dtype)
    mean = pl.load(mean_ref, (row_idx,), mask=row_mask, other=0.)
    rstd = pl.load(rstd_ref, (row_idx,), mask=row_mask, other=0.)
    a_hat = (a - mean[:, None]) * rstd[:, None]
    dw_acc_ref, db_acc_ref = acc_ref
    dw_acc_ref[:] += (dout * a_hat).astype(jnp.float32).sum(axis=0)
    db_acc_ref[:] += dout.astype(jnp.float32).sum(axis=0)
  dw_acc, db_acc = for_loop(pl.cdiv(m, block_m), body, (jnp.zeros(block_n), jnp.zeros(block_n)))
  pl.store(dw_ref, (col_idx,), dw_acc.astype(dw_ref.dtype), mask=col_mask)
  pl.store(db_ref, (col_idx,), db_acc.astype(db_ref.dtype), mask=col_mask)


def layer_norm_backward(
    num_warps: Optional[int],
    num_stages: Optional[int],
    eps: float,
    backward_pass_impl: str,
    interpret: bool,
    res, do):
  del num_stages
  x, weight, bias, mean, rstd = res
  if backward_pass_impl == 'xla':
    return jax.vjp(layer_norm_reference, x, weight, bias)[1](do)

  *shape_prefix, n = x.shape
  reshaped_x = x.reshape((-1, n))
  reshaped_mean = mean.reshape((-1,))
  reshaped_rstd = rstd.reshape((-1,))
  reshaped_do = do.reshape((-1, n))
  # Triton heuristics
  # Less than 64KB per feature: enqueue fused kernel
  max_fused_size = 65536 // x.dtype.itemsize
  block_size = min(max_fused_size, pl.next_power_of_2(n))
  block_size = min(max(block_size, 128), 4096)
  num_warps = min(max(block_size // 256, 1), 8)

  # layer_norm_backward_kernel_dx parallel over batch dims
  kernel = functools.partial(layer_norm_backward_kernel_dx, eps=eps,
                             block_size=block_size)
  out_shape_dx = jax.ShapeDtypeStruct(shape=(n,), dtype=x.dtype)
  method = pl.pallas_call(kernel, num_warps=num_warps,
                          grid=(), out_shape=out_shape_dx, debug=False,
                          interpret=interpret, name='ln_backward_dx')

  method = jax.vmap(method, in_axes=(0, None, None, 0, 0, 0))

  dx = method(reshaped_x, weight, bias, reshaped_do, reshaped_mean, reshaped_rstd)
  dx = dx.reshape((*shape_prefix, n))

  # layer_norm_backward_kernel_dw_db reduce over batch dims
  # Triton heuristics
  if n > 10240:
    block_n = 128
    block_m = 32
    num_warps = 4
  else:
    # maximize occupancy for small N
    block_n = 16
    block_m = 16
    num_warps = 8
  kernel = functools.partial(layer_norm_backward_kernel_dw_db, eps=eps,
                             block_m=block_m, block_n=block_n)
  out_shape_dwbias = [
          jax.ShapeDtypeStruct(shape=weight.shape, dtype=weight.dtype),
          jax.ShapeDtypeStruct(shape=bias.shape, dtype=bias.dtype)
  ]
  grid_ = (pl.cdiv(reshaped_x.shape[1], block_n),)
  method = pl.pallas_call(kernel, num_warps=num_warps,
                          grid=grid_, out_shape=out_shape_dwbias, debug=False,
                          interpret=interpret, name='ln_backward_dw_db')
  dw, dbias = method(reshaped_x, weight, bias, reshaped_do, reshaped_mean, reshaped_rstd)
  return dx, dw, dbias


@functools.partial(jax.custom_vjp, nondiff_argnums=[3, 4, 5, 6, 7])
@functools.partial(jax.jit, static_argnames=["num_warps", "num_stages",
                                             "num_stages", "eps",
                                             "backward_pass_impl",
                                             "interpret"])
def layer_norm(
    x, weight, bias,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = 3,
    eps: float = 1e-5,
    backward_pass_impl: str = 'triton',
    interpret: bool = False):
  n = x.shape[-1]
  # Triton heuristics
  # Less than 64KB per feature: enqueue fused kernel
  max_fused_size = 65536 // x.dtype.itemsize
  block_size = min(max_fused_size, pl.next_power_of_2(n))
  block_size = min(max(block_size, 128), 4096)
  num_warps = min(max(block_size // 256, 1), 8)

  kernel = functools.partial(layer_norm_forward_kernel, eps=eps,
                             block_size=block_size)
  out_shape = jax.ShapeDtypeStruct(shape=(n,), dtype=x.dtype)
  method = pl.pallas_call(kernel, num_warps=num_warps, num_stages=num_stages,
                          grid=(), out_shape=out_shape, debug=False,
                          interpret=interpret)
  method = jax.vmap(jax.vmap(method, in_axes=(0, None, None)), in_axes=(0, None, None))
  return method(x, weight, bias)
layer_norm.defvjp(layer_norm_forward, layer_norm_backward)


@functools.partial(jax.jit, static_argnames=["eps"])
@functools.partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
def layer_norm_reference(x, weight, bias, *, eps: float = 1e-5):
  mean = jnp.mean(x, axis=1, dtype=jnp.float32)
  mean2 = jnp.mean(jnp.square(x), axis=1, dtype=jnp.float32)
  var = jnp.maximum(0., mean2 - jnp.square(mean))
  y = (x - mean[:, None].astype(x.dtype))
  mul = lax.rsqrt(var + eps).astype(x.dtype)
  return y * mul[:, None] * weight[None].astype(x.dtype) + bias[None].astype(x.dtype)

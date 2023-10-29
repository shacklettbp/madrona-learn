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

"""Module containing fused attention forward and backward pass."""
from __future__ import annotations

import functools
from typing import Any, Optional

import jax
from jax import lax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)

DELAYED_SOFTMAX_NORMALIZE = True
USE_UNMASKED_LOOP_BODY = False
ALLOW_QK_FP16_ACC = False
ALLOW_PV_FP16_ACC = True

def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,  # Input arrays
    segment_ids_ref: jax.Array | None,  # segment_id arrays
    o_ref: Any,  # Output
    *residual_refs: Any,  # Residual outputs
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_d: int,
    block_k: int,
):
  seq_len = q_ref.shape[0]
  start_q = pl.program_id(0)

  pv_acc_dtype = o_ref.dtype if ALLOW_PV_FP16_ACC else jnp.float32
  qk_acc_dtype = q_ref.dtype if ALLOW_QK_FP16_ACC else jnp.float32

  # acc is the buffer where we accumulate the output on sram.
  # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
  m_i = jnp.zeros(block_q, dtype=jnp.float32) - float('inf')
  l_i = jnp.zeros(block_q, dtype=jnp.float32)
  # acc is the buffer where we accumulate the output on sram.
  acc = jnp.zeros((block_q, block_d), pv_acc_dtype)

  # Load q: it will stay in L1 throughout. Indices form a matrix because we
  # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
  # q tile has shape [block_q, block_d], block_d == head_dim.
  q = pl.load(q_ref, (pl.dslice(start_q * block_q, block_q), pl.dslice(None)))
  stability_factor = jnp.log2(seq_len) if DELAYED_SOFTMAX_NORMALIZE else 0.
  q_scale = sm_scale * 1.44269504089
  if q_scale != 1.:
    q *= q_scale

  q_segment_ids = (
      None
      if segment_ids_ref is None
      else pl.load(segment_ids_ref, (pl.dslice(start_q * block_q, block_q),))
  )
  # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
  # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
  # Here we only loop over blocks of kv to process entire seq_len, the loop over
  # blocks of q is carried out by the grid.
  def body(start_k, carry, masked):
    acc, m_prev, l_prev = carry

    k = pl.load(k_ref, (pl.dslice(start_k * block_k, block_k), slice(None)))
    v = pl.load(v_ref, (pl.dslice(start_k * block_k, block_k), slice(None)))
    if masked:
      kv_segment_ids = (
          None
          if segment_ids_ref is None
          else pl.load(segment_ids_ref, (pl.dslice(start_k * block_k, block_k),))
      )
      qk = pl.dot(q, k.T, out_dtype=qk_acc_dtype)  # [block_q, block_k]
      # Bring closer to XLA:GPU numerics.
      qk = qk.astype(jnp.float32)
      if causal or segment_ids_ref is not None:
        mask = None
        if segment_ids_ref is not None:
          mask = segment_mask(q_segment_ids, kv_segment_ids)
        if causal:
          span_q = start_q * block_q + jnp.arange(block_q)
          span_k = start_k * block_k + jnp.arange(block_k)
          causal_mask = span_q[:, None] >= span_k[None, :]
          mask = (
              causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
          )
        # Apply mask to qk.
        qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)
    else:
      qk = pl.dot(q, k.T, out_dtype=qk_acc_dtype).astype(q_ref.dtype)   # [block_q, block_k]
      # Bring closer to XLA:GPU numerics.
      qk = qk.astype(jnp.float32)
    m_curr = jnp.maximum(jnp.max(qk, axis=1), m_prev)
    alpha = jnp.exp2(m_prev - m_curr)
    p = jnp.exp2(qk - m_curr[:, None])

    if DELAYED_SOFTMAX_NORMALIZE:
      l_curr = jnp.sum(p, axis=1) + alpha * l_prev

      acc = (acc * alpha[:, None]).astype(acc.dtype)
    else:
      l_prev *= alpha
      l_curr = jnp.sum(p, axis=1) + l_prev

      l_rcp = 1. / l_curr
      p = p * l_rcp[:, None]

      acc = (acc * (l_prev * l_rcp)[:, None]).astype(acc.dtype)

    p = p.astype(jnp.float16)
    acc += pl.dot(p.astype(v.dtype), v, out_dtype=acc.dtype)
    return acc, m_curr, l_curr
  if causal:
    # Ceildiv (`pl.cdiv` and `//` do not work due to type of start_q)
    upper_bound = lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
    causal_lower_bound = lax.div(block_q * start_q, block_k) if USE_UNMASKED_LOOP_BODY else upper_bound
  else:
    upper_bound = pl.cdiv(seq_len, block_k)  # type: ignore
    causal_lower_bound = upper_bound
  must_mask = segment_ids_ref is not None
  acc, m_i, l_i = lax.fori_loop(causal_lower_bound, upper_bound, functools.partial(body, masked=causal or must_mask),
                                (acc, m_i, l_i))
  acc, m_i, l_i = lax.fori_loop(0, causal_lower_bound, functools.partial(body, masked=must_mask),
                                  (acc, m_i, l_i))
  if DELAYED_SOFTMAX_NORMALIZE:
    acc = acc / l_i[:, None]

  if residual_refs:
    l_ref, m_ref = residual_refs
    pl.store(l_ref, (pl.ds(start_q * block_q, block_q),), l_i)
    pl.store(m_ref, (pl.ds(start_q * block_q, block_q),), m_i)
  # Write output to dram.
  acc = acc.astype(o_ref.dtype)
  pl.store(o_ref, (pl.dslice(start_q * block_q, block_q), pl.dslice(None)), acc)


def segment_mask(
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
):
  # [B, T, 1] or [T, 1]
  q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
  # [B, 1, S] or [1, S]
  if kv_segment_ids.ndim == 1:
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=0)
  else:
    kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=1)
  return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)


@functools.partial(
    jax.custom_vjp, nondiff_argnums=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "block_q",
        "block_k",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "swap_seq_axis",
        "interpret",
        "debug",
    ],
)
def mha(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale: float = 1.0,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    backward_pass_impl: str = "triton",
    swap_seq_axis: bool = False,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    grid=None,
    interpret: bool = False,
    debug: bool = False,
):
  del backward_pass_impl
  if swap_seq_axis:
    batch_size, num_heads, seq_len, head_dim = q.shape
    qkv_block_spec = pl.BlockSpec(lambda _, j, k: (j, k, 0, 0), (None, None, seq_len, head_dim))
  else:
    batch_size, seq_len, num_heads, head_dim = q.shape
    qkv_block_spec = pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim))

  block_q = min(block_q, seq_len)
  block_k = min(block_k, seq_len)
  # Heuristics.
  grid_ = grid
  if grid_ is None:
    grid_ = (pl.cdiv(seq_len, block_q), batch_size, num_heads)

  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4 if head_dim <= 64 else 8
  kernel = functools.partial(mha_forward_kernel, sm_scale=sm_scale,
                             block_q=block_q, block_k=block_k,
                             block_d=head_dim,
                             causal=causal)
  in_specs = [qkv_block_spec for _ in range(3)]
  in_specs.append(
      None  # type: ignore[arg-type]
      if segment_ids is None
      else pl.BlockSpec(lambda _, j, k: (j, 0), (None, seq_len))
  )
  out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
  return pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=in_specs,
      out_specs=qkv_block_spec,
      num_warps=num_warps_,
      num_stages=num_stages,
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_forward",
  )(q, k, v, segment_ids)


def _mha_forward(
    q,
    k,
    v,
    segment_ids: jax.Array | None,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    backward_pass_impl: str,
    swap_seq_axis: bool,
    num_warps: Optional[int],
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
):
  del backward_pass_impl
  if swap_seq_axis:
    batch_size, num_heads, seq_len, head_dim = q.shape
    qkv_block_spec = pl.BlockSpec(lambda _, j, k: (j, k, 0, 0), (None, None, seq_len, head_dim))
  else:
    batch_size, seq_len, num_heads, head_dim = q.shape
    qkv_block_spec = pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None, head_dim))

  block_q = min(block_q, seq_len)
  block_k = min(block_k, seq_len)
  # Heuristics.
  grid_ = grid
  if grid_ is None:
    grid_ = (pl.cdiv(seq_len, block_q), batch_size, num_heads)

  num_warps_ = num_warps
  if num_warps_ is None:
    num_warps_ = 4 if head_dim <= 64 else 8
  kernel = functools.partial(mha_forward_kernel, sm_scale=sm_scale,
                             causal=causal, block_q=block_q, block_k=block_k,
                             block_d=head_dim)
  out_shape = [
      jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype), # out
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), # l
                           dtype=jnp.float32),
      jax.ShapeDtypeStruct(shape=(batch_size, num_heads, seq_len), # m
                           dtype=jnp.float32)
  ]
  in_specs = [qkv_block_spec for _ in range(3)]
  in_specs.append(
      None  # type: ignore[arg-type]
      if segment_ids is None
      else pl.BlockSpec(lambda _, j, k: (j, 0), (None, seq_len))
  )
  out, l, m = pl.pallas_call(
      kernel,
      grid=grid_,
      in_specs=in_specs,
      out_specs=[
          qkv_block_spec,
          pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
          pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
      ],
      num_warps=num_warps_,
      num_stages=num_stages,
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_forward",
  )(q, k, v, segment_ids)
  return out, (q, k, v, segment_ids, out, l, m)


def _preprocess_backward_kernel(out_ref, dout_ref, l_ref,
                                new_dout_ref, delta_ref, *,
                                block_q: int):
  pid_m = pl.program_id(0)

  off_m = pl.ds(pid_m * block_q, block_q)
  # load
  o = pl.load(out_ref, (off_m, slice(None))).astype(jnp.float32)
  do = pl.load(dout_ref, (off_m, slice(None))).astype(jnp.float32)
  denom = pl.load(l_ref, (off_m,)).astype(jnp.float32)
  # compute
  do = do / denom[:, None]
  delta = jnp.sum(o * do, axis=1)
  # write-back
  pl.store(new_dout_ref, (off_m, slice(None)),
           do.astype(new_dout_ref.dtype))
  pl.store(delta_ref, (off_m,), delta.astype(delta_ref.dtype))

def _preprocess_backward(out, do, l, block_q: int, swap_seq_axis: bool,
                         debug: bool, interpret: bool):
  if swap_seq_axis:
    batch_size, num_heads, seq_len, head_dim = out.shape
    out_block_spec = pl.BlockSpec(lambda _, j, k: (j, k, 0, 0), (None, None, seq_len, head_dim))
  else:
    batch_size, seq_len, num_heads, head_dim = out.shape
    out_block_spec = pl.BlockSpec(lambda _, j, k: (j, 0, k, 0), (None, seq_len, None,  head_dim))
  out_shape = [
      jax.ShapeDtypeStruct(do.shape, do.dtype),
      jax.ShapeDtypeStruct(l.shape, l.dtype),
  ]
  do_scaled, delta = pl.pallas_call(
      functools.partial(_preprocess_backward_kernel, block_q=block_q),
      grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
      in_specs=[
        out_block_spec,
        out_block_spec,
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
      ],
      out_specs=[
        out_block_spec,
        pl.BlockSpec(lambda _, j, k: (j, k, 0), (None, None, seq_len)),
      ],
      num_warps=4,
      num_stages=3,
      out_shape=out_shape,
      debug=debug,
      interpret=interpret,
      name="mha_preprocess_backward")(out, do, l)
  return do_scaled, delta


def mha_backward_kernel(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    segment_ids_ref: jax.Array | None,
    out_ref,
    do_scaled_ref,
    l_ref,
    m_ref,
    delta_ref,
    _,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_d: int,
    block_k: int,
):
  del out_ref, l_ref  # Not needed
  seq_len = q_ref.shape[0]

  def outer_loop(start_k, _):

    dv = jnp.zeros([block_k, block_d], dtype=jnp.float32)
    dk = jnp.zeros([block_k, block_d], dtype=jnp.float32)
    k = pl.load(k_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
    v = pl.load(v_ref, (pl.ds(start_k * block_k, block_k), slice(None)))
    span_k = start_k * block_k + jnp.arange(block_k)
    kv_segment_ids = (
        None
        if segment_ids_ref is None
        else pl.load(segment_ids_ref, (pl.ds(start_k * block_k, block_k),))
    )

    def inner_loop(start_q, carry):
      dv, dk = carry
      q = pl.load(q_ref, (pl.ds(start_q * block_q, block_q), slice(None)))

      q_scale = sm_scale * 1.44269504089
      if q_scale != 1.0:
        q_scaled = q * q_scale
      qk = pl.dot(q_scaled, k.T)
      qk = qk.astype(q_ref.dtype)
      qk = qk.astype(jnp.float32)

      q_segment_ids = (
          None
          if segment_ids_ref is None
          else pl.load(segment_ids_ref, (pl.ds(start_q * block_q, block_q),))
      )

      if causal or segment_ids_ref is not None:
        mask = None
        if segment_ids_ref is not None:
          mask = segment_mask(q_segment_ids, kv_segment_ids)

        if causal:
          span_q = start_q * block_q + jnp.arange(block_q)
          causal_mask = span_q[:, None] >= span_k[None, :]
          mask = (
              causal_mask
              if mask is None
              else jnp.logical_and(mask, causal_mask)
          )
        qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

      m = pl.load(m_ref, (pl.ds(start_q * block_q, block_q),))
      p = jnp.exp2(qk - m[:, None])
      do = pl.load(do_scaled_ref, (pl.ds(start_q * block_q, block_q), slice(None)))
      dv = dv + pl.dot(p.astype(do.dtype).T, do)
      di = pl.load(delta_ref, (pl.ds(start_q * block_q, block_q),))
      dp = jnp.zeros((block_q, block_k), dtype=jnp.float32) - di[:, None]
      dp = dp + pl.dot(do, v.T)
      ds = p * dp
      if sm_scale != 1.0:
        ds = ds * sm_scale
      dk = dk + pl.dot(ds.astype(q_ref.dtype).T, q)
      dq = pl.load(dq_ref, (pl.ds(start_q * block_q, block_q),
                            slice(None)), eviction_policy="evict_last")
      dq = dq + pl.dot(ds.astype(k.dtype), k).astype(dq.dtype)
      pl.store(dq_ref, (pl.ds(start_q * block_q, block_q),
                        slice(None)), dq, eviction_policy="evict_last")
      return dv, dk
    if causal:
      lower_bound = lax.div(start_k * block_k, block_q)
    else:
      lower_bound = 0
    dv, dk = lax.fori_loop(lower_bound, pl.cdiv(seq_len, block_q), inner_loop,
                           (dv, dk))
    pl.store(dv_ref, (pl.ds(start_k * block_k, block_k),
                      slice(None)), dv.astype(dv_ref.dtype))
    pl.store(dk_ref, (pl.ds(start_k * block_k, block_k),
                      slice(None)), dk.astype(dk_ref.dtype))
  lax.fori_loop(0, pl.cdiv(seq_len, block_k), outer_loop, None)


def _mha_backward(sm_scale: float, causal: bool, block_q: int, block_k: int,
                  backward_pass_impl: str, swap_seq_axis: bool, num_warps: Optional[int],
                  num_stages: int, grid: Any, interpret: bool,
                  debug: bool, res, do):
  del num_warps, num_stages, grid
  q, k, v, segment_ids, out, l, m = res

  if swap_seq_axis:
    batch_size, num_heads, seq_len, head_dim = q.shape
    qkv_block_spec = pl.BlockSpec(lambda j, k: (j, k, 0, 0), (None, None, seq_len, head_dim))
  else:
    batch_size, seq_len, num_heads, head_dim = q.shape
    qkv_block_spec = pl.BlockSpec(lambda j, k: (j, 0, k, 0), (None, seq_len, None,  head_dim))

  block_q = min(block_q, seq_len)
  block_k = min(block_k, seq_len)
  do_scaled, delta = _preprocess_backward(out, do, l, block_q, swap_seq_axis, debug, interpret)

  if backward_pass_impl == "xla":
    # TODO(jon-chuang): Handle the `swap_seq_axis=True` case for "xla" 
    return jax.vjp(
        functools.partial(mha_reference, sm_scale=sm_scale, causal=causal),
        q,
        k,
        v,
        segment_ids,
    )[1](do)
  elif backward_pass_impl == "triton":
    # We accumulate into dq so we need to initialize it to zeros.
    dq = jnp.zeros(q.shape, jnp.float32)
    out_shapes = [
      jax.ShapeDtypeStruct(dq.shape, dq.dtype),
      jax.ShapeDtypeStruct(k.shape, k.dtype),
      jax.ShapeDtypeStruct(v.shape, v.dtype),
    ]

    in_specs = [qkv_block_spec for _ in range(5)] + [
        pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
        pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
        pl.BlockSpec(lambda j, k: (j, k, 0), (None, None, seq_len)),
        qkv_block_spec,
    ]
    if segment_ids is None:
      in_specs.insert(3, None)  # type: ignore[arg-type]
      input_output_aliases = {8: 0}
    else:
      in_specs.insert(3, pl.BlockSpec(lambda j, k: (j, 0), (None, seq_len)))
      input_output_aliases = {9: 0}
    grid = (batch_size, num_heads)
    # TODO(sharadmv): figure out why num_warps=8 doesn't work!
    num_warps = 4
    dq, dk, dv = pl.pallas_call(
        functools.partial(
            mha_backward_kernel,
            block_q=block_q,
            block_d=head_dim,
            block_k=block_k,
            sm_scale=sm_scale,
            causal=causal,
        ),
        grid=grid,
        out_shape=out_shapes,
        in_specs=in_specs,
        out_specs=[qkv_block_spec for _ in range(3)],
        name="mha_backward",
        debug=debug,
        interpret=interpret,
        num_warps=num_warps,
        num_stages=1,
        input_output_aliases=input_output_aliases,
    )(q, k, v, segment_ids, out, do_scaled, l, m, delta, dq)
  else:
    raise ValueError(f"Invalid backward pass implementation: {backward_pass_impl}")
  return dq.astype(q.dtype), dk, dv, None
mha.defvjp(_mha_forward, _mha_backward)


@functools.partial(jax.jit, static_argnames=['sm_scale', 'causal'])
def mha_reference(
    q,
    k,
    v,
    segment_ids: jnp.ndarray | None,
    sm_scale=1.0,
    causal: bool = False,
):
  q_seq_len = q.shape[1]
  kv_seq_len = k.shape[1]
  logits = jnp.einsum('bqhc,bkhc->bhqk', q, k).astype(jnp.float32)
  mask = None
  if segment_ids is not None:
    mask = jnp.expand_dims(segment_mask(segment_ids, segment_ids), 1)
    mask = jnp.broadcast_to(mask, logits.shape)
  if causal:
    causal_mask = jnp.tril(jnp.ones((1, 1, q_seq_len, kv_seq_len), dtype=bool))
    causal_mask = jnp.broadcast_to(causal_mask, logits.shape)
    mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
  logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
  weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
  return jnp.einsum('bhqk,bkhc->bqhc', weights, v)

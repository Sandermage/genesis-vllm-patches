# SPDX-License-Identifier: Apache-2.0
"""PN350 — fused GDN post-conv Q/K/V split Triton kernel.

Convergent design: SGLang PR #26206 + TensorRT-LLM PR #12966 independently
introduced the same fused split kernel to replace ``torch.split + reshape +
.contiguous`` chain on the GDN post-conv path. Both engines measured +2.65 %
output tok/s on Qwen3.6-35B-A3B; per-layer GDN QKV split time dropped from
18.97 ms → 3.33 ms.

This file ships the kernel. The text-patch wiring is in
``sndr/engines/vllm/patches/attention/gdn/pn350_gdn_qkv_fused_split.py``.

Why kernel-level fusion is a win
================================

Current ``Qwen3_GatedDeltaNet.rearrange_mixed_qkv`` in our pin's
``vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`` does::

    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    fused = torch.cat([query.reshape(-1), key.reshape(-1), value.reshape(-1)], dim=0)
    # then 3 .view(1, seq_len, ...) slices

That's:
  * 1 ``torch.split`` (3 slice views, contiguous tensor)
  * 1 ``torch.cat`` (full-buffer COPY across 3 source tensors)
  * 3 ``.view`` reshape ops (zero-copy)

Net: 1 full-buffer copy + 4-5 kernel launches per call. Called 3 times
per GDN layer in our hot path (spec verify + non-spec prefill + non-spec
decode). On Qwen3.6-35B-A3B that's ~120 layer-calls per forward.

This kernel replaces all of that with **one launch**:
  * 1 program per token row (grid=(seq_len,))
  * Per program: 1 load from `mixed_qkv` row, 3 stores to q/k/v rows
  * Output buffers are freshly-allocated contiguous tensors (no .view chain)

Bench from SGLang #26206 on B200 + Qwen3.6-35B-A3B-FP8:
  * Per-layer GDN QKV split: 18.97 ms → 3.33 ms (~5.7× kernel speedup)
  * Per-layer contiguous microbench: 2.38×
  * End-to-end output tok/s: 7205 → 7397 (+2.66 %)

On Ampere SM 8.6 the speedup carries (memory-bandwidth-bound kernel,
no SM-specific intrinsics, no tcgen05 / wgmma). Bandwidth ratio
A5000:B200 = 768 GB/s : 8 TB/s ≈ 10× — so absolute μs savings scale
proportionally. As a fraction of slower A5000 forward, the % gain
compresses to ~+1-1.5 % single-stream TPS but absolute μs savings
per layer remain.

Shmem budget check for our 99 KiB A5000 budget
==============================================

On Qwen3.6-35B-A3B with TP=2 per-rank dims:
  * ``num_q_heads = num_k_heads = 32 // 2 = 16``, ``head_q = head_k = 128``
  * ``num_v_heads = 32 // 2 = 16``, ``head_v = 128``
  * ``q_dim = 16 * 128 = 2048``, ``k_dim = 2048``, ``v_dim = 2048``
  * ``qkv_dim = 6144``, ``BLOCK_SIZE = triton.next_power_of_2(6144) = 8192``
  * BF16 shmem per program: 8192 * 2 = 16 KiB ≪ 99 KiB ✓
  * num_warps=4, num_stages=2 (tuned for SM 8.6 budget — see PN299 family)

For Qwen3.6-27B-int4 (smaller dims), shmem is even lower. Safe.

Composition + safety
====================

  * No interaction with PN340/PN341 (MTP decode bubble fixes — different
    files / methods).
  * No interaction with PN345 (FLA chunk kernels — different files).
  * Sequential in data-flow with PN204 (in_proj dual-stream — upstream of
    the conv → upstream of this kernel).
  * Sequential in data-flow with PN54 (post-rearrange contiguous dedup —
    PN350 outputs are already contiguous, PN54 .contiguous() becomes no-op).
  * Strict no-regression fallback: caller wraps kernel in try/except;
    on any exception falls back to upstream cat-based split. Operator
    can disable via env ``GENESIS_DISABLE_PN350_GDN_QKV_FUSED_SPLIT=1``.

Limitations
===========

  * ``mixed_qkv.dim() == 2`` (rank 2). Higher-rank inputs route to
    fallback.
  * ``qkv_dim <= 16384`` (BLOCK_SIZE cap). Our PROD shapes are
    well within this.
  * Output dtype matches input dtype. We do NOT support implicit
    dtype conversion in the kernel.
  * Strided input from ``causal_conv1d_fn().transpose(0,1)`` is
    supported — the kernel takes explicit ``stride_t`` /
    ``stride_d`` constexprs.

Author: Sander (Sandermage / Aleksandr Barzov), Odessa, Ukraine.
Source: convergent port of SGLang PR #26206 + TRT-LLM PR #12966 algorithm.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("genesis.kernels.pn350_gdn_qkv_fused_split")

# Try to import Triton; if not available, the kernel can't be used and
# the caller must fall back. We still expose the symbol so the
# integration site's import doesn't fail.
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


# Max payload width — keeps BLOCK_SIZE within Triton's tile-shape sanity
# range and within Ampere SM 8.6 shmem budget for BF16. Our PROD shapes
# (35B q+k+v=6144 per rank, 27B smaller) are well within.
PN350_MAX_QKV_DIM = 16384


if _HAS_TRITON:
    @triton.jit
    def _pn350_fused_qkv_split_kernel(
        # Output pointers (contiguous)
        q_ptr,
        k_ptr,
        v_ptr,
        # Input pointer (may be strided from .transpose())
        mixed_qkv_ptr,
        # Input strides (constexpr for compile-time)
        MIXED_QKV_STRIDE_T: tl.constexpr,
        MIXED_QKV_STRIDE_D: tl.constexpr,
        # Dim metadata (constexpr)
        NUM_Q_HEADS: tl.constexpr,
        NUM_K_HEADS: tl.constexpr,
        NUM_V_HEADS: tl.constexpr,
        HEAD_Q: tl.constexpr,
        HEAD_K: tl.constexpr,
        HEAD_V: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """One program per token row. Loads the row once, scatters to q/k/v.

        Memory pattern (per program):
          * Load: ``mixed_qkv[i_t, 0:qkv_dim]``  (qkv_dim BF16 = 12 KiB)
          * Store q: ``q[i_t, 0:q_dim]``           (q_dim BF16 ≤ 4 KiB)
          * Store k: ``k[i_t, 0:k_dim]``           (k_dim BF16 ≤ 4 KiB)
          * Store v: ``v[i_t, 0:v_dim]``           (v_dim BF16 ≤ 4 KiB)

        Net: 1 read + 3 writes, no on-chip cross-warp shuffle. Pure
        memory copy → bandwidth-bound. On A5000 768 GB/s HBM →
        ~6 KiB/cycle effective copy rate → microseconds per layer.
        """
        i_t = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)

        q_dim: tl.constexpr = NUM_Q_HEADS * HEAD_Q
        k_dim: tl.constexpr = NUM_K_HEADS * HEAD_K
        v_dim: tl.constexpr = NUM_V_HEADS * HEAD_V
        qk_dim: tl.constexpr = q_dim + k_dim
        qkv_dim: tl.constexpr = qk_dim + v_dim

        # Load full row of mixed_qkv (with optional source stride).
        mask = offsets < qkv_dim
        values = tl.load(
            mixed_qkv_ptr
            + i_t * MIXED_QKV_STRIDE_T
            + offsets * MIXED_QKV_STRIDE_D,
            mask=mask,
        )

        # Store q slice [0:q_dim]
        q_mask = offsets < q_dim
        tl.store(q_ptr + i_t * q_dim + offsets, values, mask=q_mask)

        # Store k slice [q_dim:q_dim+k_dim], shifted to k_ptr origin.
        k_offsets = offsets - q_dim
        k_mask = (offsets >= q_dim) & (offsets < qk_dim)
        tl.store(k_ptr + i_t * k_dim + k_offsets, values, mask=k_mask)

        # Store v slice [qk_dim:qkv_dim], shifted to v_ptr origin.
        v_offsets = offsets - qk_dim
        v_mask = (offsets >= qk_dim) & (offsets < qkv_dim)
        tl.store(v_ptr + i_t * v_dim + v_offsets, values, mask=v_mask)


def pn350_fused_qkv_split(
    mixed_qkv: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused GDN post-conv Q/K/V split + reshape on a single Triton launch.

    Args:
        mixed_qkv: 2-D tensor ``[seq_len, q_dim + k_dim + v_dim]`` where
            ``q_dim = num_q_heads * head_q`` (same for k, v). May be
            strided (e.g. from ``causal_conv1d_fn().transpose(0, 1)``).
            Dtype must be one of ``{bf16, fp16, fp32}``.
        num_q_heads, num_k_heads, num_v_heads: per-rank head counts.
        head_q, head_k, head_v: per-head dim sizes.

    Returns:
        Tuple ``(q, k, v)`` shaped ``[1, seq_len, num_*_heads, head_*]``,
        each freshly allocated + contiguous + same dtype as input.

    Raises:
        RuntimeError if Triton not available.
        ValueError if input shape/qkv_dim out of supported range.
    """
    if not _HAS_TRITON:
        raise RuntimeError("PN350 requires Triton (not available)")

    if mixed_qkv.dim() != 2:
        raise ValueError(
            f"PN350 expects rank-2 mixed_qkv, got rank {mixed_qkv.dim()}"
        )

    seq_len, total_dim = mixed_qkv.shape
    q_dim = num_q_heads * head_q
    k_dim = num_k_heads * head_k
    v_dim = num_v_heads * head_v
    qkv_dim = q_dim + k_dim + v_dim
    if total_dim != qkv_dim:
        raise ValueError(
            f"PN350 mixed_qkv last dim {total_dim} != q+k+v dim {qkv_dim}"
        )
    if qkv_dim > PN350_MAX_QKV_DIM:
        raise ValueError(
            f"PN350 qkv_dim {qkv_dim} exceeds supported max "
            f"{PN350_MAX_QKV_DIM} — fall back to upstream cat-based split"
        )

    device = mixed_qkv.device
    dtype = mixed_qkv.dtype

    # Allocate contiguous output tensors. PyTorch zero-init via empty is
    # fine — the kernel writes all valid positions.
    q = torch.empty(
        (1, seq_len, num_q_heads, head_q), device=device, dtype=dtype,
    )
    k = torch.empty(
        (1, seq_len, num_k_heads, head_k), device=device, dtype=dtype,
    )
    v = torch.empty(
        (1, seq_len, num_v_heads, head_v), device=device, dtype=dtype,
    )

    # BLOCK_SIZE = next pow2 of qkv_dim. At BF16 with qkv_dim ≤ 8192,
    # block = 8192 → 16 KiB shmem per program — well under A5000 99 KiB.
    block_size = triton.next_power_of_2(qkv_dim)

    # Input strides — explicit constexpr so the kernel handles both
    # contiguous decode input and strided prefill input (transpose).
    stride_t, stride_d = mixed_qkv.stride()

    # Grid = (seq_len,) — one program per token row.
    grid = (seq_len,)

    _pn350_fused_qkv_split_kernel[grid](
        q,
        k,
        v,
        mixed_qkv,
        MIXED_QKV_STRIDE_T=stride_t,
        MIXED_QKV_STRIDE_D=stride_d,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        NUM_V_HEADS=num_v_heads,
        HEAD_Q=head_q,
        HEAD_K=head_k,
        HEAD_V=head_v,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=2,
    )

    return q, k, v

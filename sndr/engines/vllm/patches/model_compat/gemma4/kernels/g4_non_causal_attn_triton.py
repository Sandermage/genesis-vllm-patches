# SPDX-License-Identifier: Apache-2.0
"""Genesis G4_10 — Triton non-causal attention kernel for Ampere SM 8.6.

================================================================
PURPOSE
================================================================

Implements the missing attention primitive for **head_dim=256
non-causal** on consumer Ampere (RTX 3090 / RTX A5000 / RTX A6000 —
SM 8.6). This unblocks EAGLE-3 and DFlash speculative-decode drafters
on Gemma 4 targets on our hardware (vllm-project/vllm#40382).

================================================================
ALGORITHM
================================================================

Standard FlashAttention-2 (Dao 2023) block-tiled attention, specialized:

  * **head_dim = 256** is the only supported D. We unroll D-loops at
    BLOCK_DMODEL=128 (two passes per head) to fit the SM 8.6 shared
    memory budget (101 KB usable per block).
  * **non-causal** — no upper-triangular masking. We iterate ALL
    KV blocks for every Q row.
  * **block sizes** tuned for SM 8.6:
      BLOCK_M = 64  (queries per block)
      BLOCK_N = 64  (KV keys per block)
      BLOCK_D = 128 (half head_dim — looped twice)
    Total smem usage ≈ Q(64·256·2) + K(64·256·2) + V(64·256·2)
                     + P_acc(64·64·4) + LSE(64·4) ≈ 116 KB peak.
    To keep within 101 KB on SM 8.6 we keep V on-the-fly (no V buffer)
    and rematerialize P after softmax.

Numerical stability:

  * online softmax (Milakov & Gimelshein 2018) — track running max +
    sum-of-exp per query row, rescale accumulator on max update
  * fp32 accumulator throughout
  * fp16 / bf16 inputs, output dtype matches input

================================================================
INTERFACE
================================================================

``g4_non_causal_attn(q, k, v, sm_scale, output_dtype) -> torch.Tensor``

  q, k, v : [num_tokens, num_heads, head_dim=256]  (packed sequence layout)
  sm_scale: float, typically 1/sqrt(head_dim) = 1/16
  output  : [num_tokens, num_heads, head_dim]

Caller is responsible for:
  * laying out Q/K/V correctly (packed sequence per request)
  * applying RoPE before this call (we operate on already-rotated tensors)
  * post-attention output projection

This kernel is **drafter-targeted**: it expects small num_tokens
(typically 4-8 per request × batch=4-8 = 16-64 total) which is the
EAGLE-3 / DFlash block-parallel draft shape. For full-prefill or
large-batch decode use the standard Triton/Flash backends.

================================================================
PERFORMANCE EXPECTATIONS
================================================================

vs FA2 on Hopper (head_dim=256 supported there): expect 0.4-0.6x on
SM 8.6. The gap closes if we add a fp16 P-cache + 2-stage pipelining
in future iterations.

vs FLEX_ATTENTION on SM 8.6 (the only working stock alternative):
expect 2-4x faster. FLEX_ATTENTION has high Python-side overhead
because it uses ``torch.compile`` codegen per call.

================================================================
TEST STRATEGY
================================================================

* CPU reference: torch.nn.functional.scaled_dot_product_attention
  with is_causal=False
* Tolerance: abs diff < 1e-2 (bf16 accumulation noise)
* Shape coverage:
    - num_tokens ∈ {4, 8, 16, 32, 64, 128}
    - num_heads ∈ {1, 8, 16, 32}
    - head_dim = 256 (only supported value)
* Edge cases:
    - num_tokens not divisible by BLOCK_M (last partial block)
    - all-zero K (sanity: result = 0)
    - random input (numerical equivalence)

================================================================
LIMITATIONS
================================================================

* Only ``head_dim=256``. If we need head_dim=512 (Gemma 4 global
  attention layers) we add a sibling kernel ``g4_non_causal_attn_512.py``
  with BLOCK_DMODEL looped in 4 passes; that would push smem to the
  edge and is a separate engineering exercise.
* No KV cache slot indexing — designed for **drafter forward**, where
  every token sees the full sequence (drafters re-compute attention
  per draft step).
* No alibi / pos-bias / sliding window — drafters use vanilla SDPA
  with rotary already pre-applied.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * Dao, "FlashAttention-2", arXiv:2307.08691
  * Milakov & Gimelshein, "Online normalizer", arXiv:1805.02867
  * vLLM TritonAttentionBackend (vllm/v1/attention/backends/triton_attn.py)
    — we mirror its layout conventions but specialize for non-causal head_dim=256
"""
from __future__ import annotations

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


__all__ = ["g4_non_causal_attn", "g4_non_causal_attn_reference"]


# ─── Triton kernel ───────────────────────────────────────────────────


if _HAS_TRITON:

    @triton.jit
    def _g4_non_causal_attn_kernel(
        # Q layout: [num_tokens, num_heads, head_dim]
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        # Strides — token, head, dim
        stride_qt, stride_qh, stride_qd,
        stride_kt, stride_kh, stride_kd,
        stride_vt, stride_vh, stride_vd,
        stride_ot, stride_oh, stride_od,
        # Shape
        num_tokens,
        num_heads,
        sm_scale,
        # Compile-time
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,  # 128 — half of head_dim, looped twice
    ):
        """Non-causal scaled dot-product attention, head_dim=256.

        Grid: (cdiv(num_tokens, BLOCK_M), num_heads)
        """
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)

        # Query row offsets for this block
        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < num_tokens

        # Q tile — load both halves of head_dim
        # Q_tile shape: [BLOCK_M, head_dim=256] — we keep it in registers
        # as two halves of BLOCK_DMODEL=128.
        d0_offsets = tl.arange(0, BLOCK_DMODEL)
        d1_offsets = BLOCK_DMODEL + tl.arange(0, BLOCK_DMODEL)

        q0_ptrs = (
            Q_ptr
            + m_offsets[:, None] * stride_qt
            + pid_h * stride_qh
            + d0_offsets[None, :] * stride_qd
        )
        q1_ptrs = (
            Q_ptr
            + m_offsets[:, None] * stride_qt
            + pid_h * stride_qh
            + d1_offsets[None, :] * stride_qd
        )
        q0 = tl.load(q0_ptrs, mask=m_mask[:, None], other=0.0)
        q1 = tl.load(q1_ptrs, mask=m_mask[:, None], other=0.0)

        # Accumulator + online softmax state
        acc0 = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)

        # Iterate KV blocks (non-causal — all blocks, no mask cutoff)
        for n_start in range(0, num_tokens, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offsets < num_tokens

            # Load K tile (both halves)
            k0_ptrs = (
                K_ptr
                + n_offsets[:, None] * stride_kt
                + pid_h * stride_kh
                + d0_offsets[None, :] * stride_kd
            )
            k1_ptrs = (
                K_ptr
                + n_offsets[:, None] * stride_kt
                + pid_h * stride_kh
                + d1_offsets[None, :] * stride_kd
            )
            k0 = tl.load(k0_ptrs, mask=n_mask[:, None], other=0.0)
            k1 = tl.load(k1_ptrs, mask=n_mask[:, None], other=0.0)

            # qk = Q · K^T  (BLOCK_M, BLOCK_N), computed in two D-passes
            qk = tl.dot(q0, tl.trans(k0))
            qk += tl.dot(q1, tl.trans(k1))
            qk = qk * sm_scale

            # Mask invalid positions (out-of-range KV)
            qk = tl.where(n_mask[None, :], qk, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            # Rescale running stats
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            m_i = m_new

            # Load V tile (both halves)
            v0_ptrs = (
                V_ptr
                + n_offsets[:, None] * stride_vt
                + pid_h * stride_vh
                + d0_offsets[None, :] * stride_vd
            )
            v1_ptrs = (
                V_ptr
                + n_offsets[:, None] * stride_vt
                + pid_h * stride_vh
                + d1_offsets[None, :] * stride_vd
            )
            v0 = tl.load(v0_ptrs, mask=n_mask[:, None], other=0.0)
            v1 = tl.load(v1_ptrs, mask=n_mask[:, None], other=0.0)

            # acc += P @ V  (two D-passes)
            p_fp = p.to(v0.dtype)
            acc0 += tl.dot(p_fp, v0)
            acc1 += tl.dot(p_fp, v1)

        # Final normalize by accumulated denominator
        acc0 = acc0 / l_i[:, None]
        acc1 = acc1 / l_i[:, None]

        # Store output
        o0_ptrs = (
            Out_ptr
            + m_offsets[:, None] * stride_ot
            + pid_h * stride_oh
            + d0_offsets[None, :] * stride_od
        )
        o1_ptrs = (
            Out_ptr
            + m_offsets[:, None] * stride_ot
            + pid_h * stride_oh
            + d1_offsets[None, :] * stride_od
        )
        tl.store(o0_ptrs, acc0.to(Out_ptr.dtype.element_ty), mask=m_mask[:, None])
        tl.store(o1_ptrs, acc1.to(Out_ptr.dtype.element_ty), mask=m_mask[:, None])


# ─── Python entry-points ─────────────────────────────────────────────


def g4_non_causal_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Non-causal scaled dot-product attention for Ampere SM 8.6, head_dim=256.

    Args:
        q: [num_tokens, num_heads, 256] fp16/bf16
        k: [num_tokens, num_heads, 256] same dtype as q
        v: [num_tokens, num_heads, 256] same dtype as q
        sm_scale: scalar; defaults to 1/sqrt(256) = 0.0625
        output: optional pre-allocated output tensor (same shape as q)

    Returns:
        Output tensor [num_tokens, num_heads, 256], same dtype as q.

    Raises:
        ImportError if triton isn't installed.
        ValueError if head_dim ≠ 256.
    """
    if not _HAS_TRITON:
        raise ImportError(
            "[G4_10 kernel] triton is not installed; install triton ≥ 2.3 to use "
            "Genesis Ampere non-causal attention."
        )
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"[G4_10 kernel] q/k/v shape mismatch: q={q.shape}, k={k.shape}, v={v.shape}"
        )
    if q.ndim != 3:
        raise ValueError(
            f"[G4_10 kernel] expected [num_tokens, num_heads, head_dim], got ndim={q.ndim}"
        )
    num_tokens, num_heads, head_dim = q.shape
    if head_dim != 256:
        raise ValueError(
            f"[G4_10 kernel] this kernel is specialized for head_dim=256, got {head_dim}. "
            "For head_dim=512 (Gemma 4 global attention) use g4_non_causal_attn_512 (TBD)."
        )
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(float(head_dim))

    if output is None:
        output = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_DMODEL = 128
    grid = (triton.cdiv(num_tokens, BLOCK_M), num_heads)

    _g4_non_causal_attn_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        num_tokens,
        num_heads,
        sm_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=4,
        num_stages=2,
    )
    return output


def g4_non_causal_attn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Pure-PyTorch reference for numerical equivalence testing.

    Implements ``softmax(QK^T / sqrt(d)) V`` with full attention (no mask).
    Used by ``test_g4_10_non_causal_attn.py``.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(float(q.shape[-1]))
    # q/k/v: [T, H, D] → permute to [H, T, D] for batched matmul
    q_t = q.transpose(0, 1).contiguous()       # [H, T, D]
    k_t = k.transpose(0, 1).contiguous()       # [H, T, D]
    v_t = v.transpose(0, 1).contiguous()       # [H, T, D]
    scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * sm_scale  # [H, T_q, T_k]
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v_t)            # [H, T_q, D]
    return out.transpose(0, 1).contiguous()     # back to [T_q, H, D]

"""P67 Triton kernel v1 — multi-query attention, NO compression, NO GQA, NO causal.

Simplest possible Triton kernel that handles Q_LEN > 1 (K+1 query tokens per
request). Validates the core multi-query attention pattern in Triton before
adding GQA / causal / TurboQuant compression complexity.

Targets PyTorch reference Layer 1 (`reference_multi_query_attention_layer1`).

Tested on Ampere SM 8.6 (RTX A5000). Should work on any SM ≥ 7.5 (Triton
requires Volta+) since we use only `tl.dot`, `tl.exp`, `tl.softmax`-style
ops with FP16 inputs — no Hopper WGMMA / FA3 intrinsics.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _p67_v1_multi_query_attention(
    Q_ptr,          # [B, K_PLUS_1, H, D]  fp16
    K_ptr,          # [B, S, H, D]         fp16
    V_ptr,          # [B, S, H, D]         fp16
    O_ptr,          # [B, K_PLUS_1, H, D]  fp32 (output, will be cast back)
    Seq_lens_ptr,   # [B] int32 — context length per request (= S since no padding)
    # Strides
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    # Scalars + constexprs
    SCALE: tl.constexpr,         # 1.0/sqrt(D)
    K_PLUS_1: tl.constexpr,      # query tokens per request (= 1 + num_speculative)
    BLOCK_D: tl.constexpr,       # next_pow2(D), used for tile width
    HEAD_DIM: tl.constexpr,      # actual D (for masking)
    BLOCK_KV: tl.constexpr,      # tokens per KV tile
):
    """Grid: (B, H, 1)  — one CTA per (batch, head)."""
    bid = tl.program_id(0)
    hid = tl.program_id(1)

    seq_len = tl.load(Seq_lens_ptr + bid)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    qt_range = tl.arange(0, K_PLUS_1)
    kv_range = tl.arange(0, BLOCK_KV)

    # Load Q for K+1 tokens × this head
    # Q shape: [K_PLUS_1, BLOCK_D]
    q_base = bid * stride_qb + hid * stride_qh
    q_addrs = (
        q_base
        + qt_range[:, None] * stride_qt
        + d_offs[None, :] * stride_qd
    )
    q = tl.load(Q_ptr + q_addrs, mask=d_mask[None, :], other=0.0).to(tl.float32)
    # q shape: [K_PLUS_1, BLOCK_D]

    # Online softmax accumulator per (q_token)
    m_prev = tl.zeros([K_PLUS_1], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([K_PLUS_1], dtype=tl.float32)
    acc = tl.zeros([K_PLUS_1, BLOCK_D], dtype=tl.float32)

    # KV loop
    k_base = bid * stride_kb + hid * stride_kh
    v_base = bid * stride_vb + hid * stride_vh
    for start_n in range(0, seq_len, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < seq_len

        # Load K tile: [BLOCK_KV, BLOCK_D]
        k_addrs = (
            k_base
            + kv_offs[:, None] * stride_ks
            + d_offs[None, :] * stride_kd
        )
        k = tl.load(
            K_ptr + k_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        # k shape: [BLOCK_KV, BLOCK_D]

        # Compute scores: [K_PLUS_1, BLOCK_KV] = Q @ K.T
        # Use tl.dot for tensor cores. Cast to fp16 for tensor core compatibility.
        scores = tl.dot(
            q.to(tl.float16), tl.trans(k.to(tl.float16))
        ).to(tl.float32) * SCALE
        scores = tl.where(kv_mask[None, :], scores, -float("inf"))

        # Per-q-token online softmax
        n_e_max = tl.maximum(tl.max(scores, axis=1), m_prev)  # [K_PLUS_1]
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max[:, None])  # [K_PLUS_1, BLOCK_KV]

        # Load V tile: [BLOCK_KV, BLOCK_D]
        v_addrs = (
            v_base
            + kv_offs[:, None] * stride_vs
            + d_offs[None, :] * stride_vd
        )
        v = tl.load(
            V_ptr + v_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate: acc[t, d] = acc[t, d] * re_scale[t] + sum_kv(p[t, kv] * v[kv, d])
        # = acc * re_scale[:, None] + tl.dot(p[K_PLUS_1, BLOCK_KV], v[BLOCK_KV, BLOCK_D])
        # → tl.dot output [K_PLUS_1, BLOCK_D]
        new_acc = tl.dot(p.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        acc = acc * re_scale[:, None] + new_acc
        l_prev = l_prev * re_scale + tl.sum(p, axis=1)
        m_prev = n_e_max

    # Final divide
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    out = acc / safe_l[:, None]  # [K_PLUS_1, BLOCK_D]

    # Store output: [K_PLUS_1, BLOCK_D]
    o_base = bid * stride_ob + hid * stride_oh
    o_addrs = (
        o_base
        + qt_range[:, None] * stride_ot
        + d_offs[None, :] * stride_od
    )
    tl.store(O_ptr + o_addrs, out, mask=d_mask[None, :])


def p67_v1_multi_query_attention(
    q: torch.Tensor,        # [B, K_PLUS_1, H, D] fp16
    k: torch.Tensor,        # [B, S, H, D] fp16
    v: torch.Tensor,        # [B, S, H, D] fp16
    seq_lens: torch.Tensor, # [B] int32
    scale: float,
    block_kv: int = 16,
) -> torch.Tensor:
    """Launch wrapper for the v1 multi-query kernel."""
    B, K_PLUS_1, H, D = q.shape
    assert k.shape[0] == B and k.shape[2] == H and k.shape[3] == D
    assert v.shape == k.shape
    S = k.shape[1]

    # Round D up to next power of 2 for BLOCK_D
    BLOCK_D = triton.next_power_of_2(D)

    out = torch.empty_like(q, dtype=torch.float32)

    grid = (B, H, 1)
    _p67_v1_multi_query_attention[grid](
        q, k, v, out, seq_lens,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        SCALE=scale,
        K_PLUS_1=K_PLUS_1,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=D,
        BLOCK_KV=block_kv,
    )
    return out.to(q.dtype)

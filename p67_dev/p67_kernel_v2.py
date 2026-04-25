"""P67 Triton kernel v2 — adds GQA grouping on top of v1.

Now Hq != Hk (Hq = Hk * KV_GROUP_SIZE). Each KV head is shared by
KV_GROUP_SIZE query heads. Pack BLOCK_H query heads per CTA (like P40
upstream grouped decode kernel) for arithmetic intensity via tl.dot.

Grid: (batch, head_group, 1) where head_group iterates over groups of
BLOCK_H query heads that share KV heads.

For our prod (Qwen3.6-35B-A3B): Hq=64, Hk=4, KV_GROUP_SIZE=16. With
BLOCK_H=16, each CTA handles 1 full GQA group (16 query heads sharing
1 KV head). Number of head_groups = Hq / BLOCK_H = 4.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _p67_v2_multi_query_gqa(
    Q_ptr,          # [B, K_PLUS_1, Hq, D]  fp16
    K_ptr,          # [B, S, Hk, D]         fp16
    V_ptr,          # [B, S, Hk, D]         fp16
    O_ptr,          # [B, K_PLUS_1, Hq, D]  fp32
    Seq_lens_ptr,   # [B] int32
    # Strides
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_ot, stride_oh, stride_od,
    # Constants
    SCALE: tl.constexpr,
    K_PLUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Hq_TOTAL: tl.constexpr,
):
    """Grid: (B, cdiv(Hq, BLOCK_H), 1)."""
    bid = tl.program_id(0)
    head_group_id = tl.program_id(1)

    # Compute which Hq heads + which Hk head this CTA handles
    heads_per_kv: tl.constexpr = tl.cdiv(KV_GROUP_SIZE, BLOCK_H)
    kv_head = head_group_id // heads_per_kv
    group_idx_in_kv = head_group_id % heads_per_kv
    cur_head = (
        kv_head * KV_GROUP_SIZE
        + group_idx_in_kv * BLOCK_H
        + tl.arange(0, BLOCK_H)
    )
    # CRITICAL: cur_head must stay within the SAME kv_head's group, AND
    # within total Hq. Without the (kv_head+1)*KV_GROUP_SIZE bound, BLOCK_H
    # would spill into the next kv_head's queries with this kv_head's KV
    # values → wrong attention. Mirrors upstream P40 _tq_grouped_decode_stage1.
    mask_h = (cur_head < (kv_head + 1) * KV_GROUP_SIZE) & (cur_head < Hq_TOTAL)

    seq_len = tl.load(Seq_lens_ptr + bid)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    qt_range = tl.arange(0, K_PLUS_1)
    kv_range = tl.arange(0, BLOCK_KV)

    # Load Q for K_PLUS_1 tokens × BLOCK_H heads
    # Shape: [K_PLUS_1, BLOCK_H, BLOCK_D]
    q_base = bid * stride_qb
    q_addrs = (
        q_base
        + qt_range[:, None, None] * stride_qt
        + cur_head[None, :, None] * stride_qh
        + d_offs[None, None, :] * stride_qd
    )
    q = tl.load(
        Q_ptr + q_addrs,
        mask=mask_h[None, :, None] & d_mask[None, None, :],
        other=0.0,
    ).to(tl.float32)
    # q shape: [K_PLUS_1, BLOCK_H, BLOCK_D]

    # Reshape Q to 2D for tl.dot: [K_PLUS_1*BLOCK_H, BLOCK_D]
    q_2d = tl.reshape(q, [K_PLUS_1 * BLOCK_H, BLOCK_D])

    # Online softmax accumulator per (q_token, head)
    m_prev = tl.zeros([K_PLUS_1, BLOCK_H], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([K_PLUS_1, BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([K_PLUS_1, BLOCK_H, BLOCK_D], dtype=tl.float32)

    # KV loop — read shared KV head
    k_base = bid * stride_kb + tl.cast(kv_head, tl.int64) * stride_kh
    v_base = bid * stride_vb + tl.cast(kv_head, tl.int64) * stride_vh

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

        # Compute scores: [K_PLUS_1*BLOCK_H, BLOCK_KV] = Q_2d @ K.T
        scores_2d = tl.dot(
            q_2d.to(tl.float16), tl.trans(k.to(tl.float16))
        ).to(tl.float32) * SCALE
        # Reshape to [K_PLUS_1, BLOCK_H, BLOCK_KV]
        scores = tl.reshape(scores_2d, [K_PLUS_1, BLOCK_H, BLOCK_KV])
        scores = tl.where(
            mask_h[None, :, None] & kv_mask[None, None, :],
            scores, -float("inf"),
        )

        # Per-(t, h) online softmax
        n_e_max = tl.maximum(tl.max(scores, axis=2), m_prev)  # [K_PLUS_1, BLOCK_H]
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max[:, :, None])  # [K_PLUS_1, BLOCK_H, BLOCK_KV]

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

        # Accumulate: acc[t, h, d] += sum_kv p[t, h, kv] * v[kv, d]
        # p shape [K_PLUS_1, BLOCK_H, BLOCK_KV] → reshape [K_PLUS_1*BLOCK_H, BLOCK_KV]
        p_2d = tl.reshape(p, [K_PLUS_1 * BLOCK_H, BLOCK_KV])
        new_acc_2d = tl.dot(
            p_2d.to(tl.float16), v.to(tl.float16)
        ).to(tl.float32)
        new_acc = tl.reshape(new_acc_2d, [K_PLUS_1, BLOCK_H, BLOCK_D])

        acc = acc * re_scale[:, :, None] + new_acc
        l_prev = l_prev * re_scale + tl.sum(p, axis=2)
        m_prev = n_e_max

    # Final divide
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    out = acc / safe_l[:, :, None]  # [K_PLUS_1, BLOCK_H, BLOCK_D]

    # Store output
    o_base = bid * stride_ob
    o_addrs = (
        o_base
        + qt_range[:, None, None] * stride_ot
        + cur_head[None, :, None] * stride_oh
        + d_offs[None, None, :] * stride_od
    )
    tl.store(
        O_ptr + o_addrs,
        out,
        mask=mask_h[None, :, None] & d_mask[None, None, :],
    )


def p67_v2_multi_query_gqa(
    q: torch.Tensor,        # [B, K_PLUS_1, Hq, D] fp16
    k: torch.Tensor,        # [B, S, Hk, D] fp16
    v: torch.Tensor,        # [B, S, Hk, D] fp16
    seq_lens: torch.Tensor, # [B] int32
    scale: float,
    block_kv: int = 16,
    block_h: int = 16,
) -> torch.Tensor:
    """Launch wrapper for v2 kernel (multi-query + GQA)."""
    B, K_PLUS_1, Hq, D = q.shape
    assert k.shape[0] == B and k.shape[3] == D
    assert v.shape == k.shape
    S, Hk = k.shape[1], k.shape[2]
    assert Hq % Hk == 0
    kv_group_size = Hq // Hk

    # tl.dot requires both dims ≥ 16. Keep BLOCK_H ≥ 16; if KV_GROUP_SIZE is
    # smaller, mask_h inside the kernel suppresses the spillover heads.
    block_h = max(16, block_h)

    BLOCK_D = triton.next_power_of_2(D)
    out = torch.empty_like(q, dtype=torch.float32)

    heads_per_kv = triton.cdiv(kv_group_size, block_h)
    num_head_groups = Hk * heads_per_kv

    grid = (B, num_head_groups, 1)
    _p67_v2_multi_query_gqa[grid](
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
        BLOCK_H=block_h,
        KV_GROUP_SIZE=kv_group_size,
        Hq_TOTAL=Hq,
    )
    return out.to(q.dtype)

"""P67 Triton kernel v3 — V2 + causal mask within current K+1 chunk + prior/chunk split.

Now matches reference Layer 3:
  - Prior cached KV: [B, S_prior, Hk, D] — read in phase 1 loop, no causal
  - Current chunk K/V: [B, K_PLUS_1, Hk, D] — read in phase 2, with causal mask

Phase 2 causal mask: q_token t can attend to chunk position k iff t >= k
(within the current chunk). Prior cached positions are always reachable.

Online softmax accumulator carries from phase 1 → phase 2. Per (q_token, head)
state.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _p67_v3_multi_query_gqa_causal(
    Q_ptr,            # [B, K_PLUS_1, Hq, D] fp16
    K_cached_ptr,     # [B, S_prior, Hk, D] fp16
    V_cached_ptr,     # [B, S_prior, Hk, D] fp16
    K_chunk_ptr,      # [B, K_PLUS_1, Hk, D] fp16
    V_chunk_ptr,      # [B, K_PLUS_1, Hk, D] fp16
    O_ptr,            # [B, K_PLUS_1, Hq, D] fp32
    Seq_lens_ptr,     # [B] int32 — prior cached length per request
    # Strides
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_kcb, stride_kcs, stride_kch, stride_kcd,
    stride_vcb, stride_vcs, stride_vch, stride_vcd,
    stride_kkb, stride_kkt, stride_kkh, stride_kkd,
    stride_vkb, stride_vkt, stride_vkh, stride_vkd,
    stride_ob, stride_ot, stride_oh, stride_od,
    # Constexprs
    SCALE: tl.constexpr,
    K_PLUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Hq_TOTAL: tl.constexpr,
):
    """Grid: (B, num_head_groups, 1)."""
    bid = tl.program_id(0)
    head_group_id = tl.program_id(1)

    heads_per_kv: tl.constexpr = tl.cdiv(KV_GROUP_SIZE, BLOCK_H)
    kv_head = head_group_id // heads_per_kv
    group_idx_in_kv = head_group_id % heads_per_kv
    cur_head = (
        kv_head * KV_GROUP_SIZE
        + group_idx_in_kv * BLOCK_H
        + tl.arange(0, BLOCK_H)
    )
    mask_h = (cur_head < (kv_head + 1) * KV_GROUP_SIZE) & (cur_head < Hq_TOTAL)

    seq_len = tl.load(Seq_lens_ptr + bid)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    qt_range = tl.arange(0, K_PLUS_1)
    kv_range = tl.arange(0, BLOCK_KV)

    # Load Q [K_PLUS_1, BLOCK_H, BLOCK_D]
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
    q_2d = tl.reshape(q, [K_PLUS_1 * BLOCK_H, BLOCK_D])

    # Online softmax state per (t, h)
    m_prev = tl.zeros([K_PLUS_1, BLOCK_H], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([K_PLUS_1, BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([K_PLUS_1, BLOCK_H, BLOCK_D], dtype=tl.float32)

    # ── PHASE 1: prior cached KV (no causal — all positions valid) ──
    kc_base = bid * stride_kcb + tl.cast(kv_head, tl.int64) * stride_kch
    vc_base = bid * stride_vcb + tl.cast(kv_head, tl.int64) * stride_vch

    for start_n in range(0, seq_len, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < seq_len

        k_addrs = (
            kc_base
            + kv_offs[:, None] * stride_kcs
            + d_offs[None, :] * stride_kcd
        )
        k = tl.load(
            K_cached_ptr + k_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        scores_2d = tl.dot(
            q_2d.to(tl.float16), tl.trans(k.to(tl.float16))
        ).to(tl.float32) * SCALE
        scores = tl.reshape(scores_2d, [K_PLUS_1, BLOCK_H, BLOCK_KV])
        scores = tl.where(
            mask_h[None, :, None] & kv_mask[None, None, :],
            scores, -float("inf"),
        )

        n_e_max = tl.maximum(tl.max(scores, axis=2), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max[:, :, None])

        v_addrs = (
            vc_base
            + kv_offs[:, None] * stride_vcs
            + d_offs[None, :] * stride_vcd
        )
        v = tl.load(
            V_cached_ptr + v_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        p_2d = tl.reshape(p, [K_PLUS_1 * BLOCK_H, BLOCK_KV])
        new_acc_2d = tl.dot(p_2d.to(tl.float16), v.to(tl.float16)).to(tl.float32)
        new_acc = tl.reshape(new_acc_2d, [K_PLUS_1, BLOCK_H, BLOCK_D])

        acc = acc * re_scale[:, :, None] + new_acc
        l_prev = l_prev * re_scale + tl.sum(p, axis=2)
        m_prev = n_e_max

    # ── PHASE 2: current chunk K/V (causal mask within K_PLUS_1) ──
    # Current chunk has exactly K_PLUS_1 positions. Load all at once.
    # K_chunk shape: [K_PLUS_1, BLOCK_D] for kv_head
    kk_base = bid * stride_kkb + tl.cast(kv_head, tl.int64) * stride_kkh
    vk_base = bid * stride_vkb + tl.cast(kv_head, tl.int64) * stride_vkh

    chunk_addrs_k = (
        kk_base
        + qt_range[:, None] * stride_kkt
        + d_offs[None, :] * stride_kkd
    )
    k_chunk = tl.load(
        K_chunk_ptr + chunk_addrs_k,
        mask=d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # k_chunk shape: [K_PLUS_1, BLOCK_D]

    chunk_addrs_v = (
        vk_base
        + qt_range[:, None] * stride_vkt
        + d_offs[None, :] * stride_vkd
    )
    v_chunk = tl.load(
        V_chunk_ptr + chunk_addrs_v,
        mask=d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Compute scores: q_2d [K_PLUS_1*BLOCK_H, BLOCK_D] @ k_chunk.T [BLOCK_D, K_PLUS_1]
    # → [K_PLUS_1*BLOCK_H, K_PLUS_1]
    # Triton tl.dot needs both dims ≥ 16. K_PLUS_1=4 violates.
    # Workaround: pad k_chunk to BLOCK_KV (16) with -inf score positions.
    # Build padded K with K_PLUS_1 valid + (BLOCK_KV - K_PLUS_1) padding.

    # Pad: create k_chunk_padded [BLOCK_KV, BLOCK_D] = k_chunk for [0..K_PLUS_1)
    # and zeros for [K_PLUS_1..BLOCK_KV). Then mask scores to -inf for padding.
    chunk_pad_offs = tl.arange(0, BLOCK_KV)
    chunk_pad_mask = chunk_pad_offs < K_PLUS_1

    # Re-load with padded shape
    chunk_addrs_k_pad = (
        kk_base
        + chunk_pad_offs[:, None] * stride_kkt
        + d_offs[None, :] * stride_kkd
    )
    k_chunk_pad = tl.load(
        K_chunk_ptr + chunk_addrs_k_pad,
        mask=chunk_pad_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    chunk_addrs_v_pad = (
        vk_base
        + chunk_pad_offs[:, None] * stride_vkt
        + d_offs[None, :] * stride_vkd
    )
    v_chunk_pad = tl.load(
        V_chunk_ptr + chunk_addrs_v_pad,
        mask=chunk_pad_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # scores_2d = q_2d @ k_chunk_pad.T → [K_PLUS_1*BLOCK_H, BLOCK_KV]
    chunk_scores_2d = tl.dot(
        q_2d.to(tl.float16), tl.trans(k_chunk_pad.to(tl.float16))
    ).to(tl.float32) * SCALE
    chunk_scores = tl.reshape(chunk_scores_2d, [K_PLUS_1, BLOCK_H, BLOCK_KV])

    # Causal mask within K_PLUS_1: q_pos[t]=t can attend to k_pos[k]=k iff t >= k
    # qt_range[:, None, None] >= chunk_pad_offs[None, None, :]
    causal_mask = qt_range[:, None, None] >= chunk_pad_offs[None, None, :]
    valid_mask = (
        mask_h[None, :, None]
        & chunk_pad_mask[None, None, :]
        & causal_mask
    )
    chunk_scores = tl.where(valid_mask, chunk_scores, -float("inf"))

    # Update softmax state with chunk scores
    n_e_max = tl.maximum(tl.max(chunk_scores, axis=2), m_prev)
    re_scale_chunk = tl.exp(m_prev - n_e_max)
    p_chunk = tl.exp(chunk_scores - n_e_max[:, :, None])

    # Accumulate with v_chunk_pad
    p_chunk_2d = tl.reshape(p_chunk, [K_PLUS_1 * BLOCK_H, BLOCK_KV])
    new_acc_chunk_2d = tl.dot(
        p_chunk_2d.to(tl.float16), v_chunk_pad.to(tl.float16)
    ).to(tl.float32)
    new_acc_chunk = tl.reshape(new_acc_chunk_2d, [K_PLUS_1, BLOCK_H, BLOCK_D])

    acc = acc * re_scale_chunk[:, :, None] + new_acc_chunk
    l_prev = l_prev * re_scale_chunk + tl.sum(p_chunk, axis=2)
    m_prev = n_e_max

    # Final divide
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    out = acc / safe_l[:, :, None]

    # Store
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


def p67_v3_multi_query_gqa_causal(
    q: torch.Tensor,         # [B, K_PLUS_1, Hq, D]
    k_cached: torch.Tensor,  # [B, S_prior, Hk, D]
    v_cached: torch.Tensor,  # [B, S_prior, Hk, D]
    k_chunk: torch.Tensor,   # [B, K_PLUS_1, Hk, D]
    v_chunk: torch.Tensor,   # [B, K_PLUS_1, Hk, D]
    seq_lens: torch.Tensor,  # [B] int32
    scale: float,
    block_kv: int = 16,
    block_h: int = 16,
) -> torch.Tensor:
    """Launch wrapper for v3 kernel."""
    B, K_PLUS_1, Hq, D = q.shape
    S_prior, Hk = k_cached.shape[1], k_cached.shape[2]
    assert Hq % Hk == 0
    kv_group_size = Hq // Hk
    assert k_chunk.shape == (B, K_PLUS_1, Hk, D)
    assert v_chunk.shape == (B, K_PLUS_1, Hk, D)

    block_h = max(16, block_h)
    BLOCK_D = triton.next_power_of_2(D)
    out = torch.empty_like(q, dtype=torch.float32)

    heads_per_kv = triton.cdiv(kv_group_size, block_h)
    num_head_groups = Hk * heads_per_kv

    grid = (B, num_head_groups, 1)
    _p67_v3_multi_query_gqa_causal[grid](
        q, k_cached, v_cached, k_chunk, v_chunk, out, seq_lens,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_cached.stride(0), k_cached.stride(1), k_cached.stride(2), k_cached.stride(3),
        v_cached.stride(0), v_cached.stride(1), v_cached.stride(2), v_cached.stride(3),
        k_chunk.stride(0), k_chunk.stride(1), k_chunk.stride(2), k_chunk.stride(3),
        v_chunk.stride(0), v_chunk.stride(1), v_chunk.stride(2), v_chunk.stride(3),
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

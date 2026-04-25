"""P67 Triton kernel v4 — V3 + compressed TurboQuant cache read for Phase 1.

Production target. Reads cached KV from TurboQuant k8v4 layout directly:
  - FP8 K (KPS bytes per slot)
  - 4-bit V indices (VAL_DATA_BYTES per slot)
  - FP16 scale + FP16 zero (4 bytes per slot)

Same Phase 1/Phase 2 structure as V3, just Phase 1 K/V load changed from
explicit FP16 tensors to compressed cache read with block_table lookup.

Per-slot byte layout (matches upstream `_tq_decode_stage1` exactly):
  bytes [0..KPS)              K data (FP8 e4nv on Hopper+, e4b15 on Ampere/Ada)
  bytes [KPS..KPS+VAL_DATA_BYTES)  4-bit V indices (2 nibbles per byte)
  bytes [KPS+VDB..KPS+VDB+2)  V scale (FP16 little-endian)
  bytes [KPS+VDB+2..KPS+VDB+4) V zero  (FP16 little-endian)

Cache memory: [num_blocks, block_size, Hk, padded_slot] uint8

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _p67_v4_compressed_cache(
    Q_ptr,                  # [B, K_PLUS_1, Hq, D] fp16
    KV_cache_ptr,           # uint8 — [num_blocks, block_size, Hk, padded_slot]
    Block_table_ptr,        # [B, max_num_blocks] int32
    Seq_lens_ptr,           # [B] int32 — prior cached length per request
    K_chunk_ptr,            # [B, K_PLUS_1, Hk, D] fp16 — current chunk K
    V_chunk_ptr,            # [B, K_PLUS_1, Hk, D] fp16 — current chunk V
    O_ptr,                  # [B, K_PLUS_1, Hq, D] fp32 output
    # Strides
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_cache_block, stride_cache_pos, stride_cache_head,
    stride_bt_b,
    stride_kkb, stride_kkt, stride_kkh, stride_kkd,
    stride_vkb, stride_vkt, stride_vkh, stride_vkd,
    stride_ob, stride_ot, stride_oh, stride_od,
    # Constexprs
    SCALE: tl.constexpr,
    K_PLUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,           # KV cache page block size
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Hq_TOTAL: tl.constexpr,
    KPS: tl.constexpr,                  # key payload size in bytes per slot
    VAL_DATA_BYTES: tl.constexpr,       # value indices bytes per slot
    FP8_E4B15: tl.constexpr = 0,        # 1 = e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
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

    # Precompute V index unpack offsets (loop-invariant)
    vb_idx = d_offs // 2
    vb_shift = (d_offs % 2) * 4

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

    bt_base = bid * stride_bt_b

    # ── PHASE 1: prior cached KV (compressed read, no causal) ──
    for start_n in range(0, seq_len, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < seq_len

        # Block table lookup
        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask, other=0,
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ── K dequant: FP8 → fp32 ──
        k_addrs = slot_bases[:, None] + d_offs[None, :]
        k_raw = tl.load(
            KV_cache_ptr + k_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        )
        if FP8_E4B15:
            k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

        # Compute scores: [K_PLUS_1*BLOCK_H, BLOCK_KV]
        scores_2d = tl.dot(
            q_2d.to(tl.float16), tl.trans(k_float.to(tl.float16))
        ).to(tl.float32) * SCALE
        scores = tl.reshape(scores_2d, [K_PLUS_1, BLOCK_H, BLOCK_KV])
        scores = tl.where(
            mask_h[None, :, None] & kv_mask[None, None, :],
            scores, -float("inf"),
        )

        # Per-(t, h) online softmax update
        n_e_max = tl.maximum(tl.max(scores, axis=2), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max[:, :, None])

        # ── V dequant: 4-bit + scale + zero ──
        val_bases = slot_bases + KPS
        val_addrs = val_bases[:, None] + vb_idx[None, :]
        val_raw = tl.load(
            KV_cache_ptr + val_addrs,
            mask=kv_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

        sc_bases = val_bases + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(tl.uint16)
        v_scales = (
            (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        )
        zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(tl.uint16)
        v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        values = v_idx * v_scales[:, None] + v_zeros[:, None]
        # values shape: [BLOCK_KV, BLOCK_D]

        # Accumulate
        p_2d = tl.reshape(p, [K_PLUS_1 * BLOCK_H, BLOCK_KV])
        new_acc_2d = tl.dot(
            p_2d.to(tl.float16), values.to(tl.float16)
        ).to(tl.float32)
        new_acc = tl.reshape(new_acc_2d, [K_PLUS_1, BLOCK_H, BLOCK_D])

        acc = acc * re_scale[:, :, None] + new_acc
        l_prev = l_prev * re_scale + tl.sum(p, axis=2)
        m_prev = n_e_max

    # ── PHASE 2: current chunk K/V (uncompressed FP16, with causal) ──
    chunk_pad_offs = tl.arange(0, BLOCK_KV)
    chunk_pad_mask = chunk_pad_offs < K_PLUS_1

    kk_base = bid * stride_kkb + tl.cast(kv_head, tl.int64) * stride_kkh
    vk_base = bid * stride_vkb + tl.cast(kv_head, tl.int64) * stride_vkh

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

    chunk_scores_2d = tl.dot(
        q_2d.to(tl.float16), tl.trans(k_chunk_pad.to(tl.float16))
    ).to(tl.float32) * SCALE
    chunk_scores = tl.reshape(chunk_scores_2d, [K_PLUS_1, BLOCK_H, BLOCK_KV])

    causal_mask = qt_range[:, None, None] >= chunk_pad_offs[None, None, :]
    valid_mask = (
        mask_h[None, :, None]
        & chunk_pad_mask[None, None, :]
        & causal_mask
    )
    chunk_scores = tl.where(valid_mask, chunk_scores, -float("inf"))

    n_e_max = tl.maximum(tl.max(chunk_scores, axis=2), m_prev)
    re_scale_chunk = tl.exp(m_prev - n_e_max)
    p_chunk = tl.exp(chunk_scores - n_e_max[:, :, None])

    p_chunk_2d = tl.reshape(p_chunk, [K_PLUS_1 * BLOCK_H, BLOCK_KV])
    new_acc_chunk_2d = tl.dot(
        p_chunk_2d.to(tl.float16), v_chunk_pad.to(tl.float16)
    ).to(tl.float32)
    new_acc_chunk = tl.reshape(new_acc_chunk_2d, [K_PLUS_1, BLOCK_H, BLOCK_D])

    acc = acc * re_scale_chunk[:, :, None] + new_acc_chunk
    l_prev = l_prev * re_scale_chunk + tl.sum(p_chunk, axis=2)
    m_prev = n_e_max

    # Final divide + store
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    out = acc / safe_l[:, :, None]

    o_base = bid * stride_ob
    o_addrs = (
        o_base
        + qt_range[:, None, None] * stride_ot
        + cur_head[None, :, None] * stride_oh
        + d_offs[None, None, :] * stride_od
    )
    tl.store(
        O_ptr + o_addrs, out,
        mask=mask_h[None, :, None] & d_mask[None, None, :],
    )


def p67_v4_compressed(
    q: torch.Tensor,                # [B, K_PLUS_1, Hq, D] fp16
    kv_cache: torch.Tensor,         # uint8 [num_blocks, block_size, Hk, padded_slot]
    block_table: torch.Tensor,      # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,         # [B] int32
    k_chunk: torch.Tensor,          # [B, K_PLUS_1, Hk, D] fp16
    v_chunk: torch.Tensor,          # [B, K_PLUS_1, Hk, D] fp16
    scale: float,
    block_size: int,                # KV cache page block size
    kps: int,                       # key payload size (bytes per slot, = HEAD_DIM for FP8)
    val_data_bytes: int,            # value index bytes (= HEAD_DIM/2 for 4-bit)
    block_kv: int = 16,
    block_h: int = 16,
    fp8_e4b15: int = 1,             # default Ampere/Ada (RTX A5000 = e4b15)
) -> torch.Tensor:
    """Launch wrapper for V4 — TurboQuant compressed cache read."""
    B, K_PLUS_1, Hq, D = q.shape
    Hk = k_chunk.shape[2]
    assert k_chunk.shape == (B, K_PLUS_1, Hk, D)
    assert v_chunk.shape == (B, K_PLUS_1, Hk, D)
    assert Hq % Hk == 0
    kv_group_size = Hq // Hk

    block_h = max(16, block_h)
    BLOCK_D = triton.next_power_of_2(D)
    out = torch.empty_like(q, dtype=torch.float32)

    heads_per_kv = triton.cdiv(kv_group_size, block_h)
    num_head_groups = Hk * heads_per_kv

    grid = (B, num_head_groups, 1)
    _p67_v4_compressed_cache[grid](
        q, kv_cache, block_table, seq_lens,
        k_chunk, v_chunk, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        k_chunk.stride(0), k_chunk.stride(1), k_chunk.stride(2), k_chunk.stride(3),
        v_chunk.stride(0), v_chunk.stride(1), v_chunk.stride(2), v_chunk.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        SCALE=scale,
        K_PLUS_1=K_PLUS_1,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        BLOCK_KV=block_kv,
        BLOCK_H=block_h,
        KV_GROUP_SIZE=kv_group_size,
        Hq_TOTAL=Hq,
        KPS=kps,
        VAL_DATA_BYTES=val_data_bytes,
        FP8_E4B15=fp8_e4b15,
    )
    return out.to(q.dtype)

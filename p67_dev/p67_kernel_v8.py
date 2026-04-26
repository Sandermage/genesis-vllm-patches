"""P67 Triton kernel v8 — BLOCK_M fusion + K transposed-load + native dtype.

Optimizations over v4 (based on research from PR #40792, vLLM kernel_unified_
attention, and triton#9830 bug analysis):

1. **BLOCK_M fusion**: drop separate BLOCK_H dim; use BLOCK_M = K_PLUS_1 × HEADS_PER_KV.
   For Qwen3.6 (K+1=4, Hq=8, Hk=1 → HEADS_PER_KV=8): BLOCK_M=32 (vs v4's BLOCK_H=16
   with 50% padding waste). Q layout becomes [B, BLOCK_M, BLOCK_D] where row-i
   indexes a (q_token, head) pair via `t = i // HEADS_PER_KV; h = i % HEADS_PER_KV`.

2. **K loaded directly in (HEAD_SIZE, TILE_SIZE) layout** — drop `tl.trans()`.
   Mirror upstream pattern: `k_offset = ... + offs_d[:, None] * stride_d + tile_offs[None, :] * stride_pos`.
   Saves a transpose op per Phase 1 iteration.

3. **Native Q dtype into tl.dot** — don't force fp16 cast. Triton#9830 shows
   `tl.dot(fp16,fp16)` on SM 8.6 returns wrong values in some shapes. Keep Q
   in fp32, let Triton's MMA picker choose the right path with TF32.

4. **Sanitization preserved** — Inf/NaN→0 in K and V dequant is still our
   unique mitigation against e4b15 saturation. Upstream has no such guard.

5. **Scale fold inside dot epilogue** — `scale * tl.dot(Q, K)` instead of
   `tl.dot(...) * scale` matches upstream pattern.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _p67_v8_compressed_cache(
    Q_ptr,                  # [B, K_PLUS_1, Hq, D] fp16 (contig)
    KV_cache_ptr,           # uint8 [num_blocks, BLOCK_SIZE, Hk, slot_size]
    Block_table_ptr,        # [B, max_num_blocks] int32
    Seq_lens_ptr,           # [B] int32 — TOTAL seq_len (prior + K_PLUS_1)
    K_chunk_ptr,            # [B, K_PLUS_1, Hk, D] fp16
    V_chunk_ptr,            # [B, K_PLUS_1, Hk, D] fp16
    O_ptr,                  # [B, K_PLUS_1, Hq, D] fp32 output
    # Strides (in elements, not bytes)
    stride_qb, stride_qt, stride_qh, stride_qd,
    stride_cache_block, stride_cache_pos, stride_cache_head,  # uint8 → bytes
    stride_bt_b,
    stride_kkb, stride_kkt, stride_kkh, stride_kkd,
    stride_vkb, stride_vkt, stride_vkh, stride_vkd,
    stride_ob, stride_ot, stride_oh, stride_od,
    # Constexprs
    SCALE: tl.constexpr,
    K_PLUS_1: tl.constexpr,
    BLOCK_D: tl.constexpr,           # padded power-of-2 head dim
    HEAD_DIM: tl.constexpr,          # actual head dim
    BLOCK_SIZE: tl.constexpr,        # KV cache page block size
    BLOCK_KV: tl.constexpr,          # tile size along KV
    HEADS_PER_KV: tl.constexpr,      # = Hq // Hk (GQA factor)
    Hq_TOTAL: tl.constexpr,
    KPS: tl.constexpr,               # bytes for K data per slot (= D for FP8)
    VAL_DATA_BYTES: tl.constexpr,    # bytes for V indices per slot (= D//2 for 4-bit)
    FP8_E4B15: tl.constexpr = 0,
):
    """Grid: (B, num_kv_heads, 1). Each CTA = one (req, kv_head) pair.

    Q/K/V tile layout:
      Q: [BLOCK_M=K_PLUS_1*HEADS_PER_KV, BLOCK_D]
      K (per tile): [BLOCK_D, BLOCK_KV]   — already transposed via stride trick
      V (per tile): [BLOCK_KV, BLOCK_D]
    """
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)

    BLOCK_M: tl.constexpr = K_PLUS_1 * HEADS_PER_KV  # 32 for Qwen3.6 (4 × 8)

    # Per-row indices: row m → (q_token t, head h within group, abs head idx)
    offs_m = tl.arange(0, BLOCK_M)
    q_t = offs_m // HEADS_PER_KV         # 0..K_PLUS_1-1
    head_in_group = offs_m % HEADS_PER_KV
    abs_head = kv_head * HEADS_PER_KV + head_in_group  # global query-head idx
    head_mask = abs_head < Hq_TOTAL

    seq_len = tl.load(Seq_lens_ptr + bid)
    prior_seq_len = seq_len - K_PLUS_1

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < HEAD_DIM
    offs_kv = tl.arange(0, BLOCK_KV)
    vb_idx = offs_d // 2
    vb_shift = (offs_d % 2) * 4

    # ───── Load Q [BLOCK_M, BLOCK_D] ─────
    q_base = bid * stride_qb
    # Q_ptr address per (m, d): bid + q_t*stride_qt + abs_head*stride_qh + d*stride_qd
    q_addrs = (
        q_base
        + q_t[:, None] * stride_qt
        + abs_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    Q = tl.load(
        Q_ptr + q_addrs,
        mask=head_mask[:, None] & d_mask[None, :],
        other=0.0,
    )  # [BLOCK_M, BLOCK_D] in q dtype

    # ───── Online softmax accumulators per row m ─────
    M_state = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    L_state = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ════════════════════════════════════════════════════════════════════
    # PHASE 1 — prior cached KV (compressed read, no causal)
    # ════════════════════════════════════════════════════════════════════
    for start_n in range(0, prior_seq_len, BLOCK_KV):
        seq_offset = start_n + offs_kv  # [BLOCK_KV]
        tile_mask = seq_offset < prior_seq_len

        # Block table lookup (one int per logical KV position)
        page_idx = seq_offset // BLOCK_SIZE
        page_off = seq_offset % BLOCK_SIZE
        physical_block = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=tile_mask, other=0,
        ).to(tl.int64)
        slot_bases = (
            physical_block * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # K dequant — load directly in [HEAD_SIZE, TILE_SIZE] layout (no tl.trans)
        # k_addrs[d, n] = slot_bases[n] + d
        k_addrs = slot_bases[None, :] + offs_d[:, None]
        k_raw = tl.load(
            KV_cache_ptr + k_addrs,
            mask=d_mask[:, None] & tile_mask[None, :],
            other=0,
        )
        if FP8_E4B15:
            k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        # Sanitize Inf/NaN: e4b15 max ~0.94, exp_field=15 encodes Inf/NaN.
        # Production cache occasionally has these from K-projection outliers
        # (RMSnorm doesn't bound activations < 0.94). Without this guard, NaN
        # propagates through softmax. Upstream lacks this; our unique mitigation.
        k_safe = tl.where(
            (k_float == k_float) & (k_float < 1e30) & (k_float > -1e30),
            k_float, 0.0,
        )

        # ───── Score: S = scale * Q @ K [BLOCK_M, BLOCK_KV] ─────
        # Native Q dtype (no force cast — Triton#9830 fp16 dot SM 8.6 bug)
        S = SCALE * tl.dot(Q, k_safe.to(Q.dtype))

        # Mask invalid (head-pad rows or out-of-tile cols) → -inf
        S = tl.where(
            head_mask[:, None] & tile_mask[None, :],
            S, float("-inf"),
        )

        # Online softmax update (per row)
        M_new = tl.maximum(tl.max(S, axis=1), M_state)
        alpha = tl.exp(M_state - M_new)
        P = tl.exp(S - M_new[:, None])
        L_state = L_state * alpha + tl.sum(P, axis=1)
        acc = acc * alpha[:, None]

        # ───── V dequant — 4-bit + scale + zero ─────
        val_bases = slot_bases + KPS
        val_addrs = val_bases[:, None] + vb_idx[None, :]  # [BLOCK_KV, BLOCK_D//2]
        val_raw = tl.load(
            KV_cache_ptr + val_addrs,
            mask=tile_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

        sc_bases = val_bases + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=tile_mask, other=0).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=tile_mask, other=0).to(tl.uint16)
        v_scales = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=tile_mask, other=0).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=tile_mask, other=0).to(tl.uint16)
        v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        V_dequant = v_idx * v_scales[:, None] + v_zeros[:, None]
        # Sanitize V — same rationale as K
        V_safe = tl.where(
            (V_dequant == V_dequant) & (V_dequant < 1e30) & (V_dequant > -1e30),
            V_dequant, 0.0,
        )  # [BLOCK_KV, BLOCK_D]

        # ───── Accumulate: acc += P @ V_safe [BLOCK_M, BLOCK_D] ─────
        acc += tl.dot(P.to(Q.dtype), V_safe.to(Q.dtype))
        M_state = M_new  # ← critical: advance state for next iter

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2 — current chunk K/V (uncompressed FP16, with causal mask)
    # ════════════════════════════════════════════════════════════════════
    chunk_offs = tl.arange(0, BLOCK_KV)
    chunk_mask = chunk_offs < K_PLUS_1

    kk_base = bid * stride_kkb + tl.cast(kv_head, tl.int64) * stride_kkh
    vk_base = bid * stride_vkb + tl.cast(kv_head, tl.int64) * stride_vkh

    # K_chunk loaded as [BLOCK_D, BLOCK_KV] (no transpose needed)
    k_chunk_addrs = (
        kk_base
        + offs_d[:, None] * stride_kkd
        + chunk_offs[None, :] * stride_kkt
    )
    K_chunk = tl.load(
        K_chunk_ptr + k_chunk_addrs,
        mask=d_mask[:, None] & chunk_mask[None, :],
        other=0.0,
    )

    # V_chunk loaded as [BLOCK_KV, BLOCK_D]
    v_chunk_addrs = (
        vk_base
        + chunk_offs[:, None] * stride_vkt
        + offs_d[None, :] * stride_vkd
    )
    V_chunk = tl.load(
        V_chunk_ptr + v_chunk_addrs,
        mask=chunk_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    S_chunk = SCALE * tl.dot(Q, K_chunk.to(Q.dtype))

    # Causal mask: q_token t can see chunk position k iff k <= t
    causal_mask = q_t[:, None] >= chunk_offs[None, :]
    valid = head_mask[:, None] & chunk_mask[None, :] & causal_mask
    S_chunk = tl.where(valid, S_chunk, float("-inf"))

    M_new = tl.maximum(tl.max(S_chunk, axis=1), M_state)
    alpha = tl.exp(M_state - M_new)
    P_chunk = tl.exp(S_chunk - M_new[:, None])
    L_state = L_state * alpha + tl.sum(P_chunk, axis=1)
    acc = acc * alpha[:, None]
    acc += tl.dot(P_chunk.to(Q.dtype), V_chunk.to(Q.dtype))

    M_state = M_new

    # ───── Epilogue: normalize and store ─────
    safe_L = tl.where(L_state > 0.0, L_state, 1.0)
    out = acc / safe_L[:, None]

    # Output layout: O[bid, q_t, abs_head, d]
    o_base = bid * stride_ob
    o_addrs = (
        o_base
        + q_t[:, None] * stride_ot
        + abs_head[:, None] * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(
        O_ptr + o_addrs, out,
        mask=head_mask[:, None] & d_mask[None, :],
    )


def p67_v8_compressed(
    q,                  # [B, K_PLUS_1, Hq, D] fp16
    kv_cache,           # uint8 [num_blocks, BLOCK_SIZE, Hk, slot_size]
    block_table,        # [B, max_blocks] int32
    seq_lens,           # [B] int32 — TOTAL (prior + K_PLUS_1)
    k_chunk,            # [B, K_PLUS_1, Hk, D] fp16
    v_chunk,            # [B, K_PLUS_1, Hk, D] fp16
    scale: float,
    block_size: int,
    kps: int,
    val_data_bytes: int,
    fp8_e4b15: int = 1,
    block_kv: int = 16,
):
    B, K_PLUS_1, Hq, D = q.shape
    Hk = k_chunk.shape[2]
    assert Hq % Hk == 0, f"Hq={Hq} must be divisible by Hk={Hk}"
    heads_per_kv = Hq // Hk

    BLOCK_D = triton.next_power_of_2(D)
    output = torch.empty_like(q, dtype=torch.float32)

    grid = (B, Hk, 1)
    _p67_v8_compressed_cache[grid](
        q, kv_cache, block_table, seq_lens,
        k_chunk, v_chunk, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        k_chunk.stride(0), k_chunk.stride(1), k_chunk.stride(2), k_chunk.stride(3),
        v_chunk.stride(0), v_chunk.stride(1), v_chunk.stride(2), v_chunk.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        SCALE=scale,
        K_PLUS_1=K_PLUS_1,
        BLOCK_D=BLOCK_D,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        BLOCK_KV=block_kv,
        HEADS_PER_KV=heads_per_kv,
        Hq_TOTAL=Hq,
        KPS=kps,
        VAL_DATA_BYTES=val_data_bytes,
        FP8_E4B15=fp8_e4b15,
        num_warps=4,
        num_stages=2,
    )
    return output.to(q.dtype)

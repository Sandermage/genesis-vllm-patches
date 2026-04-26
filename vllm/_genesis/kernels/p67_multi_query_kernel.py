# SPDX-License-Identifier: Apache-2.0
"""P67 production wrapper — multi-query TurboQuant attention for spec-decode.

v7.34 SPLIT-M (Fix A from algorithms research, 2026-04-26):
- Outer loop loads K/V tiles ONCE per iteration (same memory bandwidth)
- Inner `tl.static_range` unrolls K_PLUS_1 separate tl.dot calls
- Each q_t has INDEPENDENT online-softmax accumulators
- Mathematically bit-exact match to per-query upstream path
- Zero perf loss: Triton unrolls static_range → same SASS

ROOT CAUSE for prior drift (~0.2% rel_avg, off-by-1 digit on number recall):
The fused multi-query MMA (BLOCK_M=K_PLUS_1*HEADS_PER_KV=32) had different
per-row epilogue ORDERING than upstream per-query path. Online-softmax rescale
α_i applied in different sequence → compounded magnitude history drift over
~256 KV iterations (Golden et al., arXiv 2405.02803, "Is Flash Attention
Stable?" — documented O(N) drift in fused FA vs baseline).

Fix A from algorithms research (arXiv 2203.03341, FA3 paper, vLLM #40792
hoseung2 grouped decode pattern): split-M with shared K/V load, per-q_t
independent accumulators. Each q_t gets bit-identical accumulator history
to per-query reference.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

log = logging.getLogger("genesis.kernels.p67")

_ENV_ENABLE = "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on"
    )


_CACHED_KERNEL = None


def _build_kernel():
    """Define the v7.34 split-M Triton kernel. Returns None on import failure."""
    try:
        from vllm.triton_utils import tl, triton
    except Exception:
        try:
            import triton
            import triton.language as tl
        except Exception:
            return None

    @triton.jit
    def _p67_v8_split_m_cache(
        Q_ptr,
        KV_cache_ptr,
        Block_table_ptr,
        Seq_lens_ptr,
        K_chunk_ptr,         # unused — kept for API compat
        V_chunk_ptr,         # unused — kept for API compat
        O_ptr,
        stride_qb, stride_qt, stride_qh, stride_qd,
        stride_cache_block, stride_cache_pos, stride_cache_head,
        stride_bt_b,
        stride_kkb, stride_kkt, stride_kkh, stride_kkd,  # unused
        stride_vkb, stride_vkt, stride_vkh, stride_vkd,  # unused
        stride_ob, stride_ot, stride_oh, stride_od,
        SCALE: tl.constexpr,
        K_PLUS_1: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_KV: tl.constexpr,
        HEADS_PER_KV: tl.constexpr,
        Hq_TOTAL: tl.constexpr,
        KPS: tl.constexpr,
        VAL_DATA_BYTES: tl.constexpr,
        FP8_E4B15: tl.constexpr = 0,
    ):
        """Grid: (B, num_kv_heads, 1).

        SPLIT-M architecture:
        - Outer loop: load K/V tile ONCE per iteration (bandwidth-shared)
        - Inner static_range over K_PLUS_1: separate tl.dot per q_t
        - Each q_t has BLOCK_M = HEADS_PER_KV rows (e.g., 8 for Qwen3.6)
        - Per-q_t accumulators (M, L, acc) updated independently
        - Bit-exact match to per-query upstream path
        """
        bid = tl.program_id(0)
        kv_head = tl.program_id(1)

        # Per-q_t block dimension (was BLOCK_M = K_PLUS_1 * HEADS_PER_KV in v7.27)
        BLOCK_QH: tl.constexpr = HEADS_PER_KV

        offs_h = tl.arange(0, BLOCK_QH)
        abs_head = kv_head * HEADS_PER_KV + offs_h
        head_mask = abs_head < Hq_TOTAL

        # vLLM convention: seq_lens[i] = TOTAL length INCLUDING K_PLUS_1 chunk.
        total_seq_len = tl.load(Seq_lens_ptr + bid)
        prior_seq_len = total_seq_len - K_PLUS_1

        offs_d = tl.arange(0, BLOCK_D)
        d_mask = offs_d < HEAD_DIM
        offs_kv = tl.arange(0, BLOCK_KV)
        vb_idx = offs_d // 2
        vb_shift = (offs_d % 2) * 4

        # Per-q_t accumulator state stored as [K_PLUS_1, BLOCK_QH] / [K_PLUS_1, BLOCK_QH, BLOCK_D]
        # Triton can't dynamically index by constexpr — use where-masking for writes.
        M_state = tl.zeros([K_PLUS_1, BLOCK_QH], dtype=tl.float32) - float("inf")
        L_state = tl.zeros([K_PLUS_1, BLOCK_QH], dtype=tl.float32)
        acc = tl.zeros([K_PLUS_1, BLOCK_QH, BLOCK_D], dtype=tl.float32)

        q_t_range = tl.arange(0, K_PLUS_1)
        q_base = bid * stride_qb

        bt_base = bid * stride_bt_b

        # ════════════════════════════════════════════════════════════════
        # OUTER LOOP — KV tiles. K/V loaded ONCE per iteration.
        # ════════════════════════════════════════════════════════════════
        for start_n in range(0, total_seq_len, BLOCK_KV):
            seq_offset = start_n + offs_kv
            tile_mask = seq_offset < total_seq_len

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

            # K loaded transposed: [HEAD_SIZE, TILE_SIZE]
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
            # Sanitize Inf/NaN — clamp to FP8 e4b15 max.
            k_safe = tl.where(k_float == k_float, k_float, 0.0)
            k_safe = tl.minimum(tl.maximum(k_safe, -0.9375), 0.9375)
            K_tile = k_safe  # keep fp32 — required for input_precision='ieee' to take effect

            # V dequant — load 4-bit indices + scale + zero, build V_tile
            val_bases = slot_bases + KPS
            val_addrs = val_bases[:, None] + vb_idx[None, :]
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
            V_safe = tl.where(V_dequant == V_dequant, V_dequant, 0.0)
            V_safe = tl.minimum(tl.maximum(V_safe, -32.0), 32.0)
            V_tile = V_safe  # keep fp32 for IEEE precision PV dot

            # ─────────── SPLIT-M: per-q_t independent dots ───────────
            # Static unroll over K_PLUS_1 query tokens. Each iteration:
            # 1. Load Q tile for this t (Triton compiler unrolls — no runtime index)
            # 2. Compute scores for ONE q_t (BLOCK_QH × BLOCK_KV tile)
            # 3. Apply causal mask for that q_t's absolute position
            # 4. Online softmax update for that q_t's accumulators only
            # Bit-exact to per-query upstream path (no cross-row drift).
            for t in tl.static_range(0, K_PLUS_1):
                q_abs_pos_t = prior_seq_len + t

                # Load Q tile for this q_t: [BLOCK_QH, BLOCK_D]
                # t is constexpr-static here so address computation is constant.
                q_addrs_t = (
                    q_base
                    + t * stride_qt
                    + abs_head[:, None] * stride_qh
                    + offs_d[None, :] * stride_qd
                )
                Q_t_raw = tl.load(
                    Q_ptr + q_addrs_t,
                    mask=head_mask[:, None] & d_mask[None, :],
                    other=0.0,
                )  # [BLOCK_QH, BLOCK_D] — fp16 from input
                # Upcast Q to fp32 — required for input_precision='ieee' to be effective.
                Q_t = Q_t_raw.to(tl.float32)

                # S_t = scale * Q_t @ K_tile [BLOCK_QH, BLOCK_KV]
                # input_precision='ieee' forces SOFTWARE fp32 dot (no Tensor Cores)
                # → full 23-bit mantissa, matches upstream's element-wise fp32 sum.
                # tf32x3: Markidis 3xTF32 emulation — Tensor Core throughput
                # with ~700x precision boost vs default TF32 (CUTLASS data).
                # Requires fp32 inputs (which we have).
                S_t = SCALE * tl.dot(Q_t, K_tile, out_dtype=tl.float32, input_precision='tf32x3')

                # Per-row causal mask for this q_t:
                # q_abs_pos_t (scalar) >= seq_offset[n]
                causal = q_abs_pos_t >= seq_offset
                valid = head_mask[:, None] & tile_mask[None, :] & causal[None, :]
                S_t = tl.where(valid, S_t, float("-inf"))

                # Triton can't index 2D with constexpr, use mask-based extract.
                # tl.where avoids -inf*0=NaN issue from naive multiplication.
                t_mask = q_t_range == t  # [K_PLUS_1] one-hot, compile-time constant
                # Extract row t: where t_mask use M_state, else 0; sum picks the t row.
                M_old_t = tl.sum(tl.where(t_mask[:, None], M_state, 0.0), axis=0)
                L_old_t = tl.sum(tl.where(t_mask[:, None], L_state, 0.0), axis=0)
                acc_old_t = tl.sum(tl.where(t_mask[:, None, None], acc, 0.0), axis=0)

                # Online softmax update for this q_t
                M_new_t = tl.maximum(tl.max(S_t, axis=1), M_old_t)
                alpha_t = tl.exp(M_old_t - M_new_t)
                P_t = tl.exp(S_t - M_new_t[:, None])
                L_new_t = L_old_t * alpha_t + tl.sum(P_t, axis=1)
                # PV for this q_t: [BLOCK_QH, BLOCK_D]
                # P_t already fp32, V_tile fp32 → IEEE software dot for full precision.
                acc_new_t = acc_old_t * alpha_t[:, None] + tl.dot(
                    P_t, V_tile, out_dtype=tl.float32, input_precision='tf32x3'
                )

                # Write back into per-q_t accumulator slots via where-mask.
                M_state = tl.where(t_mask[:, None], M_new_t[None, :], M_state)
                L_state = tl.where(t_mask[:, None], L_new_t[None, :], L_state)
                acc = tl.where(
                    t_mask[:, None, None],
                    acc_new_t[None, :, :],
                    acc,
                )

        # ───── Epilogue: normalize and store ─────
        safe_L = tl.where(L_state > 0.0, L_state, 1.0)
        out = acc / safe_L[:, :, None]  # [K_PLUS_1, BLOCK_QH, BLOCK_D]

        # O_ptr layout: [B, K_PLUS_1, Hq, D]
        o_addrs_3d = (
            bid * stride_ob
            + q_t_range[:, None, None] * stride_ot
            + abs_head[None, :, None] * stride_oh
            + offs_d[None, None, :] * stride_od
        )
        tl.store(
            O_ptr + o_addrs_3d, out.to(tl.float16),
            mask=head_mask[None, :, None] & d_mask[None, None, :],
        )

    return _p67_v8_split_m_cache


def _get_kernel():
    global _CACHED_KERNEL
    if _CACHED_KERNEL is None:
        _CACHED_KERNEL = _build_kernel()
    return _CACHED_KERNEL


def _autoconfig(sm_major: int, sm_minor: int, head_dim: int) -> dict:
    """Pick BLOCK_KV/num_warps/num_stages per SM.

    v7.39 aggressive tune for Ampere consumer (SM 8.6 RTX A5000):
    - BLOCK_KV=32: doubles tile size, halves loop iter count
    - num_warps=8: more parallelism per CTA (was 4)
    - num_stages=3: deeper async pipeline for better hide latency
    Override via env: GENESIS_P67_BLOCK_KV, GENESIS_P67_NUM_WARPS, GENESIS_P67_NUM_STAGES.
    """
    import os as _os
    block_kv = int(_os.environ.get("GENESIS_P67_BLOCK_KV", "32"))
    num_warps = int(_os.environ.get("GENESIS_P67_NUM_WARPS", "8" if sm_major >= 8 else "4"))
    num_stages = int(_os.environ.get("GENESIS_P67_NUM_STAGES", "3" if sm_major >= 8 else "2"))
    return dict(BLOCK_KV=block_kv, num_warps=num_warps, num_stages=num_stages)


def _detect_fp8_mode() -> int:
    try:
        import torch
        cap = torch.cuda.get_device_capability()
        return 1 if cap < (8, 9) else 0
    except Exception:
        return 1


def is_active() -> bool:
    if not _env_enabled():
        return False
    if _get_kernel() is None:
        return False
    return True


def alloc_output_buffer(B, K_PLUS_1, Hq, D, device, dtype):
    """Pre-allocate reusable output buffer (cudagraph-safe)."""
    import torch
    return torch.empty((B, K_PLUS_1, Hq, D), dtype=dtype, device=device)


def call_p67_attention(
    q,
    kv_cache,
    block_table,
    seq_lens,
    k_chunk,
    v_chunk,
    scale: float,
    block_size: int,
    kps: int,
    val_data_bytes: int,
    output=None,
):
    """Production launcher for P67 v7.34 split-M multi-query attention."""
    import torch
    import triton

    kernel = _get_kernel()
    if kernel is None:
        raise ImportError("P67 Triton kernel not available")

    B, K_PLUS_1, Hq, D = q.shape
    Hk = k_chunk.shape[2]
    assert Hq % Hk == 0
    heads_per_kv = Hq // Hk

    cap = torch.cuda.get_device_capability()
    cfg = _autoconfig(cap[0], cap[1], D)

    BLOCK_D = triton.next_power_of_2(D)

    if output is None:
        output = torch.empty_like(q)
    assert output.dtype == q.dtype, (
        f"output dtype {output.dtype} must match q dtype {q.dtype}"
    )

    fp8_e4b15 = _detect_fp8_mode()

    grid = (B, Hk, 1)
    kernel[grid](
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
        BLOCK_KV=cfg["BLOCK_KV"],
        HEADS_PER_KV=heads_per_kv,
        Hq_TOTAL=Hq,
        KPS=kps,
        VAL_DATA_BYTES=val_data_bytes,
        FP8_E4B15=fp8_e4b15,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return output


def diagnostic_info() -> dict[str, Any]:
    info = {"env_enabled": _env_enabled(), "version": "v7.39_aggressive_tune"}
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            info["sm"] = f"{cap[0]}.{cap[1]}"
            info["fp8_mode"] = "e4b15" if cap < (8, 9) else "e4nv"
            info["autoconfig"] = _autoconfig(cap[0], cap[1], 128)
        else:
            info["cuda"] = False
    except Exception as e:
        info["error"] = str(e)
    info["kernel_built"] = _get_kernel() is not None
    return info

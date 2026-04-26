"""P67 wrapper end-to-end test — mimics production call shape exactly.

Tests the production wrapper `call_p67_attention` against a synthetic
TurboQuant cache built with realistic prod parameters:
  - BLOCK_SIZE=2832 (mamba-aligned page size used by Qwen3.6 hybrid)
  - Hq=64, Hk=4, D=128 (Qwen3.6-35B-A3B-FP8)
  - K_PLUS_1=4 (MTP num_speculative_tokens=3 → verify K+1=4)
  - kps=128 (full-FP8 key), val_data_bytes=64 (4-bit value)
  - seq_lens (vLLM convention) = prior + K_PLUS_1

Goal: catch shape/stride/NaN bugs that unit tests with smaller BLOCK_SIZE
miss. Run inside the same image vLLM uses so Triton compile semantics match.

Usage (from inside container):
  PYTHONPATH=. python3 p67_test_wrapper_e2e.py
"""
from __future__ import annotations

import math
import sys

import torch


def _pack_e4b15_byte(values: torch.Tensor) -> torch.Tensor:
    """fp32 values → uint8 bytes interpreted as fp8e4b15 (Ampere/Ada bias=15).

    fp8e4b15: 1 sign + 4 exponent (bias 15) + 3 mantissa.
    """
    # Clamp to representable range
    abs_v = values.abs()
    sign = (values < 0).to(torch.uint8) << 7
    # exponent = floor(log2(abs)) + bias, clamped
    safe = abs_v.clamp(min=1e-10)
    exp_unbiased = torch.floor(torch.log2(safe)).to(torch.int32)
    exp_biased = (exp_unbiased + 15).clamp(0, 30)  # 30 = max-1, reserve 31 for inf/nan
    # mantissa = (abs / 2^exp_unbiased - 1) * 8, rounded
    mant_f = (abs_v / torch.pow(2.0, exp_unbiased.to(torch.float32)) - 1.0) * 8.0
    mant_i = mant_f.round().to(torch.int32).clamp(0, 7)
    # Handle zero
    zero_mask = abs_v < 1e-10
    exp_biased = torch.where(zero_mask, torch.zeros_like(exp_biased), exp_biased)
    mant_i = torch.where(zero_mask, torch.zeros_like(mant_i), mant_i)
    byte = sign | ((exp_biased.to(torch.uint8) & 0x0F) << 3) | (mant_i.to(torch.uint8) & 0x07)
    return byte


def _unpack_e4b15_byte(byte: torch.Tensor) -> torch.Tensor:
    """uint8 fp8e4b15 byte → fp32. Used for ground-truth dequant."""
    sign = ((byte >> 7) & 1).to(torch.float32)
    exp_b = ((byte >> 3) & 0x0F).to(torch.int32)
    mant = (byte & 0x07).to(torch.float32)
    exp_unb = exp_b - 15
    val = (1.0 + mant / 8.0) * torch.pow(2.0, exp_unb.to(torch.float32))
    val = torch.where(exp_b == 0, torch.zeros_like(val), val)
    return torch.where(sign > 0, -val, val)


def build_realistic_cache(
    B: int, prior_S: int, K1: int,
    Hk: int, D: int, BLOCK_SIZE: int,
    fp8_e4b15: bool = True,
):
    """Build vLLM-shaped TQ cache (num_blocks, BLOCK_SIZE, Hk, slot_size).

    Slot layout (kps=D for full FP8 key, val_data_bytes=D//2 for 4-bit V):
      [0..D)              FP8 K
      [D..D+D//2)         4-bit V indices (2 nibbles per byte)
      [D+D//2..D+D//2+2)  V scale (FP16 LE)
      [D+D//2+2..+4)      V zero  (FP16 LE)
    slot_size = D + D//2 + 4

    Returns:
      kv_cache, k_dequant, v_dequant, block_table, layout_dict
    """
    KPS = D
    VDB = D // 2
    SLOT = KPS + VDB + 4

    # Total positions to store in cache = prior + K_PLUS_1 (vLLM stores chunk too)
    total_S = prior_S + K1
    num_blocks_per_req = (total_S + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = max(1, num_blocks_per_req * B)

    kv_cache = torch.zeros((num_blocks, BLOCK_SIZE, Hk, SLOT), dtype=torch.uint8)
    block_table = torch.arange(num_blocks_per_req * B, dtype=torch.int32).view(B, -1)

    # Build raw K/V values
    g = torch.Generator().manual_seed(11)
    k_raw_full = torch.randn((B, total_S, Hk, D), generator=g, dtype=torch.float32) * 0.3
    v_raw_full = torch.randn((B, total_S, Hk, D), generator=g, dtype=torch.float32) * 0.3

    # Build dequant ground truth (after compress→decompress round-trip)
    k_dequant_full = torch.zeros_like(k_raw_full)
    v_dequant_full = torch.zeros_like(v_raw_full)

    for b in range(B):
        for pos in range(total_S):
            block_idx = pos // BLOCK_SIZE
            slot_idx = pos % BLOCK_SIZE
            cache_block = block_table[b, block_idx].item()
            for h in range(Hk):
                # K: FP8 e4b15 round-trip
                k_vals = k_raw_full[b, pos, h]
                k_bytes = _pack_e4b15_byte(k_vals)
                kv_cache[cache_block, slot_idx, h, :KPS] = k_bytes
                k_dequant_full[b, pos, h] = _unpack_e4b15_byte(k_bytes)
                # V: 4-bit per-vector quantization
                v_vals = v_raw_full[b, pos, h]
                v_min = v_vals.min().item()
                v_max = v_vals.max().item()
                v_scale = max((v_max - v_min) / 15.0, 1e-6)
                v_zero = v_min
                v_idx = ((v_vals - v_zero) / v_scale).round().clamp(0, 15).to(torch.uint8)
                # Pack 2 nibbles per byte
                packed = (v_idx[0::2] | (v_idx[1::2] << 4)).to(torch.uint8)
                kv_cache[cache_block, slot_idx, h, KPS:KPS + VDB] = packed
                # Scale + zero as FP16 little-endian
                scale_fp16 = torch.tensor([v_scale], dtype=torch.float16)
                zero_fp16 = torch.tensor([v_zero], dtype=torch.float16)
                kv_cache[cache_block, slot_idx, h, KPS + VDB:KPS + VDB + 2] = scale_fp16.view(torch.uint8)
                kv_cache[cache_block, slot_idx, h, KPS + VDB + 2:KPS + VDB + 4] = zero_fp16.view(torch.uint8)
                # Dequant ground truth
                v_dequant_full[b, pos, h] = (v_idx.to(torch.float32) * v_scale + v_zero)

    layout = {"KPS": KPS, "VDB": VDB, "SLOT": SLOT, "BLOCK_SIZE": BLOCK_SIZE}
    return kv_cache, k_dequant_full, v_dequant_full, block_table, layout


def reference_attention(q, k_dequant, v_dequant, prior_S, K1, scale, kv_group_size):
    """Multi-query causal attention reference using DEQUANTIZED full cache.

    q: [B, K1, Hq, D]
    k_dequant, v_dequant: [B, prior_S + K1, Hk, D]
    Phase boundary at prior_S — all K/V <= prior_S+t reachable from q[t].
    """
    B, K1_q, Hq, D = q.shape
    Hk = k_dequant.shape[2]
    out = torch.zeros((B, K1, Hq, D), dtype=torch.float32)
    for b in range(B):
        for t in range(K1):
            q_pos = prior_S + t  # absolute position of this query token
            for h in range(Hq):
                hk = h // kv_group_size
                # Build K/V for [0, q_pos+1) — causal
                k = k_dequant[b, :q_pos + 1, hk, :]   # (q_pos+1, D)
                v = v_dequant[b, :q_pos + 1, hk, :]
                qq = q[b, t, h, :].to(torch.float32)
                scores = (qq.unsqueeze(0) * k.to(torch.float32)).sum(-1) * scale
                w = torch.softmax(scores, dim=0)
                out[b, t, h, :] = (w.unsqueeze(-1) * v.to(torch.float32)).sum(0)
    return out


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
    from vllm._genesis.kernels.p67_multi_query_kernel import (
        call_p67_attention, diagnostic_info,
    )
    print(f"Diag: {diagnostic_info()}")

    device = "cuda"
    cap = torch.cuda.get_device_capability()
    fp8_e4b15 = cap < (8, 9)
    print(f"GPU SM={cap} fp8_mode={'e4b15' if fp8_e4b15 else 'e4nv'}")

    # Production-shape tests
    test_cases = [
        # (B, prior_S, K1, Hq, Hk, D, BLOCK_SIZE, name)
        (1, 16,  4, 64, 4, 128, 2832, "prod_small_S16_BS2832"),
        (1, 256, 4, 64, 4, 128, 2832, "prod_S256_BS2832"),
        (2, 16,  4, 64, 4, 128, 2832, "prod_B2_S16_BS2832"),
    ]

    failures = []
    for B, prior_S, K1, Hq, Hk, D, BLOCK_SIZE, name in test_cases:
        print(f"\n=== {name} ===")
        g = torch.Generator().manual_seed(42)
        q_in = torch.randn((B, K1, Hq, D), generator=g, dtype=torch.float16) * 0.3
        kv_cache, k_dequant_full, v_dequant_full, block_table, layout = build_realistic_cache(
            B, prior_S, K1, Hk, D, BLOCK_SIZE, fp8_e4b15=fp8_e4b15
        )

        # k_chunk/v_chunk = the K_PLUS_1 chunk in fp16 (raw K/V before quantization)
        # In production, this is what the model passes as `key`/`value`.
        # Use the ALREADY-DEQUANTIZED version of the chunk so reference matches kernel.
        k_chunk = k_dequant_full[:, prior_S:prior_S + K1, :, :].to(torch.float16)
        v_chunk = v_dequant_full[:, prior_S:prior_S + K1, :, :].to(torch.float16)

        scale = 1.0 / math.sqrt(D)
        kv_group_size = Hq // Hk

        # Reference: full causal attention
        ref = reference_attention(q_in, k_dequant_full, v_dequant_full, prior_S, K1, scale, kv_group_size)

        # Kernel
        seq_lens_kernel = torch.tensor([prior_S + K1] * B, dtype=torch.int32)
        try:
            out = call_p67_attention(
                q=q_in.to(device),
                kv_cache=kv_cache.to(device),
                block_table=block_table.to(device),
                seq_lens=seq_lens_kernel.to(device),
                k_chunk=k_chunk.to(device),
                v_chunk=v_chunk.to(device),
                scale=scale,
                block_size=BLOCK_SIZE,
                kps=layout["KPS"],
                val_data_bytes=layout["VDB"],
            )
        except Exception as e:
            print(f"  FAIL: kernel raised {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failures.append(name)
            continue

        out_cpu = out.cpu().to(torch.float32)
        has_nan = bool(torch.isnan(out_cpu).any())
        has_inf = bool(torch.isinf(out_cpu).any())
        diff = (out_cpu - ref).abs()
        ref_max = float(ref.abs().max())
        max_abs = float(diff.max())
        rel_avg = float(diff.mean() / max(ref.abs().mean().item(), 1e-6))

        status = "PASS" if (not has_nan and not has_inf and rel_avg < 0.20) else "FAIL"
        print(f"  {status}: nan={has_nan} inf={has_inf} max_abs={max_abs:.4e} ref_max={ref_max:.3f} rel_avg={rel_avg:.4f}")
        if status == "FAIL":
            failures.append(name)

    print(f"\n{len(test_cases) - len(failures)}/{len(test_cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

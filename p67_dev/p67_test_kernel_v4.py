"""TDD test for V4 — TurboQuant compressed cache read.

Builds synthetic compressed cache from random K/V, dequants it back, then
verifies V4(compressed) ≈ reference_layer3(dequant(compressed)).
"""
from __future__ import annotations
import math
import sys


def e4b15_byte_to_fp32_tensor(byte_tensor):
    """Decode uint8 tensor as FP8 e4b15 (Ampere/Ada format) → fp32 tensor.

    Format: 1 sign + 4 exp + 3 mantissa, bias = 15.
    Mirrors what `tl.load(...).to(tl.float8e4b15, bitcast=True).to(tl.float32)`
    produces in Triton. Used for reference dequant in test fixtures.
    """
    import torch
    b = byte_tensor.to(torch.int32)
    sign = ((b >> 7) & 1).to(torch.float32) * -2 + 1  # 0→+1, 1→-1
    exp = (b >> 3) & 0xF
    mant = (b & 0x7).to(torch.float32) / 8.0

    # Subnormal: exp == 0
    sub_val = mant * (2.0 ** -14)
    # Normal: exp != 0
    norm_val = (1.0 + mant) * torch.pow(2.0, (exp.to(torch.float32) - 15))

    val = torch.where(exp == 0, sub_val, norm_val)
    val = sign * val
    # Mask out NaN-encoding (exp=0xF, mant=0x7) — treat as 0
    is_nan = (exp == 0xF) & ((b & 0x7) == 0x7)
    val = torch.where(is_nan, torch.zeros_like(val), val)
    return val


def fp32_to_e4b15_byte_tensor(x_tensor):
    """Quantize fp32 → uint8 byte representing FP8 e4b15.

    Round-to-nearest. Saturate at e4b15 max (~120). Negative zero collapses to +0.
    """
    import torch
    x = x_tensor.to(torch.float32).clone()

    sign_bit = (x < 0).to(torch.int32)
    x_abs = x.abs()

    # Treat exact zero
    is_zero = x_abs < 1e-30

    # Compute exp + mant
    # value = (1 + m/8) * 2^(e-15)
    # log2(value) = e - 15 + log2(1 + m/8)
    # e_real = floor(log2(value)) + 15
    log2_val = torch.log2(x_abs.clamp_min(1e-30))
    e_real = torch.floor(log2_val) + 15
    e_real = e_real.clamp(0, 15).to(torch.int32)

    # Extract normalized mantissa
    pow_e = torch.pow(2.0, (e_real.to(torch.float32) - 15))
    mant_norm = x_abs / pow_e - 1.0  # in [0, 1)
    mant_norm = mant_norm.clamp(0, 0.875)  # max representable: 7/8
    mant_int = (mant_norm * 8.0).round().to(torch.int32).clamp(0, 7)

    # If exp is 0, treat as subnormal: value = (mant/8) * 2^(1-15) = mant/8 * 2^-14
    # mant_sub = round(value * 2^14 * 8) = round(value * 2^17)
    mant_sub = (x_abs * (2.0 ** 17)).round().to(torch.int32).clamp(0, 7)
    mant_int = torch.where(e_real == 0, mant_sub, mant_int)

    byte = (sign_bit << 7) | (e_real << 3) | mant_int
    byte = torch.where(is_zero, torch.zeros_like(byte), byte)

    return byte.to(torch.uint8)


def build_compressed_cache_and_dequant(k_fp16, v_fp16, block_size, fp8_e4b15=True):
    """Build TurboQuant k8v4 compressed cache from FP16 K/V.

    Returns (kv_cache_bytes, k_dequant_fp16, v_dequant_fp16, layout_meta).
    Layout per slot: [KPS bytes K (FP8)] [VDB bytes V_idx (4-bit)] [2 V_scale] [2 V_zero]
    Cache shape: [num_blocks, block_size, Hk, slot_size]
    """
    import torch
    B, S, Hk, D = k_fp16.shape
    assert v_fp16.shape == (B, S, Hk, D)
    assert B == 1, "test only supports B=1 for now (one block_table)"

    KPS = D                  # FP8 K = 1 byte per dim
    VDB = D // 2             # 4-bit V = 0.5 bytes per dim
    SLOT_SIZE = KPS + VDB + 4  # +2 for scale, +2 for zero

    # Number of blocks needed
    num_blocks = (S + block_size - 1) // block_size

    # Allocate cache buffer
    kv_cache = torch.zeros((num_blocks, block_size, Hk, SLOT_SIZE), dtype=torch.uint8)

    # K dequant target (will be FP8-rounded version of input)
    k_dequant = torch.zeros_like(k_fp16, dtype=torch.float32)
    v_dequant = torch.zeros_like(v_fp16, dtype=torch.float32)

    # Block table: positions [0, S) → blocks [0, num_blocks)
    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, -1)

    for s in range(S):
        page_idx = s // block_size
        page_off = s % block_size
        for h in range(Hk):
            slot_view = kv_cache[page_idx, page_off, h]  # [SLOT_SIZE] uint8

            # K: fp16 → e4b15 byte → re-decode (exact bit-level match with kernel)
            k_vec = k_fp16[0, s, h].to(torch.float32)
            if fp8_e4b15:
                k_bytes = fp32_to_e4b15_byte_tensor(k_vec)
                slot_view[:KPS] = k_bytes
                k_dequant[0, s, h] = e4b15_byte_to_fp32_tensor(k_bytes)
            else:
                # e4nv = PyTorch float8_e4m3fn
                k_fp8 = k_vec.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                slot_view[:KPS] = k_fp8.view(torch.uint8)
                k_dequant[0, s, h] = k_fp8.to(torch.float32)

            # V: 4-bit min-max quant per slot
            v_vec = v_fp16[0, s, h].to(torch.float32)
            v_min, v_max = v_vec.min().item(), v_vec.max().item()
            scale = (v_max - v_min) / 15.0 if v_max > v_min else 1.0
            zero = v_min  # idx 0 maps to v_min
            v_idx_f = ((v_vec - zero) / scale).round().clamp(0, 15)
            v_idx_int = v_idx_f.to(torch.int32)

            # Pack 2 nibbles per byte
            for d_byte in range(VDB):
                lo = int(v_idx_int[2 * d_byte].item())
                hi = int(v_idx_int[2 * d_byte + 1].item())
                slot_view[KPS + d_byte] = (lo & 0xF) | ((hi & 0xF) << 4)

            # Scale (FP16 little-endian)
            scale_fp16 = torch.tensor([scale], dtype=torch.float16)
            scale_bytes = scale_fp16.view(torch.uint8)
            slot_view[KPS + VDB] = scale_bytes[0]
            slot_view[KPS + VDB + 1] = scale_bytes[1]

            # Zero (FP16 little-endian)
            zero_fp16 = torch.tensor([zero], dtype=torch.float16)
            zero_bytes = zero_fp16.view(torch.uint8)
            slot_view[KPS + VDB + 2] = zero_bytes[0]
            slot_view[KPS + VDB + 3] = zero_bytes[1]

            # V dequant target
            v_dequant[0, s, h] = v_idx_f * scale + zero

    return kv_cache, k_dequant.to(torch.float16), v_dequant.to(torch.float16), block_table, dict(KPS=KPS, VDB=VDB, SLOT_SIZE=SLOT_SIZE)


def main():
    try:
        import torch
        if not torch.cuda.is_available():
            print("SKIP: CUDA not available"); return 0
    except Exception as e:
        print(f"SKIP: {e}"); return 0

    from p67_reference import reference_multi_query_attention_layer3
    from p67_kernel_v4 import p67_v4_compressed

    device = "cuda"

    # Detect arch (e4b15 vs e4nv)
    cap = torch.cuda.get_device_capability()
    fp8_e4b15 = 1 if cap < (8, 9) else 0
    mode_name = "e4b15 (Ampere/Ada custom decoder)" if fp8_e4b15 else "e4nv (PyTorch native)"
    print(f"GPU SM={cap}, FP8 mode: {mode_name}")

    test_cases = [
        # (B, K_PLUS_1, Hq, Hk, D, S_prior, block_size, name)
        (1, 4, 16,  4, 64,  32, 16, "B1_T4_GQA4_D64_S32_BS16"),
        (1, 4, 64,  4, 128, 64, 16, "B1_T4_Hq64_GQA16_D128_S64_BS16_prod"),
        (1, 4, 64,  4, 128, 256, 32, "B1_T4_Hq64_GQA16_D128_S256_BS32_prod"),
    ]

    failures = []
    for B, K1, Hq, Hk, D, S, BS, name in test_cases:
        # Generate random uncompressed K/V (smaller magnitudes for fp8 fit)
        g = torch.Generator().manual_seed(7)
        q = torch.randn((B, K1, Hq, D), generator=g, dtype=torch.float16) * 0.5
        k_raw = torch.randn((B, S, Hk, D), generator=g, dtype=torch.float16) * 0.5
        v_raw = torch.randn((B, S, Hk, D), generator=g, dtype=torch.float16) * 0.5
        k_chunk = torch.randn((B, K1, Hk, D), generator=g, dtype=torch.float16) * 0.5
        v_chunk = torch.randn((B, K1, Hk, D), generator=g, dtype=torch.float16) * 0.5

        # Build compressed cache + get dequant ground truth
        kv_cache, k_dequant, v_dequant, block_table, layout = (
            build_compressed_cache_and_dequant(k_raw, v_raw, block_size=BS, fp8_e4b15=bool(fp8_e4b15))
        )

        seq_lens = torch.tensor([S], dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        kv_group_size = Hq // Hk

        # Reference: Layer 3 with dequant ground-truth K/V
        ref = reference_multi_query_attention_layer3(
            q, k_dequant, v_dequant, k_chunk, v_chunk,
            scale, kv_group_size, seq_lens
        )

        # V4: compressed read
        try:
            out = p67_v4_compressed(
                q.to(device), kv_cache.to(device), block_table.to(device),
                seq_lens.to(device), k_chunk.to(device), v_chunk.to(device),
                scale=scale, block_size=BS,
                kps=layout["KPS"], val_data_bytes=layout["VDB"],
                fp8_e4b15=fp8_e4b15,
            )
        except Exception as e:
            print(f"  ✗ {name}: kernel raised {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failures.append(name); continue

        out_cpu = out.cpu()
        diff = (out_cpu.to(torch.float32) - ref.to(torch.float32)).abs()
        max_diff = diff.max().item()
        # Use rel_diff with denominator floor at output magnitude (not 1e-3)
        # to avoid blow-up at near-zero values
        ref_abs_max = ref.to(torch.float32).abs().max().item()
        rel_diff_avg = (diff.mean() / (ref.to(torch.float32).abs().mean() + 1e-3)).item()
        # Tolerance: fp8 + 4-bit quant noise dominates; check abs is small relative
        # to peak output magnitude (typical attn output ~0.5-2.0)
        ok = max_diff < 0.3 and rel_diff_avg < 0.10

        status = "✓" if ok else "✗"
        print(f"  {status} {name:55s} max_abs={max_diff:.4e} ref_max={ref_abs_max:.3f} rel_avg={rel_diff_avg:.4f}")
        if not ok: failures.append(name)

    print()
    print(f"{len(test_cases) - len(failures)}/{len(test_cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

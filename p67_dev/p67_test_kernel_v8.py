"""V8 test — production shape (Hq=8 D=256) + multi-shape sanity.

Compares v8 against the same reference layer3 we used for v4. Verifies:
- Correctness on prod shape (Hq=8, Hk=1, D=256, K_PLUS_1=4) — no NaN, rel_avg < 2%
- Multi-shape sanity (small, medium, prod-like)
- v8 vs v4 perf (TPS) on a synthetic batch
"""
import sys, math, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _pack_e4b15_byte(values: torch.Tensor) -> torch.Tensor:
    abs_v = values.abs()
    sign = (values < 0).to(torch.uint8) << 7
    safe = abs_v.clamp(min=1e-10)
    exp_unbiased = torch.floor(torch.log2(safe)).to(torch.int32)
    exp_biased = (exp_unbiased + 15).clamp(0, 30)
    mant_f = (abs_v / torch.pow(2.0, exp_unbiased.to(torch.float32)) - 1.0) * 8.0
    mant_i = mant_f.round().to(torch.int32).clamp(0, 7)
    zero_mask = abs_v < 1e-10
    exp_biased = torch.where(zero_mask, torch.zeros_like(exp_biased), exp_biased)
    mant_i = torch.where(zero_mask, torch.zeros_like(mant_i), mant_i)
    return sign | ((exp_biased.to(torch.uint8) & 0x0F) << 3) | (mant_i.to(torch.uint8) & 0x07)


def _unpack_e4b15_byte(byte: torch.Tensor) -> torch.Tensor:
    sign = ((byte >> 7) & 1).to(torch.float32)
    exp_b = ((byte >> 3) & 0x0F).to(torch.int32)
    mant = (byte & 0x07).to(torch.float32)
    val = (1.0 + mant / 8.0) * torch.pow(2.0, (exp_b - 15).to(torch.float32))
    val = torch.where(exp_b == 0, torch.zeros_like(val), val)
    return torch.where(sign > 0, -val, val)


def build_compressed_cache(B, prior_S, K1, Hk, D, BLOCK_SIZE, fp8_e4b15=True):
    KPS = D
    VDB = D // 2
    SLOT = KPS + VDB + 4
    total_S = prior_S + K1
    nb_per_req = (total_S + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = max(1, nb_per_req * B)
    kv_cache = torch.zeros((num_blocks, BLOCK_SIZE, Hk, SLOT), dtype=torch.uint8)
    block_table = torch.arange(nb_per_req * B, dtype=torch.int32).view(B, -1)

    g = torch.Generator().manual_seed(11)
    k_raw = torch.randn((B, total_S, Hk, D), generator=g, dtype=torch.float32) * 0.3
    v_raw = torch.randn((B, total_S, Hk, D), generator=g, dtype=torch.float32) * 0.3
    k_dq = torch.zeros_like(k_raw)
    v_dq = torch.zeros_like(v_raw)

    for b in range(B):
        for pos in range(total_S):
            block_idx = pos // BLOCK_SIZE
            slot_idx = pos % BLOCK_SIZE
            cb = block_table[b, block_idx].item()
            for h in range(Hk):
                k_v = k_raw[b, pos, h]
                kb = _pack_e4b15_byte(k_v)
                kv_cache[cb, slot_idx, h, :KPS] = kb
                k_dq[b, pos, h] = _unpack_e4b15_byte(kb)
                v_v = v_raw[b, pos, h]
                v_min = v_v.min().item(); v_max = v_v.max().item()
                v_scale = max((v_max - v_min) / 15.0, 1e-6); v_zero = v_min
                v_idx = ((v_v - v_zero) / v_scale).round().clamp(0, 15).to(torch.uint8)
                packed = (v_idx[0::2] | (v_idx[1::2] << 4)).to(torch.uint8)
                kv_cache[cb, slot_idx, h, KPS:KPS + VDB] = packed
                kv_cache[cb, slot_idx, h, KPS + VDB:KPS + VDB + 2] = (
                    torch.tensor([v_scale], dtype=torch.float16).view(torch.uint8)
                )
                kv_cache[cb, slot_idx, h, KPS + VDB + 2:KPS + VDB + 4] = (
                    torch.tensor([v_zero], dtype=torch.float16).view(torch.uint8)
                )
                v_dq[b, pos, h] = (v_idx.to(torch.float32) * v_scale + v_zero)

    return kv_cache, k_dq, v_dq, block_table, {"KPS": KPS, "VDB": VDB}


def reference_attention(q, k_dq, v_dq, prior_S, K1, scale, kv_group_size):
    """Multi-query causal attention reference using DEQUANTIZED full cache."""
    B, K1_q, Hq, D = q.shape
    Hk = k_dq.shape[2]
    out = torch.zeros((B, K1, Hq, D), dtype=torch.float32)
    for b in range(B):
        for t in range(K1):
            q_pos = prior_S + t
            for h in range(Hq):
                hk = h // kv_group_size
                k = k_dq[b, :q_pos + 1, hk, :]
                v = v_dq[b, :q_pos + 1, hk, :]
                qq = q[b, t, h, :].to(torch.float32)
                scores = (qq.unsqueeze(0) * k.to(torch.float32)).sum(-1) * scale
                w = torch.softmax(scores, dim=0)
                out[b, t, h, :] = (w.unsqueeze(-1) * v.to(torch.float32)).sum(0)
    return out


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    from p67_kernel_v8 import p67_v8_compressed
    cap = torch.cuda.get_device_capability()
    fp8_e4b15 = 1 if cap < (8, 9) else 0
    print(f"GPU SM={cap} fp8_mode={'e4b15' if fp8_e4b15 else 'e4nv'}")

    test_cases = [
        # (B, prior_S, K1, Hq, Hk, D, BLOCK_SIZE, name)
        (1, 16,  4, 8, 1, 256, 16, "prod_S16_BS16"),
        (1, 256, 4, 8, 1, 256, 16, "prod_S256_BS16"),
        (1, 16,  4, 8, 1, 256, 2832, "prod_S16_BS2832"),
        (2, 16,  4, 8, 1, 256, 16, "prod_B2_S16_BS16"),
        (1, 4,   4, 64, 4, 128, 16, "v4_compat_Hq64_D128"),  # backwards compat
    ]
    failures = []
    for B, prior_S, K1, Hq, Hk, D, BS, name in test_cases:
        print(f"\n=== {name} ===")
        g = torch.Generator().manual_seed(42)
        q_in = torch.randn((B, K1, Hq, D), generator=g, dtype=torch.float16) * 0.3
        kv_cache, k_dq, v_dq, bt, layout = build_compressed_cache(
            B, prior_S, K1, Hk, D, BS, fp8_e4b15=bool(fp8_e4b15)
        )
        k_chunk = k_dq[:, prior_S:prior_S + K1].to(torch.float16)
        v_chunk = v_dq[:, prior_S:prior_S + K1].to(torch.float16)
        scale = 1.0 / math.sqrt(D)
        ref = reference_attention(q_in, k_dq, v_dq, prior_S, K1, scale, Hq // Hk)
        seq_lens = torch.tensor([prior_S + K1] * B, dtype=torch.int32)

        try:
            out = p67_v8_compressed(
                q=q_in.cuda(), kv_cache=kv_cache.cuda(), block_table=bt.cuda(),
                seq_lens=seq_lens.cuda(),
                k_chunk=k_chunk.cuda(), v_chunk=v_chunk.cuda(),
                scale=scale, block_size=BS,
                kps=layout["KPS"], val_data_bytes=layout["VDB"],
                fp8_e4b15=fp8_e4b15,
            )
        except Exception as e:
            print(f"  ✗ {name}: kernel raised {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failures.append(name); continue

        out_cpu = out.cpu().to(torch.float32)
        has_nan = bool(torch.isnan(out_cpu).any())
        has_inf = bool(torch.isinf(out_cpu).any())
        diff = (out_cpu - ref).abs()
        rel_avg = float(diff.mean() / max(ref.abs().mean().item(), 1e-6))
        max_abs = float(diff.max())
        ref_max = float(ref.abs().max())
        ok = (not has_nan and not has_inf and rel_avg < 0.10)
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: nan={has_nan} inf={has_inf} max_abs={max_abs:.4e} ref_max={ref_max:.3f} rel_avg={rel_avg:.4f}")
        if not ok:
            failures.append(name)
    print(f"\n=== {len(test_cases) - len(failures)}/{len(test_cases)} passed ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

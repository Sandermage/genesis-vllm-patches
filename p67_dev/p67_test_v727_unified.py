"""V7.27 unified-loop test — verify correctness of single-loop variant.

Compares against same reference as v8 (multi-query causal attention against
dequantized full cache). Should produce identical-or-better accuracy since
all positions read from same source (FP8 quantized).

Production wrapper test: 3 prod-shape cases (Hq=8, D=256, BS=2832).
"""
import sys, math, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p67_test_wrapper_e2e import build_realistic_cache, reference_attention


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    # Force PYTHONPATH to find _genesis
    sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
    from vllm._genesis.kernels.p67_multi_query_kernel import (
        call_p67_attention, diagnostic_info,
    )
    print(f"Diag: {diagnostic_info()}")

    cap = torch.cuda.get_device_capability()
    fp8_e4b15 = cap < (8, 9)
    print(f"GPU SM={cap} fp8_mode={'e4b15' if fp8_e4b15 else 'e4nv'}")

    test_cases = [
        # Production shape: Hq=8 Hk=1 D=256 K_PLUS_1=4 BLOCK_SIZE=2832
        (1, 16,  4, 8, 1, 256, 2832, "prod_S16"),
        (1, 256, 4, 8, 1, 256, 2832, "prod_S256"),
        (2, 16,  4, 8, 1, 256, 2832, "prod_B2_S16"),
        # Larger context to test loop iterations
        (1, 1024, 4, 8, 1, 256, 2832, "prod_S1024"),
    ]

    failures = []
    for B, prior_S, K1, Hq, Hk, D, BS, name in test_cases:
        print(f"\n=== {name} ===")
        g = torch.Generator().manual_seed(42)
        q_in = torch.randn((B, K1, Hq, D), generator=g, dtype=torch.float16) * 0.3
        kv_cache, k_dq, v_dq, bt, layout = build_realistic_cache(
            B, prior_S, K1, Hk, D, BS, fp8_e4b15=fp8_e4b15
        )
        # k_chunk/v_chunk no longer used by v7.27, but pass dequant for API compat
        k_chunk = k_dq[:, prior_S:prior_S + K1].to(torch.float16)
        v_chunk = v_dq[:, prior_S:prior_S + K1].to(torch.float16)

        scale = 1.0 / math.sqrt(D)
        ref = reference_attention(q_in, k_dq, v_dq, prior_S, K1, scale, Hq // Hk)
        seq_lens = torch.tensor([prior_S + K1] * B, dtype=torch.int32)

        try:
            out = call_p67_attention(
                q=q_in.cuda(), kv_cache=kv_cache.cuda(), block_table=bt.cuda(),
                seq_lens=seq_lens.cuda(),
                k_chunk=k_chunk.cuda(), v_chunk=v_chunk.cuda(),
                scale=scale, block_size=BS,
                kps=layout["KPS"], val_data_bytes=layout["VDB"],
            )
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failures.append(name)
            continue

        out_cpu = out.cpu().to(torch.float32)
        has_nan = bool(torch.isnan(out_cpu).any())
        has_inf = bool(torch.isinf(out_cpu).any())
        diff = (out_cpu - ref).abs()
        rel_avg = float(diff.mean() / max(ref.abs().mean().item(), 1e-6))
        max_abs = float(diff.max())
        ref_max = float(ref.abs().max())
        ok = (not has_nan and not has_inf and rel_avg < 0.10)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}: nan={has_nan} inf={has_inf} max_abs={max_abs:.4e} ref_max={ref_max:.3f} rel_avg={rel_avg:.4f}")
        if not ok:
            failures.append(name)

    print(f"\n=== {len(test_cases) - len(failures)}/{len(test_cases)} passed ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

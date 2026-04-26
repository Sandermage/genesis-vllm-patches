"""P67 long-prior accuracy test — reproduce production drift in isolation.

Tests P67 against reference attention with PROGRESSIVELY LONGER prior contexts:
S=256, 1024, 4096, 8192. Measures rel_avg, max_abs, and per-token output
divergence from reference. Goal: identify if precision drift exists in
standalone, where it kicks in, and quantify it.

Production observed: long-context (6K prior) tool-call output truncates after
2 needle words. If standalone shows similar drift at 4K-8K, we have a
reproducible isolated test for fix iteration.
"""
import sys, math, os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p67_test_wrapper_e2e import build_realistic_cache, reference_attention


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return 0

    sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
    from vllm._genesis.kernels.p67_multi_query_kernel import (
        call_p67_attention, diagnostic_info,
    )
    print(f"Diag: {diagnostic_info()}")

    cap = torch.cuda.get_device_capability()
    fp8_e4b15 = cap < (8, 9)

    # Production shape: Hq=8, Hk=1, D=256, K_PLUS_1=4
    Hq, Hk, D, K1, BS = 8, 1, 256, 4, 2832

    test_cases = [
        # (B, prior_S, name)
        (1, 256,  "prior_256"),
        (1, 1024, "prior_1K"),
        (1, 2048, "prior_2K"),
        (1, 4096, "prior_4K"),
        (1, 6144, "prior_6K_prod"),  # match production observed length
        (1, 8192, "prior_8K"),
        (1, 16384, "prior_16K"),
    ]

    results = []
    for B, prior_S, name in test_cases:
        g = torch.Generator().manual_seed(42)
        # Use realistic activation magnitudes (RMSnorm-like, ~0.5 std)
        q_in = torch.randn((B, K1, Hq, D), generator=g, dtype=torch.float16) * 0.5
        kv_cache, k_dq, v_dq, bt, layout = build_realistic_cache(
            B, prior_S, K1, Hk, D, BS, fp8_e4b15=fp8_e4b15
        )
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
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            results.append((name, prior_S, None, None, None, None))
            continue

        out_cpu = out.cpu().to(torch.float32)
        diff = (out_cpu - ref).abs()
        rel_avg = float(diff.mean() / max(ref.abs().mean().item(), 1e-6))
        max_abs = float(diff.max())
        per_token_max_abs = diff.amax(dim=-1).amax(dim=-1).flatten().tolist()
        per_token_rel = [
            float(d / max(r, 1e-6)) for d, r in zip(
                per_token_max_abs,
                diff.amax(dim=-1).amax(dim=-1).flatten().tolist(),
            )
        ]
        amax = float(out_cpu.abs().max())
        results.append((name, prior_S, rel_avg, max_abs, amax, per_token_max_abs))
        print(f"  {name:18s} prior={prior_S:5d}  rel_avg={rel_avg:.5f}  max_abs={max_abs:.4e}  amax={amax:.3f}")
        print(f"    per_t_max_abs: {[f'{x:.4e}' for x in per_token_max_abs]}")

    # Trend analysis
    print("\n=== DRIFT TREND ===")
    print(f"{'name':18s}  {'prior':>6s}  {'rel_avg':>10s}  {'max_abs':>11s}")
    for name, prior_S, rel_avg, max_abs, amax, per_t in results:
        if rel_avg is None:
            print(f"{name:18s}  {prior_S:6d}  {'FAILED':>10s}")
            continue
        marker = "  ← DRIFT" if rel_avg > 0.01 else ""
        print(f"{name:18s}  {prior_S:6d}  {rel_avg:10.5f}  {max_abs:11.4e}{marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

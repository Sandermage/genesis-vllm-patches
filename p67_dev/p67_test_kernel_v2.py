"""TDD test for P67 kernel v2 — multi-query + GQA against reference Layer 2."""
from __future__ import annotations
import sys


def main():
    try:
        import torch
        if not torch.cuda.is_available():
            print("SKIP: CUDA not available"); return 0
    except Exception as e:
        print(f"SKIP: {e}"); return 0

    from p67_reference import (
        make_synthetic_test_inputs,
        reference_multi_query_attention_layer2,
    )
    from p67_kernel_v2 import p67_v2_multi_query_gqa

    device = "cuda"

    # GQA test cases (Hk < Hq, kv_group_size > 1)
    test_cases = [
        # (B, K_PLUS_1, Hq, Hk, D, S, name)
        (1, 4, 16,  4, 64,  64, "B1_T4_Hq16_Hk4_D64_S64_GQA4"),
        (1, 4, 32,  4, 128, 256, "B1_T4_Hq32_Hk4_D128_S256_GQA8"),
        (1, 4, 64,  4, 128, 290, "B1_T4_Hq64_Hk4_D128_S290_prod_like_GQA16"),
        (2, 4, 16,  4, 64,  128, "B2_T4_Hq16_Hk4_D64_S128_GQA4"),
    ]

    failures = []
    for B, K1, Hq, Hk, D, S, name in test_cases:
        if Hq % Hk != 0:
            continue
        inputs = make_synthetic_test_inputs(
            B=B, K_PLUS_1=K1, Hq=Hq, Hk=Hk, D=D, S_prior=S
        )
        q = inputs["q"].to(device)
        k = inputs["k_cached"].to(device)
        v = inputs["v_cached"].to(device)
        seq_lens = inputs["seq_lens"].to(device)
        scale = inputs["scale"]
        kv_group_size = inputs["kv_group_size"]

        # Reference (CPU)
        ref = reference_multi_query_attention_layer2(
            inputs["q"], inputs["k_cached"], inputs["v_cached"],
            scale, kv_group_size
        )

        # Triton kernel
        try:
            out = p67_v2_multi_query_gqa(q, k, v, seq_lens, scale)
        except Exception as e:
            print(f"  ✗ {name}: kernel raised {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failures.append(name); continue

        out_cpu = out.cpu()
        diff = (out_cpu.to(torch.float32) - ref.to(torch.float32)).abs()
        max_diff = diff.max().item()
        rel_diff = (diff / (ref.to(torch.float32).abs() + 1e-3)).max().item()
        ok = max_diff < 1e-1 and rel_diff < 5e-2

        status = "✓" if ok else "✗"
        print(f"  {status} {name:50s} max_abs={max_diff:.4e} max_rel={rel_diff:.4e}")
        if not ok: failures.append(name)

    print()
    print(f"{len(test_cases) - len(failures)}/{len(test_cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

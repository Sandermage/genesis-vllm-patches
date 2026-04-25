"""TDD test for P67 kernel v1 — compare Triton output vs PyTorch reference Layer 1.

Requires CUDA-capable GPU. On CPU-only host this module skips imports gracefully.

Run on server: python3 p67_test_kernel_v1.py
"""
from __future__ import annotations

import math
import sys


def main():
    try:
        import torch
        if not torch.cuda.is_available():
            print("SKIP: CUDA not available")
            return 0
    except Exception as e:
        print(f"SKIP: {e}")
        return 0

    from p67_reference import (
        make_synthetic_test_inputs,
        reference_multi_query_attention_layer1,
    )
    from p67_kernel_v1 import p67_v1_multi_query_attention

    device = "cuda"

    test_cases = [
        # (B, K_PLUS_1, H, D, S, name)
        (1, 1,  4,  64,  16, "B1_T1_H4_D64_S16_smoke"),
        (1, 4,  8,  64,  64, "B1_T4_H8_D64_S64"),
        (1, 4, 16, 128, 256, "B1_T4_H16_D128_S256"),
        (2, 4,  8,  64, 128, "B2_T4_H8_D64_S128"),
        (1, 4, 32, 128, 290, "B1_T4_H32_D128_S290_prod_like"),
    ]

    failures = []
    for B, K_PLUS_1, H, D, S, name in test_cases:
        # Generate test input (Hk == Hq for v1 — no GQA)
        inputs = make_synthetic_test_inputs(
            B=B, K_PLUS_1=K_PLUS_1, Hq=H, Hk=H, D=D, S_prior=S
        )
        q = inputs["q"].to(device)
        k = inputs["k_cached"].to(device)
        v = inputs["v_cached"].to(device)
        seq_lens = inputs["seq_lens"].to(device)
        scale = inputs["scale"]

        # Reference (CPU)
        ref = reference_multi_query_attention_layer1(
            inputs["q"], inputs["k_cached"], inputs["v_cached"], scale
        )

        # Triton kernel (GPU)
        try:
            out = p67_v1_multi_query_attention(q, k, v, seq_lens, scale)
        except Exception as e:
            print(f"  ✗ {name}: kernel raised {type(e).__name__}: {e}")
            failures.append(name)
            continue

        # Compare (fp16 tolerance — looser due to accumulation)
        out_cpu = out.cpu()
        diff = (out_cpu.to(torch.float32) - ref.to(torch.float32)).abs()
        max_diff = diff.max().item()
        rel_diff = (diff / (ref.to(torch.float32).abs() + 1e-3)).max().item()
        ok = max_diff < 1e-1 and rel_diff < 5e-2

        status = "✓" if ok else "✗"
        print(f"  {status} {name:35s} max_abs={max_diff:.4e} max_rel={rel_diff:.4e}")
        if not ok:
            failures.append(name)

    print()
    print(f"{len(test_cases) - len(failures)}/{len(test_cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

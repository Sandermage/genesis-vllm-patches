# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the KV-cache / VRAM / max-context calculator.

Verifies the math against hand-computed values (the standard transformer KV
formula), GQA / MoE / tensor-parallel handling, and calibration.
"""
from __future__ import annotations

from sndr.product_api.legacy import kv_math


def test_kv_bytes_per_token_matches_formula():
    arch = kv_math.ModelArch(name="t", num_layers=64, num_kv_heads=8, head_dim=128, params_b=27.0, weight_bits=4)
    # 2 (K+V) * layers * kv_heads * head_dim * dtype_bytes
    assert kv_math.kv_bytes_per_token(arch, kv_bytes=2.0) == 2 * 64 * 8 * 128 * 2  # = 262144
    # fp8 KV halves it.
    assert kv_math.kv_bytes_per_token(arch, kv_bytes=1.0) == 131072


def test_weights_bytes_respect_quant_bits():
    arch = kv_math.ModelArch(name="t", num_layers=64, num_kv_heads=8, head_dim=128, params_b=27.0, weight_bits=4)
    # 27e9 params * 4 bits / 8 = 13.5 GB.
    assert kv_math.weights_bytes(arch) == int(27.0e9 * 4 / 8)


def test_estimate_fits_and_breakdown():
    arch = kv_math.ModelArch(name="27b-int4", num_layers=64, num_kv_heads=8, head_dim=128, params_b=27.0, weight_bits=4)
    r = kv_math.estimate(arch, context=8192, concurrency=1, tp=2, kv_bytes=1.0,
                         gpu_count=2, gpu_vram_mib=24564, util=0.90, overhead_mib=1500)
    # Weights are sharded across TP=2 → ~6.75 GB/GPU.
    assert r["weights_per_gpu_mib"] == round(int(27.0e9 * 4 / 8) / 2 / (1024 * 1024))
    assert r["kv_per_gpu_mib"] > 0 and r["total_per_gpu_mib"] > r["weights_per_gpu_mib"]
    assert r["budget_per_gpu_mib"] == round(24564 * 0.90)
    assert isinstance(r["fits"], bool)
    # Max context is positive and consistent: plugging it back roughly saturates.
    assert r["max_context"] > 0


def test_max_context_grows_with_fp8_kv():
    arch = kv_math.ModelArch(name="m", num_layers=64, num_kv_heads=8, head_dim=128, params_b=27.0, weight_bits=4)
    base = dict(concurrency=1, tp=2, gpu_count=2, gpu_vram_mib=24564, util=0.9, overhead_mib=1500, context=8192)
    fp16 = kv_math.estimate(arch, kv_bytes=2.0, **base)["max_context"]
    fp8 = kv_math.estimate(arch, kv_bytes=1.0, **base)["max_context"]
    # Halving KV bytes roughly doubles the achievable context.
    assert fp8 > fp16 * 1.8


def test_moe_weights_are_total_and_sharded_by_tp():
    # MoE: VRAM holds ALL expert weights (dense), sharded by TP (issue #260).
    moe = kv_math.ModelArch(name="35b-a3b", num_layers=64, num_kv_heads=8, head_dim=128,
                            params_b=35.0, weight_bits=8, is_moe=True, active_params_b=3.0)
    r = kv_math.estimate(moe, context=4096, concurrency=1, tp=2, kv_bytes=1.0,
                         gpu_count=2, gpu_vram_mib=24564, util=0.9, overhead_mib=1500)
    # Full 35B (not active 3B) at 8-bit, divided by TP=2.
    assert r["weights_per_gpu_mib"] == round(int(35.0e9 * 8 / 8) / 2 / (1024 * 1024))


def test_calibration_adjusts_overhead():
    arch = kv_math.ModelArch(name="m", num_layers=64, num_kv_heads=8, head_dim=128, params_b=27.0, weight_bits=4)
    # A measured total at a known point implies an overhead we can back out.
    overhead = kv_math.calibrate_overhead(
        arch, measured_total_mib=14000, context=2048, concurrency=1, tp=1, kv_bytes=1.0)
    assert overhead >= 0
    # Re-estimating at that point with the calibrated overhead reproduces ~measured.
    r = kv_math.estimate(arch, context=2048, concurrency=1, tp=1, kv_bytes=1.0,
                         gpu_count=1, gpu_vram_mib=24564, util=1.0, overhead_mib=overhead)
    assert abs(r["total_per_gpu_mib"] - 14000) <= 2


def test_fit_envelope_grid_shape_and_monotonicity():
    arch = kv_math.ModelArch(name="m", num_layers=64, num_kv_heads=8, head_dim=128, params_b=27.0, weight_bits=4)
    grid = kv_math.fit_envelope(arch, contexts=[4096, 32768, 131072], concurrencies=[1, 4],
                                kv_bytes=1.0, tp=2, gpu_vram_mib=24564, util=0.9, overhead_mib=1500)
    assert len(grid) == 2 and len(grid[0]) == 3
    # Headroom shrinks as context grows (more KV) within a concurrency row.
    assert grid[0][0]["headroom_mib"] > grid[0][2]["headroom_mib"]
    # Higher concurrency uses more KV → less headroom at the same context.
    assert grid[1][1]["headroom_mib"] < grid[0][1]["headroom_mib"]


def test_recommend_picks_highest_fidelity_kv_that_fits():
    arch = kv_math.ModelArch(name="27b", num_layers=64, num_kv_heads=8, head_dim=128, params_b=27.0, weight_bits=4)
    # Big context where fp16 won't fit but a smaller KV dtype will.
    recs = kv_math.recommend(arch, target_context=200000, target_concurrency=1, tp=2,
                             gpu_vram_mib=24564, util=0.9, overhead_mib=1500)
    rec = next(r for r in recs if r["recommended"])
    assert rec["fits"] is True
    # Nothing higher-fidelity than the pick fits (it's the best that fits).
    idx = recs.index(rec)
    assert all(not r["fits"] for r in recs[:idx])
    # And the pick is byte-for-byte the highest among the fitting ones.
    assert rec["kv_bytes"] == max(r["kv_bytes"] for r in recs if r["fits"])


def test_sliding_window_shrinks_long_context_kv():
    # Two identical models except one has sliding-window attention on 5/6 layers.
    dense = kv_math.ModelArch(name="d", num_layers=60, num_kv_heads=8, head_dim=128, params_b=20, weight_bits=4)
    swa = kv_math.ModelArch(name="s", num_layers=60, num_kv_heads=8, head_dim=128, params_b=20, weight_bits=4,
                            sliding_window=4096, global_layers=10)
    big = dict(context=131072, concurrency=1, tp=2, kv_bytes=1.0, gpu_vram_mib=24564, util=0.9, overhead_mib=1500)
    kv_dense = kv_math.estimate(dense, **big)["kv_per_gpu_mib"]
    kv_swa = kv_math.estimate(swa, **big)["kv_per_gpu_mib"]
    # 50 of 60 layers cap KV at 4K instead of 131K → far less KV, far more context.
    assert kv_swa < kv_dense / 3
    assert kv_math.estimate(swa, **big)["max_context"] > kv_math.estimate(dense, **big)["max_context"] * 3
    # Below the window, sliding has no effect (both layers count fully).
    small = {**big, "context": 2048}
    assert kv_math.estimate(swa, **small)["kv_per_gpu_mib"] == kv_math.estimate(dense, **small)["kv_per_gpu_mib"]


def test_exact_weight_bytes_override_used():
    arch = kv_math.ModelArch(name="m", num_layers=60, num_kv_heads=16, head_dim=256, params_b=31, weight_bits=4,
                             weights_bytes_total=20936945655)
    # Uses the exact du size, not params×bits.
    assert kv_math.weights_bytes(arch) == 20936945655


def test_known_models_registry_is_usable():
    models = kv_math.known_models()
    assert models and all(m.num_layers > 0 and m.params_b > 0 for m in models.values())

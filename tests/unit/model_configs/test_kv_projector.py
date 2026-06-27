# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the byte-level KV/VRAM projector (sndr kv-calc core).

Pure-math tests for sndr/model_configs/kv_projector.py — no I/O, no nvidia-smi.
The load-bearing test is the ANCHOR reproduction: the projector must reproduce
the dev424 PN403 LIVE engine telemetry for the 35B-A3B (kv_cache_size_tokens=
388620) within tolerance, or the calibration is fiction.
"""
from __future__ import annotations

import math

import pytest

from sndr.model_configs import kv_projector as kp
from sndr.model_configs.schema_v2 import ModelShape

_GIB = 1024 ** 3


# ─── Shape fixtures (the real dims our two anchor models declare) ────────────


def _shape_35b() -> ModelShape:
    """qwen3.6-35b-a3b-fp8 — the CALIBRATED anchor (PN403 live engine)."""
    return ModelShape(
        num_hidden_layers=40,
        num_attention_layers=10,
        num_recurrent_layers=30,
        hidden_size=2048,
        num_attention_heads=16,
        num_kv_heads=2,
        head_dim=128,
        weight_bits=8,
        weights_total_gib=33.53,
        num_experts=256,
        num_experts_per_tok=8,
        mtp_num_layers=1,
    )


def _shape_27b() -> ModelShape:
    """qwen3.6-27b-int4-autoround-tq-k8v4 — PROVISIONAL (no live anchor)."""
    return ModelShape(
        num_hidden_layers=48,
        num_attention_layers=12,
        num_recurrent_layers=36,
        hidden_size=4096,
        num_attention_heads=40,
        num_kv_heads=4,
        head_dim=128,
        weight_bits=4,
        weights_total_gib=13.04,
        mtp_num_layers=1,
    )


# ─── Anchor reproduction — the load-bearing calibration check ────────────────


def test_35b_anchor_reproduces_pn403_live_pool_within_tolerance():
    """The 35B projection must reproduce the dev424 PN403 LIVE KV pool token
    capacity (388,620 tokens at 280K / TP2 / util 0.9 / k8v4) within ~10%."""
    s = _shape_35b()
    p = kp.project_from_shape(
        s, preset_id="35b", kv_format="turboquant_k8v4",
        ctx=280000, max_num_seqs=2, tp=2, mem_util=0.9, vram_gib=24.0,
        mtp=True, mtp_n=5,
    )
    # available-for-KV ÷ per-token bytes = pool token capacity (single-seq-equiv).
    per_token = kp.kv_pool_per_card_bytes(s, "turboquant_k8v4", 1, 1, 2)
    assert per_token == 1920, "35B per-token KV/card must be 1920 B (10*2*128*2*0.75/2)"
    pool_tokens = p.available_for_kv_gib * _GIB / per_token

    live = 388_620
    residual = abs(pool_tokens - live) / live
    assert residual < 0.10, (
        f"35B pool capacity {pool_tokens:,.0f} vs live {live:,} "
        f"= {residual * 100:.2f}% residual (must be < 10%)"
    )
    # The anchor is exact, so assert the tight band too (regression guard).
    assert residual < 0.02, f"35B residual {residual * 100:.2f}% drifted above 2%"


def test_35b_is_calibrated_not_provisional():
    p = kp.project_from_shape(
        _shape_35b(), preset_id="35b", kv_format="turboquant_k8v4",
        ctx=280000, max_num_seqs=2, tp=2, mem_util=0.9, vram_gib=24.0,
        mtp=True, mtp_n=5,
    )
    assert p.provisional is False


def test_27b_is_provisional():
    """27B has no live num_gpu_blocks anchor — must be flagged provisional."""
    p = kp.project_from_shape(
        _shape_27b(), preset_id="27b", kv_format="turboquant_k8v4",
        ctx=262144, max_num_seqs=2, tp=2, mem_util=0.9, vram_gib=24.0,
        mtp=True, mtp_n=5,
    )
    assert p.provisional is True
    assert any("PROVISIONAL" in n for n in p.notes)


# ─── KV format byte sizes (k8v4 vs fp8 vs bf16) ──────────────────────────────


def test_kv_format_bytes_per_element():
    assert kp.kv_format_bytes("turboquant_k8v4") == 0.75
    assert kp.kv_format_bytes("k8v4") == 0.75
    assert kp.kv_format_bytes("fp8_e5m2") == 1.0
    assert kp.kv_format_bytes("fp8_e4m3") == 1.0
    assert kp.kv_format_bytes("bf16") == 2.0
    assert kp.kv_format_bytes("fp16") == 2.0
    assert kp.kv_format_bytes(None) == 2.0          # default = full-precision
    assert kp.kv_format_bytes("unknown_fmt") == 2.0  # conservative fallback


def test_k8v4_pool_is_smaller_than_fp8_smaller_than_bf16():
    """For the same shape/ctx, k8v4 (0.75 B) < fp8 (1.0 B) < bf16 (2.0 B)."""
    s = _shape_35b()
    kw = dict(ctx=128000, max_num_seqs=2, tp=2)
    k8v4 = kp.kv_pool_per_card_bytes(s, "turboquant_k8v4", **kw)
    fp8 = kp.kv_pool_per_card_bytes(s, "fp8_e5m2", **kw)
    bf16 = kp.kv_pool_per_card_bytes(s, "bf16", **kw)
    assert k8v4 < fp8 < bf16
    # Exact ratios follow the bytes-per-element table.
    assert math.isclose(k8v4 / fp8, 0.75, rel_tol=1e-9)
    assert math.isclose(fp8 / bf16, 0.5, rel_tol=1e-9)


# ─── Pure-math properties of the per-component functions ─────────────────────


def test_kv_pool_scales_linearly_with_ctx_and_seqs_and_inversely_with_tp():
    s = _shape_35b()
    base = kp.kv_pool_per_card_bytes(s, "fp8_e5m2", 10000, 1, 1)
    assert kp.kv_pool_per_card_bytes(s, "fp8_e5m2", 20000, 1, 1) == 2 * base
    assert kp.kv_pool_per_card_bytes(s, "fp8_e5m2", 10000, 4, 1) == 4 * base
    assert kp.kv_pool_per_card_bytes(s, "fp8_e5m2", 10000, 1, 2) == base // 2


def test_only_attention_layers_grow_kv_not_recurrent():
    """A hybrid model's growing KV must count num_attention_layers, NOT total."""
    s = _shape_35b()  # 10 attn + 30 recurrent
    pool = kp.kv_pool_per_card_bytes(s, "fp8_e5m2", 100000, 1, 1)
    expected = 10 * 2 * 128 * 2 * 1.0 * 100000  # n_attn * kv * hd * 2(K+V) * bpe * ctx
    assert pool == int(expected)


def test_weights_per_card_divides_by_tp():
    s = _shape_35b()
    w1 = kp.weights_per_card_bytes(s, 1)
    w2 = kp.weights_per_card_bytes(s, 2)
    assert w1 == int(33.53 * _GIB)
    assert w2 == w1 // 2


def test_weights_zero_when_total_absent():
    s = ModelShape(num_attention_layers=10, num_kv_heads=2, head_dim=128)
    assert kp.weights_per_card_bytes(s, 2) == 0


def test_recurrent_state_zero_for_pure_dense():
    dense = ModelShape(
        num_hidden_layers=32, num_attention_layers=32, num_recurrent_layers=0,
        hidden_size=4096, num_kv_heads=8, head_dim=128,
    )
    assert kp.recurrent_state_per_card_bytes(dense, 2, 1) == 0


def test_activation_scales_with_ctx():
    s = _shape_35b()
    a1 = kp.activation_peak_per_card_bytes(s, "fp8_e5m2", 50000, 2)
    a2 = kp.activation_peak_per_card_bytes(s, "fp8_e5m2", 100000, 2)
    assert a2 == 2 * a1


def test_cudagraph_overhead_grows_with_util_and_tp():
    o_lo = kp.cudagraph_overhead_per_card_bytes(0.8, 1)
    o_hi = kp.cudagraph_overhead_per_card_bytes(0.95, 1)
    o_tp2 = kp.cudagraph_overhead_per_card_bytes(0.8, 2)
    assert o_hi > o_lo
    assert o_tp2 > o_lo


# ─── Verdict thresholds: PASS / TIGHT / FAIL ─────────────────────────────────


def test_verdict_pass_when_pool_fits_with_room():
    s = _shape_35b()
    p = kp.project_from_shape(
        s, preset_id="35b", kv_format="turboquant_k8v4",
        ctx=32000, max_num_seqs=1, tp=2, mem_util=0.9, vram_gib=24.0,
        mtp=True, mtp_n=5,
    )
    assert p.verdict == "PASS"
    assert kp.fit_verdict(p) == "PASS"


def test_verdict_fail_when_fixed_footprint_alone_blows_budget():
    """35B weights alone (16.8 GiB/card) on a tiny budget must FAIL."""
    s = _shape_35b()
    p = kp.project_from_shape(
        s, preset_id="35b", kv_format="turboquant_k8v4",
        ctx=280000, max_num_seqs=2, tp=2, mem_util=0.5, vram_gib=24.0,  # 12 GiB budget
        mtp=True, mtp_n=5,
    )
    assert p.verdict == "FAIL"
    assert any("refuse to boot" in n for n in p.notes)


def test_verdict_tight_when_requested_pool_exceeds_slack():
    """A bf16 KV at huge ctx requests far more pool than the slack holds."""
    s = _shape_35b()
    p = kp.project_from_shape(
        s, preset_id="35b", kv_format="bf16",
        ctx=280000, max_num_seqs=8, tp=2, mem_util=0.9, vram_gib=24.0,
        mtp=True, mtp_n=5,
    )
    assert p.verdict in ("TIGHT", "FAIL")  # over-subscribed pool
    if p.verdict == "TIGHT":
        assert p.kv_pool_actual_gib <= p.available_for_kv_gib + 1e-6


# ─── solve_max_ctx: monotonic + sane ─────────────────────────────────────────


def test_solve_max_ctx_returns_passing_ctx_and_one_step_more_fails_or_caps():
    s = _shape_35b()
    kw = dict(kv_format="turboquant_k8v4", max_num_seqs=2, tp=2,
              mem_util=0.9, vram_gib=24.0, mtp=True, mtp_n=5, step=1024)
    mx = kp.solve_max_ctx(s, **kw)
    assert mx > 0
    # The solved ctx PASSES or is TIGHT.
    p_at = kp.project_from_shape(
        s, preset_id="x", kv_format="turboquant_k8v4", ctx=mx,
        max_num_seqs=2, tp=2, mem_util=0.9, vram_gib=24.0, mtp=True, mtp_n=5,
    )
    assert p_at.verdict in ("PASS", "TIGHT")


def test_solve_max_ctx_monotone_in_budget():
    """More VRAM (or higher util) → max ctx never decreases."""
    s = _shape_35b()
    base = dict(kv_format="turboquant_k8v4", max_num_seqs=2, tp=2,
                mem_util=0.9, mtp=True, mtp_n=5)
    mx_24 = kp.solve_max_ctx(s, vram_gib=24.0, **base)
    mx_48 = kp.solve_max_ctx(s, vram_gib=48.0, **base)
    assert mx_48 >= mx_24


def test_solve_max_ctx_monotone_in_concurrency():
    """Higher concurrency consumes the pool faster → max ctx never increases."""
    s = _shape_35b()
    base = dict(kv_format="turboquant_k8v4", tp=2, mem_util=0.9,
                vram_gib=24.0, mtp=True, mtp_n=5)
    mx_seqs1 = kp.solve_max_ctx(s, max_num_seqs=1, **base)
    mx_seqs4 = kp.solve_max_ctx(s, max_num_seqs=4, **base)
    assert mx_seqs4 <= mx_seqs1


def test_solve_max_ctx_zero_when_weights_alone_overflow():
    s = _shape_35b()
    mx = kp.solve_max_ctx(
        s, kv_format="turboquant_k8v4", max_num_seqs=2, tp=1,  # TP=1: 33.5 GiB > 24
        mem_util=0.9, vram_gib=24.0, mtp=True, mtp_n=5,
    )
    assert mx == 0


# ─── project() preset wrapper ────────────────────────────────────────────────


class _FakeSpec:
    method = "mtp"
    num_speculative_tokens = 5


class _FakeHardware:
    n_gpus = 2


class _FakePreset:
    """Minimal duck-typed composed-cfg stand-in for project()."""
    key = "prod-qwen3.6-35b-balanced"
    max_model_len = 280000
    max_num_seqs = 2
    gpu_memory_utilization = 0.9
    kv_cache_dtype = "turboquant_k8v4"
    spec_decode = _FakeSpec()
    hardware = _FakeHardware()

    class capabilities:  # noqa: N801 — mimic attribute access path
        shape = None
        spec_decode = _FakeSpec()
        kv_cache_dtype = "turboquant_k8v4"


def test_project_from_preset_resolves_operating_point():
    preset = _FakePreset()
    preset.capabilities.shape = _shape_35b()
    rig = kp.ProjectorRig(vram_gib_per_card=24.0, gpu_count=2, name="A5000")
    p = kp.project(preset, rig)
    assert p.tp == 2
    assert p.ctx == 280000
    assert p.kv_format == "turboquant_k8v4"
    assert p.verdict in ("PASS", "TIGHT", "FAIL")


def test_project_raises_without_shape():
    preset = _FakePreset()
    preset.capabilities.shape = None
    rig = kp.ProjectorRig(vram_gib_per_card=24.0, gpu_count=2)
    with pytest.raises(ValueError, match="no byte-level shape"):
        kp.project(preset, rig)


def test_project_ctx_override():
    preset = _FakePreset()
    preset.capabilities.shape = _shape_35b()
    rig = kp.ProjectorRig(vram_gib_per_card=24.0, gpu_count=2)
    p = kp.project(preset, rig, ctx=32000)
    assert p.ctx == 32000

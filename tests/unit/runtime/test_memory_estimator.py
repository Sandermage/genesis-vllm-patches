# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.runtime.memory_estimator` — T1.3.

Cover three layers:

  1. Per-component math (estimate_weights / estimate_kv_cache /
     estimate_activations / estimate_cuda_graph_reserve /
     estimate_marlin_scratch) on synthetic ModelShape inputs.
  2. `read_model_shape()` against fabricated config.json files
     (no real models needed — uses tmp_path).
  3. `estimate_for_config()` end-to-end against a synthesized
     ModelConfig-like duck type.

Tests are CPU-only / deterministic; no model downloads or live VRAM probes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sndr.runtime.memory_estimator import (
    DEFAULT_GPU_VRAM_BYTES,
    MemoryComponent,
    MemoryEstimate,
    ModelShape,
    estimate_activations,
    estimate_cuda_graph_reserve,
    estimate_for_config,
    estimate_kv_cache,
    estimate_marlin_scratch,
    estimate_weights,
    lookup_gpu_vram,
    read_model_shape,
    render_waterfall,
)


# ─── Per-component math ─────────────────────────────────────────────────


class TestEstimateWeights:
    def test_no_safetensors_returns_zero(self):
        s = ModelShape(model_path="/fake", weights_size_bytes=None)
        assert estimate_weights(s, tp_size=2) == 0

    def test_divides_by_tp_size(self):
        s = ModelShape(model_path="/fake", weights_size_bytes=8 * (1 << 30))
        assert estimate_weights(s, tp_size=1) == 8 * (1 << 30)
        assert estimate_weights(s, tp_size=2) == 4 * (1 << 30)
        assert estimate_weights(s, tp_size=4) == 2 * (1 << 30)

    def test_clamps_tp_size_to_one(self):
        s = ModelShape(model_path="/fake", weights_size_bytes=4 * (1 << 30))
        assert estimate_weights(s, tp_size=0) == 4 * (1 << 30)


class TestEstimateKVCache:
    def test_qwen3_35b_a3b_fp8_at_64k(self):
        # Qwen3.6-35B-A3B: 64 layers × 8 KV heads × 128 head_dim
        s = ModelShape(
            model_path="/fake",
            n_layers=64, n_heads=64, n_kv_heads=8, head_dim=128,
        )
        # 2 × 64 × 8 × 128 × 65536 × 1 byte / TP=2 = 4 GiB
        bytes_ = estimate_kv_cache(
            s, max_model_len=65536, max_num_seqs=1,
            kv_dtype="fp8_e5m2", tp_size=2,
        )
        assert bytes_ == 2 * 64 * 8 * 128 * 65536 // 2
        assert 3 * (1 << 30) < bytes_ < 5 * (1 << 30)

    def test_dtype_doubles_for_fp16(self):
        s = ModelShape(
            model_path="/fake",
            n_layers=64, n_heads=64, n_kv_heads=8, head_dim=128,
        )
        fp8 = estimate_kv_cache(
            s, max_model_len=65536, kv_dtype="fp8", tp_size=1,
        )
        fp16 = estimate_kv_cache(
            s, max_model_len=65536, kv_dtype="fp16", tp_size=1,
        )
        assert fp16 == fp8 * 2

    def test_n_kv_heads_falls_back_to_n_heads(self):
        s = ModelShape(
            model_path="/fake",
            n_layers=32, n_heads=32, n_kv_heads=None, head_dim=128,
        )
        out = estimate_kv_cache(
            s, max_model_len=4096, kv_dtype="fp16", tp_size=1,
        )
        # = 2 × 32L × 32h × 128 × 4096 × 2B
        assert out == 2 * 32 * 32 * 128 * 4096 * 2

    def test_returns_zero_when_shape_incomplete(self):
        s = ModelShape(model_path="/fake")
        assert estimate_kv_cache(s, max_model_len=4096) == 0


class TestEstimateActivations:
    def test_floor_when_no_hidden(self):
        s = ModelShape(model_path="/fake")
        out = estimate_activations(s)
        assert out == 512 * (1 << 20)

    def test_capped_at_2_gib(self):
        s = ModelShape(model_path="/fake", hidden_size=99999)
        out = estimate_activations(s, max_num_batched_tokens=99999)
        assert out == 2 * (1 << 30)


class TestEstimateCudaGraphReserve:
    def test_low_concurrency_low_band(self):
        assert estimate_cuda_graph_reserve(max_num_seqs=1) == 400 * (1 << 20)

    def test_high_concurrency_capped(self):
        assert estimate_cuda_graph_reserve(max_num_seqs=1024) == 1200 * (1 << 20)

    def test_handles_zero_clamps(self):
        assert estimate_cuda_graph_reserve(max_num_seqs=0) == 400 * (1 << 20)


class TestEstimateMarlinScratch:
    def test_no_scratch_for_fp8(self):
        s = ModelShape(model_path="/fake", quant_method="fp8_e5m2")
        assert estimate_marlin_scratch(s) == 0

    def test_scratch_for_marlin(self):
        s = ModelShape(
            model_path="/fake",
            hidden_size=5120, intermediate_size=17408,
            quant_method="gptq_marlin",
        )
        out = estimate_marlin_scratch(s)
        assert out > 0
        # Should be hidden × intermediate × 2 × 1.2
        assert abs(out - 5120 * 17408 * 2 * 1.2) < 1024

    def test_fallback_when_shapes_missing(self):
        s = ModelShape(model_path="/fake", quant_method="auto_round")
        assert estimate_marlin_scratch(s) == int(1.5 * (1 << 30))

    def test_recognizes_auto_round_aliases(self):
        for q in ("auto_round", "gptq_marlin", "awq_marlin"):
            s = ModelShape(
                model_path="/fake",
                hidden_size=4096, intermediate_size=11008,
                quant_method=q,
            )
            assert estimate_marlin_scratch(s) > 0

    def test_uses_weights_when_available(self):
        """T1.9: when weights_size_bytes + n_layers are known, use the
        per-layer × 1.5× rule from the audit."""
        s = ModelShape(
            model_path="/fake",
            n_layers=64,
            hidden_size=5120, intermediate_size=17408,
            quant_method="gptq_marlin",
            weights_size_bytes=20 * (1 << 30),  # 20 GiB
        )
        out = estimate_marlin_scratch(s)
        # per_layer ≈ 20 GiB / 64 = 320 MiB; × 1.5 × 2 = 960 MiB
        assert out >= 800 * (1 << 20)
        assert out <= 1100 * (1 << 20)

    def test_min_floor_64_mib(self):
        """Even tiny models incur fixed overhead."""
        s = ModelShape(
            model_path="/fake",
            n_layers=2,
            hidden_size=128, intermediate_size=256,
            quant_method="gptq_marlin",
            weights_size_bytes=10 * (1 << 20),  # 10 MiB
        )
        out = estimate_marlin_scratch(s)
        assert out >= 64 * (1 << 20)


class TestMarlinScratchWarns:
    def test_no_warn_for_non_marlin(self):
        from sndr.runtime.memory_estimator import marlin_scratch_warns
        s = ModelShape(model_path="/fake", quant_method="fp8_e5m2")
        warn, msg = marlin_scratch_warns(
            s, free_vram_bytes=24 * (1 << 30),
            weights_bytes=20 * (1 << 30),
        )
        assert warn is False
        assert msg == ""

    def test_warns_when_peak_exceeds_free(self):
        from sndr.runtime.memory_estimator import marlin_scratch_warns
        # Construct a scenario where scratch peak + weights overshoot.
        # Per-layer estimate = (24 GiB / 4 layers) × 1.5 × 2 = 18 GiB.
        # weights (23) + scratch (18) = 41 GiB >> 24 GiB free → warn.
        s = ModelShape(
            model_path="/fake",
            n_layers=4,  # small layer count → big per-layer share
            hidden_size=5120, intermediate_size=17408,
            quant_method="gptq_marlin",
            weights_size_bytes=24 * (1 << 30),
        )
        warn, msg = marlin_scratch_warns(
            s, free_vram_bytes=24 * (1 << 30),
            weights_bytes=23 * (1 << 30),
        )
        assert warn is True
        assert "Marlin repack peak" in msg
        assert "exceeds free VRAM" in msg

    def test_no_warn_when_room_to_spare(self):
        from sndr.runtime.memory_estimator import marlin_scratch_warns
        s = ModelShape(
            model_path="/fake",
            n_layers=32,
            hidden_size=4096, intermediate_size=11008,
            quant_method="gptq_marlin",
            weights_size_bytes=8 * (1 << 30),
        )
        warn, _ = marlin_scratch_warns(
            s, free_vram_bytes=80 * (1 << 30),  # H100 80GB
            weights_bytes=8 * (1 << 30),
        )
        assert warn is False


# ─── read_model_shape ───────────────────────────────────────────────────


class TestReadModelShape:
    def test_returns_empty_when_no_config(self, tmp_path):
        s = read_model_shape(str(tmp_path))
        assert s.n_layers is None
        assert s.head_dim is None

    def test_reads_qwen3_style_config(self, tmp_path):
        cfg = {
            "model_type": "qwen3",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 17408,
            "quantization_config": {"quant_method": "auto_round"},
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        s = read_model_shape(str(tmp_path))
        assert s.n_layers == 64
        assert s.head_dim == 128
        assert s.n_kv_heads == 8
        assert s.intermediate_size == 17408
        assert s.quant_method == "auto_round"

    def test_derives_head_dim_when_missing(self, tmp_path):
        cfg = {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            # head_dim absent — should derive 4096/32 = 128
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        s = read_model_shape(str(tmp_path))
        assert s.head_dim == 128

    def test_merges_text_config_for_multimodal(self, tmp_path):
        cfg = {
            "model_type": "qwen3_5_vl",
            "text_config": {
                "num_hidden_layers": 48,
                "hidden_size": 4096,
                "num_attention_heads": 32,
            },
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        s = read_model_shape(str(tmp_path))
        assert s.n_layers == 48
        assert s.hidden_size == 4096

    def test_scans_safetensors_size(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
        # Create two fake shards
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * 4096)
        (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\x00" * 8192)
        s = read_model_shape(str(tmp_path))
        assert s.weights_size_bytes == 4096 + 8192

    def test_handles_corrupt_json(self, tmp_path):
        (tmp_path / "config.json").write_text("{not valid json")
        # Should not raise
        s = read_model_shape(str(tmp_path))
        assert s.n_layers is None


# ─── lookup_gpu_vram ────────────────────────────────────────────────────


class TestLookupGpuVram:
    def test_exact_match(self):
        assert lookup_gpu_vram("RTX A5000") == 24 * (1 << 30)
        assert lookup_gpu_vram("A100 80GB") == 80 * (1 << 30)

    def test_fuzzy_substring_match(self):
        assert lookup_gpu_vram("a5000") == 24 * (1 << 30)
        assert lookup_gpu_vram("nvidia rtx 4090") == 24 * (1 << 30)

    def test_unknown_returns_default(self):
        assert lookup_gpu_vram("totally-fake-gpu") == DEFAULT_GPU_VRAM_BYTES

    def test_none_returns_default(self):
        assert lookup_gpu_vram(None) == DEFAULT_GPU_VRAM_BYTES


# ─── estimate_for_config (end-to-end with duck-typed cfg) ───────────────


@dataclass
class _FakeHardware:
    gpu_match_keys: list = field(default_factory=lambda: ["RTX A5000"])
    n_gpus: int = 2
    min_vram_per_gpu_mib: int = 0


@dataclass
class _FakeConfig:
    """Minimal duck-type matching ModelConfig surface used by estimator."""
    key: str = "fake-test"
    model_path: str = ""
    hardware: _FakeHardware = field(default_factory=_FakeHardware)
    max_model_len: int = 65536
    max_num_seqs: int = 2
    max_num_batched_tokens: int = 4096
    kv_cache_dtype: str = "fp8_e5m2"


class TestEstimateForConfig:
    def _write_synthetic_model(self, tmp_path: Path, *, with_safetensors: bool):
        cfg = {
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 17408,
            "quantization_config": {"quant_method": "fp8_e5m2"},
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg))
        if with_safetensors:
            (tmp_path / "model.safetensors").write_bytes(
                b"\x00" * (1 * (1 << 30))  # 1 GiB synthetic shard
            )

    def test_full_path_produces_components(self, tmp_path):
        self._write_synthetic_model(tmp_path, with_safetensors=True)
        cfg = _FakeConfig(model_path=str(tmp_path))
        est = estimate_for_config(cfg)
        # 4 components min (weights, KV, activations, CUDA graph), Marlin off for fp8
        assert len(est.components) >= 4
        names = {c.name for c in est.components}
        assert any("weights" in n.lower() for n in names)
        assert any("KV cache" in n for n in names)

    def test_low_confidence_warns_when_no_safetensors(self, tmp_path):
        self._write_synthetic_model(tmp_path, with_safetensors=False)
        cfg = _FakeConfig(model_path=str(tmp_path))
        est = estimate_for_config(cfg)
        assert any("safetensors" in w for w in est.warnings)

    def test_recommendation_when_tight(self, tmp_path):
        self._write_synthetic_model(tmp_path, with_safetensors=True)
        # Force tight: huge ctx + many seqs
        cfg = _FakeConfig(
            model_path=str(tmp_path),
            max_model_len=512 * 1024,  # 512K
            max_num_seqs=4,
        )
        est = estimate_for_config(cfg)
        assert est.utilization > 0.95
        assert any("tight" in r.lower() or "consider" in r.lower()
                   for r in est.recommendations)

    def test_recommendation_when_underused(self, tmp_path):
        self._write_synthetic_model(tmp_path, with_safetensors=True)
        cfg = _FakeConfig(
            model_path=str(tmp_path),
            max_model_len=4096,
            max_num_seqs=1,
        )
        est = estimate_for_config(cfg)
        assert est.utilization < 0.6
        assert any("utilized" in r.lower() or "raise" in r.lower()
                   for r in est.recommendations)

    def test_marlin_scratch_excluded_from_committed_total(self, tmp_path):
        # Set quant to Marlin family so the component appears.
        cfg_dict = {
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "head_dim": 128,
            "intermediate_size": 11008,
            "quantization_config": {"quant_method": "gptq_marlin"},
        }
        (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * (1 << 30))
        cfg = _FakeConfig(model_path=str(tmp_path))
        est = estimate_for_config(cfg)
        names = {c.name for c in est.components}
        assert any("Marlin repack" in n for n in names)
        # total_bytes excludes the transient peak
        marlin_c = next(c for c in est.components if "Marlin" in c.name)
        non_marlin = sum(
            c.bytes_ for c in est.components
            if c.name != marlin_c.name
        )
        assert est.total_bytes == non_marlin


# ─── render_waterfall ───────────────────────────────────────────────────


class TestRenderWaterfall:
    def test_renders_text_no_color(self, tmp_path):
        components = (
            MemoryComponent("Weights", 4 * (1 << 30), confidence="high"),
            MemoryComponent("KV", 6 * (1 << 30), confidence="high"),
        )
        est = MemoryEstimate(
            preset_key="x",
            model_path="/fake",
            gpu_count=2,
            gpu_vram_bytes=24 * (1 << 30),
            components=components,
            warnings=("test warning",),
            recommendations=("test rec",),
        )
        out = render_waterfall(est, use_color=False)
        # No ANSI escapes when color disabled
        assert "\033[" not in out
        assert "Weights" in out
        assert "test warning" in out
        assert "test rec" in out

    def test_total_excludes_marlin(self):
        components = (
            MemoryComponent("Weights", 8 * (1 << 30), confidence="high"),
            MemoryComponent("Marlin repack scratch (peak)",
                            2 * (1 << 30), confidence="medium"),
        )
        est = MemoryEstimate(
            preset_key="x",
            model_path="/fake",
            gpu_count=1,
            gpu_vram_bytes=24 * (1 << 30),
            components=components,
        )
        # Subtotal = 8 GiB (Marlin transient excluded)
        assert est.total_bytes == 8 * (1 << 30)

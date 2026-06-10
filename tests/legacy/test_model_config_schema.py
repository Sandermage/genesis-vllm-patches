# SPDX-License-Identifier: Apache-2.0
"""TDD for sndr.model_configs.schema — comprehensive ModelConfig.

Schema must capture EVERYTHING needed to reproduce + verify a launch:
identity, hardware, model, vLLM args, spec_decode, structured_output,
genesis_env, system_env, vllm_extra_args, docker setup, mounts,
reference metrics, verify tolerances, provenance, notes, lifecycle.

No partial schemas — operator should be able to commit a YAML and
expect "launch + bench + verify" all work.
"""
from __future__ import annotations

import pytest
import yaml as _yaml_check  # ensure pyyaml imports

from sndr.model_configs.schema import (
    ModelConfig,
    ReferenceMetrics,
    VerifyTolerances,
    HardwareSpec,
    SpecDecodeConfig,
    DockerConfig,
    load_yaml,
    dump_yaml,
    validate,
    SchemaError,
)


# ─── Identity ─────────────────────────────────────────────────────────


class TestIdentity:
    def test_minimal_valid_config(self):
        cfg = ModelConfig(
            key="test-config",
            title="Test config",
            description="Minimal test",
            schema_version=1,
            maintainer="testuser",
            model_path="/models/Qwen3.6-35B-A3B-FP8",
            hardware=HardwareSpec(
                gpu_match_keys=["rtx a5000"], n_gpus=2,
                min_vram_per_gpu_mib=24000,
            ),
        )
        assert cfg.key == "test-config"
        assert cfg.schema_version == 1

    def test_key_must_be_kebab_case(self):
        with pytest.raises(SchemaError, match="kebab-case"):
            validate(ModelConfig(
                key="Test_Config",  # underscore not allowed
                title="x", description="x", schema_version=1,
                maintainer="x", model_path="/x",
                hardware=HardwareSpec(gpu_match_keys=["x"], n_gpus=1,
                                       min_vram_per_gpu_mib=1),
            ))

    def test_schema_version_required(self):
        with pytest.raises(SchemaError, match="schema_version"):
            validate(ModelConfig(
                key="x", title="x", description="x",
                schema_version=0,  # invalid
                maintainer="x", model_path="/x",
                hardware=HardwareSpec(gpu_match_keys=["x"], n_gpus=1,
                                       min_vram_per_gpu_mib=1),
            ))


# ─── Hardware ─────────────────────────────────────────────────────────


class TestHardware:
    def test_n_gpus_must_be_positive(self):
        with pytest.raises(SchemaError, match="n_gpus"):
            HardwareSpec(gpu_match_keys=["a"], n_gpus=0,
                         min_vram_per_gpu_mib=1).validate()

    def test_min_vram_required(self):
        with pytest.raises(SchemaError, match="min_vram"):
            HardwareSpec(gpu_match_keys=["a"], n_gpus=1,
                         min_vram_per_gpu_mib=0).validate()


# ─── Reference metrics ────────────────────────────────────────────────


class TestReferenceMetrics:
    def test_reference_metrics_optional(self):
        cfg = _make_min()
        assert cfg.reference_metrics is None

    def test_reference_metrics_full_capture(self):
        rm = ReferenceMetrics(
            measured_at="2026-05-05T18:35:00Z",
            bench_method="bench_35b.sh × 5 sections",
            long_gen_sustained_tps=192.6,
            long_gen_mean_lat_s=5.19,
            short_gen_tps=225.6,
            tool_call_score="10/10",
            stability_mean_s=1.387,
            stability_cv_pct=1.80,
            concurrent_4_total_s=5.14,
            vram_used_mib_per_gpu=[22265, 21558],
            vram_total_mib=43823,
            genesis_pin="991dc1a",
            vllm_pin="0.20.2rc1.dev9+g01d4d1ad3",
        )
        assert rm.long_gen_sustained_tps == 192.6
        assert rm.tool_call_score == "10/10"
        assert sum(rm.vram_used_mib_per_gpu) == rm.vram_total_mib


# ─── Tolerances ───────────────────────────────────────────────────────


class TestVerifyTolerances:
    def test_default_tolerances(self):
        t = VerifyTolerances()
        assert t.tps_drop_pct_max == 5.0
        assert t.tool_call_min == "9/10"
        assert t.stability_cv_pct_max == 6.0

    def test_tolerances_must_be_positive(self):
        with pytest.raises(SchemaError, match="tps_drop"):
            VerifyTolerances(tps_drop_pct_max=-1).validate()


# ─── Spec decode ──────────────────────────────────────────────────────


class TestSpecDecode:
    def test_spec_decode_optional(self):
        cfg = _make_min()
        assert cfg.spec_decode is None

    def test_mtp_k3(self):
        sd = SpecDecodeConfig(method="mtp", num_speculative_tokens=3)
        sd.validate()  # no exception

    def test_invalid_method_rejected(self):
        with pytest.raises(SchemaError, match="method"):
            SpecDecodeConfig(method="bogus", num_speculative_tokens=3).validate()


# ─── YAML round-trip ──────────────────────────────────────────────────


class TestYAMLRoundtrip:
    def test_dump_then_load_preserves_all_fields(self, tmp_path):
        cfg = _make_full()
        path = tmp_path / "test.yaml"
        path.write_text(dump_yaml(cfg))
        loaded = load_yaml(path.read_text())
        assert loaded == cfg

    def test_load_yaml_with_unknown_field_raises(self):
        with pytest.raises(SchemaError, match="unknown field"):
            load_yaml("""
key: x
title: x
description: x
schema_version: 1
maintainer: x
model_path: /x
hardware:
  gpu_match_keys: [x]
  n_gpus: 1
  min_vram_per_gpu_mib: 1
totally_bogus_field: yes
""")


# ─── Render to launch script ──────────────────────────────────────────


class TestRenderLaunchScript:
    def test_render_includes_all_env_vars(self):
        cfg = _make_full()
        script = cfg.to_launch_script()
        for k, v in cfg.genesis_env.items():
            assert f"-e {k}={v}" in script or f"export {k}={v}" in script
        for k, v in cfg.system_env.items():
            assert f"-e {k}=" in script or f"export {k}=" in script

    def test_render_includes_all_vllm_flags(self):
        cfg = _make_full()
        script = cfg.to_launch_script()
        assert "--max-model-len" in script
        assert "--tensor-parallel-size" in script
        assert "--gpu-memory-utilization" in script
        assert "--kv-cache-dtype" in script
        # Spec decode embedded as JSON
        if cfg.spec_decode:
            assert cfg.spec_decode.method in script

    def test_render_includes_reference_in_header_comment(self):
        cfg = _make_full()
        script = cfg.to_launch_script()
        if cfg.reference_metrics:
            assert "192.6" in script  # tps reference visible


# ─── Validation: required fields per category ─────────────────────────


class TestRequiredFields:
    def test_missing_genesis_env_for_known_critical(self, caplog):
        """If kv_cache_dtype=turboquant_k8v4 + hybrid model, P98 must be ON."""
        cfg = _make_min()
        cfg.kv_cache_dtype = "turboquant_k8v4"
        cfg.model_path = "/models/Qwen3.6-27B-int4-AutoRound"  # hybrid GDN
        cfg.genesis_env = {}  # missing P98
        warnings = cfg.audit()
        assert any("P98" in w for w in warnings)

    def test_no_warnings_when_all_correct(self):
        cfg = _make_full()
        warnings = cfg.audit()
        assert not any("P98" in w for w in warnings)


# ─── Helpers ──────────────────────────────────────────────────────────


def _make_min() -> ModelConfig:
    return ModelConfig(
        key="min-test", title="Min", description="Min test",
        schema_version=1, maintainer="t", model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["x"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
    )


def _make_full() -> ModelConfig:
    return ModelConfig(
        key="full-test", title="Full",
        description="Full coverage test",
        schema_version=1, maintainer="sandermage",
        last_validated="2026-05-05",
        genesis_pin="991dc1a",
        vllm_pin_required="0.20.2rc1.dev9+g01d4d1ad3",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=2,
            min_vram_per_gpu_mib=24000,
        ),
        model_path="/models/Qwen3.6-35B-A3B-FP8",
        served_model_name="qwen3.6-35b-a3b",
        kv_cache_dtype="turboquant_k8v4",
        max_model_len=320000,
        gpu_memory_utilization=0.90,
        max_num_seqs=2,
        max_num_batched_tokens=4096,
        enable_chunked_prefill=True,
        dtype="float16",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        enable_auto_tool_choice=True,
        spec_decode=SpecDecodeConfig(method="mtp", num_speculative_tokens=3),
        genesis_env={
            "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL": "1",
            "GENESIS_ENABLE_P98": "1",
            "GENESIS_ENABLE_PN59_STREAMING_GDN": "1",
        },
        system_env={
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        },
        vllm_extra_args=["--no-scheduler-reserve-full-isl"],
        docker=DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="vllm-server-mtp-test",
            port=8000, shm_size="8g",
            network="genesis-vllm-patches_default",
            mounts=["/nfs/genesis/models:/models:ro"],
        ),
        reference_metrics=ReferenceMetrics(
            measured_at="2026-05-05T18:35:00Z",
            bench_method="bench_35b.sh",
            long_gen_sustained_tps=192.6,
            long_gen_mean_lat_s=5.19,
            short_gen_tps=225.6,
            tool_call_score="10/10",
            stability_mean_s=1.387,
            stability_cv_pct=1.80,
            concurrent_4_total_s=5.14,
            vram_used_mib_per_gpu=[22265, 21558],
            vram_total_mib=43823,
            genesis_pin="991dc1a",
            vllm_pin="0.20.2rc1.dev9+g01d4d1ad3",
        ),
        verify_tolerances=VerifyTolerances(),
        verified_on=["sandermage/2xA5000-A2: 192.6 TPS, 10/10 tool, 991dc1a"],
        notes=[
            "ℹ Requires GENESIS_ENABLE_P98=1 for TQ k8v4 hybrid",
            "⚠ Do NOT enable --enable-prefix-caching",
        ],
        workload_tag="balanced",
    )

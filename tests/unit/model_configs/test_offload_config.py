# SPDX-License-Identifier: Apache-2.0
"""club-3090 #58 Path A (UNIFIED_CONFIG plan 2026-05-09) — OffloadConfig.

Covers:
  - Dataclass validation (numeric, non-negative)
  - to_vllm_args() emits `--cpu-offload-gb` only when > 0
  - Hybrid-GDN guard: validate() raises when PN59 enabled + offload set
  - YAML round-trip preserves the block
  - Renderer emits the flag at the right place in vllm extra args
"""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    OffloadConfig, ModelConfig, HardwareSpec, DockerConfig,
    SchemaError, dump_yaml, load_yaml,
)


# ─── OffloadConfig dataclass

def test_offload_default_disabled():
    o = OffloadConfig()
    o.validate()
    assert o.cpu_offload_gib == 0.0
    assert o.swap_space_gib == 0.0
    assert o.to_vllm_args() == []


def test_offload_emits_cpu_flag_when_set():
    o = OffloadConfig(cpu_offload_gib=24.0)
    o.validate()
    args = o.to_vllm_args()
    assert "--cpu-offload-gb 24" in args


def test_offload_emits_both_flags_when_both_set():
    o = OffloadConfig(cpu_offload_gib=16.0, swap_space_gib=8.0)
    o.validate()
    args = o.to_vllm_args()
    assert "--cpu-offload-gb 16" in args
    assert "--swap-space 8" in args


def test_offload_validate_rejects_negative():
    with pytest.raises(SchemaError, match=">= 0"):
        OffloadConfig(cpu_offload_gib=-1.0).validate()
    with pytest.raises(SchemaError, match=">= 0"):
        OffloadConfig(swap_space_gib=-0.5).validate()


def test_offload_validate_rejects_non_numeric():
    with pytest.raises(SchemaError, match="must be number"):
        OffloadConfig(cpu_offload_gib="lots").validate()  # type: ignore


# ─── Hybrid-GDN guard (the headline safety net)

def _cfg_with_offload(*, hybrid_gdn: bool, offload_gib: float = 0.0) -> ModelConfig:
    env: dict[str, str] = {}
    if hybrid_gdn:
        env["GENESIS_ENABLE_PN59_STREAMING_GDN"] = "1"
    return ModelConfig(
        key="test-offload",
        title="Test offload config",
        description="Minimal docker config for OffloadConfig tests.",
        schema_version=1,
        maintainer="sandermage",
        model_path="/models/dummy",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx 3090"], n_gpus=1,
            min_vram_per_gpu_mib=24000,
        ),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        genesis_env=env,
        offload=OffloadConfig(cpu_offload_gib=offload_gib) if offload_gib else None,
    )


def test_offload_zero_on_hybrid_gdn_is_allowed():
    """offload block present but cpu_offload_gib=0 on hybrid GDN is fine."""
    cfg = ModelConfig(
        key="test-offload-zero",
        title="hybrid GDN, no actual offload",
        description="Tests the guard ignores zero-offload blocks.",
        schema_version=1,
        maintainer="sandermage",
        model_path="/models/dummy",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=2,
            min_vram_per_gpu_mib=22000,
        ),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        genesis_env={"GENESIS_ENABLE_PN59_STREAMING_GDN": "1"},
        offload=OffloadConfig(cpu_offload_gib=0.0, swap_space_gib=4.0),
    )
    cfg.validate()  # no raise


def test_offload_nonzero_on_hybrid_gdn_blocked():
    """The headline guard: cpu_offload_gib>0 + hybrid GDN raises with
    a precise pointer to the research report.
    """
    cfg = _cfg_with_offload(hybrid_gdn=True, offload_gib=8.0)
    with pytest.raises(SchemaError, match="hybrid-GDN"):
        cfg.validate()


def test_offload_nonzero_on_dense_model_allowed():
    """No PN59 → dense model → cpu_offload_gib>0 is fine."""
    cfg = _cfg_with_offload(hybrid_gdn=False, offload_gib=8.0)
    cfg.validate()


# ─── YAML round-trip

def test_offload_yaml_roundtrip():
    cfg = _cfg_with_offload(hybrid_gdn=False, offload_gib=12.0)
    yaml_str = dump_yaml(cfg)
    cfg2 = load_yaml(yaml_str)
    assert cfg2.offload is not None
    assert cfg2.offload.cpu_offload_gib == 12.0


# ─── Renderer integration

def test_renderer_emits_cpu_offload_flag():
    cfg = _cfg_with_offload(hybrid_gdn=False, offload_gib=24.0)
    script = cfg.to_launch_script()
    assert "--cpu-offload-gb 24" in script


def test_renderer_no_flag_when_offload_none():
    cfg = _cfg_with_offload(hybrid_gdn=False, offload_gib=0.0)
    # offload is None when offload_gib=0 in the helper
    script = cfg.to_launch_script()
    assert "--cpu-offload-gb" not in script

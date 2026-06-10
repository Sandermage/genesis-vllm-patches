# SPDX-License-Identifier: Apache-2.0
"""Y8 + Y14 (UNIFIED_CONFIG plan 2026-05-09) — GpuTuning + Observability."""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    GpuTuningConfig, ObservabilityConfig, ModelConfig, HardwareSpec,
    DockerConfig, SchemaError, dump_yaml, load_yaml,
)


# ─── GpuTuningConfig

def test_gpu_tuning_default_safe():
    g = GpuTuningConfig()
    g.validate()
    assert g.unsafe_apply is False
    assert g.persistence_mode is None


def test_gpu_tuning_safe_fields_only_pass():
    g = GpuTuningConfig(persistence_mode=True,
                         transparent_hugepages="madvise",
                         ulimits={"memlock": "unlimited"})
    g.validate()


def test_gpu_tuning_power_limit_requires_unsafe():
    with pytest.raises(SchemaError, match="unsafe_apply"):
        GpuTuningConfig(power_limit_watts=200).validate()


def test_gpu_tuning_clocks_gfx_requires_unsafe():
    with pytest.raises(SchemaError, match="unsafe_apply"):
        GpuTuningConfig(clocks_gfx_mhz=1500).validate()


def test_gpu_tuning_clocks_mem_requires_unsafe():
    with pytest.raises(SchemaError, match="unsafe_apply"):
        GpuTuningConfig(clocks_mem_mhz=8000).validate()


def test_gpu_tuning_unsafe_apply_passes():
    g = GpuTuningConfig(power_limit_watts=200, unsafe_apply=True)
    g.validate()


def test_gpu_tuning_power_limit_min():
    with pytest.raises(SchemaError, match="power_limit_watts"):
        GpuTuningConfig(power_limit_watts=10, unsafe_apply=True).validate()


def test_gpu_tuning_clocks_out_of_range():
    with pytest.raises(SchemaError, match="clocks_gfx_mhz"):
        GpuTuningConfig(clocks_gfx_mhz=50, unsafe_apply=True).validate()


def test_gpu_tuning_thp_validation():
    with pytest.raises(SchemaError, match="transparent_hugepages"):
        GpuTuningConfig(transparent_hugepages="sometimes").validate()


# ─── ObservabilityConfig

def test_observability_default():
    o = ObservabilityConfig()
    o.validate()
    assert o.memory_trace_enabled is False
    assert o.per_patch_telemetry is True


def test_observability_memory_trace_requires_csv_path():
    with pytest.raises(SchemaError, match="memory_trace_csv_path"):
        ObservabilityConfig(memory_trace_enabled=True).validate()


def test_observability_full_config():
    o = ObservabilityConfig(
        memory_trace_enabled=True,
        memory_trace_csv_path="/var/log/genesis/memory.csv",
        cudagraph_dispatch_trace=True,
        per_patch_telemetry=False,
    )
    o.validate()


# ─── YAML round-trip

def test_gpu_tuning_obs_yaml_roundtrip():
    cfg = ModelConfig(
        key="test-tuning-obs",
        title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=2,
                              min_vram_per_gpu_mib=22000),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        gpu_tuning=GpuTuningConfig(
            persistence_mode=True,
            transparent_hugepages="madvise",
            ulimits={"memlock": "unlimited"},
        ),
        observability=ObservabilityConfig(
            cudagraph_dispatch_trace=True,
            per_patch_telemetry=True,
        ),
    )
    yaml_str = dump_yaml(cfg)
    cfg2 = load_yaml(yaml_str)
    assert cfg2.gpu_tuning is not None
    assert cfg2.gpu_tuning.persistence_mode is True
    assert cfg2.gpu_tuning.transparent_hugepages == "madvise"
    assert cfg2.observability is not None
    assert cfg2.observability.cudagraph_dispatch_trace is True

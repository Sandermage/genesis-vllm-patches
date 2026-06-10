# SPDX-License-Identifier: Apache-2.0
"""Path C v7.73.x Day 1 (PN95) — CacheTier + extended CacheConfig tests.

Covers:
  - CacheTier dataclass validation (device whitelist, capacity range,
    eviction policy, water-mark ordering, nvme_path requirement)
  - CacheConfig.tiers back-compat fall-through (empty list → PN91)
  - CacheConfig multi-tier shape constraints (≤1 gpu, ==1 cpu when n≥2)
  - YAML round-trip preserves nested CacheTier list
  - Hybrid-GDN guard relaxation: Path A guard SKIPS when PN95 tiers
    + exclude_mamba_ssm=True
  - Hybrid-GDN guard insists exclude_mamba_ssm=True for PN95 configs
"""
from __future__ import annotations

import pytest

from sndr.model_configs.schema import (
    CacheTier, CacheConfig, ModelConfig, HardwareSpec, DockerConfig,
    OffloadConfig, SchemaError, dump_yaml, load_yaml,
)


# ─── CacheTier dataclass

def test_cache_tier_minimal_valid():
    t = CacheTier(device="gpu", capacity_gib=20.0)
    t.validate()
    assert t.eviction_policy == "lru"
    assert t.promote_on_hit is True
    assert t.demote_threshold_pct == 0.92
    assert t.low_water_pct == 0.75


def test_cache_tier_cpu_default():
    t = CacheTier(device="cpu", capacity_gib=40.0, vision_first=True)
    t.validate()
    assert t.pinned is True
    assert t.vision_first is True


def test_cache_tier_nvme_requires_path():
    with pytest.raises(SchemaError, match="nvme_path"):
        CacheTier(device="nvme", capacity_gib=100.0).validate()
    # nvme_path supplied → ok
    CacheTier(device="nvme", capacity_gib=100.0,
              nvme_path="/mnt/nvme/sndr-cache").validate()


def test_cache_tier_rejects_bad_device():
    with pytest.raises(SchemaError, match="device must be one of"):
        CacheTier(device="hbm", capacity_gib=20.0).validate()


def test_cache_tier_rejects_zero_or_negative_capacity():
    with pytest.raises(SchemaError, match="capacity_gib"):
        CacheTier(device="gpu", capacity_gib=0.0).validate()
    with pytest.raises(SchemaError, match="capacity_gib"):
        CacheTier(device="gpu", capacity_gib=-1.0).validate()


def test_cache_tier_rejects_unknown_eviction_policy():
    with pytest.raises(SchemaError, match="eviction_policy"):
        CacheTier(device="gpu", capacity_gib=20.0,
                   eviction_policy="random").validate()


def test_cache_tier_rejects_inverted_water_marks():
    """low_water_pct must be < demote_threshold_pct."""
    with pytest.raises(SchemaError, match="low_water_pct"):
        CacheTier(device="gpu", capacity_gib=20.0,
                   demote_threshold_pct=0.5,
                   low_water_pct=0.6).validate()


def test_cache_tier_rejects_threshold_out_of_range():
    with pytest.raises(SchemaError, match="demote_threshold_pct"):
        CacheTier(device="gpu", capacity_gib=20.0,
                   demote_threshold_pct=1.5).validate()


# ─── CacheConfig back-compat (empty tiers = PN91)

def test_cache_config_default_no_tiers():
    cc = CacheConfig()
    cc.validate()
    assert cc.tiers == []
    assert cc.exclude_mamba_ssm is True
    assert cc.vision_demote_first is True


def test_cache_config_pn91_fields_unchanged():
    cc = CacheConfig(eviction_policy="2q", arc_capacity=2048,
                     q2_a1_ratio=0.3)
    cc.validate()
    assert cc.eviction_policy == "2q"


# ─── CacheConfig multi-tier shape constraints

def test_cache_config_two_tiers_gpu_cpu():
    cc = CacheConfig(tiers=[
        CacheTier(device="gpu", capacity_gib=20.0),
        CacheTier(device="cpu", capacity_gib=40.0, vision_first=True),
    ])
    cc.validate()
    assert len(cc.tiers) == 2


def test_cache_config_three_tiers_gpu_cpu_nvme():
    cc = CacheConfig(tiers=[
        CacheTier(device="gpu", capacity_gib=20.0),
        CacheTier(device="cpu", capacity_gib=40.0),
        CacheTier(device="nvme", capacity_gib=200.0,
                  nvme_path="/mnt/nvme/sndr"),
    ])
    cc.validate()


def test_cache_config_rejects_two_gpu_tiers():
    with pytest.raises(SchemaError, match="at most one gpu"):
        CacheConfig(tiers=[
            CacheTier(device="gpu", capacity_gib=20.0),
            CacheTier(device="gpu", capacity_gib=10.0),
        ]).validate()


def test_cache_config_rejects_no_cpu_tier_when_multitier():
    """If you have ≥2 tiers, exactly one must be cpu (gpu+nvme alone is not allowed)."""
    with pytest.raises(SchemaError, match="exactly one cpu tier"):
        CacheConfig(tiers=[
            CacheTier(device="gpu", capacity_gib=20.0),
            CacheTier(device="nvme", capacity_gib=100.0,
                      nvme_path="/mnt/nvme/x"),
        ]).validate()


def test_cache_config_rejects_two_cpu_tiers():
    with pytest.raises(SchemaError, match="exactly one cpu tier"):
        CacheConfig(tiers=[
            CacheTier(device="gpu", capacity_gib=20.0),
            CacheTier(device="cpu", capacity_gib=40.0),
            CacheTier(device="cpu", capacity_gib=10.0),
        ]).validate()


def test_cache_config_rejects_bad_low_water_pct():
    with pytest.raises(SchemaError, match="tier_low_water_pct"):
        CacheConfig(tier_low_water_pct=1.5).validate()


# ─── YAML round-trip

def _cfg(cc: CacheConfig, *, hybrid: bool = False) -> ModelConfig:
    env: dict[str, str] = {}
    if hybrid:
        env["GENESIS_ENABLE_PN59_STREAMING_GDN"] = "1"
    return ModelConfig(
        key="test-cache-tier",
        title="Test cache tier config",
        description="Path C Day 1 schema test fixture.",
        schema_version=1,
        maintainer="sandermage",
        model_path="/models/dummy",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx 3090"], n_gpus=1,
            min_vram_per_gpu_mib=24000,
        ),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        cache_config=cc,
        genesis_env=env,
    )


def test_cache_config_yaml_roundtrip_preserves_tiers():
    cc = CacheConfig(tiers=[
        CacheTier(device="gpu", capacity_gib=20.0,
                  eviction_policy="2q"),
        CacheTier(device="cpu", capacity_gib=40.0,
                  vision_first=True, low_water_pct=0.5,
                  demote_threshold_pct=0.85),
    ], vision_demote_first=True, tier_low_water_pct=0.10)
    cfg = _cfg(cc)
    yaml_str = dump_yaml(cfg)
    cfg2 = load_yaml(yaml_str)
    assert cfg2.cache_config is not None
    assert len(cfg2.cache_config.tiers) == 2
    assert cfg2.cache_config.tiers[0].device == "gpu"
    assert cfg2.cache_config.tiers[0].eviction_policy == "2q"
    assert cfg2.cache_config.tiers[1].device == "cpu"
    assert cfg2.cache_config.tiers[1].vision_first is True
    assert cfg2.cache_config.tier_low_water_pct == 0.10


# ─── Hybrid-GDN guard relaxation (Path A → Path C interaction)

def test_path_a_guard_blocks_hybrid_gdn_without_path_c():
    """No tiers declared → Path A guard fires on hybrid-GDN + offload."""
    cc = CacheConfig()  # no tiers
    cfg = ModelConfig(
        key="test",
        title="hybrid GDN no Path C",
        description="x",
        schema_version=1,
        maintainer="sandermage",
        model_path="/models/dummy",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=2,
            min_vram_per_gpu_mib=22000,
        ),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        genesis_env={"GENESIS_ENABLE_PN59_STREAMING_GDN": "1"},
        offload=OffloadConfig(cpu_offload_gib=8.0),
        cache_config=cc,
    )
    with pytest.raises(SchemaError, match="hybrid-GDN"):
        cfg.validate()


def test_path_a_guard_relaxed_when_path_c_active():
    """Hybrid-GDN + offload + tiers + exclude_mamba_ssm=True → ok."""
    cc = CacheConfig(
        tiers=[
            CacheTier(device="gpu", capacity_gib=20.0),
            CacheTier(device="cpu", capacity_gib=40.0),
        ],
        exclude_mamba_ssm=True,
    )
    cfg = ModelConfig(
        key="test",
        title="hybrid GDN with Path C tiers",
        description="x",
        schema_version=1,
        maintainer="sandermage",
        model_path="/models/dummy",
        hardware=HardwareSpec(
            gpu_match_keys=["rtx a5000"], n_gpus=2,
            min_vram_per_gpu_mib=22000,
        ),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        genesis_env={"GENESIS_ENABLE_PN59_STREAMING_GDN": "1"},
        offload=OffloadConfig(cpu_offload_gib=8.0),
        cache_config=cc,
    )
    cfg.validate()  # no raise


def test_path_c_rejects_exclude_mamba_ssm_false_on_hybrid_gdn():
    """Even without offload, Path C tiers + hybrid GDN with exclude=False raises."""
    cc = CacheConfig(
        tiers=[
            CacheTier(device="gpu", capacity_gib=20.0),
            CacheTier(device="cpu", capacity_gib=40.0),
        ],
        exclude_mamba_ssm=False,  # ← deliberately wrong
    )
    cfg = _cfg(cc, hybrid=True)
    with pytest.raises(SchemaError, match="exclude_mamba_ssm"):
        cfg.validate()


def test_path_c_dense_model_can_disable_exclude_mamba_ssm():
    """No PN59 → not hybrid GDN → exclude_mamba_ssm=False is allowed
    (and a no-op since there's no SSM state to exclude)."""
    cc = CacheConfig(
        tiers=[
            CacheTier(device="gpu", capacity_gib=20.0),
            CacheTier(device="cpu", capacity_gib=40.0),
        ],
        exclude_mamba_ssm=False,
    )
    cfg = _cfg(cc, hybrid=False)
    cfg.validate()  # no raise — dense model doesn't need exclusion

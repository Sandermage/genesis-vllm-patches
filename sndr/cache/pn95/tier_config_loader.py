# SPDX-License-Identifier: Apache-2.0
"""PN95 tier_config loader — reads PN95-internal tier_specs YAML files.

Background — V1→V2 architectural unblock (2026-06-01):

Before this module, PN95 hooks loaded its tier_specs by calling the
V1 ModelConfig loader (`sndr.model_configs.registry.get`)
on a fully-populated V1 YAML that carried 100+ lines of launch args
+ hardware + tool-call config in addition to the `cache_config` block
PN95 actually needed. That coupling blocked the V1 sunset of
`a5000-2x-tier-aware-EXAMPLE.yaml` and `a5000-1x-tier-aware-pn95.yaml`
— neither could be retired without an architectural change.

This module separates concerns:

  - V2 deployment config (model + hardware + profile triplet) lives
    in `sndr/model_configs/builtin/presets/example-2x-tier-
    aware.yaml` etc.
  - PN95 tier_specs (the data PN95 actually reads at runtime) lives
    in `sndr/cache/pn95/tier_configs/<key>.yaml`.

Resolution order in `hooks.py:lazy_init`:

  1. PN95-internal tier_configs/<key>.yaml (preferred)
  2. V1 ModelConfig.get(<key>) fallback (backward compat for any
     operator still pointing GENESIS_PN95_CONFIG_KEY at a V1 key)

The PN95 tier_config YAML is a TRIMMED form of the V1 cache_config
block — only the fields make_tier_manager actually reads:

    eviction_policy: lru
    tiers:
      - device: gpu
        capacity_gib: 22.0
        ...
      - device: cpu
        capacity_gib: 8.0
        ...
    vision_demote_first: true
    exclude_mamba_ssm: true
    tier_low_water_pct: 0.05
    async_demote: true

This module returns a thin adapter object whose attribute access
matches what `make_tier_manager(cfg)` expects from a ModelConfig
(specifically: `cfg.cache_config.tiers`, `cfg.cache_config.
vision_demote_first`, and `cfg.kv_transfer_config` for the upstream-
offload detection — which is always None in this PN95-only path).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


_TIER_CONFIG_DIR = Path(__file__).parent / "tier_configs"


@dataclass
class _TierSpec:
    """Mirror of the tier shape `make_tier_manager` reads.

    The real schema lives in
    `sndr/model_configs/types/cache.py:CacheTier` —
    we only need a structural duck-type here, not strict validation,
    since `TierManager(cc.tiers, ...)` consumes attribute access.
    """
    device: str
    capacity_gib: float
    eviction_policy: str = "lru"
    promote_on_hit: bool = True
    vision_first: bool = False
    pinned: bool = False
    demote_threshold_pct: Optional[float] = None
    low_water_pct: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class _CacheConfig:
    """Mirror of the cache_config shape `make_tier_manager` reads."""
    tiers: list[_TierSpec] = field(default_factory=list)
    vision_demote_first: bool = False
    exclude_mamba_ssm: bool = False
    tier_low_water_pct: float = 0.05
    async_demote: bool = True
    eviction_policy: str = "lru"
    offload_connector: Optional[Any] = None  # always None in PN95-only path


@dataclass
class _PN95TierConfigAdapter:
    """Thin adapter so `make_tier_manager(cfg)` can read PN95-internal
    tier configs the same way it reads V1 ModelConfig."""
    cache_config: _CacheConfig
    kv_transfer_config: Optional[Any] = None  # always None — upstream
                                              # connector detection is
                                              # only meaningful when
                                              # using V1 ModelConfig
                                              # which carries kv_transfer
                                              # state from launch args.


def load_by_key(key: str) -> Optional[_PN95TierConfigAdapter]:
    """Load a PN95 tier_config YAML by basename (without .yaml).

    Returns None when no file exists at `tier_configs/<key>.yaml` —
    caller falls back to V1 ModelConfig.get(key) for backward compat.

    Raises ValueError on parse error so misconfigured files are
    surfaced loud, not silently fallen-back-from.
    """
    path = _TIER_CONFIG_DIR / f"{key}.yaml"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"PN95 tier_config {path} must be a YAML mapping at top level, "
            f"got {type(raw).__name__}"
        )
    tiers_raw = raw.get("tiers", [])
    if not isinstance(tiers_raw, list):
        raise ValueError(
            f"PN95 tier_config {path}: `tiers` must be a list, "
            f"got {type(tiers_raw).__name__}"
        )
    tiers = []
    for i, t in enumerate(tiers_raw):
        if not isinstance(t, dict):
            raise ValueError(
                f"PN95 tier_config {path}: tier {i} must be a mapping"
            )
        # device + capacity_gib are required; everything else has a default
        if "device" not in t or "capacity_gib" not in t:
            raise ValueError(
                f"PN95 tier_config {path}: tier {i} missing required "
                f"`device` or `capacity_gib` field"
            )
        tiers.append(_TierSpec(
            device=t["device"],
            capacity_gib=float(t["capacity_gib"]),
            eviction_policy=t.get("eviction_policy", "lru"),
            promote_on_hit=bool(t.get("promote_on_hit", True)),
            vision_first=bool(t.get("vision_first", False)),
            pinned=bool(t.get("pinned", False)),
            demote_threshold_pct=t.get("demote_threshold_pct"),
            low_water_pct=t.get("low_water_pct"),
            notes=t.get("notes"),
        ))
    cc = _CacheConfig(
        tiers=tiers,
        vision_demote_first=bool(raw.get("vision_demote_first", False)),
        exclude_mamba_ssm=bool(raw.get("exclude_mamba_ssm", False)),
        tier_low_water_pct=float(raw.get("tier_low_water_pct", 0.05)),
        async_demote=bool(raw.get("async_demote", True)),
        eviction_policy=raw.get("eviction_policy", "lru"),
    )
    return _PN95TierConfigAdapter(cache_config=cc, kv_transfer_config=None)


def known_keys() -> list[str]:
    """List basenames of PN95 tier_config YAMLs shipping in this dir.

    Helps operators discover available keys via `sndr` CLI surfaces and
    is used by tests to assert the dir is non-empty.
    """
    if not _TIER_CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in _TIER_CONFIG_DIR.glob("*.yaml"))

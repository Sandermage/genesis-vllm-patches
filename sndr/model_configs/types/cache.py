# SPDX-License-Identifier: Apache-2.0
"""CacheTier + CacheConfig + OffloadConfig.

All three were inline classes in ``model_configs/schema.py`` before
M.5.1. Bodies unchanged; only the import path for :class:`SchemaError`
is new.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ._base import SchemaError


@dataclass
class OffloadConfig:
    """club-3090 #58 Path A (UNIFIED_CONFIG plan 2026-05-09): VRAM→CPU/disk
    spillover knobs (interim). Surfaces vLLM's existing
    `--cpu-offload-gb` flag in a typed schema slot AND reserves
    namespace for future tier-aware Genesis-original CacheConfig
    extension (Path C, planned for v7.73.x).

    Today this block translates to one engine arg:
      `--cpu-offload-gb <cpu_offload_gib>`

    Future fields (planned, not yet wired):
      - `tiers: list[CacheTier]`      — hierarchical cache (gpu/cpu/nvme)
      - `vision_token_demote_first`   — image tokens evicted first
      - `exclude_mamba_ssm`           — keep Mamba SSM state on GPU
        (mandatory for hybrid-GDN; vLLM/SGLang/LMCache all crash on
        hybrid-GDN offload — see club-3090 #58 research report)

    Hybrid-GDN guard: when set on a config whose `kv_cache_dtype`
    indicates hybrid GDN (turboquant_k8v4 + GDN model), `validate()`
    raises with a precise pointer to the research report. Operators
    on dense models can use this freely.
    """
    cpu_offload_gib: float = 0.0
    swap_space_gib: float = 0.0
    notes: str = ""

    def validate(self) -> None:
        if not isinstance(self.cpu_offload_gib, (int, float)):
            raise SchemaError(
                "OffloadConfig.cpu_offload_gib must be number (got "
                f"{type(self.cpu_offload_gib).__name__})"
            )
        if self.cpu_offload_gib < 0:
            raise SchemaError(
                "OffloadConfig.cpu_offload_gib must be >= 0 (got "
                f"{self.cpu_offload_gib})"
            )
        if not isinstance(self.swap_space_gib, (int, float)):
            raise SchemaError(
                "OffloadConfig.swap_space_gib must be number"
            )
        if self.swap_space_gib < 0:
            raise SchemaError(
                "OffloadConfig.swap_space_gib must be >= 0"
            )

    def to_vllm_args(self) -> list[str]:
        """Render as vllm engine flags. Empty list when offload is disabled."""
        args: list[str] = []
        if self.cpu_offload_gib > 0:
            args.append(f"--cpu-offload-gb {self.cpu_offload_gib:g}")
        if self.swap_space_gib > 0:
            args.append(f"--swap-space {self.swap_space_gib:g}")
        return args


@dataclass
class CacheTier:
    """Path C v7.73.x (PN95): one level of the KV cache hierarchy.

    Lower-index tiers are closer to compute (typically tier 0 = GPU,
    tier 1 = CPU pinned RAM, tier 2 = NVMe). Each tier carries its own
    capacity and eviction policy; demote crosses tiers, evict drops
    from the bottom tier.

    Operators declare tiers in `cache_config.tiers`. Empty list →
    PN91 single-tier behavior (zero impact for existing PROD configs).

    Field semantics:
      - `device`: 'gpu' | 'cpu' | 'nvme'
      - `capacity_gib`: hard cap on this tier's allocation
      - `eviction_policy`: forwarded to make_policy() per-tier
      - `promote_on_hit`: demoted page hit → bring back to upper tier
      - `demote_threshold_pct`: tier fill ratio that triggers demote
        (default 0.92 — start demoting when 92% full)
      - `low_water_pct`: demote until this ratio reached (default 0.75)
      - `vision_first`: if True, evict mm/image pages first
      - `pinned`: cpu tier uses cudaMallocHost-backed memory (default True)
      - `nvme_path`: required when device == 'nvme'
    """
    device: str
    capacity_gib: float
    eviction_policy: str = "lru"
    promote_on_hit: bool = True
    demote_threshold_pct: float = 0.92
    low_water_pct: float = 0.75
    vision_first: bool = False
    pinned: bool = True
    nvme_path: Optional[str] = None
    notes: str = ""

    def validate(self) -> None:
        from sndr.cache.eviction_policies import list_policies
        valid_devices = {"gpu", "cpu", "nvme"}
        if self.device not in valid_devices:
            raise SchemaError(
                f"CacheTier.device must be one of {sorted(valid_devices)} "
                f"(got {self.device!r})"
            )
        if not isinstance(self.capacity_gib, (int, float)):
            raise SchemaError("CacheTier.capacity_gib must be number")
        if self.capacity_gib <= 0:
            raise SchemaError(
                f"CacheTier.capacity_gib must be > 0 "
                f"(got {self.capacity_gib})"
            )
        valid_pol = list_policies()
        if self.eviction_policy not in valid_pol:
            raise SchemaError(
                f"CacheTier.eviction_policy must be one of {valid_pol} "
                f"(got {self.eviction_policy!r})"
            )
        if not (0.0 < self.demote_threshold_pct <= 1.0):
            raise SchemaError(
                f"CacheTier.demote_threshold_pct must be in (0,1] "
                f"(got {self.demote_threshold_pct})"
            )
        if not (0.0 <= self.low_water_pct < 1.0):
            raise SchemaError(
                f"CacheTier.low_water_pct must be in [0,1) "
                f"(got {self.low_water_pct})"
            )
        if self.low_water_pct >= self.demote_threshold_pct:
            raise SchemaError(
                f"CacheTier.low_water_pct ({self.low_water_pct}) must be "
                f"strictly less than demote_threshold_pct "
                f"({self.demote_threshold_pct})"
            )
        if self.device == "nvme" and not self.nvme_path:
            raise SchemaError(
                "CacheTier.nvme_path is required when device == 'nvme'"
            )


@dataclass
class CacheConfig:
    """T2.1 (vllm#40270 backport / PN91) + Path C v7.73.x (PN95):
    pluggable KV cache eviction with optional multi-tier hierarchy.

    PN91 fields (single-tier, back-compat):
      - `eviction_policy`: 'lru' | '2q' | 'arc'
      - `arc_capacity`: ARC capacity (entries). Ignored for LRU/2Q.
      - `q2_a1_ratio`: 2Q probationary ratio. Ignored for LRU/ARC.

    PN95 / Path C fields (multi-tier extension):
      - `tiers`: ordered list of CacheTier (empty = PN91 single-tier).
      - `exclude_mamba_ssm`: keep Mamba SSM state on GPU. MUST stay
        True on hybrid-GDN models (relaxes the Path A guard via this
        flag instead of blocking the config outright).
      - `vision_demote_first`: image/MM pages demoted before text.
      - `tier_low_water_pct`: GPU free-VRAM threshold to trigger demote
        (e.g. 0.05 = start demoting when free VRAM < 5%).
      - `async_demote`: cudaMemcpyAsync vs sync (default True).

    Back-compat: `tiers == []` → no PN95 behavior at all; existing
    PROD configs are unaffected.
    """
    # ── PN91 single-tier (back-compat) ──
    eviction_policy: str = "lru"
    arc_capacity: int = 4096
    q2_a1_ratio: float = 0.25
    notes: str = ""
    # ── PN95 / Path C multi-tier extensions ──
    tiers: list[CacheTier] = field(default_factory=list)
    exclude_mamba_ssm: bool = True
    vision_demote_first: bool = True
    tier_low_water_pct: float = 0.05
    async_demote: bool = True

    def validate(self) -> None:
        from sndr.cache.eviction_policies import list_policies
        valid = list_policies()
        if self.eviction_policy not in valid:
            raise SchemaError(
                f"CacheConfig.eviction_policy must be one of {valid} "
                f"(got {self.eviction_policy!r})"
            )
        if self.arc_capacity <= 0:
            raise SchemaError(
                "CacheConfig.arc_capacity must be > 0 "
                f"(got {self.arc_capacity})"
            )
        if not (0.0 < self.q2_a1_ratio < 1.0):
            raise SchemaError(
                f"CacheConfig.q2_a1_ratio must be in (0,1) "
                f"(got {self.q2_a1_ratio})"
            )
        if not (0.0 <= self.tier_low_water_pct < 1.0):
            raise SchemaError(
                f"CacheConfig.tier_low_water_pct must be in [0,1) "
                f"(got {self.tier_low_water_pct})"
            )
        # Multi-tier shape constraints
        for t in self.tiers:
            t.validate()
        if self.tiers:
            gpu_tiers = [t for t in self.tiers if t.device == "gpu"]
            if len(gpu_tiers) > 1:
                raise SchemaError(
                    f"CacheConfig.tiers may declare at most one gpu tier "
                    f"(got {len(gpu_tiers)})"
                )
            if len(self.tiers) >= 2:
                cpu_tiers = [t for t in self.tiers if t.device == "cpu"]
                if len(cpu_tiers) != 1:
                    raise SchemaError(
                        f"CacheConfig.tiers requires exactly one cpu tier "
                        f"when len(tiers) >= 2 (got {len(cpu_tiers)})"
                    )

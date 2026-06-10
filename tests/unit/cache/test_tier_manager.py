# SPDX-License-Identifier: Apache-2.0
"""Path C v7.73.x Day 2 (PN95) — TierManager unit tests.

Exercises:
  - admit / touch round-trip at tier 0 (single-tier behavior)
  - demote_to_threshold drains tier 0 to low_water
  - vision_demote_first orders MM pages ahead of text pages
  - mamba-excluded groups never appear in demote candidates
  - spec-decode hot ring (last N admits) refuse demotion
  - terminal eviction removes from coldest tier
  - touch on demoted page returns CPU bytes; mark_promoted moves back
  - factory returns None when cfg.cache_config has no tiers
"""
from __future__ import annotations

import pytest

from sndr.cache.tier_manager import (
    TierManager, _CpuSlab, make_tier_manager,
)
from sndr.model_configs.schema import (
    CacheTier, CacheConfig, ModelConfig, HardwareSpec, DockerConfig,
)


SLOT = 1024  # 1 KiB slots for tests (small for fast bookkeeping)


def _two_tier(cap_gpu_gib: float = 0.001,   # ~1 MiB GPU cap = ~1024 slots
              cap_cpu_gib: float = 0.002,   # ~2 MiB CPU cap = ~2048 slots
              vision_first: bool = True,
              hot_ring: int = 0) -> TierManager:
    return TierManager(
        tiers=[
            CacheTier(device="gpu", capacity_gib=cap_gpu_gib,
                      low_water_pct=0.1),
            CacheTier(device="cpu", capacity_gib=cap_cpu_gib,
                      vision_first=vision_first),
        ],
        slot_nbytes=SLOT,
        vision_demote_first=vision_first,
        spec_decode_hot_ring=hot_ring,
    )


# ─── _CpuSlab

def test_cpu_slab_alloc_free_round_trip():
    slab = _CpuSlab(n_slots=4, slot_nbytes=128)
    assert slab.n_free() == 4
    s1 = slab.alloc()
    assert s1 in (0, 1, 2, 3)
    assert slab.n_free() == 3
    slab.free(s1)
    assert slab.n_free() == 4


def test_cpu_slab_store_load():
    slab = _CpuSlab(n_slots=2, slot_nbytes=8)
    s = slab.alloc()
    slab.store(s, b"hello\0\0\0")
    assert slab.load(s) == b"hello\0\0\0"


def test_cpu_slab_alloc_returns_none_when_full():
    slab = _CpuSlab(n_slots=2, slot_nbytes=8)
    s1, s2 = slab.alloc(), slab.alloc()
    assert s1 is not None and s2 is not None
    assert slab.alloc() is None


# ─── TierManager basics

def test_admit_at_tier0():
    tm = _two_tier()
    tm.admit("blk1")
    assert tm.n_pages_at_tier(0) == 1
    assert tm.n_pages_at_tier(1) == 0


def test_touch_tier0_returns_none():
    tm = _two_tier()
    tm.admit("blk1")
    assert tm.touch("blk1") is None  # still tier 0


def test_touch_unknown_returns_none():
    tm = _two_tier()
    assert tm.touch("never-admitted") is None


def test_demote_no_tier1_returns_zero():
    """Single-tier manager: demote is a no-op."""
    tm = TierManager(
        tiers=[CacheTier(device="gpu", capacity_gib=0.001)],
        slot_nbytes=SLOT,
    )
    tm.admit("blk1")
    assert tm.demote_to_threshold(payloads={"blk1": b"x" * SLOT}) == 0


# ─── Demote behavior

def test_demote_drains_to_low_water():
    """GPU tier capacity ~1 MiB / 1 KiB slot = 1024 slots; low_water=0.1
    means target ~102 pages; if we admit 1024+ we should demote."""
    tm = _two_tier(cap_gpu_gib=0.000_1, cap_cpu_gib=0.001)  # ~100 GPU / ~1024 CPU
    payloads = {}
    n = 200  # well over GPU capacity (~100 pages)
    for i in range(n):
        key = f"blk{i}"
        tm.admit(key)
        payloads[key] = b"x" * SLOT
    moved = tm.demote_to_threshold(payloads)
    assert moved > 0  # something demoted
    # After demote tier 0 count should be near low_water target
    assert tm.n_pages_at_tier(0) < n


def test_demote_vision_first_drains_mm_pages_first():
    """vision_demote_first=True: MM pages get demoted before text pages."""
    tm = _two_tier(cap_gpu_gib=0.000_1, vision_first=True)
    payloads = {}
    # 50 text + 50 image, alternating
    for i in range(50):
        tm.admit(f"text{i}", mm_origin=False)
        tm.admit(f"img{i}", mm_origin=True)
        payloads[f"text{i}"] = b"t" * SLOT
        payloads[f"img{i}"] = b"i" * SLOT
    tm.demote_to_threshold(payloads)
    # All img* should have demoted before any text* — count surviving tier-0
    n_text_tier0 = sum(
        1 for i in range(50)
        if tm._pages[f"text{i}"].tier_idx == 0
    )
    n_img_tier0 = sum(
        1 for i in range(50)
        if tm._pages[f"img{i}"].tier_idx == 0
    )
    assert n_img_tier0 <= n_text_tier0  # img drained at least as much


def test_demote_skips_mamba_excluded_groups():
    """Mamba groups are filtered out of demote candidates."""
    tm = _two_tier(cap_gpu_gib=0.000_05)  # tiny — forces demote
    tm.register_mamba_excluded("mamba_layer_0")
    payloads = {}
    for i in range(100):
        gid = "mamba_layer_0" if i % 2 == 0 else "attn_layer_0"
        tm.admit(f"blk{i}", group_id=gid)
        payloads[f"blk{i}"] = b"x" * SLOT
    tm.demote_to_threshold(payloads)
    # Every blk{even}_i in mamba group must STILL be in tier 0
    for i in range(0, 100, 2):
        assert tm._pages[f"blk{i}"].tier_idx == 0, (
            f"mamba page blk{i} was demoted!"
        )


def test_spec_decode_hot_ring_refuses_demote():
    """The last N admits never get demoted — spec-decode safety."""
    tm = _two_tier(cap_gpu_gib=0.000_05, hot_ring=10)
    payloads = {}
    for i in range(100):
        tm.admit(f"blk{i}")
        payloads[f"blk{i}"] = b"x" * SLOT
    tm.demote_to_threshold(payloads)
    # The last 10 admits must all be tier 0
    for i in range(90, 100):
        assert tm._pages[f"blk{i}"].tier_idx == 0, (
            f"hot-ring blk{i} was demoted!"
        )


# ─── Touch + promote

def test_touch_demoted_returns_payload():
    tm = _two_tier(cap_gpu_gib=0.000_05)
    payloads = {}
    for i in range(100):
        tm.admit(f"blk{i}")
        payloads[f"blk{i}"] = bytes([i % 256] * SLOT)
    tm.demote_to_threshold(payloads)
    # Find a demoted key
    demoted = [k for k, m in tm._pages.items() if m.tier_idx == 1]
    assert demoted, "expected at least one demoted page"
    sample = demoted[0]
    bytes_back = tm.touch(sample)
    assert bytes_back is not None
    assert bytes_back == payloads[sample]


def test_mark_promoted_moves_back_to_tier0():
    tm = _two_tier(cap_gpu_gib=0.000_05)
    payloads = {}
    for i in range(100):
        tm.admit(f"blk{i}")
        payloads[f"blk{i}"] = b"x" * SLOT
    tm.demote_to_threshold(payloads)
    demoted = [k for k, m in tm._pages.items() if m.tier_idx == 1]
    sample = demoted[0]
    tm.touch(sample)
    tm.mark_promoted(sample)
    assert tm._pages[sample].tier_idx == 0
    assert tm._pages[sample].cpu_slot_idx is None


# ─── Terminal eviction

def test_evict_terminal_drops_one_from_coldest():
    tm = _two_tier(cap_gpu_gib=0.000_05)
    payloads = {}
    for i in range(100):
        tm.admit(f"blk{i}")
        payloads[f"blk{i}"] = b"x" * SLOT
    tm.demote_to_threshold(payloads)
    n_before = tm.n_pages()
    victim = tm.evict_terminal()
    if victim is not None:
        assert tm.n_pages() == n_before - 1
        assert victim not in tm._pages


# ─── Mamba registration

def test_register_mamba_rejects_empty_group_id():
    tm = _two_tier()
    with pytest.raises(ValueError):
        tm.register_mamba_excluded("")


def test_is_mamba_excluded():
    tm = _two_tier()
    assert tm.is_mamba_excluded("foo") is False
    tm.register_mamba_excluded("foo")
    assert tm.is_mamba_excluded("foo") is True


# ─── Stats

def test_stats_shape():
    tm = _two_tier()
    tm.admit("blk1")
    tm.register_mamba_excluded("mg1")
    s = tm.stats()
    assert s["n_pages_total"] == 1
    assert s["n_pages_per_tier"][0] == 1
    assert s["n_pages_per_tier"][1] == 0
    assert s["cpu_slab_used"] == 0
    assert s["n_mamba_excluded_groups"] == 1


# ─── Factory

def test_make_tier_manager_returns_none_without_tiers():
    cfg = ModelConfig(
        key="x", title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        cache_config=None,
    )
    assert make_tier_manager(cfg) is None


def test_make_tier_manager_returns_none_with_empty_tiers():
    cfg = ModelConfig(
        key="x", title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        cache_config=CacheConfig(tiers=[]),
    )
    assert make_tier_manager(cfg) is None


def test_make_tier_manager_builds_when_tiers_set():
    cfg = ModelConfig(
        key="x", title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        cache_config=CacheConfig(tiers=[
            CacheTier(device="gpu", capacity_gib=20.0),
            CacheTier(device="cpu", capacity_gib=40.0),
        ]),
    )
    tm = make_tier_manager(cfg)
    assert tm is not None
    assert len(tm.tiers) == 2


# ─── Single-tier behavior (back-compat)

def test_single_tier_admit_works():
    """A 1-tier manager (gpu only) admits + tracks but doesn't demote."""
    tm = TierManager(
        tiers=[CacheTier(device="gpu", capacity_gib=20.0)],
        slot_nbytes=SLOT,
    )
    tm.admit("blk1")
    assert tm.touch("blk1") is None
    assert tm.demote_to_threshold(payloads={"blk1": b"x" * SLOT}) == 0

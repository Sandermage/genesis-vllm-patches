# SPDX-License-Identifier: Apache-2.0
"""Path C v7.73.x Day 3-4 (PN95) — _pn95_runtime hooks + text-patch shape.

Covers:
  - notify_admit / notify_touch fail-silent when:
      - PN95 disabled (env unset)
      - TierManager singleton not initialized
      - request object missing expected attrs
  - init_from_config returns False when env disabled
  - init_from_config returns False when cfg has no tiers
  - init_from_config installs the singleton when env+cfg both set
  - reset_for_tests cleans up
  - text-patch module: anchor strings + replacement strings round-trip
  - dispatcher registry has PN95 entry with right shape
  - register_patch is wired in _per_patch_dispatch
"""
from __future__ import annotations

import os

import pytest

from sndr.cache import _pn95_runtime as P
from sndr.cache.tier_manager import TierManager
from sndr.model_configs.schema import (
    CacheTier, CacheConfig, ModelConfig, HardwareSpec, DockerConfig,
)


@pytest.fixture(autouse=True)
def _reset_pn95():
    """Always start each test with a clean singleton."""
    P.reset_for_tests()
    yield
    P.reset_for_tests()


# ─── Fail-silent contract

def test_notify_admit_silent_when_disabled():
    """No env flag → no singleton → no exception."""
    os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
    # Should never raise
    P.notify_admit(request=object(), prev_n_cached=0, new_n_cached=4, group_id=0)


def test_notify_touch_silent_when_disabled():
    os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
    P.notify_touch(block_hash=b"abc", group_ids=[0], cached_blocks=None)


def test_notify_admit_silent_with_minimal_request_object():
    """Even with a wholly empty object, notify_admit must not throw."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        # Install a singleton via direct ctor (bypass cfg)
        tm = TierManager(
            tiers=[
                CacheTier(device="gpu", capacity_gib=1.0),
                CacheTier(device="cpu", capacity_gib=2.0),
            ],
            slot_nbytes=1024,
        )
        with P._LOCK:
            P._TM = tm
        # Empty object lacks request_id, has_mm_input, etc.
        P.notify_admit(request=object(), prev_n_cached=0, new_n_cached=2,
                        group_id=7)
        # Bookkeeping should have absorbed it
        assert tm.n_pages_at_tier(0) == 2
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_notify_touch_silent_with_unknown_block_hash():
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = TierManager(
            tiers=[CacheTier(device="gpu", capacity_gib=1.0)],
            slot_nbytes=1024,
        )
        with P._LOCK:
            P._TM = tm
        # Never admitted — should be a clean no-op
        P.notify_touch(block_hash=b"never-seen", group_ids=[0],
                        cached_blocks=None)
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


# ─── init_from_config

def _cfg_with_tiers(*, enable_path_c: bool = True) -> ModelConfig:
    cc = CacheConfig(tiers=[
        CacheTier(device="gpu", capacity_gib=1.0),
        CacheTier(device="cpu", capacity_gib=2.0),
    ]) if enable_path_c else CacheConfig()
    return ModelConfig(
        key="test-pn95",
        title="x", description="x",
        schema_version=1, maintainer="sandermage",
        model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
        docker=DockerConfig(image="i", container_name="c", port=8000),
        cache_config=cc,
    )


def test_init_from_config_skips_when_env_disabled():
    os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
    assert P.init_from_config(_cfg_with_tiers()) is False
    assert P.tier_manager() is None


def test_init_from_config_skips_when_no_tiers():
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        assert P.init_from_config(_cfg_with_tiers(enable_path_c=False)) is False
        assert P.tier_manager() is None
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_init_from_config_installs_singleton_when_enabled():
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        assert P.init_from_config(_cfg_with_tiers()) is True
        tm = P.tier_manager()
        assert tm is not None
        assert len(tm.tiers) == 2
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_init_from_config_replacement_keeps_running():
    """Re-init with a different cfg replaces the singleton (with a warning logged)."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        cfg1 = _cfg_with_tiers()
        assert P.init_from_config(cfg1) is True
        tm1 = P.tier_manager()
        assert tm1 is not None
        cfg2 = _cfg_with_tiers()  # different obj, same shape
        assert P.init_from_config(cfg2) is True
        tm2 = P.tier_manager()
        assert tm2 is not None
        # Either the same TM (idempotent) or a fresh one — both acceptable
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_reset_for_tests_drops_singleton():
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        P.init_from_config(_cfg_with_tiers())
        assert P.tier_manager() is not None
        P.reset_for_tests()
        assert P.tier_manager() is None
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


# ─── Text-patch module shape

def test_pn95_text_patch_module_exports_apply():
    from sndr.engines.vllm.patches.kv_cache import pn95_tier_aware_cache as M
    assert callable(M.apply)


def test_pn95_text_patch_anchors_well_formed():
    """Anchor + replacement strings must contain the marker + sentinel lines."""
    from sndr.engines.vllm.patches.kv_cache import pn95_tier_aware_cache as M
    assert M.GENESIS_PN95_MARKER
    assert "[Genesis PN95" in M.PN95_SITE1_NEW
    assert "[Genesis PN95" in M.PN95_SITE2_NEW
    # Each sentinel line of OLD must appear in NEW. (NEW interleaves a
    # `[Genesis PN95]` comment block, so OLD as a whole substring may
    # not appear contiguously — but every original line must persist.)
    for line in M.PN95_SITE1_OLD.splitlines():
        if line.strip():
            assert line in M.PN95_SITE1_NEW, f"missing line: {line!r}"
    for line in M.PN95_SITE2_OLD.splitlines():
        if line.strip():
            assert line in M.PN95_SITE2_NEW, f"missing line: {line!r}"


def test_pn95_apply_returns_skipped_when_vllm_absent():
    """On Mac dev (no vllm), apply() returns skipped cleanly — never raises."""
    from sndr.engines.vllm.patches.kv_cache import pn95_tier_aware_cache as M
    os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
    status, reason = M.apply()
    # PN95 is default OFF in registry → should_apply → "skipped"
    # OR vllm not importable → "skipped"
    assert status == "skipped"
    assert isinstance(reason, str) and len(reason) > 0


# ─── Registry + dispatcher integration

def test_pn95_in_dispatcher_registry():
    from sndr.dispatcher.registry import PATCH_REGISTRY
    assert "PN95" in PATCH_REGISTRY
    entry = PATCH_REGISTRY["PN95"]
    assert entry["env_flag"] == "GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"
    assert entry["default_on"] is False
    assert entry["category"] == "kv_cache"
    assert entry["family"] == "kv_cache"
    assert entry["lifecycle"] == "experimental"


def test_pn95_in_per_patch_dispatch():
    """The apply hook is registered (apply_patch_N95_* exists)."""
    import sndr.apply._per_patch_dispatch as M
    assert hasattr(M, "apply_patch_N95_tier_aware_cache")


def test_pn95_in_env_flags():
    from sndr.env import Flags
    assert hasattr(Flags, "PN95_TIER_AWARE_CACHE")

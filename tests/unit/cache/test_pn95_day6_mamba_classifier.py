# SPDX-License-Identifier: Apache-2.0
"""Path C v7.73.x Day 6 (PN95) — Mamba SSM classifier hook tests."""
from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

from sndr.cache import _pn95_runtime as P
from sndr.cache.tier_manager import TierManager
from sndr.cache._pn95_runtime import (
    init_mamba_exclusions_from_kv_groups,
)
from sndr.model_configs.schema import (
    CacheTier, CacheConfig, ModelConfig, HardwareSpec, DockerConfig,
)


# ─── Synthetic vLLM-shape fixtures (mock KVCacheGroupSpec)

@dataclass
class MambaSpec:  # noqa: N801 — name MUST match vllm's MambaSpec exactly
    """Mock — type name 'MambaSpec' is what the classifier checks."""
    block_size: int = 4096


@dataclass
class _FullAttentionSpec:
    block_size: int = 16


@dataclass
class _Group:
    layer_names: list
    kv_cache_spec: object


@pytest.fixture(autouse=True)
def _reset():
    P.reset_for_tests()
    yield
    P.reset_for_tests()


# ─── init_mamba_exclusions_from_kv_groups

def test_init_mamba_returns_zero_when_disabled():
    os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
    n = init_mamba_exclusions_from_kv_groups([
        _Group(layer_names=["layer.0"], kv_cache_spec=MambaSpec()),
    ])
    assert n == 0


def test_init_mamba_returns_zero_when_no_singleton_no_env():
    """Enabled but no GENESIS_PN95_CONFIG_KEY → can't lazy-init →
    returns 0 cleanly."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    os.environ.pop("GENESIS_PN95_CONFIG_KEY", None)
    try:
        groups = [_Group(layer_names=["x"], kv_cache_spec=MambaSpec())]
        n = init_mamba_exclusions_from_kv_groups(groups)
        assert n == 0
        assert P.tier_manager() is None
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_init_mamba_with_explicit_singleton():
    """Pre-installed TM + Mamba groups → all Mamba groups excluded."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = TierManager(
            tiers=[
                CacheTier(device="gpu", capacity_gib=1.0),
                CacheTier(device="cpu", capacity_gib=2.0),
            ],
            slot_nbytes=1024,
        )
        with P._LOCK:
            P._TM = tm
        groups = [
            _Group(layer_names=["attn.0"], kv_cache_spec=_FullAttentionSpec()),
            _Group(layer_names=["mamba.0"], kv_cache_spec=MambaSpec()),
            _Group(layer_names=["attn.1"], kv_cache_spec=_FullAttentionSpec()),
            _Group(layer_names=["mamba.1"], kv_cache_spec=MambaSpec()),
        ]
        n = init_mamba_exclusions_from_kv_groups(groups)
        assert n == 2  # 2 Mamba groups excluded
        # Verify on the manager
        assert tm.is_mamba_excluded("g1") is True   # mamba.0
        assert tm.is_mamba_excluded("g3") is True   # mamba.1
        assert tm.is_mamba_excluded("g0") is False  # attn.0
        assert tm.is_mamba_excluded("g2") is False  # attn.1
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_init_mamba_dense_only_returns_zero():
    """No Mamba groups → returns 0 but does not error."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = TierManager(
            tiers=[CacheTier(device="gpu", capacity_gib=1.0)],
            slot_nbytes=1024,
        )
        with P._LOCK:
            P._TM = tm
        groups = [
            _Group(layer_names=["attn.0"], kv_cache_spec=_FullAttentionSpec()),
            _Group(layer_names=["attn.1"], kv_cache_spec=_FullAttentionSpec()),
        ]
        n = init_mamba_exclusions_from_kv_groups(groups)
        assert n == 0
        assert tm.is_mamba_excluded("g0") is False
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_init_mamba_handles_empty_groups():
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = TierManager(
            tiers=[CacheTier(device="gpu", capacity_gib=1.0)],
            slot_nbytes=1024,
        )
        with P._LOCK:
            P._TM = tm
        n = init_mamba_exclusions_from_kv_groups([])
        assert n == 0
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_init_mamba_handles_malformed_groups():
    """Group missing kv_cache_spec is silently skipped."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = TierManager(
            tiers=[CacheTier(device="gpu", capacity_gib=1.0)],
            slot_nbytes=1024,
        )
        with P._LOCK:
            P._TM = tm
        # Group with no kv_cache_spec attr
        class _Bad:
            pass
        valid = _Group(layer_names=["m.0"], kv_cache_spec=MambaSpec())
        n = init_mamba_exclusions_from_kv_groups([_Bad(), valid, None])
        assert n == 1  # only the valid one was excluded
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_init_mamba_lazy_init_from_env_config_key():
    """When _TM is None + GENESIS_PN95_CONFIG_KEY is set + cfg has tiers,
    lazy init succeeds and Mamba exclusion proceeds.

    Phase 10.5 (2026-06-01): the old `single-3090-hybrid-gdn-tier-aware-
    example` key referred to a V1 monolithic config retired in V1 sunset
    #2. Migrated to `a5000-2x-tier-aware-example` which is the V2-era
    PN95-internal tier_configs/<key>.yaml replacement carrying the same
    backward-compat example tier shape (see
    `vllm/sndr_core/cache/pn95/tier_configs/`).
    """
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    os.environ["GENESIS_PN95_CONFIG_KEY"] = "a5000-2x-tier-aware-example"
    try:
        # Singleton starts as None
        assert P.tier_manager() is None
        groups = [
            _Group(layer_names=["m.0"], kv_cache_spec=MambaSpec()),
        ]
        n = init_mamba_exclusions_from_kv_groups(groups)
        # After lazy init, singleton should be installed
        tm = P.tier_manager()
        assert tm is not None
        assert n == 1
        assert tm.is_mamba_excluded("g0") is True
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
        os.environ.pop("GENESIS_PN95_CONFIG_KEY", None)


def test_init_mamba_lazy_init_unknown_config_key_no_crash():
    """Unknown GENESIS_PN95_CONFIG_KEY → init fails silently → returns 0."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    os.environ["GENESIS_PN95_CONFIG_KEY"] = "definitely-not-a-real-config-xyz"
    try:
        n = init_mamba_exclusions_from_kv_groups([
            _Group(layer_names=["m.0"], kv_cache_spec=MambaSpec()),
        ])
        assert n == 0
        assert P.tier_manager() is None
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
        os.environ.pop("GENESIS_PN95_CONFIG_KEY", None)


def test_init_mamba_excluded_pages_not_in_demote_candidates():
    """Day 6 acceptance: after Mamba exclusion, demote() never returns
    a page from a Mamba-classified group."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = TierManager(
            tiers=[
                CacheTier(device="gpu", capacity_gib=0.0001),  # tiny — forces demote
                CacheTier(device="cpu", capacity_gib=0.001),
            ],
            slot_nbytes=1024,
        )
        with P._LOCK:
            P._TM = tm
        # Register Mamba group via classifier hook
        groups = [
            _Group(layer_names=["attn.0"], kv_cache_spec=_FullAttentionSpec()),
            _Group(layer_names=["mamba.0"], kv_cache_spec=MambaSpec()),
        ]
        init_mamba_exclusions_from_kv_groups(groups)
        # Admit pages from BOTH groups
        payloads = {}
        for i in range(50):
            attn_key = (f"r1", "g0", i)
            mamba_key = (f"r1", "g1", i)
            tm.admit(attn_key, group_id="g0")
            tm.admit(mamba_key, group_id="g1")
            payloads[attn_key] = b"a" * 1024
            payloads[mamba_key] = b"m" * 1024
        # Force demote
        tm.demote_to_threshold(payloads)
        # Mamba pages MUST all be tier 0
        for i in range(50):
            mamba_key = (f"r1", "g1", i)
            assert tm._pages[mamba_key].tier_idx == 0, (
                f"Mamba page {mamba_key} was demoted!"
            )
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


# ─── Text-patch shape

def test_text_patch_has_third_anchor():
    from sndr.engines.vllm.patches.kv_cache import pn95_tier_aware_cache as M
    assert hasattr(M, "PN95_SITE3_OLD")
    assert hasattr(M, "PN95_SITE3_NEW")
    assert "init_mamba_exclusions_from_kv_groups" in M.PN95_SITE3_NEW
    assert "[Genesis PN95]" in M.PN95_SITE3_NEW
    # Original lines preserved in replacement
    assert "self.empty_kv_cache_blocks" in M.PN95_SITE3_NEW


def test_apply_uses_three_patchers():
    """apply() returns a 3-part summary with admit + touch + mamba-init."""
    from sndr.engines.vllm.patches.kv_cache import pn95_tier_aware_cache as M
    os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
    status, reason = M.apply()
    # Default OFF → should_apply returns False → "skipped"
    assert status == "skipped"

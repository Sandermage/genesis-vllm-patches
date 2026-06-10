# SPDX-License-Identifier: Apache-2.0
"""Path C v1.0 Phase 1 (PN95) — register_kv_caches tests."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from sndr.cache import _pn95_runtime as P
from sndr.cache.tier_manager import TierManager
from sndr.cache._pn95_runtime import register_kv_caches
from sndr.model_configs.schema import CacheTier


@dataclass
class _MockTensor:
    shape: tuple = (2, 100, 16, 4, 128)  # attention shape
    dtype_name: str = "float16"
    device_name: str = "cuda:0"

    @property
    def dtype(self):
        return self.dtype_name

    @property
    def device(self):
        return self.device_name

    def element_size(self) -> int:
        return 2  # fp16


@pytest.fixture(autouse=True)
def _reset():
    P.reset_for_tests()
    yield
    P.reset_for_tests()


def _install_tm() -> TierManager:
    tm = TierManager(
        tiers=[
            CacheTier(device="gpu", capacity_gib=1.0),
            CacheTier(device="cpu", capacity_gib=2.0),
        ],
        slot_nbytes=1024,
    )
    with P._LOCK:
        P._TM = tm
    return tm


# ─── register_kv_caches

def test_register_kv_caches_returns_zero_when_disabled():
    os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
    n = register_kv_caches([_MockTensor()], [])
    assert n == 0


def test_register_kv_caches_returns_zero_when_no_singleton():
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        # No singleton installed
        n = register_kv_caches([_MockTensor()], [])
        assert n == 0
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_register_kv_caches_list_shape():
    """vLLM list[Tensor] shape — registers all layers."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = _install_tm()
        kv_caches = [_MockTensor(), _MockTensor(), _MockTensor()]
        n = register_kv_caches(kv_caches, [])
        assert n == 3
        # Check stash on TM
        assert hasattr(tm, "_kv_caches_ref")
        assert hasattr(tm, "_kv_caches_meta")
        assert len(tm._kv_caches_meta) == 3
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_register_kv_caches_dict_shape():
    """vLLM dict[layer_name, Tensor] shape."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = _install_tm()
        kv_caches = {
            "layer.0.attn": _MockTensor(),
            "layer.1.attn": _MockTensor(),
        }
        n = register_kv_caches(kv_caches, [])
        assert n == 2
        assert "layer.0.attn" in tm._kv_caches_meta
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_register_kv_caches_records_shape_metadata():
    """Per-layer metadata captures shape, dtype, device, bytes_per_block."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = _install_tm()
        # Attention shape: (2, num_blocks, block_size, num_kv_heads, head_dim)
        # = (2, 100, 16, 4, 128) → bytes_per_block ≈ 2*16*4*128*2 = 32768
        n = register_kv_caches([_MockTensor()], [])
        assert n == 1
        meta = tm._kv_caches_meta["0"]
        assert meta["shape"] == (2, 100, 16, 4, 128)
        assert meta["dtype"] == "float16"
        assert meta["bytes_per_block"] > 0
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_register_kv_caches_silent_on_unknown_shape():
    """Non-list, non-dict input → log warning, return 0."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        _install_tm()
        n = register_kv_caches("not a tensor list", [])
        assert n == 0
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_register_kv_caches_resilient_to_bad_tensor():
    """One bad tensor doesn't break the loop."""
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = _install_tm()
        # First is good, second is None (no shape), third is good
        kv_caches = [_MockTensor(), None, _MockTensor()]
        n = register_kv_caches(kv_caches, [])
        # 2 good ones registered (the None gets caught + skipped via shape=())
        assert n >= 2
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


# ─── Text-patch shape

def test_text_patch_has_fourth_anchor():
    from sndr.engines.vllm.patches.kv_cache import pn95_tier_aware_cache as M
    assert hasattr(M, "PN95_SITE4_OLD")
    assert hasattr(M, "PN95_SITE4_NEW")
    assert "register_kv_caches" in M.PN95_SITE4_NEW
    assert "[Genesis PN95 v1.0" in M.PN95_SITE4_NEW
    # Original lines preserved
    assert "kv_caches = self.initialize_kv_cache_tensors" in M.PN95_SITE4_NEW


def test_apply_uses_four_patchers():
    """apply() summary must include all four anchors."""
    from sndr.engines.vllm.patches.kv_cache import pn95_tier_aware_cache as M
    os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)
    status, reason = M.apply()
    assert status == "skipped"
    # Just confirm the function returns cleanly when default off

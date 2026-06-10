# SPDX-License-Identifier: Apache-2.0
"""Path C v7.73.x Day 5 (PN95) — per-block MM tagging tests.

Validates `_mm_block_overlap_set` + `notify_admit` correctness when
real `request.mm_features` (each carrying `mm_position.offset/length`)
is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from sndr.cache import _pn95_runtime as P
from sndr.cache.tier_manager import TierManager
from sndr.cache._pn95_runtime import _mm_block_overlap_set
from sndr.model_configs.schema import (
    CacheTier, CacheConfig, ModelConfig, HardwareSpec, DockerConfig,
)


# ─── Synthetic vLLM-shape fixtures

@dataclass
class _PlaceholderRange:
    offset: int
    length: int


@dataclass
class _MMFeature:
    mm_position: _PlaceholderRange
    modality: str = "image"


class _Request:
    def __init__(self, *, request_id: str = "req-test",
                  mm_features: list = None,
                  has_mm_input: bool = False):
        self.request_id = request_id
        self.mm_features = mm_features or []
        self.has_mm_input = has_mm_input


@pytest.fixture(autouse=True)
def _reset():
    P.reset_for_tests()
    yield
    P.reset_for_tests()


# ─── _mm_block_overlap_set helper

def test_overlap_empty_features():
    assert _mm_block_overlap_set([], range(0, 4), 16) == set()


def test_overlap_single_feature_within_one_block():
    """1 image at offset=4, length=8 (tokens 4..12). Block size 16 →
    overlaps block 0 only."""
    f = _MMFeature(mm_position=_PlaceholderRange(offset=4, length=8))
    assert _mm_block_overlap_set([f], range(0, 4), 16) == {0}


def test_overlap_single_feature_spans_two_blocks():
    """1 image at offset=12, length=8 (tokens 12..20). Block size 16 →
    overlaps blocks 0 + 1."""
    f = _MMFeature(mm_position=_PlaceholderRange(offset=12, length=8))
    assert _mm_block_overlap_set([f], range(0, 4), 16) == {0, 1}


def test_overlap_multiple_features():
    """Two images: img1 at [0..16), img2 at [40..56). Block size 16 →
    overlaps blocks 0 (img1) + 2,3 (img2)."""
    f1 = _MMFeature(mm_position=_PlaceholderRange(offset=0, length=16))
    f2 = _MMFeature(mm_position=_PlaceholderRange(offset=40, length=16))
    out = _mm_block_overlap_set([f1, f2], range(0, 4), 16)
    assert out == {0, 2, 3}


def test_overlap_feature_outside_block_range():
    """Feature at offset 1000 with block_range only covering blocks 0..3
    → no overlap."""
    f = _MMFeature(mm_position=_PlaceholderRange(offset=1000, length=10))
    assert _mm_block_overlap_set([f], range(0, 4), 16) == set()


def test_overlap_zero_length_feature_ignored():
    f = _MMFeature(mm_position=_PlaceholderRange(offset=4, length=0))
    assert _mm_block_overlap_set([f], range(0, 4), 16) == set()


def test_overlap_handles_malformed_feature():
    """Feature missing mm_position is silently skipped."""
    class _Bad:
        pass
    f_good = _MMFeature(mm_position=_PlaceholderRange(offset=4, length=8))
    out = _mm_block_overlap_set([_Bad(), f_good, None], range(0, 4), 16)
    assert out == {0}


def test_overlap_zero_block_size_returns_empty():
    f = _MMFeature(mm_position=_PlaceholderRange(offset=4, length=8))
    assert _mm_block_overlap_set([f], range(0, 4), 0) == set()


def test_overlap_partial_block_range():
    """block_range starting from blk_idx=2 (token offset 32+)."""
    f = _MMFeature(mm_position=_PlaceholderRange(offset=40, length=16))
    out = _mm_block_overlap_set([f], range(2, 5), 16)
    assert out == {2, 3}


# ─── notify_admit integration with real mm_features

def _install_two_tier() -> TierManager:
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


def test_notify_admit_per_block_mm_tagging():
    """Day 5 acceptance: only blocks overlapping MM range are mm_origin."""
    import os
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = _install_two_tier()
        # Image at tokens [0..32), block size 16 → blocks 0 + 1 are MM
        f = _MMFeature(mm_position=_PlaceholderRange(offset=0, length=32))
        req = _Request(request_id="r1", mm_features=[f])
        # Cached blocks 0..3 (4 blocks total)
        P.notify_admit(req, prev_n_cached=0, new_n_cached=4,
                        group_id=0, block_size=16)
        # Inspect bookkeeping
        rid = "r1"
        gid = "g0"
        # Block 0 + 1 should be mm_origin=True
        assert tm._pages[(rid, gid, 0)].mm_origin is True
        assert tm._pages[(rid, gid, 1)].mm_origin is True
        # Block 2 + 3 should be mm_origin=False (outside MM range)
        assert tm._pages[(rid, gid, 2)].mm_origin is False
        assert tm._pages[(rid, gid, 3)].mm_origin is False
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_notify_admit_no_mm_features_marks_all_text():
    """When mm_features is empty + no coarse fallback → all blocks text."""
    import os
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = _install_two_tier()
        req = _Request(request_id="r2", mm_features=[], has_mm_input=False)
        P.notify_admit(req, prev_n_cached=0, new_n_cached=4,
                        group_id=0, block_size=16)
        for blk_idx in range(4):
            assert tm._pages[("r2", "g0", blk_idx)].mm_origin is False
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_notify_admit_block_size_zero_falls_back_to_coarse():
    """When block_size=0 (legacy patch caller) + coarse has_mm_input=True,
    all blocks marked mm_origin=True (coarse fallback)."""
    import os
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = _install_two_tier()
        req = _Request(request_id="r3", mm_features=[], has_mm_input=True)
        P.notify_admit(req, prev_n_cached=0, new_n_cached=2,
                        group_id=0, block_size=0)
        # Coarse fallback: both blocks mm_origin=True
        assert tm._pages[("r3", "g0", 0)].mm_origin is True
        assert tm._pages[("r3", "g0", 1)].mm_origin is True
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_notify_admit_two_images_far_apart():
    """Two images at distant offsets → only their blocks tagged."""
    import os
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        tm = _install_two_tier()
        # img1 at [0..16) (block 0); img2 at [80..96) (block 5)
        f1 = _MMFeature(mm_position=_PlaceholderRange(offset=0, length=16))
        f2 = _MMFeature(mm_position=_PlaceholderRange(offset=80, length=16))
        req = _Request(request_id="r4", mm_features=[f1, f2])
        P.notify_admit(req, prev_n_cached=0, new_n_cached=8,
                        group_id=0, block_size=16)
        for blk_idx in range(8):
            expected_mm = blk_idx in (0, 5)
            actual = tm._pages[("r4", "g0", blk_idx)].mm_origin
            assert actual is expected_mm, (
                f"block {blk_idx}: expected mm_origin={expected_mm}, got {actual}"
            )
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


def test_notify_admit_silent_when_block_size_invalid():
    """block_size negative or non-int does not crash."""
    import os
    os.environ["GENESIS_ENABLE_PN95_TIER_AWARE_CACHE"] = "1"
    try:
        _install_two_tier()
        req = _Request(request_id="r5", mm_features=[
            _MMFeature(mm_position=_PlaceholderRange(offset=0, length=8))
        ])
        # Negative block_size should be treated as 0 → coarse fallback
        P.notify_admit(req, prev_n_cached=0, new_n_cached=2,
                        group_id=0, block_size=-1)
        # No crash; whether marked or not is fallback-defined
    finally:
        os.environ.pop("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", None)


# ─── Wire-in shape: text-patch passes block_size

def test_text_patch_passes_block_size():
    """The PN95 text-patch must pass `self.block_size` to notify_admit."""
    from sndr.engines.vllm.patches.kv_cache import pn95_tier_aware_cache as M
    assert "self.block_size" in M.PN95_SITE1_NEW, (
        "Day 5 wire-in must pass block_size to notify_admit"
    )

# SPDX-License-Identifier: Apache-2.0
"""Path C v1.0 Phase 2 (PN95) — real bytes movement tests.

Tests `TierManager.demote_block(layer, block_idx)` + `.promote_block(...)`
using mock GPU tensors that simulate vllm dev93's `dict[str, Tensor]`
attention layer shape: `(num_blocks, block_size, K_or_V, packed_features)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from sndr.cache.tier_manager import TierManager
from sndr.model_configs.schema import CacheTier


SLOT = 49664  # actual per-block bytes for 27B PROD: 64 × 2 × 388 = 49664


# ─── Mock GPU tensor that mimics torch.Tensor's bare-bones API

class _MockGpuTensor:
    """Mimics torch.Tensor[block_idx] view + .copy_() semantics in Python."""

    def __init__(self, num_blocks: int, bytes_per_block: int):
        self.num_blocks = num_blocks
        self.bytes_per_block = bytes_per_block
        # Underlying byte storage as bytearray (one slot per block)
        self._storage = [bytearray(bytes_per_block) for _ in range(num_blocks)]

    def __getitem__(self, idx):
        # Return a view = the inner bytearray
        return _MockGpuView(self._storage[idx])

    @property
    def shape(self):
        return (self.num_blocks, self.bytes_per_block, 1, 1)


class _MockGpuView:
    """Mimics torch.Tensor view: contiguous() + view(-1) + copy_() + .numel()."""

    def __init__(self, ba: bytearray):
        self._ba = ba

    def contiguous(self):
        return self

    def view(self, *args):
        return self

    def numel(self):
        return len(self._ba)

    @property
    def shape(self):
        return (len(self._ba),)

    @property
    def device(self):
        return "cuda:0"

    @property
    def dtype(self):
        return "uint8"

    def to(self, dtype, copy=False):
        return self

    def copy_(self, src, non_blocking=False):
        # If src has _ba attr, byte-copy
        if hasattr(src, "_ba"):
            n = min(len(self._ba), len(src._ba))
            self._ba[:n] = src._ba[:n]
        return self

    def __setitem__(self, idx, val):
        # For slice assignments
        if hasattr(val, "_ba"):
            self._ba[idx] = val._ba[idx if isinstance(idx, slice) else slice(0, len(val._ba))]


# ─── Test fixtures

def _two_tier_manager_with_views(force_bytearray: bool = True) -> TierManager:
    """Build TierManager with attention_views populated synthetically."""
    tm = TierManager(
        tiers=[
            CacheTier(device="gpu", capacity_gib=0.001),  # tiny
            CacheTier(device="cpu", capacity_gib=0.01),
        ],
        slot_nbytes=SLOT,
    )
    # Force bytearray slab for tests (no torch needed)
    tm._cpu_slab._force_bytearray = True
    # Populate _attention_views (what register_kv_caches would do)
    tm._attention_views = {
        "layer.0.self_attn.attn": {
            "tensor": _MockGpuTensor(num_blocks=10, bytes_per_block=SLOT),
            "num_blocks": 10,
            "bytes_per_block": SLOT,
            "device": "cuda:0",
        },
        "layer.4.self_attn.attn": {
            "tensor": _MockGpuTensor(num_blocks=10, bytes_per_block=SLOT),
            "num_blocks": 10,
            "bytes_per_block": SLOT,
            "device": "cuda:1",
        },
    }
    return tm


# ─── demote_block

def test_demote_block_returns_false_when_no_attention_views():
    tm = TierManager(
        tiers=[
            CacheTier(device="gpu", capacity_gib=0.001),
            CacheTier(device="cpu", capacity_gib=0.01),
        ],
        slot_nbytes=SLOT,
    )
    # No _attention_views set
    assert tm.demote_block("layer.0.self_attn.attn", 0) is False


def test_demote_block_returns_false_unknown_layer():
    tm = _two_tier_manager_with_views()
    assert tm.demote_block("layer.99.self_attn.attn", 0) is False


def test_demote_block_returns_false_block_out_of_range():
    tm = _two_tier_manager_with_views()
    assert tm.demote_block("layer.0.self_attn.attn", 100) is False
    assert tm.demote_block("layer.0.self_attn.attn", -1) is False


def test_demote_block_returns_false_when_single_tier():
    tm = TierManager(
        tiers=[CacheTier(device="gpu", capacity_gib=0.001)],
        slot_nbytes=SLOT,
    )
    tm._attention_views = {"layer.0": {
        "tensor": _MockGpuTensor(10, SLOT), "num_blocks": 10,
        "bytes_per_block": SLOT, "device": "cuda:0",
    }}
    assert tm.demote_block("layer.0", 0) is False


def test_demote_block_succeeds_round_trip():
    """demote_block + promote_block must round-trip bytes correctly."""
    tm = _two_tier_manager_with_views()
    # Write distinctive bytes to GPU block
    layer = "layer.0.self_attn.attn"
    src_tensor = tm._attention_views[layer]["tensor"]
    sentinel = bytes([0xAB] * SLOT)
    src_tensor._storage[3][:] = sentinel

    # Demote
    ok = tm.demote_block(layer, 3)
    assert ok is True
    # Page bookkeeping
    key = (layer, 3)
    assert key in tm._pages
    assert tm._pages[key].tier_idx == 1
    assert tm._pages[key].cpu_slot_idx is not None

    # Zero out the GPU block (simulate overwrite by next request)
    src_tensor._storage[3][:] = bytes(SLOT)

    # Promote back
    ok = tm.promote_block(layer, 3)
    assert ok is True
    assert tm._pages[key].tier_idx == 0
    # Bytes restored to GPU block
    assert bytes(src_tensor._storage[3]) == sentinel


def test_demote_block_increments_cpu_slab_used():
    tm = _two_tier_manager_with_views()
    n_before = tm._cpu_slab.n_used()
    tm.demote_block("layer.0.self_attn.attn", 0)
    n_after = tm._cpu_slab.n_used()
    assert n_after == n_before + 1


def test_demote_multiple_blocks_in_same_layer():
    tm = _two_tier_manager_with_views()
    layer = "layer.0.self_attn.attn"
    for blk in range(5):
        ok = tm.demote_block(layer, blk)
        assert ok is True
    # All 5 in tier 1
    for blk in range(5):
        assert tm._pages[(layer, blk)].tier_idx == 1
    assert tm._cpu_slab.n_used() == 5


# ─── promote_block

def test_promote_block_returns_false_for_unknown_page():
    tm = _two_tier_manager_with_views()
    # Never demoted → no page in bookkeeping
    assert tm.promote_block("layer.0.self_attn.attn", 0) is False


def test_promote_block_returns_false_when_already_in_tier0():
    tm = _two_tier_manager_with_views()
    layer = "layer.0.self_attn.attn"
    tm.admit((layer, 0), group_id="attn_eligible")  # admits at tier 0
    assert tm.promote_block(layer, 0) is False


# ─── n_attention_layers_eligible

def test_n_attention_layers_eligible_zero_default():
    tm = TierManager(
        tiers=[CacheTier(device="gpu", capacity_gib=0.001)],
        slot_nbytes=SLOT,
    )
    assert tm.n_attention_layers_eligible() == 0


def test_n_attention_layers_eligible_after_register():
    tm = _two_tier_manager_with_views()
    assert tm.n_attention_layers_eligible() == 2


# ─── Cross-layer demote (different GPU devices)

def test_demote_two_layers_on_different_devices():
    tm = _two_tier_manager_with_views()
    # layer.0 is cuda:0, layer.4 is cuda:1
    ok0 = tm.demote_block("layer.0.self_attn.attn", 0)
    ok1 = tm.demote_block("layer.4.self_attn.attn", 0)
    assert ok0 is True
    assert ok1 is True
    assert tm._cpu_slab.n_used() == 2

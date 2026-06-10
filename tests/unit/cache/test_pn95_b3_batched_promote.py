# SPDX-License-Identifier: Apache-2.0
"""PN95 Quality-First Sprint Q1 B3 — batched async promote tests.

Mirror of B2 (test_pn95_b2_batched_demote) for CPU→GPU copy direction.
Validates `_pn95_cpu_to_gpu_copy_batch` correctness:
- Batch result identical to N sequential _pn95_cpu_to_gpu_copy calls
- Mismatched len(views) != len(src_bytes_list) → returns 0
- Empty input handled
- Stats counters incremented correctly
- Order preserved
"""
from __future__ import annotations

import pytest

from sndr.cache import _pn95_runtime as rt


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(rt, "_PN95_CUDA_STREAM", None)
    base_stats = {
        **rt._PN95_STATS,
        "async_promote_count": 0,
        "async_batch_promote_count": 0,
    }
    monkeypatch.setattr(rt, "_PN95_STATS", base_stats)
    yield


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─── API contract (no CUDA needed) ─────────────────────────────────────


def test_promote_batch_empty_returns_zero():
    """Empty inputs return 0."""
    assert rt._pn95_cpu_to_gpu_copy_batch([], []) == 0


def test_promote_batch_mismatched_lengths_returns_zero():
    """Length mismatch is unsafe — return 0 instead of partial copy."""
    assert rt._pn95_cpu_to_gpu_copy_batch([None], [b"a", b"b"]) == 0
    assert rt._pn95_cpu_to_gpu_copy_batch([None, None], [b"a"]) == 0


def test_promote_batch_signature():
    import inspect
    sig = inspect.signature(rt._pn95_cpu_to_gpu_copy_batch)
    assert len(sig.parameters) == 2


def test_get_pn95_stats_exposes_b3_field(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    stats = rt.get_pn95_stats()
    assert "async_batch_promote_count" in stats
    assert isinstance(stats["async_batch_promote_count"], int)


# ─── CUDA correctness ──────────────────────────────────────────────────


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_promote_batch_writes_correct_bytes_per_view():
    """Batch CPU→GPU copies must populate each view with correct bytes."""
    import torch
    # 5 distinct GPU target views, all initially zeros
    views = [torch.zeros(64, dtype=torch.uint8, device="cuda") for _ in range(5)]
    # 5 distinct source byte arrays
    src_bytes_list = [bytes([i] * 64) for i in (10, 20, 30, 40, 50)]

    n = rt._pn95_cpu_to_gpu_copy_batch(views, src_bytes_list)
    assert n == 5

    torch.cuda.synchronize()
    for i, (view, expected_val) in enumerate(zip(views, (10, 20, 30, 40, 50))):
        assert bool((view == expected_val).all().item()), \
            f"View {i}: expected all bytes={expected_val}"


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_promote_batch_increments_counters_correctly(monkeypatch):
    """Single batch call: async_promote_count += N, async_batch_promote_count += 1."""
    monkeypatch.setattr(rt, "_PN95_STATS", {
        **rt._PN95_STATS,
        "async_promote_count": 0,
        "async_batch_promote_count": 0,
    })
    import torch
    views = [torch.zeros(32, dtype=torch.uint8, device="cuda") for _ in range(7)]
    src_list = [bytes(32) for _ in range(7)]
    rt._pn95_cpu_to_gpu_copy_batch(views, src_list)
    assert rt._PN95_STATS["async_promote_count"] == 7
    assert rt._PN95_STATS["async_batch_promote_count"] == 1


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_promote_batch_fallback_when_async_disabled(monkeypatch):
    """Sync fallback when GENESIS_PN95_ASYNC_STREAM=0 still correct."""
    monkeypatch.setenv("GENESIS_PN95_ASYNC_STREAM", "0")
    import torch
    views = [torch.zeros(64, dtype=torch.uint8, device="cuda") for _ in range(3)]
    src_list = [bytes([100] * 64), bytes([200] * 64), bytes([99] * 64)]
    n = rt._pn95_cpu_to_gpu_copy_batch(views, src_list)
    torch.cuda.synchronize()
    assert n == 3
    assert int(views[0][0].item()) == 100
    assert int(views[1][0].item()) == 200
    assert int(views[2][0].item()) == 99


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_promote_batch_round_trip_matches_demote_batch():
    """Critical: B2 demote batch + B3 promote batch round-trip = byte-identical."""
    import torch
    # Source: 5 distinct GPU views
    src_views = [
        torch.full((128,), i, dtype=torch.uint8, device="cuda")
        for i in (11, 22, 33, 44, 55)
    ]
    # Demote (B2)
    cpu_bytes_list = rt._pn95_gpu_to_cpu_bytes_batch(src_views)
    # Wipe targets
    targets = [torch.zeros(128, dtype=torch.uint8, device="cuda") for _ in range(5)]
    # Promote (B3)
    n = rt._pn95_cpu_to_gpu_copy_batch(targets, cpu_bytes_list)
    torch.cuda.synchronize()
    assert n == 5
    for i, (t, src) in enumerate(zip(targets, src_views)):
        assert bool((t == src).all().item()), \
            f"Round-trip view {i} byte mismatch"

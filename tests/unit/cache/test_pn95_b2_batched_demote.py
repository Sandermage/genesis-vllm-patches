# SPDX-License-Identifier: Apache-2.0
"""PN95 Quality-First Sprint Q1 B2 — batched async demote tests.

Validates `_pn95_gpu_to_cpu_bytes_batch` correctness:
- Batch result identical to N sequential _pn95_gpu_to_cpu_bytes calls
- Empty list handled
- Stats counters incremented correctly
- CUDA stream batching reduces sync count (verified indirectly)

CUDA-dependent tests skipped when no CUDA is available.
"""
from __future__ import annotations

import pytest

from sndr.cache import _pn95_runtime as rt


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset module-level cache state per test."""
    monkeypatch.setattr(rt, "_PN95_CUDA_STREAM", None)
    # Reset stat counters via fresh dict
    base_stats = {
        **rt._PN95_STATS,
        "async_demote_count": 0,
        "async_batch_demote_count": 0,
    }
    monkeypatch.setattr(rt, "_PN95_STATS", base_stats)
    yield


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ─── API contract tests (work without CUDA) ──────────────────────────


def test_batch_empty_returns_empty():
    """Empty list input → empty list output."""
    assert rt._pn95_gpu_to_cpu_bytes_batch([]) == []


def test_batch_signature():
    """Helper takes one list arg, returns list."""
    import inspect
    sig = inspect.signature(rt._pn95_gpu_to_cpu_bytes_batch)
    assert len(sig.parameters) == 1


def test_get_pn95_stats_exposes_batch_field(monkeypatch):
    """B2 stat field must appear in stats snapshot."""
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    stats = rt.get_pn95_stats()
    assert "async_batch_demote_count" in stats
    assert isinstance(stats["async_batch_demote_count"], int)


# ─── CUDA correctness tests (skipped without CUDA) ───────────────────


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_batch_produces_identical_bytes_to_sequential():
    """Batch result must be byte-identical to sequential helper calls."""
    import torch
    # Create 5 distinct GPU tensors
    views = [
        torch.arange(start=i * 256, end=(i + 1) * 256, dtype=torch.uint8, device="cuda")
        for i in range(5)
    ]
    expected = [bytes(v.cpu().numpy().tobytes()) for v in views]

    batched = rt._pn95_gpu_to_cpu_bytes_batch(views)

    assert len(batched) == len(expected)
    for i, (got, want) in enumerate(zip(batched, expected)):
        assert got == want, f"View {i} byte mismatch in batch path"


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_batch_increments_counters_correctly(monkeypatch):
    """Single batch call increments async_demote_count by N (views count)
    AND async_batch_demote_count by 1 (one batch op)."""
    monkeypatch.setattr(rt, "_PN95_STATS", {
        **rt._PN95_STATS,
        "async_demote_count": 0,
        "async_batch_demote_count": 0,
    })
    import torch
    views = [
        torch.zeros(64, dtype=torch.uint8, device="cuda")
        for _ in range(7)
    ]
    rt._pn95_gpu_to_cpu_bytes_batch(views)
    assert rt._PN95_STATS["async_demote_count"] == 7
    assert rt._PN95_STATS["async_batch_demote_count"] == 1


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_batch_fallback_when_async_disabled(monkeypatch):
    """When GENESIS_PN95_ASYNC_STREAM=0, batch helper still works correctly
    via sync fallback path."""
    monkeypatch.setenv("GENESIS_PN95_ASYNC_STREAM", "0")
    import torch
    views = [
        torch.arange(64, dtype=torch.uint8, device="cuda"),
        torch.arange(start=64, end=128, dtype=torch.uint8, device="cuda"),
    ]
    expected = [bytes(v.cpu().numpy().tobytes()) for v in views]
    out = rt._pn95_gpu_to_cpu_bytes_batch(views)
    assert out == expected


@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
def test_batch_preserves_order():
    """Output bytes must be in same order as input views."""
    import torch
    # Distinct values so order matters
    views = [
        torch.full((32,), i, dtype=torch.uint8, device="cuda")
        for i in (10, 20, 30, 40)
    ]
    out = rt._pn95_gpu_to_cpu_bytes_batch(views)
    assert len(out) == 4
    for i, expected_val in enumerate((10, 20, 30, 40)):
        assert out[i][0] == expected_val, f"Position {i}: order violated"

# SPDX-License-Identifier: Apache-2.0
"""PN95 Quality-First Sprint Q1 B4 — parallel batched compression tests.

Validates `_pn95_compress_bytes_batch` correctness:
- Output identical to sequential _pn95_compress_bytes calls (per-element)
- Order preserved
- Empty list, single-element edge cases
- Decompresses correctly back to original
- Pool lazy-init
"""
from __future__ import annotations

import pytest

from sndr.cache import _pn95_runtime as rt


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(rt, "_PN95_CUDA_STREAM", None)
    monkeypatch.setattr(rt, "_PN95_COMPRESS_LIB", None)
    monkeypatch.setattr(rt, "_PN95_COMPRESS_LEVEL", None)
    yield


def _kv_like_bytes(n: int, seed: int = 0) -> bytes:
    """KV-cache-like sparse byte pattern."""
    out = bytearray(n)
    for i in range(0, n, 20):
        out[i] = ((i + seed) * 31) & 0xFF
        if i + 1 < n:
            out[i + 1] = (((i + seed) * 17) >> 4) & 0xFF
    return bytes(out)


# ─── Edge cases ───────────────────────────────────────────────────────


def test_compress_batch_empty():
    """Empty input returns empty output."""
    assert rt._pn95_compress_bytes_batch([]) == []


def test_compress_batch_single_element_uses_sequential():
    """Single element doesn't need pool overhead — uses sequential path."""
    data = [_kv_like_bytes(50_000)]
    out = rt._pn95_compress_bytes_batch(data)
    expected = [rt._pn95_compress_bytes(data[0])]
    assert out == expected


def test_compress_batch_signature():
    import inspect
    sig = inspect.signature(rt._pn95_compress_bytes_batch)
    assert len(sig.parameters) == 1


# ─── Correctness: batch == sequential per-element ────────────────────


def test_compress_batch_matches_sequential_per_element(monkeypatch):
    """Critical: batch results must equal sequential per-element compression."""
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")

    # 17 distinct payloads (matching 27B attention layer count)
    payloads = [_kv_like_bytes(50_000, seed=i) for i in range(17)]

    batched = rt._pn95_compress_bytes_batch(payloads)
    sequential = [rt._pn95_compress_bytes(p) for p in payloads]

    assert len(batched) == len(sequential) == 17
    for i, (b, s) in enumerate(zip(batched, sequential)):
        assert b == s, f"Position {i}: batch != sequential"


def test_compress_batch_preserves_order(monkeypatch):
    """Output order matches input order."""
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")

    # Distinct payloads — order verifiable via decompression
    payloads = [_kv_like_bytes(5_000, seed=i) for i in (10, 20, 30, 40, 50)]
    out = rt._pn95_compress_bytes_batch(payloads)

    # Decompress each and verify matches expected position
    for i, (compressed, original) in enumerate(zip(out, payloads)):
        restored = rt._pn95_decompress_bytes(compressed)
        assert restored == original, f"Position {i}: order violated"


def test_compress_batch_round_trip_byte_identical(monkeypatch):
    """Critical: round-trip via batch compress + sequential decompress = identical."""
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")

    payloads = [_kv_like_bytes(20_000, seed=i) for i in range(8)]
    compressed = rt._pn95_compress_bytes_batch(payloads)
    decompressed = [rt._pn95_decompress_bytes(c) for c in compressed]

    for i, (orig, restored) in enumerate(zip(payloads, decompressed)):
        assert restored == orig, f"Round-trip view {i} byte mismatch"


# ─── Pool initialization ─────────────────────────────────────────────


def test_compress_pool_lazy_init():
    """Pool created on first call."""
    rt._PN95_COMPRESS_POOL = None
    pool = rt._pn95_compress_pool()
    assert pool is not None  # ThreadPoolExecutor available in CPython


def test_compress_pool_cached():
    """Subsequent calls return same instance."""
    rt._PN95_COMPRESS_POOL = None
    pool1 = rt._pn95_compress_pool()
    pool2 = rt._pn95_compress_pool()
    assert pool1 is pool2


# ─── Disabled compression handled correctly ──────────────────────────


def test_compress_batch_with_compression_disabled(monkeypatch):
    """When GENESIS_PN95_CPU_COMPRESS=none, batch returns originals."""
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "none")
    payloads = [_kv_like_bytes(5_000, seed=i) for i in range(3)]
    out = rt._pn95_compress_bytes_batch(payloads)
    assert out == payloads

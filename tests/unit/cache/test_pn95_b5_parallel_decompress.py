# SPDX-License-Identifier: Apache-2.0
"""PN95 Quality-First Sprint Q1 B5 — parallel batched decompression tests.

Mirror of B4 for decompress path. Validates:
- Output identical to sequential _pn95_decompress_bytes per-element
- Order preserved
- Empty list, single-element edge cases
- Backward-compatible (uncompressed data passes through)
- Round-trip with B4 compress batch byte-identical
"""
from __future__ import annotations

import pytest

from sndr.cache import _pn95_runtime as rt


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    monkeypatch.setattr(rt, "_PN95_COMPRESS_LIB", None)
    monkeypatch.setattr(rt, "_PN95_COMPRESS_LEVEL", None)
    yield


def _kv_like_bytes(n: int, seed: int = 0) -> bytes:
    out = bytearray(n)
    for i in range(0, n, 20):
        out[i] = ((i + seed) * 31) & 0xFF
        if i + 1 < n:
            out[i + 1] = (((i + seed) * 17) >> 4) & 0xFF
    return bytes(out)


# ─── Edge cases ──────────────────────────────────────────────────────


def test_decompress_batch_empty():
    assert rt._pn95_decompress_bytes_batch([]) == []


def test_decompress_batch_single_uses_sequential():
    pytest.importorskip("zstandard")
    raw = _kv_like_bytes(10_000)
    compressed = [rt._pn95_compress_bytes(raw)]
    out = rt._pn95_decompress_bytes_batch(compressed)
    assert len(out) == 1
    assert out[0] == raw


def test_decompress_batch_signature():
    import inspect
    sig = inspect.signature(rt._pn95_decompress_bytes_batch)
    assert len(sig.parameters) == 1


# ─── Correctness ─────────────────────────────────────────────────────


def test_decompress_batch_matches_sequential(monkeypatch):
    """Batch decompress result must equal sequential per-element."""
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")

    payloads = [_kv_like_bytes(20_000, seed=i) for i in range(17)]
    compressed = [rt._pn95_compress_bytes(p) for p in payloads]

    batched = rt._pn95_decompress_bytes_batch(compressed)
    sequential = [rt._pn95_decompress_bytes(c) for c in compressed]

    assert len(batched) == len(sequential) == 17
    for i, (b, s) in enumerate(zip(batched, sequential)):
        assert b == s, f"Position {i}: batch != sequential"


def test_decompress_batch_round_trip_with_compress_batch(monkeypatch):
    """Critical: B4 compress batch → B5 decompress batch = byte-identical."""
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")

    payloads = [_kv_like_bytes(30_000, seed=i) for i in range(8)]
    compressed = rt._pn95_compress_bytes_batch(payloads)
    decompressed = rt._pn95_decompress_bytes_batch(compressed)

    for i, (orig, restored) in enumerate(zip(payloads, decompressed)):
        assert restored == orig, f"Round-trip view {i} byte mismatch"


def test_decompress_batch_preserves_order(monkeypatch):
    """Output order matches input order."""
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")

    payloads = [_kv_like_bytes(5_000, seed=i) for i in (10, 20, 30, 40, 50)]
    compressed = [rt._pn95_compress_bytes(p) for p in payloads]

    decompressed = rt._pn95_decompress_bytes_batch(compressed)
    for i, (orig, restored) in enumerate(zip(payloads, decompressed)):
        assert restored == orig, f"Position {i}: order violated"


# ─── Backward compatibility ──────────────────────────────────────────


def test_decompress_batch_handles_uncompressed_passthrough():
    """Mixed compressed + uncompressed entries: backward-compat."""
    pytest.importorskip("zstandard")
    raw = _kv_like_bytes(10_000)
    compressed = rt._pn95_compress_bytes(raw)

    # Mixed list: compressed + raw (backward-compat with pre-A1 entries)
    mixed_input = [compressed, raw, compressed]
    out = rt._pn95_decompress_bytes_batch(mixed_input)

    assert len(out) == 3
    assert out[0] == raw       # compressed → decompressed
    assert out[1] == raw       # raw → passthrough (no magic bytes)
    assert out[2] == raw       # compressed → decompressed


def test_decompress_batch_with_compression_disabled(monkeypatch):
    """When compression disabled at write time, batch passes through reads."""
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "none")
    raws = [_kv_like_bytes(5_000, seed=i) for i in range(3)]
    # Stored raws (no compression)
    out = rt._pn95_decompress_bytes_batch(raws)
    assert out == raws

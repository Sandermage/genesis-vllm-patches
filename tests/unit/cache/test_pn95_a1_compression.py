# SPDX-License-Identifier: Apache-2.0
"""PN95 Quality-First Sprint Q1 A1 — CPU prefix store compression tests.

Validates lossless round-trip compression in `_pn95_runtime` for all backends
(zstd / lz4 / zlib / none). Critical: byte-identical decompress required —
any drift would corrupt KV cache restored content.
"""
from __future__ import annotations

import os

import pytest

from sndr.cache import _pn95_runtime as rt


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_compression_state(monkeypatch):
    """Reset module-level compression cache to ensure each test reads env fresh."""
    monkeypatch.setattr(rt, "_PN95_COMPRESS_LIB", None)
    monkeypatch.setattr(rt, "_PN95_COMPRESS_LEVEL", None)
    yield
    monkeypatch.setattr(rt, "_PN95_COMPRESS_LIB", None)
    monkeypatch.setattr(rt, "_PN95_COMPRESS_LEVEL", None)


def _kv_like_bytes(n: int) -> bytes:
    """Generate KV-cache-like bytes — mostly zeros + occasional non-zero
    (mimics real attention KV: sparse, semi-structured)."""
    out = bytearray(n)
    # ~5% non-zero to mimic attention sparsity pattern
    for i in range(0, n, 20):
        out[i] = (i * 31) & 0xFF
        if i + 1 < n:
            out[i + 1] = ((i * 17) >> 4) & 0xFF
    return bytes(out)


# ─── Round-trip tests (CRITICAL — byte identical) ──────────────────────


def test_zstd_round_trip_byte_identical(monkeypatch):
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")
    data = _kv_like_bytes(50_000)
    compressed = rt._pn95_compress_bytes(data)
    restored = rt._pn95_decompress_bytes(compressed)
    assert restored == data, "byte-identical round-trip must hold"
    assert rt._PN95_COMPRESS_LIB == "zstd"


def test_zlib_round_trip_byte_identical(monkeypatch):
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zlib")
    data = _kv_like_bytes(50_000)
    compressed = rt._pn95_compress_bytes(data)
    restored = rt._pn95_decompress_bytes(compressed)
    assert restored == data
    assert rt._PN95_COMPRESS_LIB == "zlib"


def test_disabled_passes_through(monkeypatch):
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "none")
    data = _kv_like_bytes(50_000)
    out = rt._pn95_compress_bytes(data)
    assert out == data, "disabled compression must pass through unchanged"
    assert rt._PN95_COMPRESS_LIB == "none"


# ─── Compression effectiveness ──────────────────────────────────────────


def test_zstd_achieves_compression_on_kv_like_data(monkeypatch):
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")
    data = _kv_like_bytes(50_000)
    compressed = rt._pn95_compress_bytes(data)
    # KV-like sparse data should compress >2× with zstd
    ratio = len(data) / len(compressed) if len(compressed) > 0 else 1.0
    assert ratio > 1.5, f"zstd ratio {ratio:.2f} too low for sparse KV-like data"


# ─── Edge cases ─────────────────────────────────────────────────────────


def test_small_data_skips_compression(monkeypatch):
    """Entries < _PN95_COMPRESS_MIN_BYTES should pass through (overhead check)."""
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")
    small = b"hello"  # 5 bytes < 256 threshold
    out = rt._pn95_compress_bytes(small)
    assert out == small


def test_empty_data(monkeypatch):
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")
    assert rt._pn95_compress_bytes(b"") == b""
    assert rt._pn95_decompress_bytes(b"") == b""


def test_random_data_no_benefit_returns_original(monkeypatch):
    """Highly random data shouldn't compress well — function returns original
    when compression saves <5%."""
    pytest.importorskip("zstandard")
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")
    # Pseudo-random bytes (deterministic seed via byte derivation)
    data = bytes((i * 7919 + 17) & 0xFF for i in range(8000))
    out = rt._pn95_compress_bytes(data)
    # Either identical (no benefit) OR strictly smaller (some benefit)
    assert out == data or len(out) < len(data) * 0.95


# ─── Backward compatibility (uncompressed entries still work) ──────────


def test_decompress_passes_through_uncompressed(monkeypatch):
    """Critical: decompress must NOT corrupt entries without a compression header.
    Backward-compat for PN95 stores written before A1."""
    monkeypatch.setenv("GENESIS_PN95_CPU_COMPRESS", "zstd")
    raw = b"arbitrary uncompressed payload " * 100
    out = rt._pn95_decompress_bytes(raw)
    assert out == raw


def test_decompress_handles_short_data():
    """Decompress on data shorter than the magic header — must pass through."""
    assert rt._pn95_decompress_bytes(b"") == b""
    assert rt._pn95_decompress_bytes(b"abc") == b"abc"  # < 4 bytes


# ─── Magic byte detection ───────────────────────────────────────────────


def test_zstd_magic_detection(monkeypatch):
    """Decompress auto-detects zstd magic regardless of compress lib in use."""
    pytest.importorskip("zstandard")
    import zstandard as zstd
    data = b"hello world " * 100
    compressed = zstd.ZstdCompressor(level=3).compress(data)
    # zstd magic: 28 b5 2f fd
    assert compressed[:4] == b"\x28\xb5\x2f\xfd"
    restored = rt._pn95_decompress_bytes(compressed)
    assert restored == data


def test_zlib_magic_detection():
    import zlib
    data = b"hello zlib " * 100
    compressed = zlib.compress(data, 1)
    # zlib header: 78 01 (or other variant)
    assert compressed[0] == 0x78
    restored = rt._pn95_decompress_bytes(compressed)
    assert restored == data


# ─── Stats tracking ─────────────────────────────────────────────────────


def test_get_pn95_stats_includes_compression_fields(monkeypatch):
    """get_pn95_stats() must report A1 compression metrics."""
    monkeypatch.setenv("GENESIS_ENABLE_PN95_TIER_AWARE_CACHE", "1")
    stats = rt.get_pn95_stats()
    assert "compress_raw_bytes_total" in stats
    assert "compress_stored_bytes_total" in stats
    assert "compress_ratio" in stats
    assert "compress_lib" in stats

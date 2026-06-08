# SPDX-License-Identifier: Apache-2.0
"""PN95 CPU prefix compression + pickle pack/unpack helpers.

Eight helpers split across two concerns:

  Pack / unpack (used by L1 pinned pool + disk tier round-trips):
    * ``_pn95_pack_layer_data``    — pickle.dumps wrapper for layer
                                      tuples
    * ``_pn95_unpack_layer_data``  — safe-allowlist pickle.Unpickler
                                      (review finding #16)

  Lossless byte compression (Sprint Q1 A1):
    * ``_pn95_init_compression``       — lazy-init zstd|lz4|zlib|none
                                          per GENESIS_PN95_CPU_COMPRESS
    * ``_pn95_compress_bytes``         — single-payload compressor
    * ``_pn95_compress_pool``          — lazy ThreadPoolExecutor for
                                          parallel batched compress
    * ``_pn95_compress_bytes_batch``   — parallel batched compress
                                          (Sprint Q1 B4)
    * ``_pn95_decompress_bytes_batch`` — parallel batched decompress
                                          (Sprint Q1 B5)
    * ``_pn95_decompress_bytes``       — single-payload decompressor;
                                          magic-byte auto-detect for
                                          backward-compat with
                                          pre-A1 uncompressed entries

M.4.2.C scope: function extraction only. The five mutable state
singletons that govern compression behaviour stay defined in
``_pn95_runtime`` because four test files (``test_pn95_a1_compression``,
``test_pn95_b4_parallel_compress``, ``test_pn95_b5_parallel_decompress``)
actively rebind them via ``monkeypatch.setattr(rt, ...)`` and direct
``rt._PN95_COMPRESS_POOL = None`` writes. Moving the names would break
those test contracts (the alias-fragility class first surfaced as the
``_PN95_STATS`` drift in M.4.1; same fix applies):

  _PN95_COMPRESS_LIB        — selected backend or "none"
  _PN95_COMPRESS_LEVEL      — backend-specific tuning level
  _PN95_COMPRESS_MIN_BYTES  — skip threshold (entries smaller pass
                              through uncompressed)
  _PN95_ZSTD_TL             — threading.local() for per-thread cached
                              zstd compressor / decompressor objects
  _PN95_COMPRESS_POOL       — ThreadPoolExecutor for batch calls

All five live in ``_pn95_runtime`` and the moved functions reach
them via lazy ``_rt.X``. The two original ``global`` rebind sites
(``_pn95_init_compression`` mutates LIB+LEVEL, ``_pn95_compress_pool``
mutates POOL) are replicated via explicit attribute mutation —
``_rt._PN95_COMPRESS_LIB = ...`` evaluates as
``setattr(_rt, "_PN95_COMPRESS_LIB", ...)`` and operates on the same
module-attribute slot the original ``global`` declaration mutated.

The legacy module re-exports all eight functions so existing tests
that do ``rt._pn95_compress_bytes(...)`` etc. keep working without
edit; no text-anchor regen (no patch references any of these names).
"""
from __future__ import annotations

import io
import logging
import os
import pickle
from typing import Optional

log = logging.getLogger("genesis.pn95")


# ─── Pack / unpack ──────────────────────────────────────────────────────


def _pn95_pack_layer_data(layer_data: list) -> bytes:
    return pickle.dumps(layer_data, protocol=pickle.HIGHEST_PROTOCOL)


def _pn95_unpack_layer_data(blob: bytes) -> Optional[list]:
    """Unpickle the layer-data blob from a pinned-pool slot or disk tier.

    Uses a strict allow-list of class lookups (review finding #16):
    a corrupted slot or a maliciously-crafted disk file MUST NOT be able
    to invoke arbitrary code via pickle's `__reduce__` / `find_class`.
    KV payloads are pure (str, bytes) tuples in a list, no custom classes
    needed; everything else is rejected.
    """
    if not blob:
        return None

    class _PN95SafeUnpickler(pickle.Unpickler):
        _ALLOWED = frozenset({
            ("builtins", "str"),
            ("builtins", "bytes"),
            ("builtins", "list"),
            ("builtins", "tuple"),
            ("builtins", "int"),
            ("builtins", "float"),
            ("builtins", "bool"),
            ("builtins", "dict"),
            ("builtins", "NoneType"),
        })

        def find_class(self, module: str, name: str):
            if (module, name) in self._ALLOWED:
                return super().find_class(module, name)
            raise pickle.UnpicklingError(
                f"[PN95] pickle class not allow-listed: {module}.{name}"
            )

    try:
        obj = _PN95SafeUnpickler(io.BytesIO(blob)).load()
    except (pickle.UnpicklingError, EOFError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, list) else None


# ─── Compression backend init + bytes-level compress / decompress ──────


def _pn95_init_compression() -> None:
    """Lazy-init compression backend on first use.

    Reads GENESIS_PN95_CPU_COMPRESS env: 'zstd'|'lz4'|'zlib'|'none'|'auto'.
    Default 'auto' = prefer zstd > lz4 > zlib > none.

    GENESIS_PN95_COMPRESS_LEVEL controls compression level:
      zstd: 1-22 (default 3 = balanced speed/ratio)
      zlib: 1-9 (default 1 = fast)
      lz4: ignored (single level)
    """
    from sndr.cache import _pn95_runtime as _rt
    if _rt._PN95_COMPRESS_LIB is not None:
        return
    requested = os.environ.get("GENESIS_PN95_CPU_COMPRESS", "auto").strip().lower()
    if requested in ("none", "off", "0", "disabled"):
        _rt._PN95_COMPRESS_LIB = "none"
        return
    # Try zstd first (best ratio + decent speed)
    if requested in ("auto", "zstd"):
        try:
            import zstandard  # noqa: F401
            _rt._PN95_COMPRESS_LIB = "zstd"
            try:
                _rt._PN95_COMPRESS_LEVEL = int(os.environ.get(
                    "GENESIS_PN95_COMPRESS_LEVEL", "3"))
            except (ValueError, TypeError):
                _rt._PN95_COMPRESS_LEVEL = 3
            log.info("[PN95 A1] CPU compression: zstd level=%d",
                     _rt._PN95_COMPRESS_LEVEL)
            return
        except ImportError:
            if requested == "zstd":
                log.warning("[PN95 A1] zstandard not installed, trying lz4")
    # Try lz4 (faster, less compression)
    if requested in ("auto", "lz4"):
        try:
            import lz4.frame  # noqa: F401
            _rt._PN95_COMPRESS_LIB = "lz4"
            log.info("[PN95 A1] CPU compression: lz4")
            return
        except ImportError:
            if requested == "lz4":
                log.warning("[PN95 A1] lz4 not installed, trying zlib")
    # Fallback to stdlib zlib (always available)
    if requested in ("auto", "zlib"):
        _rt._PN95_COMPRESS_LIB = "zlib"
        try:
            _rt._PN95_COMPRESS_LEVEL = int(os.environ.get(
                "GENESIS_PN95_COMPRESS_LEVEL", "1"))
        except (ValueError, TypeError):
            _rt._PN95_COMPRESS_LEVEL = 1
        log.info("[PN95 A1] CPU compression: zlib level=%d (stdlib fallback)",
                 _rt._PN95_COMPRESS_LEVEL)
        return
    _rt._PN95_COMPRESS_LIB = "none"


def _pn95_compress_bytes(data: bytes) -> bytes:
    """Compress bytes via configured backend. Returns compressed OR original
    if compression disabled / failed / no benefit.

    Compression backend writes a magic header (zstd/lz4/zlib all do); the
    symmetric _pn95_decompress_bytes auto-detects via magic check.
    """
    _pn95_init_compression()
    from sndr.cache import _pn95_runtime as _rt
    lib = _rt._PN95_COMPRESS_LIB
    if lib in ("none", None):
        return data
    if len(data) < _rt._PN95_COMPRESS_MIN_BYTES:
        return data
    try:
        if lib == "zstd":
            # Sprint Q1 B6 — per-thread cached compressor (avoid alloc per call,
            # avoid race in B4 ThreadPool path).
            cctx = getattr(_rt._PN95_ZSTD_TL, "cctx", None)
            if cctx is None:
                import zstandard as zstd
                cctx = zstd.ZstdCompressor(level=_rt._PN95_COMPRESS_LEVEL or 3)
                _rt._PN95_ZSTD_TL.cctx = cctx
            compressed = cctx.compress(data)
        elif lib == "lz4":
            import lz4.frame
            compressed = lz4.frame.compress(data)
        elif lib == "zlib":
            import zlib
            compressed = zlib.compress(data, _rt._PN95_COMPRESS_LEVEL or 1)
        else:
            return data
    except Exception:
        return data
    # Only use compression if it saved >5% (avoid overhead on already-compressed data)
    if len(compressed) >= int(len(data) * 0.95):
        return data
    return compressed


def _pn95_compress_pool():
    """Path C v1.0 Sprint Q1 B4 — lazy-init ThreadPoolExecutor for parallel
    compression. zstd/lz4/zlib release GIL during compression — multiple
    threads truly parallel.

    Returns None if threading unavailable (which doesn't happen in CPython).
    Default 4 workers (env GENESIS_PN95_COMPRESS_THREADS).
    """
    from sndr.cache import _pn95_runtime as _rt
    if _rt._PN95_COMPRESS_POOL is None:
        try:
            from concurrent.futures import ThreadPoolExecutor
            try:
                workers = int(os.environ.get("GENESIS_PN95_COMPRESS_THREADS", "4"))
            except (ValueError, TypeError):
                workers = 4
            workers = max(1, min(workers, 16))  # clamp [1, 16]
            _rt._PN95_COMPRESS_POOL = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="pn95-compress"
            )
        except Exception:
            return None
    return _rt._PN95_COMPRESS_POOL


def _pn95_compress_bytes_batch(data_list: list) -> list:
    """Path C v1.0 Sprint Q1 B4 — parallel batched compression.

    Compress N bytes objects concurrently via ThreadPool. zstd/lz4/zlib
    release Python GIL during native compression → real parallelism.

    For 17-layer demote with ~100KB blocks: sequential = ~1.7ms total,
    parallel (4 threads) = ~0.5ms total = ~3-4× speedup.

    Returns list of compressed bytes in same order. Empty list if input empty.
    Falls back to sequential if pool unavailable.
    """
    if not data_list:
        return []
    pool = _pn95_compress_pool()
    if pool is None or len(data_list) <= 1:
        return [_pn95_compress_bytes(d) for d in data_list]
    # Parallel — submit all, collect ordered results
    futures = [pool.submit(_pn95_compress_bytes, d) for d in data_list]
    return [f.result() for f in futures]


def _pn95_decompress_bytes_batch(data_list: list) -> list:
    """Path C v1.0 Sprint Q1 B5 — parallel batched decompression.

    Mirror of B4 (_pn95_compress_bytes_batch) for the promote path. zstd/lz4/zlib
    release Python GIL during decompression → real parallelism.

    For 17-layer promote with mixed compressed sizes:
    sequential ~340μs total, parallel (4 threads) ~85μs total = ~4× speedup.

    Returns list of decompressed bytes in same order. Backward-compatible:
    uncompressed entries pass through unchanged (auto-detected via magic bytes
    in the underlying _pn95_decompress_bytes).
    """
    if not data_list:
        return []
    pool = _pn95_compress_pool()
    if pool is None or len(data_list) <= 1:
        return [_pn95_decompress_bytes(d) for d in data_list]
    # Parallel — submit all, collect ordered results
    futures = [pool.submit(_pn95_decompress_bytes, d) for d in data_list]
    return [f.result() for f in futures]


def _pn95_decompress_bytes(data: bytes) -> bytes:
    """Auto-detect compression via magic bytes and decompress. Returns
    original bytes if no compression detected (backward-compatible —
    handles uncompressed entries from before A1, mixed-format stores).
    """
    if len(data) < 4:
        return data
    from sndr.cache import _pn95_runtime as _rt
    # zstd frame magic: 28 b5 2f fd
    if data[:4] == b'\x28\xb5\x2f\xfd':
        try:
            # Sprint Q1 B6 — per-thread cached decompressor.
            dctx = getattr(_rt._PN95_ZSTD_TL, "dctx", None)
            if dctx is None:
                import zstandard as zstd
                dctx = zstd.ZstdDecompressor()
                _rt._PN95_ZSTD_TL.dctx = dctx
            return dctx.decompress(data)
        except Exception:
            return data
    # lz4 frame magic: 04 22 4d 18
    if data[:4] == b'\x04\x22\x4d\x18':
        try:
            import lz4.frame
            return lz4.frame.decompress(data)
        except Exception:
            return data
    # zlib header (RFC 1950): 0x78 (CMF) + check byte (variable)
    # Common values: 0x78 0x01, 0x78 0x5e, 0x78 0x9c, 0x78 0xda
    if data[0] == 0x78 and data[1] in (0x01, 0x5e, 0x9c, 0xda):
        try:
            import zlib
            return zlib.decompress(data)
        except Exception:
            return data
    return data

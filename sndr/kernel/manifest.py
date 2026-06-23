# SPDX-License-Identifier: Apache-2.0
"""SNDR Core — Site Map manifest cache (process-wide singleton).

Manifest fast-path (P2.1 Phase 3, 2026-05-07) — module-private cache
loaded lazily on the first `TextPatcher.apply()` call that has a
non-None `patch_id`. Sentinel distinguishes "not yet tried" from
"tried, manifest absent" to avoid retry storm (50+ patches each
trying to load the same missing file).

Public helpers consumed by `core.text_patch.TextPatcher`:
  - `cached_load_manifest()` — process-wide manifest dict, or None.
  - `reset_manifest_cache_for_tests()` — TEST-ONLY reset hook.
  - `derive_rel_path_from_target(target_file)` — strip vllm prefix
    to get manifest-relative path.
  - `md5_bytes(data)` — MD5 hex used for anchor_md5 comparison.

Migration history:
  - Original: vllm/_genesis/wiring/text_patch.py (Stage 0).
  - Stage 3 (CURRENT): split out of TextPatcher into this module.
    The legacy `text_patch.py` becomes a re-export shim.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("genesis.wiring.text_patch.manifest_cache")


# ─────────────────────────────────────────────────────────────────────────
# Module-private cache state.
#
# Three states for the cache:
#   _MANIFEST_NOT_LOADED — load not attempted yet (initial state).
#   _MANIFEST_INVALID    — load tried, returned None (file absent /
#                          corrupted / pin mismatch). Subsequent calls
#                          return None without reloading.
#   <dict>               — successfully loaded manifest dict.
# ─────────────────────────────────────────────────────────────────────────
_MANIFEST_NOT_LOADED = object()
_MANIFEST_INVALID = object()
_MANIFEST_CACHE: Any = _MANIFEST_NOT_LOADED


def cached_load_manifest() -> Optional[dict]:
    """Return process-wide manifest dict, or None if unavailable.

    Loaded once on first call. All TextPatcher instances share the
    cached dict. Returns None when:
      - the manifest JSON is not present on disk
      - the JSON is corrupted
      - the manifest's pinned vllm/genesis versions don't match the
        installed versions

    Pin verification: `vllm.__version__` AND `sndr.version.GENESIS_VERSION`
    are both checked against `manifest.pins.*`. Any mismatch returns None
    and falls through to legacy O(N×M) anchor scan in TextPatcher.
    """
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is _MANIFEST_INVALID:
        return None
    if _MANIFEST_CACHE is not _MANIFEST_NOT_LOADED:
        return _MANIFEST_CACHE  # cached dict

    # First call — attempt load
    try:
        import vllm  # type: ignore
        vllm_pin = getattr(vllm, "__version__", None)
    except Exception:
        vllm_pin = None
    try:
        # Prefer canonical sndr_core version; fall back to legacy alias.
        try:
            from sndr import GENESIS_VERSION as gver
        except Exception:
            from sndr.version import __version__ as gver
        genesis_pin = gver
    except Exception:
        genesis_pin = None

    # v9.0 (2026-05-07): manifest path resolved via sndr_paths registry
    # (single source of truth) instead of hardcoded `_genesis/manifests/`.
    # Honors `SNDR_MANIFEST_DIR` env override; falls back to auto-detect
    # of legacy/canonical location.
    try:
        from sndr.engines.vllm.wiring.anchor_manifest import (
            load_manifest_for_pins,
            per_pin_manifest_path,
        )
        from sndr.engines.vllm.locations.project_paths import manifest_json_path
        # Phase 3 (2026-06-21): prefer the per-pin source-of-truth manifest
        # pins/<pin>/anchors.json; fall back to the legacy single committed
        # manifest when no per-pin file exists for the running pin. Either way
        # load_manifest_for_pins enforces the pin match, and any md5 mismatch
        # at apply time falls back to the inline anchor (authoritative+fallback).
        per_pin = per_pin_manifest_path(vllm_pin)
        manifest_path = (
            str(per_pin)
            if (per_pin is not None and per_pin.is_file())
            else str(manifest_json_path())
        )
        manifest = load_manifest_for_pins(
            manifest_path,
            vllm_pin=vllm_pin, genesis_pin=genesis_pin,
        )
    except Exception as e:
        log.debug("manifest load raised: %s — falling back to legacy", e)
        manifest = None

    if manifest is None:
        _MANIFEST_CACHE = _MANIFEST_INVALID
        return None
    _MANIFEST_CACHE = manifest
    return manifest


def reset_manifest_cache_for_tests() -> None:
    """Test-only: reset manifest cache so tests can set up custom
    fixture state across test cases. NEVER call from production."""
    global _MANIFEST_CACHE
    _MANIFEST_CACHE = _MANIFEST_NOT_LOADED


def derive_rel_path_from_target(target_file: str) -> Optional[str]:
    """Strip vllm install prefix from `target_file` to get manifest rel_path.

    Examples:
      `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/foo.py`
        → "model_executor/foo.py"
      `/home/dev/genesis-vllm-patches/vllm/_genesis/tests/pristine_fixtures/chunk.py`
        (test fixture path, "vllm" present but inside Genesis tree)
        → "_genesis/tests/pristine_fixtures/chunk.py"  (manifest miss → fallback)

    Strategy: take everything after the LAST `vllm` segment. Forward-slash
    separated for posix consistency (manifest keys are posix-style).

    Returns None when no `vllm` segment found.
    """
    if not target_file:
        return None
    parts = Path(target_file).parts
    try:
        idx = len(parts) - 1 - list(reversed(parts)).index("vllm")
    except ValueError:
        return None
    rel_parts = parts[idx + 1:]
    if not rel_parts:
        return None
    return "/".join(rel_parts)


def md5_bytes(data: bytes) -> str:
    """MD5 hex of raw bytes — must match anchor_manifest.compute_anchor_meta."""
    return hashlib.md5(data).hexdigest()


# ─────────────────────────────────────────────────────────────────────────
# Back-compat aliases for existing code that imports the leading-underscore
# names (`_cached_load_manifest`, etc.). Removed at v9.0.
# ─────────────────────────────────────────────────────────────────────────
_cached_load_manifest = cached_load_manifest
_reset_manifest_cache_for_tests = reset_manifest_cache_for_tests
_derive_rel_path_from_target = derive_rel_path_from_target
_md5_bytes = md5_bytes


__all__ = [
    "cached_load_manifest",
    "reset_manifest_cache_for_tests",
    "derive_rel_path_from_target",
    "md5_bytes",
    # legacy aliases:
    "_cached_load_manifest",
    "_reset_manifest_cache_for_tests",
    "_derive_rel_path_from_target",
    "_md5_bytes",
]

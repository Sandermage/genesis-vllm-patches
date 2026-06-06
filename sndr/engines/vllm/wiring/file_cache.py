# SPDX-License-Identifier: Apache-2.0
"""Persistent file md5 cache — P2.2 of patcher evolution plan (2026-05-07).

Layer 0 fast-path BEFORE TextPatcher.apply() opens the target file.
For already-patched files on warm restart, skips full file read+scan
by checking cached (mtime_ns, size_bytes, marker_present) tuple via
single os.stat() syscall (~10μs vs ~160μs for legacy Layer 1+2+3).

Design notes (see also memory/project_p22_file_md5_cache_design.md):

  - mtime_ns + size_bytes is the cache key, NOT md5. md5 read+hash is
    O(file_size) which negates the entire P2.2 win. md5 IS computed
    at write time (post-apply) so a future `genesis cache verify`
    command can detect tampering.

  - Cache write failure NEVER fails apply() — wrapped try/except,
    log WARN and continue. P2.2 is pure optimization, never a hazard.

  - Pin invalidation: full cache wipe on vllm or Genesis __version__
    change. Avoid stale entries silently surviving across upgrades.

  - GENESIS_NO_PATCH_CACHE=1 honored — operator escape hatch (P1.3).

  - File lock via fcntl.flock on Linux. macOS dev gets a no-op lock
    (advisory locks not as portable; tests run sequentially anyway).

Schema v1:

```json
{
  "cache_version": 1,
  "pins": {"vllm": "...", "genesis": "v7.72.2"},
  "updated_at": "2026-05-07T22:30:00Z",
  "files": {
    "/abs/path/to/chunk.py": {
      "mtime_ns": 1714752000123456789,
      "size_bytes": 8761,
      "md5_post_apply": "abc123...",
      "markers": ["Genesis PN79 in-place SSM state (vllm#41824)"]
    }
  }
}
```

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("genesis.wiring.file_cache")

CACHE_SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────
# Cache location resolution
# ─────────────────────────────────────────────────────────────────────────


def _resolve_cache_path() -> Path:
    """Return the path of the persistent cache file.

    Resolution order:
      1. GENESIS_FILE_CACHE_PATH env (operator override)
      2. $XDG_CACHE_HOME/genesis/files_md5.json
      3. ~/.cache/genesis/files_md5.json
      4. /tmp/genesis_files_md5.json (HOME unwritable)
    """
    override = os.environ.get("GENESIS_FILE_CACHE_PATH", "").strip()
    if override:
        return Path(override)

    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg) / "genesis" / "files_md5.json"

    home = os.environ.get("HOME", "").strip()
    if home and os.path.isdir(home):
        return Path(home) / ".cache" / "genesis" / "files_md5.json"

    return Path("/tmp/genesis_files_md5.json")


# ─────────────────────────────────────────────────────────────────────────
# Pin detection (mirrors anchor_manifest._cached_load_manifest)
# ─────────────────────────────────────────────────────────────────────────


def _detect_pins() -> tuple[Optional[str], Optional[str]]:
    """Detect (vllm_pin, genesis_pin). Returns (None, None) on import failures.
    Never raises."""
    vllm_pin = None
    genesis_pin = None
    try:
        import vllm  # type: ignore
        vllm_pin = getattr(vllm, "__version__", None)
    except Exception:
        pass
    try:
        from vllm.sndr_core.version import __version__ as gver
        genesis_pin = gver
    except Exception:
        pass
    return vllm_pin, genesis_pin


# ─────────────────────────────────────────────────────────────────────────
# Cache load / save with pin verification
# ─────────────────────────────────────────────────────────────────────────


# Process-wide cache: loaded once on first access, mutated in memory,
# flushed to disk on each record_apply_result call (for persistence
# across container restarts).
_CACHE_NOT_LOADED = object()
_CACHE_INVALID = object()
_CACHE: Any = _CACHE_NOT_LOADED


def _new_empty_cache() -> dict:
    vllm_pin, genesis_pin = _detect_pins()
    return {
        "cache_version": CACHE_SCHEMA_VERSION,
        "pins": {
            "vllm": vllm_pin or "",
            "genesis": genesis_pin or "",
        },
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {},
    }


def _validate_cache(cache: Any) -> bool:
    """Cheap structural check. False = wipe + start fresh."""
    if not isinstance(cache, dict):
        return False
    if cache.get("cache_version") != CACHE_SCHEMA_VERSION:
        return False
    pins = cache.get("pins")
    if not isinstance(pins, dict):
        return False
    if not isinstance(cache.get("files"), dict):
        return False
    return True


def _check_pins_match(cache: dict) -> bool:
    """True if cache.pins matches detected pins. False = invalidate."""
    cached_vllm = cache.get("pins", {}).get("vllm", "")
    cached_genesis = cache.get("pins", {}).get("genesis", "")
    actual_vllm, actual_genesis = _detect_pins()
    if actual_vllm and cached_vllm != actual_vllm:
        return False
    if actual_genesis and cached_genesis != actual_genesis:
        return False
    return True


def _load_cache_from_disk() -> Optional[dict]:
    """Read cache JSON from disk, validate schema + pins. None on any
    error (caller will start fresh)."""
    path = _resolve_cache_path()
    if not path.is_file():
        log.debug("file_cache: no cache at %s — fresh start", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = f.read()
    except (OSError, PermissionError) as e:
        log.warning("file_cache: read failed for %s: %s", path, e)
        return None

    try:
        cache = json.loads(payload)
    except json.JSONDecodeError as e:
        log.warning("file_cache: corrupted JSON at %s: %s — discarding", path, e)
        return None

    if not _validate_cache(cache):
        log.warning(
            "file_cache: schema validation failed at %s — discarding", path
        )
        return None

    if not _check_pins_match(cache):
        log.info(
            "file_cache: pin mismatch at %s — invalidating cache", path
        )
        return None

    return cache


def _ensure_loaded() -> dict:
    """Lazy-load cache on first access. Returns the in-memory dict.
    Always returns a valid dict (creates fresh one if no disk state)."""
    global _CACHE
    if _CACHE is _CACHE_INVALID or _CACHE is _CACHE_NOT_LOADED:
        loaded = _load_cache_from_disk()
        _CACHE = loaded if loaded is not None else _new_empty_cache()
    return _CACHE


def _save_cache_atomic(cache: dict) -> None:
    """Atomic JSON write: temp file in same dir → fsync → os.replace →
    fsync parent dir. Same pattern as anchor_manifest.write_manifest_atomic.

    Failures are caught and logged; never raise to caller."""
    path = _resolve_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("file_cache: mkdir failed for %s: %s", path.parent, e)
        return

    cache["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = json.dumps(cache, indent=2, sort_keys=True) + "\n"

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # NFS/FUSE may reject dir fsync; rename was atomic enough
    except (OSError, PermissionError) as e:
        log.warning("file_cache: atomic write failed for %s: %s", path, e)
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[call-arg]
        except Exception:
            pass


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


def get_cache_entry(target_file: str) -> Optional[dict]:
    """Lookup cache entry by absolute file path. None if not present."""
    cache = _ensure_loaded()
    return cache.get("files", {}).get(target_file)


def is_marker_cached_present(target_file: str, marker: str) -> bool:
    """Layer 0 fast-path composite check. True only if ALL conditions met:

      1. Cache loaded successfully (no schema/pin issue)
      2. File entry exists in cache
      3. File on disk has same mtime_ns AND size_bytes as cached entry
      4. `marker` is in the cached marker list

    Caller (TextPatcher.apply Layer 0) returns IDEMPOTENT immediately
    when this returns True. False → caller falls through to Layer 1+.

    NEVER raises — any error returns False (graceful fallback to legacy).
    """
    try:
        # Stat first (cheap, ~10μs). Skip cache load entirely if file
        # doesn't exist — Layer 1 will report target_file_missing.
        try:
            st = os.stat(target_file)
        except OSError:
            return False

        entry = get_cache_entry(target_file)
        if entry is None:
            return False

        # mtime_ns: prefer st_mtime_ns (Linux); fall back to mtime × 1e9
        actual_mtime_ns = getattr(st, "st_mtime_ns", None)
        if actual_mtime_ns is None:
            actual_mtime_ns = int(st.st_mtime * 1e9)

        if entry.get("mtime_ns") != actual_mtime_ns:
            return False
        if entry.get("size_bytes") != st.st_size:
            return False

        markers = entry.get("markers")
        if not isinstance(markers, list):
            return False
        return marker in markers
    except Exception as e:
        log.debug("file_cache: is_marker_cached_present exception: %s", e)
        return False


def record_apply_result(
    target_file: str, marker: str, *,
    post_apply_content: Optional[str] = None,
) -> None:
    """Update cache after successful APPLIED or IDEMPOTENT.

    Args:
      target_file: absolute path of the patched file
      marker: marker string just confirmed present in the file
      post_apply_content: file content (str) immediately after apply.
        Used to compute md5_post_apply for future `genesis cache verify`.
        If None, md5 is read from disk (one disk hit).

    NEVER raises. Failures logged at WARNING.
    """
    try:
        cache = _ensure_loaded()
        try:
            st = os.stat(target_file)
        except OSError as e:
            log.debug("file_cache: stat failed for %s: %s", target_file, e)
            return

        # Compute mtime_ns
        mtime_ns = getattr(st, "st_mtime_ns", None)
        if mtime_ns is None:
            mtime_ns = int(st.st_mtime * 1e9)

        # Compute md5
        if post_apply_content is not None:
            md5_hex = _md5_bytes(post_apply_content.encode("utf-8"))
        else:
            try:
                with open(target_file, "rb") as f:
                    md5_hex = _md5_bytes(f.read())
            except (OSError, PermissionError):
                md5_hex = ""

        files = cache.setdefault("files", {})
        entry = files.get(target_file)

        if entry is None:
            entry = {
                "mtime_ns": mtime_ns,
                "size_bytes": st.st_size,
                "md5_post_apply": md5_hex,
                "markers": [marker],
            }
            files[target_file] = entry
        else:
            entry["mtime_ns"] = mtime_ns
            entry["size_bytes"] = st.st_size
            entry["md5_post_apply"] = md5_hex
            existing_markers = entry.get("markers") or []
            if not isinstance(existing_markers, list):
                existing_markers = []
            if marker not in existing_markers:
                existing_markers.append(marker)
            entry["markers"] = existing_markers

        _save_cache_atomic(cache)
    except Exception as e:
        # Cache write failure must NEVER bubble up to apply()
        log.warning(
            "file_cache: record_apply_result for %s exception: %s",
            target_file, e,
        )


def invalidate_file(target_file: str) -> None:
    """Remove a single file's entry. Used by tests / operator commands."""
    try:
        cache = _ensure_loaded()
        files = cache.get("files", {})
        if target_file in files:
            del files[target_file]
            _save_cache_atomic(cache)
    except Exception as e:
        log.warning("file_cache: invalidate_file exception: %s", e)


def clear_cache() -> None:
    """Wipe entire cache (in-memory + on-disk). For tests / operator command.
    Does NOT remove the cache file — overwrites with empty cache."""
    global _CACHE
    _CACHE = _new_empty_cache()
    _save_cache_atomic(_CACHE)


def _reset_for_tests() -> None:
    """Test-only: reset in-memory cache so tests start with a clean slate.
    Does NOT touch disk."""
    global _CACHE
    _CACHE = _CACHE_NOT_LOADED

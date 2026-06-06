# SPDX-License-Identifier: Apache-2.0
"""PN95 Tier 3 — disk-backed prefix cache (LMCache / SGLang HiCache parity).

When the in-memory CPU prefix store (Tier 2) is full and a fresh
demote needs to land, the LRU victim spills to this disk tier rather
than being discarded. Disk-tier entries survive container restarts,
let prefix cache scale to the size of the host filesystem (typically
tens or hundreds of GiB on a modern NVMe), and are bytes-on-the-wire
identical to the RAM entries — the same demote / promote layer_data
shape (list of `(layer_name, stored_bytes)` tuples).

Public API
----------

    disk_tier_set(block_hash, layer_data)        → True on success
    disk_tier_get(block_hash) → layer_data | None
    disk_tier_delete(block_hash)                  → True on success
    disk_tier_stats() → dict
    disk_tier_evict_oldest() → bytes evicted
    disk_tier_reset_for_tests() → None

All entries live as `<dir>/<HEX>.pn95` blobs whose filename is a
SHA-256 of the block hash representation (so any vllm BlockHash type
projects to a safe filesystem name). The blob format is a
length-prefixed pickle of layer_data — no shared in-memory state with
the caller, no torch / cuda dependencies, single thread safety
guaranteed by a module-level lock.

Env knobs
---------

    GENESIS_PN95_DISK_TIER_DIR
        Storage directory. Default: $HOME/.sndr/cache/pn95_disk.
        Created on first set() if missing.

    GENESIS_PN95_DISK_TIER_CAPACITY_GIB
        Soft byte budget. Default: 50.0 (≈ 50 GiB). When the
        on-disk total would exceed this, the oldest blob (by atime)
        is unlinked before the new write.

    GENESIS_PN95_DISK_TIER_ENABLE
        Master gate. Default: "0" (off). Set to "1" to enable. The
        disk tier is opt-in — operators who run on a small root
        partition or do not want disk activity must explicitly turn
        it on.

Design choices
--------------

* No mmap. Each blob is short-lived (read in full on promote, written
  in full on demote). mmap would amortize across many promotes but
  would also require careful lifecycle management with the GIL-held
  prefix-store lock. A straight read/write is simpler and tail-latency
  is dominated by the cudaMemcpyHostToDevice anyway.

* Pickle. We already serialize the layer_data list as the in-memory
  Tier 2 value (a list of `(str, bytes)` tuples). Pickle preserves the
  exact shape. zstd compression has ALREADY been applied at Tier 2
  ingestion (`_pn95_compress_bytes_batch`) — no recompression here.

* LRU via atime. Filesystems on Linux honour atime by default, so we
  use it as cheap LRU. On filesystems mounted with `noatime` the
  eviction picks the oldest mtime instead — still safe because the
  file was last written when it landed.

* No background sweeper. Eviction runs synchronously inside set() on
  the demote hot path. Operators who want async eviction can override
  `GENESIS_PN95_DISK_TIER_CAPACITY_GIB` to a value comfortably above
  steady-state working set so set() rarely needs to evict.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple

log = logging.getLogger("genesis.pn95.disk_tier")

# Module-level state.
_LOCK = threading.Lock()
_DIR_CACHE: Optional[Path] = None
_CAPACITY_BYTES_CACHE: Optional[int] = None
_ENABLED_CACHE: Optional[bool] = None
_STATS: dict = {
    "disk_writes_total": 0,
    "disk_reads_total": 0,
    "disk_read_hits_total": 0,
    "disk_evictions_total": 0,
    "disk_bytes_written_total": 0,
    "disk_bytes_read_total": 0,
    "disk_bytes_evicted_total": 0,
    "disk_last_io_error": None,
}


def _enabled() -> bool:
    """Master gate. Reads `GENESIS_PN95_DISK_TIER_ENABLE` once per
    process and caches. `reset_for_tests` clears the cache."""
    global _ENABLED_CACHE
    if _ENABLED_CACHE is None:
        raw = os.environ.get("GENESIS_PN95_DISK_TIER_ENABLE", "0")
        _ENABLED_CACHE = raw.strip().lower() in ("1", "true", "yes", "on")
    return _ENABLED_CACHE


def _resolve_dir() -> Path:
    """Lazy-resolve and create the storage directory."""
    global _DIR_CACHE
    if _DIR_CACHE is not None:
        return _DIR_CACHE
    raw = os.environ.get("GENESIS_PN95_DISK_TIER_DIR", "")
    if raw:
        path = Path(raw).expanduser()
    else:
        home = Path(os.environ.get("HOME", "/root"))
        path = home / ".sndr" / "cache" / "pn95_disk"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("[PN95-disk] cannot create %s: %s — disk tier disabled", path, e)
        _STATS["disk_last_io_error"] = f"mkdir: {e!s}"
    _DIR_CACHE = path
    return _DIR_CACHE


def _capacity_bytes() -> int:
    """Cached soft byte cap from `GENESIS_PN95_DISK_TIER_CAPACITY_GIB`."""
    global _CAPACITY_BYTES_CACHE
    if _CAPACITY_BYTES_CACHE is None:
        try:
            gib = float(os.environ.get("GENESIS_PN95_DISK_TIER_CAPACITY_GIB", "50"))
        except ValueError:
            gib = 50.0
        _CAPACITY_BYTES_CACHE = int(max(0.0, gib) * (1 << 30))
    return _CAPACITY_BYTES_CACHE


def _hash_key(block_hash: Any) -> str:
    """SHA-256 a stringified block hash into a hex filename suffix.

    Using sha256 is overkill for collision avoidance but the cost is
    negligible (one block_hash is < 256 bytes) and it gives us a
    fixed-length filesystem-safe name across every BlockHash shape
    vllm emits — tuple, namedtuple, custom dataclass — `repr` is
    stable enough.
    """
    return hashlib.sha256(repr(block_hash).encode("utf-8")).hexdigest()


def _path_for(block_hash: Any) -> Path:
    return _resolve_dir() / f"{_hash_key(block_hash)}.pn95"


def _total_bytes_on_disk() -> int:
    """Walk the dir and sum file sizes. O(N) where N is entry count;
    called only by `_evict_until_fit` so amortized fine."""
    try:
        total = 0
        for fp in _resolve_dir().glob("*.pn95"):
            try:
                total += fp.stat().st_size
            except OSError:
                continue
        return total
    except OSError as e:
        _STATS["disk_last_io_error"] = f"_total_bytes: {e!s}"
        return 0


def _evict_until_fit(incoming_bytes: int) -> int:
    """Drop oldest entries (by atime, falling back to mtime when atime
    is unreliable) until `total + incoming <= capacity`. Returns the
    total bytes evicted.
    """
    cap = _capacity_bytes()
    if cap <= 0:
        return 0
    current = _total_bytes_on_disk()
    if current + incoming_bytes <= cap:
        return 0
    target = max(0, cap - incoming_bytes)
    try:
        entries: List[Tuple[float, int, Path]] = []
        for fp in _resolve_dir().glob("*.pn95"):
            try:
                st = fp.stat()
                # atime first, fall back to mtime (noatime mount).
                t = st.st_atime if st.st_atime > 0 else st.st_mtime
                entries.append((t, st.st_size, fp))
            except OSError:
                continue
        entries.sort(key=lambda e: e[0])  # oldest first
        evicted = 0
        for _t, size, fp in entries:
            if current <= target:
                break
            try:
                fp.unlink()
                current -= size
                evicted += size
                _STATS["disk_evictions_total"] += 1
                _STATS["disk_bytes_evicted_total"] += size
            except OSError as e:
                _STATS["disk_last_io_error"] = f"unlink {fp.name}: {e!s}"
                continue
        return evicted
    except OSError as e:
        _STATS["disk_last_io_error"] = f"_evict: {e!s}"
        return 0


def disk_tier_set(
    block_hash: Any,
    layer_data: List[Tuple[str, bytes]],
) -> bool:
    """Persist a (block_hash → layer_data) entry.

    Fail-silent: any OSError / pickle error returns False and leaves
    no partial file behind. The caller (demote_on_evict CPU-pool-full
    branch) treats False as 'spillover unavailable' and continues
    operating on Tier 2 only.
    """
    if not _enabled():
        return False
    if not layer_data:
        return False
    try:
        payload = pickle.dumps(layer_data, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PicklingError, TypeError) as e:
        _STATS["disk_last_io_error"] = f"pickle: {e!s}"
        return False
    with _LOCK:
        _evict_until_fit(len(payload))
        path = _path_for(block_hash)
        tmp = path.with_suffix(".pn95.tmp")
        try:
            with open(tmp, "wb") as fh:
                fh.write(payload)
            os.replace(tmp, path)
        except OSError as e:
            _STATS["disk_last_io_error"] = f"write {path.name}: {e!s}"
            try:
                tmp.unlink()
            except OSError:
                pass
            return False
        _STATS["disk_writes_total"] += 1
        _STATS["disk_bytes_written_total"] += len(payload)
        return True


def disk_tier_get(
    block_hash: Any,
) -> Optional[List[Tuple[str, bytes]]]:
    """Read an entry. Returns layer_data on hit, None on miss / error.

    Hit also bumps the file's atime so LRU eviction sees it as recent.
    """
    if not _enabled():
        return None
    with _LOCK:
        _STATS["disk_reads_total"] += 1
        path = _path_for(block_hash)
        if not path.is_file():
            return None
        try:
            with open(path, "rb") as fh:
                payload = fh.read()
            # Bump atime so this entry survives the next LRU sweep.
            try:
                os.utime(path, None)
            except OSError:
                pass
            _STATS["disk_bytes_read_total"] += len(payload)
        except OSError as e:
            _STATS["disk_last_io_error"] = f"read {path.name}: {e!s}"
            return None
        try:
            data = pickle.loads(payload)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            _STATS["disk_last_io_error"] = f"unpickle {path.name}: {e!s}"
            return None
        # Soft validation — same shape as Tier 2.
        if not isinstance(data, list):
            return None
        _STATS["disk_read_hits_total"] += 1
        return data


def disk_tier_delete(block_hash: Any) -> bool:
    """Remove a single entry. Returns True iff a file was unlinked."""
    if not _enabled():
        return False
    with _LOCK:
        path = _path_for(block_hash)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            _STATS["disk_last_io_error"] = f"delete {path.name}: {e!s}"
            return False


def disk_tier_evict_oldest() -> int:
    """Drop a single oldest entry (caller-driven manual eviction).
    Returns the bytes evicted, 0 on empty or error."""
    if not _enabled():
        return 0
    with _LOCK:
        try:
            entries: List[Tuple[float, int, Path]] = []
            for fp in _resolve_dir().glob("*.pn95"):
                try:
                    st = fp.stat()
                    t = st.st_atime if st.st_atime > 0 else st.st_mtime
                    entries.append((t, st.st_size, fp))
                except OSError:
                    continue
            if not entries:
                return 0
            entries.sort(key=lambda e: e[0])
            _t, size, fp = entries[0]
            try:
                fp.unlink()
                _STATS["disk_evictions_total"] += 1
                _STATS["disk_bytes_evicted_total"] += size
                return size
            except OSError as e:
                _STATS["disk_last_io_error"] = f"unlink {fp.name}: {e!s}"
                return 0
        except OSError as e:
            _STATS["disk_last_io_error"] = f"evict_oldest: {e!s}"
            return 0


def disk_tier_stats() -> dict:
    """Snapshot for observability. Walks the dir to compute live
    `disk_entries` and `disk_bytes_on_disk` — cheap when entry count
    is modest, the caller (CLI / stats dump) is throttled."""
    out = dict(_STATS)
    try:
        if _enabled():
            d = _resolve_dir()
            entries = list(d.glob("*.pn95"))
            out["disk_entries"] = len(entries)
            total = 0
            for fp in entries:
                try:
                    total += fp.stat().st_size
                except OSError:
                    continue
            out["disk_bytes_on_disk"] = total
            out["disk_capacity_bytes"] = _capacity_bytes()
            out["disk_dir"] = str(d)
        else:
            out["disk_entries"] = 0
            out["disk_bytes_on_disk"] = 0
            out["disk_capacity_bytes"] = 0
            out["disk_dir"] = None
    except OSError as e:
        _STATS["disk_last_io_error"] = f"stats: {e!s}"
    return out


def reset_for_tests() -> None:
    """Drop in-process caches so a test can use tempdir + env override."""
    global _DIR_CACHE, _CAPACITY_BYTES_CACHE, _ENABLED_CACHE
    _DIR_CACHE = None
    _CAPACITY_BYTES_CACHE = None
    _ENABLED_CACHE = None
    for k in (
        "disk_writes_total", "disk_reads_total", "disk_read_hits_total",
        "disk_evictions_total", "disk_bytes_written_total",
        "disk_bytes_read_total", "disk_bytes_evicted_total",
    ):
        _STATS[k] = 0
    _STATS["disk_last_io_error"] = None

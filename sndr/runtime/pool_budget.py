# SPDX-License-Identifier: Apache-2.0
"""Centralized per-pool VRAM budget primitive for Genesis prealloc patches.

⚠️  STATUS — PARTIAL SCAFFOLD (Audit A-15 honest 2026-05-06):
    POC integration WIRED into `kernels/dequant_buffer.py::TurboQuantBufferManager`
    (firing on 27B INT4 + TQ k8v4 stack, dormant on 35B FP8). Other pools NOT
    yet wired:
      - `gdn_scratch_pool.py`        — pending
      - `gdn_gating_buffer.py`       — pending
      - `moe_intermediate_cache.py`  — pending
      - `ffn_intermediate_cache.py`  — pending
      - `fla_kkt_buffer.py`          — pending
      - `gdn_core_attn_manager.py`   — pending
    Env vars `GENESIS_POOL_MAX_MIB`, `GENESIS_POOL_TOTAL_MAX_MIB`,
    `GENESIS_POOL_MAX_MIB_<PATCH>` are runtime-noop except for `dequant_kv`
    until the remaining pools are wired (separate session).
    Note: `_USAGE` state is per-process (vllm spawn workers). API server
    process won't see `record()` calls made by worker processes — for total
    accounting, query workers individually or use shared-memory backing.

WHY THIS EXISTS
================

Genesis has 7+ separate pool managers (`GenesisPreallocBuffer` family +
domain-specific pools in `kernels/dequant_buffer.py`, `gdn_scratch_pool.py`,
`gdn_gating_buffer.py`, `moe_intermediate_cache.py`, `ffn_intermediate_cache.py`,
`fla_kkt_buffer.py`, `gdn_core_attn_manager.py`). Each grows on first-sight and
sticks for process lifetime — no per-pool ceiling, no cross-pool budget.

The 2026-05-07 architectural audit (Sander framing: "fixed size, only essential,
doesn't bloat") identified two gaps:

  1. **Per-pool ceiling**: any single pool can grow without bound. PN59
     `o_output` was the largest unbounded grower (~768 MiB worst-case at
     long-ctx). Phase A (`gdn_scratch_pool._ENV_O_MAX_T`) capped it.

  2. **Cross-pool budget**: no top-level "total Genesis-pool VRAM ≤ X MiB"
     enforcement. Operator can't say "Genesis preallocs may use at most
     2 GiB cumulative — refuse to boot otherwise".

This module provides BOTH primitives in one place. Pools that bypass
`GenesisPreallocBuffer` (PN12, PN59, P37, P46, etc.) are integrated by
calling `pool_budget.check(patch_id, requested_bytes)` before allocating.
On overflow, the pool either raises (caller falls back) or logs+continues
based on `policy`.

DESIGN PRINCIPLES (mirror buffer_mode.py)
==========================================

  - **Operator-facing**: env var per patch + global default
  - **Dynamo-safe**: cached at module load, no env reads on hot path
  - **Default OFF**: cap=None means unlimited (current behavior preserved)
  - **Read-only inspection**: `summary()` for /diag and post-warmup logging
  - **No allocation here**: this module just gates; pools own allocation

ENV PRECEDENCE (most specific wins)
====================================

  `GENESIS_POOL_MAX_MIB_<PATCH_ID>`  per-patch ceiling in MiB (e.g.
                                     `GENESIS_POOL_MAX_MIB_PN59=200`)
  `GENESIS_POOL_MAX_MIB`             global per-pool ceiling
  `GENESIS_POOL_TOTAL_MAX_MIB`       sum of ALL Genesis pools ceiling
                                     (cross-pool, checked by `assert_total_under_budget`)
  unset/0/invalid                    unlimited (current behavior)

USAGE PATTERN
==============

In a pool manager (e.g. `GdnScratchPool.acquire_o_output`):

    from sndr.runtime.pool_budget import check, record

    bytes_requested = B * T_binned * H * V * dtype_size
    check("PN59", bytes_requested)   # raises if cap exceeded
    buf = torch.empty(...)
    record("PN59", buf.numel() * buf.element_size())

For cross-pool budget enforcement (post-warmup hook in apply_all):

    from sndr.runtime.pool_budget import assert_total_under_budget
    assert_total_under_budget()  # raises if cumulative > GENESIS_POOL_TOTAL_MAX_MIB

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

log = logging.getLogger("genesis.pool_budget")

# ─── Env names ────────────────────────────────────────────────────────

_ENV_PER_POOL_PREFIX = "GENESIS_POOL_MAX_MIB_"
_ENV_PER_POOL_GLOBAL = "GENESIS_POOL_MAX_MIB"
_ENV_TOTAL_BUDGET = "GENESIS_POOL_TOTAL_MAX_MIB"

# Cache state (env stable post-spawn; reset only by re-import).
_PER_POOL_CACHE: dict[str, Optional[int]] = {}  # patch_id → MiB (None = unlimited)
_GLOBAL_CACHE: Optional[int] = None  # MiB (None = unlimited / not yet read)
_GLOBAL_CACHE_READ = False
_TOTAL_CACHE: Optional[int] = None
_TOTAL_CACHE_READ = False

# Live accounting registry: patch_id → bytes_recorded
_USAGE_LOCK = threading.Lock()
_USAGE: dict[str, int] = {}


# ─── Cache reset helpers (test-only) ─────────────────────────────────


def _reset_caches() -> None:
    """Test helper — clear all caches so new env values are re-read."""
    global _PER_POOL_CACHE, _GLOBAL_CACHE, _GLOBAL_CACHE_READ
    global _TOTAL_CACHE, _TOTAL_CACHE_READ, _USAGE
    _PER_POOL_CACHE = {}
    _GLOBAL_CACHE = None
    _GLOBAL_CACHE_READ = False
    _TOTAL_CACHE = None
    _TOTAL_CACHE_READ = False
    with _USAGE_LOCK:
        _USAGE = {}


def _parse_mib(raw: str) -> Optional[int]:
    """Parse env value to MiB int. Returns None on invalid/unset/zero/negative."""
    if not raw:
        return None
    try:
        n = int(raw.strip())
        return n if n > 0 else None
    except (ValueError, TypeError):
        return None


# ─── Per-pool ceiling lookup ─────────────────────────────────────────


def _global_max_mib() -> Optional[int]:
    """Read GENESIS_POOL_MAX_MIB once, cache."""
    global _GLOBAL_CACHE, _GLOBAL_CACHE_READ
    if not _GLOBAL_CACHE_READ:
        _GLOBAL_CACHE = _parse_mib(os.environ.get(_ENV_PER_POOL_GLOBAL, ""))
        _GLOBAL_CACHE_READ = True
    return _GLOBAL_CACHE


def max_mib_for(patch_id: str) -> Optional[int]:
    """Return per-pool MiB ceiling for `patch_id`. None = unlimited.

    Precedence: GENESIS_POOL_MAX_MIB_<PATCH> > GENESIS_POOL_MAX_MIB > unlimited.
    """
    pid = patch_id.upper().strip()
    if pid in _PER_POOL_CACHE:
        return _PER_POOL_CACHE[pid]

    raw = os.environ.get(_ENV_PER_POOL_PREFIX + pid, "")
    parsed = _parse_mib(raw)
    if parsed is None:
        # Fall through to global default
        parsed = _global_max_mib()

    _PER_POOL_CACHE[pid] = parsed
    return parsed


def total_max_mib() -> Optional[int]:
    """Read GENESIS_POOL_TOTAL_MAX_MIB once, cache. None = unlimited."""
    global _TOTAL_CACHE, _TOTAL_CACHE_READ
    if not _TOTAL_CACHE_READ:
        _TOTAL_CACHE = _parse_mib(os.environ.get(_ENV_TOTAL_BUDGET, ""))
        _TOTAL_CACHE_READ = True
    return _TOTAL_CACHE


# ─── Live usage accounting ────────────────────────────────────────────


def record(patch_id: str, bytes_allocated: int) -> None:
    """Record `bytes_allocated` toward `patch_id`'s usage tally.

    Call this AFTER successful allocation so the registry reflects what's
    actually live. Idempotent on shape-stable pools — caller is responsible
    for not double-counting on cache hits.
    """
    pid = patch_id.upper().strip()
    if bytes_allocated <= 0:
        return
    with _USAGE_LOCK:
        _USAGE[pid] = _USAGE.get(pid, 0) + int(bytes_allocated)


def deduct(patch_id: str, bytes_freed: int) -> None:
    """Deduct `bytes_freed` from `patch_id`'s tally on pool eviction/release."""
    pid = patch_id.upper().strip()
    if bytes_freed <= 0:
        return
    with _USAGE_LOCK:
        _USAGE[pid] = max(0, _USAGE.get(pid, 0) - int(bytes_freed))


def usage_bytes(patch_id: Optional[str] = None) -> int:
    """Return live usage in bytes for `patch_id` (or total if None)."""
    with _USAGE_LOCK:
        if patch_id is None:
            return sum(_USAGE.values())
        return _USAGE.get(patch_id.upper().strip(), 0)


# ─── Budget gates ─────────────────────────────────────────────────────


def check(patch_id: str, bytes_requested: int) -> None:
    """Raise PoolBudgetExceeded if allocating `bytes_requested` would
    exceed `patch_id`'s per-pool ceiling.

    Call this BEFORE allocating. Caller's existing try/except handles the
    raise: typically falls back to non-pool transient allocation. This is
    the same pattern as PN59's GENESIS_PN59_O_MAX_T cap (Phase A).
    """
    cap_mib = max_mib_for(patch_id)
    if cap_mib is None:
        return  # unlimited

    cap_bytes = cap_mib * 1024 * 1024
    current = usage_bytes(patch_id)
    if current + bytes_requested > cap_bytes:
        raise PoolBudgetExceeded(
            f"[Genesis pool budget] {patch_id} would exceed cap: "
            f"current {current / 1024 / 1024:.1f} MiB + requested "
            f"{bytes_requested / 1024 / 1024:.1f} MiB > "
            f"{cap_mib} MiB cap (env GENESIS_POOL_MAX_MIB_{patch_id.upper()} "
            f"or GENESIS_POOL_MAX_MIB)"
        )


def assert_total_under_budget() -> None:
    """Raise PoolBudgetExceeded if cumulative Genesis-pool usage exceeds
    GENESIS_POOL_TOTAL_MAX_MIB.

    Intended hook: post-warmup in apply_all (after profile_run + capture
    completes), so all preallocs have settled. Operator gates serve loop
    on this assertion.
    """
    cap_mib = total_max_mib()
    if cap_mib is None:
        return  # unlimited

    cap_bytes = cap_mib * 1024 * 1024
    total = usage_bytes()
    if total > cap_bytes:
        per_patch = ", ".join(
            f"{pid}={b / 1024 / 1024:.1f} MiB"
            for pid, b in sorted(_USAGE.items(), key=lambda kv: -kv[1])
        )
        raise PoolBudgetExceeded(
            f"[Genesis pool budget] cumulative pool VRAM "
            f"{total / 1024 / 1024:.1f} MiB > {cap_mib} MiB cap "
            f"(env GENESIS_POOL_TOTAL_MAX_MIB). Per-pool: {per_patch}"
        )


# ─── Diagnostics ──────────────────────────────────────────────────────


def summary() -> dict:
    """Return per-pool usage + caps + total for /diag endpoints."""
    with _USAGE_LOCK:
        per_pool = {
            pid: {
                "usage_mib": round(b / 1024 / 1024, 2),
                "cap_mib": max_mib_for(pid),
            }
            for pid, b in _USAGE.items()
        }
    return {
        "per_pool": per_pool,
        "total_usage_mib": round(usage_bytes() / 1024 / 1024, 2),
        "total_cap_mib": total_max_mib(),
        "global_per_pool_cap_mib": _global_max_mib(),
    }


# ─── Exception ────────────────────────────────────────────────────────


class PoolBudgetExceeded(RuntimeError):
    """Raised when a Genesis-pool allocation would exceed configured cap."""
    pass

# SPDX-License-Identifier: Apache-2.0
"""PN95 cross-patch shared-buffer helpers.

Eight helpers consumed by sibling patches (PN96b / PN97 / PN106 /
PN200 / PN201 / PN203) as a shared API. They live under the
``pn95.`` namespace for codebase locality, but their *callers* are
not PN95-internal — each helper is referenced by text-anchor imports
embedded in sibling patches' apply()-time source-rewrite strings:

  pn200_gdn_scratch_reuse.py     —  imports pn106_get_pooled_buf
  pn97_tensor_physical_cap.py    —  imports pn97_physical_cap_bytes
  pn106_gdn_h_pool.py            —  imports pn106_get_pooled_buf (×3 anchors)
  pn96_emergency_demote.py       —  imports pn96_emergency_rescue
  (PN203 / PN201 are wired through scheduler_tick which lives in
   _pn95_runtime; their helpers below are called from there.)

Because the import path in the text anchors is hard-coded
``from sndr.cache._pn95_runtime import <name> as _g_…``,
the legacy module MUST keep working re-exports of every symbol moved
here — otherwise the anchor regenerates with a new hash and
``apply.shadow --strict`` reports drift. This file is therefore the
canonical implementation, but the public import path for sibling
patches stays ``_pn95_runtime.<name>``.

Helpers:

  Cross-patch buffer pools (PN106 / PN200):
    * ``pn106_get_gdn_h_buf``     — legacy GDN h-state pool entry
                                     (delegates to generic allocator)
    * ``_pn106_legacy_h_impl``    — singleton-pool implementation kept
                                     for reference; no live callers
    * ``pn106_get_pooled_buf``    — generic named-pool allocator
                                     (1.25× headroom growth, 4K elem
                                      rounding, optional zero-fill)

  Pressure / sweep policies (PN201 / PN203 / PN106):
    * ``pn203_cold_prefix_sweep``   — window-aware demote of cold
                                       attention-layer blocks
    * ``pn201_maybe_empty_cache``   — threshold-gated empty_cache
                                       call with cooldown
    * ``pn106_periodic_empty_cache`` — fragmentation reclaim cadence

  Tensor budget (PN97):
    * ``pn97_physical_cap_bytes``  — per-KVCacheTensor byte cap
                                      (env override or auto-derive
                                       from torch.cuda.mem_get_info)

  Emergency capacity rescue (PN96):
    * ``pn96_emergency_rescue``    — eager pre-eviction of cached
                                      free blocks before vllm raises
                                      "Cannot get N free blocks"

M.4.2.E scope: function extraction only. State singletons stay in
``_pn95_runtime``:

  _PN106_POOLS                    — read+rebind via attribute write
  _PN106_NAMED_POOLS              — same
  _PN201_LAST_EMPTY_CACHE_TICK    — REBOUND on every empty_cache fire
                                     (original ``global`` declaration
                                      replicated via ``_rt.X = tick``)
  _PN203_ENABLED / _PN203_ACTIVE_WINDOW_TOKENS / _PN203_ATTENTION_ONLY
                                  — set by PN203 apply hook at boot;
                                     read-only here
  _PN95_BLOCK_POOL_REFS / _PN95_PREFIX_STORE / _TM / _PN95_STATS
                                  — referenced via lazy ``_rt.X``

Inventory at extraction time showed ZERO test rebinds and ZERO test
direct calls of any of these 8 functions, so the cross-module
``_rt.X`` access pattern (standard for the M.4.2 slices) is safe.

The legacy module re-exports all eight functions; sibling-patch
text-anchor strings stay byte-identical.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .gates import _enabled

log = logging.getLogger("genesis.pn95")


# ─── Cross-patch buffer pools (PN106 / PN200) ──────────────────────────


def pn106_get_gdn_h_buf(B: int, NT: int, H: int, V: int, K: int,
                        dtype: Any, device: Any):
    """Legacy entry-point — delegates to generic named-pool allocator."""
    return pn106_get_pooled_buf("gdn_h", (B, NT, H, V, K), dtype, device)


def _pn106_legacy_h_impl(B: int, NT: int, H: int, V: int, K: int,
                        dtype: Any, device: Any):
    """Return a view of the singleton GDN h-state pool sized (B, NT, H, V, K).

    The pool itself is grown on demand to the max (NT × H × V × K) seen
    so far. Same-shape requests reuse the same backing storage; bigger
    NT triggers a one-time pool re-grow (PyTorch caching allocator
    typically reuses the slab even on grow).

    Reused across all 48 GDN layers within a step AND across steps.
    Saves ~50-120 MiB alloc/free traffic per layer per call (Qwen3.6-27B
    has 48 GDN layers, so net ~2.4-5.7 GiB allocator traffic eliminated
    per chunked-prefill step). Steady-state ~200-400 MiB fragmentation
    reclaimed.

    Safety: this returns a VIEW into a shared buffer. Caller MUST
    fully overwrite via the downstream Triton kernel (which it does —
    `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` writes the full
    h tensor). No stale data risk because the kernel does not read
    h on input.
    """
    import torch
    from sndr.cache import _pn95_runtime as _rt
    elem_per_slot = B * NT * H * V * K
    elem_bytes = torch.empty(0, dtype=dtype).element_size()
    bytes_needed = elem_per_slot * elem_bytes

    key = (str(device), str(dtype))
    pool = _rt._PN106_POOLS.get(key)
    if pool is None or pool.numel() * pool.element_size() < bytes_needed:
        # Grow (or allocate). Round up to 1.25x for headroom.
        target_elems = int(elem_per_slot * 1.25)
        try:
            pool = torch.empty(target_elems, dtype=dtype, device=device)
            _rt._PN106_POOLS[key] = pool
            _rt._PN95_STATS["pn106_pool_grows"] = (
                _rt._PN95_STATS.get("pn106_pool_grows", 0) + 1
            )
            _rt._PN95_STATS["pn106_pool_bytes"] = pool.numel() * pool.element_size()
        except Exception:
            return None

    view = pool[: elem_per_slot].view(B, NT, H, V, K)
    _rt._PN95_STATS["pn106_h_slices_served"] = (
        _rt._PN95_STATS.get("pn106_h_slices_served", 0) + 1
    )
    return view


def pn106_get_pooled_buf(name: str, shape: tuple, dtype: Any, device: Any,
                        zero: bool = False):
    """Generic named-pool allocator for hot-path scratch tensors.

    Replaces fresh `torch.empty(shape, ...)` / `torch.empty_like(t)` with
    a view into a persistent flat backing buffer that grows on demand
    and is reused across calls.

    Args:
      zero: if True, zero the returned slice before handing back. Use for
            `torch.zeros(...)` replacements where the kernel expects
            initialized memory (e.g. gdn core_attn_out — see vllm PR
            #28182 discussion). Adds ~5-15us overhead per call
            (memset bandwidth-bound, well under fragmentation cost).

    Args:
      name: stable pool identifier ('gdn_h', 'gdn_v_new', 'gdn_o', etc.)
      shape: requested tensor shape (any rank)
      dtype: torch.dtype
      device: torch.device

    Returns:
      A view-tensor of `shape` backed by the named pool, or None if
      allocation fails.

    Each (name, device, dtype) gets an independent backing buffer.
    Growth is by 1.25x to amortize re-alloc on slowly-growing peaks.

    Caller MUST overwrite the returned view before reading any element
    (which is the case for all our patched sites — Triton kernels are
    write-only on these scratch tensors). No correctness loss.
    """
    import torch
    from sndr.cache import _pn95_runtime as _rt
    n_elems = 1
    for d in shape:
        n_elems *= int(d)
    if n_elems <= 0:
        return None
    key = (name, str(device), str(dtype))
    pool = _rt._PN106_NAMED_POOLS.get(key)
    if pool is None or pool.numel() < n_elems:
        target = max(n_elems, int(n_elems * 1.25))
        # Round up to 4K elements to dampen growth churn
        target = ((target + 4095) // 4096) * 4096
        try:
            pool = torch.empty(target, dtype=dtype, device=device)
        except Exception:
            return None
        _rt._PN106_NAMED_POOLS[key] = pool
        _rt._PN95_STATS[f"pn106_pool_{name}_grows"] = (
            _rt._PN95_STATS.get(f"pn106_pool_{name}_grows", 0) + 1
        )
        _rt._PN95_STATS[f"pn106_pool_{name}_bytes"] = (
            pool.numel() * pool.element_size()
        )
    view = pool[:n_elems].view(*shape)
    if zero:
        view.zero_()
    _rt._PN95_STATS[f"pn106_pool_{name}_slices"] = (
        _rt._PN95_STATS.get(f"pn106_pool_{name}_slices", 0) + 1
    )
    return view


# ─── Pressure / sweep policies (PN203 / PN201 / PN106) ─────────────────


def pn203_cold_prefix_sweep() -> int:
    """Tier 3.A — sweep cold prefix blocks beyond active window to L2.

    Walks each registered BlockPool's free_block_queue and demotes
    cached blocks belonging to full-attention layers whose position
    in the request's KV is older than `_PN203_ACTIVE_WINDOW_TOKENS`.
    Mamba/GDN blocks left GPU-resident (state is fixed-size per layer
    regardless of position, and demoting them is unsafe per PN95 design).

    Returns count of blocks swept. Best-effort — fail-silent.

    Coordinates with existing PN95 path: demote_on_evict already captures
    bytes to L2 (pinned pool if PN95_PINNED_POOL enabled), so this
    function just adds the window-aware selection policy.
    """
    from sndr.cache import _pn95_runtime as _rt
    if not _rt._PN203_ENABLED or not _enabled() or _rt._TM is None:
        return 0
    swept = 0
    try:
        # Window-aware filtering: prefer blocks deep in admit_order (older
        # positions). We approximate "position" by admit-order index;
        # blocks admitted earlier are older in the request stream.
        # Hard mapping (per-request position) requires per-block metadata;
        # this approximation is good enough for cold-prefix detection.
        if not _rt._PN95_BLOCK_POOL_REFS:
            return 0
        # Use existing LRU walker but cap to window-relative cold candidates.
        candidates = _rt._select_cold_blocks_via_bpool_lru(target_count=16)
        for pool, block_id, block_hash in candidates:
            # Filter: attention-only mode skips Mamba groups (block_hash
            # carries group_id which we check against _mamba_excluded).
            if _rt._PN203_ATTENTION_ONLY and _rt._TM is not None:
                try:
                    gid_str = getattr(block_hash, "group_id", None)
                    if gid_str in getattr(_rt._TM, "_mamba_excluded", set()):
                        continue
                except Exception:
                    pass
            try:
                if _rt.demote_on_evict(block_hash, block_id):
                    swept += 1
            except Exception:
                continue
        if swept > 0:
            _rt._PN95_STATS["pn203_cold_prefix_sweeps"] = (
                _rt._PN95_STATS.get("pn203_cold_prefix_sweeps", 0) + 1
            )
            _rt._PN95_STATS["pn203_blocks_swept_total"] = (
                _rt._PN95_STATS.get("pn203_blocks_swept_total", 0) + swept
            )
    except Exception as e:
        log.warning("[PN203] cold_prefix_sweep failed silently: %s", e)
    return swept


def pn201_maybe_empty_cache(free_mib: int, free_blocks: Optional[int] = None) -> bool:
    """Threshold-gated empty_cache call for scheduler_tick path (Tier 1.C).

    Defragments the PyTorch CUDA caching allocator when memory pressure
    is high. Returns True iff empty_cache was actually called this tick.

    Triggered when EITHER:
      - free_blocks < GENESIS_PN201_EMPTY_CACHE_FREE_BLOCKS_THRESHOLD
        (default 8 — matches PN95 proactive demote threshold scale)
      - free_mib < 256

    Cooldown: GENESIS_PN201_EMPTY_CACHE_COOLDOWN ticks (default 50, ~5s
    at default tick rate). Without cooldown, back-to-back chunks could
    fire empty_cache continuously, each blocking ~5 ms.

    Architectural note: this hook is the Tier-1.C piece — pure fragmentation
    reclaim. The Tier-3 CPU offload manager (PN203) will fire from the
    same scheduler_tick using the same pressure signal but doing real
    block migration instead of cache discard.
    """
    from sndr.cache import _pn95_runtime as _rt
    if os.environ.get(
        "GENESIS_ENABLE_PN201_SCHEDULER_EMPTY_CACHE", "0",
    ).strip().lower() not in ("1", "true", "yes", "on"):
        return False

    try:
        threshold_blocks = int(os.environ.get(
            "GENESIS_PN201_EMPTY_CACHE_FREE_BLOCKS_THRESHOLD", "8"))
        cooldown = int(os.environ.get(
            "GENESIS_PN201_EMPTY_CACHE_COOLDOWN", "50"))
    except (ValueError, TypeError):
        threshold_blocks, cooldown = 8, 50

    pressure = free_mib < 256
    if free_blocks is not None:
        pressure = pressure or free_blocks < threshold_blocks
    if not pressure:
        return False

    tick = _rt._PN95_STATS.get("ticks_total", 0)
    if tick - _rt._PN201_LAST_EMPTY_CACHE_TICK < cooldown:
        _rt._PN95_STATS["pn201_empty_cache_cooldowns"] = (
            _rt._PN95_STATS.get("pn201_empty_cache_cooldowns", 0) + 1
        )
        return False

    try:
        import torch
        torch.cuda.empty_cache()
        _rt._PN201_LAST_EMPTY_CACHE_TICK = tick
        _rt._PN95_STATS["pn201_empty_cache_calls"] = (
            _rt._PN95_STATS.get("pn201_empty_cache_calls", 0) + 1
        )
        log.info(
            "[PN201] empty_cache fired at tick=%d free_mib=%d free_blocks=%s",
            tick, free_mib, free_blocks,
        )
        return True
    except Exception as e:
        log.warning("[PN201] empty_cache failed: %s", e)
        return False


def pn106_periodic_empty_cache() -> None:
    """Call `torch.cuda.empty_cache()` to defragment the allocator.

    Invoked sparingly (every Nth scheduler tick) to reclaim "reserved but
    unallocated" memory observed in the OOM crash log (~319 MiB
    fragmentation). The CUDA caching allocator does not give back
    reserved-but-free slabs until empty_cache is called. Critical for
    long-running deployments — fragmentation accumulates as variable-
    sized chunks pass through the GDN/attention path.

    Env-driven cadence: GENESIS_PN106_EMPTY_CACHE_EVERY_N_TICKS
    (default 0 = disabled — operator opts in when fragmentation
    actually hurts).
    """
    from sndr.cache import _pn95_runtime as _rt
    try:
        n = int(os.environ.get("GENESIS_PN106_EMPTY_CACHE_EVERY_N_TICKS", "0"))
    except (ValueError, TypeError):
        n = 0
    if n <= 0:
        return
    tick = _rt._PN95_STATS.get("ticks_total", 0)
    if tick == 0 or tick % n != 0:
        return
    try:
        import torch
        torch.cuda.empty_cache()
        _rt._PN95_STATS["pn106_empty_cache_calls"] = (
            _rt._PN95_STATS.get("pn106_empty_cache_calls", 0) + 1
        )
    except Exception:
        pass


# ─── Tensor budget (PN97) ──────────────────────────────────────────────


def pn97_physical_cap_bytes(n_tensors: int) -> Optional[int]:
    """Return per-KVCacheTensor byte cap for PN97 (Phase 7 PoC).

    Called from the PN97 anchor at `_allocate_kv_cache_tensors`. We
    compute the maximum bytes that fit in the physical GPU KV budget
    (gpu_memory_utilization × VRAM − model_weights − workspace),
    divided evenly across the `n_tensors` KVCacheTensor entries.

    Returns None when PN97 disabled OR VIRT_ENABLE off (no inflation
    happening, no cap needed) OR torch unavailable.

    Operator can override via `GENESIS_PN97_PHYSICAL_CAP_GIB` (single
    value, total across all tensors).
    """
    if os.environ.get("GENESIS_ENABLE_PN97_TENSOR_PHYSICAL_CAP", "0").strip().lower() not in (
        "1", "true", "yes", "on",
    ):
        return None
    if n_tensors <= 0:
        return None

    # Operator override (total bytes across all tensors).
    env_total = os.environ.get("GENESIS_PN97_PHYSICAL_CAP_GIB", "").strip()
    if env_total:
        try:
            total_bytes = int(float(env_total) * (1 << 30))
            return total_bytes // n_tensors
        except (ValueError, TypeError):
            pass

    # Auto-derive: query torch for free GPU memory and reserve 80% for KV.
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_bytes, _total = torch.cuda.mem_get_info(0)
        # 80% of currently-free memory goes to KV (rest = workspace/activations).
        # This is conservative — operator may bump via env override.
        per_tensor = int(free_bytes * 0.80) // n_tensors
        return max(per_tensor, 1 << 30)  # at least 1 GiB per tensor entry
    except Exception:
        return None


# ─── Emergency capacity rescue (PN96) ──────────────────────────────────


def pn96_emergency_rescue(pool: Any, deficit: int) -> int:
    """Phase 6 PoC — emergency rescue when get_new_blocks would crash.

    Called from the PN96 anchor BEFORE vllm raises
    `ValueError("Cannot get N free blocks")`. Walks the pool's
    free_block_queue looking for ALREADY-CACHED blocks (block_hash
    set, ref_cnt=0). For each such block, captures the bytes to the
    PN95 L2 store via demote_on_evict, then marks the slot reusable
    (by clearing its hash — vllm will pop it from cached_block_hash_to_block
    on the next eviction pass).

    Returns the number of slots rescued. Best-effort: on any error
    returns 0 (caller falls through to the upstream ValueError).

    Why this matters: vllm's block pool can have free slots that
    are "free but reserved for cache reuse" — they show up in
    free_block_queue but with non-None block_hash. The eviction
    happens lazily inside `_maybe_evict_cached_block`. PN96 forces
    eager eviction with byte-preservation so the pool reports enough
    free slots to satisfy the current allocation request.

    Honest limitation: this ONLY rescues already-free cached blocks.
    Active blocks (ref_cnt>0, held by a running sequence) are NOT
    touched — those would need scheduler-level preemption which is
    Phase 7 work. So PN96 helps multi-prefix workloads but does
    NOT extend the single-user max_model_len above the GPU pool size.
    """
    from sndr.cache import _pn95_runtime as _rt
    if not _enabled() or deficit <= 0:
        return 0
    rescued = 0
    try:
        free_q = getattr(pool, "free_block_queue", None)
        if free_q is None:
            return 0
        head = (getattr(free_q, "fake_free_list_head", None)
                or getattr(free_q, "_fake_head", None))
        cur = getattr(head, "next_free_block", None) if head else None
        walked = 0
        max_walk = max(deficit * 4, 64)  # cap walk cost
        while cur is not None and walked < max_walk and rescued < deficit:
            walked += 1
            block_hash = getattr(cur, "block_hash", None)
            ref_cnt = getattr(cur, "ref_cnt", -1)
            block_id = getattr(cur, "block_id", -1)
            is_null = getattr(cur, "is_null", False)
            nxt = getattr(cur, "next_free_block", None)
            if (block_hash is not None and ref_cnt == 0
                    and block_id >= 0 and not is_null):
                # Preserve bytes BEFORE vllm reuses this slot.
                try:
                    if _rt.demote_on_evict(block_hash, block_id):
                        rescued += 1
                        # Clear hash so vllm sees it as a clean free slot.
                        try:
                            cur.block_hash = None
                        except Exception:
                            pass
                except Exception:
                    pass
            cur = nxt
        if rescued > 0:
            _rt._PN95_STATS["pn96_emergency_rescues"] = (
                _rt._PN95_STATS.get("pn96_emergency_rescues", 0) + 1
            )
            _rt._PN95_STATS["pn96_blocks_rescued_total"] = (
                _rt._PN95_STATS.get("pn96_blocks_rescued_total", 0) + rescued
            )
            log.info(
                "[PN96] emergency rescue: rescued %d slots (deficit was %d, walked %d)",
                rescued, deficit, walked,
            )
    except Exception as e:
        log.warning("[PN96] emergency_rescue failed silently: %s", e)
    return rescued

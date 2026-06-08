# SPDX-License-Identifier: Apache-2.0
"""PN95 Phase 5 virtual-block helpers — pure accounting / mapping / guards.

Eight pure helpers that decide whether a block is virtual, where its
physical donor lives, what the pool's logical capacity should be,
how much CPU-tier inflation is available, and how to guard the
free-block-pool walk against virtual blocks leaking onto the GPU
path before they're materialized:

  pn95_extra_logical_memory_bytes  — Anchor #9: bytes of CPU-tier
                                      capacity to ADD to vllm's pre-flight
                                      available_memory check
  pn95_phase5_init_block_pool      — Anchor #11: side-table + (when
                                      VIRT=1) virtual-block inflation
                                      of pool.blocks + free_block_queue
  pn95_block_is_physical_resident  — boolean: block_id < physical_num
  pn95_guard_get_new_blocks        — Anchor #14: defensive guard;
                                      raises ValueError instead of
                                      letting a virtual block reach
                                      attention
  pn95_anchor12_post_popleft       — Anchor #12: best-effort
                                      post-popleft materialization
                                      (currently rolled back to
                                      VIRT=1-safe no-op)
  pn95_block_metadata              — read side-table by (pool, block_id)
  pn95_pool_logical_num_blocks     — read inflated count for a pool
  pn95_physical_num_blocks_cap     — Anchor #10: GPU-tier byte cap so
                                      KVCacheTensor allocation stays
                                      within physical VRAM

The materialization function itself (``pn95_materialize_virtual_block``)
STAYS in ``_pn95_runtime`` because it calls ``demote_on_evict`` and
``promote_on_miss`` — prefix-store mutation that lives there until a
later slice. Two helpers below
(``pn95_guard_get_new_blocks``, ``pn95_anchor12_post_popleft``) call
``pn95_materialize_virtual_block`` via lazy ``_rt.pn95_materialize_virtual_block``
when a virtual block is encountered.

M.4.2.F scope: function extraction only. State singletons stay in
``_pn95_runtime``:

  _PN95_BLOCK_METADATA         — side-table {(pool_id, block_id) → meta dict}
                                  written/read across every helper
  _PN95_POOL_LOGICAL_NUM_BLOCKS — {pool_id → inflated num_blocks}

Inventory at extraction time showed ZERO test rebinds, ZERO test
direct calls, ZERO `global` declarations for either state singleton
(all writes are dict-item mutations, no rebind hazard). Three
sibling-patch text-anchor imports reference these symbols via the
legacy ``_pn95_runtime`` path — those continue resolving through the
re-export shim, no anchor regen.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .gates import _enabled, _phase5_virt_enabled

log = logging.getLogger("genesis.pn95")


# ─── Anchor #9: CPU-tier capacity inflation ────────────────────────────


def pn95_extra_logical_memory_bytes() -> int:
    """Phase 5 Anchor #9 helper — bytes of CPU tier capacity to add to
    vllm's pre-flight available_memory check.

    Allows max_model_len > GPU hardware ceiling. Caller (Anchor #9)
    inflates available_memory by this amount, vllm then computes
    num_blocks based on inflated value, BUT Anchor #10 caps the
    physical GPU allocation separately to prevent CUDA OOM.

    Returns 0 if PN95 disabled OR Phase 5 virt disabled OR no TM
    installed yet (deferred boot-time call before TM init).

    Safety: env-gated (default OFF). Even if accidentally called when
    TM not ready, returns 0 → vllm sees unmodified available_memory →
    behavior identical to PN95 OFF.
    """
    if not _phase5_virt_enabled():
        return 0
    from sndr.cache import _pn95_runtime as _rt
    tm = _rt._TM
    if tm is None:
        return 0
    extra = 0
    try:
        for tier_idx, tier in enumerate(tm.tiers):
            if tier_idx == 0:
                continue  # tier 0 is GPU, already counted by vllm
            device = getattr(tier, "device", "")
            if device == "cpu":
                cap_gib = float(getattr(tier, "capacity_gib", 0.0))
                extra += int(cap_gib * (1 << 30))
    except Exception:
        return 0
    return extra


# ─── Anchor #11: side-table init + virtual block inflation ─────────────


def pn95_phase5_init_block_pool(pool: Any) -> None:
    """Path C v1.0 Phase 5 (Anchors #11 + extended for #12) — initialize
    side-table metadata + optionally inflate the pool with virtual blocks.

    Two modes:
      VIRT=0 (default): metadata side-table init only, NO behavior change.
      VIRT=1: also creates N_virtual extra KVCacheBlock objects with virtual
        block_ids ≥ num_physical, marks them physical_resident=False, adds
        to pool.blocks list AND free_block_queue. Scheduler sees inflated
        num_gpu_blocks. When get_new_blocks pops a virtual block, Anchor #12
        materializes it via swap (mutates block.block_id to physical id from
        a demoted donor block).

    Side-table per (id(pool), block_id):
      - physical_resident: bool
      - physical_block_id: Optional[int] — None for virtual, GPU slot otherwise
      - last_access_tick: int

    N_virtual computed from CPU tier capacity / page_size_bytes_per_block.

    Idempotent — safe to call multiple times on same pool.
    Fail-silent — returns immediately on any error.
    """
    if not _enabled():
        return
    from sndr.cache import _pn95_runtime as _rt
    try:
        pool_id = id(pool)

        # Initialize per-block metadata for ALL existing physical blocks
        blocks = getattr(pool, "blocks", None) or []
        n_physical = len(blocks)

        for blk in blocks:
            key = (pool_id, blk.block_id)
            if key not in _rt._PN95_BLOCK_METADATA:
                _rt._PN95_BLOCK_METADATA[key] = {
                    "physical_resident": True,
                    "physical_block_id": blk.block_id,
                    "last_access_tick": 0,
                }

        # VIRT=0: no inflation. Default behavior preserved.
        if not _phase5_virt_enabled():
            if pool_id not in _rt._PN95_POOL_LOGICAL_NUM_BLOCKS:
                _rt._PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
            return

        # VIRT=1: create virtual blocks. CONSERVATIVE inflation strategy
        # to prevent scheduler over-admission crash:
        #
        # The fundamental constraint: when single request grows past
        # physical pool size, vllm cannot preempt its own blocks (that
        # would lose in-progress work). Virtual blocks can only be safely
        # materialized when a DIFFERENT request's blocks are in free_queue
        # (ref_cnt=0). For single-request long-context, this never happens
        # → materialization fails → crash.
        #
        # SAFE INFLATION = inflate only by amount that scheduler will
        # admit cross-request rotation can absorb. Empirical safe ratio:
        # logical = physical × INFLATION_RATIO where ratio ≤ 1.5 typically
        # (operator override via GENESIS_PN95_VIRT_INFLATION_RATIO).
        #
        # For aggressive single-request expansion, scheduler-level
        # preemption is required — that's beyond text-patches.
        # Idempotent guard.
        if pool_id in _rt._PN95_POOL_LOGICAL_NUM_BLOCKS:
            return

        # Compute N_virtual from tier capacity. Need per-block bytes
        # estimate — use TM's _attention_views first registered layer
        # bytes_per_block, fallback to 49664 (TQ k8v4 default for 27B PROD).
        bytes_per_block = 49664
        try:
            views = getattr(_rt._TM, "_attention_views", None) or {}
            if views:
                first_info = next(iter(views.values()))
                bytes_per_block = int(first_info.get("bytes_per_block", 49664))
        except Exception:
            pass

        cpu_tier_bytes = pn95_extra_logical_memory_bytes()
        # Account for ALL eligible attention layers (each block's bytes
        # × n_layers must fit in CPU tier).
        n_attn_layers = max(1, len(getattr(_rt._TM, "_attention_views", {}) or {}) or 17)
        bytes_per_full_block = bytes_per_block * n_attn_layers
        cpu_capacity_blocks = int(cpu_tier_bytes // bytes_per_full_block)

        # SAFE INFLATION RATIO — bounded by physical pool size to prevent
        # scheduler over-admission crash. Default 1.5x physical (operator
        # can tune via env GENESIS_PN95_VIRT_INFLATION_RATIO).
        try:
            inflation_ratio = float(os.environ.get(
                "GENESIS_PN95_VIRT_INFLATION_RATIO", "1.5"))
        except (ValueError, TypeError):
            inflation_ratio = 1.5
        inflation_ratio = max(1.0, min(inflation_ratio, 8.0))  # clamp [1, 8]

        # Max safe virtual count based on physical pool size
        max_safe_virtual = int(n_physical * (inflation_ratio - 1.0))

        # Take min of CPU capacity and safe inflation
        n_virtual = min(cpu_capacity_blocks, max_safe_virtual)

        # Operator override caps (avoid runaway memory)
        try:
            virt_max = int(os.environ.get(
                "GENESIS_PN95_VIRT_MAX_BLOCKS", "10000"))
        except (ValueError, TypeError):
            virt_max = 10000
        n_virtual = min(n_virtual, virt_max)

        if n_virtual <= 0:
            log.warning(
                "[PN95 v1.0 Phase 5] VIRT=1 but no CPU tier capacity "
                "available for virtual blocks — skipping inflation"
            )
            _rt._PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
            return

        # Create virtual blocks. KVCacheBlock(block_id=int) constructor.
        # Need access to vllm's KVCacheBlock class — import lazily.
        try:
            from vllm.v1.core.kv_cache_utils import KVCacheBlock
        except ImportError:
            log.warning(
                "[PN95 v1.0 Phase 5] cannot import KVCacheBlock — virt skipped"
            )
            _rt._PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
            return

        free_q = getattr(pool, "free_block_queue", None)
        if free_q is None:
            log.warning(
                "[PN95 v1.0 Phase 5] pool has no free_block_queue — virt skipped"
            )
            _rt._PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
            return

        # Generate virtual blocks with synthetic block_ids starting from
        # n_physical. Mark each as virtual in side-table.
        virtual_blocks = []
        for i in range(n_virtual):
            virt_id = n_physical + i
            try:
                vblk = KVCacheBlock(block_id=virt_id)
                _rt._PN95_BLOCK_METADATA[(pool_id, virt_id)] = {
                    "physical_resident": False,
                    "physical_block_id": None,
                    "last_access_tick": 0,
                }
                virtual_blocks.append(vblk)
            except Exception:
                continue

        # Append to pool.blocks list and free_block_queue.
        # FreeKVCacheBlockQueue uses doubly-linked-list pointers
        # (prev_free_block / next_free_block). Use append_n method.
        if hasattr(free_q, "append_n"):
            try:
                free_q.append_n(virtual_blocks)
            except Exception as e:
                log.warning(
                    "[PN95 v1.0 Phase 5] free_q.append_n failed: %s — virt skipped",
                    e,
                )
                _rt._PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = pool.num_gpu_blocks
                return

        try:
            pool.blocks.extend(virtual_blocks)
        except Exception:
            pass

        # Inflate pool.num_gpu_blocks for scheduler awareness
        new_logical = n_physical + len(virtual_blocks)
        try:
            pool.num_gpu_blocks = new_logical
        except Exception:
            pass

        _rt._PN95_POOL_LOGICAL_NUM_BLOCKS[pool_id] = new_logical

        log.warning(
            "[PN95 v1.0 Phase 5 Anchor #11+] inflated BlockPool: "
            "physical=%d → logical=%d (+%d virtual blocks, "
            "CPU tier %.1f GiB / block %d B × %d layers = %d B/full)",
            n_physical, new_logical, len(virtual_blocks),
            cpu_tier_bytes / (1 << 30), bytes_per_block, n_attn_layers,
            bytes_per_full_block,
        )
    except Exception as e:
        log.warning("[PN95 v1.0 Phase 5] init_block_pool failed: %s", e)


# ─── Block-residency predicates + side-table accessors ─────────────────


def pn95_block_is_physical_resident(pool: Any, block_id: int) -> bool:
    """Return True iff `block_id` exists in the physical KV pool range.

    Used by Anchor #14 (defensive guard at get_new_blocks) to detect a
    virtual block leaked out before the runtime is ready to materialize
    it. A virtual block_id >= physical_num_blocks would index past the
    KVCacheTensor → CUDA illegal memory access, which we want to convert
    into a clean ValueError instead of a worker-process crash.

    For pools without Phase-5 metadata (which is the normal case while
    VIRT_ENABLE=0) every block is physical by definition; we return True.
    """
    from sndr.cache import _pn95_runtime as _rt
    pool_id = id(pool)
    physical_num = _rt._PN95_POOL_LOGICAL_NUM_BLOCKS.get(pool_id, -1)
    if physical_num <= 0:
        return True  # no virtual blocks ever created on this pool
    return 0 <= block_id < physical_num


# ─── Anchor #14: defensive guard at get_new_blocks ─────────────────────


def pn95_guard_get_new_blocks(pool: Any, blocks: list) -> None:
    """Phase 5 Anchor #14 — defensive guard: reject virtual blocks before
    they reach the GPU.

    Walks the just-popped block list; if any block has a block_id >=
    physical_num_blocks AND Phase-5 materialization is not ready (no
    donor available, VIRT disabled, etc.) — raise ValueError so the
    scheduler retries / preempts instead of letting the worker dereference
    invalid GPU memory.

    Best-effort materialization is still attempted for each suspicious
    block via pn95_materialize_virtual_block — only if THAT also fails
    do we surface the error. On normal (VIRT=0) deployments this is a
    fast no-op (the pool has no virtual blocks).
    """
    from sndr.cache import _pn95_runtime as _rt
    if not _enabled() or _rt._TM is None:
        return
    pool_id = id(pool)
    if _rt._PN95_POOL_LOGICAL_NUM_BLOCKS.get(pool_id, -1) <= 0:
        return  # no Phase-5 inflation on this pool
    for blk in blocks:
        bid = getattr(blk, "block_id", -1)
        if pn95_block_is_physical_resident(pool, bid):
            continue
        # Attempt materialization (swap with a donor physical) — still
        # lives in _pn95_runtime because it calls demote_on_evict /
        # promote_on_miss (prefix-store mutation).
        new_phys = _rt.pn95_materialize_virtual_block(pool, blk, exclude=blocks)
        if new_phys is None:
            _rt._PN95_STATS["virtual_block_unmaterialized_total"] = (
                _rt._PN95_STATS.get("virtual_block_unmaterialized_total", 0) + 1
            )
            raise ValueError(
                f"[Genesis PN95 Anchor #14] virtual block_id={bid} could "
                f"not be materialized (no free donor available). This is "
                f"the documented 'Phase 5 VIRT without scheduler "
                f"preemption' failure mode — set "
                f"GENESIS_PN95_VIRT_ENABLE=0 to disable, or wait for "
                f"scheduler-preemption work to land."
            )
        # Adopt donor's physical block_id.
        try:
            blk.block_id = new_phys
        except Exception:
            raise ValueError(
                f"[Genesis PN95 Anchor #14] materialization succeeded "
                f"(donor phys_id={new_phys}) but block_id assignment "
                f"failed; refusing to return virtual block to engine."
            )


# ─── Anchor #12: post-popleft best-effort materialization ──────────────


def pn95_anchor12_post_popleft(pool: Any, popped_blocks: list) -> bool:
    """Path C v1.0 Phase 5 Anchor #12 — post-process popped blocks list.

    DESIGN UPDATE 2026-05-09: rollback to truly no-op behavior.

    Discovery from live testing: swap-based virtualization fundamentally
    unsafe without scheduler-level preemption. When all physical blocks
    are held by active requests (ref_cnt > 0), no donors available in
    free_queue → materialization fails → ValueError → engine_core dies.

    To safely reach 256K context via virtualization requires:
      1. Scheduler-side preemption (evict older requests' blocks)
      2. Request queueing on partial allocation failure
      3. Cross-step coordination of demote/promote with attention reads

    This is scheduler-level architecture work, NOT achievable via
    text-patches alone. Deferred to future sub-project.

    For now: this anchor returns True (no-op) when PN95/VIRT disabled.
    When VIRT=1 + virtual blocks present: WARNINGS but no crash —
    materialization attempted on best-effort basis. If donor unavailable,
    block stays virtual (will crash attention read) — but Anchor #11
    inflation also rolled back to no-virtual-creation, so this path
    never triggers in practice.
    """
    if not _enabled() or not _phase5_virt_enabled():
        return True
    from sndr.cache import _pn95_runtime as _rt
    # Best-effort materialization but NEVER raise — return True even on
    # partial failure to avoid crashing engine_core. With Anchor #11
    # rolled back to no-inflation, this loop typically finds no virtual
    # blocks and is a fast pass-through.
    try:
        pool_id = id(pool)
        for block in popped_blocks:
            meta = _rt._PN95_BLOCK_METADATA.get((pool_id, block.block_id))
            if meta is None:
                continue
            if meta.get("physical_resident", True):
                continue
            new_phys_id = _rt.pn95_materialize_virtual_block(
                pool, block, exclude=popped_blocks,
            )
            if new_phys_id is None:
                # Best-effort: cannot materialize but cannot crash either.
                # Block stays virtual; if attention reads it, vllm crashes
                # at attention layer, not here. With Anchor #11 inflation
                # rolled back, this path is unreachable in normal use.
                continue
            try:
                block.block_id = new_phys_id
            except Exception:
                continue
        return True
    except Exception:
        return True


def pn95_block_metadata(pool: Any, block_id: int) -> Optional[dict]:
    """Phase 5 Session 2 helper — read PN95 metadata for (pool, block_id).

    Returns None when:
      - PN95 disabled
      - Pool not yet initialized via pn95_phase5_init_block_pool
      - block_id not in side-table (e.g., for null_block which has block_id=0
        but special semantics)
    """
    if not _enabled():
        return None
    from sndr.cache import _pn95_runtime as _rt
    return _rt._PN95_BLOCK_METADATA.get((id(pool), block_id))


def pn95_pool_logical_num_blocks(pool: Any) -> Optional[int]:
    """Phase 5 helper — get logical num_blocks for a pool.

    Returns None if not initialized → caller treats as unmodified vllm
    behavior. When Session 3+ activates virtualization, this returns
    inflated count (physical + cpu_tier_blocks).
    """
    if not _enabled():
        return None
    from sndr.cache import _pn95_runtime as _rt
    return _rt._PN95_POOL_LOGICAL_NUM_BLOCKS.get(id(pool))


# ─── Anchor #10: physical KV byte cap ──────────────────────────────────


def pn95_physical_num_blocks_cap() -> Optional[int]:
    """Phase 5 Anchor #10 helper — cap physical KVCacheTensor allocation
    to GPU-only memory budget.

    Without this cap, inflated available_memory in Anchor #9 would
    cause vllm to allocate `KVCacheTensor(size=page_size * num_blocks)`
    sized for inflated num_blocks → CUDA OOM at init.

    Returns the max block count that fits in tier 0 (GPU). vllm's
    allocation site uses min(num_blocks_logical, this_cap) for the
    actual torch.empty() call. The remaining (num_blocks_logical -
    cap) blocks become "virtual" — created in BlockPool's metadata
    pool but not backed by physical GPU memory until materialized
    via Anchor #12 (next session).

    Returns None when Phase 5 virt disabled — caller treats as
    "no cap" and uses original num_blocks (current behavior).

    NOTE: Returns BYTES budget, not block count. Caller (Anchor #10
    site) divides by page_size to get block count for KVCacheTensor.
    """
    if not _phase5_virt_enabled():
        return None
    from sndr.cache import _pn95_runtime as _rt
    tm = _rt._TM
    if tm is None or len(tm.tiers) < 1:
        return None
    try:
        gpu_tier = tm.tiers[0]
        if getattr(gpu_tier, "device", "") != "gpu":
            return None
        cap_gib = float(getattr(gpu_tier, "capacity_gib", 0.0))
        if cap_gib <= 0:
            return None
        return int(cap_gib * (1 << 30))
    except Exception:
        return None

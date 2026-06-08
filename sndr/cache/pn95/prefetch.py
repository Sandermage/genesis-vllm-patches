# SPDX-License-Identifier: Apache-2.0
"""PN95 prefetch API — batched L1 pinned-pool warm-up.

Two public functions:

  * ``pn95_prefetch_blocks(block_hashes)`` — warm up the L1 pinned
    host pool for the given block_hashes from L2 (OrderedDict) or
    disk tier. Called from ``notify_touch`` (in ``_pn95_runtime``)
    via the neighbor-prefetch path when adjacent block hashes are
    likely about to be touched next.
  * ``pn95_get_prefetch_stats()`` — snapshot of the counter dict
    surfaced via ``sndr patches pn95-status``.

M.4.2.B scope: function extraction only. The mutable state singleton
``_PN95_PREFETCH_STATS`` and every other state singleton this code
touches (``_PN95_PREFIX_STORE``, ``_PN95_PREFIX_STORE_LOCK``,
``_PN95_PREFIX_STORE_BYTES_USED``, the L1 pool accessor
``_pn95_l1_pool``, the packer ``_pn95_pack_layer_data``, the
``_prefix_store_max_bytes`` helper) STAY defined in ``_pn95_runtime``
because either:

  * other code paths (the prefix-store family — M.4.2.E and later)
    still own those globals, or
  * metrics.py already reads ``_PN95_PREFETCH_STATS`` via lazy
    ``_rt._PN95_PREFETCH_STATS`` and consistency with the M.4.1/A
    pattern argues for keeping all state in one place until the
    full split lands.

The cross-module write to ``_PN95_PREFIX_STORE_BYTES_USED`` (formerly
``global _PN95_PREFIX_STORE_BYTES_USED`` + ``+=``) is replicated via
explicit attribute mutation ``_rt._PN95_PREFIX_STORE_BYTES_USED += …``
which works because Python evaluates that as
``setattr(_rt, "...", getattr(_rt, "...") + …)`` — operating on the
same module-attribute slot the original ``global`` declaration did.

The legacy module re-exports both functions so:

  * ``notify_touch`` (stays in ``_pn95_runtime``) calls
    ``pn95_prefetch_blocks(...)`` through the shim
  * text-patch anchors that reference these names directly stay intact
  * tests that import either function via ``rt.pn95_prefetch_blocks``
    or directly from ``_pn95_runtime`` keep working without edit
"""
from __future__ import annotations

from .gates import _enabled


def pn95_prefetch_blocks(block_hashes: list) -> dict:
    """Warm up the L1 pinned pool for the given block_hashes BEFORE the
    engine needs them. Returns a stats delta dict.

    Strategy per block_hash:
      1. If L1 already has it → no-op (count as already-warm).
      2. If L2 OrderedDict has it → pack + put into L1 pinned pool.
      3. If disk tier has it → pull from disk into L2 (re-fill RAM cache),
         then pack + put into L1.
      4. If none → record as miss; nothing to do.

    Cheap on the happy path: dict lookup + pickle.dumps + memcpy into
    pinned slot. Steps 2/3 do NOT touch the GPU — pure host-side moves.
    The slow GPU DMA happens later on `promote_on_miss`, but from the
    pinned slot (3-5× faster than pageable bytes).

    Safe to call from any thread; the pool is mutex-protected internally.
    Returns {} when PN95 disabled or pool unavailable.
    """
    if not _enabled():
        return {}
    # Late import — every state singleton + helper this function reads
    # or mutates stays in `_pn95_runtime` for M.4.2.B (see module
    # docstring for the rationale).
    from sndr.cache import _pn95_runtime as _rt
    pool = _rt._pn95_l1_pool()
    if pool is None:
        return {}

    delta = {
        "blocks_warmed_from_l2": 0,
        "blocks_already_warm": 0,
        "blocks_missing": 0,
        "blocks_warmed_from_disk": 0,
        "blocks_pool_full": 0,
    }

    _rt._PN95_PREFETCH_STATS["prefetch_calls"] += 1
    _rt._PN95_PREFETCH_STATS["prefetch_block_hashes"] += len(block_hashes)

    # Lazy disk import — only when we actually need disk fallback below.
    _disk = None

    for h in block_hashes:
        # 1) Already in L1?
        if pool.has(h):
            delta["blocks_already_warm"] += 1
            _rt._PN95_PREFETCH_STATS["prefetch_l2_already_in_l1"] += 1
            continue

        # 2) In L2?
        layer_data = _rt._PN95_PREFIX_STORE.get(h)

        # 3) Else try disk.
        if layer_data is None:
            if _disk is None:
                try:
                    from sndr.cache import _pn95_disk_tier as _disk_mod
                    _disk = _disk_mod
                except ImportError:
                    _disk = False
            if _disk and _disk._enabled():
                try:
                    layer_data = _disk.disk_tier_get(h)
                except Exception:
                    layer_data = None
            if layer_data is not None:
                # Re-insert into L2 so future direct hits stay fast too.
                try:
                    total_bytes = sum(len(b) for _n, b in layer_data)
                    if total_bytes <= _rt._prefix_store_max_bytes():
                        with _rt._PN95_PREFIX_STORE_LOCK:
                            _rt._PN95_PREFIX_STORE[h] = layer_data
                            _rt._PN95_PREFIX_STORE_BYTES_USED += total_bytes
                except Exception:
                    pass
                delta["blocks_warmed_from_disk"] += 1
                _rt._PN95_PREFETCH_STATS["prefetch_disk_hits_promoted"] += 1

        if layer_data is None:
            delta["blocks_missing"] += 1
            _rt._PN95_PREFETCH_STATS["prefetch_missing"] += 1
            continue

        # 4) Pack + put into L1 pinned slot.
        try:
            blob = _rt._pn95_pack_layer_data(layer_data)
            if pool.put(h, blob):
                if delta["blocks_warmed_from_disk"]:
                    pass  # already counted as disk
                else:
                    delta["blocks_warmed_from_l2"] += 1
                    _rt._PN95_PREFETCH_STATS["prefetch_l2_hits_promoted"] += 1
            else:
                delta["blocks_pool_full"] += 1
                _rt._PN95_PREFETCH_STATS["prefetch_pool_full_skips"] += 1
        except Exception:
            pass

    return delta


def pn95_get_prefetch_stats() -> dict:
    """Snapshot of prefetch API counters — surfaced via sndr patches pn95-status."""
    from sndr.cache import _pn95_runtime as _rt
    return dict(_rt._PN95_PREFETCH_STATS)

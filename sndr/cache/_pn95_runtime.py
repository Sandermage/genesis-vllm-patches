# SPDX-License-Identifier: Apache-2.0
"""PN95 v7.73.x runtime hooks — notify_admit / notify_touch.

Module-level singleton TierManager. Both hooks are designed to be
**fail-silent**: if GENESIS_ENABLE_PN95_TIER_AWARE_CACHE is unset OR
the singleton hasn't been initialized OR any error occurs inside the
notification, the call must return cleanly so the surrounding vLLM
code path is never destabilized.

Public entry points:
  - `init_from_config(cfg)` — install the singleton from a ModelConfig.
    Idempotent. Called once at engine startup by the dispatcher hook.
  - `notify_admit(request, prev_n_cached, new_n_cached, group_id)` —
    called from the cache_blocks() text-patch site after vLLM's
    cache_full_blocks() returns.
  - `notify_touch(block_hash, group_ids, cached_blocks)` — called
    from the get_cached_block() text-patch site before return.
  - `tier_manager()` — accessor for live observability / tests.
  - `reset_for_tests()` — drop the singleton.

Vision-token tagging + Mamba exclusion plumbing wires through this
module so the text-patches stay tiny.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

log = logging.getLogger("genesis.pn95")

_LOCK = threading.Lock()
_TM: Optional[Any] = None  # sndr.cache.tier_manager.TierManager
_LAST_GROUP_IDS_BY_HASH: dict = {}  # cleared on reset_for_tests


# M.4.1 — env-gate predicates extracted to `.pn95.gates`. The
# re-exports below keep ``_pn95_runtime._enabled`` / ``_phase5_virt_enabled``
# importable at the original dotted path for tests and text-patch anchors.
from .pn95.gates import _enabled, _phase5_virt_enabled  # noqa: E402


# M.4.2.F — `pn95_extra_logical_memory_bytes` extracted to
# `.pn95.virtual_blocks`. Sibling-patch text-anchor in
# pn95_tier_aware_cache.py:393 imports this name through _pn95_runtime
# so the shim below MUST keep working byte-identically.
from .pn95.virtual_blocks import pn95_extra_logical_memory_bytes  # noqa: E402


_PN95_CUDA_STREAM: Optional[Any] = None
# Phase 5 Session 2 — side-table for block metadata.
# KVCacheBlock is @dataclass(slots=True) → cannot add fields directly.
# Side-table keyed by (id(pool), block_id) → {"physical_resident": bool,
# "physical_block_id": Optional[int], "last_access_tick": int}.
_PN95_BLOCK_METADATA: dict = {}
_PN95_POOL_LOGICAL_NUM_BLOCKS: dict = {}  # id(pool) → logical num_blocks


# M.4.1 — `_pn95_async_enabled` + `_pn95_use_stream_pool` extracted to
# `.pn95.gates` (re-exports preserved here so test sites that do
# `rt._pn95_async_enabled` keep resolving — e.g.
# tests/unit/cache/test_pn95_b1_async_stream.py).
from .pn95.gates import _pn95_async_enabled, _pn95_use_stream_pool  # noqa: E402


# M.4.2.G — seven transfer helpers (_pn95_stream + 6 byte-copy primitives)
# extracted to `.pn95.transfer`. State `_PN95_CUDA_STREAM` stays here (5
# test sites rebind it via `monkeypatch.setattr(rt, ...)`); the moved
# `_pn95_stream` lazy-imports `_rt` and writes `_rt._PN95_CUDA_STREAM = …`
# via attribute mutation at the same module-attribute slot the original
# `global` declaration mutated.
from .pn95.transfer import (  # noqa: E402
    _pn95_stream,
    _pn95_gpu_to_cpu_bytes,
    _pn95_gpu_to_cpu_bytes_batch_v2,
    _pn95_cpu_to_gpu_copy_batch_v2,
    _pn95_gpu_to_cpu_bytes_batch,
    _pn95_cpu_to_gpu_copy_batch,
    _pn95_cpu_to_gpu_copy,
)


# M.4.2.F — `pn95_phase5_init_block_pool` extracted to
# `.pn95.virtual_blocks`. Sibling-patch text-anchor in
# pn95_tier_aware_cache.py:348 imports this name through _pn95_runtime
# so the shim below MUST keep working byte-identically.
from .pn95.virtual_blocks import pn95_phase5_init_block_pool  # noqa: E402


# M.4.2.H — `pn95_materialize_virtual_block` extracted to
# `.pn95.prefix_store`. Was parked here in M.4.2.F because it calls
# `demote_on_evict` (now also in prefix_store.py). Reaches state via
# lazy `_rt.X`; the virtual_blocks helpers that call this function
# (pn95_guard_get_new_blocks, pn95_anchor12_post_popleft) keep using
# `_rt.pn95_materialize_virtual_block` → resolves through the shim.
from .pn95.prefix_store import pn95_materialize_virtual_block  # noqa: E402


# M.4.2.E — `pn106_get_gdn_h_buf`, `_pn106_legacy_h_impl`,
# `pn106_get_pooled_buf` extracted to `.pn95.shared_buffers`. State
# (`_PN106_POOLS`, `_PN106_NAMED_POOLS`) stays defined in this module
# (see block below); the moved functions reach it via lazy `_rt.X`.
# The legacy import path stays byte-identical for sibling-patch
# text-anchor strings (PN106 / PN200 anchor strings hard-code
# ``from sndr.cache._pn95_runtime import pn106_get_pooled_buf``).
from .pn95.shared_buffers import (  # noqa: E402
    pn106_get_gdn_h_buf,
    _pn106_legacy_h_impl,
    pn106_get_pooled_buf,
)


_PN106_POOLS: dict = {}
_PN106_NAMED_POOLS: dict = {}  # name -> torch.Tensor (flat backing buffer)


_PN201_LAST_EMPTY_CACHE_TICK: int = 0

# PN203 cold-prefix offload settings (set by PN203 apply hook at boot).
# Read by scheduler_tick to decide whether to do window-aware demote
# of full-attention layer blocks older than _PN203_ACTIVE_WINDOW_TOKENS.
_PN203_ENABLED: bool = False
_PN203_ACTIVE_WINDOW_TOKENS: int = 32768
_PN203_ATTENTION_ONLY: bool = True


# M.4.2.E — `pn203_cold_prefix_sweep`, `pn201_maybe_empty_cache`,
# `pn106_periodic_empty_cache` extracted to `.pn95.shared_buffers`.
# State (`_PN201_LAST_EMPTY_CACHE_TICK`, `_PN203_*`) stays defined in
# this module; the moved functions read/rebind via lazy `_rt.X` (the
# `global _PN201_LAST_EMPTY_CACHE_TICK` rebind is replicated via
# `_rt._PN201_LAST_EMPTY_CACHE_TICK = tick` attribute write).
from .pn95.shared_buffers import (  # noqa: E402
    pn203_cold_prefix_sweep,
    pn201_maybe_empty_cache,
    pn106_periodic_empty_cache,
)


# M.4.2.E — `pn97_physical_cap_bytes` + `pn96_emergency_rescue`
# extracted to `.pn95.shared_buffers`. The legacy import path stays
# byte-identical for sibling-patch text-anchor strings (PN96 / PN97
# hard-code the ``from sndr.cache._pn95_runtime import …``
# string).
from .pn95.shared_buffers import (  # noqa: E402
    pn97_physical_cap_bytes,
    pn96_emergency_rescue,
)


# M.4.2.F — six virtual-block helpers extracted to
# `.pn95.virtual_blocks`. Sibling-patch text-anchor in
# pn95_tier_aware_cache.py:250 imports `pn95_anchor12_post_popleft`
# through _pn95_runtime; the shim below MUST keep working byte-identically.
# `pn95_guard_get_new_blocks` and `pn95_anchor12_post_popleft` call
# `pn95_materialize_virtual_block` (stays here, calls demote_on_evict /
# promote_on_miss) via lazy `_rt.pn95_materialize_virtual_block`.
from .pn95.virtual_blocks import (  # noqa: E402
    pn95_block_is_physical_resident,
    pn95_guard_get_new_blocks,
    pn95_anchor12_post_popleft,
    pn95_block_metadata,
    pn95_pool_logical_num_blocks,
    pn95_physical_num_blocks_cap,
)


# M.4.2.A — `_detect_upstream_offload_connector` + `init_from_config`
# extracted to `.pn95.runtime_state`. State ownership (`_TM`, `_LOCK`)
# stays in this module; the moved functions write through `_rt._TM = ...`
# via lazy late-import so the 36 reader sites here see the canonical
# binding.
from .pn95.runtime_state import (  # noqa: E402
    _detect_upstream_offload_connector,
    init_from_config,
)


# M.4.2.G — `_mm_block_overlap_set` extracted to `.pn95.transfer`. Pure
# helper (no torch / no CUDA) but it's used by notify_admit alongside the
# other transfer primitives, so it lives in transfer.py for cohesion.
# Tests in test_pn95_day5_mm_overlap.py import it directly from
# `_pn95_runtime`; the shim keeps that path working.
from .pn95.transfer import _mm_block_overlap_set  # noqa: E402


# M.4.2.I — notify_admit + notify_touch extracted to `.pn95.hooks`.
# Sibling-patch text-anchor imports in pn95_tier_aware_cache.py:61 / :86
# reference these names through _pn95_runtime — shim below MUST keep
# working byte-identically (anchor regen would break apply.shadow).
from .pn95.hooks import notify_admit, notify_touch  # noqa: E402


# M.4.1 — prefetch env gates extracted to `.pn95.gates`.
from .pn95.gates import (  # noqa: E402
    _pn95_prefetch_neighbors_enabled,
    _pn95_prefetch_window,
)


# M.4.2.I — register_kv_caches extracted to `.pn95.hooks`.
# Sibling-patch text-anchor in pn95_tier_aware_cache.py:144 imports
# this name through _pn95_runtime — shim preserves the path byte-identical.
from .pn95.hooks import register_kv_caches  # noqa: E402


# M.4.2.I — init_mamba_exclusions_from_kv_groups extracted to `.pn95.hooks`.
# Sibling-patch text-anchor in pn95_tier_aware_cache.py:115 imports this
# name through _pn95_runtime — shim below preserves the path byte-identical.
from .pn95.hooks import init_mamba_exclusions_from_kv_groups  # noqa: E402


_TICK_COUNTER = 0
_TICK_LAST_FREE_MIB = 0
# Path C v1.0 Phase 3 — observability counters.
#
# M.4.1 note: ownership stays in this module (not `.pn95.metrics`) because
# ~10 test sites rebind `rt._PN95_STATS` via ``monkeypatch.setattr``,
# which would break a cross-module name alias. The functions that READ
# this dict (``get_pn95_stats`` / ``_pn95_dump_stats_if_due``) live in
# `.pn95.metrics` and late-import this name so the monkeypatch path
# continues to work. State ownership reorganization is deferred to M.4.2.
_PN95_STATS = {
    "ticks_total": 0,
    "ticks_pressure_check": 0,
    "ticks_demote_triggered": 0,
    "blocks_demoted_total": 0,
    "blocks_promoted_total": 0,
    "last_free_mib": 0,
    "last_demote_count": 0,
}
# Cache config envs — read once at module init, not on every tick
# (was causing measurable overhead per call). Override via reset_env_cache().
_TICK_EVERY_CACHED: Optional[int] = None
_THRESHOLD_CACHED: Optional[int] = None
_DEMOTE_BATCH_CACHED: Optional[int] = None
_FREE_MIB_CACHE_TTL: int = 5  # cache mem_get_info for N consecutive ticks
_FREE_MIB_CACHE_VALID: int = 0


# M.4.1 — `_read_env_int` extracted to `.pn95.gates`.
from .pn95.gates import _read_env_int  # noqa: E402


# M.4.2.I — `_refresh_env_cache` + `_gpu_free_mib` extracted to
# `.pn95.hooks` (helpers for scheduler_tick). The `global` rebinds for
# `_TICK_EVERY_CACHED` / `_THRESHOLD_CACHED` / `_DEMOTE_BATCH_CACHED`
# are replicated via `_rt.X = …` attribute writes.
from .pn95.hooks import _refresh_env_cache, _gpu_free_mib  # noqa: E402


# M.4.1 — `get_pn95_stats` and `_pn95_dump_stats_if_due` extracted to
# `.pn95.metrics`. They late-import the foreign state (`_PN95_PREFIX_STORE`,
# `_PN95_PREFETCH_STATS`, `_PN95_LAYER_ACCESS_COUNTS`, `_PN95_COMPRESS_LIB`,
# `_pn95_l1_pool`, `_TICK_COUNTER`) from this module to avoid a circular
# import; M.4.2 will move those singletons into focused modules too.
from .pn95.metrics import get_pn95_stats, _pn95_dump_stats_if_due  # noqa: E402


# ─── Path C v1.0 Phase 4 — prefix-cache extension to CPU pinned RAM ──────
#
# Strategy: instead of demoting ARBITRARY GPU blocks (race-prone), we
# intercept exactly two BlockPool events that are already safe:
#
#   1. demote_on_evict — called from `_maybe_evict_cached_block` AFTER
#      the block has been removed from `cached_block_hash_to_block` and
#      BEFORE `block.reset_hash()`. At this moment the block has ref_cnt=0
#      (no readers), is not in any active `block_table`, and vllm is
#      about to recycle the GPU slot. We safely copy the bytes to CPU.
#
#   2. promote_on_miss — called from `get_cached_block` when vllm's own
#      lookup returned None (cache miss). We check our CPU store; if the
#      hash is there, we allocate a fresh GPU block via `get_new_blocks(1)`,
#      copy CPU→GPU, re-insert into vllm's prefix cache, and return it.
#      vllm sees a normal cache hit — no further changes needed.
#
# Effect: prefix cache effective capacity = N_gpu_blocks + N_cpu_entries
# Multi-turn / long-history workloads see dramatically higher hit rate
# without any CUDA OOM risk and without any hot-path overhead (no polling,
# no per-tick mem_get_info — the path only fires on actual eviction events).
#
# Compatible with hybrid-GDN models (Mamba SSM groups never enter the
# prefix cache to begin with — only attention groups have block hashes).
# Compatible with TP=2+ — each worker has its own _PN95_PREFIX_STORE
# scoped to that worker's GPU.

from collections import OrderedDict as _OrderedDict  # noqa: E402  — block-local import after PN95 section header
_PN95_PREFIX_STORE: "_OrderedDict[Any, list]" = _OrderedDict()
_PN95_PREFIX_STORE_BYTES_USED: int = 0
_PN95_PREFIX_STORE_MAX_BYTES_CACHED: Optional[int] = None
_PN95_BLOCK_POOL_REFS: list = []
# Lock protecting concurrent writers to _PN95_PREFIX_STORE +
# _PN95_PREFIX_STORE_BYTES_USED. Multiple paths can mutate the store:
# demote_on_evict (scheduler thread), prefetch_blocks (prefetch worker
# thread), _prefix_store_evict_until_fit (called recursively from demote).
# Pre-PN95 the dict was single-threaded so a lock would have been overhead;
# with the new prefetch API + worker_side_proactive_demote we explicitly
# advertise thread-safety, so the lock is required (review finding #12).
_PN95_PREFIX_STORE_LOCK: threading.Lock = threading.Lock()


# ── L1 pinned host cache (optional, gated by GENESIS_ENABLE_PN95_PINNED_POOL).
# Held in a separate module (_pn95_pinned_pool) so the heavy import (torch
# pin_memory) doesn't run at sndr_core boot when the feature is OFF.
#
# Layer payload (list of (layer_name, bytes)) is serialized to a single bytes
# blob via pickle.HIGHEST_PROTOCOL before being placed in the pool — the pool
# itself works on byte slabs of equal slot size. Unpack reverses pickle.
# Pickle overhead is ~5-10 μs per blob, dwarfed by the PCIe transfer savings
# from non-pageable memory (3-5 GB/s pinned vs ~600 MB/s pageable bounce).
# M.4.2.C — `_pn95_pack_layer_data` + `_pn95_unpack_layer_data` extracted
# to `.pn95.compression`. Pure pickle helpers with no state touch.
from .pn95.compression import (  # noqa: E402
    _pn95_pack_layer_data,
    _pn95_unpack_layer_data,
)


# M.4.2.H — `_pn95_l1_pool` extracted to `.pn95.prefix_store`. Pure
# accessor for the pinned-pool singleton.
from .pn95.prefix_store import _pn95_l1_pool  # noqa: E402


# ── Prefetch / warmup API ───────────────────────────────────────────────
# Inspired by SGLang HiCache layer-by-layer prefetch overlap: the engine
# tells PN95 which block_hashes are about to be needed; PN95 warms up the
# fast L1 pinned pool from the slow L2 OrderedDict (or, if not in L2, the
# disk tier) so the actual `promote_on_miss` call lands in L1.
#
# Without prefetch the path on a cold block is:
#   promote_on_miss → L2 OrderedDict.get → numpy.frombuffer → torch.from_numpy
#   → .to(cuda, non_blocking=True from pageable mem) → bounce-buffer copy
#   ~400 μs for a 32 KB block (single attention layer's K+V for one block).
#
# With prefetch the L1 slot is already pinned by the time vllm calls
# promote_on_miss; the GPU read is a single pinned-host DMA at PCIe Gen4
# line rate, ~80 μs.
#
# Stats track hits/misses so operators can see whether prefetch is paying
# off (vs raw L1 demote-side fills).
_PN95_PREFETCH_STATS = {
    "prefetch_calls": 0,
    "prefetch_block_hashes": 0,
    "prefetch_l2_hits_promoted": 0,  # L2 entry copied into L1 pinned
    "prefetch_l2_already_in_l1": 0,  # L1 already warm — no-op
    "prefetch_missing": 0,           # not in L2 or disk — nothing to do
    "prefetch_disk_hits_promoted": 0,
    "prefetch_pool_full_skips": 0,
}


# M.4.2.B — `pn95_prefetch_blocks` + `pn95_get_prefetch_stats` extracted
# to `.pn95.prefetch`. The `_PN95_PREFETCH_STATS` dict + every other state
# singleton this code reads (`_PN95_PREFIX_STORE`, the L1 pool, prefix
# store helpers, the packer) stay defined in this module; the moved
# functions mutate them through `_rt.X` via lazy late-import — including
# the `_rt._PN95_PREFIX_STORE_BYTES_USED += …` attribute rebind that
# replaces the original `global` declaration.
from .pn95.prefetch import (  # noqa: E402
    pn95_prefetch_blocks,
    pn95_get_prefetch_stats,
)


# ── Layer-aware demote priority ──────────────────────────────────────────
# Tracks per-layer access frequency from the promote path so demote can
# prioritize cold layers when capacity is constrained. Implementation is a
# small dict keyed by layer_name; on a 17-attention-layer Qwen3.6 27B model
# the structure stays trivial (<200 bytes). Single-process, single-rank —
# no cross-worker sync needed (each rank decides its own demote order).
#
# Update on every promote restoration; read on every demote sort. The
# heuristic is intentionally simple: layers with the highest cumulative
# promote-read count are deemed "hot" and pushed to the end of the demote
# queue. Cold layers (low counts) are demoted first, freeing GPU memory
# along the path the GPU's attention forward least frequently touches.
#
# Bounded growth: counts are reset on overflow (>10M) to prevent integer
# bloat. The relative ordering is what matters, not absolute values.
_PN95_LAYER_ACCESS_COUNTS: dict = {}
_PN95_LAYER_ACCESS_RESET_THRESHOLD = 10_000_000


# ── store_threshold reuse-frequency gate (upstream PR #40020 pattern) ─────
#
# Tracks how many times each block_hash has been *looked up* during
# promote_on_miss. Blocks with hits below GENESIS_PN95_STORE_THRESHOLD
# are NOT demoted on evict — the engine pays no compression/copy cost
# on a block that's about to disappear from the request stream forever.
#
# Inspired by upstream `FilterReusedOffloadingManager` in cpu/manager.py
# (only stores keys observed `store_threshold` times via lookup).
#
# Default off (threshold=0). Operators set >=2 when serving chat workloads
# where most prefill blocks are one-shot.
# Lookup hit tracker — ownership stays here for M.4.1 (same reason
# as ``_PN95_STATS``: test sites may rebind via monkeypatch). The
# ``_pn95_record_lookup`` function lives in `.pn95.metrics` and
# late-imports this state.
_PN95_HIT_COUNTS: dict = {}
_PN95_HIT_TRACKER_MAX = 64_000

from .pn95.metrics import _pn95_record_lookup  # noqa: E402


# M.4.1 — `_pn95_store_threshold` extracted to `.pn95.gates`.
from .pn95.gates import _pn95_store_threshold  # noqa: E402


# M.4.2.D — `_pn95_should_demote` extracted to `.pn95.demote_policy`.
# Reads `_PN95_HIT_COUNTS` (stays here) via lazy `_rt.X`.
from .pn95.demote_policy import _pn95_should_demote  # noqa: E402


# ── block_size_factor — PCIe transaction amortization ────────────────────
#
# Upstream PR #40020 lets the offload layer operate on `block_size_factor`
# adjacent KV blocks as a single super-block. This amortizes the PCIe
# transaction setup cost (~10-20us per DMA submit) over a larger payload,
# critical when each KV block is small (Qwen3.6 fp8 32KB/block).
#
# At factor=4 we batch four ~32KB blocks into one ~128KB transfer:
#   - submit/sync overhead drops 4×
#   - PCIe is more BW-efficient on larger packets (closer to line rate)
#   - tradeoff: the L1 pinned pool slot_size auto-derives from first
#     super-block payload, so 4× larger slots → fewer slots within
#     GENESIS_PN95_PINNED_POOL_MB budget
#
# Default 1 (no grouping). 2-4 typical sweet spots for production.
# M.4.1 — `_pn95_block_size_factor` extracted to `.pn95.gates`.
from .pn95.gates import _pn95_block_size_factor  # noqa: E402


# M.4.2.H — `pn95_demote_batch` extracted to `.pn95.prefix_store`.
# Was parked here in M.4.2.D because it calls demote_on_evict.
from .pn95.prefix_store import pn95_demote_batch  # noqa: E402


# M.4.1 — `_pn95_layer_aware_enabled` extracted to `.pn95.gates`.
from .pn95.gates import _pn95_layer_aware_enabled  # noqa: E402


# M.4.2.D — `_pn95_record_layer_promote` + `_pn95_sort_layers_cold_first`
# extracted to `.pn95.demote_policy`. `_PN95_LAYER_ACCESS_COUNTS` +
# `_PN95_LAYER_ACCESS_RESET_THRESHOLD` stay defined in this module; the
# moved functions read/rebind them via lazy `_rt.X` (the «halve all
# counters» overflow path uses explicit attribute mutation that hits
# the same module-attribute slot the original `global` declaration did).
from .pn95.demote_policy import (  # noqa: E402
    _pn95_record_layer_promote,
    _pn95_sort_layers_cold_first,
)

# Path C v1.0 Quality-First Sprint Q1 A1 — lossless CPU prefix compression.
# Reduces effective CPU tier capacity 2-3× via zstd (or 1.5-2× via lz4).
# Detection at decompress is via magic bytes — no per-entry header overhead.
# Quality: 100% (lossless by construction).
_PN95_COMPRESS_LIB: Optional[str] = None  # 'zstd'|'lz4'|'zlib'|'none'|None
_PN95_COMPRESS_LEVEL: Optional[int] = None
_PN95_COMPRESS_MIN_BYTES = 256  # entries smaller skip compression (overhead)
# Sprint Q1 B6 — per-thread cached compressor/decompressor instances.
# threading.local ensures each ThreadPool worker has own cached instance
# (avoids race in singleton init AND any potential thread-safety nuance
# of underlying C library context state).
_PN95_ZSTD_TL = threading.local()


# M.4.2.C — six compression helpers extracted to `.pn95.compression`.
# State singletons (`_PN95_COMPRESS_LIB`, `_PN95_COMPRESS_LEVEL`,
# `_PN95_COMPRESS_MIN_BYTES`, `_PN95_ZSTD_TL`, `_PN95_COMPRESS_POOL`)
# stay defined in this module — four test files actively rebind them via
# ``monkeypatch.setattr(rt, ...)`` and direct ``rt._PN95_COMPRESS_POOL = None``
# writes, so moving the names would break the test contract. The moved
# functions reach the state through lazy ``_rt.X`` and replicate the
# original ``global ... = …`` rebinds via explicit attribute mutation.
# ``_PN95_COMPRESS_POOL`` keeps its definition in this module — see the
# state-singleton block below.
from .pn95.compression import (  # noqa: E402
    _pn95_init_compression,
    _pn95_compress_bytes,
    _pn95_compress_pool,
    _pn95_compress_bytes_batch,
    _pn95_decompress_bytes_batch,
    _pn95_decompress_bytes,
)

_PN95_COMPRESS_POOL: Optional[Any] = None  # ThreadPoolExecutor for parallel compress


# M.4.2.H — prefix-store accounting + register_block_pool extracted to
# `.pn95.prefix_store`. State (`_PN95_PREFIX_STORE`,
# `_PN95_PREFIX_STORE_BYTES_USED`, `_PN95_PREFIX_STORE_MAX_BYTES_CACHED`,
# `_PN95_BLOCK_POOL_REFS`, `_PN95_PREFIX_STORE_LOCK`) stays in this
# module. The original `global` rebinds become `_rt.X` attribute writes.
# Sibling-patch anchors at pn95_tier_aware_cache.py:181/334/342 import
# `register_block_pool` through _pn95_runtime — shim preserves the path.
from .pn95.prefix_store import (  # noqa: E402
    _prefix_store_max_bytes,
    _prefix_store_evict_until_fit,
    register_block_pool,
)


# M.4.2.H — `demote_on_evict` extracted to `.pn95.prefix_store`.
# Sibling-patch text-anchor in pn95_tier_aware_cache.py:213 imports
# this name through _pn95_runtime — shim preserves the path.
from .pn95.prefix_store import demote_on_evict  # noqa: E402


# M.4.2.H — `promote_on_miss` extracted to `.pn95.prefix_store`.
# Sibling-patch text-anchor in pn95_tier_aware_cache.py:437 imports
# this name through _pn95_runtime — shim preserves the path.
from .pn95.prefix_store import promote_on_miss  # noqa: E402


# M.4.2.D — `_select_cold_blocks_via_bpool_lru` extracted to
# `.pn95.demote_policy`. Reads `_PN95_BLOCK_POOL_REFS`, `_TM`, and
# `_PN95_PREFIX_STORE` (all stay here) via lazy `_rt.X`.
from .pn95.demote_policy import _select_cold_blocks_via_bpool_lru  # noqa: E402


# M.4.2.H — `worker_side_proactive_demote` + `_proactive_demote_cold`
# extracted to `.pn95.prefix_store`. Both were parked here in M.4.2.D
# because they call demote_on_evict.
from .pn95.prefix_store import (  # noqa: E402
    worker_side_proactive_demote,
    _proactive_demote_cold,
)


# M.4.2.I — scheduler_tick extracted to `.pn95.hooks`. Sibling-patch
# text-anchor in pn95_tier_aware_cache.py:160 imports this name through
# _pn95_runtime — shim preserves the path byte-identical. The 3 `global`
# rebinds (_TICK_COUNTER, _TICK_LAST_FREE_MIB, _FREE_MIB_CACHE_VALID)
# are replicated via `_rt.X = …` attribute writes inside the moved fn.
from .pn95.hooks import scheduler_tick  # noqa: E402


# M.4.2.A — `tier_manager` + `reset_for_tests` extracted to
# `.pn95.runtime_state`. State ownership (`_TM`, `_LOCK`,
# `_LAST_GROUP_IDS_BY_HASH`) stays in this module; the moved functions
# read/rebind via lazy late-import (`_rt._TM = None` / `return _rt._TM`)
# so the local module attribute remains the canonical name.
from .pn95.runtime_state import tier_manager, reset_for_tests  # noqa: E402

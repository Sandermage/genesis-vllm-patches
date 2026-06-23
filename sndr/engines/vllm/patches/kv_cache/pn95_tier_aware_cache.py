# SPDX-License-Identifier: Apache-2.0
"""PN95 v7.73.x — tier-aware KV cache wire-in (Path C, club-3090 #58).

Two minimal text-patch anchors that route into the runtime singleton
at `sndr/cache/_pn95_runtime.py`:

  1. `vllm/v1/core/single_type_kv_cache_manager.py::cache_blocks` —
     after vLLM's `cache_full_blocks()` updates `num_cached_block`,
     notify the TierManager so it admits the newly cached block range.

  2. `vllm/v1/core/block_pool.py::get_cached_block` — before the
     final `return cached_blocks`, notify the TierManager so it
     touches the block hash (returns demoted bytes on tier-1 hit;
     promote logic stays inside the manager).

Both injections are wrapped in `try/except` so any runtime error is
swallowed — the surrounding vLLM code path stays alive.

Default OFF (`GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1` to enable). When
the env flag is unset, both `notify_admit()` and `notify_touch()` are
fast-path no-ops (single dict lookup + None check).

The `init_from_config()` singleton install happens in the dispatcher
hook; the wire-in module here only injects the runtime call sites.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatcher,
    TextPatch,
)

log = logging.getLogger("genesis.pn95")

GENESIS_PN95_MARKER = (
    "Genesis PN95 tier-aware KV cache + vision sub-tier (Path C v7.73.x_v11.3.0_hotpath)"
)

# ─── Anchor 1: single_type_kv_cache_manager.py::cache_blocks ──────────
#
# Original (in the BASE class `SingleTypeKVCacheManager.cache_blocks`):
#
#     self.num_cached_block[request.request_id] = num_full_blocks
#
# (this is the LAST line of the method; safe injection point — after
# `cache_full_blocks()` returns, before the method exits)
#
# Replacement: append a no-op fail-silent notify_admit() call.

PN95_SITE1_OLD = (
    "        self.num_cached_block[request.request_id] = num_full_blocks\n"
)
PN95_SITE1_NEW = (
    "        self.num_cached_block[request.request_id] = num_full_blocks\n"
    "        # [Genesis PN95 v2 — hot-path optimized] tier-aware admit.\n"
    "        _g_pn95_admit = globals().get('_GENESIS_PN95_admit_fn')\n"
    "        if _g_pn95_admit is None:\n"
    "            try:\n"
    "                from sndr.cache._pn95_runtime import notify_admit as _g_pn95_admit\n"
    "            except Exception:\n"
    "                _g_pn95_admit = False\n"
    "            globals()['_GENESIS_PN95_admit_fn'] = _g_pn95_admit\n"
    "        if _g_pn95_admit:\n"
    "            try:\n"
    "                _g_pn95_admit(request, num_cached_blocks, num_full_blocks,\n"
    "                              self.kv_cache_group_id, self.block_size)\n"
    "            except Exception:\n"
    "                pass\n"
)

# ─── Anchor 2: block_pool.py::get_cached_block ────────────────────────
#
# Original (tail of get_cached_block, after the for-loop that builds
# `cached_blocks`):
#
#             cached_blocks.append(block)
#         return cached_blocks
#
# Replacement: insert touch-notify before the final return.

PN95_SITE2_OLD = (
    "            cached_blocks.append(block)\n"
    "        return cached_blocks\n"
)
PN95_SITE2_NEW = (
    "            cached_blocks.append(block)\n"
    "        # [Genesis PN95 v2 — hot-path optimized] tier-aware touch.\n"
    "        # Cache the notify_touch callable in upstream-file globals;\n"
    "        # called per block_pool.get_cached_block — high-frequency.\n"
    "        _g_pn95_touch = globals().get('_GENESIS_PN95_touch_fn')\n"
    "        if _g_pn95_touch is None:\n"
    "            try:\n"
    "                from sndr.cache._pn95_runtime import notify_touch as _g_pn95_touch\n"
    "            except Exception:\n"
    "                _g_pn95_touch = False\n"
    "            globals()['_GENESIS_PN95_touch_fn'] = _g_pn95_touch\n"
    "        if _g_pn95_touch:\n"
    "            try:\n"
    "                _g_pn95_touch(block_hash, kv_cache_group_ids, cached_blocks)\n"
    "            except Exception:\n"
    "                pass\n"
    "        return cached_blocks\n"
)


# ─── Anchor 3: kv_cache_manager.py::KVCacheManager.__init__ ────────────
#
# Day 6 (UNIFIED_CONFIG plan 2026-05-09): hook into the manager's
# __init__ to register every MambaSpec group as excluded from demotion
# AND lazy-init the TierManager singleton from GENESIS_PN95_CONFIG_KEY
# env var when running inside vllm worker process (spawn-based).
#
# Anchor: end of `__init__` method, after `self.empty_kv_cache_blocks = ...`
# block. Inject the call right after.

PN95_SITE3_OLD = (
    "        self.empty_kv_cache_blocks = KVCacheBlocks(\n"
    "            tuple(() for _ in range(self.num_kv_cache_groups))\n"
    "        )\n"
)
PN95_SITE3_NEW = (
    "        self.empty_kv_cache_blocks = KVCacheBlocks(\n"
    "            tuple(() for _ in range(self.num_kv_cache_groups))\n"
    "        )\n"
    "        # [Genesis PN95] Mamba SSM exclusion + lazy TierManager init\n"
    "        try:\n"
    "            from sndr.cache._pn95_runtime import init_mamba_exclusions_from_kv_groups as _g_pn95_init\n"
    "            _g_pn95_init(kv_cache_config.kv_cache_groups)\n"
    "        except Exception:\n"
    "            pass\n"
)


# ─── Anchor 4: gpu_model_runner.py::initialize_kv_cache (Path C v1.0 Phase 1)
#
# Path C v1.0 Phase 1 (UNIFIED_CONFIG plan 2026-05-09): bridge from
# vLLM worker-level GPU tensor refs to TierManager. Anchored
# immediately after `kv_caches = self.initialize_kv_cache_tensors(...)`
# so we have the per-layer GPU tensor list in scope.
#
# Phase 1: register_kv_caches() records shapes + tensor refs on the
# TierManager (observability). Phase 2 (v1.0 final) will use those
# refs to perform actual cudaMemcpyAsync demote/promote.

PN95_SITE4_OLD = (
    "        kv_caches = self.initialize_kv_cache_tensors(\n"
    "            kv_cache_config, kernel_block_sizes\n"
    "        )\n"
)
PN95_SITE4_NEW = (
    "        kv_caches = self.initialize_kv_cache_tensors(\n"
    "            kv_cache_config, kernel_block_sizes\n"
    "        )\n"
    "        # [Genesis PN95 v1.0 Phase 1] register kv_caches refs into TierManager\n"
    "        try:\n"
    "            from sndr.cache._pn95_runtime import register_kv_caches as _g_pn95_reg\n"
    "            _g_pn95_reg(kv_caches, self.kv_cache_config.kv_cache_groups)\n"
    "        except Exception:\n"
    "            pass\n"
)


# ─── Anchor 5: scheduler.py::Scheduler.schedule (Path C v1.0 Phase 2) ──
# v2 (2026-06-08, archive-drift forensics): upstream inserted
# ``self.current_step += 1`` between the method signature and the NOTE
# comment for per-step accounting. PN95's tick hook still injects at the
# top of schedule(), it just moves AFTER the new step-counter assignment
# instead of before the comment. Functionally identical — the step
# counter fires per call regardless of when we tick.
PN95_SITE5_OLD = (
    "    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:\n"
    "        self.current_step += 1\n"
    "        # NOTE(woosuk) on the scheduling algorithm:\n"
)
PN95_SITE5_NEW = (
    "    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:\n"
    "        self.current_step += 1\n"
    "        # [Genesis PN95 v2 — hot-path optimized] periodic tier maintenance.\n"
    "        # schedule() fires per scheduler step (~50/sec at sustained TPS).\n"
    "        _g_pn95_tick = globals().get('_GENESIS_PN95_tick_fn')\n"
    "        if _g_pn95_tick is None:\n"
    "            try:\n"
    "                from sndr.cache._pn95_runtime import scheduler_tick as _g_pn95_tick\n"
    "            except Exception:\n"
    "                _g_pn95_tick = False\n"
    "            globals()['_GENESIS_PN95_tick_fn'] = _g_pn95_tick\n"
    "        if _g_pn95_tick:\n"
    "            try:\n"
    "                _g_pn95_tick()\n"
    "            except Exception:\n"
    "                pass\n"
    "        # NOTE(woosuk) on the scheduling algorithm:\n"
)


# ─── Anchor 6: block_pool.py::BlockPool.__init__ (Path C v1.0 Phase 4) ──
#
# Phase 4: register self with PN95 runtime so promote_on_miss can call
# self.get_new_blocks(1) to materialize a CPU-stored block back to GPU.
# Anchor at end of __init__, after metrics_collector assignment.

PN95_SITE6_OLD = (
    "        self.metrics_collector = metrics_collector\n"
)
PN95_SITE6_NEW = (
    "        self.metrics_collector = metrics_collector\n"
    "        # [Genesis PN95 v1.0 Phase 4] register BlockPool for promote-on-miss\n"
    "        try:\n"
    "            from sndr.cache._pn95_runtime import register_block_pool as _g_pn95_regpool\n"
    "            _g_pn95_regpool(self)\n"
    "        except Exception:\n"
    "            pass\n"
)


# ─── Anchor 7: block_pool.py::_maybe_evict_cached_block (Phase 4) ───────
#
# Phase 4: demote-on-evict — capture GPU block bytes to CPU pinned
# storage right before vllm wipes the hash + recycles the slot.
# At this point block has ref_cnt=0 (no readers) and was just removed
# from cached_block_hash_to_block — safe to copy without race.
#
# Anchor BEFORE block.reset_hash() — we still have block_hash + block_id.

PN95_SITE7_OLD = (
    "        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:\n"
    "            # block not found in cached_block_hash_to_block,\n"
    "            # eviction is not needed\n"
    "            return False\n"
    "\n"
    "        block.reset_hash()\n"
)
PN95_SITE7_NEW = (
    "        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:\n"
    "            # block not found in cached_block_hash_to_block,\n"
    "            # eviction is not needed\n"
    "            return False\n"
    "\n"
    "        # [Genesis PN95 v1.0 Phase 4] demote-on-evict — fail-silent\n"
    "        try:\n"
    "            from sndr.cache._pn95_runtime import demote_on_evict as _g_pn95_demote_ev\n"
    "            _g_pn95_demote_ev(block_hash, block.block_id)\n"
    "        except Exception:\n"
    "            pass\n"
    "        block.reset_hash()\n"
)


# ─── Anchor 11 (SITE11): block_pool.py::get_new_blocks (Phase 5 Session 3)
#
# Path C v1.0 Phase 5 Session 3 — virtual block materialization on
# allocation. After popleft_n returns N blocks, walk the list and
# materialize any virtual blocks (those with physical_resident=False
# in PN95 side-table) via swap-based virtualization:
#
#   1. Find a "donor" cached block in the pool (ref_cnt=0, physical)
#   2. Demote donor's bytes to CPU prefix store (Phase 4 mechanism)
#   3. Adopt donor's physical_block_id for the virtual block
#   4. Mutate block.block_id = donor.physical_block_id
#
# Result: caller gets blocks with valid physical block_ids — attention
# reads tensor[block_id] normally. ZERO patches needed on attention path.
#
# Anchor: insert AFTER popleft_n line, BEFORE the eviction loop.
# Env-gated: PN95 disabled OR VIRT=0 → no-op (preserves current behavior).

PN95_SITE11_OLD = (
    "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n"
    "\n"
    "        # In order to only iterate the list once, we duplicated code a bit\n"
    "        if self.enable_caching:\n"
)
PN95_SITE11_NEW = (
    "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n"
    "\n"
    "        # [Genesis PN95 v1.0 Phase 5 Anchor #12] virtual block materialization\n"
    "        try:\n"
    "            from sndr.cache._pn95_runtime import pn95_anchor12_post_popleft as _g_pn95_a12\n"
    "            if not _g_pn95_a12(self, ret):\n"
    "                # Virtual materialization failed — return blocks to queue\n"
    "                # and raise (caller will retry or scheduler will queue request)\n"
    "                self.free_block_queue.append_n(ret)\n"
    "                raise ValueError(\n"
    "                    f\"PN95 cannot materialize {num_blocks} virtual blocks — \"\n"
    "                    f\"GPU pool truly exhausted (no donor available)\"\n"
    "                )\n"
    "        except ValueError:\n"
    "            raise\n"
    "        except Exception:\n"
    "            pass\n"
    "\n"
    "        # In order to only iterate the list once, we duplicated code a bit\n"
    "        if self.enable_caching:\n"
)


# ─── Anchor 13: block_pool.py::get_new_blocks — worker-side proactive demote
#
# Path C v1.0 Phase 4.2 — close the multiproc gap. SITE5 scheduler_tick
# runs in the EngineCore process whose `_PN95_BLOCK_POOL_REFS` is empty
# (pools live in Worker processes), so the proactive demote branch
# never fires in a real multiproc deploy.
#
# This anchor inserts the proactive trigger BEFORE the size check in
# BlockPool.get_new_blocks — which runs in the Worker process that
# OWNS the BlockPool. When the free queue length drops below
# `GENESIS_PN95_PROACTIVE_FREE_BLOCKS_THRESHOLD` (default 32), we walk
# the head of free_block_queue and capture each cached block's bytes
# to the CPU prefix store (compressed if GENESIS_PN95_COMPRESS is set).
# vllm's allocator then proceeds normally — the GPU slot gets reused
# for the new request, but the prefix-cache content is preserved on
# CPU so a future hash hit can promote-on-miss it back.
#
# Net effect: prefix cache effectively grows beyond the GPU pool size,
# multi-turn workloads see sustained cache hits across longer windows,
# and PN95's pressure-driven preserve mechanism is finally wired to a
# process that holds the BlockPool reference.

PN95_SITE13_OLD = (
    "        if num_blocks > self.get_num_free_blocks():\n"
    "            raise ValueError(f\"Cannot get {num_blocks} free blocks from the pool\")\n"
    "\n"
    "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n"
)
PN95_SITE13_NEW = (
    "        # [Genesis PN95 v1.0 Phase 4.2 Anchor #13] worker-side proactive demote\n"
    "        try:\n"
    "            import os as _g_pn95_os\n"
    "            _g_pn95_thr = int(_g_pn95_os.environ.get(\n"
    "                \"GENESIS_PN95_PROACTIVE_FREE_BLOCKS_THRESHOLD\", \"32\"))\n"
    "            if self.get_num_free_blocks() <= _g_pn95_thr:\n"
    "                from sndr.cache._pn95_runtime import (\n"
    "                    worker_side_proactive_demote as _g_pn95_wpd,\n"
    "                )\n"
    "                _g_pn95_wpd(self, target_count=min(8, _g_pn95_thr))\n"
    "        except Exception:\n"
    "            pass\n"
    "        if num_blocks > self.get_num_free_blocks():\n"
    "            raise ValueError(f\"Cannot get {num_blocks} free blocks from the pool\")\n"
    "\n"
    "        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)\n"
)


# ─── Anchor 10: block_pool.py::BlockPool.__init__ (Phase 5 Session 2) ───
#
# Path C v1.0 Phase 5 Session 2 — initialize PN95 side-table metadata for
# blocks. KVCacheBlock is @dataclass(slots=True) — cannot add fields,
# so we use side-table in _pn95_runtime keyed by (id(pool), block_id).
#
# Anchor on the existing Phase 4 SITE6 injection's `pass\n` line (will
# only match if Phase 4 anchors already applied — explicit ordering
# dependency in apply()).
#
# Session 2 = METADATA INFRASTRUCTURE ONLY. Defaults preserve current
# vllm behavior. Sessions 3-4 will conditionally inflate logical pool
# and materialize virtual blocks using this metadata.

PN95_SITE10_OLD = (
    "        # [Genesis PN95 v1.0 Phase 4] register BlockPool for promote-on-miss\n"
    "        try:\n"
    "            from sndr.cache._pn95_runtime import register_block_pool as _g_pn95_regpool\n"
    "            _g_pn95_regpool(self)\n"
    "        except Exception:\n"
    "            pass\n"
)
PN95_SITE10_NEW = (
    "        # [Genesis PN95 v1.0 Phase 4] register BlockPool for promote-on-miss\n"
    "        try:\n"
    "            from sndr.cache._pn95_runtime import register_block_pool as _g_pn95_regpool\n"
    "            _g_pn95_regpool(self)\n"
    "        except Exception:\n"
    "            pass\n"
    "        # [Genesis PN95 v1.0 Phase 5 Anchor #11] init block metadata side-table\n"
    "        try:\n"
    "            from sndr.cache._pn95_runtime import pn95_phase5_init_block_pool as _g_pn95_p5init\n"
    "            _g_pn95_p5init(self)\n"
    "        except Exception:\n"
    "            pass\n"
)


# ─── Anchor 9: kv_cache_utils.py::_check_enough_kv_cache_memory (Phase 5)
#
# Path C v1.0 Phase 5 Session 1 — boot-time KV pool expansion.
#
# vllm's pre-flight check rejects boot when max_seq_len's KV needs
# exceed available_memory (GPU-only). PN95 Phase 5 expands the check
# by adding CPU tier capacity to the comparison budget.
#
# CRITICAL: this anchor inflates ONLY the local check variable, NOT
# the parameter that propagates downstream. Caller's `available_memory`
# stays unchanged → KVCacheTensor allocation uses GPU-only memory →
# no CUDA OOM at init. The trade-off: boot passes with inflated
# max_seq_len, but runtime requests requesting > GPU pool capacity
# will hit "Cannot get N free blocks" at allocate time. Sessions 2-4
# add virtualization (#11/#12/#13) to materialize the runtime path.
#
# Env gate: GENESIS_PN95_VIRT_ENABLE=1 (default OFF for safety).
# When OFF: helper returns 0, anchor becomes no-op.

# dev354+ form: upstream expanded the error message with a CPU-backend
# clarification ("(this flag also controls CPU memory reservation ...)").
# Anchor below matches the new form. If we revert to an older pin, retune.
PN95_SITE9_OLD = (
    "    if available_memory <= 0:\n"
    "        raise ValueError(\n"
    "            \"No available memory for the cache blocks. \"\n"
    "            \"Try increasing `gpu_memory_utilization` when initializing the engine \"\n"
    "            \"(this flag also controls CPU memory reservation on the CPU \"\n"
    "            \"backend, despite its name). \"\n"
    "            \"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ \"\n"
    "            \"for more details.\"\n"
    "        )\n"
    "\n"
    "    needed_memory = get_needed_memory()\n"
    "\n"
    "    if needed_memory > available_memory:\n"
)
PN95_SITE9_NEW = (
    "    # [Genesis PN95 v1.0 Phase 5 Anchor #9] tier-aware boot check\n"
    "    # — inflate LOCAL check budget by CPU tier capacity. Does NOT\n"
    "    # propagate to caller; downstream tensor alloc uses original.\n"
    "    available_memory_for_check = available_memory\n"
    "    try:\n"
    "        from sndr.cache._pn95_runtime import pn95_extra_logical_memory_bytes as _g_pn95_extra\n"
    "        _extra = _g_pn95_extra()\n"
    "        if _extra > 0:\n"
    "            available_memory_for_check = available_memory + _extra\n"
    "    except Exception:\n"
    "        pass\n"
    "    if available_memory_for_check <= 0:\n"
    "        raise ValueError(\n"
    "            \"No available memory for the cache blocks. \"\n"
    "            \"Try increasing `gpu_memory_utilization` when initializing the engine \"\n"
    "            \"(this flag also controls CPU memory reservation on the CPU \"\n"
    "            \"backend, despite its name). \"\n"
    "            \"See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ \"\n"
    "            \"for more details.\"\n"
    "        )\n"
    "\n"
    "    needed_memory = get_needed_memory()\n"
    "\n"
    "    if needed_memory > available_memory_for_check:\n"
)


# ─── Anchor 8: block_pool.py::get_cached_block (Phase 4) ────────────────
#
# Phase 4: promote-on-miss — when vllm's own prefix cache lookup misses,
# check our CPU prefix store and restore the block. This is the second
# half of the demote-on-evict / promote-on-miss pair.
#
# Anchor inside the for-loop, replacing the `if not block: return None`
# with a fallback path that calls promote_on_miss before giving up.

PN95_SITE8_OLD = (
    "            block = self.cached_block_hash_to_block.get_one_block(\n"
    "                block_hash_with_group_id\n"
    "            )\n"
    "            if not block:\n"
    "                return None\n"
    "            cached_blocks.append(block)\n"
)
PN95_SITE8_NEW = (
    "            block = self.cached_block_hash_to_block.get_one_block(\n"
    "                block_hash_with_group_id\n"
    "            )\n"
    "            if not block:\n"
    "                # [Genesis PN95 v1.0 Phase 4] promote-on-miss — fail-silent\n"
    "                try:\n"
    "                    from sndr.cache._pn95_runtime import promote_on_miss as _g_pn95_promote_m\n"
    "                    block = _g_pn95_promote_m(self, block_hash_with_group_id)\n"
    "                except Exception:\n"
    "                    block = None\n"
    "                if not block:\n"
    "                    return None\n"
    "            cached_blocks.append(block)\n"
)


def _make_patcher_kvcm() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/single_type_kv_cache_manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/single_type_kv_cache_manager.py — tier-aware admit"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — admit anchor",
        sub_patches=[
            TextPatch(
                name="pn95_admit_anchor_at_cache_blocks",
                anchor=PN95_SITE1_OLD,
                replacement=PN95_SITE1_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN95",
            "_g_pn95_",
            # Upstream-side markers if vLLM ships native multi-tier KV cache:
            "tier_manager",
            "TierManager",
            "cpu_offload_kv_blocks",
        ],
    )


def _make_patcher_blockpool() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/block_pool.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/block_pool.py — tier-aware touch"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — touch anchor",
        sub_patches=[
            TextPatch(
                name="pn95_touch_anchor_at_get_cached_block",
                anchor=PN95_SITE2_OLD,
                replacement=PN95_SITE2_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN95",
            "_g_pn95_",
        ],
    )


def _make_patcher_kv_manager_init() -> TextPatcher | None:
    """Day 6: anchor at KVCacheManager.__init__ for Mamba exclusion + lazy init."""
    target = resolve_vllm_file("v1/core/kv_cache_manager.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/kv_cache_manager.py — Mamba exclusion + TM init"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — mamba+init anchor",
        sub_patches=[
            TextPatch(
                name="pn95_mamba_init_anchor",
                anchor=PN95_SITE3_OLD,
                replacement=PN95_SITE3_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN95",
            "_g_pn95_",
        ],
    )


def _make_patcher_register_kv_caches() -> TextPatcher | None:
    """v1.0 Phase 1: anchor at gpu_model_runner.py::initialize_kv_cache for register_kv_caches."""
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/worker/gpu_model_runner.py — register_kv_caches (v1.0 Phase 1)"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — register_kv_caches anchor",
        sub_patches=[
            TextPatch(
                name="pn95_register_kv_caches_anchor",
                anchor=PN95_SITE4_OLD,
                replacement=PN95_SITE4_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN95",
            "_g_pn95_",
        ],
    )


def _make_patcher_scheduler_tick() -> TextPatcher | None:
    """v1.0 Phase 2: anchor at Scheduler.schedule for periodic tier maintenance."""
    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/sched/scheduler.py — scheduler_tick (v1.0 Phase 2)"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — scheduler_tick anchor",
        sub_patches=[
            TextPatch(
                name="pn95_scheduler_tick_anchor",
                anchor=PN95_SITE5_OLD,
                replacement=PN95_SITE5_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN95",
            "_g_pn95_",
        ],
    )


def _make_patcher_blockpool_register() -> TextPatcher | None:
    """v1.0 Phase 4: anchor at BlockPool.__init__ for register_block_pool."""
    target = resolve_vllm_file("v1/core/block_pool.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/block_pool.py — register_block_pool (v1.0 Phase 4)"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — blockpool register anchor",
        sub_patches=[
            TextPatch(
                name="pn95_register_block_pool_anchor",
                anchor=PN95_SITE6_OLD,
                replacement=PN95_SITE6_NEW,
                required=True,
            ),
        ],
        # Phase 4 anchors share the file with Phase 1 (touch). Drift markers
        # must be UPSTREAM-side only (not our own markers) to avoid mutual
        # blocking when multiple anchors live in the same file.
        upstream_drift_markers=[
            "tier_manager",
            "TierManager",
            "cpu_offload_kv_blocks",
        ],
    )


def _make_patcher_demote_on_evict() -> TextPatcher | None:
    """v1.0 Phase 4: anchor at _maybe_evict_cached_block for demote_on_evict."""
    target = resolve_vllm_file("v1/core/block_pool.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/block_pool.py — demote_on_evict (v1.0 Phase 4)"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — demote_on_evict anchor",
        sub_patches=[
            TextPatch(
                name="pn95_demote_on_evict_anchor",
                anchor=PN95_SITE7_OLD,
                replacement=PN95_SITE7_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "tier_manager",
            "TierManager",
            "cpu_offload_kv_blocks",
        ],
    )


def _make_patcher_worker_proactive_demote() -> TextPatcher | None:
    """Phase 4.2 Anchor #13 — worker-side proactive demote.

    Inserts the pressure-driven trigger at the top of
    `BlockPool.get_new_blocks`. Runs in the Worker process that owns
    the live BlockPool reference, closing the EngineCore-vs-Worker
    multiproc gap that left the scheduler-tick proactive branch dead
    in real deployments.

    Threshold is env-driven (default 32 free blocks); below that we
    walk the head of free_block_queue and preserve each cached
    block's bytes to the CPU prefix store before vllm reuses the slot.

    Opt-in: gated by `GENESIS_PN95_ANCHOR13_ENABLE=1`. Default off
    because the anchor modifies `vllm/v1/core/block_pool.py` source
    bytes, which can invalidate the torch.inductor compile cache and
    force a cold recompile on next boot (5-10 minute warmup). When
    the operator wants the proactive demote loop they set the env;
    the default `1` deploy stays bit-identical to the pre-anchor
    state so existing compile caches keep hitting.

    v2 (2026-06-08, archive-drift forensics) — **RETIRED on dev259+**:
    PN96 Phase 6 emergency-demote (lines 385-411 of
    ``vllm/v1/core/block_pool.py`` in dev259) has fully superseded
    PN95's Phase 4.2 pressure-driven preservation loop with a more
    robust per-block walk that runs inside ``get_new_blocks`` itself.
    The PN95 anchor #13 position is now occupied by the PN96 emergency
    demote code, so this patch's anchor cannot match and forcing it
    would conflict with PN96. The function preserves an early-return
    so the env flag remains a no-op without raising; operators flipping
    the flag get a clean skip with the retired reason logged.
    """
    # Retired: PN96 Phase 6 superseded the Phase 4.2 logic. Always None.
    # No longer invoked by apply() (deep-audit 2026-06-14 #4). Kept for
    # one-block re-enable if PN96 regresses; log at DEBUG so a manual
    # re-invocation does not spam the boot log.
    log.debug(
        "[PN95] anchor #13 worker_proactive_demote RETIRED on dev259+ — "
        "PN96 Phase 6 emergency-demote at block_pool.py:385-411 superseded "
        "the Phase 4.2 pressure-driven preservation loop."
    )
    return None
    # Original implementation preserved below for archaeology — kept dead
    # so re-enabling is a one-block edit if PN96 ever regresses upstream.
    if os.environ.get("GENESIS_PN95_ANCHOR13_ENABLE", "0").strip().lower() not in (  # noqa: F841
        "1", "true", "yes", "on",
    ):
        return None
    target = resolve_vllm_file("v1/core/block_pool.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/block_pool.py — worker-side proactive demote "
            "(v1.0 Phase 4.2 Anchor #13)"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — anchor #13 worker proactive demote",
        sub_patches=[
            TextPatch(
                name="pn95_worker_proactive_demote_anchor",
                anchor=PN95_SITE13_OLD,
                replacement=PN95_SITE13_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "tier_manager",
            "TierManager",
            "cpu_offload_kv_blocks",
        ],
    )


def _make_patcher_phase5_get_new_blocks() -> TextPatcher | None:
    """v1.0 Phase 5 Session 3 — anchor at BlockPool.get_new_blocks for
    virtual block materialization (Anchor #12).

    After popleft_n returns N blocks, materialize any virtual blocks via
    swap-based approach. Pure runtime no-op when PN95 OR VIRT disabled.
    """
    target = resolve_vllm_file("v1/core/block_pool.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/block_pool.py — phase5 get_new_blocks materialize "
            "(v1.0 Phase 5 Session 3)"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — phase5 get_new_blocks anchor",
        sub_patches=[
            TextPatch(
                name="pn95_phase5_get_new_blocks_anchor",
                anchor=PN95_SITE11_OLD,
                replacement=PN95_SITE11_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "tier_manager",
            "TierManager",
            "cpu_offload_kv_blocks",
        ],
    )


def _make_patcher_phase5_block_pool_init() -> TextPatcher | None:
    """v1.0 Phase 5 Session 2 — anchor at BlockPool.__init__ for PN95
    side-table metadata initialization.

    Depends on Phase 4 SITE6 (register_block_pool) being applied first
    — anchor matches the Phase 4 injection text. apply() guarantees
    Phase 4 runs before Phase 5.

    Session 2 = METADATA INFRASTRUCTURE ONLY (zero behavior change).
    Foundation for Sessions 3-4 virtual block tracking.
    """
    target = resolve_vllm_file("v1/core/block_pool.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/block_pool.py — phase5 block pool init "
            "(v1.0 Phase 5 Session 2)"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — phase5 block pool init anchor",
        sub_patches=[
            TextPatch(
                name="pn95_phase5_block_pool_init_anchor",
                anchor=PN95_SITE10_OLD,
                replacement=PN95_SITE10_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "tier_manager",
            "TierManager",
            "cpu_offload_kv_blocks",
        ],
    )


def _make_patcher_phase5_boot_check() -> TextPatcher | None:
    """v1.0 Phase 5 Session 1 — anchor at _check_enough_kv_cache_memory
    for tier-aware boot pre-flight expansion.

    Local-only inflation: caller's available_memory NOT propagated.
    Downstream tensor allocation uses original GPU-only budget.
    Env-gated by GENESIS_PN95_VIRT_ENABLE (default OFF).
    """
    target = resolve_vllm_file("v1/core/kv_cache_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/kv_cache_utils.py — boot check expansion "
            "(v1.0 Phase 5 Session 1)"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — phase5 boot check anchor",
        sub_patches=[
            TextPatch(
                name="pn95_phase5_boot_check_anchor",
                anchor=PN95_SITE9_OLD,
                replacement=PN95_SITE9_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "tier_manager",
            "TierManager",
            "cpu_offload_kv_blocks",
        ],
    )


def _make_patcher_promote_on_miss() -> TextPatcher | None:
    """v1.0 Phase 4: anchor at get_cached_block for promote_on_miss."""
    target = resolve_vllm_file("v1/core/block_pool.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN95 v1/core/block_pool.py — promote_on_miss (v1.0 Phase 4)"
        ),
        target_file=str(target),
        marker=GENESIS_PN95_MARKER + " — promote_on_miss anchor",
        sub_patches=[
            TextPatch(
                name="pn95_promote_on_miss_anchor",
                anchor=PN95_SITE8_OLD,
                replacement=PN95_SITE8_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "tier_manager",
            "TierManager",
            "cpu_offload_kv_blocks",
        ],
    )


def _apply_one(patcher: TextPatcher | None, label: str) -> tuple[str, str]:
    """Apply one of the two PN95 anchors (returns wiring-status tuple)."""
    if patcher is None:
        return "skipped", f"{label}: target file not resolvable"
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"{label}: target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[PN95] %s: marker present — idempotent skip", label)
        return "applied", "idempotent (marker present)"
    # Skip drift markers that are our own (any PN95 patch already applied
    # to the same file leaves these). Real upstream drift would use other
    # text like "tier_manager", "TierManager", "cpu_offload_kv_blocks".
    _own_markers = ("[Genesis PN95", "_g_pn95_")
    for m in patcher.upstream_drift_markers:
        if m in _own_markers and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"{label}: upstream drift marker {m!r} present — "
                "upstream may have absorbed multi-tier cache support",
            )

    result, failure = patcher.apply()
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            f"{label}: PN95 tier-aware anchor injected — "
            f"runtime no-op until GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1 + "
            "init_from_config() called by dispatcher."
        ),
        patch_name=patcher.patch_name,
    )


def apply() -> tuple[str, str]:
    """Apply PN95 tier-aware cache wire-in (both anchors).

    Returns a single (status, reason) tuple summarizing both:
      - "applied" if both succeed (or both idempotent)
      - "skipped" if either is skipped (with reasons concatenated)
      - "failed" if either returns "failed"
    """
    from sndr.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN95")
    log_decision("PN95", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    s1, r1 = _apply_one(_make_patcher_kvcm(), "PN95-admit")
    s2, r2 = _apply_one(_make_patcher_blockpool(), "PN95-touch")
    s3, r3 = _apply_one(_make_patcher_kv_manager_init(), "PN95-mamba-init")
    s4, r4 = _apply_one(_make_patcher_register_kv_caches(),
                          "PN95-register-kv-caches")
    s5, r5 = _apply_one(_make_patcher_scheduler_tick(),
                          "PN95-scheduler-tick")
    # Phase 4 — prefix cache extension to CPU pinned RAM
    s6, r6 = _apply_one(_make_patcher_blockpool_register(),
                          "PN95-blockpool-register")
    s7, r7 = _apply_one(_make_patcher_demote_on_evict(),
                          "PN95-demote-on-evict")
    s8, r8 = _apply_one(_make_patcher_promote_on_miss(),
                          "PN95-promote-on-miss")
    # Phase 5 Session 1 — boot-time KV pool expansion (Anchor #9)
    s9, r9 = _apply_one(_make_patcher_phase5_boot_check(),
                          "PN95-phase5-boot-check")
    # Phase 5 Session 2 — block pool metadata side-table init (Anchor #11)
    # Note: depends on Phase 4 SITE6 (s6) being applied first.
    s10, r10 = _apply_one(_make_patcher_phase5_block_pool_init(),
                          "PN95-phase5-block-pool-init")
    # Phase 5 Session 3 — get_new_blocks materialization (Anchor #12)
    s11, r11 = _apply_one(_make_patcher_phase5_get_new_blocks(),
                          "PN95-phase5-get-new-blocks")
    # Phase 4.2 Anchor #13 (worker-side proactive demote) is RETIRED on
    # dev259+ — PN96 Phase 6 emergency-demote at block_pool.py:385-411
    # superseded it. The builder is no longer invoked here: it only logged a
    # per-boot RETIRED banner and returned None, so calling it added a dead
    # "skipped" to the apply matrix and noise to every operator's boot log.
    # The archived implementation stays in _make_patcher_worker_proactive_demote
    # for one-block re-enable if PN96 ever regresses. deep-audit 2026-06-14 (#4).

    # Worst status wins: failed > skipped > applied
    rank = {"failed": 2, "skipped": 1, "applied": 0}
    overall = max(
        (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11),
        key=lambda s: rank.get(s, 0),
    )
    return overall, (
        f"admit: {r1} | touch: {r2} | mamba-init: {r3} | "
        f"register-kv-caches: {r4} | scheduler-tick: {r5} | "
        f"blockpool-register: {r6} | demote-on-evict: {r7} | "
        f"promote-on-miss: {r8} | phase5-boot-check: {r9} | "
        f"phase5-bp-init: {r10} | phase5-get-new-blocks: {r11}"
    )

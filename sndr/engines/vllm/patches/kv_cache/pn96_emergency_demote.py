# SPDX-License-Identifier: Apache-2.0
"""PN96 — emergency-demote hook (Phase 6 PoC).

Architectural goal: extend `KVCacheManager.get_new_blocks(num_blocks)`
beyond the upstream `raise ValueError("Cannot get N free blocks")` cliff.
When the GPU pool is truly out of free slots, this hook attempts to
synthesise free slots by demoting CACHED blocks from the free queue
(blocks that have ref_cnt=0 but still hold cached-prefix bytes) to the
PN95 L2/L3 tier.

This is the "intercept vllm's allocation path" piece — without it, vllm
gives up and the worker dies. With it, the engine recovers enough slots
to admit the next allocation request, at the cost of having to re-promote
those cached blocks later (handled by existing promote_on_miss path).

LIMITATIONS — documented honestly, not aspirations
====================================================

This hook ONLY rescues "free-but-cached" blocks (ref_cnt=0, block_hash
not None). It does NOT preempt active sequences (ref_cnt > 0 blocks
held by a running request). For single-user / single-request beyond
the physical GPU pool, the active sequence's OWN blocks must move to
CPU mid-attention — that requires virtual block_table addressing
inside the attention kernel and is OUT OF SCOPE for this hook.

So: PN96 unlocks "multi-prefix-cache reuse" workloads (8+ concurrent
users with shared prefix). It does NOT unlock "single 200K user on
80K pool".

The 156K / 256K single-user-single-card case still needs Phase 7
(virtual block addressing in attention kernel). PN96 is the
prerequisite that proves the allocation-path interception works.

Env gate: `GENESIS_ENABLE_PN96_EMERGENCY_DEMOTE=1` (default OFF).

Anchor surface: `vllm/v1/core/block_pool.py::BlockPool.get_new_blocks`
on the `if num_blocks > self.get_num_free_blocks()` line.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn96_emergency_demote")

GENESIS_MARKER = "Genesis PN96 emergency-demote hook (Phase 6 PoC)"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN96_EMERGENCY_DEMOTE", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor on dev209+ inspected 2026-05-13: get_new_blocks() opens with the
# free-block-count check + raise. We insert an emergency-rescue branch
# BEFORE the raise so the engine recovers instead of crashing.
PN96_OLD = (
    "    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:\n"
    "        \"\"\"Get new blocks from the free block pool.\n"
    "\n"
    "        Note that we do not check block cache in this function.\n"
    "\n"
    "        Args:\n"
    "            num_blocks: The number of blocks to allocate.\n"
    "\n"
    "        Returns:\n"
    "            A list of new block.\n"
    "        \"\"\"\n"
    "        if num_blocks > self.get_num_free_blocks():\n"
    "            raise ValueError(f\"Cannot get {num_blocks} free blocks from the pool\")\n"
)
PN96_NEW = (
    "    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:\n"
    "        \"\"\"Get new blocks from the free block pool.\n"
    "\n"
    "        Note that we do not check block cache in this function.\n"
    "\n"
    "        Args:\n"
    "            num_blocks: The number of blocks to allocate.\n"
    "\n"
    "        Returns:\n"
    "            A list of new block.\n"
    "        \"\"\"\n"
    "        # [Genesis PN96 Phase 6 PoC] emergency demote: when the pool\n"
    "        # is exhausted, walk the free queue for CACHED (ref_cnt=0,\n"
    "        # block_hash != None) entries and capture their bytes to the\n"
    "        # PN95 L2 store before vllm reuses their slots. Without this\n"
    "        # the engine crashes with ValueError; with it the slots are\n"
    "        # recovered, the eviction is reversible (promote_on_miss can\n"
    "        # restore later). Best-effort; on any failure falls through\n"
    "        # to the upstream raise.\n"
    "        try:\n"
    "            _free = self.get_num_free_blocks()\n"
    "            if num_blocks > _free:\n"
    "                from sndr.cache._pn95_runtime import (\n"
    "                    pn96_emergency_rescue as _g_pn96_rescue,\n"
    "                )\n"
    "                _g_pn96_rescue(self, deficit=num_blocks - _free)\n"
    "        except Exception:\n"
    "            pass\n"
    "        if num_blocks > self.get_num_free_blocks():\n"
    "            raise ValueError(f\"Cannot get {num_blocks} free blocks from the pool\")\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file("v1/core/block_pool.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN96 emergency-demote (Phase 6 PoC)",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn96_get_new_blocks_rescue",
                anchor=PN96_OLD,
                replacement=PN96_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN96",
            "pn96_emergency_rescue",
        ],
    )


def apply() -> tuple[str, str]:
    if not _enabled():
        return "skipped", "PN96 disabled (set GENESIS_ENABLE_PN96_EMERGENCY_DEMOTE=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file block_pool.py not resolvable"
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as fh:
        content = fh.read()
    if patcher.marker in content:
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m in content:
            return "skipped", f"drift marker {m!r} already in file"
    result, failure = patcher.apply()
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message="PN96 emergency-demote hook installed",
        patch_name=patcher.patch_name,
    )

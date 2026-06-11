# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN110 — BlockPool.free_blocks deduplication.

Backport of [vllm#42615](https://github.com/vllm-project/vllm/pull/42615)
by `AkCodes23` (OPEN at the time of backport).

================================================================
WHAT THIS PATCH DOES
================================================================

``BlockPool.free_blocks(ordered_blocks)`` currently does::

    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1
    self.free_block_queue.append_n(blocks_list)

If a caller passes the same ``KVCacheBlock`` more than once — which
happens at boundaries between sliding-window attention reuse and
prefix-cache eviction — ``ref_cnt`` decreases twice and the same
block is appended to the free queue twice. The next allocation pops a
block whose ``ref_cnt`` is already negative, which trips the
``fake_free_list_tail`` sentinel assertion (and, in release builds
where the assert is compiled out, leaves a poisoned block in
circulation).

The fix is to deduplicate by object identity (``id(block)``) before
both the ref-count decrement and the queue append, and to emit a
``logger.warning`` when duplicates are seen so the underlying caller
bug stays visible.

================================================================
RELEVANCE FOR GENESIS
================================================================

Our PN95 / PN96 / PN97 stack already wraps ``BlockPool`` (the tier-
aware cache + emergency-demote + physical-cap overlays manipulate
``free_block_queue`` and ``ref_cnt`` near this exact code path). A
caller-side regression in our own overlays could in principle hit the
same double-free as the upstream-reported case — making this a
defensive guard that *composes* with the rest of the family.

Hot reproducer in the upstream PR was sliding-window + a
``SimpleCPUOffloading`` connector; we do not use that connector, but
hybrid GDN attention also uses sliding-window-like block reuse on
27B-INT4, so the latent bug class exists in our stack.

Default ON. Cost is one ``set`` per ``free_blocks`` call (linear in
the input size) — negligible compared to the GPU work the same call
ultimately triggers.

================================================================
SAFETY MODEL
================================================================

- Behaviour-preserving for non-duplicate inputs (the only path
  exercised by clean callers).
- Idempotent via Genesis marker comment.
- Drift-marker watches the canonical upstream form so the patch
  self-skips when the fix lands in our pin.
- Adds zero VRAM, no new allocations beyond the per-call dedup set.

================================================================

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original PR: vllm#42615.
"""
from __future__ import annotations

import logging
import os

from sndr.kernel import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)
from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn110_block_pool_free_dedup")

GENESIS_PN110_MARKER = (
    "Genesis PN110 BlockPool.free_blocks deduplication (vllm#42615) v11.0.0"
)


# Anchor on the 3-line shape that is currently the FIRST thing
# `free_blocks` does after its docstring. Whitespace = 8-space indent
# (method body inside a class).
PN110_OLD = (
    "        # Materialize the iterable to allow multiple passes.\n"
    "        blocks_list = list(ordered_blocks)\n"
    "        for block in blocks_list:\n"
    "            block.ref_cnt -= 1\n"
)

PN110_NEW = (
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN110 vllm#42615 backport] Deduplicate by object\n"
    "        # identity so a caller that passes the same KVCacheBlock\n"
    "        # twice (sliding-window reuse + offload connector race;\n"
    "        # see upstream PR for repro) does not double-decrement\n"
    "        # ref_cnt or double-append into the free queue.\n"
    "        # Warns on duplicates so the upstream caller bug stays\n"
    "        # visible.\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        seen: set[int] = set()\n"
    "        blocks_list: list = []\n"
    "        num_duplicates = 0\n"
    "        for block in ordered_blocks:\n"
    "            block_obj_id = id(block)\n"
    "            if block_obj_id not in seen:\n"
    "                seen.add(block_obj_id)\n"
    "                blocks_list.append(block)\n"
    "            else:\n"
    "                num_duplicates += 1\n"
    "        if num_duplicates > 0:\n"
    "            logger.warning(\n"
    "                \"free_blocks() received %d duplicate block(s) \"\n"
    "                \"(total=%d, unique=%d). Caller-side bug — same \"\n"
    "                \"KVCacheBlock appeared multiple times in input.\",\n"
    "                num_duplicates,\n"
    "                len(blocks_list) + num_duplicates,\n"
    "                len(blocks_list),\n"
    "            )\n"
    "        for block in blocks_list:\n"
    "            block.ref_cnt -= 1\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/core/block_pool.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN110 v1/core/block_pool.py — free_blocks dedup (vllm#42615)",
        target_file=str(target),
        marker=GENESIS_PN110_MARKER,
        sub_patches=[
            TextPatch(
                name="pn110_free_blocks_dedup",
                anchor=PN110_OLD,
                replacement=PN110_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN110",
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "free_blocks() received" is the vllm#42615 warning fragment
            # baked verbatim by our own backport replacement — it cannot
            # distinguish a real upstream merge from our residue (false
            # "upstream_merged" skip, PN369 class). Real-merge detection
            # via required-anchor mismatch (Layer 5) + preflight deep-diff.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN110 — BlockPool.free_blocks deduplication."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN110")
    log_decision("PN110", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/core/block_pool.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} present — upstream "
                "PR #42615 (or equivalent fix) appears merged",
            )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return (
        "applied",
        "PN110 applied: BlockPool.free_blocks deduplicates by id() "
        "before ref_cnt -= and queue append. Closes the double-free "
        "class that trips fake_free_list_tail. Composes with PN95/96/97."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except OSError:
        return False

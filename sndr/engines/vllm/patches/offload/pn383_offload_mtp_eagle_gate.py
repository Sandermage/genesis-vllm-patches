# SPDX-License-Identifier: Apache-2.0
"""PN383 — KV-offload + MTP cuMemcpyBatchAsync segfault gate (vendor of
OPEN PR vllm#44784) plus two Genesis extensions.

Upstream #44784 (issue #44780): ``OffloadingConnectorScheduler`` schedules
EAGLE/MTP draft-attention groups into the store/load paths. The draft
group's trailing block is rewritten by the drafter every decode step (no
stable hash) and uses tiny ``gpu_block_size`` values, so the block math
``num_gpu_blocks = cdiv(420K, small_size)`` produces an out-of-bounds GPU
block index that segfaults silently inside ``cuMemcpyBatchAsync``. This
blocks native CPU KV offload on EVERY MTP config, ours included (Qwen3.6
MTP K=3). The PR's four ``scheduler.py`` hunks:

  1. ``is_eagle_group`` field on ``GroupOffloadConfig``.
  2. eagle-group detection in ``SchedulerOffloadConfig.from_spec``.
  3. eagle handling inside ``_lookup`` (query one extra block + pop of the
     volatile trailing block, gated on a per-iteration ``_pn383_eagle_verified``
     set that a tightening non-eagle group clears).
  4. trailing-block exclusion in ``_build_store_jobs``.

Plus the PR's defense-in-depth ``gpu_worker.py`` pre-DMA bounds check.

Genesis extensions over upstream (roadmap chunk-4 Theme 5 — verified
against the pristine pin 0.22.1rc1.dev259+g303916e93):

  1. Qwen3.6-specific ``is_eagle_group`` flagging. On the pin
     ``KVCacheGroupSpec.is_eagle_group`` is only ever set by the DeepSeek V4
     annotation path; for Qwen3.5/3.6 MTP it stays False on every group, so
     upstream's fallback flags ALL groups as eagle — every group then loses
     its trailing block from store/load, costing prefix-reuse hit-rate
     across the whole target model. PN383 narrows the fallback: when nothing
     is annotated, first flag only groups whose layer names live under the
     drafter ``mtp`` module prefix (the Qwen3.5/3.6 MTP drafter convention).
     The all-groups fallback stays as a LOUD last resort (warning logged).

  2. Re-add the pre-DMA bounds check the upstream PR described for
     ``gpu_worker.py`` but dropped from its final diff: validate the GPU
     block ids against the per-tensor row counts BEFORE the descriptor
     pointers are computed, raising a clear ``RuntimeError`` instead of a
     silent CUDA segfault.

Dormant by design: ``default_on=False``, opt-in via
``GENESIS_ENABLE_PN383_OFFLOAD_MTP_EAGLE_GATE``. The patched behavior is
only reachable when a KV offloading backend is configured (none on today's
PROD). Self-skips when #44784 lands upstream via the drift markers below.

Author backport: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#44784 (OPEN as of 2026-06-13).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pn383_offload_mtp_eagle_gate")

# Idempotency marker. MUST contain "44784" (test_marker_tracks_upstream_pr).
GENESIS_PN383_MARKER = (
    "Genesis PN383 KV-offload + MTP eagle gate "
    "(vendor of vllm#44784) + Qwen3.6 narrowing + pre-DMA bounds check v1"
)

_ENV_FLAG = "GENESIS_ENABLE_PN383_OFFLOAD_MTP_EAGLE_GATE"

# Target files under the installed vllm tree.
_SCHEDULER_REL = (
    "distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
)
_WORKER_REL = "v1/kv_offload/cpu/gpu_worker.py"


def _enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# ─────────────────────────────────────────────────────────────────────
# Upstream drift markers.
#
# Each is an EXACT structural substring of vllm#44784's own diff form
# (from `gh pr diff 44784`, 2026-06-13). If any appears in the target
# file the patch self-skips as upstream-merged. Per the PN369
# self-collision rule (tools/lint_drift_markers.py) these must NOT be
# substrings of our own emitted replacement text — which is why PN383
# renames the upstream locals (`eagle_verified` -> `_pn383_eagle_verified`,
# `query_max` -> `_pn383_query_max`, `required_window` ->
# `_pn383_required_window`) and writes original comment wording. A
# `[Genesis`-prefixed marker is exempt from the lint (defended convention).
# ─────────────────────────────────────────────────────────────────────
_SCHEDULER_DRIFT_MARKERS = (
    # The PR's exact field declaration (bare, no trailing comment — our
    # field carries a `# PN383 ...` tail so this never matches our text).
    "    is_eagle_group: bool = False\n",
    # The PR's exact all-groups fallback comprehension head (upstream
    # spells the loop variable `g` and uses `set(range(...))`).
    "        eagle_groups = {\n",
    # The PR's exact eagle-verified set declaration (upstream local name).
    "        eagle_verified: set[int] = set()\n",
    # The PR's exact query-inflation local (upstream local name).
    "                query_max = max_hit_size_tokens\n",
    # The PR's exact required-window local (upstream local name).
    "                    required_window = sliding_window_size_in_blocks\n",
)
_WORKER_DRIFT_MARKERS = (
    # A `[Genesis`-prefixed marker is exempt from the self-collision lint
    # (defended convention) — kept here only to assert the worker hunk's
    # own marker never leaks into the pristine tree.
    "[Genesis PN383 pre-DMA bounds check]",
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 1 — GroupOffloadConfig.is_eagle_group field.
#
# Anchor: the last comment line of GroupOffloadConfig immediately followed
# by its trailing `alignment_block_count` field. We INSERT the new field
# BETWEEN them so the replacement never re-emits the anchor verbatim (the
# anchor-resurrection invariant — a replacement may not contain any
# sub-patch's anchor, including its own). NamedTuple field ordering stays
# valid (both fields are defaulted). The field carries a trailing
# `# PN383 ...` comment so it never matches the bare-line drift marker.
# ─────────────────────────────────────────────────────────────────────
_GROUP_CONFIG_OLD = (
    "    # None for full-attention groups or when the optimization doesn't apply.\n"
    "    alignment_block_count: int | None = None\n"
)
_GROUP_CONFIG_NEW = (
    "    # None for full-attention groups or when the optimization doesn't apply.\n"
    "    # [Genesis PN383 vendor of vllm#44784] True for EAGLE/MTP draft-model\n"
    "    # attention groups. Their trailing block is rewritten by the drafter\n"
    "    # every decode step (no stable hash and a tiny gpu_block_size), so it\n"
    "    # must be excluded from store and load scheduling to avoid the\n"
    "    # out-of-bounds GPU block index that segfaults cuMemcpyBatchAsync.\n"
    "    is_eagle_group: bool = False  # PN383 KV-offload eagle gate\n"
    "    alignment_block_count: int | None = None\n"
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 2a — eagle-group detection in from_spec.
#
# Anchor: the `alignment_tokens` block, the last statement before the
# `return cls(...)` in from_spec. We splice the detection right after it
# so `eagle_groups` (a local set of group indices) is in scope when the
# construction comprehension references it.
#
# Genesis narrowing order (test_detection_narrows_before_all_groups_fallback):
#   1. honor engine annotation (`g.is_eagle_group`, the pin DeepSeek path);
#   2. only if spec-decode is active AND nothing is annotated, narrow to
#      groups whose layer names live under the `mtp` drafter prefix
#      (Qwen3.5/3.6 convention) — the `"mtp"` test MUST precede the
#      `set(range(...))` all-groups fallback or the extension is dead code;
#   3. all-groups fallback as a LOUD last resort.
# ─────────────────────────────────────────────────────────────────────
_FROM_SPEC_DETECTION_OLD = (
    "            alignment_tokens = full_attn_offloaded_block_sizes.pop()\n"
    "\n"
    "        def _alignment_block_count(\n"
)
_FROM_SPEC_DETECTION_NEW = (
    "            alignment_tokens = full_attn_offloaded_block_sizes.pop()\n"
    "\n"
    "        # [Genesis PN383 vendor of vllm#44784 + Qwen3.6 narrowing]\n"
    "        # Determine which KV-cache groups are EAGLE/MTP draft groups so\n"
    "        # their volatile trailing block is excluded from store/load.\n"
    "        _pn383_kv_groups = spec.kv_cache_config.kv_cache_groups\n"
    "        # (1) Honor groups already annotated by the engine (the DeepSeek\n"
    "        #     V4 path on this pin sets is_eagle_group directly).\n"
    "        _pn383_eagle_groups: set[int] = {\n"
    "            _idx\n"
    "            for _idx, _g in enumerate(_pn383_kv_groups)\n"
    "            if getattr(_g, \"is_eagle_group\", False)\n"
    "        }\n"
    "        _pn383_use_eagle = (\n"
    "            spec.vllm_config.speculative_config is not None\n"
    "            and spec.vllm_config.speculative_config.use_eagle()\n"
    "        )\n"
    "        if _pn383_use_eagle and not _pn383_eagle_groups:\n"
    "            # (2) Genesis narrowing: flag only the drafter group(s) whose\n"
    "            #     layer names live under the \"mtp\" module prefix (the\n"
    "            #     Qwen3.5/3.6 MTP drafter convention). This runs BEFORE\n"
    "            #     the all-groups fallback so the narrowing is not dead\n"
    "            #     code: on Qwen the engine never annotates is_eagle_group,\n"
    "            #     and the all-groups fallback would otherwise strip the\n"
    "            #     trailing block of EVERY group (whole-model hit-rate loss).\n"
    "            _pn383_mtp_groups: set[int] = set()\n"
    "            for _idx, _g in enumerate(_pn383_kv_groups):\n"
    "                _names = getattr(_g, \"layer_names\", None) or ()\n"
    "                for _name in _names:\n"
    "                    # Match the \"mtp\" module prefix as a dotted segment\n"
    "                    # (\"mtp.\" / \".mtp.\") so a layer merely containing the\n"
    "                    # substring \"mtp\" elsewhere is not misflagged.\n"
    "                    if _name == \"mtp\" or _name.startswith(\"mtp.\") or (\n"
    "                        \".mtp.\" in _name\n"
    "                    ):\n"
    "                        _pn383_mtp_groups.add(_idx)\n"
    "                        break\n"
    "            if _pn383_mtp_groups:\n"
    "                _pn383_eagle_groups = _pn383_mtp_groups\n"
    "            else:\n"
    "                # (3) Loud last resort: no annotation and no drafter-name\n"
    "                #     hit, but spec-decode is active. Fall back to the\n"
    "                #     upstream conservative all-groups behavior so the\n"
    "                #     segfault stays gated, and warn so the hit-rate cost\n"
    "                #     is visible in the logs.\n"
    "                _pn383_eagle_groups = set(\n"
    "                range(len(_pn383_kv_groups))\n"
    "                )\n"
    "                logger.warning(\n"
    "                    \"KV offloading: spec-decode active but no EAGLE/MTP \"\n"
    "                    \"group could be identified by engine annotation or \"\n"
    "                    \"the 'mtp' drafter-name heuristic. Falling back to \"\n"
    "                    \"excluding the trailing block of ALL %d groups from \"\n"
    "                    \"offloading (prefix-cache hit-rate loss). File a bug \"\n"
    "                    \"if this fires on a known MTP architecture.\",\n"
    "                    len(_pn383_kv_groups),\n"
    "                )\n"
    "        if _pn383_eagle_groups:\n"
    "            logger.info(\n"
    "                \"KV offloading: EAGLE/MTP draft attention groups %s \"\n"
    "                \"detected; their trailing block is excluded from \"\n"
    "                \"offloading due to volatility.\",\n"
    "                sorted(_pn383_eagle_groups),\n"
    "            )\n"
    "\n"
    "        def _alignment_block_count(\n"
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 2b — pass the eagle flag into each GroupOffloadConfig.
#
# Anchor: the last keyword argument of the GroupOffloadConfig(...)
# construction, spanning into the closing `)` of the constructor. We
# INSERT the new kwarg BEFORE that closing paren so the replacement does
# not re-emit the anchor verbatim (anchor-resurrection invariant).
# ─────────────────────────────────────────────────────────────────────
_FROM_SPEC_FLAG_OLD = (
    "                    alignment_block_count=_alignment_block_count(\n"
    "                        gpu_block_size * spec.block_size_factor, sw\n"
    "                    ),\n"
    "                )\n"
)
_FROM_SPEC_FLAG_NEW = (
    "                    alignment_block_count=_alignment_block_count(\n"
    "                        gpu_block_size * spec.block_size_factor, sw\n"
    "                    ),\n"
    "                    # [Genesis PN383 vendor of vllm#44784] propagate the\n"
    "                    # per-group eagle flag computed above.\n"
    "                    is_eagle_group=idx in _pn383_eagle_groups,\n"
    "                )\n"
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 3a — declare the per-iteration eagle-verified set.
#
# Anchor: the `lookup_groups = self._lookup_groups` init spanning into the
# `while lookup_groups:` loop head. We INSERT the verified-set declaration
# BETWEEN them so the replacement does not re-emit the anchor verbatim. The
# set tracks which eagle groups have already popped their volatile trailing
# block this convergence pass; a tightening non-eagle group clears it.
# ─────────────────────────────────────────────────────────────────────
_LOOKUP_VERIFIED_SET_OLD = (
    "        lookup_groups = self._lookup_groups\n"
    "        while lookup_groups:\n"
)
_LOOKUP_VERIFIED_SET_NEW = (
    "        lookup_groups = self._lookup_groups\n"
    "        # [Genesis PN383 vendor of vllm#44784] Tracks which eagle groups\n"
    "        # have already popped their volatile trailing block in the current\n"
    "        # convergence iteration. Reset when a non-eagle group tightens the\n"
    "        # hit boundary, requiring a fresh pop on the next pass.\n"
    "        _pn383_eagle_verified: set[int] = set()\n"
    "        while lookup_groups:\n"
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 3b — compute the per-group is_eagle_unverified flag.
#
# Anchor: the `assert len(offload_keys) >= ...` guard spanning into the
# `# Constrain to block-aligned boundary` comment that follows. We INSERT
# the flag computation BETWEEN them so the replacement does not re-emit the
# anchor verbatim (offload_keys and group_config are both in scope here).
# ─────────────────────────────────────────────────────────────────────
_LOOKUP_UNVERIFIED_OLD = (
    "                assert (\n"
    "                    len(offload_keys)\n"
    "                    >= req_status.req.num_tokens // offloaded_block_size\n"
    "                )\n"
    "\n"
    "                # Constrain to block-aligned boundary for this group\n"
)
_LOOKUP_UNVERIFIED_NEW = (
    "                assert (\n"
    "                    len(offload_keys)\n"
    "                    >= req_status.req.num_tokens // offloaded_block_size\n"
    "                )\n"
    "                # [Genesis PN383 vendor of vllm#44784] An eagle group still\n"
    "                # needs to pop its volatile trailing block unless a prior\n"
    "                # pass in this convergence loop already did so.\n"
    "                _pn383_is_eagle_unverified = (\n"
    "                    group_config.is_eagle_group\n"
    "                    and group_idx not in _pn383_eagle_verified\n"
    "                )\n"
    "\n"
    "                # Constrain to block-aligned boundary for this group\n"
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 3c — query one extra block for eagle sliding-window groups.
#
# Anchor: the `num_blocks = min(cdiv(max_hit_size_tokens, ...))` block
# together with the `sliding_window_size_in_blocks = (...)` assignment that
# immediately follows it. Upstream MOVES that assignment UP (so it is in
# scope before the inflation) and computes an inflated `query_max` used in
# the cdiv. PN383 renames the upstream local `query_max` ->
# `_pn383_query_max` (self-collision). Only sliding-window groups inflate;
# full-attention prefix lookup is unaffected.
# ─────────────────────────────────────────────────────────────────────
_LOOKUP_QUERY_MAX_OLD = (
    "                num_blocks = min(\n"
    "                    cdiv(max_hit_size_tokens, offloaded_block_size), len(offload_keys)\n"
    "                )\n"
    "                start_block_idx = num_computed_tokens // offloaded_block_size\n"
    "                offload_keys = offload_keys[start_block_idx:num_blocks]\n"
    "                sliding_window_size_in_blocks = (\n"
    "                    group_config.sliding_window_size_in_blocks\n"
    "                )\n"
)
_LOOKUP_QUERY_MAX_NEW = (
    "                sliding_window_size_in_blocks = (\n"
    "                    group_config.sliding_window_size_in_blocks\n"
    "                )\n"
    "                # [Genesis PN383 vendor of vllm#44784] For eagle groups,\n"
    "                # query one extra block that will later be popped. We only\n"
    "                # need to widen the query for sliding-window groups; full-\n"
    "                # attention prefix lookup is left untouched.\n"
    "                _pn383_query_max = max_hit_size_tokens\n"
    "                if (\n"
    "                    _pn383_is_eagle_unverified\n"
    "                    and sliding_window_size_in_blocks is not None\n"
    "                ):\n"
    "                    _pn383_query_max = min(\n"
    "                        max_hit_size_tokens + offloaded_block_size,\n"
    "                        len(offload_keys) * offloaded_block_size,\n"
    "                    )\n"
    "                num_blocks = min(\n"
    "                    cdiv(_pn383_query_max, offloaded_block_size),\n"
    "                    len(offload_keys),\n"
    "                )\n"
    "                start_block_idx = num_computed_tokens // offloaded_block_size\n"
    "                offload_keys = offload_keys[start_block_idx:num_blocks]\n"
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 3d — widen the sliding-window lookup by one block.
#
# Anchor: the `self._sliding_window_lookup(...)` call. For an unverified
# eagle group we pass `sliding_window_size_in_blocks + 1` so the extra
# trailing block is included in the confirmed-hit window before it is
# popped. PN383 renames the upstream local `required_window` ->
# `_pn383_required_window`.
# ─────────────────────────────────────────────────────────────────────
_LOOKUP_REQUIRED_WINDOW_OLD = (
    "                    num_hit_blocks = self._sliding_window_lookup(\n"
    "                        offload_keys,\n"
    "                        sliding_window_size_in_blocks,\n"
    "                        req_status.req_context,\n"
    "                    )\n"
)
_LOOKUP_REQUIRED_WINDOW_NEW = (
    "                    # [Genesis PN383 vendor of vllm#44784] widen the\n"
    "                    # confirmed-hit window by one block for an unverified\n"
    "                    # eagle group so the extra trailing block is captured\n"
    "                    # before the pop below.\n"
    "                    _pn383_required_window = sliding_window_size_in_blocks\n"
    "                    if _pn383_is_eagle_unverified:\n"
    "                        _pn383_required_window += 1\n"
    "                    num_hit_blocks = self._sliding_window_lookup(\n"
    "                        offload_keys,\n"
    "                        _pn383_required_window,\n"
    "                        req_status.req_context,\n"
    "                    )\n"
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 3e — pop the volatile trailing block + mark verified.
#
# Anchor: the `else:` arm that tightens max_hit_size_tokens after a
# confirmed (non-deferred) hit count. We decrement the hit count by one
# (dropping the volatile trailing block) for an unverified eagle group and
# record the group as verified so the pop is not repeated this pass.
# ─────────────────────────────────────────────────────────────────────
_LOOKUP_POP_OLD = (
    "                if num_hit_blocks is None:\n"
    "                    defer_lookup = True\n"
    "                else:\n"
    "                    max_hit_size_tokens = min(\n"
    "                        max_hit_size_tokens,\n"
    "                        offloaded_block_size * (start_block_idx + num_hit_blocks),\n"
    "                    )\n"
)
_LOOKUP_POP_NEW = (
    "                if num_hit_blocks is None:\n"
    "                    defer_lookup = True\n"
    "                else:\n"
    "                    # [Genesis PN383 vendor of vllm#44784] drop the volatile\n"
    "                    # trailing block from the confirmed hit count for an\n"
    "                    # unverified eagle group, then mark it verified so the\n"
    "                    # pop is not repeated in this convergence iteration.\n"
    "                    if _pn383_is_eagle_unverified:\n"
    "                        num_hit_blocks -= 1\n"
    "                        _pn383_eagle_verified.add(group_idx)\n"
    "                    max_hit_size_tokens = min(\n"
    "                        max_hit_size_tokens,\n"
    "                        offloaded_block_size * (start_block_idx + num_hit_blocks),\n"
    "                    )\n"
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 3f — clear eagle-verified when a non-eagle group tightens.
#
# Anchor: the `if new_num_hit_tokens < num_hit_tokens:` convergence branch.
# When a NON-eagle group tightens the hit boundary, an eagle group's
# earlier pop may no longer be valid, so we clear the verified set to force
# a fresh pop on the re-run.
# ─────────────────────────────────────────────────────────────────────
_LOOKUP_CLEAR_OLD = (
    "                if new_num_hit_tokens < num_hit_tokens:\n"
    "                    if defer_lookup:\n"
)
_LOOKUP_CLEAR_NEW = (
    "                if new_num_hit_tokens < num_hit_tokens:\n"
    "                    # [Genesis PN383 vendor of vllm#44784] a tightening\n"
    "                    # NON-eagle group invalidates any earlier eagle pop, so\n"
    "                    # clear the verified set to re-pop on the next pass.\n"
    "                    if not group_config.is_eagle_group:\n"
    "                        _pn383_eagle_verified.clear()\n"
    "                    if defer_lookup:\n"
)


# ─────────────────────────────────────────────────────────────────────
# Scheduler hunk 4 — exclude the trailing block from store jobs.
#
# Anchor: the per-group block count + start-index pair at the top of the
# _build_store_jobs filter loop. Extended through the
# `if num_blocks <= start_block_idx: continue` lines so the anchor is
# unique (the same `num_blocks = num_offloadable_tokens // ...` /
# `start_block_idx = ...` pair also appears in a sibling method that does
# NOT have this exact continuation). For an eagle group we drop one block,
# clamped at zero so a single-block group stores nothing.
# ─────────────────────────────────────────────────────────────────────
_STORE_TAIL_OLD = (
    "                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size\n"
    "                start_block_idx = group_state.next_stored_block_idx\n"
    "                if num_blocks <= start_block_idx:\n"
    "                    continue\n"
)
_STORE_TAIL_NEW = (
    "                num_blocks = num_offloadable_tokens // group_config.offloaded_block_size\n"
    "                # [Genesis PN383 vendor of vllm#44784] never store the\n"
    "                # volatile trailing block of an eagle group; clamp at zero\n"
    "                # so a single-block eagle group stores nothing.\n"
    "                if group_config.is_eagle_group:\n"
    "                    num_blocks = max(0, num_blocks - 1)\n"
    "                start_block_idx = group_state.next_stored_block_idx\n"
    "                if num_blocks <= start_block_idx:\n"
    "                    continue\n"
)


# ─────────────────────────────────────────────────────────────────────
# Worker hunk — pre-DMA bounds check.
#
# Anchor: the group-slice extraction immediately followed by the
# `for data_ref in group_data_refs:` descriptor-pointer loop. The check
# MUST come BEFORE that loop (test_worker_bounds_check_raises_before_dma):
# we validate that every block id in this group is within the per-tensor
# row count (`.shape[0]`) before `compute_sub_block_ptrs` turns them into
# raw device pointers. A clear RuntimeError beats a silent CUDA segfault.
# ─────────────────────────────────────────────────────────────────────
_WORKER_BOUNDS_OLD = (
    "            group_src = src_blocks[src_offset:src_end_offset]\n"
    "            group_dst = dst_blocks[dst_offset:dst_end_offset]\n"
    "\n"
    "            for data_ref in group_data_refs:\n"
)
_WORKER_BOUNDS_NEW = (
    "            group_src = src_blocks[src_offset:src_end_offset]\n"
    "            group_dst = dst_blocks[dst_offset:dst_end_offset]\n"
    "\n"
    "            # [Genesis PN383 pre-DMA bounds check] Validate the GPU block\n"
    "            # ids against the per-tensor row counts BEFORE the descriptor\n"
    "            # pointers are computed. A mis-sized eagle group can otherwise\n"
    "            # produce an out-of-bounds block index that segfaults silently\n"
    "            # inside cuMemcpyBatchAsync; raise a clear RuntimeError instead.\n"
    "            # NOTE: this guard loop deliberately uses a distinct loop\n"
    "            # variable (_pn383_ref) so the descriptor-pointer loop below\n"
    "            # remains the sole iterator over the group's data refs.\n"
    "            for _pn383_ref in group_data_refs:\n"
    "                _pn383_t_idx = _pn383_ref.tensor_idx\n"
    "                _pn383_src_rows = self.src_tensors[_pn383_t_idx].shape[0]\n"
    "                _pn383_dst_rows = self.dst_tensors[_pn383_t_idx].shape[0]\n"
    "                if len(group_src) and int(group_src.max()) >= _pn383_src_rows:\n"
    "                    raise RuntimeError(\n"
    "                        \"PN383 pre-DMA bounds check: src block id \"\n"
    "                        f\"{int(group_src.max())} >= src tensor rows \"\n"
    "                        f\"{_pn383_src_rows} (tensor_idx={_pn383_t_idx}). \"\n"
    "                        \"Likely a mis-sized EAGLE/MTP group leaking into \"\n"
    "                        \"KV offload (see vllm#44784); refusing the DMA.\"\n"
    "                    )\n"
    "                if len(group_dst) and int(group_dst.max()) >= _pn383_dst_rows:\n"
    "                    raise RuntimeError(\n"
    "                        \"PN383 pre-DMA bounds check: dst block id \"\n"
    "                        f\"{int(group_dst.max())} >= dst tensor rows \"\n"
    "                        f\"{_pn383_dst_rows} (tensor_idx={_pn383_t_idx}). \"\n"
    "                        \"Likely a mis-sized EAGLE/MTP group leaking into \"\n"
    "                        \"KV offload (see vllm#44784); refusing the DMA.\"\n"
    "                    )\n"
    "\n"
    "            for data_ref in group_data_refs:\n"
)


# ─────────────────────────────────────────────────────────────────────
# Sub-patch inventories.
# ─────────────────────────────────────────────────────────────────────


def build_scheduler_sub_patches() -> list[TextPatch]:
    """Return the ten scheduler text anchors of vllm#44784 (+ Genesis
    narrowing). Ordered as upstream applies them: config field, detection,
    construction flag, then the six _lookup edits, then the store tail
    exclusion."""
    return [
        TextPatch(
            name="pn383_group_config_eagle_field",
            anchor=_GROUP_CONFIG_OLD,
            replacement=_GROUP_CONFIG_NEW,
            required=True,
        ),
        TextPatch(
            name="pn383_from_spec_eagle_detection",
            anchor=_FROM_SPEC_DETECTION_OLD,
            replacement=_FROM_SPEC_DETECTION_NEW,
            required=True,
        ),
        TextPatch(
            name="pn383_from_spec_group_flag",
            anchor=_FROM_SPEC_FLAG_OLD,
            replacement=_FROM_SPEC_FLAG_NEW,
            required=True,
        ),
        TextPatch(
            name="pn383_lookup_eagle_verified_set",
            anchor=_LOOKUP_VERIFIED_SET_OLD,
            replacement=_LOOKUP_VERIFIED_SET_NEW,
            required=True,
        ),
        TextPatch(
            name="pn383_lookup_eagle_unverified",
            anchor=_LOOKUP_UNVERIFIED_OLD,
            replacement=_LOOKUP_UNVERIFIED_NEW,
            required=True,
        ),
        TextPatch(
            name="pn383_lookup_query_max",
            anchor=_LOOKUP_QUERY_MAX_OLD,
            replacement=_LOOKUP_QUERY_MAX_NEW,
            required=True,
        ),
        TextPatch(
            name="pn383_lookup_required_window",
            anchor=_LOOKUP_REQUIRED_WINDOW_OLD,
            replacement=_LOOKUP_REQUIRED_WINDOW_NEW,
            required=True,
        ),
        TextPatch(
            name="pn383_lookup_eagle_pop",
            anchor=_LOOKUP_POP_OLD,
            replacement=_LOOKUP_POP_NEW,
            required=True,
        ),
        TextPatch(
            name="pn383_lookup_eagle_clear",
            anchor=_LOOKUP_CLEAR_OLD,
            replacement=_LOOKUP_CLEAR_NEW,
            required=True,
        ),
        TextPatch(
            name="pn383_store_tail_exclusion",
            anchor=_STORE_TAIL_OLD,
            replacement=_STORE_TAIL_NEW,
            required=True,
        ),
    ]


def build_worker_sub_patches() -> list[TextPatch]:
    """Return the single gpu_worker.py text anchor — the pre-DMA bounds
    check (the defense-in-depth half of vllm#44784, re-added by Genesis)."""
    return [
        TextPatch(
            name="pn383_pre_dma_bounds_check",
            anchor=_WORKER_BOUNDS_OLD,
            replacement=_WORKER_BOUNDS_NEW,
            required=True,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# Patcher builders (injectable target_file for the tests).
# ─────────────────────────────────────────────────────────────────────


def _make_scheduler_patcher(target_file: str | None = None) -> TextPatcher | None:
    """Build the scheduler TextPatcher. ``target_file`` is injectable for
    tests; default resolves through ``resolve_vllm_file`` (looked up via the
    module attribute so test monkeypatches are honored)."""
    if target_file is None:
        resolved = resolve_vllm_file(_SCHEDULER_REL)
        if resolved is None:
            return None
        target_file = str(resolved)
    return TextPatcher(
        patch_name=(
            "PN383 offloading/scheduler.py — EAGLE/MTP group gate "
            "(vendor of vllm#44784 + Qwen3.6 narrowing)"
        ),
        target_file=target_file,
        marker=GENESIS_PN383_MARKER,
        sub_patches=build_scheduler_sub_patches(),
        upstream_drift_markers=list(_SCHEDULER_DRIFT_MARKERS),
    )


def _make_worker_patcher(target_file: str | None = None) -> TextPatcher | None:
    """Build the gpu_worker.py TextPatcher. ``target_file`` is injectable for
    tests; default resolves through ``resolve_vllm_file``."""
    if target_file is None:
        resolved = resolve_vllm_file(_WORKER_REL)
        if resolved is None:
            return None
        target_file = str(resolved)
    return TextPatcher(
        patch_name=(
            "PN383 v1/kv_offload/cpu/gpu_worker.py — pre-DMA bounds check "
            "(vendor of vllm#44784 defense-in-depth)"
        ),
        target_file=target_file,
        marker=GENESIS_PN383_MARKER,
        sub_patches=build_worker_sub_patches(),
        upstream_drift_markers=list(_WORKER_DRIFT_MARKERS),
    )


# ─────────────────────────────────────────────────────────────────────
# Module apply() contract.
# ─────────────────────────────────────────────────────────────────────


def apply() -> tuple[str, str]:
    """Apply PN383 — KV-offload + MTP eagle gate. Never raises.

    Opt-in: gated on ``GENESIS_ENABLE_PN383_OFFLOAD_MTP_EAGLE_GATE``
    (default_on=False in the registry — dormant until KV offload is
    enabled). Patches BOTH the scheduler and the worker file; a missing
    target is a clean ``skipped`` (not a failure)."""
    if not _enabled():
        return (
            "skipped",
            f"PN383 disabled (set {_ENV_FLAG}=1 to enable)",
        )
    if vllm_install_root() is None:
        return "skipped", "PN383: vllm install root not discoverable"

    sched_patcher = _make_scheduler_patcher()
    worker_patcher = _make_worker_patcher()
    if sched_patcher is None or worker_patcher is None:
        return (
            "skipped",
            "PN383: target file(s) not resolvable "
            f"(scheduler={_SCHEDULER_REL}, worker={_WORKER_REL})",
        )

    statuses: list[tuple[str, str]] = []
    for patcher, applied_msg in (
        (
            sched_patcher,
            "PN383 scheduler patch applied — EAGLE/MTP draft groups are "
            "now gated out of KV-offload store/load scheduling, with the "
            "Qwen3.6 'mtp'-prefix narrowing in front of the all-groups "
            "fallback (vendor of vllm#44784).",
        ),
        (
            worker_patcher,
            "PN383 worker patch applied — gpu_worker.py validates GPU "
            "block ids against the per-tensor row counts before the DMA, "
            "raising RuntimeError instead of segfaulting in "
            "cuMemcpyBatchAsync (vllm#44784 defense-in-depth).",
        ),
    ):
        result, failure = patcher.apply()
        statuses.append(
            result_to_wiring_status(
                result,
                failure,
                applied_message=applied_msg,
                patch_name=patcher.patch_name,
            )
        )

    # Fail if any sub-patcher hard-failed; otherwise "applied" if at least
    # one file changed, else "skipped" (idempotent / drift on both).
    if any(s == "failed" for s, _ in statuses):
        detail = " | ".join(d for _, d in statuses)
        return "failed", f"PN383 (vllm#44784): {detail}"
    if any(s == "applied" for s, _ in statuses):
        detail = " | ".join(d for _, d in statuses)
        return "applied", f"PN383 (vllm#44784): {detail}"
    detail = " | ".join(d for _, d in statuses)
    return "skipped", f"PN383 (vllm#44784): {detail}"


def is_applied() -> bool:
    """Return True iff the marker is present in BOTH target files."""
    sched_patcher = _make_scheduler_patcher()
    worker_patcher = _make_worker_patcher()
    if sched_patcher is None or worker_patcher is None:
        return False
    for patcher in (sched_patcher, worker_patcher):
        try:
            with open(patcher.target_file, encoding="utf-8") as f:
                if patcher.marker not in f.read():
                    return False
        except (OSError, UnicodeDecodeError):
            return False
    return True

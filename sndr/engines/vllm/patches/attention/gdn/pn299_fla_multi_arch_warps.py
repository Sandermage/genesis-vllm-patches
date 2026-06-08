# SPDX-License-Identifier: Apache-2.0
"""Patch PN299 — FLA multi-file arch-aware NUM_WARPS prune (3 hot kernels).

Genesis-original 2026-06-05 — extension of PN298 pattern to additional
FLA ops used by GDN path. Targets:

  1. `model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py` — KKT path
     `num_warps in [2, 4, 8]` × `num_stages in [2, 3, 4]`
  2. `model_executor/layers/fla/ops/wy_fast.py` — recompute_w_u_fwd
     `num_warps in [2, 4, 8]` × `num_stages in [2, 3, 4]`
  3. `model_executor/layers/fla/ops/l2norm.py` — TWO autotune sites
     • kernel1: `num_warps in [1, 2, 4, 8, 16, 32]` (!! up to 32)
     • kernel:  `num_warps in [1, 2, 4, 8, 16]`

All three run PER GDN LAYER on prefill (48 layers in 27B, 30 in 35B).
Reading `GENESIS_TRITON_AUTOTUNE_MAX_WARPS` and `GENESIS_TRITON_AUTOTUNE_MAX_STAGES`
env vars (auto-set by PN296 to 4 and 2 respectively on SM 8.6).

================================================================
APPROACH
================================================================

For each file:
  - Replace literal `num_warps in [....]` with comprehension that
    filters by GENESIS_TRITON_AUTOTUNE_MAX_WARPS env var.
  - Same for num_stages where present.

Env vars are read at module IMPORT time (Triton autotune decorator
evaluates at decorator-eval time, which is module load). PN296 sets
env BEFORE any patches apply, so by the time these modules import,
env is correctly populated.

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-05.
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

log = logging.getLogger("genesis.wiring.pn299_fla_multi_arch_warps")

GENESIS_PN299_MARKER_KKT = (
    "Genesis PN299 chunk_scaled_dot_kkt arch warps (SM 8.6 prune) v1"
)
GENESIS_PN299_MARKER_WY = (
    "Genesis PN299 wy_fast arch warps (SM 8.6 prune) v1"
)
GENESIS_PN299_MARKER_L2 = (
    "Genesis PN299 l2norm arch warps (SM 8.6 prune) v1"
)


# ─── chunk_scaled_dot_kkt.py: replace [2, 4, 8] × [2, 3, 4] ──────────────
PN299_KKT_OLD = (
    "        for num_warps in [2, 4, 8]\n"
    "        for num_stages in [2, 3, 4]\n"
)
PN299_KKT_NEW = (
    "        # [Genesis PN299] arch-aware: filter by GENESIS_TRITON_AUTOTUNE_MAX_*\n"
    "        # (env vars auto-set by PN296 — SM 8.6 A5000: max_warps=4, max_stages=2)\n"
    "        for num_warps in [w for w in [2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "        for num_stages in [s for s in [2, 3, 4]\n"
    "                           if s <= int(__import__('os').environ.get(\n"
    "                               'GENESIS_TRITON_AUTOTUNE_MAX_STAGES', '4'))]\n"
)

# ─── wy_fast.py: same pattern ───────────────────────────────────────────
PN299_WY_OLD = (
    "        for num_warps in [2, 4, 8]\n"
    "        for num_stages in [2, 3, 4]\n"
)
PN299_WY_NEW = (
    "        # [Genesis PN299] arch-aware filter (see PN296 for env source)\n"
    "        for num_warps in [w for w in [2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "        for num_stages in [s for s in [2, 3, 4]\n"
    "                           if s <= int(__import__('os').environ.get(\n"
    "                               'GENESIS_TRITON_AUTOTUNE_MAX_STAGES', '4'))]\n"
)

# ─── l2norm.py site 1: kernel1 with num_warps up to 32 ──────────────────
PN299_L2_KERNEL1_OLD = (
    "        triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]\n"
)
PN299_L2_KERNEL1_NEW = (
    "        # [Genesis PN299] arch-aware filter — SM 8.6 caps at 4; num_warps=32 spills hard\n"
    "        triton.Config({}, num_warps=num_warps)\n"
    "        for num_warps in [w for w in [1, 2, 4, 8, 16, 32]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
)

# ─── l2norm.py site 2: kernel with num_warps up to 16 ───────────────────
PN299_L2_KERNEL2_OLD = (
    "        triton.Config({\"BT\": BT}, num_warps=num_warps)\n"
    "        for num_warps in [1, 2, 4, 8, 16]\n"
    "        for BT in BT_LIST\n"
)
PN299_L2_KERNEL2_NEW = (
    "        # [Genesis PN299] arch-aware filter — SM 8.6 caps at 4\n"
    "        triton.Config({\"BT\": BT}, num_warps=num_warps)\n"
    "        for num_warps in [w for w in [1, 2, 4, 8, 16]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "        for BT in BT_LIST\n"
)


def _apply_one(rel_path: str, marker: str, sub_patches: list[TextPatch]) -> tuple[str, int]:
    """Apply a TextPatcher to one file. Returns (status_msg, num_applied).

    Returns ``num_applied = len(sub_patches)`` for both APPLIED (new splice
    this boot) and IDEMPOTENT (marker already in file from a prior boot —
    the bind-mounted vllm tree is shared across container restarts). The
    caller only uses the count to decide success vs failure, so both
    states satisfy "the patch is live in this file".
    """
    target = resolve_vllm_file(rel_path)
    if target is None:
        return f"{rel_path}: file not found", 0
    patcher = TextPatcher(
        patch_name=f"PN299 {rel_path} — arch-aware NUM_WARPS prune",
        target_file=str(target),
        marker=marker,
        sub_patches=sub_patches,
        upstream_drift_markers=["[Genesis PN299", "GENESIS_TRITON_AUTOTUNE_MAX_WARPS"],
    )
    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return f"{rel_path}: FAILED — {failure.reason if failure else 'unknown'}", 0
    if result == TextPatchResult.SKIPPED:
        return f"{rel_path}: skipped — {failure.reason if failure else 'unknown'}", 0
    if result == TextPatchResult.IDEMPOTENT:
        return (
            f"{rel_path}: idempotent ({len(sub_patches)} sub-patches already applied)",
            len(sub_patches),
        )
    n = len(patcher.applied_sub_patches)
    return f"{rel_path}: applied {n}/{len(sub_patches)} sub-patches", n


_APPLIED = False


def apply() -> tuple[str, str]:
    """Apply PN299 — multi-file FLA arch-aware NUM_WARPS prune."""
    global _APPLIED

    if os.environ.get(
        "GENESIS_ENABLE_PN299_FLA_MULTI_ARCH_WARPS", ""
    ).lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "PN299 default OFF — set GENESIS_ENABLE_PN299_FLA_MULTI_ARCH_WARPS=1. "
            "Extends PN298 pattern to chunk_scaled_dot_kkt + wy_fast + l2norm."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    results = []
    total_applied = 0

    # File 1: chunk_scaled_dot_kkt.py
    msg, n = _apply_one(
        "model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py",
        GENESIS_PN299_MARKER_KKT,
        [TextPatch(name="pn299_kkt_warps", anchor=PN299_KKT_OLD,
                   replacement=PN299_KKT_NEW, required=True)],
    )
    results.append(msg)
    total_applied += n

    # File 2: wy_fast.py
    msg, n = _apply_one(
        "model_executor/layers/fla/ops/wy_fast.py",
        GENESIS_PN299_MARKER_WY,
        [TextPatch(name="pn299_wy_warps", anchor=PN299_WY_OLD,
                   replacement=PN299_WY_NEW, required=True)],
    )
    results.append(msg)
    total_applied += n

    # File 3: l2norm.py — TWO sub-patches in one TextPatcher
    msg, n = _apply_one(
        "model_executor/layers/fla/ops/l2norm.py",
        GENESIS_PN299_MARKER_L2,
        [
            TextPatch(name="pn299_l2_kernel1_warps", anchor=PN299_L2_KERNEL1_OLD,
                      replacement=PN299_L2_KERNEL1_NEW, required=True),
            TextPatch(name="pn299_l2_kernel_warps", anchor=PN299_L2_KERNEL2_OLD,
                      replacement=PN299_L2_KERNEL2_NEW, required=True),
        ],
    )
    results.append(msg)
    total_applied += n

    if total_applied == 0:
        return "failed", " | ".join(results)
    _APPLIED = True
    return "applied", (
        f"PN299 installed: {total_applied} sub-patches across 3 FLA files. "
        f"On SM 8.6 (max_warps=4, max_stages=2) Triton autotune drops 8-warp "
        f"and 16/32-warp configs that spill. Details: {' | '.join(results)}"
    )


def is_applied() -> bool:
    return _APPLIED

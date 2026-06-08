# SPDX-License-Identifier: Apache-2.0
"""PN299B — extend PN299 arch-aware NUM_WARPS prune to kda + cumsum + solve_tril.

Kernels-audit agent (2026-06-08) flagged a gap in PN299 coverage: it
prunes ``num_warps=8`` autotune configs from 3 FLA files
(``chunk_scaled_dot_kkt.py``, ``wy_fast.py``, ``l2norm.py``) but leaves
**5 other files** with the same spill-prone configs untouched:

* ``cumsum.py``       (2 configs) — chunk_local_cumsum variants
* ``chunk_delta_h.py`` (0 — already capped at [2, 4]; skip)
* ``kda.py``          (5 configs) — **the Qwen3.6 KDA hot path**
* ``solve_tril.py``   (3 configs) — solve_tril_16x16, merge_*_inverse

On SM 8.6 (RTX A5000 / 3090) the 100 KB shared memory per SM cannot
sustain 8-warp configs for these kernel shapes — autotune still picks
one of them on cold runs, then evicts after spill. The kda.py blocks
are particularly bad because Qwen3.6 hybrid_gdn_moe uses Kimi Delta
Attention layers on every block; cold-start autotune evictions there
add ~50-200 ms TTFT and contribute to the ~10 % observed wall_TPS gap
vs. Sprint-1 baseline.

PN299B is a pure structural copy of PN299 — same env-var (PN296 auto-
sets ``GENESIS_TRITON_AUTOTUNE_MAX_WARPS=4`` on Ampere), same list-
comprehension filter, same per-sub-patch ``required=False`` so partial-
apply is allowed when upstream layouts drift.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Audit lineage: kernels-audit agent finding #2 sub-issue, 2026-06-08.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn299b_fla_kda_cumsum_solve_tril")

GENESIS_PN299B_MARKER_CUMSUM = (
    "Genesis PN299B cumsum arch warps (SM 8.6 prune) v1"
)
GENESIS_PN299B_MARKER_KDA = (
    "Genesis PN299B kda arch warps (SM 8.6 prune) v1"
)
GENESIS_PN299B_MARKER_SOLVE_TRIL = (
    "Genesis PN299B solve_tril arch warps (SM 8.6 prune) v1"
)

_FILTER_NOTE = (
    "        # [Genesis PN299B] arch-aware filter (PN296 auto-sets env on SM 8.6)\n"
)


# ════════════════════════════════════════════════════════════════════════
# cumsum.py
# ════════════════════════════════════════════════════════════════════════

# Site 1: scalar kernel — single-line list comprehension.
# Anchor uses the unique key= signature.
PN299B_CUMSUM_SCALAR_OLD = (
    "    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],\n"
    "    key=[\"B\", \"H\", \"BT\", \"IS_VARLEN\", \"REVERSE\"],\n"
)
PN299B_CUMSUM_SCALAR_NEW = (
    "    # [Genesis PN299B] arch-aware filter (PN296 auto-sets env on SM 8.6)\n"
    "    configs=[triton.Config({}, num_warps=num_warps)\n"
    "             for num_warps in [w for w in [1, 2, 4, 8]\n"
    "                               if w <= int(__import__('os').environ.get(\n"
    "                                   'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]],\n"
    "    key=[\"B\", \"H\", \"BT\", \"IS_VARLEN\", \"REVERSE\"],\n"
)

# Site 2: vector kernel — multi-line with BS_LIST.
PN299B_CUMSUM_VECTOR_OLD = (
    "        triton.Config({\"BS\": BS}, num_warps=num_warps)\n"
    "        for BS in BS_LIST\n"
    "        for num_warps in [2, 4, 8]\n"
    "    ],\n"
    "    key=[\"B\", \"H\", \"S\", \"BT\", \"IS_VARLEN\", \"REVERSE\"],\n"
)
PN299B_CUMSUM_VECTOR_NEW = (
    "        # [Genesis PN299B] arch-aware filter on SM 8.6\n"
    "        triton.Config({\"BS\": BS}, num_warps=num_warps)\n"
    "        for BS in BS_LIST\n"
    "        for num_warps in [w for w in [2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "    ],\n"
    "    key=[\"B\", \"H\", \"S\", \"BT\", \"IS_VARLEN\", \"REVERSE\"],\n"
)


# ════════════════════════════════════════════════════════════════════════
# kda.py — 5 sites
# ════════════════════════════════════════════════════════════════════════

# Site 1 (~515): BK loop + num_warps [1,2,4,8] + num_stages [2,3,4], key=["BC"].
PN299B_KDA_SITE1_OLD = (
    "        triton.Config({\"BK\": BK}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for BK in [32, 64]\n"
    "        for num_warps in [1, 2, 4, 8]\n"
    "        for num_stages in [2, 3, 4]\n"
    "    ],\n"
    "    key=[\"BC\"],\n"
)
PN299B_KDA_SITE1_NEW = (
    "        # [Genesis PN299B] arch-aware filter on SM 8.6\n"
    "        triton.Config({\"BK\": BK}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for BK in [32, 64]\n"
    "        for num_warps in [w for w in [1, 2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "        for num_stages in [s for s in [2, 3, 4]\n"
    "                           if s <= int(__import__('os').environ.get(\n"
    "                               'GENESIS_TRITON_AUTOTUNE_MAX_STAGES', '4'))]\n"
    "    ],\n"
    "    key=[\"BC\"],\n"
)

# Site 2 (~623): single-line, key=["BK", "BT"].
PN299B_KDA_SITE2_OLD = (
    "    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],\n"
    "    key=[\"BK\", \"BT\"],\n"
)
PN299B_KDA_SITE2_NEW = (
    "    # [Genesis PN299B] arch-aware filter on SM 8.6\n"
    "    configs=[triton.Config({}, num_warps=num_warps)\n"
    "             for num_warps in [w for w in [1, 2, 4, 8]\n"
    "                               if w <= int(__import__('os').environ.get(\n"
    "                                   'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]],\n"
    "    key=[\"BK\", \"BT\"],\n"
)

# Site 3 (~810): num_warps [2,4,8] + num_stages [2,3,4], key with H,K,V,BT,BK,BV,IS_VARLEN.
PN299B_KDA_SITE3_OLD = (
    "        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for num_warps in [2, 4, 8]\n"
    "        for num_stages in [2, 3, 4]\n"
    "    ],\n"
    "    key=[\"H\", \"K\", \"V\", \"BT\", \"BK\", \"BV\", \"IS_VARLEN\"],\n"
)
PN299B_KDA_SITE3_NEW = (
    "        # [Genesis PN299B] arch-aware filter on SM 8.6\n"
    "        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for num_warps in [w for w in [2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "        for num_stages in [s for s in [2, 3, 4]\n"
    "                           if s <= int(__import__('os').environ.get(\n"
    "                               'GENESIS_TRITON_AUTOTUNE_MAX_STAGES', '4'))]\n"
    "    ],\n"
    "    key=[\"H\", \"K\", \"V\", \"BT\", \"BK\", \"BV\", \"IS_VARLEN\"],\n"
)

# Site 4 (~1013): BK,BV loop + num_warps [2,4,8] + num_stages [2,3,4], key=["BT"].
PN299B_KDA_SITE4_OLD = (
    "        triton.Config({\"BK\": BK, \"BV\": BV}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for BK in [32, 64]\n"
    "        for BV in [64, 128]\n"
    "        for num_warps in [2, 4, 8]\n"
    "        for num_stages in [2, 3, 4]\n"
    "    ],\n"
    "    key=[\"BT\"],\n"
)
PN299B_KDA_SITE4_NEW = (
    "        # [Genesis PN299B] arch-aware filter on SM 8.6\n"
    "        triton.Config({\"BK\": BK, \"BV\": BV}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for BK in [32, 64]\n"
    "        for BV in [64, 128]\n"
    "        for num_warps in [w for w in [2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "        for num_stages in [s for s in [2, 3, 4]\n"
    "                           if s <= int(__import__('os').environ.get(\n"
    "                               'GENESIS_TRITON_AUTOTUNE_MAX_STAGES', '4'))]\n"
    "    ],\n"
    "    key=[\"BT\"],\n"
)

# Site 5 (~1177): BD loop + num_warps [2,4,8], key with H,D,BT,IS_VARLEN.
PN299B_KDA_SITE5_OLD = (
    "        triton.Config({\"BD\": BD}, num_warps=num_warps)\n"
    "        for BD in [32, 64]\n"
    "        for num_warps in [2, 4, 8]\n"
    "    ],\n"
    "    key=[\"H\", \"D\", \"BT\", \"IS_VARLEN\"],\n"
)
PN299B_KDA_SITE5_NEW = (
    "        # [Genesis PN299B] arch-aware filter on SM 8.6\n"
    "        triton.Config({\"BD\": BD}, num_warps=num_warps)\n"
    "        for BD in [32, 64]\n"
    "        for num_warps in [w for w in [2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "    ],\n"
    "    key=[\"H\", \"D\", \"BT\", \"IS_VARLEN\"],\n"
)


# ════════════════════════════════════════════════════════════════════════
# solve_tril.py — 3 sites
# ════════════════════════════════════════════════════════════════════════

# Site 1 (~31): num_warps [1,2,4,8] + num_stages [2,3,4,5], key=["BT"].
PN299B_SOLVE_SITE1_OLD = (
    "        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for num_warps in [1, 2, 4, 8]\n"
    "        for num_stages in [2, 3, 4, 5]\n"
    "    ],\n"
    "    key=[\"BT\"],\n"
)
PN299B_SOLVE_SITE1_NEW = (
    "        # [Genesis PN299B] arch-aware filter on SM 8.6\n"
    "        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for num_warps in [w for w in [1, 2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "        for num_stages in [s for s in [2, 3, 4, 5]\n"
    "                           if s <= int(__import__('os').environ.get(\n"
    "                               'GENESIS_TRITON_AUTOTUNE_MAX_STAGES', '5'))]\n"
    "    ],\n"
    "    key=[\"BT\"],\n"
)

# Site 2 (~106): same num_warps [1,2,4,8] but key=["H", "BT", "IS_VARLEN"].
PN299B_SOLVE_SITE2_OLD = (
    "        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for num_warps in [1, 2, 4, 8]\n"
    "        for num_stages in [2, 3, 4, 5]\n"
    "    ],\n"
    "    key=[\"H\", \"BT\", \"IS_VARLEN\"],\n"
)
PN299B_SOLVE_SITE2_NEW = (
    "        # [Genesis PN299B] arch-aware filter on SM 8.6\n"
    "        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for num_warps in [w for w in [1, 2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "        for num_stages in [s for s in [2, 3, 4, 5]\n"
    "                           if s <= int(__import__('os').environ.get(\n"
    "                               'GENESIS_TRITON_AUTOTUNE_MAX_STAGES', '5'))]\n"
    "    ],\n"
    "    key=[\"H\", \"BT\", \"IS_VARLEN\"],\n"
)

# Site 3 (~231): num_warps [2,4,8] only, same key as Site 2 but different.
# This anchor is differentiated from Site 2 by the leading [2,4,8] vs [1,2,4,8].
PN299B_SOLVE_SITE3_OLD = (
    "        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for num_warps in [2, 4, 8]\n"
    "        for num_stages in [2, 3, 4, 5]\n"
    "    ],\n"
    "    key=[\"H\", \"BT\", \"IS_VARLEN\"],\n"
)
PN299B_SOLVE_SITE3_NEW = (
    "        # [Genesis PN299B] arch-aware filter on SM 8.6\n"
    "        triton.Config({}, num_warps=num_warps, num_stages=num_stages)\n"
    "        for num_warps in [w for w in [2, 4, 8]\n"
    "                          if w <= int(__import__('os').environ.get(\n"
    "                              'GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8'))]\n"
    "        for num_stages in [s for s in [2, 3, 4, 5]\n"
    "                           if s <= int(__import__('os').environ.get(\n"
    "                               'GENESIS_TRITON_AUTOTUNE_MAX_STAGES', '5'))]\n"
    "    ],\n"
    "    key=[\"H\", \"BT\", \"IS_VARLEN\"],\n"
)


# ════════════════════════════════════════════════════════════════════════
# Driver
# ════════════════════════════════════════════════════════════════════════

def _apply_one(rel_path: str, marker: str, sub_patches: list[TextPatch]) -> tuple[str, int]:
    target = resolve_vllm_file(rel_path)
    if target is None:
        return f"{rel_path}: file not found", 0
    patcher = TextPatcher(
        patch_name=f"PN299B {rel_path} — arch-aware NUM_WARPS prune",
        target_file=str(target),
        marker=marker,
        sub_patches=sub_patches,
        upstream_drift_markers=["[Genesis PN299B", "GENESIS_TRITON_AUTOTUNE_MAX_WARPS"],
    )
    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return f"{rel_path}: FAILED — {failure.reason if failure else 'unknown'}", 0
    if result == TextPatchResult.SKIPPED:
        return f"{rel_path}: skipped — {failure.reason if failure else 'unknown'}", 0
    if result == TextPatchResult.IDEMPOTENT:
        return f"{rel_path}: idempotent ({len(sub_patches)} sub-patches already applied)", len(sub_patches)
    n = len(patcher.applied_sub_patches)
    return f"{rel_path}: applied {n}/{len(sub_patches)} sub-patches", n


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN299B", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply PN299B — arch-aware NUM_WARPS prune for kda + cumsum + solve_tril."""
    if _env_disabled():
        return "skipped", "PN299B disabled via GENESIS_DISABLE_PN299B=1"

    # File 1: cumsum.py — 2 sub-patches
    cumsum_msg, cumsum_n = _apply_one(
        "model_executor/layers/fla/ops/cumsum.py",
        GENESIS_PN299B_MARKER_CUMSUM,
        [
            TextPatch(name="pn299b_cumsum_scalar_warps",
                      anchor=PN299B_CUMSUM_SCALAR_OLD,
                      replacement=PN299B_CUMSUM_SCALAR_NEW,
                      required=False),
            TextPatch(name="pn299b_cumsum_vector_warps",
                      anchor=PN299B_CUMSUM_VECTOR_OLD,
                      replacement=PN299B_CUMSUM_VECTOR_NEW,
                      required=False),
        ],
    )

    # File 2: kda.py — 5 sub-patches
    kda_msg, kda_n = _apply_one(
        "model_executor/layers/fla/ops/kda.py",
        GENESIS_PN299B_MARKER_KDA,
        [
            TextPatch(name="pn299b_kda_site1_bk_warps_stages",
                      anchor=PN299B_KDA_SITE1_OLD,
                      replacement=PN299B_KDA_SITE1_NEW,
                      required=False),
            TextPatch(name="pn299b_kda_site2_warps_bk_bt",
                      anchor=PN299B_KDA_SITE2_OLD,
                      replacement=PN299B_KDA_SITE2_NEW,
                      required=False),
            TextPatch(name="pn299b_kda_site3_warps_stages_hkvbtbkbv",
                      anchor=PN299B_KDA_SITE3_OLD,
                      replacement=PN299B_KDA_SITE3_NEW,
                      required=False),
            TextPatch(name="pn299b_kda_site4_bk_bv_warps_stages",
                      anchor=PN299B_KDA_SITE4_OLD,
                      replacement=PN299B_KDA_SITE4_NEW,
                      required=False),
            TextPatch(name="pn299b_kda_site5_bd_warps",
                      anchor=PN299B_KDA_SITE5_OLD,
                      replacement=PN299B_KDA_SITE5_NEW,
                      required=False),
        ],
    )

    # File 3: solve_tril.py — 3 sub-patches
    solve_msg, solve_n = _apply_one(
        "model_executor/layers/fla/ops/solve_tril.py",
        GENESIS_PN299B_MARKER_SOLVE_TRIL,
        [
            TextPatch(name="pn299b_solve_site1_bt_warps_stages",
                      anchor=PN299B_SOLVE_SITE1_OLD,
                      replacement=PN299B_SOLVE_SITE1_NEW,
                      required=False),
            TextPatch(name="pn299b_solve_site2_hbtvarlen_warps_stages",
                      anchor=PN299B_SOLVE_SITE2_OLD,
                      replacement=PN299B_SOLVE_SITE2_NEW,
                      required=False),
            TextPatch(name="pn299b_solve_site3_64x64",
                      anchor=PN299B_SOLVE_SITE3_OLD,
                      replacement=PN299B_SOLVE_SITE3_NEW,
                      required=False),
        ],
    )

    total = cumsum_n + kda_n + solve_n
    summary = f"cumsum: {cumsum_msg} | kda: {kda_msg} | solve_tril: {solve_msg}"
    if total == 0:
        return "skipped", f"PN299B no anchors matched. Details: {summary}"
    return "applied", (
        f"PN299B applied: {total}/10 sub-patches across cumsum + kda + "
        f"solve_tril (extends PN299 coverage to the Qwen3.6 KDA hot path "
        f"and 5 other Mamba/GDN kernels). Details: {summary}"
    )


def is_applied() -> bool:
    """True iff any of the three target files carry our marker."""
    for rel, marker in [
        ("model_executor/layers/fla/ops/cumsum.py", GENESIS_PN299B_MARKER_CUMSUM),
        ("model_executor/layers/fla/ops/kda.py", GENESIS_PN299B_MARKER_KDA),
        ("model_executor/layers/fla/ops/solve_tril.py", GENESIS_PN299B_MARKER_SOLVE_TRIL),
    ]:
        target = resolve_vllm_file(rel)
        if target is None:
            continue
        try:
            if marker in target.read_text(encoding="utf-8"):
                return True
        except (OSError, UnicodeDecodeError):
            continue
    return False

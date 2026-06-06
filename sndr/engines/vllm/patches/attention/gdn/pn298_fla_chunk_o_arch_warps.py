# SPDX-License-Identifier: Apache-2.0
"""Patch PN298 — FLA chunk_o NUM_WARPS arch-aware pruning.

Genesis-original 2026-06-05 — reads `gpu_arch_profile` to prune Triton
autotune configs that don't fit our shared-mem budget.

================================================================
PROBLEM
================================================================

Upstream `vllm/model_executor/layers/fla/ops/chunk_o.py:23`:

    NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]

The intent: Hopper (SM 9.0+, 228KB shared/SM, WGMMA) does well with 2-4
warps. Everything else gets [2, 4, 8] — but Ampere SM 8.6 (A5000, A6000,
RTX 30xx) has ONLY 100KB shared/SM — `num_warps=8` configs SPILL
registers and run slower.

Triton autotune at first-call time TRIES every config in the search
space. Bad configs add cold-start time (TTFT cost) AND may even win
autotune if their bench iteration happens to be cached differently.

================================================================
FIX
================================================================

Replace the module-level `NUM_WARPS = ...` line with arch-aware version
that consults `get_gpu_arch_profile().max_safe_num_warps`:

  - SM 8.x consumer (A5000, A6000, RTX 30xx, A40, RTX 4090): max=4 → [2, 4]
  - A100 SM 8.0 (164KB shared): max=8 → [2, 4, 8]
  - SM 9.0+ (Hopper/Blackwell, 228KB shared): max=8 → [2, 4, 8]
    (but file already restricts these to [2, 4] via is_nvidia_hopper)

On our 27B Lorbus + 48 GDN layers, chunk_o.fwd_o is called per layer per
prefill iteration. Cutting autotune search by 33% (3 configs → 2) means
faster autotune convergence on cold start and removes the num_warps=8
spilling risk on Ampere.

================================================================
SAFETY
================================================================

- Pure config pruning — kernel correctness unchanged.
- Operator override: if `GENESIS_PN298_FORCE_ALL_WARPS=1`, restore
  upstream behavior (rare diagnostic case).
- Falls through to upstream if arch profile not available (no detection).

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-05.
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.core import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)
from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn298_fla_chunk_o_arch_warps")

GENESIS_PN298_MARKER = (
    "Genesis PN298 FLA chunk_o NUM_WARPS arch-aware (SM 8.6 prune) v1"
)


PN298_OLD = (
    "BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]\n"
    "NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]\n"
)

PN298_NEW = (
    "BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]\n"
    "# [Genesis PN298 2026-06-05] arch-aware NUM_WARPS — drop num_warps=8\n"
    "# on Ampere SM 8.6 (A5000 100KB shared/SM cannot fit 8-warp configs\n"
    "# with BV=128 — spills registers). is_nvidia_hopper covers SM 9.0+\n"
    "# but Ampere consumer also benefits. A100 (SM 8.0, 164KB) keeps 8.\n"
    "try:\n"
    "    from vllm.sndr_core.detection.gpu_arch_profile import (\n"
    "        get_gpu_arch_profile as _genesis_pn298_get_profile,\n"
    "    )\n"
    "    _genesis_pn298_prof = _genesis_pn298_get_profile()\n"
    "    if _genesis_pn298_prof is not None:\n"
    "        _genesis_pn298_max = _genesis_pn298_prof.max_safe_num_warps\n"
    "        NUM_WARPS = [w for w in [2, 4, 8] if w <= _genesis_pn298_max]\n"
    "    else:\n"
    "        NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]\n"
    "except Exception:\n"
    "    NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/fla/ops/chunk_o.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN298 model_executor/layers/fla/ops/chunk_o.py — arch-aware "
            "NUM_WARPS prune (SM 8.6 100KB shared mem budget)"
        ),
        target_file=str(target),
        marker=GENESIS_PN298_MARKER,
        sub_patches=[
            TextPatch(
                name="pn298_chunk_o_num_warps_arch",
                anchor=PN298_OLD,
                replacement=PN298_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN298",
            "_genesis_pn298_max",
        ],
    )


_APPLIED = False


def apply() -> tuple[str, str]:
    """Apply PN298 — chunk_o NUM_WARPS arch-aware prune."""
    global _APPLIED

    if os.environ.get(
        "GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS", ""
    ).lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "PN298 default OFF — set "
            "GENESIS_ENABLE_PN298_FLA_CHUNK_O_ARCH_WARPS=1 to engage. "
            "Drops num_warps=8 from FLA chunk_o autotune search space on "
            "Ampere SM 8.6 (registers spill with BV=128). Composes with "
            "PN296 arch profiler."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "model_executor/layers/fla/ops/chunk_o.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown TextPatch failure"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown TextPatch skip"
    applied = patcher.applied_sub_patches or [sp.name for sp in patcher.sub_patches]
    _APPLIED = True
    return "applied", (
        f"PN298 installed: FLA chunk_o.py NUM_WARPS now reads "
        f"get_gpu_arch_profile().max_safe_num_warps. On SM 8.6 A5000 "
        f"(100KB shared) Triton autotune skips num_warps=8 configs that "
        f"would spill registers. Sub-patches: {', '.join(applied)}."
    )


def is_applied() -> bool:
    return _APPLIED

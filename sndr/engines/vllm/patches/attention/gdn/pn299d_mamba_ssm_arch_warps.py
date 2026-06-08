# SPDX-License-Identifier: Apache-2.0
"""PN299D — arch-aware cap on Mamba2 selective_state_update fallback heuristic.

``vllm/model_executor/layers/mamba/ops/mamba_ssm.py::
_get_default_ssm_launch_config`` is the **fallback** path used when no
tuned JSON config exists for the live (headdim, dstate, cache_dtype,
device) combination. The heuristic body::

    BLOCK_SIZE_M, num_warps = 4, 8
    if dstate <= 16:
        BLOCK_SIZE_M, num_warps = 32, 4
    elif dstate <= 32:
        BLOCK_SIZE_M, num_warps = 16, 4
    elif dstate <= 64:
        BLOCK_SIZE_M, num_warps = 8, 4
    else:
        if is_blackwell:
            BLOCK_SIZE_M, num_warps = 32, 8
        elif dstate <= 128:
            BLOCK_SIZE_M, num_warps = 4, 4
    return BLOCK_SIZE_M, num_warps

Two paths leave the function with ``num_warps = 8``:
  1. ``dstate > 128`` AND **not** Blackwell → stays at the initial
     ``(4, 8)`` assignment.
  2. ``dstate > 128`` AND Blackwell → ``(32, 8)``.

On SM 8.6 (A5000 / 3090, NOT Blackwell), path 1 fires whenever a
Mamba layer is configured with ``dstate > 128``. Some Qwen3.6 variants
ship ``dstate = 256`` for the Mamba blocks. The 8-warp choice spills
the 100 KB shared per SM on Ampere → eviction-recompile loop in
autotune → ~50-100 ms TTFT inflation on first Mamba forward.

PN299D appends a single ``num_warps = min(num_warps, env_max)`` cap
just before the ``return`` — defensive on tuned-config-present (no
effect; heuristic is bypassed), corrective on heuristic-fallback
when ``dstate > 128``. The env var is the same one PN296 auto-sets.

This patch is a NO-OP whenever the tuned config JSON for the live
(headdim, dstate, cache_dtype, device) tuple is present in the
vllm install — those configs were generated on the matching
hardware and are authoritative.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn299d_mamba_ssm")

GENESIS_PN299D_MARKER = (
    "Genesis PN299D mamba_ssm SSU fallback arch warps cap (SM 8.6) v1"
)


# The heuristic ends with ``return BLOCK_SIZE_M, num_warps``. We insert
# the cap just BEFORE the return. The leading 4-space indent matches
# the function body.
PN299D_OLD = (
    "        elif dstate <= 128:\n"
    "            BLOCK_SIZE_M, num_warps = 4, 4\n"
    "    return BLOCK_SIZE_M, num_warps\n"
)
PN299D_NEW = (
    "        elif dstate <= 128:\n"
    "            BLOCK_SIZE_M, num_warps = 4, 4\n"
    "    # [Genesis PN299D] arch-aware cap on the SSU fallback heuristic.\n"
    "    # PN296 auto-sets GENESIS_TRITON_AUTOTUNE_MAX_WARPS=4 on SM 8.6\n"
    "    # (A5000 / 3090) where the upstream ``dstate > 128, not Blackwell``\n"
    "    # branch would leave num_warps=8 — spills on 100 KB shared/SM.\n"
    "    # Without env (non-Ampere host), default '8' = upstream behaviour.\n"
    "    num_warps = min(\n"
    "        num_warps,\n"
    "        int(__import__('os').environ.get('GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8')),\n"
    "    )\n"
    "    return BLOCK_SIZE_M, num_warps\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN299D", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply PN299D — cap the Mamba2 SSU fallback heuristic."""
    if _env_disabled():
        return "skipped", "PN299D disabled via GENESIS_DISABLE_PN299D=1"

    target = resolve_vllm_file("model_executor/layers/mamba/ops/mamba_ssm.py")
    if target is None:
        return "skipped", (
            "PN299D: mamba_ssm.py not found in vllm install — pin may "
            "predate the Mamba2 ssm op or have a different layout"
        )

    patcher = TextPatcher(
        patch_name=(
            "PN299D model_executor/layers/mamba/ops/mamba_ssm.py — "
            "SSU fallback heuristic arch-aware NUM_WARPS cap"
        ),
        target_file=str(target),
        marker=GENESIS_PN299D_MARKER,
        sub_patches=[
            TextPatch(
                name="pn299d_ssu_fallback_num_warps_cap",
                anchor=PN299D_OLD,
                replacement=PN299D_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["[Genesis PN299D", "GENESIS_TRITON_AUTOTUNE_MAX_WARPS"],
    )

    try:
        result, failure = patcher.apply()
    except Exception as e:
        log.warning("[PN299D] apply() raised %s — leaving upstream heuristic", e)
        return "skipped", f"PN299D raised at apply: {e!r}"

    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "anchor drift / not eligible"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "skipped", f"PN299D: {reason}{detail}"

    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "failed", f"PN299D: {reason}{detail}"

    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN299D idempotent: marker already present"

    return "applied", (
        "PN299D applied: mamba_ssm.py SSU fallback heuristic now caps "
        "num_warps via GENESIS_TRITON_AUTOTUNE_MAX_WARPS (PN296 auto-sets "
        "=4 on SM 8.6). Defensive against dstate>128 layers on Ampere."
    )


def is_applied() -> bool:
    target = resolve_vllm_file("model_executor/layers/mamba/ops/mamba_ssm.py")
    if target is None:
        return False
    try:
        return GENESIS_PN299D_MARKER in target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

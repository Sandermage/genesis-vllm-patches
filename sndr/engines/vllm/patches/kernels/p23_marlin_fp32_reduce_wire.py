# SPDX-License-Identifier: Apache-2.0
"""Wiring for P23 — Marlin FP32_REDUCE env override text-patch (FIX WIRE).

Previously P23 shipped only a working env reader (`should_disable_fp32_reduce()`)
and platform auto-detect, but the dispatch handler ONLY LOGGED the
decision — never propagated it into the live kernel. As a result the
env override was inert at runtime even with all conditions met.

This wire-apply landed 2026-06-04 after P23/P29 deep study confirmed:
- Genesis env reader works correctly (`sndr/engines/vllm/kernels_legacy/marlin_fp32_reduce.py`)
- Platform guard correctly disables on SM 9.0+ (Hopper has native FP32 TCs)
- BUT upstream `marlin_utils.py:36` `USE_FP32_REDUCE_DEFAULT = True` was
  module-level constant + `marlin_moe.py:158, 217` hardcoded the arg.
  Decision never reached the kernel.

The fix: TWO text-patches.

(A) `model_executor/layers/quantization/utils/marlin_utils.py:36`
    Convert `USE_FP32_REDUCE_DEFAULT = True` to env-driven read.

(B) `model_executor/layers/fused_moe/experts/marlin_moe.py:158, 217`
    Replace hardcoded `use_fp32_reduce=True` with env-driven variable.
    Two call sites: W13 (line 158) and W2 (line 217). Both patched.

Effect on PROD 27B Qwen3.6 MoE (GPTQ/Marlin, 2× A5000 SM 8.6):
- VLLM_MARLIN_FP32_REDUCE not set → keep default (True), no behavior change
- VLLM_MARLIN_FP32_REDUCE=0 → both call sites use False → +1.5-3% TGS
  per Genesis empirical data, no quality drop on GSM8K/MMLU sweeps.

Operator opt-in: set `VLLM_MARLIN_FP32_REDUCE=0` in launcher env
(or via the P23 platform-auto-detect that disables on SM 8.x by default).

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-04 fix-wire pass.
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

log = logging.getLogger("genesis.wiring.p23_marlin_fp32_reduce_wire")

GENESIS_P23_WIRE_MARKER = "Genesis P23 Marlin FP32_REDUCE wire (env-driven) v1"


# ───────────────────────── marlin_utils.py:36 ─────────────────────────
_MARLIN_UTILS_OLD = (
    "# In case there is a performance issue with Marlin, the variable below can be\n"
    "# changed to False, which allows Marlin to perform global reductions in fp16\n"
    "# precision (instead of fp32), and therefore, save on some memory movements.\n"
    "USE_FP32_REDUCE_DEFAULT = True"
)

_MARLIN_UTILS_NEW = (
    "# In case there is a performance issue with Marlin, the variable below can be\n"
    "# changed to False, which allows Marlin to perform global reductions in fp16\n"
    "# precision (instead of fp32), and therefore, save on some memory movements.\n"
    "# [Genesis P23] env-driven: VLLM_MARLIN_FP32_REDUCE=0 disables fp32 reduce\n"
    "# (recommended on SM 8.x — Ampere has no FP32 tensor cores, +1.5-3% TGS).\n"
    "import os as _genesis_p23_os\n"
    "USE_FP32_REDUCE_DEFAULT = _genesis_p23_os.environ.get(\n"
    "    \"VLLM_MARLIN_FP32_REDUCE\", \"1\"\n"
    ").strip().lower() not in (\"0\", \"false\", \"no\", \"off\")"
)


# ───────────────────────── marlin_moe.py:158 (W13) ─────────────────────────
_MARLIN_MOE_W13_OLD = (
    "        size_k=K,\n"
    "        is_k_full=is_k_full,\n"
    "        use_atomic_add=False,\n"
    "        use_fp32_reduce=True,\n"
    "        is_zp_float=False,\n"
    "    )"
)

_MARLIN_MOE_W13_NEW = (
    "        size_k=K,\n"
    "        is_k_full=is_k_full,\n"
    "        use_atomic_add=False,\n"
    "        # [Genesis P23] env-driven fp32 reduce\n"
    "        use_fp32_reduce=__import__('os').environ.get(\n"
    "            'VLLM_MARLIN_FP32_REDUCE', '1'\n"
    "        ).strip().lower() not in ('0', 'false', 'no', 'off'),\n"
    "        is_zp_float=False,\n"
    "    )"
)


def _make_marlin_utils_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/quantization/utils/marlin_utils.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P23 Marlin FP32_REDUCE wire — marlin_utils.py:36",
        target_file=str(target),
        marker=GENESIS_P23_WIRE_MARKER + " — utils",
        sub_patches=[
            TextPatch(
                name="p23_marlin_utils_env",
                anchor=_MARLIN_UTILS_OLD,
                replacement=_MARLIN_UTILS_NEW,
                required=True,
            ),
        ],
    )


def _make_marlin_moe_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/fused_moe/experts/marlin_moe.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P23 Marlin FP32_REDUCE wire — marlin_moe.py:158,217",
        target_file=str(target),
        marker=GENESIS_P23_WIRE_MARKER + " — moe",
        sub_patches=[
            # The same anchor appears TWICE in the file (W13 + W2 call sites).
            # Genesis TextPatcher counts duplicates as AMBIGUOUS, so we use
            # a multi-occurrence-tolerant pattern via replace_all-style intent:
            # the anchor is unique enough that replacing both is safe and
            # desired (both must read the env identically). We do this via
            # a single sub-patch with the duplicate-tolerant variant below.
            #
            # NOTE: TextPatcher rejects count > 1 by default for safety.
            # Use a pair of sub-patches where each targets one site via
            # added uniqueness (the line BEFORE the duplicate context).
            # In practice: both sites have identical 6-line context, so
            # we patch the SHARED block with a marker-suppressed replacement
            # written ONCE, then run the patcher TWICE in sequence — each
            # run picks up the next unpatched occurrence.
            TextPatch(
                name="p23_marlin_moe_env_w13_w2",
                anchor=_MARLIN_MOE_W13_OLD,
                replacement=_MARLIN_MOE_W13_NEW,
                required=True,
            ),
        ],
    )


_APPLIED = False


def apply() -> tuple[str, str]:
    """Apply P23 wire — env-driven Marlin FP32_REDUCE on both upstream sites."""
    global _APPLIED

    if os.environ.get("GENESIS_ENABLE_P23_MARLIN_FP32_REDUCE_WIRE", "").lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "P23 wire default OFF — set GENESIS_ENABLE_P23_MARLIN_FP32_REDUCE_WIRE=1 "
            "to engage. Then VLLM_MARLIN_FP32_REDUCE=0 actually disables fp32 reduce. "
            "Recommended on SM 8.x (Ampere); contra-productive on SM 9.0+ (Hopper has "
            "native FP32 TCs)."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    # Patch 1: marlin_utils.py
    p1 = _make_marlin_utils_patcher()
    if p1 is None:
        return "skipped", "marlin_utils.py not found"
    r1, f1 = p1.apply()
    if r1 == TextPatchResult.FAILED:
        return "failed", f"marlin_utils: {f1.reason if f1 else 'unknown'}"

    # Patch 2: marlin_moe.py — apply TWICE (W13 + W2 sites share anchor).
    # First apply patches one occurrence; second apply patches the other.
    # TextPatcher's idempotent marker prevents triple-apply.
    p2_first = _make_marlin_moe_patcher()
    if p2_first is None:
        return "skipped", "marlin_moe.py not found"
    # Need separate markers for W13 and W2 sites because both reuse
    # the same shared marker key — to allow second apply, instantiate
    # patcher with a slightly different marker for each site. Simpler:
    # just call apply twice with same patcher; second call sees marker
    # already present → IDEMPOTENT, returns without modifying. Need a
    # different approach: directly do the dual replace below.
    target_moe = resolve_vllm_file("model_executor/layers/fused_moe/experts/marlin_moe.py")
    if target_moe is None:
        return "failed", "marlin_moe.py disappeared between patcher instantiation and apply"
    try:
        with open(target_moe, "r") as fh:
            content = fh.read()
        marker_line = f"# [Genesis wiring marker: {GENESIS_P23_WIRE_MARKER + ' — moe'}]\n"
        if marker_line in content:
            # already applied
            _APPLIED = True
            return "applied", "P23 wire idempotent (both sites already patched)"
        # Replace ALL occurrences (W13 + W2 share identical anchor block).
        if _MARLIN_MOE_W13_OLD not in content:
            return "failed", "marlin_moe anchor not found at any site"
        n_occurrences = content.count(_MARLIN_MOE_W13_OLD)
        if n_occurrences not in (1, 2):
            return "failed", f"marlin_moe anchor appears {n_occurrences} times (expected 1 or 2)"
        new_content = content.replace(_MARLIN_MOE_W13_OLD, _MARLIN_MOE_W13_NEW)
        new_content = marker_line + new_content
        with open(target_moe, "w") as fh:
            fh.write(new_content)
        sites_patched = n_occurrences
    except (OSError, PermissionError) as e:
        return "failed", f"marlin_moe write_error: {e}"

    _APPLIED = True
    return "applied", (
        f"P23 wire installed: marlin_utils.py:36 + marlin_moe.py "
        f"({sites_patched} sites) now read VLLM_MARLIN_FP32_REDUCE env. "
        f"Set VLLM_MARLIN_FP32_REDUCE=0 in launcher to engage."
    )


def is_applied() -> bool:
    return _APPLIED

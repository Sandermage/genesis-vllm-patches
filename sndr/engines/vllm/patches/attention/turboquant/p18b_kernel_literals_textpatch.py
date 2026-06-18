# SPDX-License-Identifier: Apache-2.0
"""Wiring for P18B_TEXT — TurboQuant decode stage1 kernel-literal tune.

Genesis-original. The original P18b (kernels_legacy/tq_decode_tune.py +
dispatch hook in sndr/apply/_per_patch_dispatch.py:6275) reads the
``VLLM_TQ_DECODE_{BLOCK_KV,NUM_WARPS,NUM_STAGES}`` env vars and **logs**
their resolved value, but never patches the actual Triton launcher.

Kernels-audit agent (2026-06-08) flagged this as dead code: 35B + 27B
production has been running with the upstream H100 defaults
(``num_warps=4, num_stages=2`` on the GQA branch, ``num_warps=1,
num_stages=1`` on the MHA branch) on every boot, regardless of env
overrides — under-utilising Ampere SM 8.6 (RTX A5000 / 3090) shared-
memory budgets.

This patch is the missing text-patch half. It rewrites the two launch-
parameter blocks of ``vllm/v1/attention/ops/triton_turboquant_decode.py``
in place at boot using the values from ``resolve_decode_tune()``. The
SM-8.6-validated tune is ``num_warps=8, num_stages=3``, but that is a
RECOMMENDED OVERRIDE, not the shipped default: ``resolve_decode_tune()``
returns the upstream values (``num_warps=4, num_stages=2`` GQA /
``1, 1`` MHA) unless ``VLLM_TQ_DECODE_NUM_WARPS=8`` /
``VLLM_TQ_DECODE_NUM_STAGES=3`` are set in the environment. Without those
env vars this patch rewrites the launcher to the same upstream literals
(inert). Set the env to actually realise the SM-8.6 tune.

Expected impact (HIGH confidence on the fix actually applying, MEDIUM
on the TPS number): +3-8 % on 35B-A3B-FP8 + TQ k8v4 + MTP K=3. Bench
A/B before promoting from experimental.

Safety:
  - Exact text-anchor match, soft-skip on drift.
  - Per-branch (GQA / MHA) sub-patches, both optional — partial-apply
    is allowed (some pins ship only one branch).
  - Operator override ``GENESIS_DISABLE_P18B_TEXT=1`` keeps upstream
    literals; ``VLLM_TQ_DECODE_NUM_WARPS`` / ``VLLM_TQ_DECODE_NUM_STAGES``
    flow through ``resolve_decode_tune()`` and tune the replacement.
  - Self-suppresses on non-NVIDIA / pre-Ampere via
    ``tq_decode_tune.should_apply()``.
  - Idempotent marker — re-apply is a no-op.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.engines.vllm.kernels_legacy.tq_decode_tune import (
    resolve_decode_tune,
    should_apply as tq_should_apply,
)
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.p18b_kernel_literals_textpatch")

GENESIS_P18B_TEXT_MARKER = (
    "Genesis P18b TEXT TurboQuant decode stage1 kernel-literal tune "
    "(SM 8.6 num_warps/num_stages override)"
)


# GQA path — line ~787-792 of triton_turboquant_decode.py. The
# four-line literal block is unique in the file.
P18B_GQA_OLD = (
    "            FP8_E4B15=fp8_e4b15,\n"
    "            num_warps=4,\n"
    "            num_stages=2,\n"
    "        )\n"
    "    else:\n"
)

# MHA path — line ~828-832. Same launcher, MHA branch
# (kv_group_size==1). The trailing comment ("# Stage 2:") anchors the
# replacement uniquely.
P18B_MHA_OLD = (
    "            FP8_E4B15=fp8_e4b15,\n"
    "            num_warps=1,\n"
    "            num_stages=1,\n"
    "        )\n"
    "\n"
    "    # Stage 2:"
)


def _build_replacement(num_warps: int, num_stages: int, branch: str) -> str:
    """Render the new launch-param block with our resolved tune.

    ``branch`` is ``"GQA"`` or ``"MHA"`` — only used in the comment.
    """
    note = (
        f"            # [Genesis P18b TEXT, 2026-06-08] {branch} launcher\n"
        f"            # tuned for Ampere SM 8.6 (RTX A5000 / 3090).\n"
        f"            # Upstream defaults were H100-shaped (1-4 warps,\n"
        f"            # 1-2 stages) — under-utilised the 100 KB shared\n"
        f"            # / 64 KB L1 budget per SM on consumer Ampere.\n"
        f"            # Override via VLLM_TQ_DECODE_NUM_WARPS /\n"
        f"            # VLLM_TQ_DECODE_NUM_STAGES (tq_decode_tune.py).\n"
    )
    return (
        "            FP8_E4B15=fp8_e4b15,\n"
        + note
        + f"            num_warps={num_warps},\n"
        + f"            num_stages={num_stages},\n"
        + "        )\n"
    )


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/ops/triton_turboquant_decode.py")
    if target is None:
        return None

    _bkv, num_warps, num_stages = resolve_decode_tune()

    gqa_new = _build_replacement(num_warps, num_stages, "GQA") + "    else:\n"
    mha_new = (
        _build_replacement(num_warps, num_stages, "MHA") + "\n    # Stage 2:"
    )

    return TextPatcher(
        patch_name=(
            "P18b TEXT v1/attention/ops/triton_turboquant_decode.py — "
            "kernel-literal tune (num_warps/num_stages SM 8.6)"
        ),
        target_file=str(target),
        marker=GENESIS_P18B_TEXT_MARKER,
        sub_patches=[
            TextPatch(
                name="p18b_text_gqa_launch_tune",
                anchor=P18B_GQA_OLD,
                replacement=gqa_new,
                required=False,
            ),
            TextPatch(
                name="p18b_text_mha_launch_tune",
                anchor=P18B_MHA_OLD,
                replacement=mha_new,
                required=False,
            ),
        ],
        upstream_drift_markers=["[Genesis P18b TEXT"],
    )


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_P18B_TEXT", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply P18b TEXT — text-patch the TQ decode launch literals."""
    if _env_disabled():
        return "skipped", (
            "P18b TEXT disabled via GENESIS_DISABLE_P18B_TEXT=1 — leaving "
            "upstream H100-default kernel launch params"
        )

    if not tq_should_apply():
        return "skipped", (
            "P18b TEXT: TurboQuant not applicable on this device "
            "(non-CUDA or pre-Ampere) — kernel literals not patched"
        )

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", (
            "P18b TEXT: triton_turboquant_decode.py not found in vllm "
            "install — pin may predate TurboQuant or have a different layout"
        )

    bkv, num_warps, num_stages = resolve_decode_tune()

    try:
        result, failure = patcher.apply()
    except Exception as e:  # never raise out of an apply hook
        log.warning(
            "[P18b TEXT] apply() raised %s — leaving upstream kernel literals",
            e,
        )
        return "skipped", f"P18b TEXT raised at apply: {e!r}"

    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "anchor drift / not eligible"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "skipped", (
            f"P18b TEXT: {reason}{detail}. Resolved tune was "
            f"BLOCK_KV={bkv} num_warps={num_warps} num_stages={num_stages}; "
            f"kernel literals NOT overridden — upstream H100 defaults remain."
        )

    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "failed", f"P18b TEXT: {reason}{detail}"

    if result == TextPatchResult.IDEMPOTENT:
        return "applied", (
            f"P18b TEXT idempotent: marker already present (num_warps="
            f"{num_warps} num_stages={num_stages} previously installed)."
        )

    applied = ", ".join(patcher.applied_sub_patches) or "(unknown)"
    return "applied", (
        f"P18b TEXT applied: TQ decode stage1 launch literals overridden "
        f"to num_warps={num_warps} num_stages={num_stages} via sub-patches "
        f"[{applied}]. Closes the dead-code finding (upstream H100 defaults "
        f"4/2 + 1/1 were silently in use despite env overrides; tq_decode_tune "
        f"was logging-only)."
    )


def is_applied() -> bool:
    """Best-effort check by reading the target file for our marker."""
    target = resolve_vllm_file("v1/attention/ops/triton_turboquant_decode.py")
    if target is None:
        return False
    try:
        return GENESIS_P18B_TEXT_MARKER in target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

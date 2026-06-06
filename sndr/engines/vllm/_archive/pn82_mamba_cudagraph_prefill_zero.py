# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN82 — Mamba CUDA-graph stale `is_prefilling` padded rows.

Backport of [vllm-project/vllm#41873](https://github.com/vllm-project/vllm/pull/41873)
(OPEN as of 2026-05-07).

================================================================
WHAT THIS PATCH DOES
================================================================

In `v1/worker/gpu_model_runner.py`, the input batch is padded out to
`num_reqs_padded` so CUDA-graph captures replay at a fixed shape. The
boolean tensor `is_prefilling` is computed as

    is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu

over the **padded** range. After `condense()` rotates the input batch,
trailing padded slots `[num_reqs:num_reqs_padded]` keep whatever
`num_computed_tokens_cpu` / `num_prompt_tokens_cpu` they had from a
previous step. Often those happen to satisfy the inequality and end up
as `True` — telling Mamba/hybrid attention backends that the padded
rows are still in prefill phase. Backends then route those rows
through prefill kernels even though no real request occupies them,
which can corrupt Mamba state on hybrid models or trip assertions.

The upstream fix is a one-line zero of the padding region right after
the assignment:

    is_prefilling[num_reqs:] = False

================================================================
WHY GENESIS BACKPORT
================================================================

The bug is most visible on:

  - Qwen3.6-27B Lorbus INT4 (hybrid GDN), CUDA-graph capture +
    chunked prefill — the Mamba layers read `is_prefilling` directly.
  - Any DFlash + hybrid path that touches Mamba state.

35B-A3B-FP8 is NOT affected (no Mamba layers). The patch is
`applies_to.is_hybrid=True` gated, so dense MoE deployments will skip
it cleanly.

================================================================
ENV
================================================================

GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO=1 to opt in.

Default OFF until live smoke confirms no regression on hybrid +
CUDA-graph configs. Plan §3.2 PR38 calls for explicit enable in 27B
hybrid configs after one round of bench validation.

================================================================
RISK
================================================================

LOW.

  - Single extra line; no flow-control change for non-padded rows.
  - Anchor stable on vllm dev93 (and main HEAD as of 2026-05-07);
    upstream drift markers detect when the line is already present
    natively (i.e. PR #41873 merges).
  - Idempotent (marker-protected); safe to re-apply.

The only true regression vector would be if some downstream code
reads `is_prefilling[num_reqs:]` and expects the unmasked value. We
have audited gpu_model_runner.py callers — none read past `num_reqs`
on the prefilling slice; only `seq_lens_cpu_upper_bound` and the
condense-aware paths do, and those already account for padding.

================================================================
STATE
================================================================

PR38 Day 1 (2026-05-07): patch landed default OFF. Anchor exact-match
on vllm dev93 confirmed; upstream PR #41873 still OPEN at this writing.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Backport reference: vllm-project/vllm#41873 (algorithm 80-90% upstream;
Genesis adds packaging, gating, idempotent TextPatch, drift markers,
tests, model-config integration).
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import (
    TextPatch, TextPatcher, TextPatchResult,
)


log = logging.getLogger("genesis.wiring.pN82_mamba_cudagraph_prefill_zero")


GENESIS_PN82_MARKER = (
    "Genesis PN82 Mamba CUDA-graph prefill zero "
    "(vllm#41873 backport)"
)


# Anchor: the exact assignment line + 2 preceding comment lines.
# The comments make this anchor strongly disambiguating — there is no
# other `is_prefilling = ...` site in gpu_model_runner.py.
PN82_ANCHOR = (
    "        # is_prefilling: True if request is still in prefill phase.\n"
    "        # Used by mamba backends to distinguish actual decodes from\n"
    "        # short extends.\n"
    "        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu\n"
)


PN82_REPLACEMENT = (
    "        # is_prefilling: True if request is still in prefill phase.\n"
    "        # Used by mamba backends to distinguish actual decodes from\n"
    "        # short extends.\n"
    "        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu\n"
    "        # [Genesis PN82 vllm#41873] zero padded CUDA-graph rows so\n"
    "        # condense() leftover True values don't mislead Mamba/hybrid\n"
    "        # backends into treating padding as prefill.\n"
    "        is_prefilling[num_reqs:] = False\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN82 v1/worker/gpu_model_runner.py — Mamba CUDA-graph "
            "padded `is_prefilling` zero (vllm#41873 backport)"
        ),
        target_file=str(target),
        marker=GENESIS_PN82_MARKER,
        sub_patches=[
            TextPatch(
                name="pN82_mamba_cudagraph_prefill_zero",
                anchor=PN82_ANCHOR,
                replacement=PN82_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN82",
            # If upstream PR #41873 merges, the modified file will
            # contain literally `is_prefilling[num_reqs:] = False` as
            # the line right after the assignment. Detect that exact
            # form to self-retire.
            "is_prefilling[num_reqs:] = False",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN82 — Mamba CUDA-graph padded `is_prefilling` zero.

    Default OFF; opt in via GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO=1
    after smoke validation on hybrid + CUDA-graph configs.
    """
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN82")
    log_decision("PN82", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/worker/gpu_model_runner.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[PN82] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} "
                "— upstream PR #41873 (or equivalent) appears merged. "
                "PN82 self-retires; verify on next vllm pin bump.",
            )

    # TextPatcher.apply() returns (TextPatchResult, TextPatchFailure | None).
    # Use the canonical mapping helper instead of the obsolete single-value
    # contract (caused "unexpected result type" on first apply 2026-05-09).
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        log.info("[PN82] applied")
        return "applied", "PN82: is_prefilling[num_reqs:] = False guard injected"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "idempotent (marker present)"
    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "unknown_skip"
        detail = f" — {failure.detail}" if (failure and failure.detail) else ""
        log.info("[PN82] SKIPPED: %s%s", reason, detail)
        return "skipped", f"{reason}{detail}"
    # FAILED
    reason = failure.reason if failure else "unknown_failure"
    detail = f" ({failure.detail})" if (failure and failure.detail) else ""
    log.warning("[PN82] FAILED: %s%s", reason, detail)
    return "failed", f"{reason}{detail}"

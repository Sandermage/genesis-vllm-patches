# SPDX-License-Identifier: Apache-2.0
"""PN347 — vendor of OPEN PR vllm#44113 (shernshiou) MarlinFP8 N==K correctness.

MarlinFP8 weight transpose silently skipped for square (N==K) matrices on sm_75-88
=================================================================================

The bug (and why it is a CORRECTNESS, not a perf, regression)
-------------------------------------------------------------

vLLM PR #38092 added this shape-tuple guard in
``MarlinFP8ScaledMMLinearKernel.process_weights_after_loading``::

    if w_q.shape != (
        layer.input_size_per_partition,
        layer.output_size_per_partition,
    ):
        replace_parameter(layer, "weight", w_q.t())

Intent: skip the transpose when the caller already pre-transposed to ``(K, N)``
(modelopt path); apply the transpose when the caller hands ``(N, K)`` from a
checkpoint (CompressedTensorsW8A16Fp8 / Fp8LinearMethod use_marlin branch).

The bug: for a **square** weight (``N == K``), the tuples ``(N, K)`` and
``(K, N)`` are IDENTICAL — the guard always evaluates ``False`` and the
transpose is silently skipped, regardless of the actual memory layout.
Marlin then multiplies a wrongly-laid-out weight tensor → silent data
corruption in the affected layers' outputs. NO exception, NO log, NO check.
The kernel returns garbage that LOOKS like a valid activation.

Concrete blast radius on our hardware
-------------------------------------

We run 2× RTX A5000 (sm_86 Ampere). MarlinFP8 fires on sm_75-88 (we lack
native FP8 compute → Marlin INT4-style packed kernels emulate). Our
Qwen3.6 27B INT4 and 35B FP8 models both contain square Linear weights in
attention:

  * Qwen3.6 27B: hidden=4096 → q_proj 4096×4096, k_proj/v_proj 4096×4096,
    o_proj 4096×4096 — all square.
  * Qwen3.6 35B FP8: hidden=5120 → q_proj 5120×5120, o_proj 5120×5120 —
    square.

All of these layers, when loaded through the CompressedTensorsW8A16Fp8 or
Fp8LinearMethod use_marlin branch, hand a ``(N, K)`` contiguous tensor
into ``process_weights_after_loading``. Pre-fix, the buggy guard treats
``(N, K)`` and ``(K, N)`` as equal → no transpose → Marlin reads the
weight in the wrong layout.

Upstream test result on A40 (sm_86, identical layout to our A5000)::

    BF16: The capital of France is Paris...
    FP8:  ,,,,,,,,,,,,,,,,,,,,                  ← total token-stream
                                                  collapse on square layers

The exact fix
-------------

Switch from a (broken) shape-tuple compare to a (correct) memory-layout
check: a contiguous tensor is in checkpoint ``(N, K)`` layout and needs
transpose; a non-contiguous tensor is a ``.t()`` view in ``(K, N)``
layout and must NOT be transposed again::

    if w_q.is_contiguous():
        replace_parameter(layer, "weight", w_q.t())

The contiguity flag is robust to square shapes because PyTorch's ``.t()``
on a 2-D tensor returns a non-contiguous view (strides swapped) — even
when shape happens to be symmetric.

Why we vendor an OPEN PR
------------------------

* The PR is OPEN as of 2026-06-09; merge ETA unknown.
* The fix is one method, six lines net, behaviour-preserving on the
  non-buggy path (non-square layers already worked; the new check still
  routes them correctly through the contiguous branch).
* The corruption is silent — we cannot wait. Quality-bench delta on
  square attention layers is unknown without applying the fix.

ROI / risk
----------

ROI: correctness restoration on every square Linear layer in attention
projections (q/k/v/o) of Qwen3.6 27B and 35B on A5000. Expected:
measurable improvement on any quality bench (perplexity, MMLU, tool-call
correctness) currently bottlenecked by hidden FP8 attention corruption.

Risk: LOW. Behaviour preserved on:
  * non-square layers (was-contiguous branch still transposes)
  * modelopt pre-transposed input (was-non-contiguous branch still
    no-ops); modelopt itself is touched by the upstream PR with the same
    semantics — we only vendor the Marlin-kernel side, not the modelopt
    workaround removal, so callers see no change.
  * sm_89+ (Marlin path not taken; FP8 native compute used instead).
  * block-quant branch (entirely separate ``if self.block_quant`` branch).

Composition
-----------

  * No anchor overlap with any existing Genesis Marlin patch
    (``kernels/p87_marlin_pad_sub_tile.py`` exists but targets the Marlin
    INT4 kernel internals, not the FP8 weight-loading hook).
  * Composes with PN77 (FP8 lm_head — different layer), PN81 (FP8
    block-scaled — different branch), PN91/PN91B (INT4 AutoRound —
    different scheme), P87 (Marlin INT4 sub-tile pad — different kernel).
  * CORRECTNESS category — not perf — quality gain is restoration, not
    improvement.

Author: Sander Barzov Aleksandr (Sandermage, Ukraine, Odessa).
Vendor target: vllm-project/vllm#44113 (Closes vllm#44110; OPEN 2026-06-09).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn347_marlin_fp8_nk_correctness")

GENESIS_PN347_MARKER = (
    "Genesis PN347 vendor of vllm#44113 (MarlinFP8 N==K correctness) v1"
)

# Target file in our pin's site-packages layout (dev259+). Verified via
# `ssh ... docker exec ... ls /usr/local/lib/python3.12/dist-packages/vllm/`
# matches a single MarlinFP8 implementation file.
PN347_TARGET_REL = "model_executor/kernels/linear/scaled_mm/marlin.py"


# Anchor: the buggy shape-tuple guard plus the entire surrounding comment
# block. 18 lines of context for unique match + drift detection. The
# anchor is unique in the file: ``w_q, *_ = self._get_layer_params(``
# and ``if w_q.shape != (`` each appear exactly once.
PN347_GUARD_OLD = (
    "            w_q, *_ = self._get_layer_params(layer)\n"
    "            # Compressed tensors transposes the weight to (K, N)\n"
    "            # for channel and tensor quant strategies.\n"
    "            # So we can skip the transpose if the layout is\n"
    "            # already (K, N).\n"
    "            # TODO: Remove this check once the layouts have been\n"
    "            # canonicalized to a standard (N, K) dimension. See issue\n"
    "            # #33314 for more details.\n"
    "            if w_q.shape != (\n"
    "                layer.input_size_per_partition,\n"
    "                layer.output_size_per_partition,\n"
    "            ):\n"
    "                # transpose the weights to (K,N)\n"
    "                replace_parameter(\n"
    "                    layer,\n"
    "                    \"weight\",\n"
    "                    w_q.t(),\n"
    "                )\n"
)

PN347_GUARD_NEW = (
    "            w_q, *_ = self._get_layer_params(layer)\n"
    "            # [Genesis PN347 vendor of vllm#44113] CORRECTNESS FIX\n"
    "            # ----------------------------------------------------\n"
    "            # The shape-tuple guard `w_q.shape != (in, out)` was a\n"
    "            # silent no-op for SQUARE (N==K) weights — both tuples\n"
    "            # `(N, K)` and `(K, N)` are identical when N==K, so the\n"
    "            # transpose was skipped even though the layout was\n"
    "            # wrong. Marlin then multiplied a wrongly-laid-out\n"
    "            # weight tensor → silent data corruption. On our 2×\n"
    "            # A5000 (sm_86) this fires on every square q/k/v/o_proj\n"
    "            # in Qwen3.6 27B (4096²) and 35B (5120²) FP8 attn.\n"
    "            #\n"
    "            # Fix: switch to a memory-layout check. `.t()` on a 2-D\n"
    "            # tensor returns a non-contiguous view (strides swapped)\n"
    "            # regardless of whether the shape happens to be square.\n"
    "            #   - contiguous     → (N, K) from checkpoint → transpose\n"
    "            #   - non-contiguous → (K, N) pre-transposed  → no-op\n"
    "            if w_q.is_contiguous():\n"
    "                # (N, K) from checkpoint -- transpose to (K, N)\n"
    "                replace_parameter(\n"
    "                    layer,\n"
    "                    \"weight\",\n"
    "                    w_q.t(),\n"
    "                )\n"
    "            # else: already (K, N) from caller's pre-transpose -- no-op\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN347", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _make_patcher_for_drift() -> TextPatcher | None:
    """Build PN347's TextPatcher WITHOUT applying it.

    PN347 historically constructed its ``TextPatcher`` inline inside
    ``apply()`` (no module-level ``_make_patcher``), which made it
    invisible to the static anchor-drift detector
    (``tools/check_upstream_drift.py``) — a whole class of inline-builder
    patches was silently dropped. This shim factors the builder out so the
    drift tool can opt in and verify PN347's anchor against an upstream
    clone, while ``apply()`` reuses the exact same patcher (no drift
    between the applied edit and the checked anchor).

    Returns the ``TextPatcher`` or ``None`` when the target file is absent
    in the current vllm tree (e.g. the pin moved/renamed the kernel file —
    a legitimate "needs re-anchor" signal the drift tool surfaces).
    """
    target = resolve_vllm_file(PN347_TARGET_REL)
    if target is None:
        return None

    return TextPatcher(
        patch_name="PN347 MarlinFP8 N==K correctness (vendor of OPEN vllm#44113)",
        target_file=str(target),
        marker=GENESIS_PN347_MARKER,
        sub_patches=[
            TextPatch(
                name="pn347_marlin_fp8_contiguity_guard",
                anchor=PN347_GUARD_OLD,
                replacement=PN347_GUARD_NEW,
                required=True,
            ),
        ],
        # Drift markers — only Genesis-injected sentinel; the upstream
        # PR's `if w_q.is_contiguous():` and the long-standing TODO
        # comment `canonicalized to a standard (N, K)` would both
        # false-positive against the unpatched original file (the TODO
        # has been in upstream since vllm#33314 was filed), so they
        # MUST NOT be used as drift markers.
        upstream_drift_markers=[
            "[Genesis PN347",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN347 — MarlinFP8 (N==K) correctness fix."""
    if _env_disabled():
        return "skipped", "PN347 disabled via GENESIS_DISABLE_PN347=1"

    # Honour the registry version gate BEFORE anchor-matching. PN347's cap
    # (>=0.21.0, <0.22.1rc1.dev491) is correct, but apply() previously fell
    # straight through to the patcher on out-of-window pins (e.g. dev148,
    # where the upstream scaled_mm refactor structurally removed the N==K
    # bug), emitting a noisy per-boot `required_anchor_missing` DRIFT warning
    # instead of a clean VERSION-GATE skip. Mirrors the PN50/PN111 pattern.
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN347")
    log_decision("PN347", decision, reason)
    if not decision:
        return "skipped", reason

    patcher = _make_patcher_for_drift()
    if patcher is None:
        return "skipped", (
            f"PN347: target file not found ({PN347_TARGET_REL}); "
            "pin may have moved the file or this kernel is not present."
        )

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001
        return "failed", f"PN347 apply raised {e!r}"

    if result == TextPatchResult.FAILED:
        return "failed", (
            f"PN347 FAILED — {failure.reason if failure else 'unknown'}"
        )
    if result == TextPatchResult.SKIPPED:
        return "skipped", (
            f"PN347 skipped — {failure.reason if failure else 'unknown'}"
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", (
            "PN347 idempotent (already applied) — MarlinFP8 N==K "
            "correctness fix live in marlin.py."
        )
    return "applied", (
        "PN347 applied — MarlinFP8 `w_q.shape != (in,out)` guard replaced "
        "with `w_q.is_contiguous()` contiguity check. Square (N==K) weights "
        "on sm_75-88 (our A5000 sm_86) now transpose correctly; silent data "
        "corruption in square q/k/v/o_proj attn layers of Qwen3.6 27B + 35B "
        "FP8 is fixed. Vendor of OPEN PR vllm#44113 (Closes vllm#44110). "
        "Composes with PN77 + PN81 + PN91 + PN91B + P87."
    )


def is_applied() -> bool:
    target = resolve_vllm_file(PN347_TARGET_REL)
    if target is None:
        return False
    try:
        return GENESIS_PN347_MARKER in Path(str(target)).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

# SPDX-License-Identifier: Apache-2.0
"""PN133 — MTP scheduler empty-output accounting fix (backport vllm#42722).

================================================================
PROBLEM
================================================================

PR #42722 (OPEN, 2026-05-15) fixes a scheduler bug where MTP/spec
draft tokens are scheduled but model_runner returns an empty
generated_token_ids list.

Current scheduler code:

    if scheduled_spec_token_ids and generated_token_ids:
        num_draft_tokens = len(scheduled_spec_token_ids)
        num_accepted = len(generated_token_ids) - 1
        num_rejected = num_draft_tokens - num_accepted

When `generated_token_ids` is empty:
  - Condition False → scheduler does NOT account rejected draft tokens
  - num_computed_tokens stays caught up with num_tokens_with_spec
  - Scheduler thinks the request made progress
  - But the request is NOT finished → scheduler stops issuing work
  - Request is permanently stuck (unschedulable)

Pre-fix bug: scheduler crash via `len([]) - 1 = -1` →
Prometheus counter ValueError.

Applicable to us?
  - MTP K=3 sync mode — our setup
  - Bench is usually stable, but under error conditions
    (request abortion, async race, model OOM partial output)
    can trigger this stuck-request bug

================================================================
FIX
================================================================

PR diff:

    -if scheduled_spec_token_ids and generated_token_ids:
    +if scheduled_spec_token_ids:
         num_draft_tokens = len(scheduled_spec_token_ids)
    -    num_accepted = len(generated_token_ids) - 1
    +    num_accepted = max(len(generated_token_ids) - 1, 0)
         num_rejected = num_draft_tokens - num_accepted

PN133 backports via runtime monkey-patch on
`Scheduler.update_from_output`.

================================================================
V2 — vllm#45060 OBSERVABILITY ARM (2026-06-11)
================================================================

PR #45060 root-caused the empty-output condition this patch handles:
``sample_recovered_tokens_kernel`` returns an out-of-vocab id
(``recovered_id == vocab_size``) on all-NaN logits when the vocab is
not a multiple of the kernel BLOCK_SIZE (8192 — Qwen's 151936 % 8192
!= 0, so the condition is live on both PROD models). The upstream PR
fixes the kernel AND replaces this scheduler site with
``assert generated_token_ids``.

Per the roadmap (chunk-3 Theme A, 2026-06-11) we vendor the KERNEL
half as Genesis PN378 (separate patch, rejection_sampler.py) and do
NOT take the assert: an all-NaN forward already produced garbage for
one request — killing the whole engine core on top of it converts one
bad request into an outage. v2 instead extends the replacement with a
``logger.error`` arm on the exact invariant the upstream assert
enforces, so the condition is loudly observable in docker logs while
the #42722 accounting fix keeps the request schedulable.

COMPOSITION with PN378: PN378 removes the out-of-vocab source in the
kernel; PN133 v2 keeps accounting correct + logs if the condition
somehow still fires (e.g. PN378 disabled, or a different empty-row
producer). Different files, zero anchor overlap.

================================================================
SAFETY
================================================================

  - Default OFF — opt-in via GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX=1
  - Defensive imports + idempotency
  - Does not raise — failure path = log.warning + fallback
  - log.error arm fires only on the broken-invariant row; zero cost on
    the healthy path (one truthiness test already paid by the old
    ``and generated_token_ids`` condition)

Author: Sandermage 2026-05-15. Backport vllm#42722 (OPEN).
v2 2026-06-11: vllm#45060 observability arm (log.error, NOT the
upstream assert) + #45060 drift marker. Coordinates with PN378.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn133_mtp_scheduler_empty_output")

GENESIS_PN133_MARKER = (
    "Genesis PN133 MTP scheduler empty-output fix v2 "
    "(vllm#42722 + vllm#45060 observability arm)"
)
_ENV_ENABLE = "GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX"
_ENV_DISABLE = "GENESIS_DISABLE_PN133_MTP_EMPTY_OUTPUT_FIX"

_APPLIED = False

# ── Anchor / replacement (hoisted to module constants in v2 for
# testability + pristine-pin byte-verification) ──────────────────────
# Two-part fix (#42722) + observability arm (#45060):
# 1. `if scheduled_spec_token_ids and generated_token_ids:` →
#    `if scheduled_spec_token_ids:`
# 2. `num_accepted = len(generated_token_ids) - 1` →
#    `num_accepted = max(len(generated_token_ids) - 1, 0)`
# 3. v2: log.error on the empty row (the invariant #45060 asserts on)
# Anchor count==1 byte-verified against the pristine pin tree
# (/private/tmp/candidate_pin_current/vllm at g303916e93,
# v1/core/sched/scheduler.py line 1417).

PN133_OLD = (
    "            if scheduled_spec_token_ids and generated_token_ids:\n"
    "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
    "                num_accepted = len(generated_token_ids) - 1\n"
)
PN133_NEW = (
    "            # [Genesis PN133 vllm#42722] empty generated_token_ids\n"
    "            # must still account scheduled draft tokens as rejected,\n"
    "            # otherwise request can become permanently unschedulable.\n"
    "            if scheduled_spec_token_ids:\n"
    "                if not generated_token_ids:\n"
    "                    # [Genesis PN133 v2] vllm#45060 observability\n"
    "                    # arm: a scheduled-spec request that commits\n"
    "                    # zero tokens means the rejection sampler\n"
    "                    # emitted only ids parse_output dropped\n"
    "                    # (out-of-vocab recovered id on all-NaN logits\n"
    "                    # — the source Genesis PN378 masks in the\n"
    "                    # kernel). Upstream asserts here; PROD must\n"
    "                    # degrade loudly, not crash the engine core.\n"
    "                    logger.error(\n"
    '                        "[Genesis PN133] request %s: %d spec "\n'
    '                        "tokens scheduled but no tokens committed "\n'
    '                        "— empty sampled row (all-NaN logits / "\n'
    '                        "out-of-vocab recovered id; see vllm#45060 "\n'
    '                        "and Genesis PN378). Accounting all drafts "\n'
    '                        "as rejected.",\n'
    "                        req_id,\n"
    "                        len(scheduled_spec_token_ids),\n"
    "                    )\n"
    "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
    "                num_accepted = max(len(generated_token_ids) - 1, 0)\n"
)

# Drift markers — BOTH upstream forms of this scheduler site:
#   * #42722's accounting fix (the form we backport). Substring of our
#     own replacement BY DESIGN — defended below by the
#     `"PN133" not in content` guard (custom apply(), not enumerated by
#     tools/lint_drift_markers.py builder discovery).
#   * #45060's scheduler half (exact indented line from
#     `gh pr diff 45060`, 2026-06-11). NOT emitted by us; if upstream
#     lands the assert, the anchor is gone and PN133 self-retires
#     loudly instead of failing the anchor scan.
_DRIFT_MARKERS = (
    # Operand-agnostic (2026-06-14): dev491 merged vllm#42722 but emits
    # `max(len(generated_token_ids) - num_sampled, 0)` (scheduler.py:1549),
    # while the <dev491 form was `- 1`. Match the stable invariant — the
    # `max(..., 0)` clamp — regardless of the subtrahend, so PN133 self-
    # retires cleanly as upstream-merged on dev491 instead of falling through
    # to a generic `required_anchor_missing` DRIFT warning. Still a substring
    # of PN133_NEW BY DESIGN; the `"PN133" not in content` guard in apply()
    # prevents self-detection of our own injected marker.
    "max(len(generated_token_ids) - ",
    "                assert generated_token_ids\n",
)


def _env_enabled() -> bool:
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    val = os.environ.get(_ENV_ENABLE, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """Text-patch on scheduler.py — fix empty generated_token_ids accounting."""
    global _APPLIED

    if not _env_enabled():
        return "skipped", (
            f"PN133 disabled (set {_ENV_ENABLE}=1 — backport vllm#42722 "
            f"MTP scheduler stuck-request fix when generated_token_ids "
            f"empty)"
        )

    if _APPLIED:
        return "applied", "PN133 already installed (idempotent)"

    try:
        from sndr.engines.vllm.detection.guards import resolve_vllm_file
        from sndr.kernel import TextPatcher, TextPatch
    except ImportError as e:
        return "skipped", f"genesis core not importable: {e}"

    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return "skipped", "scheduler.py not resolvable"

    patcher = TextPatcher(
        patch_name=(
            "PN133 scheduler.py — MTP empty-output accounting "
            "(vllm#42722) + vllm#45060 observability arm"
        ),
        target_file=str(target),
        marker="[Genesis PN133 vllm#42722]",
        sub_patches=[
            TextPatch(
                name="pn133_empty_output_fix",
                anchor=PN133_OLD,
                replacement=PN133_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target file missing: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        _APPLIED = True
        return "applied", "PN133 marker present — idempotent skip"
    for m in patcher.upstream_drift_markers:
        if m in content and "PN133" not in content:
            log.info("[PN133] upstream lands the fix — self-retire")
            return "skipped", f"upstream_merged — drift marker {m!r} present"

    result, failure = patcher.apply()
    from sndr.kernel import result_to_wiring_status
    status, reason = result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN133 installed: scheduler MTP empty-output fix wired "
            "(vllm#42722 backport) + v2 log.error observability arm on "
            "the empty-row invariant (vllm#45060 scheduler half, "
            "demoted from assert; kernel half = Genesis PN378). Fixes "
            "permanently-stuck-request bug when model_runner returns "
            "empty generated_token_ids with scheduled draft tokens."
        ),
        patch_name=patcher.patch_name,
    )
    if status == "applied":
        _APPLIED = True
    return status, reason


def is_applied() -> bool:
    return _APPLIED

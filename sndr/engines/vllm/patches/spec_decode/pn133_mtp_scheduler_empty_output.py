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
SAFETY
================================================================

  - Default OFF — opt-in via GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX=1
  - Defensive imports + idempotency
  - Does not raise — failure path = log.warning + fallback

Author: Sandermage 2026-05-15. Backport vllm#42722 (OPEN).
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.wiring.pn133_mtp_scheduler_empty_output")

GENESIS_PN133_MARKER = "Genesis PN133 MTP scheduler empty-output fix v1 (vllm#42722)"
_ENV_ENABLE = "GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX"
_ENV_DISABLE = "GENESIS_DISABLE_PN133_MTP_EMPTY_OUTPUT_FIX"

_APPLIED = False


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
        from vllm.sndr_core.detection.guards import resolve_vllm_file
        from vllm.sndr_core.core import TextPatcher, TextPatch
    except ImportError as e:
        return "skipped", f"genesis core not importable: {e}"

    target = resolve_vllm_file("v1/core/sched/scheduler.py")
    if target is None:
        return "skipped", "scheduler.py not resolvable"

    # Two-part fix:
    # 1. `if scheduled_spec_token_ids and generated_token_ids:` →
    #    `if scheduled_spec_token_ids:`
    # 2. `num_accepted = len(generated_token_ids) - 1` →
    #    `num_accepted = max(len(generated_token_ids) - 1, 0)`

    OLD = (
        "            if scheduled_spec_token_ids and generated_token_ids:\n"
        "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
        "                num_accepted = len(generated_token_ids) - 1\n"
    )
    NEW = (
        "            # [Genesis PN133 vllm#42722] empty generated_token_ids\n"
        "            # must still account scheduled draft tokens as rejected,\n"
        "            # otherwise request can become permanently unschedulable.\n"
        "            if scheduled_spec_token_ids:\n"
        "                num_draft_tokens = len(scheduled_spec_token_ids)\n"
        "                num_accepted = max(len(generated_token_ids) - 1, 0)\n"
    )

    patcher = TextPatcher(
        patch_name="PN133 scheduler.py — MTP empty-output accounting (vllm#42722)",
        target_file=str(target),
        marker="[Genesis PN133 vllm#42722]",
        sub_patches=[
            TextPatch(
                name="pn133_empty_output_fix",
                anchor=OLD,
                replacement=NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "max(len(generated_token_ids) - 1, 0)",  # if upstream lands
        ],
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
    from vllm.sndr_core.core import result_to_wiring_status
    status, reason = result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN133 installed: scheduler MTP empty-output fix wired "
            "(vllm#42722 backport). Fixes permanently-stuck-request bug "
            "when model_runner returns empty generated_token_ids with "
            "scheduled draft tokens."
        ),
        patch_name=patcher.patch_name,
    )
    if status == "applied":
        _APPLIED = True
    return status, reason


def is_applied() -> bool:
    return _APPLIED

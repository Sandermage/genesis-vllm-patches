# SPDX-License-Identifier: Apache-2.0
"""PN381 — vendor OPEN PR vllm#44742 (allowed_token_ids metadata
hardening for spec-decode), PN67-playbook style.

Upstream PR `vllm-project/vllm#44742`_ fixes GHSA-8c65-hq7q-r7jm:
when a request sets ONLY ``allowed_token_ids`` (no penalties, no bad
words, no thinking budget), ``InputBatch._make_sampling_metadata``
ships ``output_token_ids == []`` while ``allowed_token_ids_mask`` and
the speculative draft-token counts are non-empty. Any consumer that
derives the request count from ``len(output_token_ids)`` then
mis-expands the mask rows during draft verification — the 0.17.1
worker crash in the advisory.

Single anchored sub-patch on ``v1/worker/gpu_input_batch.py``
``_make_sampling_metadata`` (anchor byte-verified count==1 on pin
0.22.1rc1.dev259+g303916e93): add the ``allowed_token_ids`` clause to
``needs_output_token_ids`` so the metadata row counts stay
self-consistent whenever the mask is in play.

================================================================
WHY VENDOR — THE CONSUMER FIX IS ALREADY IN THE PIN
================================================================

Our pin's ``RejectionSampler.apply_logits_processors`` already sizes
the draft expansion by ``len(metadata.num_draft_tokens)`` (#35654), so
TODAY no in-tree consumer trips over the empty list. PN381 is
deliberate defense-in-depth (roadmap chunk 5, 2026-06-11): the
PN369/P71 rewritten rejection-sampler paths and any future logits
processor consume the SAME SamplingMetadata — a producer that emits
internally-inconsistent metadata (mask rows present, output rows
absent) is a row-parity landmine for every consumer we add. Populate
once at the producer; every consumer inherits consistency.

Cost analysis (iron rule #10 — adapt, don't blind-copy): the clause
only flips ``needs_output_token_ids`` when ``no_allowed_token_ids`` is
False, i.e. when at least one live request actually set
``allowed_token_ids``. ``req_output_token_ids`` is a list of per-request
list REFERENCES (no tensor copy, no D2H) — the upstream fast path for
the common no-mask case is untouched.

================================================================
DRIFT-MARKER DISCIPLINE (lint_drift_markers contract)
================================================================

Our emitted text parenthesizes the clause —
``or (not self.no_allowed_token_ids)`` — so the drift marker for the
PR's merged form (``or not self.no_allowed_token_ids``, no parens) can
NEVER match our own output. When #44742 merges at a future pin, the
marker fires and the patcher self-skips (Layer 3) instead of
double-adding the clause beside upstream's.

================================================================
RELATIONSHIP TO OTHER GENESIS PATCHES
================================================================

- Retired PN67 (vllm#41674) patched the SAME ``needs_output_token_ids``
  block — its fix (``or not thinking_budget_tracks_reqs`` -> ``or
  thinking_budget_tracks_reqs``) merged upstream at dev371 and IS the
  pristine shape our anchor encodes. PN381 is the same playbook, one
  clause further.
- PN369 / P71 patch ``v1/sample/rejection_sampler.py`` (different
  file) — PN381 protects the metadata they consume. The ported #44742
  regression test exercises allowed_token_ids + draft tokens through
  the PN369/P71-patched module
  (tests/unit/integrations/spec_decode/test_pn381_sampler_regression_torch.py).
- No other active patch anchors this region (PN52 retired-era
  neighborhood is gone from the pin).

================================================================
SAFETY MODEL
================================================================

- Opt-in: ``GENESIS_ENABLE_PN381_ALLOWED_TOKEN_IDS_METADATA=1``
  (default OFF).
- Anchor missing (upstream rewrote the region) -> SKIPPED with reason.
- Merged form already present -> drift-marker SKIP (no double-apply).
- Behavior change is metadata-only: NULL on requests that never set
  ``allowed_token_ids`` (the entire Genesis PROD workload today);
  populated-vs-empty ``output_token_ids`` only changes which rows the
  consumers may index — never the sampled tokens themselves.

Expected effect: zero perf either way; removes the row-parity
silent-corruption class for masked + speculative batches (the
GHSA-8c65-hq7q-r7jm shape) from every present and future consumer of
SamplingMetadata.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Upstream: https://github.com/vllm-project/vllm/pull/44742 (OPEN at
vendor time, 2026-06-11).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn381_allowed_token_ids_spec_metadata")

GENESIS_PN381_MARKER = (
    "Genesis PN381 vendor of vllm#44742 (allowed_token_ids metadata hardening) v1"
)

_INPUT_BATCH_REL = "v1/worker/gpu_input_batch.py"

# Fires when vllm#44742 merges. The PR adds the UNparenthesized form;
# our replacement deliberately emits ``or (not self.no_allowed_token_ids)``
# so this marker is disjoint from our own output
# (tools/lint_drift_markers.py contract; asserted in tests).
_DRIFT_MARKERS = (
    "[Genesis PN381",
    "or not self.no_allowed_token_ids",
)


# ─── Single sub-patch: needs_output_token_ids += allowed_token_ids ────
# Anchor: the full pristine condition block (post-#41674 shape — the
# retired PN67 fix is merged upstream at our pin). Unique count==1 on
# pin g303916e93.
PN381_OLD = (
    "        needs_output_token_ids = (\n"
    "            not self.no_penalties\n"
    "            or bool(self.bad_words_token_ids)\n"
    "            or self.logitsprocs_need_output_token_ids\n"
    "            or thinking_budget_tracks_reqs\n"
    "        )\n"
)
PN381_NEW = (
    "        needs_output_token_ids = (\n"
    "            not self.no_penalties\n"
    "            or bool(self.bad_words_token_ids)\n"
    "            or self.logitsprocs_need_output_token_ids\n"
    "            or thinking_budget_tracks_reqs\n"
    "            # [Genesis PN381 vendor of vllm#44742] GHSA-8c65-hq7q-r7jm:\n"
    "            # when any request uses allowed_token_ids, populate\n"
    "            # output_token_ids so SamplingMetadata stays row-consistent\n"
    "            # with allowed_token_ids_mask during speculative decoding\n"
    "            # (consumers must never see mask rows without output rows).\n"
    "            # Parenthesized on purpose: the unparenthesized upstream\n"
    "            # form is our merge-detection drift marker.\n"
    "            or (not self.no_allowed_token_ids)\n"
    "        )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_INPUT_BATCH_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN381 v1/worker/gpu_input_batch.py — allowed_token_ids "
            "metadata hardening for spec-decode (vendor vllm#44742)"
        ),
        target_file=str(target),
        marker=GENESIS_PN381_MARKER,
        sub_patches=[
            TextPatch(
                name="pn381_needs_output_token_ids_allowed_clause",
                anchor=PN381_OLD,
                replacement=PN381_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_DRIFT_MARKERS),
    )


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN381_ALLOWED_TOKEN_IDS_METADATA", ""
    ).strip().lower() in ("1", "true", "yes", "on")


def apply() -> tuple[str, str]:
    """Apply PN381 — vendor vllm#44742. Never raises."""
    if not _enabled():
        return "skipped", (
            "PN381 default OFF — set "
            "GENESIS_ENABLE_PN381_ALLOWED_TOKEN_IDS_METADATA=1 to engage. "
            "Producer-side allowed_token_ids metadata hardening for "
            "spec-decode (vendor of OPEN PR vllm#44742, "
            "GHSA-8c65-hq7q-r7jm defense-in-depth)."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN381: {_INPUT_BATCH_REL} not resolvable"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"PN381: target disappeared: {patcher.target_file}"

    try:
        result, failure = patcher.apply()
    except Exception as e:  # noqa: BLE001 — wiring must never raise
        log.warning("[PN381] apply() raised %s — leaving upstream", e)
        return "skipped", f"PN381 raised at apply: {e!r}"

    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN381 already applied (marker present, idempotent)"
    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "anchor drift / not eligible"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "skipped", f"PN381: {reason}{detail}"
    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "failed", f"PN381: {reason}{detail}"

    return "applied", (
        "PN381 applied (vendor of OPEN PR vllm#44742): "
        "_make_sampling_metadata now populates output_token_ids whenever "
        "any request uses allowed_token_ids, keeping SamplingMetadata "
        "row-consistent with allowed_token_ids_mask under speculative "
        "decoding (GHSA-8c65-hq7q-r7jm producer-side hardening; the pin's "
        "consumer fix #35654 stays as the second layer). NULL on requests "
        "without allowed_token_ids — the upstream fast path is untouched."
    )


def is_applied() -> bool:
    target = resolve_vllm_file(_INPUT_BATCH_REL)
    if target is None:
        return False
    try:
        return GENESIS_PN381_MARKER in open(
            str(target), encoding="utf-8"
        ).read()
    except (OSError, UnicodeDecodeError):
        return False

# SPDX-License-Identifier: Apache-2.0
"""Wiring for P61c — qwen3coder deferred-commit until <function= header.

Closes club-3090 issue #72 (troymroberts 2026-05-06).

================================================================
Bug
---
`vllm/tool_parsers/qwen3coder_tool_parser.py::extract_tool_calls_streaming`
flips `is_tool_call_started=True` permanently on EITHER:

  1. `tool_call_start_token_id` in `delta_token_ids` (tokenizer special)
  2. `tool_call_start_token` in `delta_text` (string match `<tool_call>`)

Both trigger paths mis-fire when the model emits `<tool_call>` in
narrative output — as the special token when the BPE tokenizer
recognizes the tag, OR as the string when the tag appears in
markdown / prose contexts (e.g. agent reasoning that *describes* the
tool-call protocol). The flip is sticky: subsequent deltas all return
`None` and vLLM's serving layer skips emitting `None` deltas
(`continue` in `serving.py:914-924`), so the SSE wire goes silent
until `max_tokens` ends the request. Server-side decoded tokens
never reach the client (30-120+ s of zero-chunk silence).

Fix
---
Defer commit to `is_tool_call_started=True` until `<function=`
appears in a 64-char slack window after `<tool_call>`. Real tool
calls in qwen3coder format have `<tool_call>\\n<function=...`
adjacency; if no `<function=` arrives within 64 chars past the tag,
treat the `<tool_call>` mention as benign content and continue
streaming. Both trigger paths (token-id and string) flow through the
same deferred check.

Three logical paths after detecting trigger:
  A) token-id present but tag string not in `current_text` (tokenizer
     edge case) → emit delta as content, don't commit
  B) `<function=` confirmed in slack → commit (set flag) + return
     None (preserves original behavior)
  C) no `<function=` yet → emit delta as content (never silently
     drop chunks while uncertain)

Compatibility
-------------
- **P64** (vllm#39598 backport — qwen3coder MTP streaming) — operates
  in the same file but in different sub-blocks (the `</function>`
  branch and serving.py). P61c anchor is in the *commit-trigger*
  block at the top of `extract_tool_calls_streaming`. No conflict.
- **PN56** (vllm#41466 backport — XML parse fallback) — operates in
  `_parse_xml_function_call` try/except. No conflict.
- **P61** (vllm#33041 backport — multi-tool first occurrence) — fixes
  detection when multiple `<tool_call>` blocks present. P61c covers
  the *opposite* failure: false positive without any real tool call.
  Complementary, both should land together.

Status: opt-in via `GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT=1`.
Default OFF until live verify on 27B PROD streaming workload.

Idempotent (sentinel marker check). Auto-no-op on upstream merge
(drift markers `_tc_idx`, `_slack_end`).

Risks acknowledged
------------------
- 64-char slack is heuristic. If a future Qwen3 variant inserts more
  whitespace/comments between `<tool_call>` and `<function=`, real
  tool calls may not commit — symptom would be content streaming
  instead of tool_call extraction. Mitigation: drift marker
  triggers re-evaluation on upstream pin bump.
- Loss of optimization: token-id-only matches now incur a substring
  search per delta. Cost: ~O(64) chars × frequency of false `<tool_call>`
  mentions, dominated by Python overhead — negligible at our scale.

Author: Sandermage backport (troymroberts club-3090#72, V2 deferred sketch).
Sandermage Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from vllm.sndr_core.detection.guards import resolve_vllm_file, vllm_install_root
from vllm.sndr_core.core import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p61c_qwen3coder_deferred_commit")

GENESIS_P61C_MARKER = "Genesis P61c qwen3coder deferred-commit (club-3090#72)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor: the 12-line commit-trigger block at the top of
# `extract_tool_calls_streaming`. Indentation is 12 spaces (3 levels
# inside the method). Pristine upstream (vllm 0.20.2rc1.dev9+g01d4d1ad3).
ANCHOR_OLD = (
    "            if (\n"
    "                self.tool_call_start_token_id in delta_token_ids\n"
    "                or self.tool_call_start_token in delta_text\n"
    "            ):\n"
    "                self.is_tool_call_started = True\n"
    "                # Return any content before the tool call\n"
    "                if self.tool_call_start_token in delta_text:\n"
    "                    content_before = delta_text[\n"
    "                        : delta_text.index(self.tool_call_start_token)\n"
    "                    ]\n"
    "                    if content_before:\n"
    "                        return DeltaMessage(content=content_before)\n"
    "                return None\n"
)

ANCHOR_NEW = (
    "            # [Genesis P61c club-3090#72] Defer commit until <function=\n"
    "            # header follows the <tool_call> token within a 64-char slack\n"
    "            # window. Guards against the model emitting <tool_call> (as\n"
    "            # special token OR string) in narrative reasoning without an\n"
    "            # actual tool-call header — a case that flips\n"
    "            # is_tool_call_started=True permanently and silently drops\n"
    "            # all subsequent content via the serving layer's\n"
    "            # `if delta_message is None: continue` path. Three paths:\n"
    "            #   A) token-id present but tag string not in current_text\n"
    "            #      (tokenizer edge case) → emit content, don't commit\n"
    "            #   B) <function= confirmed in slack → original commit path\n"
    "            #   C) no <function= yet → emit content, never silently drop\n"
    "            if (\n"
    "                self.tool_call_start_token_id in delta_token_ids\n"
    "                or self.tool_call_start_token in delta_text\n"
    "            ):\n"
    "                _tc_idx = current_text.find(self.tool_call_start_token)\n"
    "                if _tc_idx == -1:\n"
    "                    # Path A: token-id-only, conservative emit-as-content\n"
    "                    return DeltaMessage(content=delta_text or None)\n"
    "                _slack_end = (\n"
    "                    _tc_idx + len(self.tool_call_start_token) + 64\n"
    "                )\n"
    '                if "<function=" in current_text[_tc_idx:_slack_end]:\n'
    "                    # Path B: real tool call confirmed; original commit\n"
    "                    self.is_tool_call_started = True\n"
    "                    if self.tool_call_start_token in delta_text:\n"
    "                        content_before = delta_text[\n"
    "                            : delta_text.index(self.tool_call_start_token)\n"
    "                        ]\n"
    "                        if content_before:\n"
    "                            return DeltaMessage(content=content_before)\n"
    "                    return None\n"
    "                # Path C: <function= not yet present; emit delta as\n"
    "                # content. If <function= eventually arrives within slack\n"
    "                # we'll commit then; otherwise we've correctly streamed\n"
    "                # the prose containing literal <tool_call> mention.\n"
    "                return DeltaMessage(content=delta_text or None)\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("tool_parsers/qwen3coder_tool_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P61c qwen3coder deferred-commit (club-3090#72)",
        target_file=str(target),
        marker=GENESIS_P61C_MARKER,
        sub_patches=[TextPatch(
            name="p61c_deferred_commit",
            anchor=ANCHOR_OLD,
            replacement=ANCHOR_NEW,
            required=True,
        )],
        upstream_drift_markers=[
            # `_tc_idx` and `_slack_end` are our own variable names — would
            # only appear here in pristine source if upstream coincidentally
            # lands an equivalent fix using same names (unlikely). Safe to
            # use as drift indicators. (DO NOT add `is_tool_call_started`
            # here — it's an existing variable in pristine source, would
            # fire drift on every boot.)
            "_tc_idx",
            "_slack_end",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P61c")
    log_decision("P61c", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "qwen3coder_tool_parser.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "P61c applied: SSE silence on narrative <tool_call> mentions fixed; "
            "commit deferred until <function= confirms within 64-char slack",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return ("skipped",
                f"{msg} — likely upstream merged equivalent or block restructured")
    return "failed", failure.reason if failure else "unknown failure"

# SPDX-License-Identifier: Apache-2.0
"""Wiring for P107 — vllm#41467 backport: MTP truncation detector.

With MTP K>=1 + tools + reasoning parser there is a rare (~0.25% per
author measurement on Qwen3.6-27B-FP8) condition: the model emits EOS
at the reasoning->tool_call boundary. finish_reason=stop + tools
configured + neither tool_calls nor content delivered → silent
client-side error.

Fix: defensive guard in `chat_completion_stream_generator` that
detects this combo and raises GenerationError (retryable) instead of
emitting a silent stop. The client receives an SSE error event and
retries.

Directly affects our PROD: 27B Lorbus + MTP K=3 + tools — exactly
the config the author reported. P107 is a safety net on top of an
already-defended path (P59/P60/P61/P62/P64/P68/P69 family).

Default OFF — defensive backport. Risk: low; adds one extra branch
in the hot path, no effect on the happy path.

Author: Sandermage backport (ToastyTheBot, vllm#41467).
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

log = logging.getLogger("genesis.wiring.p107_mtp_truncation_detector")

GENESIS_P107_MARKER = "Genesis P107 MTP truncation detector (vllm#41467)"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor on stable finish_reason_ if/else block (lines 1084-1093 pristine)
ANCHOR_OLD = (
    "                        if (\n"
    "                            auto_tools_called\n"
    "                            or (tools_streamed[i] and not tool_choice_function_name)\n"
    "                            or (self.use_harmony and harmony_tools_streamed[i])\n"
    "                        ):\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )\n"
    "                        choice_data = ChatCompletionResponseStreamChoice("
)

ANCHOR_NEW = (
    "                        if (\n"
    "                            auto_tools_called\n"
    "                            or (tools_streamed[i] and not tool_choice_function_name)\n"
    "                            or (self.use_harmony and harmony_tools_streamed[i])\n"
    "                        ):\n"
    "                            finish_reason_ = \"tool_calls\"\n"
    "                        else:\n"
    "                            finish_reason_ = (\n"
    "                                output.finish_reason if output.finish_reason else \"stop\"\n"
    "                            )\n"
    "\n"
    "                        # [Genesis P107 vllm#41467] MTP truncation detector.\n"
    "                        # ~0.25% rate on Qwen3.6 27B-FP8 + MTP K=3 (per upstream\n"
    "                        # author): EOS at reasoning→tool_call boundary leaves\n"
    "                        # finish_reason=stop with no content/tool_calls. Raise\n"
    "                        # retryable error so client retries instead of seeing\n"
    "                        # silent empty response. Defensive: 6 AND-conditions,\n"
    "                        # no impact on happy path.\n"
    "                        if (\n"
    "                            finish_reason_ == \"stop\"\n"
    "                            and request.tools\n"
    "                            and not tools_streamed[i]\n"
    "                            and not auto_tools_called\n"
    "                            and reasoning_parser is not None\n"
    "                            and delta_message is not None\n"
    "                            and not delta_message.content\n"
    "                            and not delta_message.tool_calls\n"
    "                        ):\n"
    "                            from vllm.entrypoints.openai.chat_completion.protocol import GenerationError as _P107_GenError\n"
    "                            logger.warning(\n"
    "                                \"[Genesis P107] MTP truncation detected for request %s: \"\n"
    "                                \"finished with 'stop' but tools configured and only \"\n"
    "                                \"reasoning produced.\",\n"
    "                                request_id,\n"
    "                            )\n"
    "                            raise _P107_GenError(\n"
    "                                \"MTP speculative decoding truncated tool call \"\n"
    "                                \"generation. Please retry.\"\n"
    "                            )\n"
    "                        choice_data = ChatCompletionResponseStreamChoice("
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(
        "entrypoints/openai/chat_completion/serving.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name="P107 MTP truncation detector (vllm#41467)",
        target_file=str(target),
        marker=GENESIS_P107_MARKER,
        sub_patches=[TextPatch(
            name="p107_mtp_truncation",
            anchor=ANCHOR_OLD,
            replacement=ANCHOR_NEW,
            required=True,
        )],
        upstream_drift_markers=[
            "MTP truncation detected",
            "MTP speculative decoding truncated",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm.sndr_core.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P107")
    log_decision("P107", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "serving.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return "applied", "P107 applied: MTP truncation now raises retryable error"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return "skipped", f"{msg} — likely upstream merged"
    return "failed", failure.reason if failure else "unknown failure"

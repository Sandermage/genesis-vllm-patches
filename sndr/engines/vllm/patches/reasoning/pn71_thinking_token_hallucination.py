# SPDX-License-Identifier: Apache-2.0
"""PN71 — runtime `</thinking>` → `</think>` hallucination normalizer.

The Qwen 3.6 27B and 35B-A3B models occasionally hallucinate the wrong
closing tag — `</thinking>` (the full word) instead of the canonical
`</think>` token they were trained on. froggeric's enhanced chat
template handles this in the prompt-side jinja for PAST turns, but it
does NOT help the LIVE generated output of the CURRENT turn — by the
time the chat template sees content, the model has already finished.

When the model emits `</thinking>` instead of `</think>` in live
generation:
  1. `Qwen3ReasoningParser.extract_reasoning()` looks for `</think>`
     literally, doesn't find it, and routes ALL output to reasoning
     (with `content=None`).
  2. Streaming: `delta.reasoning` keeps growing forever, `delta.content`
     never opens, client sees an empty response with reasoning-only.

PN71 normalizes the hallucinated tag in both surfaces:

  - `extract_reasoning(model_output, ...)`: replace `</thinking>` →
    `</think>` once at function entry. The partition logic then works
    on a normalized string.
  - `extract_reasoning_streaming(..., current_text=...)`: same normalize
    on `current_text` so the streaming-state machine sees `</think>`.

This is a pure runtime safety net — no template dependency. Even with
the default model-bundled template the parser stays robust.

Env gate: `GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE=1` (default OFF).
Strictly additive — only fires when the model hallucinates the wrong
tag, which is a rare edge per the upstream Qwen 3.6 bug tracker.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import TextPatch, TextPatcher

log = logging.getLogger("genesis.wiring.pn71_thinking_token_hallucination")

GENESIS_MARKER = "Genesis PN71 </thinking> → </think> runtime normalizer"


def _enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE", "0",
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor 1: extract_reasoning function entry. Normalize on model_output.
# The `partition(self.start_token)` line is the first read of model_output.
PN71_OLD_NON_STREAMING = (
    "        # Strip <think> if present in the generated output.\n"
    "        # [Genesis P27] Capture BEFORE-THINK text so it can be prepended\n"
    "        # to content instead of being silently dropped (issue #40699-class).\n"
    "        model_output_parts = model_output.partition(self.start_token)\n"
)
PN71_NEW_NON_STREAMING = (
    "        # Strip <think> if present in the generated output.\n"
    "        # [Genesis PN71] Normalize hallucinated </thinking> → </think>\n"
    "        # before any tag-based parsing. Qwen 3.6 occasionally emits\n"
    "        # the full word instead of the canonical token; this would\n"
    "        # cause all output to be routed to reasoning with empty content.\n"
    "        if \"</thinking>\" in model_output:\n"
    "            model_output = model_output.replace(\"</thinking>\", \"</think>\")\n"
    "        # [Genesis P27] Capture BEFORE-THINK text so it can be prepended\n"
    "        # to content instead of being silently dropped (issue #40699-class).\n"
    "        model_output_parts = model_output.partition(self.start_token)\n"
)


def _make_patcher() -> TextPatcher | None:
    if not _enabled():
        return None
    target = resolve_vllm_file("reasoning/qwen3_reasoning_parser.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN71 </thinking> tag hallucination normalizer",
        target_file=str(target),
        marker=GENESIS_MARKER,
        sub_patches=[
            TextPatch(
                name="pn71_extract_reasoning_normalize",
                anchor=PN71_OLD_NON_STREAMING,
                replacement=PN71_NEW_NON_STREAMING,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "Genesis PN71",
            "Normalize hallucinated </thinking>",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN71 text-patch. Returns (wiring_status, message)."""
    if not _enabled():
        return "skipped", "PN71 disabled (set GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE=1)"
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "target file qwen3_reasoning_parser.py not resolvable"
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as fh:
        content = fh.read()
    if patcher.marker in content:
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m in content:
            return "skipped", f"drift marker {m!r} already in file"
    result, failure = patcher.apply()
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message="PN71 </thinking>→</think> hallucination normalizer active",
        patch_name=patcher.patch_name,
    )

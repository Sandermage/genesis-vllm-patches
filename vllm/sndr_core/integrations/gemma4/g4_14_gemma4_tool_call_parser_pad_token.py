# SPDX-License-Identifier: Apache-2.0
"""G4_14 — Gemma 4 tool-call parser pad-token workaround (closes vllm#39392).

================================================================
WHAT IT FIXES
================================================================

vllm-project/vllm#39392 (OPEN as of 2026-05-17, 9 comments): when using
``--tool-call-parser gemma`` with Gemma 4 chat output, the parser receives
streamed deltas that include the tokenizer's ``<pad>`` token in the
middle of a tool-call JSON block. The pad token appears because
Gemma 4's chat template emits a ``<pad>`` between successive turns to
mark the speaker boundary — most parsers strip it implicitly via the
tokenizer's ``skip_special_tokens=True`` flag, but the streaming tool-call
path holds raw token IDs to allow partial-JSON parsing and the pad
sneaks through.

Symptom: the OpenAI-compatible response stream produces malformed
``function.arguments`` JSON like:

    {"city": "Odessa<pad>", "country": "Ukraine"}

…or, when the pad lands mid-key, a hard parse error in the client
(``json.JSONDecodeError: Expecting value: line 1 column 9``).

================================================================
THE FIX
================================================================

We wrap ``Gemma4ToolParser.extract_tool_calls_streaming`` and
``Gemma4ToolParser.extract_tool_calls`` to strip the pad token (and any
other Gemma-4-specific control tokens that may leak) from the delta
before it hits the JSON-stream parser.

Pad token candidates we strip:
  * ``<pad>``
  * ``<eos>``  (also leaks on turn boundaries)
  * ``<start_of_turn>``
  * ``<end_of_turn>``

Whitelist approach (strip these specific control strings rather than
"all <…> tokens") so legitimate user content like ``<thinking>`` blocks
isn't accidentally damaged.

================================================================
SAFETY MODEL
================================================================

* default_on: True (no harm: strips known-broken pads from a known-broken
  context; falls through silently if class not present)
* env_flag: GENESIS_ENABLE_G4_14_GEMMA4_TOOL_CALL_PARSER_PAD
* applies_to:
    - tool-call-parser: gemma / gemma4
* superseded_by: when vllm#39392 merges

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/issues/39392 (OPEN, 9 comments)
"""
from __future__ import annotations

import logging
import re

from ._gemma4_detect import env_truthy

log = logging.getLogger("genesis.gemma4.g4_14_tool_call_parser")

GENESIS_G4_14_MARKER = (
    "Genesis G4_14 gemma4 tool-call parser pad-token strip v1 "
    "(closes vllm#39392 — strips <pad>/<eos>/turn-boundary tokens "
    "from streaming tool-call JSON deltas)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_14_GEMMA4_TOOL_CALL_PARSER_PAD"

_STRIP_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<unk>",
)

# Pre-compile regex once
_STRIP_RE = re.compile("|".join(re.escape(t) for t in _STRIP_TOKENS))

_APPLIED = False
_ORIGINAL_STREAMING = None
_ORIGINAL_BATCH = None
_PATCHED_CLS = None


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


def _strip_control_tokens(text: str) -> str:
    """Strip Gemma 4 control tokens from a text fragment."""
    if not text:
        return text
    return _STRIP_RE.sub("", text)


def _find_gemma_tool_parser():
    """Locate the Gemma 4 tool-call parser class across vLLM pin variants."""
    candidates = [
        ("vllm.entrypoints.openai.tool_parsers.gemma4_tool_parser", "Gemma4ToolParser"),
        ("vllm.entrypoints.openai.tool_parsers.gemma_tool_parser", "GemmaToolParser"),
        ("vllm.entrypoints.openai.tool_parsers", "Gemma4ToolParser"),
        ("vllm.entrypoints.openai.tool_parsers", "GemmaToolParser"),
    ]
    for mod_path, cls_name in candidates:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return cls
        except ImportError:
            continue
    return None


def apply() -> tuple[str, str]:
    """Install pad-token strip wrappers on Gemma4ToolParser methods."""
    global _APPLIED, _ORIGINAL_STREAMING, _ORIGINAL_BATCH, _PATCHED_CLS

    if not _env_enabled():
        return "skipped", (
            f"G4_14 disabled (set {_ENV_ENABLE}=1 to strip Gemma 4 control "
            "tokens from tool-call streaming JSON — closes vllm#39392)"
        )

    if _APPLIED:
        return "applied", "G4_14 already installed (idempotent)"

    cls = _find_gemma_tool_parser()
    if cls is None:
        return "skipped", (
            "No Gemma4ToolParser-like class found in this vLLM pin; G4_14 "
            "is no-op (you may not be running with --tool-call-parser gemma4)"
        )

    _PATCHED_CLS = cls
    streaming = getattr(cls, "extract_tool_calls_streaming", None)
    batch = getattr(cls, "extract_tool_calls", None)

    if streaming is not None and not getattr(streaming, "_genesis_g4_14_wrapped", False):
        _ORIGINAL_STREAMING = streaming

        def _g4_14_streaming(self, current_text, *args, **kwargs):
            stripped = _strip_control_tokens(current_text) if isinstance(current_text, str) else current_text
            return _ORIGINAL_STREAMING(self, stripped, *args, **kwargs)

        _g4_14_streaming._genesis_g4_14_wrapped = True
        _g4_14_streaming.__wrapped__ = _ORIGINAL_STREAMING
        cls.extract_tool_calls_streaming = _g4_14_streaming

    if batch is not None and not getattr(batch, "_genesis_g4_14_wrapped", False):
        _ORIGINAL_BATCH = batch

        def _g4_14_batch(self, model_output, *args, **kwargs):
            stripped = _strip_control_tokens(model_output) if isinstance(model_output, str) else model_output
            return _ORIGINAL_BATCH(self, stripped, *args, **kwargs)

        _g4_14_batch._genesis_g4_14_wrapped = True
        _g4_14_batch.__wrapped__ = _ORIGINAL_BATCH
        cls.extract_tool_calls = _g4_14_batch

    _APPLIED = True
    log.info(
        "[G4_14] installed: Gemma 4 tool-call parser will strip control "
        "tokens (<pad>, <eos>, turn-boundaries) before JSON parsing."
    )
    return "applied", (
        "G4_14 installed: Gemma 4 tool-call parser now strips control "
        "tokens from streaming + batch input. Closes vllm#39392 "
        "(malformed JSON args with embedded <pad>)."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_STREAMING, _ORIGINAL_BATCH, _PATCHED_CLS
    if not _APPLIED or _PATCHED_CLS is None:
        return False
    if _ORIGINAL_STREAMING is not None:
        _PATCHED_CLS.extract_tool_calls_streaming = _ORIGINAL_STREAMING
    if _ORIGINAL_BATCH is not None:
        _PATCHED_CLS.extract_tool_calls = _ORIGINAL_BATCH
    _APPLIED = False
    _ORIGINAL_STREAMING = None
    _ORIGINAL_BATCH = None
    _PATCHED_CLS = None
    return True


__all__ = ["GENESIS_G4_14_MARKER", "apply", "is_applied", "revert"]

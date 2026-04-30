# SPDX-License-Identifier: Apache-2.0
"""Long-context tool-call adherence hook (P68 + P69).

Genesis-original — addresses model-behavior limitation where Qwen3-class
models (and most LLMs) lose tool-call format adherence at long context
with significant prefix content.

================================================================
PROBLEM EMPIRICALLY OBSERVED (Genesis test 2026-04-25)

Ladder test on Qwen3.6-35B-A3B-FP8 with `tool_choice="auto"`:

  prompt chars | tool_call success rate
  ─────────────────────────────────────
       0-12K  |  3/3 OK (clean)
       16K+   |  0/3 (model emits JSON text, refuses, or hallucinates)

Failure modes at long context:
  - Model emits `{"name":"get_weather","arguments":...}` as TEXT
    (not wrapped in `<tool_call>...</tool_call>` markers)
  - Model emits Python-style `get_weather(city="Tokyo", ...)` text
  - Model refuses ("I cannot retrieve real-time data")
  - Model hallucinates an answer ("The weather is 22°C")
  - Model emits ONLY content text without tool_calls field

Plain text generation at same context length works correctly (3/3),
so this is NOT an attention bug — it's structured-output adherence
degradation, a known LLM "lost in the middle" + format-decay issue.

Same behavior reproduces on prod ngram baseline AND on our P64+P65+P66
test container — confirming it's MODEL-LEVEL, not engine-level.
================================================================

This module provides TWO complementary mitigations:

P68 — Auto-upgrade `tool_choice` from "auto" to "required"
------------------------------------------------------------
When the prompt is long AND tools are provided AND user did not
explicitly set tool_choice, upgrade to "required". This triggers
vLLM's parser to expect/validate tool format more strictly. Plus
client receives a guaranteed tool emission.

WARNING: This forces tool emission. If user prompt is ambiguous about
whether a tool should be called, P68 will force one. Use only when
your workload expects tool calls for long-context queries (typical
agentic / function-calling pipelines).

P69 — Append explicit format reminder to last user message
-----------------------------------------------------------
Inject a clear instruction at the END of the last user message:
  "REMINDER: Use one of these tools by emitting <tool_call>...</tool_call>
   markers. Available: <names>. Do NOT respond in plain text..."

Reminders at the END have stronger attention impact than instructions
buried in middle of context (Liu et al. 2023, "Lost in the Middle").

Both hooks are independently togglable via env flags:
  GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1
  GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1

Threshold (in characters, approx 4 chars/token):
  GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS=50000  # default = ~12.5K tokens
                                                  # (was 8000; raised
                                                  #  per Genesis Issue #9
                                                  #  — 8K fired on every
                                                  #  realistic IDE-agent
                                                  #  prompt and silently
                                                  #  forced tool_choice or
                                                  #  appended reminders,
                                                  #  causing finish_reason
                                                  #  =stop on plain text)

================================================================

Hook integration: text-patch on `create_chat_completion` calls
`apply_hook(self, request)` at top of method, before any other
processing. Hook mutates `request.tool_choice` and/or
`request.messages[-1]["content"]` in place.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.middleware.long_ctx_tool_adherence")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _get_threshold_chars() -> int:
    """Return the long-context threshold in characters.

    Default raised 8000 → 50000 in response to Genesis Issue #9
    (noonghunna 2026-04-29). At 8000 chars (~2K tokens) the threshold
    fired on every realistic IDE-agent prompt — Cline / Cursor /
    OpenCode / Copilot Gateway typically build 15-50K-char system
    prompts — and silently coerced `tool_choice: auto → required` (P68)
    or appended a "must use a tool" reminder (P69), producing
    `finish_reason=stop` with empty content for plain-text user
    messages. 50000 chars (~12.5K tokens) keeps the long-context tool
    adherence behavior for genuinely long histories while leaving
    casual IDE-agent flows alone.
    """
    raw = os.environ.get("GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS", "50000")
    try:
        return max(1000, int(raw))
    except (ValueError, TypeError):
        return 50000


def _estimate_prompt_chars(messages: list[dict[str, Any]]) -> int:
    """Rough prompt-length estimate in characters.

    Counts string content from all messages. Multimodal parts (lists of
    content blocks) are counted by their text portions only. Tool/system
    messages included.
    """
    total = 0
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if isinstance(text, str):
                        total += len(text)
    return total


def _extract_tool_names(tools: list[Any]) -> list[str]:
    """Extract function names from request.tools for the reminder text."""
    names: list[str] = []
    for t in tools or []:
        if isinstance(t, dict):
            fn = t.get("function") or {}
            name = fn.get("name")
            if isinstance(name, str):
                names.append(name)
        else:
            # pydantic Tool model
            fn = getattr(t, "function", None)
            if fn is not None:
                name = getattr(fn, "name", None)
                if isinstance(name, str):
                    names.append(name)
    return names


def _build_p69_reminder(tool_names: list[str]) -> str:
    """Build the P69 format-reminder string appended to last user message.

    Designed to be:
    - Unambiguous about expected output format (`<tool_call>...</tool_call>`)
    - Explicitly forbid common failure modes (plain text, JSON-without-wrapper,
      Python-style call, refusal, hallucination)
    - Not too long (avoid blowing the context budget further)
    - Visible at END of prompt (attention-strong region for decoder LLMs)
    """
    names_str = ", ".join(tool_names) if tool_names else "the provided tools"
    return (
        "\n\n---\n"
        "[SYSTEM REMINDER — FORMAT REQUIREMENT]\n"
        f"Use one of these tools by emitting "
        "`<tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>` markers "
        f"verbatim. Available tools: {names_str}.\n"
        "DO NOT respond with plain text. DO NOT emit JSON without the "
        "`<tool_call>` wrapper. DO NOT emit Python-style function calls. "
        "DO NOT refuse, claim inability, or fabricate the answer.\n"
        "Emit ONLY the `<tool_call>...</tool_call>` block.\n"
        "---"
    )


def apply_hook(serving_chat: Any, request: Any) -> dict[str, Any]:
    """Apply P68 + P69 hooks if conditions met.

    Mutates `request` IN PLACE. Returns dict describing applied changes
    (for telemetry / logging).

    Conditions:
      - request.tools must be non-empty (caller actually provided tools)
      - request.tool_choice must be "auto" or None (don't override explicit
        user choice like "none" or named-function or "required")
      - prompt char-count must exceed threshold

    Behavior:
      - P68 (if env=1): upgrade tool_choice → "required"
      - P69 (if env=1): append reminder to last user message (string content)

    Returns: dict with keys {applied_p68: bool, applied_p69: bool,
             prompt_chars: int, threshold: int, reason: str}
    """
    result = {
        "applied_p68": False,
        "applied_p69": False,
        "prompt_chars": 0,
        "threshold": 0,
        "reason": "",
    }

    p68_enabled = _env_flag("GENESIS_ENABLE_P68_AUTO_FORCE_TOOL")
    p69_enabled = _env_flag("GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER")

    if not (p68_enabled or p69_enabled):
        result["reason"] = "both env flags off; no-op"
        return result

    # Gate 1: tools must be provided
    tools = getattr(request, "tools", None)
    if not tools:
        result["reason"] = "no tools in request; no-op"
        return result

    # Gate 2: tool_choice must be "auto" or None (respect explicit choices)
    tool_choice = getattr(request, "tool_choice", None)
    if tool_choice not in (None, "auto"):
        result["reason"] = (
            f"tool_choice={tool_choice!r} explicit; respecting user choice"
        )
        return result

    # Gate 3: prompt length threshold
    messages = getattr(request, "messages", None)
    if not messages:
        result["reason"] = "no messages; no-op"
        return result
    chars = _estimate_prompt_chars(messages)
    threshold = _get_threshold_chars()
    result["prompt_chars"] = chars
    result["threshold"] = threshold
    if chars < threshold:
        result["reason"] = (
            f"prompt chars {chars} < threshold {threshold}; no-op"
        )
        return result

    # All gates passed. Apply requested mitigations.

    # P68 — auto force tool
    if p68_enabled:
        try:
            request.tool_choice = "required"
            result["applied_p68"] = True
            # WARN-level (was INFO): per Genesis Issue #9 — operators must
            # see this in default log levels because the rewrite changes
            # request semantics in a way that can produce
            # finish_reason=stop on what was a casual user message.
            log.warning(
                "[Genesis P68] long-ctx prompt (%d chars >= %d threshold): "
                "upgraded tool_choice 'auto' -> 'required'. To disable "
                "this rewrite set GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=0 or "
                "raise GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS.",
                chars, threshold,
            )
        except Exception as e:
            log.warning("[Genesis P68] failed to upgrade tool_choice: %s", e)

    # P69 — append reminder to last user message
    if p69_enabled:
        try:
            last = messages[-1] if isinstance(messages, list) else None
            if last is not None and isinstance(last, dict):
                role = last.get("role")
                content = last.get("content")
                if role == "user" and isinstance(content, str):
                    tool_names = _extract_tool_names(tools)
                    reminder = _build_p69_reminder(tool_names)
                    last["content"] = content + reminder
                    result["applied_p69"] = True
                    log.info(
                        "[Genesis P69] long-ctx prompt (%d chars >= %d): "
                        "appended tool-format reminder (+%d chars) to last "
                        "user message",
                        chars, threshold, len(reminder),
                    )
                else:
                    result["reason"] = (
                        f"last message role={role!r} content not string; "
                        "P69 skipped"
                    )
            else:
                result["reason"] = "last message not a dict; P69 skipped"
        except Exception as e:
            log.warning("[Genesis P69] failed to append reminder: %s", e)

    return result

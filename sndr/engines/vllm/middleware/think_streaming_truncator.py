# SPDX-License-Identifier: Apache-2.0
"""PN16 V6 — streaming `<think>` token-budget enforcer.

Sprint 4 (audit closure 2026-05-08 noonghunna): deep fix for
``london_think`` failure class without prefix-cache or CUDA-graph
collateral damage.

Problem
-------
``london_think`` failure: model enters `<think>` block on a tool-attached
request, generates 1024 reasoning tokens until ``max_tokens`` cap, never
emits the actual ``<tool_call>``. PN16 V8 mitigates by prepending a
system-message budget hint (cache-stable, prompt-engineering nudge), but
the model can still ignore the hint and graphomanize.

V6 is the deterministic backstop: watch the streaming response, count
reasoning-content tokens, and when budget exceeded + tools attached:

  • STOP forwarding `delta.reasoning_content` chunks to the client
  • Continue letting the model generate internally until `</think>`
  • Forward subsequent `delta.tool_calls` chunks normally
  • Emit one synthetic ``[Genesis] reasoning truncated at N tokens``
    note in `delta.content` so client / operator log sees what happened

This bounds client-visible TTFT-to-tool-call deterministically. The
model still consumes `max_tokens` worth of compute (truncator can't
abort mid-generation safely), but the user / agent loop sees the
tool_call promptly instead of after 1024 wasted reasoning tokens.

Implementation contract
-----------------------
``ThinkStreamingTruncator`` is per-request stateful — instantiate one
per chat completion request. Caller feeds each ``ChatCompletionStream
Response`` chunk via ``filter_chunk()`` and gets back either:

  • The original chunk (passthrough)
  • A modified chunk (reasoning_content stripped, optionally with the
    truncation note in `delta.content`)
  • ``None`` (drop chunk entirely — used when nothing to forward)

The class is intentionally framework-agnostic — it operates on dict-
shaped chunks so it works against vLLM's SSE serializer, plain
streaming JSON, or test stubs without dragging in vllm imports.

Configuration
-------------
``GENESIS_PN16_MAX_THINKING_STREAM_TOKENS=N`` — budget (default 0=disabled)
``GENESIS_PN16_TRUNCATION_NOTE`` — text injected once when truncation
    fires (default ``"[Genesis] reasoning truncated at budget"``)

The truncator only fires on requests that have tools attached
(operator-set `tools` array on the request) — non-tool requests are
unaffected. This matches V8's targeted scope.

Author: Sandermage; Sprint 4 / 2026-05-09.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger("genesis.middleware.think_streaming_truncator")


# ─── Config helpers ────────────────────────────────────────────────────


def max_thinking_stream_tokens() -> int:
    """V6 budget. 0 disables the truncator (default)."""
    try:
        v = int(os.environ.get("GENESIS_PN16_MAX_THINKING_STREAM_TOKENS", "0"))
        return max(0, v)
    except (TypeError, ValueError):
        return 0


def _truncation_note() -> str:
    return os.environ.get(
        "GENESIS_PN16_TRUNCATION_NOTE",
        "[Genesis] reasoning truncated at budget — proceeding to tool call",
    )


# ─── Stats counter ─────────────────────────────────────────────────────


_STATS = {
    "requests_seen": 0,
    "truncations_fired": 0,
    "reasoning_chunks_dropped": 0,
    "passthrough_only": 0,
}


def get_stats() -> dict[str, int]:
    return dict(_STATS)


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


# ─── Truncator state machine ──────────────────────────────────────────


@dataclass
class ThinkStreamingTruncator:
    """Per-request stateful filter.

    Caller pattern::

        truncator = ThinkStreamingTruncator(
            budget_tokens=200, has_tools=True,
        )
        async for chunk in source_generator:
            filtered = truncator.filter_chunk(chunk)
            if filtered is not None:
                yield filtered

    Once truncation fires, subsequent reasoning chunks are silently
    dropped until the model emits a non-reasoning chunk (tool_call,
    content, or finish_reason). At that point the truncator returns
    to passthrough mode for those chunks.
    """

    budget_tokens: int
    has_tools: bool = False
    # State
    reasoning_token_count: int = 0
    truncated: bool = False
    note_injected: bool = False
    # Optional override of the in-band note text (defaults to env var)
    truncation_note: Optional[str] = None
    # Counters live in module-level _STATS but per-instance copy is useful
    # for tests / observability
    counters: dict[str, int] = field(default_factory=lambda: {
        "reasoning_chunks_seen": 0,
        "reasoning_chunks_dropped": 0,
        "tool_chunks_seen": 0,
        "content_chunks_seen": 0,
    })

    def __post_init__(self) -> None:
        _STATS["requests_seen"] += 1
        if not self.is_active():
            _STATS["passthrough_only"] += 1

    def is_active(self) -> bool:
        """Truncator is active only when budget>0 AND tools attached.

        When inactive, ``filter_chunk`` is a strict passthrough — caller
        can still safely use the truncator without checking is_active()."""
        return self.budget_tokens > 0 and self.has_tools

    def filter_chunk(self, chunk: Any) -> Optional[Any]:
        """Process one streaming chunk. Returns the chunk to forward
        (possibly modified) or None to drop entirely.

        Chunk shape: caller passes a dict-like with structure::

            {"choices": [{"delta": {...}, "finish_reason": ...}]}

        Where ``delta`` may contain ``reasoning_content``, ``content``,
        or ``tool_calls`` keys. Other top-level keys (id, created, etc.)
        are passed through unchanged.
        """
        if not self.is_active():
            return chunk

        choices = self._choices_of(chunk)
        if not choices:
            return chunk

        # Inspect the first choice (vLLM streams n=1 by default for
        # tool-call workloads; we don't try to handle n>1 here).
        delta = self._delta_of(choices[0])
        if delta is None:
            return chunk

        reasoning = self._get(delta, "reasoning_content")
        tool_calls = self._get(delta, "tool_calls")
        content = self._get(delta, "content")

        if tool_calls:
            self.counters["tool_chunks_seen"] += 1
            # Tool call chunks always pass — that's what we want clients to see.
            return chunk

        if content:
            self.counters["content_chunks_seen"] += 1
            return chunk

        if reasoning is None or reasoning == "":
            # Empty delta (heartbeat / role-only / finish_reason) — pass through
            return chunk

        # Reasoning chunk path
        self.counters["reasoning_chunks_seen"] += 1
        # Approximate token count: 1 token per reasoning chunk if it's
        # not a continuation. vLLM streams typically emit one delta per
        # decoded token, so chunk-count ≈ token-count. We don't decode
        # the actual token id here.
        self.reasoning_token_count += 1

        if self.reasoning_token_count <= self.budget_tokens:
            return chunk  # under budget — passthrough

        # Over budget. Drop the reasoning_content chunk.
        self.counters["reasoning_chunks_dropped"] += 1
        _STATS["reasoning_chunks_dropped"] += 1

        if not self.truncated:
            self.truncated = True
            _STATS["truncations_fired"] += 1
            log.info(
                "[PN16 V6] reasoning budget exceeded (%d tokens) — "
                "suppressing further reasoning_content chunks; tool_call "
                "chunks will pass through normally.",
                self.budget_tokens,
            )

        # On the FIRST drop, optionally emit a truncation note via
        # delta.content so the client log sees what happened.
        if not self.note_injected:
            self.note_injected = True
            note = self.truncation_note or _truncation_note()
            return self._build_note_chunk(chunk, note)

        # Subsequent drops: silent (no note duplication).
        return None

    # ─── Chunk shape helpers (defensive against pydantic models AND
    # plain dicts; both are seen across vLLM versions) ─────────────────

    @staticmethod
    def _choices_of(chunk: Any) -> Optional[list]:
        if isinstance(chunk, dict):
            return chunk.get("choices")
        return getattr(chunk, "choices", None)

    @staticmethod
    def _delta_of(choice: Any) -> Optional[Any]:
        if isinstance(choice, dict):
            return choice.get("delta")
        return getattr(choice, "delta", None)

    @staticmethod
    def _get(delta: Any, key: str) -> Any:
        if isinstance(delta, dict):
            return delta.get(key)
        return getattr(delta, key, None)

    @staticmethod
    def _build_note_chunk(template_chunk: Any, note: str) -> Any:
        """Construct a synthetic chunk that replaces reasoning_content
        with a one-shot truncation note in delta.content.

        The note is sent to the CLIENT as content (visible in chat UI)
        so users see "thinking was truncated" instead of the silent
        behaviour that would look like a stall.
        """
        if isinstance(template_chunk, dict):
            new = {k: v for k, v in template_chunk.items() if k != "choices"}
            choices = template_chunk.get("choices") or []
            new_choices = []
            for i, ch in enumerate(choices):
                base = dict(ch) if isinstance(ch, dict) else {
                    "index": getattr(ch, "index", i),
                    "finish_reason": getattr(ch, "finish_reason", None),
                }
                base["delta"] = {"content": note + "\n"}
                new_choices.append(base)
            new["choices"] = new_choices
            return new
        # Object path: don't risk pydantic model surgery — fall back to dict
        try:
            choices = []
            for i, ch in enumerate(template_chunk.choices):  # type: ignore[attr-defined]
                choices.append({
                    "index": getattr(ch, "index", i),
                    "delta": {"content": note + "\n"},
                    "finish_reason": getattr(ch, "finish_reason", None),
                })
            return {
                "id": getattr(template_chunk, "id", "chatcmpl-genesis-truncate"),
                "object": "chat.completion.chunk",
                "choices": choices,
            }
        except Exception:
            # Worst case — emit minimal chunk
            return {
                "id": "chatcmpl-genesis-truncate",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {"content": note + "\n"},
                    "finish_reason": None,
                }],
            }


# ─── Async streaming filter helper ─────────────────────────────────────


async def filter_stream(
    source,
    *,
    budget_tokens: int,
    has_tools: bool,
):
    """Async generator wrapper — feeds ``source`` chunks through a fresh
    ``ThinkStreamingTruncator`` and yields filtered chunks.

    Designed to drop in front of vllm's ``chat_completion_stream_generator``::

        original = self.chat_completion_stream_generator(...)
        async for chunk in filter_stream(
            original, budget_tokens=cfg, has_tools=bool(request.tools),
        ):
            yield chunk
    """
    truncator = ThinkStreamingTruncator(
        budget_tokens=budget_tokens, has_tools=has_tools,
    )
    async for chunk in source:
        out = truncator.filter_chunk(chunk)
        if out is not None:
            yield out

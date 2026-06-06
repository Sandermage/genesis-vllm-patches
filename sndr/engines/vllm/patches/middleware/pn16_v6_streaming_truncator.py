# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN16 V6 — streaming `<think>` token-budget enforcer.

Sprint 4 (audit closure 2026-05-08 noonghunna): deep fix for london_think
graphomania without prefix-cache or CUDA-graph collateral damage.

Wire-in strategy
================

Class-rebind ``OpenAIServingChat.chat_completion_stream_generator`` —
NOT text-patch. The wrapper:

  1. Builds a per-request ``ThinkStreamingTruncator`` from
     ``GENESIS_PN16_MAX_THINKING_STREAM_TOKENS`` (env, default 0=disabled)
     and ``request.tools`` presence
  2. Iterates the original async SSE-stream generator
  3. For each ``data: {…}\\n\\n`` chunk: parses JSON, filters via
     truncator, re-serializes
  4. Drops chunks the truncator returns ``None`` for

Fast-path: when truncator is inactive (env=0 or no tools), the wrapper
just re-yields the original SSE strings untouched — zero cost.

Activation
==========

Default OFF — opt in via two env vars:

  GENESIS_ENABLE_PN16_V6_STREAMING_TRUNCATOR=1   # master gate
  GENESIS_PN16_MAX_THINKING_STREAM_TOKENS=200    # budget (per-request)

V6 stacks cleanly on top of V8 (system-prompt budget hint) — V8 nudges
the model to be concise; V6 enforces the bound deterministically when
the model ignores the nudge.

Author: Sandermage; Sprint 4 / 2026-05-09.
"""
from __future__ import annotations

import json
import logging
import os

from vllm.sndr_core.middleware.think_streaming_truncator import (
    ThinkStreamingTruncator,
    max_thinking_stream_tokens,
)

log = logging.getLogger("genesis.wiring.pn16_v6_streaming_truncator")


def _is_enabled() -> bool:
    val = os.environ.get(
        "GENESIS_ENABLE_PN16_V6_STREAMING_TRUNCATOR", ""
    ).strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def _wrap_stream_generator(original_method):
    """Wrap ``OpenAIServingChat.chat_completion_stream_generator`` with
    a per-request truncator filter. Fast-path passthrough when inactive."""

    async def wrapped(self, request, *args, **kwargs):
        budget = max_thinking_stream_tokens()
        has_tools = bool(getattr(request, "tools", None))
        truncator = ThinkStreamingTruncator(
            budget_tokens=budget, has_tools=has_tools,
        )

        if not truncator.is_active():
            # Inactive: passthrough — preserves the original AsyncGenerator
            # contract bit-for-bit.
            async for sse_chunk in original_method(
                self, request, *args, **kwargs,
            ):
                yield sse_chunk
            return

        async for sse_chunk in original_method(self, request, *args, **kwargs):
            # SSE format: ``data: {json}\n\n`` (or ``data: [DONE]\n\n``).
            # Anything that doesn't match the prefix is yielded as-is
            # (heartbeat, comments, unexpected shapes — defensive).
            if not isinstance(sse_chunk, str) or not sse_chunk.startswith(
                "data: "
            ):
                yield sse_chunk
                continue

            payload = sse_chunk[len("data: "):]
            # Strip trailing \n\n (SSE message terminator)
            payload = payload.rstrip("\n").rstrip()
            if payload == "[DONE]":
                yield sse_chunk
                continue

            try:
                chunk_dict = json.loads(payload)
            except Exception:
                # Don't break on unparseable chunk — pass through.
                yield sse_chunk
                continue

            filtered = truncator.filter_chunk(chunk_dict)
            if filtered is None:
                continue  # drop chunk

            # Re-serialize. If filter returned the same dict, this is
            # equivalent to the input. If it returned a synthetic note
            # chunk, this is the new payload.
            try:
                new_payload = json.dumps(filtered)
            except Exception:
                # Couldn't re-serialize — fall back to original to avoid
                # corrupting the stream.
                yield sse_chunk
                continue

            yield f"data: {new_payload}\n\n"

    wrapped.__wrapped__ = original_method
    wrapped.__pn16_v6_wrapped__ = True
    return wrapped


def apply() -> tuple[str, str]:
    """Apply PN16 V6 — class-rebind streaming generator with budget filter."""
    from vllm.sndr_core.dispatcher import should_apply, log_decision

    decision, reason = should_apply("PN16_V6")
    log_decision("PN16_V6", decision, reason)
    if not decision:
        return "skipped", reason

    if not _is_enabled():
        return (
            "skipped",
            "opt-in only — set GENESIS_ENABLE_PN16_V6_STREAMING_TRUNCATOR=1 "
            "to engage. Companion: GENESIS_PN16_MAX_THINKING_STREAM_TOKENS=N "
            "(default 0=disabled).",
        )

    try:
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
    except Exception as exc:
        return (
            "skipped",
            f"OpenAIServingChat not importable on this vllm pin: {exc!r}",
        )

    if not hasattr(OpenAIServingChat, "chat_completion_stream_generator"):
        return (
            "skipped",
            "chat_completion_stream_generator absent on this vllm pin — "
            "PN16 V6 NULL",
        )

    if getattr(
        OpenAIServingChat.chat_completion_stream_generator,
        "__pn16_v6_wrapped__", False,
    ):
        return "applied", "PN16 V6 already wrapped (idempotent)"

    original = OpenAIServingChat.chat_completion_stream_generator
    OpenAIServingChat.chat_completion_stream_generator = _wrap_stream_generator(
        original,
    )

    budget = max_thinking_stream_tokens()
    return (
        "applied",
        f"PN16 V6 wrapped chat_completion_stream_generator — budget={budget} "
        "tokens, fires only on tool-attached requests. Once budget exceeded, "
        "subsequent reasoning_content chunks are suppressed and a one-shot "
        "[Genesis] truncation note is injected as delta.content. tool_calls "
        "and content chunks always pass through. Stacks on top of V8 "
        "(system-prompt budget hint) — V8 nudges, V6 enforces.",
    )

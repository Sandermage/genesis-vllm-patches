# SPDX-License-Identifier: Apache-2.0
"""OpenAI response helpers for the memory gateway.

The gateway's augment → forward → capture orchestration lives in the async route
`sndr.product_api.routes.gateway` (it must be async to stream from httpx) and
reuses `ConversationMemory` (augment/capture) + these helpers — one
implementation, no parallel sync copy. These functions parse the assistant text
out of an upstream reply so it can be captured into memory.
"""
from __future__ import annotations

import json
from typing import Any


def extract_assistant_text(response: dict[str, Any]) -> str:
    """Pull the assistant message text from an OpenAI chat-completion response
    (non-streaming). Returns "" when absent."""
    choices = response.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def assistant_text_from_sse(sse_text: str) -> str:
    """Reassemble the assistant message from an OpenAI SSE stream — concatenate
    every `choices[0].delta.content` across `data:` events (skipping `[DONE]`
    and unparseable lines). Used to capture a streamed reply into memory."""
    parts: list[str] = []
    for raw in sse_text.splitlines():
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except (ValueError, TypeError):
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            parts.append(str(piece))
    return "".join(parts)

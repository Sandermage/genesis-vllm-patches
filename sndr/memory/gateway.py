# SPDX-License-Identifier: Apache-2.0
"""MemoryGateway — OpenAI-compatible memory middleware around any upstream.

This is the единое-ядро integration point: clients point at the gateway instead
of directly at the upstream, and it transparently adds memory to EVERY model —

    client -> MemoryGateway -> upstream(/v1/chat/completions) -> provider

The upstream is whatever you configure:
  * external models: CLIProxyAPI (router-for-me/CLIProxyAPI) -> Claude/Gemini/...
  * internal models: the vLLM OpenAI server (the 35B)

We never modify the upstream. The gateway, per request:
  1. augment  — recall the owner's relevant memories, inject them as a plain-text
                system block (works for any model; no tool-calls / special format),
  2. forward  — call the upstream with the augmented body (other fields passed
                through untouched),
  3. capture  — remember the salient user turn + the assistant reply.

This module is the transport-agnostic CORE: the upstream call is an injected
`forward(body) -> response_dict`. The HTTP/SSE shell (a FastAPI route forwarding
to the configured base URL, streaming) layers on top and reuses this.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from sndr.memory.middleware import ConversationMemory


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
    import json

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


class MemoryGateway:
    def __init__(
        self,
        *,
        memory: ConversationMemory,
        forward: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._memory = memory
        self._forward = forward

    def handle(self, *, owner_id: int, request_body: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming augment -> forward -> capture. Never mutates
        `request_body`; only the forwarded copy carries the injected memory."""
        messages = request_body.get("messages") or []
        augmented = self._memory.augment(owner_id=owner_id, messages=messages)
        forwarded = {**request_body, "messages": augmented}
        response = self._forward(forwarded)
        assistant = extract_assistant_text(response)
        self._memory.capture(owner_id=owner_id, messages=messages, assistant=assistant)
        return response

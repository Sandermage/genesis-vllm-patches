# SPDX-License-Identifier: Apache-2.0
"""ConversationMemory — universal, model-agnostic capture/inject.

This is how memory makes EVERY model smarter, local or external: before a chat
request we recall the owner's relevant memories and inject them as plain text
(a system block) — no tool-calls, no model-specific format — and after the turn
we capture the salient content. Because it speaks plain messages, it works for
Gemma, Qwen, and any external model behind the proxy.

It is duck-typed on the engine (needs `recall` + `remember`), so the same class
runs in-process (MemoryEngine) or against an HTTP client exposing that shape.
Pure, side-effect-light, and fully unit-tested with the in-memory engine.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from sndr.memory.model import SearchHit

_INJECT_HEADER = "Relevant memory (use if helpful):"


class _RecallRemember(Protocol):
    def recall(
        self, *, owner_id: int, query: str, limit: int = ..., expand_depth: int = ...,
        reinforce: bool = ...,
    ) -> list[SearchHit]: ...
    def remember(
        self, *, owner_id: int, text: str, kind: str = ..., importance: float = ...,
        properties: dict[str, Any] | None = ...,
    ) -> int: ...


def _last_user(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "user" and str(m.get("content", "")).strip():
            return str(m["content"])
    return None


class ConversationMemory:
    def __init__(
        self,
        *,
        engine: _RecallRemember,
        max_recall: int = 5,
        min_score: float = 0.05,
        expand_depth: int = 1,
    ) -> None:
        self._engine = engine
        self._max_recall = max_recall
        self._min_score = min_score
        self._expand_depth = expand_depth

    def augment(
        self, *, owner_id: int, messages: list[dict[str, Any]], query: str | None = None
    ) -> list[dict[str, Any]]:
        """Return a NEW message list with relevant memory injected as a system
        block (merged into an existing leading system message if present).
        Returns the input unchanged when nothing relevant is recalled."""
        q = query if query is not None else _last_user(messages)
        if not q:
            return messages
        hits = self._engine.recall(
            owner_id=owner_id, query=q, limit=self._max_recall,
            expand_depth=self._expand_depth, reinforce=False,
        )
        snippets = [h.node.content for h in hits if h.score >= self._min_score]
        if not snippets:
            return messages
        block = _INJECT_HEADER + "\n" + "\n".join(f"- {s}" for s in snippets)
        out = [dict(m) for m in messages]  # shallow copy; never mutate caller's list
        if out and out[0].get("role") == "system":
            out[0]["content"] = f"{out[0]['content']}\n\n{block}"
            return out
        return [{"role": "system", "content": block}, *out]

    def capture(
        self,
        *,
        owner_id: int,
        messages: list[dict[str, Any]],
        assistant: str | None = None,
        importance: float = 0.0,
    ) -> None:
        """Remember the salient turn (the last user message, plus the assistant
        reply when given). No-op when there is no user content."""
        user = _last_user(messages)
        if user:
            self._engine.remember(
                owner_id=owner_id, text=user, kind="conversation", importance=importance
            )
        if assistant and assistant.strip():
            self._engine.remember(
                owner_id=owner_id, text=assistant, kind="conversation",
                importance=importance,
            )

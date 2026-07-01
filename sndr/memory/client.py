# SPDX-License-Identifier: Apache-2.0
"""MemoryHTTPClient — drive the memory engine over HTTP from another process.

This is how EXTERNAL models connect to memory. The proxy (a separate process)
holds a MemoryHTTPClient pointed at the unified container's /api/v1/memory/*,
and feeds it to ConversationMemory exactly like the in-process MemoryEngine:

    client = MemoryHTTPClient("http://server:8811", owner_id=1, token=...)
    cm = ConversationMemory(engine=client)
    messages = cm.augment(owner_id=1, messages=messages)   # recall over HTTP
    ... call the external model with the augmented messages ...
    cm.capture(owner_id=1, messages=messages, assistant=reply)  # remember over HTTP

It exposes the same `recall` / `remember` shape as MemoryEngine (returning
SearchHit objects), so ConversationMemory is unchanged. Stdlib urllib only — no
new dependency for whoever embeds it.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from sndr.memory.model import MemoryNode, SearchHit


class MemoryHTTPClient:
    def __init__(
        self,
        base_url: str,
        *,
        owner_id: int = 1,
        token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._owner = owner_id
        self._token = token
        self._timeout = timeout

    def _call(
        self, method: str, path: str, body: dict[str, Any] | None = None,
        owner: int | None = None,
    ) -> Any:
        headers = {"X-Owner-Id": str(self._owner if owner is None else owner)}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(  # noqa: S310 - base_url is operator-configured
            f"{self._base}{path}", data=data, headers=headers, method=method
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["data"]

    def _hits(self, rows: list[dict[str, Any]], owner: int) -> list[SearchHit]:
        return [
            SearchHit(
                node=MemoryNode(
                    id=r["id"], owner_id=owner, kind=r.get("kind", "note"),
                    content=r["content"],
                ),
                score=r["score"],
            )
            for r in rows
        ]

    # ── MemoryEngine-compatible surface (what ConversationMemory needs) ───
    def recall(
        self,
        *,
        owner_id: int,
        query: str,
        limit: int = 10,
        expand_depth: int = 2,
        reinforce: bool = True,
    ) -> list[SearchHit]:
        rows = self._call("POST", "/api/v1/memory/recall", {
            "query": query, "limit": limit,
            "expand_depth": expand_depth, "reinforce": reinforce,
        }, owner=owner_id)
        return self._hits(rows, owner_id)

    def remember(
        self,
        *,
        owner_id: int,
        text: str,
        kind: str = "note",
        importance: float = 0.0,
        properties: dict[str, Any] | None = None,
    ) -> int:
        data = self._call("POST", "/api/v1/memory/remember", {
            "text": text, "kind": kind, "importance": importance,
            "properties": properties or {},
        }, owner=owner_id)
        return int(data["id"])

    def search(self, *, owner_id: int, query: str, limit: int = 10) -> list[SearchHit]:
        from urllib.parse import quote
        rows = self._call(
            "GET", f"/api/v1/memory/search?q={quote(query)}&limit={limit}", owner=owner_id
        )
        return self._hits(rows, owner_id)

    def stats(self, *, owner_id: int) -> dict[str, int]:
        """Owner memory counts: ``{"nodes": int, "edges": int}``."""
        return self._call("GET", "/api/v1/memory/stats", owner=owner_id)

# SPDX-License-Identifier: Apache-2.0
"""`InMemoryStore` — the pure-stdlib reference backend.

It is both the unit-test double and the numerical reference for the brain
mechanics: the Postgres backend must reproduce its results exactly. No numpy
(cosine is hand-rolled) so the engine core carries zero new runtime deps.

This is in-process and unbounded until `prune` lands; it is NOT the production
store. Production = Postgres + pgvector (see the design doc).
"""
from __future__ import annotations

import math
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

from sndr.memory.model import (
    CO_ACCESS_REL,
    HEBBIAN_ETA,
    HEBBIAN_LAMBDA,
    MemoryEdge,
    MemoryNode,
    SearchHit,
)
from sndr.memory.store import MemoryStore

# Relations that wire both ways during traversal / spreading activation.
_SYMMETRIC_RELS = frozenset({"co_access", "similar_to"})
_EPSILON = 1e-12
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 for a zero-magnitude OR mismatched-dimension vector
    (a dim mismatch means incomparable embedding spaces, not high similarity)."""
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryStore(MemoryStore):
    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._nodes: dict[int, MemoryNode] = {}
        self._edges: dict[tuple[int, int, str], MemoryEdge] = {}
        # incidence index: node_id -> set of edge keys touching it, so
        # neighbors() is O(degree) and prune() drops a node's edges O(degree)
        # (mirrors the B-tree adjacency indexes of the Postgres backend).
        self._incident: dict[int, set[tuple[int, int, str]]] = {}
        self._next_id = 1

    # ── nodes ────────────────────────────────────────────────────────────
    def add_node(
        self,
        *,
        owner_id: int,
        kind: str,
        content: str,
        embedding: Sequence[float],
        importance: float = 0.0,
        properties: dict[str, Any] | None = None,
    ) -> int:
        nid = self._next_id
        self._next_id += 1
        now = self._clock()
        self._nodes[nid] = MemoryNode(
            id=nid,
            owner_id=owner_id,
            kind=kind,
            content=content,
            embedding=list(embedding),
            importance=importance,
            properties=dict(properties or {}),
            created_at=now,
            accessed_at=now,
        )
        return nid

    def get_node(self, node_id: int) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def iter_nodes(self, owner_id: int) -> Iterator[MemoryNode]:
        return (n for n in self._nodes.values() if n.owner_id == owner_id)

    # ── edges ────────────────────────────────────────────────────────────
    def add_edge(
        self,
        src_id: int,
        dst_id: int,
        rel: str,
        *,
        weight: float = 0.0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        now = self._clock()
        key = (src_id, dst_id, rel)
        self._edges[key] = MemoryEdge(
            src_id=src_id,
            dst_id=dst_id,
            rel=rel,
            weight=weight,
            properties=dict(properties or {}),
            valid_at=now,
        )
        self._incident.setdefault(src_id, set()).add(key)
        self._incident.setdefault(dst_id, set()).add(key)

    def edge_weight(self, src_id: int, dst_id: int, rel: str) -> float:
        edge = self._edges.get((src_id, dst_id, rel))
        return edge.weight if edge is not None else 0.0

    # ── recall ───────────────────────────────────────────────────────────
    def search(
        self,
        *,
        owner_id: int,
        query: Sequence[float],
        limit: int = 15,
    ) -> list[SearchHit]:
        hits = [
            SearchHit(node=node, score=_cosine(query, node.embedding))
            for node in self._nodes.values()
            if node.owner_id == owner_id
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def keyword_search(
        self, *, owner_id: int, query: str, limit: int = 15
    ) -> list[SearchHit]:
        qtokens = set(_TOKEN_RE.findall(query.lower()))
        if not qtokens:
            return []
        hits: list[SearchHit] = []
        for node in self._nodes.values():
            if node.owner_id != owner_id:
                continue
            ntokens = _TOKEN_RE.findall(node.content.lower())
            if not ntokens:
                continue
            overlap = sum(1 for t in ntokens if t in qtokens)
            if overlap > 0:
                # tf-style: overlap normalized by sqrt(doc length) (BM25-ish)
                hits.append(SearchHit(node=node, score=overlap / math.sqrt(len(ntokens))))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def find_by_content(self, *, owner_id: int, content: str) -> int | None:
        for node in self._nodes.values():
            if node.owner_id == owner_id and node.content == content:
                return node.id
        return None

    def neighbors(
        self, node_id: int, *, min_weight: float = 0.0
    ) -> list[tuple[int, str, float]]:
        out: list[tuple[int, str, float]] = []
        for key in self._incident.get(node_id, ()):
            edge = self._edges[key]
            if edge.invalid_at is not None or edge.weight < min_weight:
                continue
            src, dst, rel = key
            if src == node_id:
                out.append((dst, rel, edge.weight))
            elif rel in _SYMMETRIC_RELS:  # dst == node_id, symmetric -> traverse back
                out.append((src, rel, edge.weight))
        return out

    def _touch(self, node_ids: Sequence[int], now: float) -> None:
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node is not None:
                node.access_count += 1
                node.accessed_at = now
                # Reinforcement: retrieval strengthens the memory (slows decay).
                node.strength = 1.0 + math.log1p(node.access_count)

    # ── brain mechanics ──────────────────────────────────────────────────
    # recall() and _retention() are inherited from MemoryStore (shared algorithm).
    def reinforce_co_access(self, node_ids: Sequence[int]) -> None:
        uniq = sorted(set(node_ids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]  # canonical (min, max) -> undirected
                old = self.edge_weight(a, b, CO_ACCESS_REL)
                new = min(1.0, HEBBIAN_LAMBDA * old + HEBBIAN_ETA)
                self.add_edge(a, b, CO_ACCESS_REL, weight=new)

    # ── maintenance (leak-bounding) ──────────────────────────────────────
    def prune(self, *, owner_id: int, max_nodes: int) -> int:
        now = self._clock()
        owned = [n for n in self._nodes.values() if n.owner_id == owner_id]
        if len(owned) <= max_nodes:
            return 0
        # Salience: importance + current retention + a small recency-of-use term.
        # Tie-break by id so eviction is deterministic (oldest id first out).
        def salience(n: MemoryNode) -> tuple[float, int]:
            s = n.importance + self._retention(n, now) + 0.1 * n.access_count
            return (s, -n.id)

        owned.sort(key=salience)                     # lowest salience first
        to_remove = owned[: len(owned) - max_nodes]
        remove_ids = {n.id for n in to_remove}
        for nid in remove_ids:
            self._nodes.pop(nid, None)
            # Drop every edge touching this node via the incidence index
            # (O(degree)); detach the surviving endpoint so no dangling ref.
            for key in self._incident.pop(nid, set()):
                if self._edges.pop(key, None) is None:
                    continue
                src, dst, _rel = key
                other = dst if src == nid else src
                if other not in remove_ids:
                    self._incident.get(other, set()).discard(key)
        return len(remove_ids)

    def set_communities(self, mapping: dict[int, int]) -> None:
        for nid, community in mapping.items():
            node = self._nodes.get(nid)
            if node is not None:
                node.community_id = community

    def set_importance(self, mapping: dict[int, float]) -> None:
        for nid, importance in mapping.items():
            node = self._nodes.get(nid)
            if node is not None:
                node.importance = importance

    def count_nodes(self, owner_id: int | None = None) -> int:
        if owner_id is None:
            return len(self._nodes)
        return sum(1 for n in self._nodes.values() if n.owner_id == owner_id)

    def count_edges(self) -> int:
        return len(self._edges)

    def owner_ids(self) -> list[int]:
        return sorted({n.owner_id for n in self._nodes.values()})

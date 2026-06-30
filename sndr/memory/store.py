# SPDX-License-Identifier: Apache-2.0
"""`MemoryStore` — the thin backend-agnostic storage interface.

Every backend (in-memory reference, Postgres+pgvector) implements the same
contract; `tests/unit/test_memory_store_contract.py` is the executable spec, run
against BOTH backends. Backends provide only data-access primitives (add_node,
get_node, search, neighbors, reinforce_co_access, prune, _touch, ...); the
brain-recall algorithm (two-phase spreading activation + lazy Ebbinghaus decay)
lives ONCE here as a concrete method, so both backends are provably identical.

Subclasses must set `self._clock` (a `() -> epoch_seconds` callable) in __init__
so decay is deterministic under test.

The interface is synchronous on purpose: the engine core has no async machinery,
and the Postgres backend wraps a connection pool. Async write-back is a
product-API concern layered on top, not baked into the contract.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from sndr.memory.model import EBBINGHAUS_S, SPREAD_DAMPING, SearchHit

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from sndr.memory.model import MemoryNode

_EPSILON = 1e-12


class MemoryStore(ABC):
    """Backend-agnostic persistent-memory store."""

    _clock = staticmethod(time.time)  # subclasses override per-instance for tests

    # ── nodes ────────────────────────────────────────────────────────────
    @abstractmethod
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
        """Insert a node; return its new id."""

    @abstractmethod
    def get_node(self, node_id: int) -> MemoryNode | None:
        """Return the node, or None if absent."""

    @abstractmethod
    def iter_nodes(self, owner_id: int) -> Iterator[MemoryNode]:
        """Iterate every node belonging to `owner_id` (for the batch linker)."""

    # ── edges ────────────────────────────────────────────────────────────
    @abstractmethod
    def add_edge(
        self,
        src_id: int,
        dst_id: int,
        rel: str,
        *,
        weight: float = 0.0,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or replace a directed edge (src) -[rel]-> (dst)."""

    @abstractmethod
    def edge_weight(self, src_id: int, dst_id: int, rel: str) -> float:
        """Return the edge weight, or 0.0 if the edge does not exist."""

    # ── recall primitives ────────────────────────────────────────────────
    @abstractmethod
    def search(
        self,
        *,
        owner_id: int,
        query: Sequence[float],
        limit: int = 15,
    ) -> list[SearchHit]:
        """Owner-scoped ANN search; hits sorted by descending cosine score."""

    @abstractmethod
    def keyword_search(
        self, *, owner_id: int, query: str, limit: int = 15
    ) -> list[SearchHit]:
        """Owner-scoped lexical (keyword/BM25-style) search over node content;
        returns only docs sharing terms with the query, ranked by relevance.
        Complements vector search for exact terms, names, and identifiers."""

    @abstractmethod
    def find_by_content(self, *, owner_id: int, content: str) -> int | None:
        """Return the id of an owner node with exactly this content, else None
        (the cheap exact-dedup probe)."""

    @abstractmethod
    def neighbors(
        self, node_id: int, *, min_weight: float = 0.0
    ) -> list[tuple[int, str, float]]:
        """Return (neighbor_id, rel, weight) adjacent to `node_id`. Symmetric
        relations (co_access, similar_to) are traversable from either end;
        directed relations only follow src -> dst. Excludes invalidated edges.
        """

    @abstractmethod
    def _touch(self, node_ids: Sequence[int], now: float) -> None:
        """Mark nodes as accessed: access_count += 1 and accessed_at = now."""

    # ── brain mechanics ──────────────────────────────────────────────────
    @abstractmethod
    def reinforce_co_access(self, node_ids: Sequence[int]) -> None:
        """Hebbian update: strengthen the (undirected) `co_access` edge between
        every pair of co-retrieved nodes by  w <- min(1, (1 - lambda) * w + eta).
        """

    def _retention(self, node: MemoryNode, now: float) -> float:
        """Ebbinghaus retention R = exp(-age / (S * (1 + importance)))."""
        age = max(0.0, now - node.accessed_at)
        scale = EBBINGHAUS_S * (1.0 + max(0.0, node.importance))
        return math.exp(-age / scale)

    def recall(
        self,
        *,
        owner_id: int,
        query: Sequence[float],
        limit: int = 10,
        expand_depth: int = 2,
        reinforce: bool = True,
    ) -> list[SearchHit]:
        """Two-phase brain recall (shared by every backend): ANN seeds -> bounded
        cycle-safe graph expand with spreading activation, blended with lazy
        Ebbinghaus decay. Touches the returned nodes and, when `reinforce`,
        Hebbian-strengthens their mutual co_access edges. Non-positive
        activations are dropped; result is the top `limit` by final score.
        """
        now = self._clock()
        # Phase 1 — ANN seeds (cosine == seed activation).
        activation: dict[int, float] = {}
        frontier: list[tuple[int, float, int]] = []  # (node_id, activation, depth)
        for hit in self.search(owner_id=owner_id, query=query, limit=max(limit, 1)):
            if hit.score <= _EPSILON:
                continue
            activation[hit.id] = hit.score
            frontier.append((hit.id, hit.score, 0))
        # Phase 2 — bounded, cycle-safe spreading activation along edges.
        while frontier:
            nid, act, depth = frontier.pop()
            if depth >= expand_depth:
                continue
            for neigh_id, _rel, weight in self.neighbors(nid):
                node = self.get_node(neigh_id)
                if node is None or node.owner_id != owner_id:
                    continue
                propagated = act * weight * SPREAD_DAMPING
                if propagated <= activation.get(neigh_id, 0.0) + _EPSILON:
                    continue  # no improvement -> don't re-expand (terminates)
                activation[neigh_id] = propagated
                frontier.append((neigh_id, propagated, depth + 1))
        # Blend activation with lazy decay; drop non-positive; rank; trim.
        scored: list[SearchHit] = []
        for nid, act in activation.items():
            node = self.get_node(nid)
            if node is None:
                continue
            final = act * self._retention(node, now)
            if final > _EPSILON:
                scored.append(SearchHit(node=node, score=final))
        scored.sort(key=lambda h: h.score, reverse=True)
        result = scored[:limit]
        if result:
            self._touch([h.id for h in result], now)
        if reinforce and len(result) > 1:
            self.reinforce_co_access([h.id for h in result])
        return result

    # ── maintenance (leak-bounding) ──────────────────────────────────────
    @abstractmethod
    def prune(self, *, owner_id: int, max_nodes: int) -> int:
        """Evict the lowest-salience nodes of `owner_id` until at most
        `max_nodes` remain; drop their edges (no dangling). Returns the count
        removed. Owner-scoped — never touches another owner's memory.
        """

    @abstractmethod
    def set_communities(self, mapping: dict[int, int]) -> None:
        """Bulk-assign community_id (the "cloud") for the given node ids."""

    @abstractmethod
    def set_importance(self, mapping: dict[int, float]) -> None:
        """Bulk-assign importance for the given node ids."""

    @abstractmethod
    def count_nodes(self, owner_id: int | None = None) -> int:
        """Total node count, or the count for one owner."""

    @abstractmethod
    def count_edges(self) -> int:
        """Total edge count (for leak/soak assertions)."""

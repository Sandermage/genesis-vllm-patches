# SPDX-License-Identifier: Apache-2.0
"""`MemoryStore` — the thin backend-agnostic storage interface.

Every backend (in-memory reference, Postgres+pgvector) implements this exact
contract; `tests/unit/test_memory_store_contract.py` is the executable spec.
Keeping the surface small and explicit is what lets us swap backends without a
rewrite (the design's "pluggable interface" promise) and lets the brain
mechanics live in one tested place.

The interface is synchronous on purpose: the engine core has no async
machinery, and the Postgres backend wraps a connection pool. Concurrency /
async write-back is a product-API concern layered on top, not baked into the
contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sndr.memory.model import MemoryNode, SearchHit


class MemoryStore(ABC):
    """Backend-agnostic persistent-memory store."""

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

    # ── recall ───────────────────────────────────────────────────────────
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
    def neighbors(
        self, node_id: int, *, min_weight: float = 0.0
    ) -> list[tuple[int, str, float]]:
        """Return (neighbor_id, rel, weight) adjacent to `node_id`. Symmetric
        relations (co_access, similar_to) are traversable from either end;
        directed relations only follow src -> dst. Excludes invalidated edges.
        """

    # ── brain mechanics ──────────────────────────────────────────────────
    @abstractmethod
    def reinforce_co_access(self, node_ids: Sequence[int]) -> None:
        """Hebbian update: strengthen the (undirected) `co_access` edge between
        every pair of co-retrieved nodes by  w <- (1 - lambda) * w + eta.
        """

    @abstractmethod
    def recall(
        self,
        *,
        owner_id: int,
        query: Sequence[float],
        limit: int = 10,
        expand_depth: int = 2,
        reinforce: bool = True,
    ) -> list[SearchHit]:
        """Two-phase brain recall: ANN seeds -> bounded cycle-safe graph expand
        with spreading activation, blended with lazy Ebbinghaus decay. Touches
        (accessed_at / access_count) the returned nodes and, when `reinforce`,
        Hebbian-strengthens their mutual co_access edges. Non-positive
        activations are dropped; result is the top `limit` by final score.
        """

    # ── maintenance (leak-bounding) ──────────────────────────────────────
    @abstractmethod
    def prune(self, *, owner_id: int, max_nodes: int) -> int:
        """Evict the lowest-salience nodes of `owner_id` until at most
        `max_nodes` remain; drop their edges (no dangling). Returns the count
        removed. Owner-scoped — never touches another owner's memory.
        """

    @abstractmethod
    def count_nodes(self, owner_id: int | None = None) -> int:
        """Total node count, or the count for one owner."""

    @abstractmethod
    def count_edges(self) -> int:
        """Total edge count (for leak/soak assertions)."""

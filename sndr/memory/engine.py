# SPDX-License-Identifier: Apache-2.0
"""`MemoryEngine` — the storage-agnostic, text-level facade.

This is what the proxy memory-middleware and the product-API call. It owns the
embedder, turns text into vectors, and delegates persistence + brain mechanics
to a `MemoryStore`. It also runs the semantic auto-linker (`link_semantic`):
kNN over embeddings -> `similar_to` edges, which is the mechanism that turns a
pile of isolated notes into the connected "neuron cloud" graph.

Everything here is backend-agnostic: swap `InMemoryStore` for the Postgres
backend and the engine is unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sndr.memory.model import SEMANTIC_EDGE_TAU, SearchHit

if TYPE_CHECKING:
    from sndr.memory.embedder import Embedder
    from sndr.memory.store import MemoryStore

_SIMILAR_REL = "similar_to"


def run_maintenance(
    engine: MemoryEngine, *, max_nodes: int, tau: float = SEMANTIC_EDGE_TAU
) -> dict[str, Any]:
    """One maintenance pass over every owner: consolidate (auto-link + communities
    + importance) then prune to `max_nodes` (the wired leak-bound). This is what
    the background scheduler calls on a timer — the design's "nightly batch".
    Returns a small report."""
    owners = engine.store.owner_ids()
    pruned = 0
    for owner_id in owners:
        engine.consolidate(owner_id=owner_id, tau=tau)
        pruned += engine.store.prune(owner_id=owner_id, max_nodes=max_nodes)
    return {"owners": owners, "pruned": pruned}


class MemoryEngine:
    def __init__(self, *, store: MemoryStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    # ── write ────────────────────────────────────────────────────────────
    def remember(
        self,
        *,
        owner_id: int,
        text: str,
        kind: str = "note",
        importance: float = 0.0,
        properties: dict[str, Any] | None = None,
        dedup: bool = True,
    ) -> int:
        """Store `text` as a node; return its id. With `dedup` (default), an
        existing node with identical content for this owner is returned instead
        of inserting a duplicate (prevents memory bloat from repeated facts)."""
        if dedup:
            existing = self.store.find_by_content(owner_id=owner_id, content=text)
            if existing is not None:
                return existing
        embedding = self.embedder.embed_one(text)
        return self.store.add_node(
            owner_id=owner_id,
            kind=kind,
            content=text,
            embedding=embedding,
            importance=importance,
            properties=properties,
        )

    # ── read ─────────────────────────────────────────────────────────────
    def search(
        self, *, owner_id: int, query: str, limit: int = 10
    ) -> list[SearchHit]:
        """Pure ANN search by text — no graph expand, no side effects (the
        idempotent GET path). Use `recall` for the brain op."""
        return self.store.search(
            owner_id=owner_id, query=self.embedder.embed_one(query), limit=limit
        )

    def search_hybrid(
        self, *, owner_id: int, query: str, limit: int = 10, alpha: float = 0.5
    ) -> list[SearchHit]:
        """Blend semantic (vector) and lexical (keyword) search — alpha weights
        the vector side. Each side is min-max normalized to [0,1] before mixing,
        so exact terms/names/IDs (lexical) and meaning (vector) both count."""
        vec = self.store.search(
            owner_id=owner_id, query=self.embedder.embed_one(query), limit=limit * 2
        )
        kw = self.store.keyword_search(owner_id=owner_id, query=query, limit=limit * 2)

        def _norm(hits: list[SearchHit]) -> dict[int, float]:
            top = max((h.score for h in hits), default=0.0)
            return {h.id: (h.score / top if top > 0 else 0.0) for h in hits}

        nv, nk = _norm(vec), _norm(kw)
        nodes = {h.id: h.node for h in kw}
        nodes.update({h.id: h.node for h in vec})
        scored = [
            SearchHit(node=nodes[i], score=alpha * nv.get(i, 0.0) + (1 - alpha) * nk.get(i, 0.0))
            for i in nodes
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:limit]

    def recall(
        self,
        *,
        owner_id: int,
        query: str,
        limit: int = 10,
        expand_depth: int = 2,
        reinforce: bool = True,
    ) -> list[SearchHit]:
        query_vec = self.embedder.embed_one(query)
        return self.store.recall(
            owner_id=owner_id,
            query=query_vec,
            limit=limit,
            expand_depth=expand_depth,
            reinforce=reinforce,
        )

    # ── consolidation (the "nightly batch": link -> cluster -> rank) ──────
    def detect_communities(self, *, owner_id: int, max_iter: int = 20) -> dict[int, int]:
        """Assign community ids (the "clouds") via deterministic weighted label
        propagation over the owner's graph. Pure-Python (no igraph/Leiden dep);
        good enough for homelab-scale graphs. Persists community_id and returns
        {node_id: community}."""
        nodes = list(self.store.iter_nodes(owner_id))
        ids = {n.id for n in nodes}
        adj: dict[int, list[tuple[int, float]]] = {n.id: [] for n in nodes}
        for nid in ids:
            for neigh_id, _rel, weight in self.store.neighbors(nid):
                if neigh_id in ids:
                    adj[nid].append((neigh_id, weight))
        label = {nid: nid for nid in ids}
        order = sorted(ids)  # deterministic sweep
        for _ in range(max_iter):
            changed = False
            for nid in order:
                if not adj[nid]:
                    continue
                tally: dict[int, float] = {}
                for neigh_id, weight in adj[nid]:
                    tally[label[neigh_id]] = tally.get(label[neigh_id], 0.0) + weight
                # highest weighted label; tie-break on smallest id -> deterministic
                best = max(sorted(tally), key=lambda lab: tally[lab])
                if label[nid] != best:
                    label[nid] = best
                    changed = True
            if not changed:
                break
        # renumber to dense 0..k-1 (stable)
        remap = {lab: i for i, lab in enumerate(sorted(set(label.values())))}
        result = {nid: remap[label[nid]] for nid in ids}
        self.store.set_communities(result)
        return result

    def recompute_importance(self, *, owner_id: int) -> dict[int, float]:
        """Heuristic importance in [0,1] from connectivity + use: a hub (many
        strong edges, frequently accessed) matters more than a leaf or an
        isolated note. Persists importance and returns {node_id: importance}."""
        nodes = list(self.store.iter_nodes(owner_id))
        ids = {n.id for n in nodes}
        raw: dict[int, float] = {}
        for node in nodes:
            degree = sum(
                w for nb, _rel, w in self.store.neighbors(node.id) if nb in ids
            )
            raw[node.id] = degree + 0.25 * node.access_count
        hi = max(raw.values(), default=0.0)
        result = {nid: (v / hi if hi > 0 else 0.0) for nid, v in raw.items()}
        self.store.set_importance(result)
        return result

    def consolidate(
        self, *, owner_id: int, tau: float = SEMANTIC_EDGE_TAU, k: int = 10
    ) -> dict[str, int]:
        """The full batch the design calls for, as one call: semantic auto-link
        -> community detection -> importance. Returns a small report. Cheap
        enough to run on demand (the GUI's button) or on a schedule."""
        linked = self.link_semantic(owner_id=owner_id, tau=tau, k=k)
        communities = self.detect_communities(owner_id=owner_id)
        self.recompute_importance(owner_id=owner_id)
        return {
            "linked": linked,
            "communities": len(set(communities.values())),
            "nodes": len(communities),
        }

    # ── graph view ───────────────────────────────────────────────────────
    def graph(
        self, *, owner_id: int, limit: int = 200
    ) -> tuple[list, list[tuple[int, int, str, float]]]:
        """Return (nodes, edges) for an owner's memory graph, bounded to `limit`
        nodes — the data the GUI force-graph renders. Edges are the undirected
        set among the returned nodes (deduped). Storage-agnostic."""
        nodes = list(self.store.iter_nodes(owner_id))[:limit]
        ids = {n.id for n in nodes}
        edges: list[tuple[int, int, str, float]] = []
        seen: set[tuple[int, int, str]] = set()
        for node in nodes:
            for neigh_id, rel, weight in self.store.neighbors(node.id):
                if neigh_id not in ids:
                    continue
                key = (min(node.id, neigh_id), max(node.id, neigh_id), rel)
                if key in seen:
                    continue
                seen.add(key)
                edges.append((key[0], key[1], rel, weight))
        return nodes, edges

    # ── batch graph building ─────────────────────────────────────────────
    def link_semantic(
        self,
        *,
        owner_id: int,
        tau: float = SEMANTIC_EDGE_TAU,
        k: int = 10,
    ) -> int:
        """Connect each node to its kNN neighbours above cosine `tau` with a
        (canonical, undirected) `similar_to` edge weighted by the similarity.
        Returns the number of distinct edges created. Idempotent on weight
        (re-running re-asserts the same edge). Storage-agnostic: uses only
        iter_nodes + search + add_edge.
        """
        linked: set[tuple[int, int]] = set()
        for node in list(self.store.iter_nodes(owner_id)):
            for hit in self.store.search(
                owner_id=owner_id, query=node.embedding, limit=k + 1
            ):
                if hit.id == node.id or hit.score < tau:
                    continue
                a, b = min(node.id, hit.id), max(node.id, hit.id)
                if (a, b) in linked:
                    continue
                linked.add((a, b))
                self.store.add_edge(a, b, _SIMILAR_REL, weight=hit.score)
        return len(linked)

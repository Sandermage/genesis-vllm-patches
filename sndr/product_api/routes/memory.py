# SPDX-License-Identifier: Apache-2.0
"""Persistent-memory routes (/api/v1/memory/*), owner-scoped.

The neural-graph memory engine exposed over HTTP for the proxy memory-middleware
(server-to-server) and, later, the GUI graph panel. Owner scoping comes from the
`X-Owner-Id` header — the service path the proxy uses. A session-auth dependency
plugs in here later without changing the routes (the store enforces owner
scoping regardless).

The MemoryEngine lives on `app.state.memory_engine` (set by `create_app` /
`init_memory_engine`), so tests can inject an in-memory engine.

Handlers are deliberately plain `def` (NOT `async def`): the engine/stores are
fully synchronous (CPU embedding, psycopg behind a lock, pure-Python scans), so
async handlers would run them ON the event loop and freeze the entire unified
daemon (GUI + gateway) for the duration — FastAPI runs `def` handlers in its
threadpool instead.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request

from sndr.product_api.schemas.common import Envelope, ResponseMeta
from sndr.product_api.schemas.memory import (
    ConsolidateOut,
    GraphEdgeOut,
    GraphNodeOut,
    GraphOut,
    HitOut,
    InvalidateEdgeIn,
    InvalidateEdgeOut,
    LinkIn,
    LinkOut,
    NeighborOut,
    NodeOut,
    ObsidianImportIn,
    ObsidianImportOut,
    RecallIn,
    RememberIn,
    RememberOut,
    StatsOut,
)
from sndr.product_api.security import owner_from_request

if TYPE_CHECKING:
    from sndr.memory.engine import MemoryEngine

router = APIRouter(prefix="/api/v1/memory", tags=["memory"])


def _engine(request: Request) -> MemoryEngine:
    engine = getattr(request.app.state, "memory_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="memory engine not configured")
    return engine


def _meta() -> ResponseMeta:
    return ResponseMeta(request_id=uuid4().hex, timestamp=datetime.now(timezone.utc))


def _hit(h) -> HitOut:
    return HitOut(id=h.id, content=h.node.content, kind=h.node.kind, score=h.score)


def _node_out(n) -> NodeOut:
    return NodeOut(
        id=n.id, owner_id=n.owner_id, kind=n.kind, content=n.content,
        importance=n.importance, strength=n.strength, access_count=n.access_count,
        community_id=n.community_id, properties=n.properties,
        created_at=n.created_at, accessed_at=n.accessed_at,
    )


@router.post("/remember", summary="Store a memory")
def remember(
    body: RememberIn, request: Request
) -> Envelope[RememberOut]:
    eng = _engine(request)
    owner = owner_from_request(request)
    nid = eng.remember(
        owner_id=owner, text=body.text, kind=body.kind,
        importance=body.importance, properties=body.properties,
    )
    return Envelope(data=RememberOut(id=nid), meta=_meta())


@router.get("/search", summary="Search (vector, or hybrid vector+keyword); no side effects")
def search(
    request: Request, q: str, limit: int = Query(10, ge=1, le=500), mode: str = "vector"
) -> Envelope[list[HitOut]]:
    # limit is bounded: an unbounded value made this a full-table fetch +
    # serialization of every node (hybrid mode doubles it) — a trivial DoS.
    eng = _engine(request)
    owner = owner_from_request(request)
    if mode == "hybrid":
        hits = eng.search_hybrid(owner_id=owner, query=q, limit=limit)
    else:
        hits = eng.search(owner_id=owner, query=q, limit=limit)
    return Envelope(data=[_hit(h) for h in hits], meta=_meta())


@router.post("/recall", summary="Brain recall (graph expand + reinforce)")
def recall(body: RecallIn, request: Request) -> Envelope[list[HitOut]]:
    eng = _engine(request)
    hits = eng.recall(
        owner_id=owner_from_request(request), query=body.query, limit=body.limit,
        expand_depth=body.expand_depth, reinforce=body.reinforce,
    )
    return Envelope(data=[_hit(h) for h in hits], meta=_meta())


@router.get("/node/{node_id}", summary="Fetch one node")
def get_node(node_id: int, request: Request) -> Envelope[NodeOut]:
    eng = _engine(request)
    owner = owner_from_request(request)
    node = eng.store.get_node(node_id)
    if node is None or node.owner_id != owner:
        raise HTTPException(status_code=404, detail="node not found")
    return Envelope(data=_node_out(node), meta=_meta())


@router.delete("/node/{node_id}", summary="Forget one memory (delete node + its edges)")
def delete_node(node_id: int, request: Request) -> Envelope[dict]:
    eng = _engine(request)
    owner = owner_from_request(request)
    deleted = eng.store.delete_node(node_id, owner_id=owner)
    if not deleted:
        raise HTTPException(status_code=404, detail="node not found")
    return Envelope(data={"deleted": True, "id": node_id}, meta=_meta())


@router.post("/edge/invalidate", summary="Invalidate (bi-temporally retire) an edge")
def invalidate_edge(body: InvalidateEdgeIn, request: Request) -> Envelope[InvalidateEdgeOut]:
    eng = _engine(request)
    owner = owner_from_request(request)
    src = eng.store.get_node(body.src)
    if src is None or src.owner_id != owner:
        raise HTTPException(status_code=404, detail="source node not found")
    ok = eng.store.invalidate_edge(body.src, body.dst, body.rel)
    return Envelope(data=InvalidateEdgeOut(invalidated=ok), meta=_meta())


@router.get("/neighbors/{node_id}", summary="Adjacent nodes")
def neighbors(node_id: int, request: Request) -> Envelope[list[NeighborOut]]:
    eng = _engine(request)
    owner = owner_from_request(request)
    node = eng.store.get_node(node_id)
    if node is None or node.owner_id != owner:
        raise HTTPException(status_code=404, detail="node not found")
    out = [
        NeighborOut(id=nid, rel=rel, weight=w)
        for nid, rel, w in eng.store.neighbors(node_id)
    ]
    return Envelope(data=out, meta=_meta())


@router.get("/stats", summary="Owner memory counts")
def stats(request: Request) -> Envelope[StatsOut]:
    eng = _engine(request)
    owner = owner_from_request(request)
    return Envelope(
        data=StatsOut(
            nodes=eng.store.count_nodes(owner_id=owner),
            edges=eng.store.count_edges(owner_id=owner),
            communities=eng.store.count_communities(owner_id=owner),
        ),
        meta=_meta(),
    )


@router.get("/graph", summary="Owner memory graph (nodes + edges) for visualization")
def graph(
    request: Request, limit: int = Query(200, ge=1, le=100_000)
) -> Envelope[GraphOut]:
    # Upper bound matches the GUI Export (full-graph backup legitimately asks
    # for 100k) while rejecting absurd values; runs in the threadpool anyway.
    eng = _engine(request)
    nodes, edges = eng.graph(owner_id=owner_from_request(request), limit=limit)
    return Envelope(
        data=GraphOut(
            nodes=[
                GraphNodeOut(
                    id=n.id, content=n.content, kind=n.kind,
                    community_id=n.community_id, importance=n.importance,
                    access_count=n.access_count,
                )
                for n in nodes
            ],
            edges=[
                GraphEdgeOut(src=s, dst=d, rel=r, weight=w)
                for (s, d, r, w) in edges
            ],
        ),
        meta=_meta(),
    )


@router.post("/link", summary="Run the semantic auto-linker (batch)")
def link(body: LinkIn, request: Request) -> Envelope[LinkOut]:
    eng = _engine(request)
    created = eng.link_semantic(owner_id=owner_from_request(request), tau=body.tau, k=body.k)
    return Envelope(data=LinkOut(created=created), meta=_meta())


@router.post("/consolidate", summary="Batch: auto-link + detect communities + importance")
def consolidate(body: LinkIn, request: Request) -> Envelope[ConsolidateOut]:
    eng = _engine(request)
    rep = eng.consolidate(owner_id=owner_from_request(request), tau=body.tau, k=body.k)
    return Envelope(data=ConsolidateOut(**rep), meta=_meta())


def _confined_vault(path: str) -> Path:
    """Resolve a vault path confined to GENESIS_MEMORY_VAULT_ROOT (disabled when
    unset). Blocks traversal outside the allowed root."""
    root = os.environ.get("GENESIS_MEMORY_VAULT_ROOT", "").strip()
    if not root:
        raise HTTPException(
            status_code=403,
            detail="vault import disabled (set GENESIS_MEMORY_VAULT_ROOT)",
        )
    base = Path(root).resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if resolved != base and base not in resolved.parents:
        raise HTTPException(status_code=403, detail="path escapes the allowed vault root")
    return resolved


@router.post("/import/obsidian", summary="Import an Obsidian vault into memory")
def import_obsidian(
    body: ObsidianImportIn, request: Request
) -> Envelope[ObsidianImportOut]:
    from sndr.memory.obsidian import import_vault

    eng = _engine(request)
    vault = _confined_vault(body.path)
    if not vault.is_dir():
        raise HTTPException(status_code=404, detail="vault not found")
    report = import_vault(engine=eng, owner_id=owner_from_request(request), vault_path=str(vault))
    return Envelope(data=ObsidianImportOut(**report), meta=_meta())


__all__ = ["router"]

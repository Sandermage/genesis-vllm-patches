# SPDX-License-Identifier: Apache-2.0
"""TDD contract for the memory product-API routes (/api/v1/memory/*).

Verified here with FastAPI's TestClient against a MemoryEngine backed by the
in-memory store, so no DB is needed. Owner scoping comes from the `X-Owner-Id`
header (the service path the proxy uses); a session-auth seam plugs in later.

Routes (LightRAG-shaped, owner-scoped):
  POST /remember  GET /search  POST /recall  GET /node/{id}
  GET /neighbors/{id}  GET /stats  POST /link
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sndr.memory.embedder import HashEmbedder
from sndr.memory.engine import MemoryEngine
from sndr.memory.inmemory import InMemoryStore
from sndr.product_api.routes.memory import router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.memory_engine = MemoryEngine(
        store=InMemoryStore(), embedder=HashEmbedder(dim=64)
    )
    return TestClient(app)


def _remember(client, text, owner=1, **kw):
    body = {"text": text, **kw}
    r = client.post("/api/v1/memory/remember", json=body,
                    headers={"X-Owner-Id": str(owner)})
    assert r.status_code == 200, r.text
    return r.json()["data"]["id"]


class TestRememberSearch:
    def test_remember_then_search_finds_it(self, client):
        nid = _remember(client, "postgres vector memory graph")
        r = client.get("/api/v1/memory/search",
                       params={"q": "postgres vector memory graph", "limit": 5},
                       headers={"X-Owner-Id": "1"})
        assert r.status_code == 200
        hits = r.json()["data"]
        assert nid in [h["id"] for h in hits]
        assert hits[0]["content"] == "postgres vector memory graph"
        assert hits[0]["score"] > 0.0

    def test_search_is_owner_scoped_by_header(self, client):
        _remember(client, "secret note for owner one", owner=1)
        r = client.get("/api/v1/memory/search",
                       params={"q": "secret note for owner one"},
                       headers={"X-Owner-Id": "2"})
        assert r.json()["data"] == []

    def test_hybrid_search_mode(self, client):
        tgt = _remember(client, "the api owner header is X-Owner-Id exactly")
        _remember(client, "a general note about something else entirely")
        r = client.get("/api/v1/memory/search",
                       params={"q": "X-Owner-Id", "mode": "hybrid"},
                       headers={"X-Owner-Id": "1"})
        assert r.status_code == 200
        assert r.json()["data"][0]["id"] == tgt

    def test_remember_dedup_default(self, client):
        a = _remember(client, "a fact stored exactly once")
        b = _remember(client, "a fact stored exactly once")
        assert a == b  # dedup on by default
        stats = client.get("/api/v1/memory/stats", headers={"X-Owner-Id": "1"}).json()["data"]
        assert stats["nodes"] == 1

    def test_remember_accepts_kind_and_importance(self, client):
        nid = _remember(client, "an important fact", kind="fact", importance=0.9)
        r = client.get(f"/api/v1/memory/node/{nid}", headers={"X-Owner-Id": "1"})
        node = r.json()["data"]
        assert node["kind"] == "fact"
        assert node["importance"] == pytest.approx(0.9)


class TestNodeNeighborsStats:
    def test_node_404_for_missing(self, client):
        r = client.get("/api/v1/memory/node/999999", headers={"X-Owner-Id": "1"})
        assert r.status_code == 404

    def test_stats_counts_nodes(self, client):
        _remember(client, "alpha beta")
        _remember(client, "gamma delta")
        r = client.get("/api/v1/memory/stats", headers={"X-Owner-Id": "1"})
        assert r.json()["data"]["nodes"] == 2

    def test_link_then_neighbors(self, client):
        a = _remember(client, "postgres vector memory graph")
        _remember(client, "postgres vector memory engine")
        created = client.post("/api/v1/memory/link", json={"tau": 0.5, "k": 10},
                              headers={"X-Owner-Id": "1"}).json()["data"]["created"]
        assert created >= 1
        nb = client.get(f"/api/v1/memory/neighbors/{a}",
                        headers={"X-Owner-Id": "1"}).json()["data"]
        assert any(n["rel"] == "similar_to" for n in nb)


class TestConsolidate:
    def test_consolidate_links_and_clusters(self, client):
        _remember(client, "postgres vector memory graph")
        _remember(client, "postgres vector memory engine")
        _remember(client, "banana orange weather guitar")
        rep = client.post("/api/v1/memory/consolidate", json={"tau": 0.5, "k": 10},
                          headers={"X-Owner-Id": "1"}).json()["data"]
        assert rep["linked"] >= 1
        assert rep["communities"] >= 1
        assert rep["nodes"] == 3
        # community_id is now populated (clouds exist) for at least the linked pair
        g = client.get("/api/v1/memory/graph", headers={"X-Owner-Id": "1"}).json()["data"]
        assert any(n["community_id"] is not None for n in g["nodes"])


class TestGraph:
    def test_graph_returns_nodes_and_edges(self, client):
        a = _remember(client, "postgres vector memory graph")
        b = _remember(client, "postgres vector memory engine")
        client.post("/api/v1/memory/link", json={"tau": 0.5, "k": 10},
                    headers={"X-Owner-Id": "1"})
        g = client.get("/api/v1/memory/graph", headers={"X-Owner-Id": "1"}).json()["data"]
        assert {n["id"] for n in g["nodes"]} == {a, b}
        assert any(
            e["rel"] == "similar_to" and {e["src"], e["dst"]} == {a, b}
            for e in g["edges"]
        )

    def test_graph_is_owner_scoped(self, client):
        _remember(client, "owner one note", owner=1)
        _remember(client, "owner two note", owner=2)
        g = client.get("/api/v1/memory/graph", headers={"X-Owner-Id": "2"}).json()["data"]
        assert all(n["content"] == "owner two note" for n in g["nodes"])


class TestRecallBrain:
    def test_recall_reaches_linked_neighbor_via_expand(self, client):
        a = _remember(client, "postgres vector memory graph")
        b = _remember(client, "postgres vector memory store")
        client.post("/api/v1/memory/link", json={"tau": 0.5, "k": 10},
                    headers={"X-Owner-Id": "1"})
        r = client.post("/api/v1/memory/recall",
                        json={"query": "graph", "limit": 10, "expand_depth": 1,
                              "reinforce": False},
                        headers={"X-Owner-Id": "1"})
        ids = [h["id"] for h in r.json()["data"]]
        assert a in ids        # direct hit on "graph"
        assert b in ids        # reached via the similar_to edge


class TestAppWiring:
    def test_create_app_registers_memory_routes_with_default_engine(self):
        from sndr.product_api.server import create_app

        app = create_app()
        client = TestClient(app)
        r = client.get("/api/v1/memory/stats", headers={"X-Owner-Id": "1"})
        assert r.status_code == 200
        assert r.json()["data"] == {"nodes": 0, "edges": 0}

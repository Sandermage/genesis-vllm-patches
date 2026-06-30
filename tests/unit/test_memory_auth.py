# SPDX-License-Identifier: Apache-2.0
"""TDD for the memory/gateway API-key guard (review finding C3).

When GENESIS_MEMORY_API_KEY is set, the memory + gateway routes require a matching
bearer token (or X-Api-Key). When unset, they stay open (localhost/dev). Applied
at router-include time in create_app, so it guards every memory/gateway route
uniformly without per-handler boilerplate.
"""
from __future__ import annotations

from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from sndr.memory.embedder import HashEmbedder
from sndr.memory.engine import MemoryEngine
from sndr.memory.inmemory import InMemoryStore
from sndr.product_api.routes.memory import router
from sndr.product_api.security import require_api_key


def _app() -> TestClient:
    app = FastAPI()
    app.include_router(router, dependencies=[Depends(require_api_key)])
    app.state.memory_engine = MemoryEngine(store=InMemoryStore(), embedder=HashEmbedder(dim=64))
    return TestClient(app, raise_server_exceptions=True)


def _get(client, **headers):
    return client.get("/api/v1/memory/stats", headers={"X-Owner-Id": "1", **headers})


def test_open_when_key_unset(monkeypatch):
    monkeypatch.delenv("GENESIS_MEMORY_API_KEY", raising=False)
    assert _get(_app()).status_code == 200


def test_401_without_key(monkeypatch):
    monkeypatch.setenv("GENESIS_MEMORY_API_KEY", "secret123")
    assert _get(_app()).status_code == 401


def test_200_with_bearer(monkeypatch):
    monkeypatch.setenv("GENESIS_MEMORY_API_KEY", "secret123")
    assert _get(_app(), Authorization="Bearer secret123").status_code == 200


def test_200_with_x_api_key(monkeypatch):
    monkeypatch.setenv("GENESIS_MEMORY_API_KEY", "secret123")
    assert _get(_app(), **{"X-Api-Key": "secret123"}).status_code == 200


def test_401_wrong_key(monkeypatch):
    monkeypatch.setenv("GENESIS_MEMORY_API_KEY", "secret123")
    assert _get(_app(), Authorization="Bearer nope").status_code == 401

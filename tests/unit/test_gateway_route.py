# SPDX-License-Identifier: Apache-2.0
"""TDD for the memory-gateway HTTP route (/v1/chat/completions).

Multi-upstream: the gateway holds a registry of named upstreams (your
CLIProxyAPI, another proxy, the internal vLLM, ...) and routes each request to
the one chosen by the `X-Memory-Upstream` header (falling back to the configured
default). `GET /v1/upstreams` lists the choices for the GUI.

Verified with fake upstreams on app.state (no live proxy needed); the augment ->
forward -> capture logic is exercised through the route.
"""
from __future__ import annotations

from fastapi import FastAPI
from starlette.testclient import TestClient

from sndr.memory.embedder import HashEmbedder
from sndr.memory.engine import MemoryEngine
from sndr.memory.inmemory import InMemoryStore
from sndr.product_api.routes.gateway import router


def _make_fake(tag: str, seen: dict):
    async def forward(body):
        seen["tag"] = tag
        seen["body"] = body
        return {"choices": [{"message": {"role": "assistant", "content": f"from {tag}"}}]}
    return {"forward": forward, "stream": None}


def _app(*, upstreams=True):
    app = FastAPI()
    app.include_router(router)
    app.state.memory_engine = MemoryEngine(store=InMemoryStore(), embedder=HashEmbedder(dim=64))
    seen: dict = {}
    if upstreams:
        app.state.gateway_upstreams = {
            "cliproxy": _make_fake("cliproxy", seen),
            "local": _make_fake("local", seen),
        }
        app.state.gateway_default = "cliproxy"
    return app, seen


def test_routes_to_default_upstream():
    app, seen = _app()
    eng = app.state.memory_engine
    eng.remember(owner_id=1, text="the magic number is 7788")
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "what is the magic number"}]},
        headers={"X-Owner-Id": "1"},
    )
    assert r.status_code == 200
    assert seen["tag"] == "cliproxy"                       # default upstream
    assert seen["body"]["messages"][0]["role"] == "system"  # memory injected
    assert "7788" in seen["body"]["messages"][0]["content"]
    assert r.json()["choices"][0]["message"]["content"] == "from cliproxy"


def test_selects_upstream_by_header():
    app, seen = _app()
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Owner-Id": "1", "X-Memory-Upstream": "local"},
    )
    assert r.status_code == 200
    assert seen["tag"] == "local"
    assert r.json()["choices"][0]["message"]["content"] == "from local"


def test_unknown_upstream_is_400():
    app, _ = _app()
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Owner-Id": "1", "X-Memory-Upstream": "nope"},
    )
    assert r.status_code == 400


def test_list_upstreams():
    app, _ = _app()
    client = TestClient(app)
    data = client.get("/v1/upstreams").json()
    assert sorted(data["upstreams"]) == ["cliproxy", "local"]
    assert data["default"] == "cliproxy"


def test_upstream_error_maps_to_502_not_500():
    app, _ = _app()

    async def boom(body):
        raise RuntimeError("upstream exploded")

    app.state.gateway_upstreams["cliproxy"] = {"forward": boom, "stream": None}
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Owner-Id": "1", "X-Memory-Upstream": "cliproxy"},
    )
    assert r.status_code == 502


def test_503_without_any_upstream():
    app, _ = _app(upstreams=False)
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Owner-Id": "1"},
    )
    assert r.status_code == 503

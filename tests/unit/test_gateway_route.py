# SPDX-License-Identifier: Apache-2.0
"""TDD for the memory-gateway HTTP route (/v1/chat/completions).

Verified with a fake upstream on app.state (no live CLIProxyAPI/vLLM needed):
the route augments the forwarded body with recalled memory and captures the
assistant reply. Streaming reassembly is unit-tested in test_memory_gateway.py.
"""
from __future__ import annotations

from fastapi import FastAPI
from starlette.testclient import TestClient

from sndr.memory.embedder import HashEmbedder
from sndr.memory.engine import MemoryEngine
from sndr.memory.inmemory import InMemoryStore
from sndr.product_api.routes.gateway import router


def _app(with_upstream: bool = True):
    app = FastAPI()
    app.include_router(router)
    app.state.memory_engine = MemoryEngine(store=InMemoryStore(), embedder=HashEmbedder(dim=64))
    seen: dict = {}
    if with_upstream:
        async def forward(body):
            seen["body"] = body
            return {"choices": [{"message": {"role": "assistant", "content": "answer text"}}]}
        app.state.gateway_forward = forward
        app.state.gateway_stream = None
    return app, seen


def test_gateway_augments_forwards_captures():
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
    assert r.json()["choices"][0]["message"]["content"] == "answer text"
    # upstream received the recalled memory injected as a system block
    assert seen["body"]["messages"][0]["role"] == "system"
    assert "7788" in seen["body"]["messages"][0]["content"]
    # the assistant reply was captured
    hits = eng.recall(owner_id=1, query="answer text", limit=10, reinforce=False)
    assert any("answer text" in h.node.content for h in hits)


def test_gateway_503_without_upstream():
    app, _ = _app(with_upstream=False)
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Owner-Id": "1"},
    )
    assert r.status_code == 503

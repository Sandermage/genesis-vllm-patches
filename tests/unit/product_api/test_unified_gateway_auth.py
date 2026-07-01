# SPDX-License-Identifier: Apache-2.0
"""TDD for the unified-app auth gap on the OpenAI-compatible gateway.

The unified factory mounts the memory/gateway routers with guard=False and
relies on the legacy auth middleware — but that middleware only enforced on
``/api/v1/*``. The gateway lives at ``/v1/*`` (chat completions + upstreams),
so with auth ENABLED any network caller could drive the gateway (and, via the
memory-augment path, read/write owner memories) with no credentials at all.

Contract: with auth enabled, ``/v1/*`` requires credentials exactly like
``/api/v1/*``; with auth off (loopback/dev), it stays open.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _unified_client(tmp_path, monkeypatch, *, auth: bool) -> TestClient:
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    if auth:
        monkeypatch.setenv("SNDR_AUTH", "on")
        monkeypatch.setenv("SNDR_ADMIN_PASSWORD", "admin-secret-123")
    else:
        monkeypatch.delenv("SNDR_AUTH", raising=False)
        monkeypatch.delenv("SNDR_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("SNDR_GUI_TOKEN", raising=False)
    monkeypatch.delenv("GENESIS_MEMORY_API_KEY", raising=False)
    monkeypatch.delenv("GENESIS_MEMORY_DSN", raising=False)
    from sndr.product_api.unified import create_app

    return TestClient(create_app(), raise_server_exceptions=True)


@pytest.mark.timeout(60)
def test_gateway_v1_requires_auth_when_enabled(tmp_path, monkeypatch):
    client = _unified_client(tmp_path, monkeypatch, auth=True)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Owner-Id": "7"},
    )
    assert r.status_code == 401, f"unauthenticated gateway chat must 401, got {r.status_code}"
    assert client.get("/v1/upstreams").status_code == 401


@pytest.mark.timeout(60)
def test_gateway_v1_stays_open_when_auth_off(tmp_path, monkeypatch):
    client = _unified_client(tmp_path, monkeypatch, auth=False)
    # No upstream configured -> the route answers (200 listing / 503 dormant),
    # but it must NOT be an auth rejection.
    assert client.get("/v1/upstreams").status_code not in (401, 403)

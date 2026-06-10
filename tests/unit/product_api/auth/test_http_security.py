# SPDX-License-Identifier: Apache-2.0
"""HTTP-level tests for the auth security hardening (CSRF, lockout, recovery,
session revocation) via FastAPI TestClient."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sndr.product_api.legacy.auth import totp  # noqa: E402
from sndr.product_api.legacy.http_app import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.setenv("SNDR_AUTH", "on")
    monkeypatch.setenv("SNDR_ADMIN_PASSWORD", "admin-secret-123")
    monkeypatch.delenv("SNDR_GUI_TOKEN", raising=False)
    return TestClient(create_app(enable_apply=False, bind_host="127.0.0.1"))


def _admin(client):
    return client.get("/api/v1/auth/status").json()["context"]["system_user"]


def test_login_lockout_returns_429(client):
    admin = _admin(client)
    for _ in range(8):
        client.post("/api/v1/auth/login", json={"username": admin, "password": "nope"})
    # even the correct password is now throttled
    resp = client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"})
    assert resp.status_code == 429


def test_csrf_blocks_cross_origin_cookie_mutation(client):
    admin = _admin(client)
    client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"})
    # cookie is set on the client; a cross-site fetch is blocked
    blocked = client.post(
        "/api/v1/auth/logout",
        headers={"Sec-Fetch-Site": "cross-site", "Origin": "http://evil.example"},
    )
    assert blocked.status_code == 403
    # same-origin is allowed
    ok = client.post("/api/v1/auth/logout", headers={"Sec-Fetch-Site": "same-origin"})
    assert ok.status_code == 200


def test_bearer_token_is_csrf_exempt(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.setenv("SNDR_AUTH", "on")
    monkeypatch.setenv("SNDR_GUI_TOKEN", "legacy-tok")
    c = TestClient(create_app(enable_apply=False, bind_host="127.0.0.1"))
    # bearer token + cross-site headers: no session cookie -> not a CSRF vector
    r = c.post(
        "/api/v1/services/apply",
        headers={"Authorization": "Bearer legacy-tok", "Sec-Fetch-Site": "cross-site"},
        json={"preset_id": "x", "action": "status"},
    )
    assert r.status_code != 403


def test_recovery_code_login_over_http(client):
    admin = _admin(client)
    client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"})
    enroll = client.post("/api/v1/auth/2fa/enroll").json()
    codes = client.post("/api/v1/auth/2fa/activate", json={"code": totp.hotp(enroll["secret"], int(time.time() // 30))}).json()["recovery_codes"]
    assert len(codes) == 10
    client.post("/api/v1/auth/logout", headers={"Sec-Fetch-Site": "same-origin"})
    # 2FA now required; a recovery code completes login
    first = client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"}).json()
    assert first["needs_2fa"]
    done = client.post("/api/v1/auth/login/2fa", json={"username": admin, "code": codes[0]})
    assert done.status_code == 200 and done.json()["token"]


def test_session_revoke_endpoint(client):
    admin = _admin(client)
    login = client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"}).json()
    token = login["token"]
    # revoke via the authenticated session
    assert client.post("/api/v1/auth/sessions/revoke", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    # the old bearer token is now rejected
    fresh = TestClient(create_app(enable_apply=False, bind_host="127.0.0.1"))  # noqa: F841 - new client, no cookie
    assert client.get("/api/v1/capabilities", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_audit_events_recorded(client):
    admin = _admin(client)
    client.post("/api/v1/auth/login", json={"username": admin, "password": "wrong"})
    client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"})
    events = client.get("/api/v1/events/recent").json()["events"]
    kinds = [e["detail"].get("event") for e in events if e.get("kind") == "auth"]
    assert "login.failed" in kinds and "login.success" in kinds

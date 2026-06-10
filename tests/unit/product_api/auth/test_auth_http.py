# SPDX-License-Identifier: Apache-2.0
"""End-to-end HTTP tests for the auth subsystem via FastAPI TestClient."""
from __future__ import annotations

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
    app = create_app(enable_apply=False, bind_host="127.0.0.1")
    return TestClient(app)


def test_status_reports_auth_required_and_context(client):
    body = client.get("/api/v1/auth/status").json()
    assert body["auth_required"] is True
    assert "local" in body["backends"]
    assert body["user"] is None
    assert "system_user" in body["context"]


def test_protected_endpoint_blocked_without_session(client):
    assert client.get("/api/v1/capabilities").status_code == 401


def test_login_and_access(client):
    # bootstrap admin username == system user
    status = client.get("/api/v1/auth/status").json()
    admin = status["context"]["system_user"]
    login = client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"})
    assert login.status_code == 200
    data = login.json()
    assert data["ok"] and not data["needs_2fa"] and data["token"]
    # cookie now set on the client -> protected endpoint accessible
    assert client.get("/api/v1/capabilities").status_code == 200
    me = client.get("/api/v1/auth/me").json()
    assert me["username"] == admin and me["role"] == "admin"


def test_bad_password_rejected(client):
    admin = client.get("/api/v1/auth/status").json()["context"]["system_user"]
    assert client.post("/api/v1/auth/login", json={"username": admin, "password": "wrong"}).status_code == 401


def test_admin_create_user_and_nonadmin_forbidden(client):
    admin = client.get("/api/v1/auth/status").json()["context"]["system_user"]
    client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"})
    created = client.post(
        "/api/v1/auth/users", json={"username": "operator1", "password": "operator-pass1", "role": "operator"}
    )
    assert created.status_code == 200 and created.json()["role"] == "operator"
    users = client.get("/api/v1/auth/users").json()["users"]
    assert {u["username"] for u in users} == {admin, "operator1"}
    # log out, log in as operator -> cannot list/create users
    client.post("/api/v1/auth/logout")
    client.post("/api/v1/auth/login", json={"username": "operator1", "password": "operator-pass1"})
    assert client.get("/api/v1/auth/users").status_code == 403


def test_2fa_enrol_then_required_on_login(client):
    admin = client.get("/api/v1/auth/status").json()["context"]["system_user"]
    client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"})
    enroll = client.post("/api/v1/auth/2fa/enroll").json()
    secret = enroll["secret"]
    assert enroll["otpauth_uri"].startswith("otpauth://")
    import time

    counter = int(time.time() // 30)
    activate = client.post("/api/v1/auth/2fa/activate", json={"code": totp.hotp(secret, counter)})
    assert activate.status_code == 200
    client.post("/api/v1/auth/logout")
    # next login now demands 2FA (no token yet)
    first = client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"}).json()
    assert first["needs_2fa"] is True and "token" not in first
    second = client.post(
        "/api/v1/auth/login/2fa", json={"username": admin, "code": totp.hotp(secret, int(time.time() // 30))}
    )
    assert second.status_code == 200 and second.json()["token"]


def test_legacy_token_still_authorizes(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.setenv("SNDR_AUTH", "on")
    monkeypatch.setenv("SNDR_GUI_TOKEN", "legacy-shared-token")
    app = create_app(enable_apply=False, bind_host="127.0.0.1")
    client = TestClient(app)
    assert client.get("/api/v1/capabilities").status_code == 401
    ok = client.get("/api/v1/capabilities", headers={"Authorization": "Bearer legacy-shared-token"})
    assert ok.status_code == 200


def test_auth_disabled_on_loopback_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.delenv("SNDR_AUTH", raising=False)
    monkeypatch.delenv("SNDR_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("SNDR_GUI_TOKEN", raising=False)
    app = create_app(enable_apply=False, bind_host="127.0.0.1")
    client = TestClient(app)
    # auto mode + loopback + no users -> open (unchanged dev behavior)
    assert client.get("/api/v1/capabilities").status_code == 200
    assert client.get("/api/v1/auth/status").json()["auth_required"] is False


def test_api_token_bearer_auth_end_to_end(client):
    admin = client.get("/api/v1/auth/status").json()["context"]["system_user"]
    client.post("/api/v1/auth/login", json={"username": admin, "password": "admin-secret-123"})
    # issue a managed API token (plaintext returned once)
    created = client.post("/api/v1/auth/tokens", json={"label": "ci"})
    assert created.status_code == 200
    token = created.json()["token"]
    assert token.startswith("sndr_pat_")
    assert created.json()["record"]["label"] == "ci"
    # listed (metadata only, no secret)
    listed = client.get("/api/v1/auth/tokens").json()["tokens"]
    assert len(listed) == 1 and "hash" not in listed[0]
    tid = listed[0]["id"]
    # a fresh client with ONLY the Bearer token can reach a protected endpoint
    fresh = TestClient(client.app)
    assert fresh.get("/api/v1/capabilities").status_code == 401
    assert fresh.get("/api/v1/capabilities", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    # revoke → the Bearer no longer authenticates
    assert client.delete(f"/api/v1/auth/tokens/{tid}").json()["revoked"] is True
    assert fresh.get("/api/v1/capabilities", headers={"Authorization": f"Bearer {token}"}).status_code == 401

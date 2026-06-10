# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the user store + AuthService facade."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy.auth import totp
from sndr.product_api.legacy.auth.config import AuthConfig
from sndr.product_api.legacy.auth.service import AuthError, AuthService
from sndr.product_api.legacy.auth.store import User, UserStore


def _config(**overrides) -> AuthConfig:
    base = dict(
        enabled=True,
        manage_accounts=True,
        bind_host="127.0.0.1",
        in_container=False,
        system_user="operator-bob",
        pam_enabled=False,
        session_ttl=3600,
        public_base_url="http://127.0.0.1:8765",
    )
    base.update(overrides)
    return AuthConfig(**base)


@pytest.fixture
def service(tmp_path):
    store = UserStore(auth_dir=tmp_path / "auth")
    return AuthService(store, _config())


# ---- persistence ----

def test_store_persists_across_instances(tmp_path):
    store = UserStore(auth_dir=tmp_path / "auth")
    store.put(User(username="alice", role="admin", password_hash="x"))
    # New instance reading the same dir — simulates a container restart.
    reopened = UserStore(auth_dir=tmp_path / "auth")
    assert reopened.count() == 1
    assert reopened.get("alice").role == "admin"


def test_signing_key_is_stable(tmp_path):
    store = UserStore(auth_dir=tmp_path / "auth")
    key1 = store.signing_key()
    key2 = UserStore(auth_dir=tmp_path / "auth").signing_key()
    assert key1 == key2 and len(key1) >= 32


def test_users_file_is_chmod_600(tmp_path):
    import os
    import stat

    store = UserStore(auth_dir=tmp_path / "auth")
    store.put(User(username="alice"))
    mode = stat.S_IMODE(os.stat(tmp_path / "auth" / "users.json").st_mode)
    assert mode == 0o600


# ---- bootstrap ----

def test_bootstrap_creates_admin_from_system_user(service):
    generated = service.bootstrap()
    assert generated  # auto-generated password returned once
    user = service.store.get("operator-bob")
    assert user is not None and user.role == "admin"
    # Bootstrap is idempotent.
    assert service.bootstrap() is None


def test_bootstrap_with_explicit_password(service):
    assert service.bootstrap(admin_password="supersecret123") is None
    result = service.authenticate("operator-bob", "supersecret123")
    assert result.ok and result.token


# ---- login state machine ----

def test_login_success_and_failure(service):
    service.create_user(username="carol", password="passw0rd!", role="operator")
    assert service.authenticate("carol", "passw0rd!").ok
    assert not service.authenticate("carol", "wrong").ok
    assert not service.authenticate("ghost", "x").ok


def test_login_requires_2fa_then_completes(service):
    user = service.create_user(username="dave", password="passw0rd!")
    secret, _uri = service.enroll_2fa(user)
    service.activate_2fa(user, totp.hotp(secret, _counter()))
    first = service.authenticate("dave", "passw0rd!")
    assert first.ok and first.needs_2fa and first.token is None
    second = service.complete_2fa("dave", totp.hotp(secret, _counter()))
    assert second.ok and second.token


def test_session_roundtrip(service):
    service.create_user(username="erin", password="passw0rd!")
    token = service.authenticate("erin", "passw0rd!").token
    assert service.verify_session(token).username == "erin"
    assert service.verify_session("garbage.token") is None


# ---- user management guards ----

def test_create_user_validation(service):
    with pytest.raises(AuthError):
        service.create_user(username="bad name", password="passw0rd!")
    with pytest.raises(AuthError):
        service.create_user(username="ok", password="short")
    service.create_user(username="ok", password="passw0rd!")
    with pytest.raises(AuthError):
        service.create_user(username="ok", password="passw0rd!")  # duplicate


def test_cannot_delete_last_admin(service):
    service.bootstrap(admin_password="passw0rd!")
    admin = service.store.get("operator-bob")
    other = service.create_user(username="other-admin", password="passw0rd!", role="admin")
    # deleting one admin while another remains is fine
    service.delete_user("other-admin", acting=admin)
    with pytest.raises(AuthError):
        service.delete_user("operator-bob", acting=other)  # last admin -> blocked
    with pytest.raises(AuthError):
        service.delete_user(admin.username, acting=admin)  # self -> blocked


def test_change_password_requires_current(service):
    user = service.create_user(username="frank", password="passw0rd!")
    with pytest.raises(AuthError):
        service.set_password(user, current="nope", new="newpassw0rd")
    service.set_password(user, current="passw0rd!", new="newpassw0rd")
    assert service.authenticate("frank", "newpassw0rd").ok


def _counter() -> int:
    import time

    return int(time.time() // 30)

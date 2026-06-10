# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the auth security hardening: rate-limit, session revocation,
recovery codes, OAuth privilege, session epoch."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy.auth import totp
from sndr.product_api.legacy.auth.config import AuthConfig
from sndr.product_api.legacy.auth.ratelimit import LoginGuard
from sndr.product_api.legacy.auth.service import AuthService
from sndr.product_api.legacy.auth.store import UserStore


def _config(**ov) -> AuthConfig:
    base = dict(enabled=True, manage_accounts=True, bind_host="127.0.0.1", in_container=False,
                system_user="admin", pam_enabled=False, session_ttl=3600, public_base_url="http://127.0.0.1:8765")
    base.update(ov)
    return AuthConfig(**base)


@pytest.fixture
def service(tmp_path):
    return AuthService(UserStore(auth_dir=tmp_path / "auth"), _config())


def _counter():
    import time
    return int(time.time() // 30)


# ---- rate limiter ----

def test_login_guard_locks_then_clears():
    g = LoginGuard(threshold=3, window=100, lockout=60)
    assert g.retry_after("u", now=0) == 0
    g.record_failure("u", now=1)
    g.record_failure("u", now=2)
    assert g.retry_after("u", now=3) == 0          # below threshold
    g.record_failure("u", now=3)                    # 3rd -> locked
    assert g.retry_after("u", now=4) > 0
    assert g.retry_after("u", now=70) == 0          # lockout expired
    # success clears
    g.record_failure("u", now=100); g.record_failure("u", now=101)
    g.record_success("u")
    assert g.retry_after("u", now=102) == 0


def test_authenticate_locks_after_failures(service):
    service.create_user(username="bob", password="passw0rd!")
    for _ in range(8):
        assert not service.authenticate("bob", "wrong").ok
    locked = service.authenticate("bob", "passw0rd!")  # even correct pw is throttled
    assert not locked.ok and locked.locked


# ---- session revocation / epoch ----

def test_password_change_revokes_old_sessions(service):
    user = service.create_user(username="carol", password="passw0rd!")
    token = service.authenticate("carol", "passw0rd!").token
    assert service.verify_session(token).username == "carol"
    service.set_password(user, current="passw0rd!", new="newpassw0rd")
    assert service.verify_session(token) is None        # old token invalidated


def test_explicit_revoke_invalidates(service):
    user = service.create_user(username="dave", password="passw0rd!")
    token = service.authenticate("dave", "passw0rd!").token
    assert service.verify_session(token) is not None
    service.revoke_sessions(user)
    assert service.verify_session(token) is None


# ---- 2FA recovery codes ----

def test_recovery_codes_issued_and_single_use(service):
    user = service.create_user(username="erin", password="passw0rd!")
    secret, _ = service.enroll_2fa(user)
    codes = service.activate_2fa(user, totp.hotp(secret, _counter()))
    assert len(codes) == 10
    # login needs 2FA; a recovery code completes it
    first = service.authenticate("erin", "passw0rd!")
    assert first.needs_2fa
    used = codes[0]
    assert service.complete_2fa("erin", used).ok
    # same code cannot be reused
    again = service.authenticate("erin", "passw0rd!")
    assert again.needs_2fa
    assert not service.complete_2fa("erin", used).ok
    # a different code still works
    assert service.complete_2fa("erin", codes[1]).ok


def test_disable_2fa_clears_recovery(service):
    user = service.create_user(username="frank", password="passw0rd!")
    secret, _ = service.enroll_2fa(user)
    service.activate_2fa(user, totp.hotp(secret, _counter()))
    service.disable_2fa(user)
    assert service.store.get("frank").recovery_codes == []


# ---- OAuth privilege ----

def test_oauth_user_is_never_admin(service):
    service.bootstrap(admin_password="passw0rd!")          # admin exists
    u = service.upsert_oauth_user(provider="google", subject="abc", email="x@y.com")
    assert u.role == "operator"


def test_oauth_user_operator_even_on_empty_store(tmp_path):
    svc = AuthService(UserStore(auth_dir=tmp_path / "auth"), _config())
    u = svc.upsert_oauth_user(provider="google", subject="z", email="z@y.com")
    assert u.role == "operator"   # no race-to-admin

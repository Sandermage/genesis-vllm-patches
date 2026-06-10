# SPDX-License-Identifier: Apache-2.0
"""Tests for managed API tokens (personal access tokens for the Product API).

Security contract: the plaintext token is returned exactly once at issue time;
only a scrypt hash of the secret is persisted; verification is O(1) (the token
carries its record id) and revocation is immediate.
"""
from __future__ import annotations

import json

from sndr.product_api.legacy.auth.api_tokens import TokenStore


def test_issue_returns_plaintext_and_persists_only_a_hash(tmp_path):
    store = TokenStore(tmp_path)
    plaintext, token = store.issue("ci-readonly", created_by="admin")
    assert plaintext.startswith("sndr_pat_")
    assert token.label == "ci-readonly" and token.created_by == "admin"
    assert token.prefix in plaintext and token.prefix.startswith("sndr_pat_")
    # the raw file must NOT contain the plaintext secret, only a scrypt hash
    raw = (tmp_path / "api_tokens.json").read_text()
    assert plaintext not in raw
    assert "scrypt$" in raw


def test_verify_roundtrip_and_updates_last_used(tmp_path):
    store = TokenStore(tmp_path)
    plaintext, token = store.issue("prog", created_by="sander")
    assert store.verify(plaintext) == "sander"
    again = next(t for t in store.list() if t.id == token.id)
    assert again.last_used is not None


def test_verify_rejects_bad_revoked_and_malformed(tmp_path):
    store = TokenStore(tmp_path)
    plaintext, token = store.issue("x", created_by="admin")
    assert store.verify("sndr_pat_deadbeef_00ff") is None      # unknown id
    assert store.verify("not-a-token") is None                  # malformed
    assert store.verify(plaintext[:-3] + "000") is None         # wrong secret
    assert store.revoke(token.id) is True
    assert store.verify(plaintext) is None                      # revoked
    assert store.revoke(token.id) is False                      # already gone


def test_list_excludes_secret_material(tmp_path):
    store = TokenStore(tmp_path)
    store.issue("a", created_by="admin")
    store.issue("b", created_by="admin")
    listed = store.list()
    assert len(listed) == 2
    for token in listed:
        assert not hasattr(token, "hash")
        assert token.prefix.startswith("sndr_pat_")


def test_authservice_integration(tmp_path, monkeypatch):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from sndr.product_api.legacy.auth.store import UserStore
    from sndr.product_api.legacy.auth.config import AuthConfig
    from sndr.product_api.legacy.auth.service import AuthService

    store = UserStore(tmp_path / "auth")
    service = AuthService(store, AuthConfig(
        enabled=True, manage_accounts=True, bind_host="127.0.0.1", in_container=False,
        system_user="operator", pam_enabled=False, session_ttl=3600,
        public_base_url="http://127.0.0.1:8765",
    ))
    user = service.create_user(username="opsuser", password="hunter2hunter2", role="operator")
    plaintext, token = service.issue_api_token("ci", created_by=user.username)
    # a Bearer of this token authenticates as the owner
    resolved = service.verify_api_token(plaintext)
    assert resolved is not None and resolved.username == "opsuser"
    assert any(t.id == token.id for t in service.list_api_tokens())
    assert service.revoke_api_token(token.id) is True
    assert service.verify_api_token(plaintext) is None
    # deleted owner ⇒ token no longer resolves
    plaintext2, _ = service.issue_api_token("ci2", created_by="opsuser")
    service.delete_user("opsuser", acting=service.create_user(username="root", password="adminadmin12", role="admin"))
    assert service.verify_api_token(plaintext2) is None

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the at-rest secrets store (SSH passwords / engine API keys).

The file backend is exercised here (deterministic, no OS keychain prompt). The
keyring backend is selected automatically in production when an OS keychain is
available; it shares the same public API.
"""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import secrets_store as ss


@pytest.fixture()
def file_store(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    monkeypatch.setenv("SNDR_SECRETS_BACKEND", "file")
    ss.reset_backend_cache()
    return tmp_path


def test_set_get_delete_roundtrip(file_store):
    assert ss.has_secret("ssh:host-a") is False
    ss.set_secret("ssh:host-a", "hunter2")
    assert ss.has_secret("ssh:host-a") is True
    assert ss.get_secret("ssh:host-a") == "hunter2"
    assert ss.delete_secret("ssh:host-a") is True
    assert ss.get_secret("ssh:host-a") is None
    assert ss.delete_secret("ssh:host-a") is False


def test_value_is_encrypted_at_rest(file_store):
    ss.set_secret("api:prod-35b", "genesis-local-supersecret")
    # The on-disk store must not contain the plaintext secret anywhere.
    blob = b""
    for path in file_store.rglob("*"):
        if path.is_file():
            blob += path.read_bytes()
    assert b"genesis-local-supersecret" not in blob
    # But the value still decrypts back through the API.
    assert ss.get_secret("api:prod-35b") == "genesis-local-supersecret"


def test_get_missing_returns_none(file_store):
    assert ss.get_secret("nope") is None


def test_empty_value_clears(file_store):
    ss.set_secret("k", "v")
    ss.set_secret("k", "")
    assert ss.get_secret("k") is None


def test_backend_name_is_reported(file_store):
    assert ss.backend_name() == "file"
    assert ss.available() is True

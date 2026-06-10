# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the stdlib auth crypto core: passwords, TOTP, sessions."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy.auth import passwords, sessions, totp


# ----------------------------- passwords -----------------------------

def test_password_roundtrip():
    encoded = passwords.hash_password("correct horse battery staple")
    assert encoded.startswith("scrypt$")
    assert passwords.verify_password("correct horse battery staple", encoded)


def test_password_wrong_rejected():
    encoded = passwords.hash_password("hunter2")
    assert not passwords.verify_password("hunter3", encoded)


def test_password_salt_is_random():
    a = passwords.hash_password("same")
    b = passwords.hash_password("same")
    assert a != b  # distinct salts
    assert passwords.verify_password("same", a)
    assert passwords.verify_password("same", b)


def test_password_empty_rejected():
    assert not passwords.verify_password("", "scrypt$1$1$1$x$y")
    with pytest.raises(ValueError):
        passwords.hash_password("")


def test_password_malformed_hash_rejected():
    assert not passwords.verify_password("x", "not-a-hash")
    assert not passwords.verify_password("x", "bcrypt$foo$bar")


# ----------------------------- TOTP -----------------------------

def test_totp_secret_is_base32():
    secret = totp.generate_secret()
    # decodable as base32 (with padding restored)
    import base64
    base64.b32decode(secret + "=" * (-len(secret) % 8))


def test_totp_known_vector_sha1():
    # RFC 6238 test secret "12345678901234567890" -> base32, T=59s, code 287082.
    import base64
    secret = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
    assert totp.hotp(secret, 1) == "287082"


def test_totp_verify_current_and_window():
    secret = totp.generate_secret()
    now = 1_000_000.0
    counter = int(now // 30)
    code = totp.hotp(secret, counter)
    assert totp.verify_totp(secret, code, now=now)
    # previous step accepted within window=1
    prev = totp.hotp(secret, counter - 1)
    assert totp.verify_totp(secret, prev, now=now, window=1)
    # far-out step rejected
    assert not totp.verify_totp(secret, totp.hotp(secret, counter + 5), now=now, window=1)


def test_totp_rejects_garbage():
    secret = totp.generate_secret()
    assert not totp.verify_totp(secret, "abcdef")
    assert not totp.verify_totp(secret, "")
    assert not totp.verify_totp("", "123456")


def test_provisioning_uri_shape():
    uri = totp.provisioning_uri("ABC234", "alice")
    assert uri.startswith("otpauth://totp/")
    assert "secret=ABC234" in uri
    assert "issuer=SNDR" in uri


# ----------------------------- sessions -----------------------------

def test_session_roundtrip():
    key = b"x" * 32
    token = sessions.issue_token(key, "alice", now=1000)
    payload = sessions.verify_token(key, token, now=1001)
    assert payload is not None
    assert payload["u"] == "alice"


def test_session_expired_rejected():
    key = b"x" * 32
    token = sessions.issue_token(key, "alice", ttl=10, now=1000)
    assert sessions.verify_token(key, token, now=1009) is not None
    assert sessions.verify_token(key, token, now=1011) is None


def test_session_tamper_rejected():
    key = b"x" * 32
    token = sessions.issue_token(key, "alice", now=1000)
    body, sig = token.split(".")
    forged = body + "." + ("a" * len(sig))
    assert sessions.verify_token(key, forged, now=1001) is None
    # wrong key rejected
    assert sessions.verify_token(b"y" * 32, token, now=1001) is None


def test_sign_unsign_value():
    key = b"k" * 32
    signed = sessions.sign_value(key, "state-nonce-123")
    assert sessions.unsign_value(key, signed) == "state-nonce-123"
    assert sessions.unsign_value(key, signed + "x") is None
    assert sessions.unsign_value(b"other" * 8, signed) is None

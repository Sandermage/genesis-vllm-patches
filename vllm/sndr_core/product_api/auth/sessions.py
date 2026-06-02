# SPDX-License-Identifier: Apache-2.0
"""Stateless HMAC-signed session tokens.

A session token is ``<base64url payload>.<base64url HMAC-SHA256>``. The payload
carries the username, issued-at and expiry. Verification is constant-time and
checks both the signature and the expiry. The signing key is persisted once
under ``$SNDR_HOME/auth`` so tokens survive a daemon/container restart.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Optional

_DEFAULT_TTL = 86_400  # 24h


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_token(
    secret_key: bytes, username: str, *, epoch: int = 0, ttl: int = _DEFAULT_TTL, now: float | None = None
) -> str:
    """Issue a signed session token for ``username``.

    ``epoch`` binds the token to the account's current token generation; bumping
    the account's epoch (password change / explicit revoke) invalidates every
    previously-issued token.
    """
    issued = int(time.time() if now is None else now)
    payload = {
        "u": username,
        "ep": int(epoch),
        "iat": issued,
        "exp": issued + int(ttl),
        "jti": secrets.token_hex(8),
    }
    body = _b64encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    signature = _b64encode(hmac.new(secret_key, body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{signature}"


def verify_token(secret_key: bytes, token: str, *, now: float | None = None) -> Optional[dict[str, Any]]:
    """Return the payload if ``token`` is validly signed and unexpired, else None."""
    if not token or "." not in token:
        return None
    try:
        body, signature = token.split(".", 1)
        expected = _b64encode(hmac.new(secret_key, body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(_b64decode(body))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    moment = int(time.time() if now is None else now)
    if int(payload.get("exp", 0)) < moment:
        return None
    return payload


def sign_value(secret_key: bytes, value: str) -> str:
    """Sign a short opaque value (e.g. an OAuth state nonce)."""
    mac = _b64encode(hmac.new(secret_key, value.encode("utf-8"), hashlib.sha256).digest())
    return f"{value}.{mac}"


def unsign_value(secret_key: bytes, signed: str) -> Optional[str]:
    """Recover the original value from :func:`sign_value`, or None if tampered."""
    if not signed or "." not in signed:
        return None
    value, mac = signed.rsplit(".", 1)
    expected = _b64encode(hmac.new(secret_key, value.encode("utf-8"), hashlib.sha256).digest())
    if hmac.compare_digest(mac, expected):
        return value
    return None

# SPDX-License-Identifier: Apache-2.0
"""RFC 6238 TOTP (time-based one-time passwords) for 2FA.

Implemented with the standard library only (hmac/hashlib/struct/base64) so no
``pyotp`` dependency is needed. Compatible with Google Authenticator, Authy,
1Password and any other RFC 6238 authenticator app.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import struct
import time
from urllib.parse import quote, urlencode

_DEFAULT_PERIOD = 30
_DEFAULT_DIGITS = 6
_SECRET_BYTES = 20  # 160-bit, the RFC 4226 recommendation for SHA-1.


def generate_secret() -> str:
    """Return a fresh base32 TOTP secret (no padding) for enrolment."""
    return base64.b32encode(secrets.token_bytes(_SECRET_BYTES)).decode("ascii").rstrip("=")


def _b32decode(secret_b32: str) -> bytes:
    padded = secret_b32 + "=" * (-len(secret_b32) % 8)
    return base64.b32decode(padded, casefold=True)


def hotp(secret_b32: str, counter: int, *, digits: int = _DEFAULT_DIGITS) -> str:
    """Return the HOTP value for ``counter`` (RFC 4226)."""
    key = _b32decode(secret_b32)
    message = struct.pack(">Q", counter)
    digest = hmac.new(key, message, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    binary = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(binary % (10 ** digits)).zfill(digits)


def verify_totp(
    secret_b32: str,
    code: str,
    *,
    period: int = _DEFAULT_PERIOD,
    digits: int = _DEFAULT_DIGITS,
    window: int = 1,
    now: float | None = None,
) -> bool:
    """Constant-time verify ``code`` allowing +/- ``window`` time steps of drift."""
    if not secret_b32 or not code:
        return False
    cleaned = code.strip().replace(" ", "")
    if not cleaned.isdigit():
        return False
    moment = time.time() if now is None else now
    counter = int(moment // period)
    matched = False
    # Iterate the full window even on an early match to keep timing uniform.
    for drift in range(-window, window + 1):
        candidate = hotp(secret_b32, counter + drift, digits=digits)
        if hmac.compare_digest(candidate, cleaned):
            matched = True
    return matched


def provisioning_uri(secret_b32: str, account: str, *, issuer: str = "SNDR Control Center") -> str:
    """Build the ``otpauth://`` URI an authenticator app scans as a QR code."""
    label = quote(f"{issuer}:{account}")
    params = urlencode(
        {
            "secret": secret_b32,
            "issuer": issuer,
            "algorithm": "SHA1",
            "digits": _DEFAULT_DIGITS,
            "period": _DEFAULT_PERIOD,
        }
    )
    return f"otpauth://totp/{label}?{params}"

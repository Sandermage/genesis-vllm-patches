# SPDX-License-Identifier: Apache-2.0
"""Password hashing for the GUI auth store.

Uses the standard library ``hashlib.scrypt`` — a memory-hard KDF — so no
third-party hashing dependency (bcrypt/argon2) is required. The encoded form is
self-describing (``scrypt$N$r$p$salt_b64$hash_b64``) so parameters can evolve
without breaking existing hashes.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os

# scrypt cost parameters. N must be a power of two; these are a sensible
# interactive-login default (~16 MiB, low tens of ms on modern hardware).
_SCRYPT_N = 2 ** 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32
_SALT_BYTES = 16
# scrypt requires maxmem >= 128 * N * r; the default (32 MiB) is too low for
# N=16384, r=8 (~16 MiB *with* internal overhead), so set an explicit ceiling.
_MAXMEM = 128 * _SCRYPT_N * _SCRYPT_R * 2


def hash_password(password: str) -> str:
    """Return a self-describing scrypt hash for ``password``."""
    if not password:
        raise ValueError("password must not be empty")
    salt = os.urandom(_SALT_BYTES)
    derived = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_DKLEN,
        maxmem=_MAXMEM,
    )
    return "$".join(
        [
            "scrypt",
            str(_SCRYPT_N),
            str(_SCRYPT_R),
            str(_SCRYPT_P),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(derived).decode("ascii"),
        ]
    )


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time verify ``password`` against a stored scrypt hash."""
    if not password or not encoded:
        return False
    try:
        scheme, n_str, r_str, p_str, salt_b64, hash_b64 = encoded.split("$")
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n_str),
            r=int(r_str),
            p=int(p_str),
            dklen=len(expected),
            maxmem=128 * int(n_str) * int(r_str) * 2,
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(derived, expected)

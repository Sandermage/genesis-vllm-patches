# SPDX-License-Identifier: Apache-2.0
"""Credential-verification backends.

``local`` checks the scrypt hash in the user store. ``pam`` defers to the host
PAM stack (only meaningful when the daemon runs directly on a host, or a
container deliberately bind-mounts the host auth files). PAM is imported lazily
so the dependency is optional.
"""
from __future__ import annotations

from typing import Optional

from . import passwords
from .store import User, UserStore


def verify_local(store: UserStore, username: str, password: str) -> Optional[User]:
    """Verify against the local scrypt-hashed store."""
    user = store.get(username)
    if user is None or user.disabled or not user.password_hash:
        return None
    if passwords.verify_password(password, user.password_hash):
        return user
    return None


def verify_pam(username: str, password: str) -> bool:
    """Verify ``username``/``password`` against the system PAM stack."""
    try:
        import pam  # type: ignore
    except Exception:
        return False
    try:
        authenticator = pam.pam()
        return bool(authenticator.authenticate(username, password, service="login"))
    except Exception:
        return False

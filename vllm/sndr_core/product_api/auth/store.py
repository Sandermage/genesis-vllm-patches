# SPDX-License-Identifier: Apache-2.0
"""Persistent user + credential store for the GUI auth subsystem.

State lives under ``$SNDR_HOME/auth`` (default ``~/.sndr/auth``) as ``users.json``
plus a ``session.key`` signing key. Writes are atomic (temp + ``os.replace``) and
files are created ``0600`` in a ``0700`` directory. Because the directory sits in
``$SNDR_HOME``, mounting that path as a container volume preserves all accounts,
2FA enrolments and the signing key across restarts.
"""
from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

_LOCK = threading.RLock()

VALID_ROLES = ("admin", "operator", "viewer")


def default_auth_dir() -> Path:
    """Resolve ``$SNDR_HOME/auth`` (honouring the legacy ``GENESIS_HOME`` alias)."""
    home = os.environ.get("SNDR_HOME") or os.environ.get("GENESIS_HOME")
    base = Path(home).expanduser() if home else (Path.home() / ".sndr")
    return base / "auth"


@dataclass
class User:
    username: str
    role: str = "operator"
    password_hash: Optional[str] = None  # None for OAuth-only / external accounts
    totp_secret: Optional[str] = None
    totp_enabled: bool = False
    source: str = "local"  # local | pam | oauth:google | oauth:apple
    oauth_subject: Optional[str] = None
    email: Optional[str] = None
    disabled: bool = False
    created_at: float = field(default_factory=time.time)
    last_login: Optional[float] = None
    # Session generation: bumped on password change / explicit revoke to
    # invalidate every previously-issued token.
    token_epoch: int = 0
    # Hashed single-use 2FA recovery codes (never the plaintext).
    recovery_codes: list = field(default_factory=list)

    def public_dict(self) -> dict:
        """Serializable view with all secret material stripped."""
        return {
            "username": self.username,
            "role": self.role,
            "source": self.source,
            "email": self.email,
            "totp_enabled": self.totp_enabled,
            "has_password": bool(self.password_hash),
            "recovery_codes_remaining": len(self.recovery_codes),
            "disabled": self.disabled,
            "created_at": self.created_at,
            "last_login": self.last_login,
        }


class UserStore:
    """Thread-safe, file-backed user store."""

    def __init__(self, auth_dir: Optional[Path] = None) -> None:
        self._dir = Path(auth_dir) if auth_dir else default_auth_dir()
        self._users_path = self._dir / "users.json"
        self._key_path = self._dir / "session.key"
        self._users: dict[str, User] = {}
        self._loaded = False

    # ---- persistence ----

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self._dir, 0o700)
        except OSError:
            pass  # best-effort on platforms without POSIX perms

    def load(self) -> None:
        with _LOCK:
            self._users = {}
            if self._users_path.exists():
                try:
                    raw = json.loads(self._users_path.read_text("utf-8"))
                except (json.JSONDecodeError, OSError):
                    raw = {}
                for record in raw.get("users", []):
                    try:
                        user = User(**record)
                    except TypeError:
                        continue
                    self._users[user.username] = user
            self._loaded = True

    def _save(self) -> None:
        self._ensure_dir()
        payload = {"version": 1, "users": [asdict(user) for user in self._users.values()]}
        tmp = self._users_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), "utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self._users_path)

    def _require_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ---- signing key ----

    def signing_key(self) -> bytes:
        """Load (or generate once) the persistent session-signing key."""
        with _LOCK:
            self._ensure_dir()
            if self._key_path.exists():
                # Raw key bytes — never strip(): a leading/trailing whitespace
                # byte (0x09/0x0a/0x0d/0x20) is valid key material and stripping
                # it would silently invalidate every session after a restart.
                data = self._key_path.read_bytes()
                if len(data) >= 32:
                    return data
            key = secrets.token_bytes(48)
            tmp = self._key_path.with_suffix(".key.tmp")
            tmp.write_bytes(key)
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, self._key_path)
            return key

    # ---- queries ----

    def get(self, username: str) -> Optional[User]:
        with _LOCK:
            self._require_loaded()
            return self._users.get(username)

    def list_users(self) -> list[User]:
        with _LOCK:
            self._require_loaded()
            return sorted(self._users.values(), key=lambda u: u.username)

    def count(self) -> int:
        with _LOCK:
            self._require_loaded()
            return len(self._users)

    # ---- mutations ----

    def put(self, user: User) -> User:
        with _LOCK:
            self._require_loaded()
            self._users[user.username] = user
            self._save()
            return user

    def delete(self, username: str) -> bool:
        with _LOCK:
            self._require_loaded()
            if username in self._users:
                del self._users[username]
                self._save()
                return True
            return False

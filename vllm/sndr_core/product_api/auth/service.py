# SPDX-License-Identifier: Apache-2.0
"""AuthService — the facade the HTTP layer talks to.

Ties together the user store, config, password/TOTP/session crypto and the
verification backends. Owns first-run admin bootstrap and the login state
machine (password -> optional 2FA -> session token).
"""
from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from . import backends, passwords, sessions, totp
from .config import AuthConfig
from .ratelimit import LoginGuard
from .store import VALID_ROLES, User, UserStore

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9._@-]{1,64}$")
_RECOVERY_CODE_COUNT = 10


class AuthError(Exception):
    """Raised for caller-correctable auth failures (bad input, forbidden)."""


@dataclass
class LoginResult:
    ok: bool
    user: Optional[User] = None
    needs_2fa: bool = False
    token: Optional[str] = None
    error: Optional[str] = None
    locked: bool = False


def valid_username(username: str) -> bool:
    return bool(username and _USERNAME_RE.match(username))


class AuthService:
    def __init__(self, store: UserStore, config: AuthConfig) -> None:
        self.store = store
        self.config = config
        self._key = store.signing_key()
        self.guard = LoginGuard()
        from .api_tokens import TokenStore

        self.tokens = TokenStore(store._dir)

    # ---- API tokens (managed personal access tokens) ----

    def issue_api_token(self, label: str, *, created_by: str):
        return self.tokens.issue(label, created_by=created_by)

    def list_api_tokens(self):
        return self.tokens.list()

    def revoke_api_token(self, token_id: str) -> bool:
        return self.tokens.revoke(token_id)

    def verify_api_token(self, plaintext: str) -> Optional[User]:
        """Resolve a Bearer API token to its owning user (or None)."""
        username = self.tokens.verify(plaintext)
        if not username:
            return None
        return self.store.get(username)

    # ---- bootstrap ----

    def bootstrap(self, *, admin_password: Optional[str] = None) -> Optional[str]:
        """Create the first admin if the store is empty. Returns the generated
        password if one was auto-created (so the daemon can print it once), or
        ``None`` if an admin already exists or a password was supplied.
        """
        if self.store.count() > 0:
            return None
        username = self.config.system_user if valid_username(self.config.system_user) else "admin"
        generated: Optional[str] = None
        if not admin_password:
            generated = secrets.token_urlsafe(15)
            admin_password = generated
        self.store.put(
            User(
                username=username,
                role="admin",
                password_hash=passwords.hash_password(admin_password),
                source="local",
            )
        )
        return generated

    # ---- login state machine ----

    def authenticate(self, username: str, password: str) -> LoginResult:
        if not username or not password:
            return LoginResult(ok=False, error="Username and password are required.")
        locked = self.guard.retry_after(username)
        if locked:
            return LoginResult(ok=False, locked=True, error=f"Too many attempts. Try again in {locked}s.")
        user = backends.verify_local(self.store, username, password)
        if user is None and self.config.pam_enabled and backends.verify_pam(username, password):
            user = self._ensure_pam_user(username)
        if user is None:
            self.guard.record_failure(username)
            return LoginResult(ok=False, error="Invalid username or password.")
        if user.disabled:
            return LoginResult(ok=False, error="Account is disabled.")
        # Password is correct; the 2FA step has its own throttle keyed below.
        self.guard.record_success(username)
        if user.totp_enabled:
            return LoginResult(ok=True, user=user, needs_2fa=True)
        return LoginResult(ok=True, user=user, token=self.issue_session(user))

    def complete_2fa(self, username: str, code: str) -> LoginResult:
        key = f"2fa:{username}"
        locked = self.guard.retry_after(key)
        if locked:
            return LoginResult(ok=False, locked=True, error=f"Too many attempts. Try again in {locked}s.")
        user = self.store.get(username)
        if user is None or not user.totp_enabled or not user.totp_secret:
            return LoginResult(ok=False, error="Two-factor is not enabled for this account.")
        if totp.verify_totp(user.totp_secret, code) or self._consume_recovery_code(user, code):
            self.guard.record_success(key)
            return LoginResult(ok=True, user=user, token=self.issue_session(user))
        self.guard.record_failure(key)
        return LoginResult(ok=False, error="Invalid authentication or recovery code.")

    def _ensure_pam_user(self, username: str) -> User:
        existing = self.store.get(username)
        if existing:
            return existing
        return self.store.put(User(username=username, role="operator", source="pam"))

    # ---- sessions ----

    def issue_session(self, user: User) -> str:
        user.last_login = time.time()
        self.store.put(user)
        return sessions.issue_token(
            self._key, user.username, epoch=user.token_epoch, ttl=self.config.session_ttl
        )

    def verify_session(self, token: str) -> Optional[User]:
        payload = sessions.verify_token(self._key, token)
        if not payload:
            return None
        user = self.store.get(str(payload.get("u", "")))
        if user is None or user.disabled:
            return None
        # Reject tokens issued before the account's current epoch (revoked).
        if int(payload.get("ep", 0)) != user.token_epoch:
            return None
        return user

    def revoke_sessions(self, user: User) -> None:
        """Invalidate all of this account's existing session tokens."""
        user.token_epoch += 1
        self.store.put(user)

    def sign_state(self, value: str) -> str:
        return sessions.sign_value(self._key, value)

    def unsign_state(self, signed: str) -> Optional[str]:
        return sessions.unsign_value(self._key, signed)

    # ---- user management (admin) ----

    def create_user(self, *, username: str, password: str, role: str = "operator") -> User:
        if not valid_username(username):
            raise AuthError("Username must be 1-64 chars: letters, digits, . _ @ -")
        if role not in VALID_ROLES:
            raise AuthError(f"Role must be one of {', '.join(VALID_ROLES)}.")
        if self.store.get(username):
            raise AuthError("A user with that name already exists.")
        if not password or len(password) < 8:
            raise AuthError("Password must be at least 8 characters.")
        return self.store.put(
            User(username=username, role=role, password_hash=passwords.hash_password(password), source="local")
        )

    def delete_user(self, username: str, *, acting: User) -> None:
        if username == acting.username:
            raise AuthError("You cannot delete your own account.")
        target = self.store.get(username)
        if target is None:
            raise AuthError("No such user.")
        if target.role == "admin" and self._admin_count() <= 1:
            raise AuthError("Cannot delete the last admin.")
        self.store.delete(username)

    def set_password(self, user: User, *, current: Optional[str], new: str, by_admin: bool = False) -> None:
        if not new or len(new) < 8:
            raise AuthError("Password must be at least 8 characters.")
        if not by_admin:
            if not user.password_hash or not passwords.verify_password(current or "", user.password_hash):
                raise AuthError("Current password is incorrect.")
        user.password_hash = passwords.hash_password(new)
        # A password change invalidates every existing session token.
        user.token_epoch += 1
        self.store.put(user)

    def _admin_count(self) -> int:
        return sum(1 for u in self.store.list_users() if u.role == "admin" and not u.disabled)

    # ---- 2FA enrolment ----

    def enroll_2fa(self, user: User) -> tuple[str, str]:
        """Generate a pending TOTP secret (not yet active) + provisioning URI."""
        secret = totp.generate_secret()
        user.totp_secret = secret
        user.totp_enabled = False
        self.store.put(user)
        return secret, totp.provisioning_uri(secret, user.username)

    def activate_2fa(self, user: User, code: str) -> list[str]:
        """Activate 2FA and return one-time recovery codes (shown once)."""
        if not user.totp_secret:
            raise AuthError("Start 2FA enrolment first.")
        if not totp.verify_totp(user.totp_secret, code):
            raise AuthError("Code did not match — check your authenticator app.")
        user.totp_enabled = True
        plaintext = self._fresh_recovery_codes(user)
        self.store.put(user)
        return plaintext

    def regenerate_recovery_codes(self, user: User) -> list[str]:
        if not user.totp_enabled:
            raise AuthError("Enable 2FA first.")
        plaintext = self._fresh_recovery_codes(user)
        self.store.put(user)
        return plaintext

    def disable_2fa(self, user: User) -> None:
        user.totp_enabled = False
        user.totp_secret = None
        user.recovery_codes = []
        self.store.put(user)

    def _fresh_recovery_codes(self, user: User) -> list[str]:
        plaintext = [f"{secrets.token_hex(2)}-{secrets.token_hex(2)}-{secrets.token_hex(2)}" for _ in range(_RECOVERY_CODE_COUNT)]
        user.recovery_codes = [passwords.hash_password(code) for code in plaintext]
        return plaintext

    def _consume_recovery_code(self, user: User, code: str) -> bool:
        """Verify + single-use a recovery code; persists removal on success."""
        candidate = (code or "").strip().lower()
        if not candidate:
            return False
        for stored in list(user.recovery_codes):
            if passwords.verify_password(candidate, stored):
                user.recovery_codes.remove(stored)
                self.store.put(user)
                return True
        return False

    # ---- oauth account linking ----

    def upsert_oauth_user(self, *, provider: str, subject: str, email: Optional[str]) -> User:
        # Match an existing link first, then an existing local account by email.
        for user in self.store.list_users():
            if user.source == f"oauth:{provider}" and user.oauth_subject == subject:
                return user
        username = email or f"{provider}:{subject}"
        existing = self.store.get(username)
        if existing:
            existing.oauth_subject = subject
            existing.email = email or existing.email
            return self.store.put(existing)
        # OAuth sign-ups are never auto-admin (no race-to-be-first escalation);
        # an existing admin promotes them explicitly.
        role = "operator"
        return self.store.put(
            User(
                username=username,
                role=role,
                source=f"oauth:{provider}",
                oauth_subject=subject,
                email=email,
            )
        )

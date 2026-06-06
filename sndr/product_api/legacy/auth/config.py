# SPDX-License-Identifier: Apache-2.0
"""Auth configuration + deployment-context detection.

The auth subsystem adapts to *where* it runs. We detect whether the daemon is
inside a container, which system user owns the process, and which optional
backends (PAM, Google/Apple OAuth) are available and configured, then derive a
single :class:`AuthConfig` the HTTP layer consumes.
"""
from __future__ import annotations

import getpass
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def detect_container() -> bool:
    """Best-effort detection of a container runtime (Docker/Podman/K8s)."""
    if _truthy(os.environ.get("SNDR_IN_CONTAINER")):
        return True
    for marker in ("/.dockerenv", "/run/.containerenv"):
        if Path(marker).exists():
            return True
    try:
        cgroup = Path("/proc/1/cgroup").read_text("utf-8")
        if any(tag in cgroup for tag in ("docker", "containerd", "kubepods", "libpod")):
            return True
    except OSError:
        pass
    return False


def system_user() -> str:
    """The OS account running the daemon (used to seed the first admin)."""
    try:
        return getpass.getuser() or "admin"
    except Exception:  # pragma: no cover - getuser can raise on odd environments
        return os.environ.get("USER") or os.environ.get("USERNAME") or "admin"


def pam_available() -> bool:
    """True if a PAM binding is importable (python-pam / pam)."""
    try:
        import pam  # noqa: F401

        return True
    except Exception:
        return False


@dataclass(frozen=True)
class OAuthProvider:
    name: str          # "google" | "apple"
    client_id: str
    client_secret: str
    enabled: bool = True


@dataclass(frozen=True)
class AuthConfig:
    enabled: bool
    manage_accounts: bool
    bind_host: str
    in_container: bool
    system_user: str
    pam_enabled: bool
    session_ttl: int
    # Public base URL the browser reaches us on — needed for OAuth redirect URIs.
    public_base_url: str
    oauth: dict[str, OAuthProvider] = field(default_factory=dict)
    # Legacy single shared token (kept working for API/service clients).
    legacy_token: str = ""

    @property
    def backends(self) -> list[str]:
        out = ["local"]
        if self.pam_enabled:
            out.append("pam")
        out.extend(f"oauth:{name}" for name in self.oauth)
        return out


def _resolve_oauth(public_base_url: str) -> dict[str, OAuthProvider]:
    providers: dict[str, OAuthProvider] = {}
    g_id = os.environ.get("SNDR_OAUTH_GOOGLE_CLIENT_ID", "").strip()
    g_secret = os.environ.get("SNDR_OAUTH_GOOGLE_CLIENT_SECRET", "").strip()
    if g_id and g_secret:
        providers["google"] = OAuthProvider("google", g_id, g_secret)
    a_id = os.environ.get("SNDR_OAUTH_APPLE_CLIENT_ID", "").strip()
    a_secret = os.environ.get("SNDR_OAUTH_APPLE_CLIENT_SECRET", "").strip()
    if a_id and a_secret:
        providers["apple"] = OAuthProvider("apple", a_id, a_secret)
    return providers


def load_config(*, bind_host: str = "127.0.0.1", has_users: bool = False) -> AuthConfig:
    """Resolve the effective auth configuration from the environment.

    ``SNDR_AUTH`` controls enablement: ``on``/``off`` force it; the default
    ``auto`` enables auth when the daemon is exposed beyond loopback or when an
    account store already exists (so a configured server stays protected).
    """
    mode = (os.environ.get("SNDR_AUTH") or os.environ.get("SNDR_AUTH_ENABLED") or "auto").strip().lower()
    legacy_token = os.environ.get("SNDR_GUI_TOKEN", "").strip()
    loopback = bind_host in {"127.0.0.1", "localhost", "::1", ""}
    if mode in {"1", "true", "yes", "on"}:
        enabled = True
    elif mode in {"0", "false", "no", "off"}:
        enabled = False
    else:  # auto: protect when exposed beyond loopback, when accounts exist,
        # or when a shared token is explicitly configured (legacy opt-in).
        enabled = (not loopback) or has_users or bool(legacy_token)

    # Account management (login, user store, first-run admin bootstrap) is a
    # superset of plain token enforcement: a pure shared-token deployment
    # enforces the token but never provisions user accounts.
    if mode in {"1", "true", "yes", "on"}:
        manage_accounts = True
    elif mode in {"0", "false", "no", "off"}:
        manage_accounts = False
    else:
        manage_accounts = (not loopback) or has_users

    public_base_url = (
        os.environ.get("SNDR_PUBLIC_URL", "").strip()
        or f"http://{bind_host if not loopback else '127.0.0.1'}:8765"
    )
    try:
        ttl = int(os.environ.get("SNDR_AUTH_SESSION_TTL", "86400"))
    except ValueError:
        ttl = 86_400

    return AuthConfig(
        enabled=enabled,
        manage_accounts=manage_accounts,
        bind_host=bind_host,
        in_container=detect_container(),
        system_user=system_user(),
        pam_enabled=_truthy(os.environ.get("SNDR_AUTH_PAM")) and pam_available(),
        session_ttl=ttl,
        public_base_url=public_base_url,
        oauth=_resolve_oauth(public_base_url),
        legacy_token=legacy_token,
    )

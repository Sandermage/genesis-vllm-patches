# SPDX-License-Identifier: Apache-2.0
"""GUI authentication subsystem.

Pluggable, dependency-light auth for the SNDR Product API daemon:

* local accounts (scrypt-hashed passwords) persisted under ``$SNDR_HOME/auth``
* optional system login via PAM (host deployments)
* optional Google / Apple OAuth (OIDC) when credentials are configured
* TOTP two-factor (RFC 6238)
* HMAC-signed session tokens

The configuration adapts to the deployment context (container vs host) — see
:func:`config.load_config`.
"""
from __future__ import annotations

from .config import AuthConfig, load_config
from .service import AuthError, AuthService, LoginResult
from .store import User, UserStore

__all__ = [
    "AuthConfig",
    "load_config",
    "AuthError",
    "AuthService",
    "LoginResult",
    "User",
    "UserStore",
]

# SPDX-License-Identifier: Apache-2.0
"""Google / Apple OAuth 2.0 + OIDC (authorization-code flow).

Dependency-light: uses ``urllib`` for the token exchange. Because the ID token
is received directly from the provider's token endpoint over a TLS channel with
client authentication, OIDC Core 3.1.3.7 permits consuming its claims without a
separate signature check, so no ``cryptography``/JWKS round-trip is required.

These providers are inert until the operator supplies client credentials via
``SNDR_OAUTH_GOOGLE_CLIENT_ID``/``..._SECRET`` (and the Apple equivalents) and
registers ``<public_base_url>/api/v1/auth/oauth/<provider>/callback`` as a
redirect URI.
"""
from __future__ import annotations

import base64
import json
import urllib.parse
import urllib.request
from typing import Optional

from .config import OAuthProvider

_ENDPOINTS = {
    "google": {
        "authorize": "https://accounts.google.com/o/oauth2/v2/auth",
        "token": "https://oauth2.googleapis.com/token",
        "scope": "openid email profile",
    },
    "apple": {
        "authorize": "https://appleid.apple.com/auth/authorize",
        "token": "https://appleid.apple.com/auth/token",
        "scope": "openid email name",
    },
}


def redirect_uri(public_base_url: str, provider: str) -> str:
    return f"{public_base_url.rstrip('/')}/api/v1/auth/oauth/{provider}/callback"


def authorize_url(provider: OAuthProvider, public_base_url: str, *, state: str, nonce: str) -> str:
    meta = _ENDPOINTS[provider.name]
    params = {
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri(public_base_url, provider.name),
        "response_type": "code",
        "scope": meta["scope"],
        "state": state,
        "nonce": nonce,
    }
    if provider.name == "apple":
        # Apple returns the email/name in the POST body the first time only.
        params["response_mode"] = "form_post"
    return f"{meta['authorize']}?{urllib.parse.urlencode(params)}"


def _decode_jwt_claims(id_token: str) -> dict:
    """Decode (not verify) the claims segment of a JWT. Safe here: the token came
    straight from the provider's token endpoint over TLS (OIDC 3.1.3.7)."""
    try:
        _header, payload, _sig = id_token.split(".")
        padded = payload + "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, TypeError, json.JSONDecodeError):
        return {}


def exchange_code(provider: OAuthProvider, public_base_url: str, code: str, *, timeout: float = 10.0) -> dict:
    meta = _ENDPOINTS[provider.name]
    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri(public_base_url, provider.name),
            "client_id": provider.client_id,
            "client_secret": provider.client_secret,
        }
    ).encode("ascii")
    request = urllib.request.Request(
        meta["token"],
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    from ..hub import ssl_context

    with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:  # noqa: S310 - fixed provider hosts
        return json.loads(response.read().decode("utf-8"))


def identity_from_token_response(token_response: dict, *, expected_nonce: Optional[str] = None) -> Optional[dict]:
    """Extract ``{sub, email}`` from a token response, checking the nonce."""
    id_token = token_response.get("id_token")
    if not id_token:
        return None
    claims = _decode_jwt_claims(id_token)
    subject = claims.get("sub")
    if not subject:
        return None
    if expected_nonce is not None and claims.get("nonce") not in (None, expected_nonce):
        return None
    return {"sub": str(subject), "email": claims.get("email")}

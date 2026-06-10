# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OAuth URL building + ID-token claim extraction (no network)."""
from __future__ import annotations

import base64
import json

from sndr.product_api.legacy.auth import oauth
from sndr.product_api.legacy.auth.config import OAuthProvider


def _fake_id_token(claims: dict) -> str:
    def seg(obj):
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")

    return f"{seg({'alg': 'RS256'})}.{seg(claims)}.sig"


def test_authorize_url_google():
    provider = OAuthProvider("google", "client-123", "secret")
    url = oauth.authorize_url(provider, "https://gui.example.com", state="st", nonce="nc")
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=client-123" in url
    assert "redirect_uri=https%3A%2F%2Fgui.example.com%2Fapi%2Fv1%2Fauth%2Foauth%2Fgoogle%2Fcallback" in url
    assert "state=st" in url and "nonce=nc" in url
    assert "response_mode" not in url


def test_authorize_url_apple_uses_form_post():
    provider = OAuthProvider("apple", "com.example.app", "secret")
    url = oauth.authorize_url(provider, "https://gui.example.com", state="st", nonce="nc")
    assert url.startswith("https://appleid.apple.com/auth/authorize?")
    assert "response_mode=form_post" in url


def test_identity_extracted_and_nonce_checked():
    token_response = {"id_token": _fake_id_token({"sub": "abc-123", "email": "a@b.com", "nonce": "nc"})}
    ident = oauth.identity_from_token_response(token_response, expected_nonce="nc")
    assert ident == {"sub": "abc-123", "email": "a@b.com"}
    # wrong nonce rejected
    assert oauth.identity_from_token_response(token_response, expected_nonce="other") is None
    # missing id_token rejected
    assert oauth.identity_from_token_response({}, expected_nonce="nc") is None


def test_redirect_uri():
    assert oauth.redirect_uri("https://x.io/", "google") == "https://x.io/api/v1/auth/oauth/google/callback"

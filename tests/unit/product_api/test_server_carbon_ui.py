# SPDX-License-Identifier: Apache-2.0
"""Tests for the new modular sndr.product_api server and its Carbon UI mount.

The modular server (``sndr.product_api.server:create_app``) serves the
enveloped ``{data, meta}`` API and, when a built Carbon bundle is present,
mounts it as a history-routed SPA. These tests pin both contracts.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sndr.product_api import server as server_mod  # noqa: E402
from sndr.product_api.server import create_app  # noqa: E402


def test_health_is_enveloped():
    client = TestClient(create_app())
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"data", "meta"}
    assert body["meta"]["api_version"] == "v1"
    assert "request_id" in body["meta"]


def test_carbon_ui_absent_keeps_api_only(monkeypatch):
    """With no bundle resolvable, the server stays API-only:
    '/' is not served as a static SPA (404), API still works."""
    monkeypatch.setattr(server_mod, "_resolve_carbon_static_dir", lambda: None)
    client = TestClient(create_app())
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/").status_code == 404


def _build_fake_bundle(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text(
        '<!doctype html><html><body><div id="root"></div>'
        '<script src="/assets/app-abc123.js"></script></body></html>'
    )
    (tmp_path / "assets" / "app-abc123.js").write_text("console.log('sndr');")
    return tmp_path


def test_carbon_ui_serving(monkeypatch, tmp_path):
    _build_fake_bundle(tmp_path)
    monkeypatch.setenv("SNDR_GUI_STATIC_CARBON", str(tmp_path))
    client = TestClient(create_app())

    # API wins over the "/" mount.
    assert client.get("/api/v1/health").status_code == 200

    # Root serves index.html, must revalidate.
    root = client.get("/")
    assert root.status_code == 200
    assert 'id="root"' in root.text
    assert root.headers["cache-control"] == "no-cache"

    # Client-routed deep links fall back to index.html (BrowserRouter).
    for route in ("/fleet", "/engines/vllm", "/patches"):
        r = client.get(route)
        assert r.status_code == 200, route
        assert 'id="root"' in r.text, route

    # Content-hashed assets are immutable.
    asset = client.get("/assets/app-abc123.js")
    assert asset.status_code == 200
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"

    # A missing asset is a genuine 404 — no HTML fallback for /assets/*.
    assert client.get("/assets/missing.js").status_code == 404

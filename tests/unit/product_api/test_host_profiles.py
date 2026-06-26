# SPDX-License-Identifier: Apache-2.0
"""Tests for operator-local host profile persistence."""
from __future__ import annotations

from sndr.product_api.legacy.host_profiles import (
    delete_host_profile,
    list_host_profiles,
    upsert_host_profile,
)


def test_upsert_list_and_delete_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    assert list_host_profiles() == ()

    profile = upsert_host_profile({
        "label": "GPU Build 01",
        "host": "gpu-build-01",
        "transport": "ssh",
        "ssh_target": "user@gpu-build-01",
        "port": 8765,
        "notes": "2x A5000",
    })
    assert profile.id == "gpu-build-01"
    assert profile.transport == "ssh"
    assert profile.port == 8765

    listed = list_host_profiles()
    assert len(listed) == 1
    assert listed[0].label == "GPU Build 01"

    # Upsert with same id updates rather than duplicates.
    updated = upsert_host_profile({"id": "gpu-build-01", "label": "GPU Build 01", "host": "gpu-build-01", "notes": "updated"})
    assert updated.notes == "updated"
    assert len(list_host_profiles()) == 1

    assert delete_host_profile("gpu-build-01") is True
    assert list_host_profiles() == ()
    assert delete_host_profile("gpu-build-01") is False


def test_upsert_requires_label_or_host(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    import pytest

    with pytest.raises(ValueError):
        upsert_host_profile({"notes": "no id/label/host"})


def test_enterprise_fields_persist(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    profile = upsert_host_profile({
        "label": "Prod A5000",
        "host": "192.0.2.10",
        "transport": "ssh",
        "ssh_target": "user@192.0.2.10",
        "port": 8765,
        "role": "production",
        "hardware": "2x A5000 24GB",
        "gpus": 2,
        "engine_port": 8101,
        "tags": ["27b", "tq-k8v4"],
    })
    assert profile.role == "production"
    assert profile.hardware == "2x A5000 24GB"
    assert profile.gpus == 2
    assert profile.engine_port == 8101
    assert profile.tags == ("27b", "tq-k8v4")
    again = list_host_profiles()[0]
    assert again.role == "production" and again.gpus == 2 and again.engine_port == 8101
    assert again.tags == ("27b", "tq-k8v4")


def test_legacy_rows_get_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    upsert_host_profile({"label": "Legacy", "host": "old-host"})
    p = list_host_profiles()[0]
    assert p.role == "" and p.hardware == "" and p.gpus == 0
    assert p.engine_port == 8000 and p.tags == () and p.api_key == ""


def test_ssh_fields_persist(monkeypatch, tmp_path):
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    profile = upsert_host_profile({
        "label": "Prod", "host": "192.0.2.10",
        "ssh_user": "operator", "ssh_auth": "key", "ssh_key_path": "~/.ssh/id_ed25519", "ssh_port": 2222,
    })
    assert profile.ssh_user == "operator" and profile.ssh_auth == "key"
    assert profile.ssh_key_path == "~/.ssh/id_ed25519" and profile.ssh_port == 2222
    again = list_host_profiles()[0]
    assert again.ssh_user == "operator" and again.ssh_port == 2222
    # Defaults for a legacy/minimal row.
    minimal = upsert_host_profile({"label": "Min", "host": "h2"})
    assert minimal.ssh_auth == "agent" and minimal.ssh_port == 22


def test_api_key_is_never_persisted_or_exposed_by_profile(monkeypatch, tmp_path):
    """The engine key must not land on disk or in the GUI payload — it lives
    encrypted in the secrets store (handled by the HTTP layer). Mirrors how the
    SSH password is treated."""
    from sndr.product_api.legacy.host_profiles import _read, host_profile_payload

    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    profile = upsert_host_profile({
        "label": "Prod 35B", "host": "192.0.2.10", "engine_port": 8102,
        "api_key": "genesis-local",
    })
    # Not persisted to disk and not echoed back through the profile object.
    assert "api_key" not in _read()[0]
    assert profile.api_key == ""
    assert list_host_profiles()[0].api_key == ""
    # The GUI payload never carries the raw key.
    assert "api_key" not in host_profile_payload(profile)


def test_probe_host_unreachable_is_graceful():
    from sndr.product_api.legacy import engine_client
    # 127.0.0.1 on an unused high port — refused, not an exception escape
    out = engine_client.probe_host("127.0.0.1", 65535, timeout=0.5)
    assert out["reachable"] is False
    assert out["host"] == "127.0.0.1" and out["port"] == 65535
    assert out["error"]


def test_probe_clamps_bad_port():
    from sndr.product_api.legacy import engine_client
    out = engine_client.probe_host("127.0.0.1", 999999, timeout=0.3)
    assert out["port"] == 8000


def test_host_routes(monkeypatch, tmp_path):
    import pytest
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from fastapi.testclient import TestClient

    from sndr.product_api.legacy import engine_client
    from sndr.product_api.legacy.http_app import create_app

    seen_keys: list = []

    def _probe(host, port, api_key=None):
        seen_keys.append(api_key)  # capture what the daemon resolved server-side
        return {
            "reachable": True, "host": host, "port": port, "version": "0.20.2", "models": ["qwen3.6-27b"],
            "latency_ms": 4.2, "base_url": f"http://{host}:{port}/v1", "error": None,
        }

    monkeypatch.setattr(engine_client, "probe_host", _probe)
    client = TestClient(create_app(allowed_origins=()))

    # CRUD with enterprise fields + a key-protected engine.
    up = client.post("/api/v1/hosts", json={
        "label": "Prod", "host": "192.0.2.10", "ssh_target": "user@192.0.2.10",
        "role": "production", "hardware": "2x A5000", "gpus": 2, "engine_port": 8101, "tags": ["27b"],
        "api_key": "genesis-local",
    })
    assert up.status_code == 200 and up.json()["role"] == "production" and up.json()["gpus"] == 2
    # The raw key is NEVER returned — only a boolean presence flag.
    assert "api_key" not in up.json()
    assert up.json()["has_api_key"] is True

    # GET /hosts likewise masks the key.
    listed = client.get("/api/v1/hosts").json()["hosts"][0]
    assert "api_key" not in listed and listed["has_api_key"] is True

    # /install/targets must not leak it either.
    for h in client.get("/api/v1/install/targets").json()["hosts"]:
        assert "api_key" not in h

    # probe — the daemon resolves the stored key server-side from host_id.
    probe = client.get("/api/v1/hosts/probe?host=192.0.2.10&port=8101&host_id=prod")
    assert probe.status_code == 200 and probe.json()["reachable"] is True and probe.json()["version"] == "0.20.2"
    assert seen_keys[-1] == "genesis-local"  # injected without the browser ever holding it

    # inventory
    inv = client.get("/api/v1/host/inventory")
    assert inv.status_code == 200 and {"os", "python", "docker", "nvidia", "vllm"} <= set(inv.json())

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the fleet overview aggregation (read-only, concurrent)."""
from __future__ import annotations

from dataclasses import dataclass

from sndr.product_api.legacy import fleet


@dataclass
class _Profile:
    id: str
    label: str
    host: str
    transport: str = "ssh"
    ssh_user: str = "sander"
    ssh_auth: str = "agent"
    ssh_key_path: str = ""
    ssh_target: str = ""
    ssh_port: int = 22
    role: str = ""
    gpu_arch: str = ""


def _discover_ok(target):
    return {
        "available": True, "docker": True, "error": None,
        "engines": [{"container": "vllm-gemma4", "host_port": 8102, "image": "vllm/vllm-openai:nightly",
                     "status": "Up", "ports": "", "genesis_flags": ["GENESIS_ENABLE_A", "GENESIS_ENABLE_B"]}],
        "gpus": [{"name": "RTX A5000", "memory_total_mib": "24564", "arch": "Ampere", "utilization": "0"},
                 {"name": "RTX A5000", "memory_total_mib": "24564", "arch": "Ampere", "utilization": "0"}],
        "interconnect": {"has_nvlink": False, "worst_link": "PCIe", "note": "x"},
    }


def _probe_ok(host, port, *, api_key=None):
    return {"reachable": True, "host": host, "port": port, "version": "0.21.1", "models": ["gemma-4-31b"]}


def test_summarize_host_folds_discovery_and_probe():
    out = fleet.summarize_host(_Profile("srv2", "Srv 2", "192.168.1.11"), discover=_discover_ok, probe=_probe_ok)
    assert out["ssh_ok"] is True
    assert out["models"] == ["gemma-4-31b"] and out["vllm_version"] == "0.21.1"
    assert out["gpu_count"] == 2 and out["arch"] == "Ampere" and out["interconnect"] == "PCIe"
    assert out["active_patches"] == 2  # two GENESIS_ENABLE_*=1 flags
    assert out["engines"][0]["reachable"] is True and out["engines"][0]["port"] == 8102


def test_one_host_failure_does_not_break_others():
    def discover(target):
        if "1.12" in target["host"]:
            raise OSError("connection refused")
        return _discover_ok(target)

    profiles = [_Profile("a", "A", "192.168.1.11"), _Profile("b", "B", "192.168.1.12")]
    rows = fleet.collect_fleet_overview(profiles, discover=discover, probe=_probe_ok)
    by_id = {r["id"]: r for r in rows}
    assert by_id["a"]["ssh_ok"] is True
    assert by_id["b"]["ssh_ok"] is False and "refused" in by_id["b"]["error"]


def test_only_engine_hosts_are_included():
    profiles = [
        _Profile("eng", "Engine", "10.0.0.1", transport="ssh"),
        _Profile("loc", "Local", "127.0.0.1", transport="local", ssh_user=""),
    ]
    rows = fleet.collect_fleet_overview(profiles, discover=_discover_ok, probe=_probe_ok)
    assert {r["id"] for r in rows} == {"eng"}  # the local daemon host is not a fleet server


def test_probe_failure_keeps_host_in_fleet():
    def probe(host, port, *, api_key=None):
        raise ConnectionError("engine down")

    out = fleet.summarize_host(_Profile("s", "S", "h"), discover=_discover_ok, probe=probe)
    # SSH discovery still worked → host is shown, engine just unreachable.
    assert out["ssh_ok"] is True and out["engines"][0]["reachable"] is False
    assert out["active_patches"] == 2 and out["gpu_count"] == 2


def test_resolve_key_is_forwarded_to_probe():
    seen = {}

    def probe(host, port, *, api_key=None):
        seen["key"] = api_key
        return {"reachable": True, "version": "v", "models": []}

    fleet.summarize_host(_Profile("s", "S", "h"), discover=_discover_ok, probe=probe,
                         resolve_key=lambda p: "genesis-local")
    assert seen["key"] == "genesis-local"  # the host's engine key reaches the probe


def test_fleet_overview_endpoint(monkeypatch, tmp_path):
    import pytest
    pytest.importorskip("fastapi")
    monkeypatch.setenv("SNDR_HOME", str(tmp_path))
    from fastapi.testclient import TestClient

    from sndr.product_api.legacy import engine_client, host_profiles, ssh_client
    from sndr.product_api.legacy.http_app import create_app

    host_profiles.upsert_host_profile({"label": "Srv A", "host": "10.0.0.1", "ssh_user": "sander", "role": "prod"})
    monkeypatch.setattr(ssh_client, "discover_host", _discover_ok)
    monkeypatch.setattr(engine_client, "probe_host", _probe_ok)
    client = TestClient(create_app(allowed_origins=()))

    rows = client.get("/api/v1/fleet/overview").json()["hosts"]
    assert len(rows) == 1
    r = rows[0]
    assert r["models"] == ["gemma-4-31b"] and r["active_patches"] == 2 and r["gpu_count"] == 2

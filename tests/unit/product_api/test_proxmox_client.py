# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the read-only Proxmox VE client — pure shaping + graceful
degradation, without a live Proxmox (mirrors test_k8s_client.py)."""
from __future__ import annotations

from sndr.product_api.legacy import proxmox_client as px


def test_availability_unconfigured(monkeypatch):
    for v in ("SNDR_PROXMOX_HOST", "SNDR_PROXMOX_TOKEN_ID", "SNDR_PROXMOX_TOKEN_SECRET"):
        monkeypatch.delenv(v, raising=False)
    a = px.availability()
    assert a["available"] is False and a["configured"] is False and "not configured" in a["error"]


def test_availability_missing_token(monkeypatch):
    monkeypatch.setenv("SNDR_PROXMOX_HOST", "pve.local")
    monkeypatch.delenv("SNDR_PROXMOX_TOKEN_ID", raising=False)
    monkeypatch.delenv("SNDR_PROXMOX_TOKEN_SECRET", raising=False)
    a = px.availability()
    assert a["available"] is False and "token" in a["error"]


def test_config_normalizes_host(monkeypatch):
    monkeypatch.setenv("SNDR_PROXMOX_HOST", "pve.local")
    monkeypatch.setenv("SNDR_PROXMOX_TOKEN_ID", "root@pam!sndr")
    monkeypatch.setenv("SNDR_PROXMOX_TOKEN_SECRET", "secret")
    c = px._config()
    assert c["host"] == "https://pve.local:8006"  # scheme + default API port added
    assert px.availability()["available"] is True


def test_config_keeps_explicit_scheme_and_port(monkeypatch):
    monkeypatch.setenv("SNDR_PROXMOX_HOST", "http://10.0.0.5:8006")
    assert px._config()["host"] == "http://10.0.0.5:8006"


def test_shape_node():
    n = px.shape_node({"node": "pve1", "status": "online", "cpu": 0.25, "maxcpu": 16,
                       "mem": 8 * 2**30, "maxmem": 64 * 2**30, "disk": 100, "maxdisk": 1000, "uptime": 99999})
    assert n["name"] == "pve1" and n["online"] is True
    assert n["cpu_pct"] == 25.0 and n["cpu_cores"] == 16
    assert n["mem_pct"] == 12.5 and n["disk_pct"] == 10.0


def test_shape_guest_vm_with_sndr_preset():
    g = px.shape_guest({"type": "qemu", "vmid": 101, "name": "vllm-35b", "status": "running",
                        "node": "pve1", "cpu": 0.9, "maxcpu": 8, "mem": 40 * 2**30, "maxmem": 48 * 2**30,
                        "maxdisk": 200 * 2**30, "uptime": 3600, "tags": "gpu;sndr-preset-qwen3.6-35b-balanced"})
    assert g["kind"] == "vm" and g["vmid"] == 101 and g["running"] is True
    assert g["cpu_pct"] == 90.0 and g["mem_pct"] == round(100 * 40 / 48, 1)
    assert g["sndr_preset"] == "qwen3.6-35b-balanced"
    assert "gpu" in g["tags"]


def test_shape_guest_lxc_unlinked():
    g = px.shape_guest({"type": "lxc", "vmid": 200, "status": "stopped", "node": "pve1"})
    assert g["kind"] == "lxc" and g["running"] is False and g["sndr_preset"] is None


def test_preset_from_tags_variants():
    assert px._preset_from_tags("a,b,sndr-preset-x9") == "x9"
    assert px._preset_from_tags("sndr-preset-") is None   # prefix only -> no id
    assert px._preset_from_tags("") is None


def test_live_calls_degrade_without_config(monkeypatch):
    for v in ("SNDR_PROXMOX_HOST", "SNDR_PROXMOX_TOKEN_ID", "SNDR_PROXMOX_TOKEN_SECRET"):
        monkeypatch.delenv(v, raising=False)
    assert px.cluster_status()["available"] is False
    assert px.list_nodes()["nodes"] == [] and px.list_guests()["guests"] == []


def test_cluster_status_counts_over_mocked_resources(monkeypatch):
    # _resources is called by the canonical cluster_status — patch it there.
    import sndr.product_api.legacy.proxmox_client as pxc
    monkeypatch.setenv("SNDR_PROXMOX_HOST", "pve.local")
    monkeypatch.setenv("SNDR_PROXMOX_TOKEN_ID", "root@pam!sndr")
    monkeypatch.setenv("SNDR_PROXMOX_TOKEN_SECRET", "secret")
    monkeypatch.setattr(pxc, "_resources", lambda: [
        {"type": "node", "node": "pve1", "status": "online"},
        {"type": "qemu", "vmid": 101, "status": "running", "tags": "sndr-preset-x"},
        {"type": "qemu", "vmid": 102, "status": "stopped"},
        {"type": "lxc", "vmid": 200, "status": "running"},
        {"type": "qemu", "vmid": 900, "status": "stopped", "template": 1},  # template excluded
        {"type": "storage", "storage": "local"},
    ])
    s = px.cluster_status()
    assert s["available"] is True
    assert s["node_count"] == 1 and s["nodes_online"] == 1
    assert s["vm_count"] == 2 and s["vm_running"] == 1     # template not counted
    assert s["lxc_count"] == 1 and s["lxc_running"] == 1
    assert s["sndr_managed"] == 1

# SPDX-License-Identifier: Apache-2.0
"""S-05 (2026-05-08) — runtime_caveat.probe_caveats()."""
from __future__ import annotations

import sndr.engines.vllm.detection.runtime_caveat as M


def test_probe_no_pve_markers_returns_safe(monkeypatch, tmp_path):
    """Without PVE kernel and without /etc/pve, caveat is False."""
    monkeypatch.setattr("platform.release", lambda: "6.5.0-generic")
    monkeypatch.setattr(M, "_has_pve_etc", lambda: False)
    cav = M.probe_caveats()
    assert cav.proxmox_detected is False
    assert "no Proxmox VE markers" in cav.reason


def test_probe_pve_kernel_string_detected(monkeypatch):
    """`6.17.5-pve-edge` triggers detection via kernel name."""
    monkeypatch.setattr("platform.release", lambda: "6.17.5-pve-edge")
    monkeypatch.setattr(M, "_has_pve_etc", lambda: False)
    cav = M.probe_caveats()
    assert cav.proxmox_detected is True
    assert "kernel" in cav.reason.lower()


def test_probe_etc_pve_directory_detected(monkeypatch):
    """`/etc/pve/` existence alone triggers detection."""
    monkeypatch.setattr("platform.release", lambda: "6.5.0-generic")
    monkeypatch.setattr(M, "_has_pve_etc", lambda: True)
    cav = M.probe_caveats()
    assert cav.proxmox_detected is True
    assert "/etc/pve/" in cav.reason


def test_probe_proxmox_kernel_branding(monkeypatch):
    """`6.17.0-1-proxmox` (alternate branding) also detected."""
    monkeypatch.setattr("platform.release", lambda: "6.17.0-1-proxmox")
    monkeypatch.setattr(M, "_has_pve_etc", lambda: False)
    cav = M.probe_caveats()
    assert cav.proxmox_detected is True


def test_probe_kernel_release_field_populated(monkeypatch):
    """`kernel_release` field always reflects platform.release() output."""
    monkeypatch.setattr("platform.release", lambda: "test-kernel-version")
    monkeypatch.setattr(M, "_has_pve_etc", lambda: False)
    cav = M.probe_caveats()
    assert cav.kernel_release == "test-kernel-version"

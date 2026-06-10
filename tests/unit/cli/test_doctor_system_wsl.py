# SPDX-License-Identifier: Apache-2.0
"""WSL2 probe coverage for `sndr doctor-system` (audit closure
post-MASTER_REMEDIATION_PLAN).

Reads `/proc/version` to detect WSL kernel. Verified via mocked file
read so the test passes on non-WSL hosts (Mac dev / Linux servers)
without false-positives.
"""
from __future__ import annotations

import builtins
import os
from io import StringIO

import pytest

from sndr.cli.legacy.doctor_system import _build_facts


_REAL_OPEN = open
_REAL_ISFILE = os.path.isfile
_REAL_EXISTS = os.path.exists


def _patch_proc_version(monkeypatch, contents: str | None,
                         *, present_devices=()):
    """Pretend `/proc/version` exists with the given contents.

    `contents=None` means the file doesn't exist (non-Linux host).
    `present_devices` is a tuple of `/dev/nvidia*` paths to report as present.

    Captures the real callables before patching so the fakes can fall
    through for unrelated paths without recursing.
    """

    def fake_isfile(path):
        if path == "/proc/version":
            return contents is not None
        return _REAL_ISFILE(path)

    def fake_open(path, *args, **kwargs):
        if path == "/proc/version" and contents is not None:
            return StringIO(contents)
        return _REAL_OPEN(path, *args, **kwargs)

    def fake_exists(path):
        if path.startswith("/dev/nvidia") or path == "/dev/dxg":
            return path in present_devices
        return _REAL_EXISTS(path)

    monkeypatch.setattr(os.path, "isfile", fake_isfile)
    monkeypatch.setattr(os.path, "exists", fake_exists)
    monkeypatch.setattr(builtins, "open", fake_open)


class TestWslProbe:
    def test_non_wsl_kernel_undetected(self, monkeypatch):
        _patch_proc_version(
            monkeypatch,
            "Linux version 6.5.0-generic (gcc 13.2.0)",
        )
        facts = _build_facts()
        wsl = facts.get("wsl")
        assert wsl is not None
        assert wsl["detected"] is False

    def test_wsl2_kernel_detected(self, monkeypatch):
        _patch_proc_version(
            monkeypatch,
            "Linux version 5.15.0-microsoft-standard-WSL2 #1",
        )
        facts = _build_facts()
        wsl = facts["wsl"]
        assert wsl["detected"] is True
        assert wsl["wsl2"] is True

    def test_wsl1_kernel_detected_as_not_wsl2(self, monkeypatch):
        _patch_proc_version(
            monkeypatch,
            "Linux version 4.4.0-19041-Microsoft (oe-user@oe-host)",
        )
        facts = _build_facts()
        wsl = facts["wsl"]
        assert wsl["detected"] is True
        assert wsl["wsl2"] is False

    def test_no_proc_version_file_returns_safe_default(self, monkeypatch):
        """Mac / BSD have no /proc/version — should not crash."""
        _patch_proc_version(monkeypatch, None)
        facts = _build_facts()
        wsl = facts["wsl"]
        assert wsl == {"detected": False}

    def test_wsl2_with_nvidia_devices_reports_them(self, monkeypatch):
        _patch_proc_version(
            monkeypatch,
            "WSL2 microsoft kernel",
            present_devices=("/dev/nvidia0", "/dev/nvidiactl", "/dev/dxg"),
        )
        facts = _build_facts()
        wsl = facts["wsl"]
        assert wsl["detected"] is True
        assert set(wsl["nvidia_devices"]) == {
            "/dev/nvidia0", "/dev/nvidiactl", "/dev/dxg",
        }

    def test_wsl2_with_no_devices_reports_empty(self, monkeypatch):
        _patch_proc_version(monkeypatch, "WSL2 microsoft kernel")
        facts = _build_facts()
        wsl = facts["wsl"]
        assert wsl["detected"] is True
        assert wsl["nvidia_devices"] == []

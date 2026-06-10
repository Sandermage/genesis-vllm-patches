# SPDX-License-Identifier: Apache-2.0
"""Tier 2 P3 — deps.checkers unit tests.

The checkers run real subprocess probes when available; we don't stub
those because the goal of `inspect_host()` is to surface accurate
inventory. Tests focus on:
  - dataclass shape (always returns the right fields)
  - graceful handling when the binary is absent
  - serialization round-trip via to_dict()
"""
from __future__ import annotations

import shutil

import pytest

from sndr.deps.checkers import (
    DockerInfo, NvidiaInfo, PythonInfo, VLLMInfo, OSInfo, HostInventory,
    check_os, check_python, check_docker, check_nvidia, check_vllm,
    inspect_host, _run,
)


# ─── _run helper

def test_run_handles_missing_binary():
    rc, out, err = _run(["this-binary-does-not-exist-xyz"])
    assert rc == -1
    assert out == ""
    assert err  # some message


def test_run_returns_stdout_for_real_command():
    rc, out, err = _run(["/usr/bin/env", "true"])
    assert rc == 0
    assert err == ""


# ─── check_os

def test_check_os_returns_dataclass():
    info = check_os()
    assert isinstance(info, OSInfo)
    assert info.system in ("Linux", "Darwin", "Windows", "Java")
    assert info.arch  # non-empty
    d = info.to_dict()
    assert d["system"] == info.system


# ─── check_python

def test_check_python_reports_running_interpreter():
    import sys
    info = check_python()
    assert isinstance(info, PythonInfo)
    assert info.binary_path == sys.executable
    assert info.version.startswith("3.")
    # pip should be available in a working dev env
    assert info.pip_present is True
    d = info.to_dict()
    assert d["version"] == info.version


# ─── check_docker

def test_check_docker_returns_dataclass():
    info = check_docker()
    assert isinstance(info, DockerInfo)
    if shutil.which("docker"):
        assert info.installed is True
        assert info.binary_path is not None
        assert info.via == "cli"
    else:
        # No CLI: installed only if the docker socket happens to be mounted here.
        from sndr.deps import checkers as _c
        if _c._docker_socket_present():
            assert info.installed is True and info.via == "socket"
        else:
            assert info.installed is False
            assert "docker" in info.notes.lower()


def test_check_docker_falls_back_to_socket_when_cli_absent(monkeypatch):
    """No docker CLI but a mounted socket → report installed via the socket,
    not a misleading "not installed" (the node-daemon sidecar case)."""
    from sndr.deps import checkers as _c
    monkeypatch.setattr(_c.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_c, "_docker_socket_present", lambda *a, **k: True)
    monkeypatch.setattr(_c, "_docker_socket_version", lambda *a, **k: "27.2.0")
    info = check_docker()
    assert info.installed is True
    assert info.via == "socket"
    assert info.daemon_running is True
    assert info.server_version == "27.2.0"


def test_check_docker_no_cli_no_socket_reports_absent(monkeypatch):
    from sndr.deps import checkers as _c
    monkeypatch.setattr(_c.shutil, "which", lambda _name: None)
    monkeypatch.setattr(_c, "_docker_socket_present", lambda *a, **k: False)
    info = check_docker()
    assert info.installed is False
    assert "docker" in info.notes.lower()


# ─── check_nvidia

def test_check_nvidia_returns_dataclass():
    info = check_nvidia()
    assert isinstance(info, NvidiaInfo)
    if not shutil.which("nvidia-smi"):
        assert info.installed is False
        assert info.n_gpus == 0


# ─── check_vllm

def test_check_vllm_returns_dataclass():
    """vllm may or may not be in this env; either way returns dataclass."""
    info = check_vllm()
    assert isinstance(info, VLLMInfo)
    if info.installed:
        assert info.version is not None
        assert info.location is not None


# ─── inspect_host

def test_inspect_host_returns_full_inventory():
    inv = inspect_host()
    assert isinstance(inv, HostInventory)
    assert isinstance(inv.os, OSInfo)
    assert isinstance(inv.python, PythonInfo)
    assert isinstance(inv.docker, DockerInfo)
    assert isinstance(inv.nvidia, NvidiaInfo)
    assert isinstance(inv.vllm, VLLMInfo)


def test_inspect_host_to_dict_round_trip():
    """to_dict() output must be JSON-serializable."""
    import json
    d = inspect_host().to_dict()
    s = json.dumps(d, sort_keys=True)
    parsed = json.loads(s)
    assert "os" in parsed
    assert "python" in parsed
    assert "docker" in parsed
    assert "nvidia" in parsed
    assert "vllm" in parsed

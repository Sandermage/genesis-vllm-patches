# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_network_port_consistency.py` — Entry 34."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_network_port_consistency.py"


def _import():
    name = "_audit_v2_network_port_consistency_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(p: Path, text: str) -> Path:
    p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return p


def _hw_yaml(*, hp=8000, cp=8000, shm="'8g'", net="'mynet'") -> str:
    return textwrap.dedent(f"""
        id: synth
        kind: hardware
        runtime:
          docker:
            host_port: {hp}
            container_port: {cp}
            shm_size: {shm}
            network: {net}
    """).lstrip("\n")


class TestRegex:
    def test_shm_size_canonical(self):
        mod = _import()
        for v in ["8g", "8G", "2048m", "1024", "512k"]:
            assert mod.SHM_SIZE_RE.match(v), v

    def test_shm_size_rejects_garbage(self):
        mod = _import()
        for v in ["8gb", "8 g", "huge", ""]:
            assert not mod.SHM_SIZE_RE.match(v), v


class TestCheckOneHardware:
    def test_canonical_passes(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml())
        r = mod.check_one_hardware(y)
        assert r.passed is True

    def test_privileged_port_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(hp=80))
        r = mod.check_one_hardware(y)
        assert r.passed is False

    def test_port_too_high_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(cp=99999))
        r = mod.check_one_hardware(y)
        assert r.passed is False

    def test_shm_no_unit_passes(self, tmp_path):
        """`'2048'` passes the regex (bytes default)."""
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(shm="'2048'"))
        r = mod.check_one_hardware(y)
        assert r.passed is True

    def test_shm_typo_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(shm="'huge'"))
        r = mod.check_one_hardware(y)
        assert r.passed is False

    def test_empty_network_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(net="''"))
        r = mod.check_one_hardware(y)
        assert r.passed is False


class TestLiveRepo:
    def test_committed_clean(self):
        mod = _import()
        results = mod.audit_v2_network_port_consistency()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.hardware_id}: {r.violations}" for r in failed
        )
        assert len(results) == 3


class TestScriptCLI:
    def test_cli_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_json(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "port_range" in payload

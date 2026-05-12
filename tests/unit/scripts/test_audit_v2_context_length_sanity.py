# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_context_length_sanity.py` — Entry 33."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_context_length_sanity.py"


def _import():
    name = "_audit_v2_context_length_sanity_test"
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


def _hw_yaml(*, mml: int = 65536, mbt: int = 4096) -> str:
    return textwrap.dedent(f"""
        id: synth
        kind: hardware
        sizing:
          max_model_len: {mml}
          max_num_batched_tokens: {mbt}
    """).lstrip("\n")


class TestCheckOneHardware:
    def test_canonical_passes(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml())
        r = mod.check_one_hardware(y)
        assert r.passed is True

    def test_mml_below_min_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(mml=320))   # 320 < 1024
        r = mod.check_one_hardware(y)
        assert r.passed is False

    def test_mml_above_max_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(mml=10_000_000))
        r = mod.check_one_hardware(y)
        assert r.passed is False

    def test_batch_above_max_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(mbt=100_000))
        r = mod.check_one_hardware(y)
        assert r.passed is False

    def test_batch_exceeds_mml_fails(self, tmp_path):
        """batch > ctx is physically impossible (a chunk can't be larger
        than the context window)."""
        mod = _import()
        y = _write(tmp_path / "h.yaml", _hw_yaml(mml=2048, mbt=8192))
        r = mod.check_one_hardware(y)
        assert r.passed is False
        assert any("max_num_batched_tokens" in v and "max_model_len" in v
                   for v in r.violations)

    def test_non_int_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "h.yaml", """
            id: synth
            kind: hardware
            sizing:
              max_model_len: 'huge'
              max_num_batched_tokens: 4096
        """)
        r = mod.check_one_hardware(y)
        assert r.passed is False


class TestLiveRepo:
    def test_all_committed_pass(self):
        mod = _import()
        results = mod.audit_v2_context_length_sanity()
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
        assert "bounds" in payload

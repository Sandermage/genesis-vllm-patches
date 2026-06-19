# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_quantization_coverage.py` — Entry 33."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_quantization_coverage.py"


def _import():
    name = "_audit_v2_quantization_coverage_test"
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


def _model_yaml(*, q="'auto_round'", d="'bfloat16'") -> str:
    return textwrap.dedent(f"""
        id: synth
        kind: model
        quantization: {q}
        dtype: {d}
    """).lstrip("\n")


class TestSchema:
    def test_none_in_allowed_quantization(self):
        mod = _import()
        assert None in mod.ALLOWED_QUANTIZATION

    def test_dtype_includes_common(self):
        mod = _import()
        assert "float16" in mod.ALLOWED_DTYPE
        assert "bfloat16" in mod.ALLOWED_DTYPE


class TestCheckOneModel:
    def test_canonical_passes(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", _model_yaml())
        r = mod.check_one_model(y)
        assert r.passed is True

    def test_quantization_null_passes(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", _model_yaml(q="null"))
        r = mod.check_one_model(y)
        assert r.passed is True

    def test_quantization_typo_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", _model_yaml(q="'autoround-typo'"))
        r = mod.check_one_model(y)
        assert r.passed is False
        assert any("quantization" in v for v in r.violations)

    def test_dtype_typo_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", _model_yaml(d="'float69'"))
        r = mod.check_one_model(y)
        assert r.passed is False
        assert any("dtype" in v for v in r.violations)

    def test_missing_dtype_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", "id: x\nkind: model\nquantization: null\n")
        r = mod.check_one_model(y)
        assert r.passed is False


class TestLiveRepo:
    def test_all_committed_pass(self):
        mod = _import()
        results = mod.audit_v2_quantization_coverage()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.model_id}: {r.violations}" for r in failed
        )
        # Phase 5.4 (2026-05-22): refreshed for current fleet
        # (10 V2 model YAMLs; was 6 in Wave 9/10 era).
        # Reconciled 2026-06-19 to live count: 11 model YAMLs — the 11th
        # is qwen3.6-7b-dense (committed club-3090 #58 Path A DENSE
        # reference; one quantization-check result per model YAML).
        assert len(results) == 11


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
        assert "allowed_quantization" in payload
        assert "allowed_dtype" in payload

# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_versions_pin_format.py` — Entry 32."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_versions_pin_format.py"


def _import():
    name = "_audit_v2_versions_pin_format_test"
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


def _model_yaml(*, vllm="'0.20.2rc1.dev9+g01d4d1ad3'",
                genesis="'v11.0.0'") -> str:
    return textwrap.dedent(f"""
        id: synth
        kind: model
        versions:
          vllm_pin_required: {vllm}
          genesis_pin_min: {genesis}
    """).lstrip("\n")


class TestRegex:
    def test_vllm_pin_matches_canonical(self):
        mod = _import()
        assert mod.VLLM_PIN_RE.match("0.20.2rc1.dev9+g01d4d1ad3")
        assert mod.VLLM_PIN_RE.match("0.20.2rc1.dev209+g5536fc0c0")
        assert mod.VLLM_PIN_RE.match("0.20.2rc1.dev93+g51f22dcfd")

    def test_vllm_pin_no_dev_segment_ok(self):
        mod = _import()
        # dev optional
        assert mod.VLLM_PIN_RE.match("0.20.2rc1+g01d4d1ad3")

    def test_vllm_pin_no_rc_segment_ok(self):
        mod = _import()
        assert mod.VLLM_PIN_RE.match("0.20.2+g01d4d1ad3")

    def test_vllm_pin_missing_sha_fails(self):
        mod = _import()
        assert not mod.VLLM_PIN_RE.match("0.20.2rc1.dev9")

    def test_vllm_pin_wrong_separator_fails(self):
        mod = _import()
        assert not mod.VLLM_PIN_RE.match("0.20.2rc1.dev9-g01d4d1ad3")

    def test_genesis_pin_matches(self):
        mod = _import()
        assert mod.GENESIS_PIN_RE.match("v11.0.0")
        assert mod.GENESIS_PIN_RE.match("v11.0.0+wave8")
        assert mod.GENESIS_PIN_RE.match("v0.1.0-alpha")

    def test_genesis_pin_no_v_prefix_fails(self):
        mod = _import()
        assert not mod.GENESIS_PIN_RE.match("11.0.0")


class TestCheckOneModel:
    def test_canonical_passes(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", _model_yaml())
        r = mod.check_one_model(y)
        assert r.passed is True

    def test_missing_vllm_pin_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            versions:
              genesis_pin_min: 'v11.0.0'
        """)
        r = mod.check_one_model(y)
        assert r.passed is False
        assert any("vllm_pin_required" in v for v in r.violations)

    def test_missing_genesis_pin_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            versions:
              vllm_pin_required: '0.20.2rc1.dev9+g01d4d1ad3'
        """)
        r = mod.check_one_model(y)
        assert r.passed is False

    def test_typo_in_vllm_pin_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml",
                   _model_yaml(vllm="'totally-not-a-pin'"))
        r = mod.check_one_model(y)
        assert r.passed is False

    def test_typo_in_genesis_pin_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", _model_yaml(genesis="'11.0.0'"))
        r = mod.check_one_model(y)
        assert r.passed is False


class TestLiveRepo:
    def test_all_models_clean(self):
        mod = _import()
        results = mod.audit_v2_versions_pin_format()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.model_id}: {r.violations}" for r in failed
        )
        # Phase 5.4 (2026-05-22): refreshed for current fleet
        # (10 V2 model YAMLs; was 6 in Wave 9/10 era).
        # Reconciled 2026-06-19 to live count: 11 model YAMLs — the 11th
        # is qwen3.6-7b-dense (committed club-3090 #58 Path A DENSE
        # reference; one pin-format result per model YAML).
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
        assert "regex" in payload

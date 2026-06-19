# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_capability_coverage.py` — Entry 32."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_capability_coverage.py"


def _import():
    name = "_audit_v2_capability_coverage_test"
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


class TestSchema:
    def test_required_caps_in_schema(self):
        mod = _import()
        assert "attention_arch" in mod.ALLOWED_CAPABILITIES
        assert "spec_decode.method" in mod.ALLOWED_CAPABILITIES

    def test_dense_and_hybrid_allowed(self):
        mod = _import()
        assert "dense" in mod.ALLOWED_CAPABILITIES["attention_arch"]
        assert "hybrid_gdn_moe" in mod.ALLOWED_CAPABILITIES["attention_arch"]


class TestCheckOneModel:
    def test_canonical_passes(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            capabilities:
              attention_arch: hybrid_gdn_moe
              tool_call_parser: qwen3_coder
              reasoning_parser: qwen3
              kv_cache_dtype: turboquant_k8v4
              spec_decode:
                method: mtp
        """)
        r = mod.check_one_model(y)
        assert r.passed is True

    def test_attention_arch_typo_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            capabilities:
              attention_arch: hybrid_gdn_mor
        """)
        r = mod.check_one_model(y)
        assert r.passed is False
        assert any(v["capability"] == "attention_arch"
                   for v in r.violations)

    def test_spec_decode_method_typo_fails(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            capabilities:
              spec_decode:
                method: not-a-real-method
        """)
        r = mod.check_one_model(y)
        assert r.passed is False

    def test_absent_capability_passes(self, tmp_path):
        """Required-fields gate handles presence; this gate handles
        value validity. Absent ≠ violation here."""
        mod = _import()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            capabilities:
              attention_arch: dense
        """)
        r = mod.check_one_model(y)
        assert r.passed is True

    def test_spec_decode_none_allowed(self, tmp_path):
        """7b-dense model has spec_decode: None — must pass."""
        mod = _import()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            capabilities:
              spec_decode: null
        """)
        r = mod.check_one_model(y)
        assert r.passed is True

    def test_kv_cache_dtype_none_allowed(self, tmp_path):
        mod = _import()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            capabilities:
              kv_cache_dtype: null
        """)
        r = mod.check_one_model(y)
        assert r.passed is True


class TestLiveRepo:
    def test_all_models_clean(self):
        mod = _import()
        results = mod.audit_v2_capability_coverage()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.model_id}: {r.violations}" for r in failed
        )
        # Phase 5.4 (2026-05-22): refreshed for current fleet
        # (10 V2 model YAMLs; was 6 in Wave 9/10 era).
        # Reconciled 2026-06-19 to live count: 11 model YAMLs — the 11th
        # is qwen3.6-7b-dense (committed club-3090 #58 Path A DENSE
        # reference; one capability-coverage result per model YAML).
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
        assert "allowed_capabilities" in payload

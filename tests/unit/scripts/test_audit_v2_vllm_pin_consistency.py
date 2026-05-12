# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_vllm_pin_consistency.py` — Entry 29."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_vllm_pin_consistency.py"


def _import():
    name = "_audit_v2_vllm_pin_consistency_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_model_yaml(p: Path, *, pin: str | None = "v0.6.4",
                      ref: str | None = None) -> Path:
    body = f"id: synth\nkind: model\nversions:\n  vllm_pin_required: {pin!r}\n"
    if ref is not None:
        body += f"  reference_metrics_ref: {ref!r}\n"
    p.write_text(body, encoding="utf-8")
    return p


# ─── Baseline extraction helper ───────────────────────────────────────


class TestExtractBaseline:
    def test_top_level_vllm_version(self):
        mod = _import()
        assert mod._extract_baseline_vllm_version(
            {"vllm_version": "v1.0"},
        ) == "v1.0"

    def test_top_level_vllm_pin(self):
        mod = _import()
        assert mod._extract_baseline_vllm_version(
            {"vllm_pin": "v1.0"},
        ) == "v1.0"

    def test_nested_parsed_path(self):
        """The committed baselines use nested form."""
        mod = _import()
        d = {"vllm_version": {"parsed": {"vllm_version": "v1.0"}}}
        assert mod._extract_baseline_vllm_version(d) == "v1.0"

    def test_config_vllm_pin(self):
        mod = _import()
        d = {"config": {"vllm_pin": "v1.0"}}
        assert mod._extract_baseline_vllm_version(d) == "v1.0"

    def test_returns_none_when_absent(self):
        mod = _import()
        assert mod._extract_baseline_vllm_version({}) is None


# ─── Per-model check ──────────────────────────────────────────────────


class TestCheckOneModel:
    def test_no_reference_metrics_ref_is_skipped(self, tmp_path):
        mod = _import()
        y = _write_model_yaml(tmp_path / "m.yaml", pin="v1.0", ref=None)
        r = mod.check_one_model(y)
        assert r.passed is True
        assert r.skipped_reason != ""

    def test_matching_pins_pass(self, tmp_path):
        mod = _import()
        bench = tmp_path / "bench.json"
        bench.write_text(json.dumps({"vllm_version": "v1.0"}),
                         encoding="utf-8")
        y = _write_model_yaml(tmp_path / "m.yaml",
                              pin="v1.0", ref=str(bench))
        r = mod.check_one_model(y)
        assert r.passed is True
        assert r.yaml_pin == "v1.0"
        assert r.baseline_pin == "v1.0"

    def test_mismatching_pins_fail(self, tmp_path):
        mod = _import()
        bench = tmp_path / "bench.json"
        bench.write_text(json.dumps({"vllm_version": "v9.9"}),
                         encoding="utf-8")
        y = _write_model_yaml(tmp_path / "m.yaml",
                              pin="v1.0", ref=str(bench))
        r = mod.check_one_model(y)
        assert r.passed is False
        assert r.yaml_pin == "v1.0"
        assert r.baseline_pin == "v9.9"

    def test_missing_baseline_file_errors(self, tmp_path):
        mod = _import()
        y = _write_model_yaml(tmp_path / "m.yaml",
                              pin="v1.0", ref="/nope/missing.json")
        r = mod.check_one_model(y)
        assert r.passed is False
        assert "baseline file not found" in r.error

    def test_baseline_no_vllm_field_errors(self, tmp_path):
        mod = _import()
        bench = tmp_path / "bench.json"
        bench.write_text(json.dumps({"some_other_field": 1}),
                         encoding="utf-8")
        y = _write_model_yaml(tmp_path / "m.yaml",
                              pin="v1.0", ref=str(bench))
        r = mod.check_one_model(y)
        assert r.passed is False
        assert "no recognized vllm version field" in r.error


# ─── Live repo ────────────────────────────────────────────────────────


class TestLiveRepo:
    def test_committed_models_agree(self):
        mod = _import()
        results = mod.audit_v2_vllm_pin_consistency()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.model_id}: yaml={r.yaml_pin} vs baseline={r.baseline_pin} "
            f"{('error=' + r.error) if r.error else ''}"
            for r in failed
        )
        # At least 2 models have a baseline ref (E22 fix).
        compared = [r for r in results
                    if not r.skipped_reason and not r.error]
        assert len(compared) >= 2


# ─── CLI ──────────────────────────────────────────────────────────────


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
        assert payload["failed"] == 0

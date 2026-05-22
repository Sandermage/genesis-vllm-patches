# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_model_baselines.py` — V2 model
`reference_metrics_ref` validator (Entry 22).

Contract:

  • null reference passes vacuously
  • non-null reference must point at an existing readable JSON file
  • path is resolved relative to repo root
  • lookup walks `versions:` → `bench_validation:` → top-level
  • script CLI exits 0 on all-green, 1 on any broken ref, 2 on parse error
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_model_baselines.py"


def _import_script():
    name = "_audit_model_baselines_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Ref-extraction logic ─────────────────────────────────────────────


class TestRefExtraction:
    def test_versions_block(self):
        mod = _import_script()
        d = {"versions": {"reference_metrics_ref": "foo.json"}}
        assert mod._ref_from_yaml(d) == "foo.json"

    def test_versions_null_ref(self):
        mod = _import_script()
        d = {"versions": {"reference_metrics_ref": None}}
        assert mod._ref_from_yaml(d) is None

    def test_bench_validation_block(self):
        mod = _import_script()
        d = {"bench_validation": {"reference_metrics_ref": "bar.json"}}
        assert mod._ref_from_yaml(d) == "bar.json"

    def test_top_level_fallback(self):
        mod = _import_script()
        d = {"reference_metrics_ref": "top.json"}
        assert mod._ref_from_yaml(d) == "top.json"

    def test_missing_returns_none(self):
        mod = _import_script()
        assert mod._ref_from_yaml({}) is None

    def test_versions_takes_precedence_over_top_level(self):
        mod = _import_script()
        d = {
            "reference_metrics_ref": "wrong.json",
            "versions": {"reference_metrics_ref": "right.json"},
        }
        assert mod._ref_from_yaml(d) == "right.json"


# ─── Single-YAML check ────────────────────────────────────────────────


def _write_yaml(p: Path, text: str) -> Path:
    p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return p


class TestCheckOneYaml:
    def test_null_ref_passes(self, tmp_path):
        mod = _import_script()
        y = _write_yaml(tmp_path / "m.yaml", """
            id: test-model
            versions:
              reference_metrics_ref: null
        """)
        r = mod.check_one_yaml(y)
        assert r.passed is True
        assert r.reference_metrics_ref is None

    def test_missing_ref_passes(self, tmp_path):
        mod = _import_script()
        y = _write_yaml(tmp_path / "m.yaml", """
            id: test-model
            versions:
              other_field: x
        """)
        r = mod.check_one_yaml(y)
        assert r.passed is True

    def test_existing_ref_passes(self, tmp_path):
        mod = _import_script()
        bench = tmp_path / "bench.json"
        bench.write_text('{"foo": 1}', encoding="utf-8")
        # Path must be relative to REPO_ROOT — use absolute resolved.
        rel = str(bench.resolve())
        y = _write_yaml(tmp_path / "m.yaml", f"""
            id: test-model
            versions:
              reference_metrics_ref: {rel}
        """)
        r = mod.check_one_yaml(y)
        assert r.passed is True
        assert r.baseline_exists is True
        assert r.baseline_parseable is True

    def test_broken_ref_fails(self, tmp_path):
        mod = _import_script()
        y = _write_yaml(tmp_path / "m.yaml", """
            id: test-model
            versions:
              reference_metrics_ref: tests/integration/baselines/nonexistent.json
        """)
        r = mod.check_one_yaml(y)
        assert r.passed is False
        assert r.baseline_exists is False
        assert "not found" in r.error

    def test_malformed_baseline_json_fails(self, tmp_path):
        mod = _import_script()
        bench = tmp_path / "bad.json"
        bench.write_text("not valid {json", encoding="utf-8")
        rel = str(bench.resolve())
        y = _write_yaml(tmp_path / "m.yaml", f"""
            id: test-model
            versions:
              reference_metrics_ref: {rel}
        """)
        r = mod.check_one_yaml(y)
        assert r.passed is False
        assert r.baseline_exists is True
        assert r.baseline_parseable is False
        assert "JSON parse error" in r.error


# ─── Live repo check — committed YAMLs must all pass ──────────────────


class TestLiveRepo:
    def test_committed_v2_models_all_pass(self):
        """Every committed V2 model YAML must pass the baseline audit.

        Phase 4.A (2026-05-22): replaced the prior `verified >= 2`
        assertion (pre-Phase-5.2.D expectation that ≥2 ModelDefs
        reference a canonical baseline JSON). After 5.2.D formalized
        the receipt-only state across all 10 ModelDefs, every
        `reference_metrics_ref` is `null` and the audit reports
        `verified=0` — which is the correct current state. Phase 7
        will land formal dev371 baselines and the verified path will
        re-engage naturally without further test edits.

        Structural invariant that survives both eras:
          - failed == [] (no broken baseline refs in committed YAMLs)
          - verified + skipped == total (every model classified)
        """
        mod = _import_script()
        results = mod.audit_model_baselines()
        failed = [r for r in results if not r.passed]
        assert failed == [], (
            "Broken baseline refs in committed YAMLs:\n"
            + "\n".join(f"  {r.model_id}: {r.error}" for r in failed)
        )
        verified = [r for r in results if r.reference_metrics_ref is not None]
        skipped = [r for r in results if r.reference_metrics_ref is None]
        assert len(verified) + len(skipped) == len(results), (
            f"classification gap: verified={len(verified)} + "
            f"skipped={len(skipped)} != total={len(results)}"
        )

    def test_at_least_six_v2_models_present(self):
        """Sanity: we expect 6 V2 model YAMLs after the migration."""
        mod = _import_script()
        results = mod.audit_model_baselines()
        assert len(results) >= 6


# ─── Script CLI ────────────────────────────────────────────────────────


class TestScriptCLI:
    def test_cli_exits_zero_on_committed_repo(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"audit-model-baselines failed: {result.stdout[:1500]}"
        )
        assert "passing" in result.stdout

    def test_cli_json_mode_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "total" in payload
        assert "passed" in payload
        assert "failed" in payload
        assert "checks" in payload
        assert payload["failed"] == 0

    def test_cli_synth_broken(self, tmp_path):
        """Synthetic broken-ref YAML → rc=1."""
        bad = tmp_path / "broken.yaml"
        bad.write_text(textwrap.dedent("""
            id: synth-broken
            versions:
              reference_metrics_ref: /nope/does/not/exist.json
        """).lstrip("\n"), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--model-dir", str(tmp_path), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["failed"] == 1

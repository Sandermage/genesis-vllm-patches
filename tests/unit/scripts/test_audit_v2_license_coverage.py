# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_license_coverage.py` — Entry 28 license +
maintainer coverage."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_license_coverage.py"


def _import_script():
    name = "_audit_v2_license_coverage_test"
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


def _model_yaml(*, lic: str = "'apache-2.0'", maint: str = "'alice'") -> str:
    return textwrap.dedent(f"""
        id: synth
        kind: model
        license: {lic}
        maintainer: {maint}
    """).lstrip("\n")


# ─── Schema sanity ────────────────────────────────────────────────────


class TestSchema:
    def test_allowed_licenses_non_empty(self):
        mod = _import_script()
        assert len(mod.ALLOWED_LICENSES) >= 5
        assert "apache-2.0" in mod.ALLOWED_LICENSES
        assert "mit" in mod.ALLOWED_LICENSES

    def test_normalize_license_lowercases(self):
        mod = _import_script()
        assert mod._normalize_license("Apache-2.0") == "apache-2.0"
        assert mod._normalize_license("  MIT  ") == "mit"
        assert mod._normalize_license(None) == ""
        assert mod._normalize_license(42) == ""


# ─── Per-file check ───────────────────────────────────────────────────


class TestCheckOneModel:
    def test_canonical_passes(self, tmp_path):
        mod = _import_script()
        y = _write(tmp_path / "m.yaml", _model_yaml())
        r = mod.check_one_model(y)
        assert r.passed is True
        assert r.license_ok and r.maintainer_ok

    def test_uppercase_license_accepted(self, tmp_path):
        mod = _import_script()
        y = _write(tmp_path / "m.yaml", _model_yaml(lic="'Apache-2.0'"))
        r = mod.check_one_model(y)
        assert r.passed is True

    def test_unknown_license_fails(self, tmp_path):
        mod = _import_script()
        y = _write(tmp_path / "m.yaml", _model_yaml(lic="'proprietary'"))
        r = mod.check_one_model(y)
        assert r.passed is False
        assert r.license_ok is False
        assert any("not in ALLOWED_LICENSES" in s for s in r.reasons)

    def test_missing_license_fails(self, tmp_path):
        mod = _import_script()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            maintainer: 'alice'
        """)
        r = mod.check_one_model(y)
        assert r.passed is False
        assert r.license_ok is False

    def test_empty_maintainer_fails(self, tmp_path):
        mod = _import_script()
        y = _write(tmp_path / "m.yaml", _model_yaml(maint="''"))
        r = mod.check_one_model(y)
        assert r.passed is False
        assert r.maintainer_ok is False

    def test_missing_maintainer_fails(self, tmp_path):
        mod = _import_script()
        y = _write(tmp_path / "m.yaml", """
            id: synth
            kind: model
            license: 'mit'
        """)
        r = mod.check_one_model(y)
        assert r.passed is False
        assert r.maintainer_ok is False

    def test_parse_error_recorded(self, tmp_path):
        mod = _import_script()
        bad = tmp_path / "bad.yaml"
        bad.write_text("foo: [\n", encoding="utf-8")
        r = mod.check_one_model(bad)
        assert r.passed is False
        assert r.parse_error != ""


# ─── Live repo ────────────────────────────────────────────────────────


class TestLiveRepo:
    def test_all_models_have_valid_license_and_maintainer(self):
        mod = _import_script()
        results = mod.audit_v2_license_coverage()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.model_id}: license={r.license_raw} maint={r.maintainer_raw} "
            f"reasons={r.reasons}"
            for r in failed
        )
        assert len(results) >= 6


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
        assert "allowed_licenses" in payload
        assert payload["failed"] == 0

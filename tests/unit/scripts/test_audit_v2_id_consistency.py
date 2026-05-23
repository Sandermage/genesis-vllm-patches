# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_id_consistency.py` — Entry 28 id-filename
consistency."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_id_consistency.py"


def _import_script():
    name = "_audit_v2_id_consistency_test"
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


class TestCheckFile:
    def test_match_passes(self, tmp_path):
        mod = _import_script()
        y = _write(tmp_path / "qwen3.6-27b-dflash.yaml",
                   "id: qwen3.6-27b-dflash\nkind: model\n")
        r = mod._check_file(y, "model")
        assert r.passed is True
        assert r.yaml_id == r.filename_stem

    def test_mismatch_fails(self, tmp_path):
        mod = _import_script()
        y = _write(tmp_path / "qwen3.6-27b.yaml",
                   "id: typo-id\nkind: model\n")
        r = mod._check_file(y, "model")
        assert r.passed is False
        assert r.yaml_id == "typo-id"
        assert r.filename_stem == "qwen3.6-27b"

    def test_missing_id_fails(self, tmp_path):
        mod = _import_script()
        y = _write(tmp_path / "no-id.yaml", "kind: model\n")
        r = mod._check_file(y, "model")
        # yaml_id == "" != "no-id" → fail
        assert r.passed is False

    def test_parse_error_recorded(self, tmp_path):
        mod = _import_script()
        bad = tmp_path / "bad.yaml"
        bad.write_text("foo: [\n", encoding="utf-8")
        r = mod._check_file(bad, "model")
        assert r.passed is False
        assert r.parse_error != ""

    def test_non_mapping_recorded(self, tmp_path):
        mod = _import_script()
        bad = tmp_path / "list.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        r = mod._check_file(bad, "model")
        assert r.passed is False
        assert "not a mapping" in r.parse_error


class TestLiveRepo:
    def test_all_committed_match(self):
        mod = _import_script()
        results = mod.audit_v2_id_consistency()
        failed = [r for r in results if not r.passed]
        assert failed == [], (
            "id/filename mismatch:\n"
            + "\n".join(
                f"  {r.layer} {r.filename_stem}: id={r.yaml_id!r}"
                for r in failed
            )
        )
        # Phase 5.4 (2026-05-22): refreshed for current fleet
        # (10 model + 3 hardware + 17 profile = 30; Wave 10 baseline
        # was 24 with 6 models and 15 profiles).
        # Phase 7.G4.26B-A4B.B0 (2026-05-23): +3 Gemma 4 26B-A4B
        # profiles (no new models / hardware) → 33.
        assert len(results) == 33


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

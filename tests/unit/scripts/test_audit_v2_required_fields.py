# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_required_fields.py` — V2 schema fields
gate (Entry 27).

Contract:

  • REQUIRED_FIELDS schema is frozen as code.
  • Every committed V2 YAML satisfies its layer's required fields.
  • Missing any required field surfaces in `missing_fields`.
  • CLI exits 0 on committed repo; 1 on synthetic missing field.
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
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_required_fields.py"


def _import_script():
    name = "_audit_v2_required_fields_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Schema sanity ────────────────────────────────────────────────────


class TestSchema:
    def test_four_layers_defined(self):
        mod = _import_script()
        assert set(mod.REQUIRED_FIELDS) == {
            "model", "hardware", "profile", "preset",
        }

    def test_model_layer_size(self):
        mod = _import_script()
        # 16 required model-layer fields per E27 freeze.
        assert len(mod.REQUIRED_FIELDS["model"]) == 16
        assert "last_validated" in mod.REQUIRED_FIELDS["model"]
        assert "patches" in mod.REQUIRED_FIELDS["model"]

    def test_hardware_layer_size(self):
        mod = _import_script()
        assert len(mod.REQUIRED_FIELDS["hardware"]) == 9
        assert "runtime" in mod.REQUIRED_FIELDS["hardware"]
        assert "system_env" in mod.REQUIRED_FIELDS["hardware"]

    def test_profile_layer_size(self):
        mod = _import_script()
        assert len(mod.REQUIRED_FIELDS["profile"]) == 9
        assert "parent_model" in mod.REQUIRED_FIELDS["profile"]
        assert "created" in mod.REQUIRED_FIELDS["profile"]

    def test_preset_layer_size(self):
        mod = _import_script()
        # Just three pointer fields.
        assert mod.REQUIRED_FIELDS["preset"] == frozenset(
            {"model", "hardware", "profile"},
        )


# ─── Per-file check ───────────────────────────────────────────────────


def _write_yaml(p: Path, text: str) -> Path:
    p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return p


class TestCheckFile:
    def test_complete_model_passes(self, tmp_path):
        mod = _import_script()
        body = "\n".join(f"{k}: x" for k in mod.REQUIRED_FIELDS["model"])
        y = _write_yaml(tmp_path / "m.yaml", body)
        r = mod._check_file(y, "model")
        assert r.passed is True
        assert r.missing_fields == []

    def test_missing_one_field_fails(self, tmp_path):
        mod = _import_script()
        # Drop `patches` from the model schema.
        body = "\n".join(
            f"{k}: x"
            for k in mod.REQUIRED_FIELDS["model"]
            if k != "patches"
        )
        y = _write_yaml(tmp_path / "m.yaml", body)
        r = mod._check_file(y, "model")
        assert r.passed is False
        assert r.missing_fields == ["patches"]

    def test_missing_multiple_fields_listed(self, tmp_path):
        mod = _import_script()
        body = "id: x\nkind: hardware\n"
        y = _write_yaml(tmp_path / "h.yaml", body)
        r = mod._check_file(y, "hardware")
        assert r.passed is False
        # Several missing — exact count = required - 2 present (id, kind).
        assert len(r.missing_fields) == len(mod.REQUIRED_FIELDS["hardware"]) - 2

    def test_non_mapping_top_level_parse_error(self, tmp_path):
        mod = _import_script()
        bad = tmp_path / "list.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        r = mod._check_file(bad, "model")
        assert r.passed is False
        assert "not a mapping" in r.parse_error

    def test_preset_minimal_passes(self, tmp_path):
        mod = _import_script()
        y = _write_yaml(tmp_path / "alias.yaml", """
            model: x
            hardware: y
            profile: z
        """)
        r = mod._check_file(y, "preset")
        assert r.passed is True


# ─── Live repo — regression anchor ────────────────────────────────────


class TestLiveRepo:
    def test_all_layers_pass(self):
        mod = _import_script()
        results = mod.audit_v2_required_fields()
        failed = [r for r in results if not r.passed]
        assert failed == [], (
            "V2 YAMLs missing required fields:\n"
            + "\n".join(
                f"  {r.layer} {r.yaml_id}: missing={r.missing_fields}"
                + (f" error={r.error}" if r.error else "")
                for r in failed
            )
        )
        # Phase 5.4 (2026-05-22): refreshed for current fleet
        # (10 model + 3 hardware + 17 profile + 15 preset = 45;
        # Wave 10 baseline was 39 with 6 models and 15 profiles).
        # Phase 7.G4.B1.0 (2026-05-23): +2 Gemma 4 31B presets
        # (10 model + 3 hardware + 17 profile + 17 preset = 47).
        # Phase 7.G4.26B-A4B.B0 (2026-05-23): +3 Gemma 4 26B-A4B
        # profiles + 3 preset aliases
        # (10 model + 3 hardware + 20 profile + 20 preset = 53).
        # Phase 7.G4.26B-A4B.B4-PRE (2026-05-23): +1 K=1 multiconc
        # profile + 1 preset alias (10 + 3 + 21 + 21 = 55).
        assert len(results) == 55


# ─── CLI ──────────────────────────────────────────────────────────────


class TestScriptCLI:
    def test_cli_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout[:2000]

    def test_cli_json(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["failed"] == 0
        assert "required_fields" in payload
        assert "model" in payload["required_fields"]

    def test_cli_layer_filter(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--layer", "hardware", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        layers = {r["layer"] for r in payload["results"]}
        assert layers == {"hardware"}

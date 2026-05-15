# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_cross_reference.py` — Entry 29."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_cross_reference.py"


def _import():
    name = "_audit_v2_cross_reference_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestLiveRepo:
    def test_all_refs_resolve(self):
        mod = _import()
        results = mod.audit_v2_cross_reference()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.layer} {r.label}.{r.field_name}={r.ref_value!r} "
            f"→ no such {r.target_layer}"
            for r in failed
        )
        # Wave 10 V2 layout: 15 profile parent_model + 15 preset×3 = 60 refs.
        assert len(results) == 60


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
        assert payload["total"] == 60


class TestSyntheticBroken:
    def test_profile_with_missing_parent_model_fails(self, tmp_path, monkeypatch):
        """Override PROFILE_DIR + MODEL_DIR to a synthetic tmp tree."""
        mod = _import()
        # Synthetic profile pointing at non-existent model.
        prof_dir = tmp_path / "profile"
        prof_dir.mkdir()
        (prof_dir / "synth.yaml").write_text(
            "id: synth\nkind: profile\nparent_model: nope-not-exist\n",
            encoding="utf-8",
        )
        # No model files → ref must fail.
        monkeypatch.setattr(mod, "PROFILE_DIR", prof_dir)
        monkeypatch.setattr(mod, "MODEL_DIR", tmp_path / "model_empty")
        monkeypatch.setattr(mod, "HARDWARE_DIR", tmp_path / "hw_empty")
        monkeypatch.setattr(mod, "PRESETS_DIR", tmp_path / "presets_empty")
        results = mod.audit_v2_cross_reference()
        # Single profile, single ref check.
        assert len(results) == 1
        assert not results[0].passed
        assert results[0].ref_value == "nope-not-exist"

    def test_preset_with_missing_hardware_fails(self, tmp_path, monkeypatch):
        mod = _import()
        # Synthetic preset whose hardware id doesn't exist.
        preset_dir = tmp_path / "presets"
        preset_dir.mkdir()
        (preset_dir / "synth.yaml").write_text(
            "model: nope\nhardware: nope\nprofile: nope\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "PROFILE_DIR", tmp_path / "profile_empty")
        monkeypatch.setattr(mod, "MODEL_DIR", tmp_path / "model_empty")
        monkeypatch.setattr(mod, "HARDWARE_DIR", tmp_path / "hw_empty")
        monkeypatch.setattr(mod, "PRESETS_DIR", preset_dir)
        results = mod.audit_v2_cross_reference()
        # 3 refs from one preset; all fail.
        assert len(results) == 3
        assert all(not r.passed for r in results)

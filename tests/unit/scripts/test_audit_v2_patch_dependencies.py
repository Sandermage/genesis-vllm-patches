# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_patch_dependencies.py` — Entry 31."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_patch_dependencies.py"


def _import():
    name = "_audit_v2_patch_dependencies_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestLiveRepo:
    def test_all_models_clean(self):
        mod = _import()
        results = mod.audit_v2_patch_dependencies()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.model_id}: req_viol={r.missing_requires} "
            f"conf_viol={r.conflicts_active}"
            for r in failed
        )
        # Phase 5.4 (2026-05-22): refreshed for current fleet
        # (10 V2 model YAMLs; was 6 in Wave 9/10 era).
        # Reconciled 2026-06-19 to live count: 11 model YAMLs — the 11th
        # is qwen3.6-7b-dense (committed club-3090 #58 Path A DENSE
        # reference; one dependency-check result per model YAML).
        assert len(results) == 11

    def test_enabled_count_sane(self):
        mod = _import()
        results = mod.audit_v2_patch_dependencies()
        # At least one model has ≥ 20 enabled.
        assert max(len(r.enabled_pids) for r in results) >= 20


class TestSyntheticDrift:
    def test_synthetic_requires_violation(self, tmp_path):
        """Build a fake registry where P-Z requires P-Y; enable only
        P-Z and verify violation surfaces."""
        mod = _import()
        # Use the live flag index but synthesize a model that enables
        # P67b (requires P67) without enabling P67. Note: because P67+P67b
        # share env_flag in real registry, this test has to bypass —
        # easier to verify the algorithm via direct check_one_model with
        # a fake flag_to_pids + pid_meta.
        flag_to_pids = {
            "GENESIS_ENABLE_FAKE_FLAG_Z": ["P-Z"],
            "GENESIS_ENABLE_FAKE_FLAG_Y": ["P-Y"],
        }
        pid_meta = {
            "P-Z": {"requires_patches": ["P-Y"]},
            "P-Y": {},
        }
        fake_yaml = tmp_path / "fake.yaml"
        fake_yaml.write_text(textwrap.dedent("""
            id: fake
            kind: model
            patches:
              GENESIS_ENABLE_FAKE_FLAG_Z: '1'
        """).lstrip("\n"), encoding="utf-8")
        r = mod.check_one_model(
            fake_yaml,
            flag_to_pids=flag_to_pids,
            pid_meta=pid_meta,
        )
        assert r.passed is False
        assert len(r.missing_requires) == 1
        assert r.missing_requires[0]["patch_id"] == "P-Z"
        assert r.missing_requires[0]["missing_dependency"] == "P-Y"

    def test_synthetic_conflict_violation(self, tmp_path):
        mod = _import()
        flag_to_pids = {
            "GENESIS_ENABLE_FAKE_FLAG_A": ["P-A"],
            "GENESIS_ENABLE_FAKE_FLAG_B": ["P-B"],
        }
        pid_meta = {
            "P-A": {"conflicts_with": ["P-B"]},
            "P-B": {"conflicts_with": ["P-A"]},
        }
        fake_yaml = tmp_path / "fake.yaml"
        fake_yaml.write_text(textwrap.dedent("""
            id: fake
            kind: model
            patches:
              GENESIS_ENABLE_FAKE_FLAG_A: '1'
              GENESIS_ENABLE_FAKE_FLAG_B: '1'
        """).lstrip("\n"), encoding="utf-8")
        r = mod.check_one_model(
            fake_yaml,
            flag_to_pids=flag_to_pids,
            pid_meta=pid_meta,
        )
        assert r.passed is False
        # Single emission (pid < conf de-duplication).
        assert len(r.conflicts_active) == 1
        pair = r.conflicts_active[0]
        assert {pair["patch_a"], pair["patch_b"]} == {"P-A", "P-B"}

    def test_synthetic_multi_pid_flag_satisfies_requires(self, tmp_path):
        """P67/P67b share env_flag — setting flag enables both;
        P67b.requires=['P67'] must be satisfied."""
        mod = _import()
        flag_to_pids = {
            "SHARED_FLAG": ["P67", "P67b"],
        }
        pid_meta = {
            "P67":  {},
            "P67b": {"requires_patches": ["P67"]},
        }
        fake_yaml = tmp_path / "fake.yaml"
        fake_yaml.write_text(textwrap.dedent("""
            id: fake
            kind: model
            patches:
              SHARED_FLAG: '1'
        """).lstrip("\n"), encoding="utf-8")
        r = mod.check_one_model(
            fake_yaml,
            flag_to_pids=flag_to_pids,
            pid_meta=pid_meta,
        )
        assert r.passed is True
        assert "P67"  in r.enabled_pids
        assert "P67b" in r.enabled_pids


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

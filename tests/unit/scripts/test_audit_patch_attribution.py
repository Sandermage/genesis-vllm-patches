# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_patch_attribution.py` — Phase A semantic drift gate.

Checks covered:

  AT-1  patches_attribution key references a known PATCH_REGISTRY ID
        (catches typos like `PN204b` vs `PN204`).
  AT-2  Roles asserting presence (load_bearing/defensive/optional_perf)
        require the env_flag to be present in model.patches.
  AT-3  Coverage statistic is informational (not gating without
        --min-coverage).
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_patch_attribution.py"


def _import_script():
    name = "_audit_patch_attribution_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_yaml(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)


# ─── AT-1: unknown patch ID ────────────────────────────────────────────


class TestAT1UnknownPatchId:
    def test_clean_attribution_passes(self, tmp_path):
        mod = _import_script()
        yaml_path = tmp_path / "model.yaml"
        _write_yaml(yaml_path, (
            "id: test\n"
            "patches:\n"
            "  GENESIS_ENABLE_PN204: '1'\n"
            "patches_attribution:\n"
            "  PN204:\n"
            "    role: load_bearing\n"
            "    note: 'pinned'\n"
        ))
        registry = {"PN204": "GENESIS_ENABLE_PN204"}
        chk = mod._check_one(yaml_path, registry)
        assert chk.passed
        assert chk.unknown_patch_ids == []

    def test_typo_flagged(self, tmp_path):
        mod = _import_script()
        yaml_path = tmp_path / "model.yaml"
        _write_yaml(yaml_path, (
            "id: test\n"
            "patches:\n"
            "  GENESIS_ENABLE_PN204: '1'\n"
            "patches_attribution:\n"
            "  PN204b:\n"  # typo
            "    role: load_bearing\n"
        ))
        registry = {"PN204": "GENESIS_ENABLE_PN204"}
        chk = mod._check_one(yaml_path, registry)
        assert not chk.passed
        assert "PN204b" in chk.unknown_patch_ids


# ─── AT-2: presence-asserting role without env_flag ────────────────────


class TestAT2RoleConsistency:
    def test_load_bearing_with_matching_flag_passes(self, tmp_path):
        mod = _import_script()
        yaml_path = tmp_path / "model.yaml"
        _write_yaml(yaml_path, (
            "id: test\n"
            "patches:\n"
            "  GENESIS_ENABLE_PN204: '1'\n"
            "patches_attribution:\n"
            "  PN204:\n"
            "    role: load_bearing\n"
        ))
        registry = {"PN204": "GENESIS_ENABLE_PN204"}
        chk = mod._check_one(yaml_path, registry)
        assert chk.attribution_without_flag == []

    def test_load_bearing_without_flag_flagged(self, tmp_path):
        mod = _import_script()
        yaml_path = tmp_path / "model.yaml"
        _write_yaml(yaml_path, (
            "id: test\n"
            "patches: {}\n"  # PN204 NOT in model.patches
            "patches_attribution:\n"
            "  PN204:\n"
            "    role: load_bearing\n"
        ))
        registry = {"PN204": "GENESIS_ENABLE_PN204"}
        chk = mod._check_one(yaml_path, registry)
        assert "PN204" in chk.attribution_without_flag

    def test_suspected_regression_exempt(self, tmp_path):
        """suspected_regression / no_op / unknown roles intentionally
        document patches kept OUT of the model — exempt from AT-2."""
        mod = _import_script()
        yaml_path = tmp_path / "model.yaml"
        _write_yaml(yaml_path, (
            "id: test\n"
            "patches: {}\n"
            "patches_attribution:\n"
            "  PN26B:\n"
            "    role: suspected_regression\n"
            "    note: 'kept out on 27B per bench'\n"
        ))
        registry = {"PN26B": "GENESIS_ENABLE_PN26B"}
        chk = mod._check_one(yaml_path, registry)
        assert chk.attribution_without_flag == []

    def test_defensive_role_checked(self, tmp_path):
        mod = _import_script()
        yaml_path = tmp_path / "model.yaml"
        _write_yaml(yaml_path, (
            "id: test\n"
            "patches: {}\n"
            "patches_attribution:\n"
            "  PN204:\n"
            "    role: defensive\n"
        ))
        registry = {"PN204": "GENESIS_ENABLE_PN204"}
        chk = mod._check_one(yaml_path, registry)
        assert "PN204" in chk.attribution_without_flag


# ─── Coverage % calculation ────────────────────────────────────────────


class TestCoverage:
    def test_full_coverage_100pct(self, tmp_path):
        mod = _import_script()
        yaml_path = tmp_path / "model.yaml"
        _write_yaml(yaml_path, (
            "id: test\n"
            "patches:\n"
            "  GENESIS_ENABLE_PN1: '1'\n"
            "  GENESIS_ENABLE_PN2: '1'\n"
            "patches_attribution:\n"
            "  PN1:\n"
            "    role: load_bearing\n"
            "  PN2:\n"
            "    role: defensive\n"
        ))
        registry = {"PN1": "GENESIS_ENABLE_PN1", "PN2": "GENESIS_ENABLE_PN2"}
        chk = mod._check_one(yaml_path, registry)
        assert chk.total_patches == 2
        assert chk.total_attributions == 2
        assert chk.coverage_pct == 100.0

    def test_zero_patches_returns_100pct(self, tmp_path):
        """Empty patches dict → coverage is vacuously 100%."""
        mod = _import_script()
        yaml_path = tmp_path / "model.yaml"
        _write_yaml(yaml_path, "id: test\npatches: {}\npatches_attribution: {}\n")
        registry: dict[str, str] = {}
        chk = mod._check_one(yaml_path, registry)
        assert chk.coverage_pct == 100.0


# ─── Parse error path ──────────────────────────────────────────────────


class TestParseError:
    def test_yaml_parse_error_surfaces(self, tmp_path):
        mod = _import_script()
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text(":\n:\n: invalid")
        registry: dict[str, str] = {}
        chk = mod._check_one(yaml_path, registry)
        assert not chk.passed
        assert chk.parse_error != ""


# ─── Live regression anchor ────────────────────────────────────────────


class TestLiveRepo:
    def test_main_exits_zero_on_live_repo(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0

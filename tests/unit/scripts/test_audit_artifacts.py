# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_artifacts.py` — Phase 7 artifact storage policy.

Checks covered:

  A-1 evidence ledger present in maintainer planning tree (when visible)
  A-2 evidence/patch_proof/ contents (JSON + _waivers/ only, .gitkeep OK)
  A-3 release artefacts (SBOM + constraints) — only in --public-release mode
  A-5 bench-results JSON not tracked in git
  A-6 rollback playbook present (legacy OR consolidated path with R-001 anchor)
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_artifacts.py"


def _import_script():
    name = "_audit_artifacts_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── A-1 evidence ledger ──────────────────────────────────────────────


class TestA1EvidenceLedger:
    def test_no_planning_dir_returns_empty(self, monkeypatch, tmp_path):
        """When no planning dir is visible the check is a no-op."""
        mod = _import_script()
        monkeypatch.delenv("GENESIS_PLANNING_DIR", raising=False)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod.check_evidence_ledger_present() == []

    def test_planning_dir_with_ledger_passes(self, monkeypatch, tmp_path):
        mod = _import_script()
        planning = tmp_path / "planning"
        planning.mkdir()
        (planning / "ROADMAP_EVIDENCE_LEDGER_2026-05-30.md").write_text("# ledger")
        monkeypatch.setenv("GENESIS_PLANNING_DIR", str(planning))
        assert mod.check_evidence_ledger_present() == []

    def test_planning_dir_without_ledger_flags(self, monkeypatch, tmp_path):
        mod = _import_script()
        planning = tmp_path / "planning"
        planning.mkdir()
        # No ROADMAP_EVIDENCE_LEDGER_*.md present.
        monkeypatch.setenv("GENESIS_PLANNING_DIR", str(planning))
        issues = mod.check_evidence_ledger_present()
        assert len(issues) == 1
        assert "ROADMAP_EVIDENCE_LEDGER" in issues[0]


# ─── A-2 patch proof layout ────────────────────────────────────────────


class TestA2PatchProofLayout:
    def test_missing_dir_returns_empty(self, monkeypatch, tmp_path):
        mod = _import_script()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        # evidence/patch_proof/ does not exist → return [] (optional)
        assert mod.check_patch_proof_layout() == []

    def test_json_only_passes(self, monkeypatch, tmp_path):
        mod = _import_script()
        pp = tmp_path / "evidence" / "patch_proof"
        pp.mkdir(parents=True)
        (pp / "P1_static.json").write_text("{}")
        (pp / ".gitkeep").write_text("")
        (pp / "_waivers").mkdir()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod.check_patch_proof_layout() == []

    def test_non_json_file_flagged(self, monkeypatch, tmp_path):
        mod = _import_script()
        pp = tmp_path / "evidence" / "patch_proof"
        pp.mkdir(parents=True)
        (pp / "rogue.txt").write_text("nope")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        issues = mod.check_patch_proof_layout()
        assert len(issues) == 1
        assert "rogue.txt" in issues[0]

    def test_unexpected_dir_flagged(self, monkeypatch, tmp_path):
        mod = _import_script()
        pp = tmp_path / "evidence" / "patch_proof"
        pp.mkdir(parents=True)
        (pp / "unexpected_subdir").mkdir()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        issues = mod.check_patch_proof_layout()
        assert len(issues) == 1
        assert "unexpected_subdir" in issues[0]


# ─── A-3 release artefacts ─────────────────────────────────────────────


class TestA3Release:
    def test_skipped_without_public_release(self):
        mod = _import_script()
        assert mod.check_release_artefacts_present(public_release=False) == []

    def test_missing_artefacts_flagged_in_release_mode(self, monkeypatch, tmp_path):
        mod = _import_script()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        # No release/ tree → both SBOM + constraints missing
        issues = mod.check_release_artefacts_present(public_release=True)
        assert len(issues) == 2

    def test_present_artefacts_pass(self, monkeypatch, tmp_path):
        mod = _import_script()
        release = tmp_path / "release"
        release.mkdir()
        (release / "SBOM.spdx.json").write_text("{}")
        (release / "constraints.txt").write_text("")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod.check_release_artefacts_present(public_release=True) == []


# ─── A-5 no bench results tracked ──────────────────────────────────────


class TestA5BenchResults:
    def test_clean_files_pass(self):
        mod = _import_script()
        files = ["docs/README.md", "scripts/foo.py", "tools/bar.py"]
        assert mod.check_no_bench_results_tracked(files) == []

    def test_bench_result_json_flagged(self):
        mod = _import_script()
        files = ["bench-results/run-001.json", "ok-file.py"]
        issues = mod.check_no_bench_results_tracked(files)
        assert len(issues) == 1
        assert "run-001.json" in issues[0]


# ─── A-6 rollback playbook ─────────────────────────────────────────────


class TestA6Rollback:
    def test_legacy_path_with_anchor_passes(self, monkeypatch, tmp_path):
        mod = _import_script()
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "ROLLBACK_PLAYBOOK.md").write_text("R-001 step.")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod.check_rollback_playbook_present() == []

    def test_consolidated_path_with_anchor_passes(self, monkeypatch, tmp_path):
        mod = _import_script()
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "TROUBLESHOOTING.md").write_text(
            "# Rollback playbook\n## R-001\nstep"
        )
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod.check_rollback_playbook_present() == []

    def test_missing_anchor_flags(self, monkeypatch, tmp_path):
        mod = _import_script()
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "TROUBLESHOOTING.md").write_text(
            "# Random content, no anchor"
        )
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        issues = mod.check_rollback_playbook_present()
        assert len(issues) == 1


# ─── Live regression anchor ────────────────────────────────────────────


class TestLive:
    def test_a5_no_bench_results_in_live_repo(self):
        """Live repo must not have bench-results JSON tracked."""
        mod = _import_script()
        files = mod._git_tracked_files()
        bad = mod.check_no_bench_results_tracked(files)
        assert bad == [], f"unexpected bench-result tracked: {bad}"

    def test_a6_live_repo_has_rollback_playbook(self):
        mod = _import_script()
        assert mod.check_rollback_playbook_present() == []

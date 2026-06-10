# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_wheel_contents.py`` — §9.A.1
(AUDIT-CLOSURE.3, 2026-05-27).

The audit is a thin CLI wrapper; tests cover:

  * Pyproject shape check (sndr runtime tree in packages, sndr console entry)
  * Test-file presence check (catches accidental deletion of the
    canonical wheel-boundary test files)
  * Invariant manifest documentation
  * Live corpus exit 0
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_wheel_contents.py"


def _import_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_wheel_contents", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_wheel_contents"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def audit_mod():
    return _import_audit_module()


class TestPyprojectShape:
    """Direct exercise of ``check_pyproject_shape`` against tmp pyproject."""

    def _write_pyproject(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "pyproject.toml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_clean_pyproject_passes(self, audit_mod, tmp_path):
        self._write_pyproject(tmp_path,
            "[project]\n"
            "name = \"x\"\n"
            "[project.scripts]\n"
            "sndr = \"sndr.cli.legacy:cli_main\"\n"
            "[tool.setuptools.packages.find]\n"
            "include = [\"sndr*\"]\n"
        )
        results = audit_mod.check_pyproject_shape(repo_root=tmp_path)
        assert all(r.passed for r in results), (
            f"clean pyproject should pass, got: {results}"
        )

    def test_missing_sndr_tree_in_packages_flagged(self, audit_mod, tmp_path):
        self._write_pyproject(tmp_path,
            "[project]\n"
            "name = \"x\"\n"
            "[project.scripts]\n"
            "sndr = \"x.cli:main\"\n"
            "[tool.setuptools.packages.find]\n"
            "include = [\"some_other_package\"]\n"
        )
        results = audit_mod.check_pyproject_shape(repo_root=tmp_path)
        failed = [r for r in results if not r.passed]
        assert any("sndr runtime tree MISSING" in r.detail for r in failed), (
            f"missing sndr runtime tree must be flagged, got: {results}"
        )

    def test_missing_sndr_console_entry_flagged(self, audit_mod, tmp_path):
        self._write_pyproject(tmp_path,
            "[project]\n"
            "name = \"x\"\n"
            "[project.scripts]\n"
            "other = \"x.cli:main\"\n"
            "[tool.setuptools.packages.find]\n"
            "include = [\"sndr*\"]\n"
        )
        results = audit_mod.check_pyproject_shape(repo_root=tmp_path)
        failed = [r for r in results if not r.passed]
        assert any("sndr console entry MISSING" in r.detail for r in failed)

    def test_missing_pyproject_flagged(self, audit_mod, tmp_path):
        results = audit_mod.check_pyproject_shape(repo_root=tmp_path)
        assert results
        assert not results[0].passed
        assert "not found" in results[0].detail


class TestTestFilePresence:
    """``check_test_files_exist`` flags missing boundary test files."""

    def test_live_tree_all_present(self, audit_mod):
        results = audit_mod.check_test_files_exist()
        assert results
        assert all(r.passed for r in results), (
            f"all wheel-boundary test files must exist, got missing: "
            f"{[r for r in results if not r.passed]}"
        )

    def test_missing_test_file_flagged(self, audit_mod, tmp_path):
        # Don't create any test files in tmp_path.
        results = audit_mod.check_test_files_exist(repo_root=tmp_path)
        assert all(not r.passed for r in results)
        for r in results:
            assert "MISSING" in r.detail


class TestInvariantManifest:
    """The invariant list is the public API of this audit."""

    def test_invariants_nonempty(self, audit_mod):
        assert len(audit_mod._INVARIANTS) >= 7, (
            f"expected ≥7 wheel-boundary invariants documented, "
            f"got {len(audit_mod._INVARIANTS)}"
        )

    def test_invariants_have_test_locations(self, audit_mod):
        for inv in audit_mod._INVARIANTS:
            assert "::" in inv.location, (
                f"invariant {inv.name!r} should reference a "
                f"test_file::class[::method], got: {inv.location}"
            )


class TestLiveCorpus:
    """Audit must exit 0 on real tree."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *args],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        )

    def test_live_default_exit_zero(self):
        result = self._run()
        assert result.returncode == 0, (
            f"live wheel-boundary should be intact, got rc="
            f"{result.returncode}\nstdout:\n{result.stdout}"
        )
        assert "wheel boundary surface intact" in result.stdout

    def test_live_json_shape(self):
        result = self._run("--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["passed"] is True
        assert len(data["pyproject_results"]) >= 2
        assert len(data["test_file_results"]) >= 2
        assert len(data["invariants"]) >= 7

    def test_show_tests_listing(self):
        result = self._run("--show-tests")
        assert result.returncode == 0
        assert "invariants covered" in result.stdout

    def test_help_works(self):
        result = self._run("--help")
        assert result.returncode == 0
        assert "audit_wheel_contents" in result.stdout

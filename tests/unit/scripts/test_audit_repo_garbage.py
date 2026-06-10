# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_repo_garbage.py`` — §9.A.2
(AUDIT-CLOSURE.2, 2026-05-27).

Coverage:

  * Each forbidden-filename category fires on a canonical example
  * Path-aware ``temp-output-at-root`` rule only triggers at depth-0
  * Allowlist (``.claude/``, ``CLAUDE.md``, ``sndr_private/``) silent
  * Live tracked-tree must stay clean
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_repo_garbage.py"


def _import_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_repo_garbage", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_repo_garbage"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def audit_mod():
    return _import_audit_module()


class TestClassifyFilename:
    """Direct exercise of ``_classify_filename``."""

    def test_orig_flagged(self, audit_mod):
        assert audit_mod._classify_filename("foo.py.orig") == "merge-leftover"

    def test_rej_flagged(self, audit_mod):
        assert audit_mod._classify_filename("bar.txt.rej") == "merge-leftover"

    def test_ds_store_flagged(self, audit_mod):
        assert audit_mod._classify_filename(".DS_Store") == "macos-metadata"
        # Even in a subdir, .DS_Store is metadata garbage.
        assert audit_mod._classify_filename("docs/.DS_Store") == "macos-metadata"

    def test_editor_backup_flagged(self, audit_mod):
        assert audit_mod._classify_filename("foo.py~") == "editor-backup"

    def test_vim_swap_flagged(self, audit_mod):
        assert audit_mod._classify_filename("foo.py.swp") == "vim-swap"

    def test_emacs_autosave_flagged(self, audit_mod):
        assert audit_mod._classify_filename("#foo.py#") == "emacs-autosave"

    def test_stray_pyc_flagged(self, audit_mod):
        assert audit_mod._classify_filename("misplaced.pyc") == "stray-pyc"

    def test_illegal_chars_quote_flagged(self, audit_mod):
        cat = audit_mod._classify_filename('weird"name.txt')
        assert cat == "illegal-filename-chars"

    def test_illegal_chars_question_flagged(self, audit_mod):
        assert audit_mod._classify_filename("what?.md") == "illegal-filename-chars"

    def test_temp_output_at_root_flagged(self, audit_mod):
        # Repo-root (no slash in path).
        assert audit_mod._classify_filename("output.txt") == "temp-output-at-root"
        assert audit_mod._classify_filename("tmp.json") == "temp-output-at-root"
        assert audit_mod._classify_filename("debug.log") == "temp-output-at-root"
        assert audit_mod._classify_filename("scratch.md") == "temp-output-at-root"

    def test_temp_output_in_subdir_not_flagged(self, audit_mod):
        # ``test.yml`` under .github/workflows is a legitimate CI file.
        assert audit_mod._classify_filename(".github/workflows/test.yml") is None
        # ``debug.md`` under docs/ is a real doc.
        assert audit_mod._classify_filename("docs/debug.md") is None
        # ``output.json`` under tests/ is a fixture.
        assert audit_mod._classify_filename("tests/fixtures/output.json") is None

    def test_clean_filename_returns_none(self, audit_mod):
        assert audit_mod._classify_filename("scripts/audit_repo_garbage.py") is None
        assert audit_mod._classify_filename("README.md") is None
        assert audit_mod._classify_filename("docs/USAGE.md") is None


class TestTopLevelAllowlist:
    """Allowlist hides operator-local untracked entries."""

    def test_dot_claude_allowed(self, audit_mod):
        assert audit_mod._in_top_level_allowlist(".claude/agents/foo.md") is True

    def test_claude_md_allowed(self, audit_mod):
        assert audit_mod._in_top_level_allowlist("CLAUDE.md") is True

    def test_sndr_private_allowed(self, audit_mod):
        assert audit_mod._in_top_level_allowlist(
            "sndr_private/planning/foo.md"
        ) is True

    def test_legitimate_path_not_allowed(self, audit_mod):
        assert audit_mod._in_top_level_allowlist("docs/USAGE.md") is False


class TestCriticalZone:
    """Untracked-zone scan focuses on critical surfaces."""

    def test_docs_in_zone(self, audit_mod):
        assert audit_mod._in_critical_zone("docs/foo.md") is True

    def test_scripts_in_zone(self, audit_mod):
        assert audit_mod._in_critical_zone("scripts/foo.py") is True

    def test_vllm_sndr_core_in_zone(self, audit_mod):
        assert audit_mod._in_critical_zone(
            "sndr/engines/vllm/patches/foo.py"
        ) is True

    def test_tests_in_zone(self, audit_mod):
        assert audit_mod._in_critical_zone("tests/unit/foo.py") is True

    def test_repo_root_not_in_zone(self, audit_mod):
        """Repo-root level is not a 'critical zone' for untracked-scan
        purposes (tracked-scan covers it)."""
        assert audit_mod._in_critical_zone("README.md") is False


class TestLiveCorpus:
    """Live tracked-tree must stay clean post-audit."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *args],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        )

    def test_live_default_exit_zero(self):
        result = self._run()
        assert result.returncode == 0, (
            f"live tracked tree should be clean of garbage, "
            f"got rc={result.returncode}\nstdout:\n{result.stdout}"
        )
        assert "No garbage detected" in result.stdout

    def test_live_json_shape(self):
        result = self._run("--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["count"] == 0
        assert data["findings"] == []

    def test_tracked_only_mode_runs(self):
        result = self._run("--tracked-only")
        assert result.returncode == 0

    def test_help_works(self):
        result = self._run("--help")
        assert result.returncode == 0
        assert "audit_repo_garbage" in result.stdout

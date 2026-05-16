# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_no_hardcoded_paths.py` — operator path drift
detector (Entry 26).

Contract:

  • Comments (lines starting with `#`) are ignored — operator may
    document host paths in comments.
  • `${var}` placeholders are OK.
  • `/home/<USER>/` or `/Users/<USER>/` with a plausible username flags.
  • Generic non-user dirs (`/home/models/`, `/Users/Public/`) don't flag.
  • Files under `_archive/` are skipped.
  • EXEMPT_FILES list bypasses the check.
  • Live committed repo passes (regression anchor).
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
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_no_hardcoded_paths.py"


def _import_script():
    name = "_audit_no_hardcoded_paths_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Predicates + regex ────────────────────────────────────────────────


class TestDetection:
    def test_comment_line_skipped(self):
        mod = _import_script()
        assert mod._line_is_comment("# - /home/sander/models")
        assert mod._line_is_comment("    #  comment")
        assert not mod._line_is_comment("  - /home/sander/models")

    def test_regex_matches_home_user(self):
        mod = _import_script()
        line = "  - /home/sander/models:/models:ro"
        ms = list(mod._PATH_RE.finditer(line))
        assert len(ms) == 1
        assert ms[0].group("user") == "sander"

    def test_regex_matches_users_macos(self):
        mod = _import_script()
        line = "  - /Users/alice/repo:/code:ro"
        ms = list(mod._PATH_RE.finditer(line))
        assert len(ms) == 1
        assert ms[0].group("user") == "alice"

    def test_generic_user_dirs_skipped(self):
        """`/home/models/` is a generic container path, not operator."""
        mod = _import_script()
        # `/home/models/` matches the regex but `models` is in _GENERIC_USERS,
        # so the violation filter drops it.
        # The regex still produces a hit; the file-scanner is what filters.
        ms = list(mod._PATH_RE.finditer("- /home/models/x:/foo:ro"))
        assert len(ms) == 1
        assert ms[0].group("user") == "models"
        assert "models" in mod._GENERIC_USERS


# ─── Single-file scanner ──────────────────────────────────────────────


class TestScanOneFile:
    def test_clean_file_no_violations(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "clean.yaml"
        f.write_text(textwrap.dedent("""
            # this comment mentions /home/sander/ in description
            mounts:
              - "${models_dir}:/models:ro"
              - "${hf_cache}:/root/.cache/huggingface:ro"
        """).lstrip("\n"), encoding="utf-8")
        r = mod._scan_one_file(f)
        assert r.passed is True
        assert r.violations == []

    def test_hardcoded_path_flagged(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "dirty.yaml"
        f.write_text(textwrap.dedent("""
            mounts:
              - "/home/sander/models:/models:ro"
        """).lstrip("\n"), encoding="utf-8")
        r = mod._scan_one_file(f)
        assert r.passed is False
        assert len(r.violations) == 1
        assert "/home/sander/" in r.violations[0].matched

    def test_two_hardcoded_paths_both_flagged(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "dirty2.yaml"
        f.write_text(textwrap.dedent("""
            mounts:
              - "/home/sander/models:/models:ro"
              - "/Users/sander/projects/x:/code:ro"
        """).lstrip("\n"), encoding="utf-8")
        r = mod._scan_one_file(f)
        assert r.passed is False
        assert len(r.violations) == 2

    def test_comment_with_path_not_flagged(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "doc.yaml"
        f.write_text(textwrap.dedent("""
            # mount /home/sander/data into /models in production
            mounts:
              - "${models_dir}:/models:ro"
        """).lstrip("\n"), encoding="utf-8")
        r = mod._scan_one_file(f)
        assert r.passed is True
        assert r.violations == []

    def test_generic_user_not_flagged(self, tmp_path):
        mod = _import_script()
        f = tmp_path / "generic.yaml"
        f.write_text(textwrap.dedent("""
            mounts:
              - "/home/models/qwen:/models:ro"
              - "/Users/Public/share:/share:ro"
        """).lstrip("\n"), encoding="utf-8")
        r = mod._scan_one_file(f)
        assert r.passed is True


# ─── EXEMPT_FILES allowlist ───────────────────────────────────────────


class TestExempt:
    def test_exempt_entries_resolve_to_real_files(self):
        """Whatever lives in the live exempt list must be a real file
        in the tree. Empty list is OK — the privacy-consolidation pass
        moved the only previously-exempt example off the public surface,
        so the gate currently runs with zero exemptions."""
        mod = _import_script()
        for rel in mod.EXEMPT_FILES:
            assert (REPO_ROOT / rel).is_file(), f"exempt file missing: {rel}"

    def test_exempt_file_has_justification_comment(self):
        """Each exempt file must self-document why."""
        mod = _import_script()
        for rel in mod.EXEMPT_FILES:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8")
            assert "audit-no-hardcoded-paths EXEMPT" in text, (
                f"{rel} lacks `audit-no-hardcoded-paths EXEMPT` header marker"
            )


# ─── Whole-repo sweep — regression anchor ────────────────────────────


class TestLiveRepo:
    def test_no_violations_in_active_config(self):
        """After E26 fixes, sweep must be clean."""
        mod = _import_script()
        results = mod.audit_no_hardcoded_paths()
        violating = [r for r in results if not r.passed]
        assert violating == [], (
            "hardcoded paths in active config:\n"
            + "\n".join(
                f"  {r.path}: {[v.matched for v in r.violations[:3]]}"
                for r in violating
            )
        )


# ─── CLI ──────────────────────────────────────────────────────────────


class TestScriptCLI:
    def test_cli_zero_on_committed(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout[:2000]

    def test_cli_json_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "total_files" in payload
        assert "exempt" in payload
        assert payload["violating"] == 0

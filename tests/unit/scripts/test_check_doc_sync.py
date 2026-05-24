# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/check_doc_sync.py` —
CONFIG-HYGIENE.docs-reconcile.1.GATE-EXTEND additions.

Covers:
  - DOC_PATTERNS coverage of the 5 newly-tracked files
    (BENCHMARKS / FAQ / CONFIGURATION / QUICKSTART / RELEASE_POLICY)
  - `_TRANSITION_ALLOWLIST` behavior:
      allowlisted mismatch → `transition_pending=True`, --strict exits 0
      unallowlisted mismatch → --strict exits 1
  - Live committed corpus passes --strict (everything either matches
    expected OR is in the transition allowlist scheduled for
    CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL)
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_doc_sync.py"


def _import():
    name = "_check_doc_sync_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── DOC_PATTERNS coverage ────────────────────────────────────────────


class TestExtendedCoverage:
    """GATE-EXTEND added 5 files. Confirm each is in DOC_PATTERNS with
    at least one regex."""

    @pytest.mark.parametrize("filename", [
        "BENCHMARKS.md",
        "FAQ.md",
        "CONFIGURATION.md",
        "QUICKSTART.md",
        "RELEASE_POLICY.md",
    ])
    def test_file_in_doc_patterns(self, filename):
        mod = _import()
        target = REPO_ROOT / "docs" / filename
        assert target in mod.DOC_PATTERNS, (
            f"docs/{filename} should be in DOC_PATTERNS after GATE-EXTEND"
        )
        assert mod.DOC_PATTERNS[target], (
            f"docs/{filename} has no patterns — empty list is invalid"
        )


# ─── Transition allowlist ────────────────────────────────────────────


class TestTransitionAllowlist:
    def test_allowlist_is_frozenset_of_tuples(self):
        mod = _import()
        assert isinstance(mod._TRANSITION_ALLOWLIST, frozenset)
        for entry in mod._TRANSITION_ALLOWLIST:
            assert isinstance(entry, tuple) and len(entry) == 3
            rel, line, found = entry
            assert isinstance(rel, str)
            assert isinstance(line, int) and line > 0
            assert isinstance(found, int) and found > 0

    def test_allowlist_is_empty_after_mechanical(self):
        """CONFIG-HYGIENE.docs-reconcile.1.MECHANICAL emptied the
        transition allowlist. The scaffolding remains (frozenset of
        3-tuples) so the next counter-bump cycle can populate without
        redesigning the gate."""
        mod = _import()
        assert mod._TRANSITION_ALLOWLIST == frozenset(), (
            "Transition allowlist must be empty post-MECHANICAL; any "
            "entry implies pending mechanical work outside scope."
        )


# ─── check_doc with synthetic doc + allowlist injection ──────────────


class TestCheckDocBehavior:
    def test_unallowlisted_mismatch_flagged(self, tmp_path, monkeypatch):
        mod = _import()
        synthetic = tmp_path / "FAKE.md"
        synthetic.write_text("Total entries: **42**\n")
        # Use one of the new BENCHMARKS regexes against synthetic file
        # by monkey-patching DOC_PATTERNS to point at the temp path.
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        mismatches = mod.check_doc(
            synthetic, expected_count=227,
            patterns=[(r"Total entries: \*\*(\d+)\*\*", 1)],
        )
        assert len(mismatches) == 1
        mm = mismatches[0]
        assert mm["found"] == 42
        assert mm["expected"] == 227
        assert mm["transition_pending"] is False

    def test_allowlisted_mismatch_marked_pending(self, tmp_path, monkeypatch):
        mod = _import()
        synthetic = tmp_path / "FAKE.md"
        synthetic.write_text("foo\nbar baz 99 qux\n")
        # Inject the synthetic file into the allowlist via monkeypatch.
        # `check_doc` reads `_TRANSITION_ALLOWLIST` at call time.
        rel = "FAKE.md"
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(
            mod, "_TRANSITION_ALLOWLIST",
            frozenset({(rel, 2, 99)}),
        )
        mismatches = mod.check_doc(
            synthetic, expected_count=100,
            patterns=[(r"baz (\d+) qux", 1)],
        )
        assert len(mismatches) == 1
        assert mismatches[0]["transition_pending"] is True
        assert mismatches[0]["line"] == 2


# ─── Live committed corpus / CLI smoke ───────────────────────────────


class TestLiveCorpus:
    def test_default_mode_clean(self):
        """Post-MECHANICAL: every doc anchor matches the registry count.
        Default mode prints the clean summary and exits 0."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"default mode failed:\n{result.stdout}\n{result.stderr}"
        )
        out = result.stdout
        assert "PATCH_REGISTRY count: 227" in out
        assert "claim 227 patches consistently" in out

    def test_strict_mode_passes_on_clean_corpus(self):
        """--strict exits 0 because every detected anchor matches and
        the transition allowlist is empty (MECHANICAL closed)."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--strict"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, (
            f"--strict mode failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_json_mode_clean_status(self):
        import json as _json
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0
        data = _json.loads(result.stdout)
        assert data["expected_registry_count"] == 227
        assert "transition_pending" in data
        assert "errors" in data
        assert "status" in data
        # Post-MECHANICAL: no drift at all.
        assert data["status"] == "OK"
        assert data["errors"] == []
        assert data["transition_pending"] == []
        assert data["mismatches"] == []

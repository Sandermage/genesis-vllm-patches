# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_anchor_fragility.py`` — Phase 3.1
TextPatcher anchor fragility ratchet (2026-06-01)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_anchor_fragility.py"


def _import_script():
    name = "_audit_anchor_fragility_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── _collect_string AST walker ────────────────────────────────────────────


class TestCollectString:
    def test_plain_str_literal(self):
        import ast
        mod = _import_script()
        node = ast.parse('x = "hello world"').body[0].value
        assert mod._collect_string(node) == "hello world"

    def test_parenthesised_concat(self):
        import ast
        mod = _import_script()
        # Source: x = ("a\n" "b\n" "c\n")
        node = ast.parse('x = ("a\\n" "b\\n" "c\\n")').body[0].value
        assert mod._collect_string(node) == "a\nb\nc\n"

    def test_binop_concat(self):
        import ast
        mod = _import_script()
        node = ast.parse('x = "ab" + "cd"').body[0].value
        assert mod._collect_string(node) == "abcd"

    def test_tuple_concat(self):
        import ast
        mod = _import_script()
        # Tuple form: x = ("a\n", "b\n")
        node = ast.parse('x = ("a\\n", "b\\n")').body[0].value
        assert mod._collect_string(node) == "a\nb\n"

    def test_non_static_returns_none(self):
        import ast
        mod = _import_script()
        # f-string is not a static literal — returns None
        node = ast.parse('x = f"hello {who}"').body[0].value
        assert mod._collect_string(node) is None


# ─── Name-matcher heuristic ─────────────────────────────────────────────────


class TestIsAnchorName:
    def test_anchor_prefix(self):
        mod = _import_script()
        assert mod._is_anchor_name("ANCHOR_1A_IMPORT_OLD") is True
        assert mod._is_anchor_name("MY_ANCHOR_FOO") is True

    def test_suffix_matches(self):
        mod = _import_script()
        assert mod._is_anchor_name("FORWARD_OLD") is True
        assert mod._is_anchor_name("P85_SITE2_OLD") is True
        assert mod._is_anchor_name("ANCHOR_FORWARD_BEFORE") is True
        assert mod._is_anchor_name("FOO_PRE") is True

    def test_non_anchor_rejected(self):
        mod = _import_script()
        assert mod._is_anchor_name("DEFAULT_PAGE_SIZE") is False
        assert mod._is_anchor_name("foo_old") is False  # case-sensitive


# ─── Live corpus — informational, no hard error ─────────────────────────────


class TestLiveCorpus:
    def test_default_state_exit_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        # Default hard_cap=70 is set above the largest current anchor;
        # the gate is informational on the inherited fragility baseline.
        assert result.returncode == 0, result.stdout + result.stderr

    def test_strict_mode_surfaces_warnings(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--strict"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        # In strict mode any anchor >= threshold is an error — the
        # inherited 13-file baseline has documented warnings.
        assert result.returncode == 1

    def test_json_payload_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        for key in ("files", "counts", "threshold", "hard_cap", "passed"):
            assert key in payload
        assert "warn" in payload["counts"]
        assert "error" in payload["counts"]


# ─── Synthetic fragility regression ─────────────────────────────────────────


def _write_fake_patch(path: Path, anchor_lines: int) -> None:
    """Write a tiny .py file with one ANCHOR_FAKE_OLD of the given size."""
    body = "fake line\n" * anchor_lines
    path.write_text(
        f'ANCHOR_FAKE_OLD = """{body}"""\n',
        encoding="utf-8",
    )


class TestSyntheticRegression:
    def test_anchor_above_hard_cap_fires_error(self, tmp_path, monkeypatch):
        mod = _import_script()
        # Point the audit at a synthesized tree.
        fake = tmp_path / "fragile.py"
        _write_fake_patch(fake, anchor_lines=100)
        monkeypatch.setattr(mod, "SCAN_ROOT", tmp_path)
        report = mod.audit(threshold=25, hard_cap=70)
        assert report["counts"]["error"] >= 1
        assert report["passed"] is False

    def test_anchor_below_warn_threshold_invisible(self, tmp_path, monkeypatch):
        mod = _import_script()
        fake = tmp_path / "clean.py"
        _write_fake_patch(fake, anchor_lines=10)
        monkeypatch.setattr(mod, "SCAN_ROOT", tmp_path)
        report = mod.audit(threshold=25, hard_cap=70)
        # A 10-line anchor stays under the 25 threshold — no warn or
        # error. The file itself still appears in `files` so operators
        # can see the inventory.
        assert report["counts"]["error"] == 0
        assert report["counts"]["warn"] == 0
        assert report["passed"] is True

    def test_anchor_at_warn_threshold_fires_warn_only(self, tmp_path, monkeypatch):
        mod = _import_script()
        fake = tmp_path / "warn.py"
        _write_fake_patch(fake, anchor_lines=30)
        monkeypatch.setattr(mod, "SCAN_ROOT", tmp_path)
        report = mod.audit(threshold=25, hard_cap=70)
        assert report["counts"]["warn"] == 1
        assert report["counts"]["error"] == 0
        # passed=True because warn-only is non-blocking by default.
        assert report["passed"] is True

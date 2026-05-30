# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_private_namespace.py` — hard rule #27.

Contract enforced:

  Rule 1 (ERROR): no `vllm/**/sndr_private` directory anywhere.
  Rule 2 (WARN):  tests/unit/test_wheel_contents.py exists with the
                  test_no_sndr_private_anywhere_in_wheel function.
  Rule 3 (WARN):  repo-root `sndr_private/` is listed in .gitignore.

  Default mode: ERRORs fail; WARNs pass with banner.
  --strict mode: WARNs are upgraded to ERROR.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_private_namespace.py"


def _import_script():
    name = "_audit_private_namespace_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_tree(root: Path) -> None:
    """Build a minimal tree that satisfies all 3 rules (clean baseline)."""
    (root / "vllm").mkdir()
    (root / "vllm" / "sndr_core").mkdir()
    (root / "vllm" / "sndr_core" / "__init__.py").write_text("")
    (root / "tests" / "unit").mkdir(parents=True)
    test_file = root / "tests" / "unit" / "test_wheel_contents.py"
    test_file.write_text(
        "def test_no_sndr_private_anywhere_in_wheel():\n    pass\n"
    )
    (root / ".gitignore").write_text("sndr_private/\n")


# ─── Live repo (regression anchor) ─────────────────────────────────────


class TestLiveRepo:
    def test_live_passes_default_mode(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_live_passes_strict_mode(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--strict"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0


# ─── Rule 1 — no vllm/**/sndr_private ──────────────────────────────────


class TestRule1:
    def test_clean_tree_returns_empty(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        violations = mod._check_no_sndr_private_under_vllm()
        assert violations == []

    def test_detects_sndr_private_in_vllm(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        bad = tmp_path / "vllm" / "sndr_core" / "sndr_private"
        bad.mkdir()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        violations = mod._check_no_sndr_private_under_vllm()
        assert len(violations) == 1
        assert "sndr_private" in violations[0]

    def test_detects_deeply_nested_violation(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        bad = tmp_path / "vllm" / "a" / "b" / "c" / "sndr_private"
        bad.mkdir(parents=True)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        violations = mod._check_no_sndr_private_under_vllm()
        assert len(violations) == 1


# ─── Rule 2 — wheel-contract test guard ────────────────────────────────


class TestRule2:
    def test_clean_when_test_present(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod._check_wheel_contract_test_exists() is None

    def test_warns_when_test_file_missing(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        (tmp_path / "tests" / "unit" / "test_wheel_contents.py").unlink()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        warning = mod._check_wheel_contract_test_exists()
        assert warning is not None
        assert "missing" in warning

    def test_warns_when_test_function_missing(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        # File exists but lacks the canonical function name.
        (tmp_path / "tests" / "unit" / "test_wheel_contents.py").write_text(
            "def test_something_else():\n    pass\n"
        )
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        warning = mod._check_wheel_contract_test_exists()
        assert warning is not None
        assert "test_no_sndr_private_anywhere_in_wheel" in warning


# ─── Rule 3 — top-level sndr_private gitignored ────────────────────────


class TestRule3:
    def test_clean_when_in_gitignore(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod._check_top_level_sndr_private_gitignored() is None

    def test_warns_when_gitignore_missing(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        (tmp_path / ".gitignore").unlink()
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        warning = mod._check_top_level_sndr_private_gitignored()
        assert warning is not None

    def test_warns_when_not_in_gitignore(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        (tmp_path / ".gitignore").write_text("# nothing about private\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        warning = mod._check_top_level_sndr_private_gitignored()
        assert warning is not None

    def test_accepts_simple_form(self, tmp_path, monkeypatch):
        mod = _import_script()
        _make_tree(tmp_path)
        (tmp_path / ".gitignore").write_text("sndr_private\n")
        monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
        assert mod._check_top_level_sndr_private_gitignored() is None

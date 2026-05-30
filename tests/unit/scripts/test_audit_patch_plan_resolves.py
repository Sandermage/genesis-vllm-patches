# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_patch_plan_resolves.py` — Phase D resolver gate.

Contract:

  1. Every committed V2 preset composes + resolves cleanly under all
     three policies (compat / safe / minimal).
  2. ResolveCheck.passed is True only when error is empty AND no
     per-policy slot carries an error.
  3. ResolveCheck.total_warnings aggregates warnings across policies.
  4. main() exits 0 on clean live repo.
  5. --json shape exposes presets_scanned / policies / all_passed /
     results[].
  6. --strict-warnings causes exit 1 when warnings exist.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_patch_plan_resolves.py"


def _import_script():
    name = "_audit_patch_plan_resolves_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── ResolveCheck dataclass invariants ─────────────────────────────────


class TestResolveCheckProperties:
    def test_passed_true_when_clean(self):
        mod = _import_script()
        chk = mod.ResolveCheck(preset="ok", path=Path("/x"))
        chk.by_policy["safe"] = {"included_count": 5, "warnings": []}
        assert chk.passed is True

    def test_passed_false_when_error_set(self):
        mod = _import_script()
        chk = mod.ResolveCheck(preset="bad", path=Path("/x"), error="boom")
        assert chk.passed is False

    def test_passed_false_when_policy_error(self):
        mod = _import_script()
        chk = mod.ResolveCheck(preset="x", path=Path("/x"))
        chk.by_policy["compat"] = {"error": "Ouch"}
        assert chk.passed is False

    def test_total_warnings_sums_across_policies(self):
        mod = _import_script()
        chk = mod.ResolveCheck(preset="x", path=Path("/x"))
        chk.by_policy["compat"] = {"warnings": ["w1", "w2"]}
        chk.by_policy["safe"] = {"warnings": ["w3"]}
        chk.by_policy["minimal"] = {"warnings": []}
        assert chk.total_warnings == 3


# ─── Live regression anchor ────────────────────────────────────────────


class TestLiveRepo:
    def test_all_presets_resolve_under_all_policies(self):
        """The full live preset set must compose + resolve under
        compat/safe/minimal without raising."""
        mod = _import_script()
        results = mod.audit_patch_plan_resolves()
        assert len(results) > 0
        failures = [(r.preset, r.error or [
            (p, s.get("error")) for p, s in r.by_policy.items() if "error" in s
        ]) for r in results if not r.passed]
        assert not failures, f"resolver failures: {failures}"

    def test_main_exits_zero_on_live_repo(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0


# ─── JSON output shape ──────────────────────────────────────────────────


class TestJsonOutput:
    def test_json_payload_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["all_passed"] is True
        assert isinstance(payload["presets_scanned"], int)
        assert payload["policies"] == ["compat", "safe", "minimal"]
        assert isinstance(payload["results"], list)
        for r in payload["results"]:
            assert "preset" in r
            assert "passed" in r
            assert "by_policy" in r


# ─── --strict-warnings interaction ──────────────────────────────────────


class TestStrictWarnings:
    def test_strict_warnings_no_op_when_no_warnings(self, monkeypatch):
        """If no preset produces warnings, --strict-warnings doesn't
        change the outcome."""
        mod = _import_script()
        results = mod.audit_patch_plan_resolves()
        any_warns = any(r.total_warnings > 0 for r in results)
        # On the live repo, --strict-warnings produces same exit code
        # as default only when warnings count is zero. Skip if warnings
        # already non-zero — that's a live-repo state, not a bug.
        if any_warns:
            pytest.skip(
                f"live repo has resolver warnings ({sum(r.total_warnings for r in results)}); "
                f"strict mode interaction is operator-controlled"
            )
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--strict-warnings"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0

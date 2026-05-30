# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/check_dirty_state.py` — three-tier dirty-state gate.

Contract:

  1. _matches_any: fnmatch + `/**` recursive prefix patterns.
  2. _check_entry routes untracked vs modified-tracked correctly.
  3. Untracked file: forbidden pattern → reject; allow pattern → accept;
     neither → reject (default-deny on untracked).
  4. Modified-tracked file: forbidden pattern → reject; otherwise accept
     (default-allow on tracked modifications unless tier forbids).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_dirty_state.py"


def _import_script():
    name = "_check_dirty_state_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Pattern matcher ───────────────────────────────────────────────────


class TestMatchesAny:
    def test_exact_literal_matches(self):
        mod = _import_script()
        assert mod._matches_any("foo.py", ["foo.py"])

    def test_no_match_returns_false(self):
        mod = _import_script()
        assert not mod._matches_any("foo.py", ["bar.py"])

    def test_simple_glob(self):
        mod = _import_script()
        assert mod._matches_any("foo.py", ["*.py"])
        assert not mod._matches_any("foo.txt", ["*.py"])

    def test_recursive_double_star(self):
        """`a/**` matches a/file and a/sub/file."""
        mod = _import_script()
        assert mod._matches_any("a/file.py", ["a/**"])
        assert mod._matches_any("a/sub/file.py", ["a/**"])
        assert not mod._matches_any("b/file.py", ["a/**"])

    def test_dir_prefix_with_trailing_double_star(self):
        """`a/**` shortcut: any path that starts with `a/` matches."""
        mod = _import_script()
        assert mod._matches_any("a/x/y/z.py", ["a/**"])

    def test_empty_pattern_list(self):
        mod = _import_script()
        assert not mod._matches_any("anything", [])


# ─── Untracked file routing ────────────────────────────────────────────


class TestUntrackedFile:
    def test_untracked_in_allow_accepted(self):
        mod = _import_script()
        policy = {
            "forbidden_untracked": [],
            "allow_untracked": ["build/**"],
            "forbidden_tracked_modified": [],
        }
        ok, reason = mod._check_entry("??", "build/output.txt", policy)
        assert ok
        assert "allowed untracked" in reason

    def test_untracked_in_forbidden_rejected(self):
        mod = _import_script()
        policy = {
            "forbidden_untracked": ["secrets/**"],
            "allow_untracked": [],
            "forbidden_tracked_modified": [],
        }
        ok, reason = mod._check_entry("??", "secrets/key.txt", policy)
        assert not ok
        assert "forbidden untracked" in reason

    def test_untracked_no_match_defaults_to_reject(self):
        """Default-deny: untracked file not on either list → reject."""
        mod = _import_script()
        policy = {
            "forbidden_untracked": [],
            "allow_untracked": [],
            "forbidden_tracked_modified": [],
        }
        ok, reason = mod._check_entry("??", "random.txt", policy)
        assert not ok
        assert "not in tier allowlist" in reason


# ─── Modified-tracked file routing ─────────────────────────────────────


class TestModifiedTracked:
    def test_modified_default_accepted(self):
        """Default-allow: modified-tracked file not in forbidden list → accept."""
        mod = _import_script()
        policy = {
            "forbidden_untracked": [],
            "allow_untracked": [],
            "forbidden_tracked_modified": [],
        }
        ok, reason = mod._check_entry("M", "src/file.py", policy)
        assert ok

    def test_modified_in_forbidden_rejected(self):
        """When tier forbids a modified-tracked path, reject."""
        mod = _import_script()
        policy = {
            "forbidden_untracked": [],
            "allow_untracked": [],
            "forbidden_tracked_modified": ["release/**"],
        }
        ok, reason = mod._check_entry("M", "release/SBOM.json", policy)
        assert not ok
        assert "modified tracked file not allowed" in reason

    def test_staged_modified_treated_as_modified(self):
        """Two-char code 'M ' (staged) and ' M' (unstaged) both → tracked-modified."""
        mod = _import_script()
        policy = {
            "forbidden_untracked": [],
            "allow_untracked": [],
            "forbidden_tracked_modified": ["forbidden/**"],
        }
        # Either form is treated identically as a tracked modification.
        ok, _ = mod._check_entry("M", "forbidden/x.py", policy)
        assert not ok
        ok, _ = mod._check_entry("MM", "forbidden/x.py", policy)
        assert not ok


# ─── Policy file loading ───────────────────────────────────────────────


class TestPolicyLoading:
    def test_live_policy_exists(self):
        """Regression anchor — the policy file must exist."""
        mod = _import_script()
        assert mod.POLICY_PATH.exists(), (
            f"policy file expected at {mod.POLICY_PATH}"
        )

    def test_live_policy_has_three_tiers(self):
        mod = _import_script()
        policy = mod._load_policy()
        for tier in ("dev", "audit", "release"):
            assert tier in policy, f"policy missing tier: {tier}"

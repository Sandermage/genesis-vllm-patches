# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_patch_lifecycle.py` — Entry 30."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_patch_lifecycle.py"


def _import():
    name = "_audit_v2_patch_lifecycle_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   # register before exec for dataclass introspection
    spec.loader.exec_module(mod)
    return mod


class TestSchema:
    def test_retired_is_disallowed(self):
        mod = _import()
        assert "retired" in mod.DISALLOWED_LIFECYCLES

    def test_allowlist_has_known_retired_patches(self):
        mod = _import()
        # E30 freeze: three retired patches operator-allowlisted.
        assert "PN19" in mod.ALLOWED_RETIRED_PATCHES
        assert "PN52" in mod.ALLOWED_RETIRED_PATCHES
        assert "P94"  in mod.ALLOWED_RETIRED_PATCHES


class TestLiveRepo:
    def test_committed_models_clean(self):
        """All retired-enabled patches in committed V2 models must be
        on the allowlist."""
        mod = _import()
        results = mod.audit_v2_patch_lifecycle()
        failed = [r for r in results if not r.passed]
        assert failed == [], "\n".join(
            f"  {r.model_id}: violations={r.violations}"
            for r in failed
        )
        # Phase 5.4 (2026-05-22): refreshed for current fleet
        # (10 V2 model YAMLs; was 6 in Wave 9/10 era).
        # Reconciled 2026-06-19 to live count: 11 model YAMLs — the 11th
        # is qwen3.6-7b-dense (committed club-3090 #58 Path A DENSE
        # reference; one lifecycle-check result per model YAML).
        assert len(results) == 11

    def test_enabled_patches_actually_count(self):
        """Sanity: at least one model has ≥ 20 enabled patches (not all 0)."""
        mod = _import()
        results = mod.audit_v2_patch_lifecycle()
        counts = [r.enabled_patches for r in results]
        assert max(counts) >= 20

    def test_lifecycle_groups_populated(self):
        """At least one model shows retired count = 1-3 (allowlisted)."""
        mod = _import()
        results = mod.audit_v2_patch_lifecycle()
        retired_seen = any(r.by_lifecycle.get("retired", 0) >= 1
                           for r in results)
        assert retired_seen


class TestScriptCLI:
    def test_cli_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_json(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "allowed_retired_patches" in payload
        assert payload["failed"] == 0

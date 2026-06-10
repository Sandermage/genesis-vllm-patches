# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_env_flag_consumers.py`` — orphan env-flag
defensive gate (Phase 10.5 D-extension 2026-06-01)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_env_flag_consumers.py"


def _import_script():
    name = "_audit_env_flag_consumers_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Strip-prefix helper ────────────────────────────────────────────────────


class TestStripPrefix:
    def test_strip_genesis_enable(self):
        mod = _import_script()
        assert mod._strip_prefix("GENESIS_ENABLE_P67") == "P67"

    def test_strip_sndr_enable(self):
        mod = _import_script()
        assert mod._strip_prefix("SNDR_ENABLE_PN283_PROC_BRIDGE") == "PN283_PROC_BRIDGE"

    def test_strip_info_prefix(self):
        mod = _import_script()
        # Phase 10.5 added INFO_ as canonical 5th category.
        assert mod._strip_prefix(
            "GENESIS_INFO_G4_T1_PR42006_OVERLAY_MOUNTED"
        ) == "G4_T1_PR42006_OVERLAY_MOUNTED"

    def test_no_known_prefix_passthrough(self):
        mod = _import_script()
        assert mod._strip_prefix("WEIRD_NAME_NOT_PREFIXED") == (
            "WEIRD_NAME_NOT_PREFIXED"
        )


# ─── Live corpus — regression anchor ───────────────────────────────────────


class TestLiveCorpus:
    def test_no_orphans_on_committed_tree(self):
        mod = _import_script()
        orphans = mod.audit()
        assert orphans == [], (
            "Orphan env_flags (registered but no consumer in sndr/):\n"
            + "\n".join(f"  {o['patch_id']}: {o['env_flag']}" for o in orphans)
        )

    def test_cli_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_cli_json_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert {"orphan_count", "orphans", "passed"} <= set(payload)
        assert payload["passed"] is True
        assert payload["orphan_count"] == 0


# ─── Synthetic orphan detection ────────────────────────────────────────────


class TestSyntheticOrphan:
    def test_orphan_fires_when_no_consumer(self, monkeypatch):
        """Inject a synthetic registry entry whose env_flag is read
        nowhere in vllm/sndr_core — the audit must flag it."""
        mod = _import_script()
        from sndr.dispatcher import registry as live_registry
        # Synth registry with one active orphan + one valid entry.
        # Use a name guaranteed to not appear in the real source tree.
        synthetic = {
            "PX_TOTALLY_UNUSED_FLAG_XYZ_2026": {
                "title": "synthetic test orphan",
                "env_flag": "GENESIS_ENABLE_TOTALLY_UNUSED_FLAG_XYZ_2026",
                "lifecycle": "experimental",
                "tier": "community",
                "family": "memory",
                "default_on": False,
                "category": "kv_cache",
            },
        }
        monkeypatch.setattr(live_registry, "PATCH_REGISTRY", synthetic)
        orphans = mod.audit()
        flagged = {o["patch_id"] for o in orphans}
        assert "PX_TOTALLY_UNUSED_FLAG_XYZ_2026" in flagged

    def test_retired_lifecycle_exempt(self, monkeypatch):
        """Retired entries are exempt — they're not supposed to have
        live consumers (the wiring lives under _retired/)."""
        mod = _import_script()
        from sndr.dispatcher import registry as live_registry
        synthetic = {
            "PX_RETIRED_SYNTH_2026": {
                "title": "retired",
                "env_flag": "GENESIS_ENABLE_TOTALLY_UNUSED_FLAG_XYZ_2026",
                "lifecycle": "retired",  # ← exempt
                "tier": "community",
                "family": "memory",
                "default_on": False,
                "category": "kv_cache",
            },
        }
        monkeypatch.setattr(live_registry, "PATCH_REGISTRY", synthetic)
        orphans = mod.audit()
        assert not any(o["patch_id"] == "PX_RETIRED_SYNTH_2026" for o in orphans)

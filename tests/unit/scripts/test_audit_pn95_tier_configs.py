# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_pn95_tier_configs.py`` — Phase 10.5 D.2
PN95 tier_configs audit gate.

Contract:

  1. ``audit()`` returns no findings on the committed corpus.
  2. ``--json`` payload has the documented shape.
  3. CLI exits 0 on the live tree.
  4. PN95-T-002 (unknown device) fires on a synthesized bad tier.
  5. PN95-T-003 (over-capacity) fires when GPU tier total > rig total.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_pn95_tier_configs.py"


def _import_script():
    name = "_audit_pn95_tier_configs_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Live corpus ────────────────────────────────────────────────────────────


class TestLiveCorpus:
    def test_audit_clean_on_committed_tree(self):
        mod = _import_script()
        findings = mod.audit()
        errors = [f for f in findings if f.severity == "error"]
        assert errors == [], (
            "PN95 tier_configs invariants violated:\n"
            + "\n".join(f"  {f.rule_id} {f.yaml_id}: {f.message}" for f in errors)
        )

    def test_cli_exits_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_cli_json_payload_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert {"errors", "warnings", "counts", "passed"} <= set(payload)
        assert payload["counts"]["error"] == 0
        assert payload["passed"] is True


# ─── Synthetic regression anchors ──────────────────────────────────────────


class TestSyntheticBad:
    def test_unknown_device_fires_PN95_T_002(self, tmp_path, monkeypatch):
        mod = _import_script()
        # Synthesize a bad tier_config
        bad = tmp_path / "bad-config.yaml"
        bad.write_text(
            "tiers:\n"
            "  - device: nvram\n"  # ← unknown device
            "    capacity_gib: 8.0\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "TIER_DIR", tmp_path)
        # Also patch loader directory
        from sndr.cache.pn95 import tier_config_loader as loader
        monkeypatch.setattr(loader, "_TIER_CONFIG_DIR", tmp_path)
        findings = mod.audit()
        codes = {f.rule_id for f in findings if f.severity == "error"}
        assert "PN95-T-002" in codes, [f.__dict__ for f in findings]

    def test_over_capacity_fires_PN95_T_003(self, tmp_path, monkeypatch):
        mod = _import_script()
        # 80 GiB GPU declared on an a5000-2x rig (advertised 48 GiB).
        bad = tmp_path / "a5000-2x-over.yaml"
        bad.write_text(
            "tiers:\n"
            "  - device: gpu\n"
            "    capacity_gib: 80.0\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "TIER_DIR", tmp_path)
        from sndr.cache.pn95 import tier_config_loader as loader
        monkeypatch.setattr(loader, "_TIER_CONFIG_DIR", tmp_path)
        findings = mod.audit()
        codes = {f.rule_id for f in findings if f.severity == "error"}
        assert "PN95-T-003" in codes, [f.__dict__ for f in findings]


# ─── Schema invariant — loader rejects malformed top-level ──────────────────


class TestLoaderRejections:
    def test_top_level_list_raises_PN95_T_001(self, tmp_path, monkeypatch):
        mod = _import_script()
        bad = tmp_path / "list-top.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        monkeypatch.setattr(mod, "TIER_DIR", tmp_path)
        from sndr.cache.pn95 import tier_config_loader as loader
        monkeypatch.setattr(loader, "_TIER_CONFIG_DIR", tmp_path)
        findings = mod.audit()
        codes = {f.rule_id for f in findings if f.severity == "error"}
        assert "PN95-T-001" in codes, [f.__dict__ for f in findings]

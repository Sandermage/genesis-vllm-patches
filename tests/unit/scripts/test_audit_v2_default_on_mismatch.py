# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_default_on_mismatch.py` — Entry 31."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_default_on_mismatch.py"


def _import():
    name = "_audit_v2_default_on_mismatch_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestLiveRepo:
    # Legitimate default_on overrides that the V2 surface intentionally
    # documents. Each entry is (model_id, env_flag, expected_value)
    # and must carry a YAML comment explaining the override.
    #
    # Phase 5.4 (2026-05-22): appended 4 documented Gemma overrides
    # discovered by the read-only triage. The Gemma 26B-A4B variants
    # explicitly disable the Ampere stability guards G4_02 and G4_13
    # (the guards are scoped to specific shape mismatches not present
    # on the canonical 26B-A4B variant). Each disable carries a YAML
    # comment in the respective gemma-4-26b-* ModelDef.
    _ALLOWED_OVERRIDES: set[tuple[str, str, str]] = {
        ("qwen3.6-35b-a3b-fp8", "GENESIS_LEGACY_P7", "0"),
        ("gemma-4-26b-a4b-it-awq",
         "GENESIS_ENABLE_G4_02_GEMMA4_MARLIN_KDIM_GUARD", "0"),
        ("gemma-4-26b-a4b-it-awq",
         "GENESIS_ENABLE_G4_13_GEMMA4_PER_TOKEN_HEAD_KV_GUARD", "0"),
        ("gemma-4-26b-a4b-it-awq-experimental",
         "GENESIS_ENABLE_G4_02_GEMMA4_MARLIN_KDIM_GUARD", "0"),
        ("gemma-4-26b-a4b-it-awq-experimental",
         "GENESIS_ENABLE_G4_13_GEMMA4_PER_TOKEN_HEAD_KV_GUARD", "0"),
        # 2026-06 vendor wave: PN119 targets the upstream TQ decode path;
        # Gemma 4 serves TQ via the wrapper path, so the patch is N/A
        # there (documented in the ModelDef YAML comment).
        ("gemma-4-31b-it-awq", "GENESIS_ENABLE_PN119", "0"),
    }

    def test_only_documented_overrides_present(self):
        """Every default_on override that surfaces must be on the
        allowlist above (with a documenting comment in the YAML). The
        audit script itself is informational and never fails CI; this
        test is what ratchets the allowlist."""
        mod = _import()
        results = mod.audit_v2_default_on_mismatch()
        observed: set[tuple[str, str, str]] = set()
        for r in results:
            assert r.error == ""
            for ov in r.overrides:
                observed.add((r.model_id, ov["env_flag"], ov["model_value"]))
        unexpected = observed - self._ALLOWED_OVERRIDES
        assert not unexpected, (
            f"undocumented default_on overrides: {sorted(unexpected)} — "
            "either add a YAML justification comment and update the "
            "allowlist, or revert the override"
        )
        # Phase 5.4 (2026-05-22): refreshed for current fleet
        # (10 V2 model YAMLs; was 6 in Wave 9/10 era).
        # Reconciled 2026-06-19 to live count: 11 model YAMLs — the 11th
        # is qwen3.6-7b-dense (committed club-3090 #58 Path A DENSE
        # reference; one override-check result per model YAML). The
        # _ALLOWED_OVERRIDES allowlist is unchanged: 7b-dense carries no
        # default_on overrides, so it adds a clean (empty-overrides)
        # result without expanding the allowlist.
        assert len(results) == 11


class TestSyntheticOverride:
    def test_synthetic_override_surfaced(self, tmp_path):
        mod = _import()
        # Live registry has 33 default_on=True patches; pick one's env_flag
        # and disable it. Use P1 (default_on=True per E30 survey).
        try:
            from sndr.dispatcher.registry import PATCH_REGISTRY
        except ImportError:
            pytest.skip("PATCH_REGISTRY not importable in test env")
        p1_flag = PATCH_REGISTRY["P1"].get("env_flag")
        assert p1_flag, "P1 must have env_flag"
        fake_yaml = tmp_path / "synth.yaml"
        fake_yaml.write_text(textwrap.dedent(f"""
            id: synth
            kind: model
            patches:
              {p1_flag}: '0'
        """).lstrip("\n"), encoding="utf-8")
        flag_idx = mod._build_flag_to_default_on_pids()
        r = mod.check_one_model(fake_yaml, flag_idx)
        # Informational: passes (never blocks).
        assert r.passed is True
        # But the override IS surfaced.
        assert len(r.overrides) == 1
        assert r.overrides[0]["patch_id"] == "P1"
        assert r.overrides[0]["model_value"] == "0"


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
        assert "total_overrides" in payload

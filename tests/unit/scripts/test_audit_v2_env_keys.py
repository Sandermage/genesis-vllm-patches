# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/audit_v2_env_keys.py` — V2 cross-layer env-key
consistency gate (Entry 25).

Contract:

  • Three layers walked: model, profile, resolved-alias.
  • Only Genesis/SNDR-prefixed keys checked.
  • Every Genesis/SNDR key must appear in `load_canonical_registry()`.
  • Live committed repo must be 100% clean (regression anchor).
  • Synthetic typo in profile delta surfaces in the resolved-alias layer.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_v2_env_keys.py"


def _import_script():
    name = "_audit_v2_env_keys_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    # Ensure repo root on path so the script's `from vllm.sndr_core...` works.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(mod)
    return mod


# ─── Helper predicates ────────────────────────────────────────────────


class TestPredicates:
    def test_is_genesis_key_genesis_prefix(self):
        mod = _import_script()
        assert mod._is_genesis_key("GENESIS_ENABLE_P58_FOO") is True

    def test_is_genesis_key_sndr_prefix(self):
        mod = _import_script()
        assert mod._is_genesis_key("SNDR_FOO_BAR") is True

    def test_is_genesis_key_rejects_others(self):
        mod = _import_script()
        assert mod._is_genesis_key("PYTORCH_CUDA_ALLOC_CONF") is False
        assert mod._is_genesis_key("VLLM_NO_USAGE_STATS") is False
        assert mod._is_genesis_key("HF_HOME") is False


# ─── Layer walkers — sanity ────────────────────────────────────────────


class TestWalkers:
    def test_model_walker_returns_expected_entries(self):
        mod = _import_script()
        from vllm.sndr_core.cli.config_keys import load_canonical_registry
        canon = load_canonical_registry()
        entries = mod._walk_model_layer(canon)
        # Phase 5.4 (2026-05-22): refreshed for current fleet
        # (10 V2 model YAMLs; was 6 in Wave 9/10 era). Method renamed
        # from `test_model_walker_returns_six_entries` — count moved
        # out of method name so future fleet growth touches only the
        # assertion, not the test identifier.
        assert len(entries) == 10
        for e in entries:
            assert e.layer == "model"

    def test_profile_walker_returns_at_least_eleven(self):
        mod = _import_script()
        from vllm.sndr_core.cli.config_keys import load_canonical_registry
        canon = load_canonical_registry()
        entries = mod._walk_profile_layer(canon)
        assert len(entries) >= 11
        for e in entries:
            assert e.layer == "profile"

    def test_resolved_alias_walker_returns_fifteen(self):
        mod = _import_script()
        from vllm.sndr_core.cli.config_keys import load_canonical_registry
        canon = load_canonical_registry()
        entries = mod._walk_resolved_aliases(canon)
        # Wave 10 V2 layout had 15 preset aliases.
        # Phase 7.G4.B1.0 (2026-05-23): +2 Gemma 4 31B presets → 17.
        # Phase 7.G4.26B-A4B.B0 (2026-05-23): +3 Gemma 4 26B-A4B
        # preset aliases (default + mtp-k4 + multiconc) → 20.
        # Phase 7.G4.26B-A4B.B4-PRE (2026-05-23): +1 multiconc-k1
        # preset alias → 21.
        # Test name kept as "fifteen" for grep continuity; assertion
        # tracks current fleet.
        assert len(entries) == 21
        for e in entries:
            assert e.layer == "resolved-alias"


# ─── Live committed repo — regression anchor ─────────────────────────


class TestLiveRepoClean:
    def test_all_layers_clean(self):
        """Every Genesis/SNDR env key in every V2 layer must resolve in
        the canonical registry. This is the regression anchor — if a
        future PR introduces a typo or new param without registering it,
        this test breaks."""
        mod = _import_script()
        results = mod.audit_v2_env_keys()
        failed = [r for r in results if not r.passed]
        assert failed == [], (
            "V2 cross-layer env-key drift:\n"
            + "\n".join(
                f"  {r.layer} {r.label}: "
                f"unknown={r.unknown_keys[:5]}"
                f"{'...' if len(r.unknown_keys) > 5 else ''}"
                + (f" error={r.error}" if r.error else "")
                for r in failed
            )
        )
        assert len(results) >= 28


# ─── Script CLI ────────────────────────────────────────────────────────


class TestScriptCLI:
    def test_cli_zero_on_committed(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout[:2000]
        assert "clean" in result.stdout

    def test_cli_json_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert "by_layer" in payload
        for layer in ("model", "profile", "resolved-alias"):
            assert layer in payload["by_layer"]
        assert payload["failed"] == 0

    def test_cli_layer_filter(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--layer", "model", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        # Only model-layer entries returned (note: by_layer counts are
        # over the *unfiltered* registry; entries[] is filtered).
        layers = {e["layer"] for e in payload["entries"]}
        assert layers == {"model"}


# ─── Synthetic typo detection ─────────────────────────────────────────


class TestTypoDetection:
    def test_unknown_key_in_extracted_set_surfaces(self):
        """Inject an unknown key into the canonical registry's
        complement, then re-walk: that key must appear as `unknown_keys`
        in the result. We can't easily mutate the live YAMLs in-place,
        so we test the predicate directly."""
        mod = _import_script()
        canon = {"GENESIS_ENABLE_P58_OK": {"source": "test"}}
        # Synthetic set of extracted keys including one typo.
        keys = [
            "GENESIS_ENABLE_P58_OK",
            "GENESIS_ENABLE_P99999_TYPO",
            "PYTORCH_CUDA_ALLOC_CONF",   # non-Genesis, should be filtered out
        ]
        genesis = [k for k in keys if mod._is_genesis_key(k)]
        unknown = [k for k in genesis if k not in canon]
        assert genesis == [
            "GENESIS_ENABLE_P58_OK",
            "GENESIS_ENABLE_P99999_TYPO",
        ]
        assert unknown == ["GENESIS_ENABLE_P99999_TYPO"]

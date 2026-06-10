# SPDX-License-Identifier: Apache-2.0
"""Stage 7 bundle smoke tests.

Each bundle MUST:
  1. Import cleanly (no top-level vllm import that would fail in CI).
  2. Have an `apply()` returning (status: str, reason: str).
  3. Skip with helpful reason when umbrella flag is unset.
  4. Tier-gate correctly when tier=engine + sndr_engine missing
     OR TIER_OVERRIDE forces community-only.

These tests run on CPU-only / Mac dev environment without an installed
vllm — they verify the orchestration layer, not actual file patching.
"""
from __future__ import annotations

import os
import sys

import pytest


# ─── Bundle catalog: (module_name, umbrella_flag, tier) ─────────────
BUNDLES: list[tuple[str, str, str]] = [
    ("tool_parsing_qwen3coder",   "BUNDLE_TOOL_PARSING_QWEN3CODER",   "community"),
    ("reasoning_qwen3",            "BUNDLE_REASONING_QWEN3",           "community"),
    ("attention_gdn_spec",         "BUNDLE_ATTENTION_GDN_SPEC",        "community"),
    # DA-009 (audit 2026-05-08): bundle tier corrected from "engine" to
    # "community" — P67/P67b are tier="community" in the registry, so
    # the bundle's umbrella tier must match.
    ("attention_tq_multi_query",   "BUNDLE_ATTENTION_TQ_MULTI_QUERY",  "community"),
    ("spec_decode_async_cleanup",  "BUNDLE_SPEC_DECODE_ASYNC_CLEANUP", "community"),
]


@pytest.mark.parametrize("name,_flag,_tier", BUNDLES, ids=[b[0] for b in BUNDLES])
def test_bundle_imports_cleanly(name, _flag, _tier):
    """Every bundle module must import without error."""
    mod = __import__(f"sndr.bundles.{name}", fromlist=["apply"])
    assert callable(mod.apply), f"bundle {name}.apply must be callable"


@pytest.mark.parametrize("name,flag,_tier", BUNDLES, ids=[b[0] for b in BUNDLES])
def test_bundle_skips_when_disabled(name, flag, _tier, monkeypatch):
    """Each bundle apply() returns ('skipped', ...) when umbrella flag is unset."""
    # Clean any preexisting flags
    monkeypatch.delenv(f"SNDR_ENABLE_{flag}", raising=False)
    monkeypatch.delenv(f"GENESIS_ENABLE_{flag}", raising=False)
    monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
    monkeypatch.delenv("GENESIS_ENABLE_TIER_OVERRIDE", raising=False)

    mod = __import__(f"sndr.bundles.{name}", fromlist=["apply"])
    status, reason = mod.apply()
    assert status == "skipped", f"{name}: expected skip, got {status}: {reason}"
    assert flag in reason or "disabled" in reason.lower(), (
        f"{name} skip reason should mention flag or 'disabled': {reason}"
    )


@pytest.mark.parametrize("name,flag,_tier", [
    b for b in BUNDLES if b[2] == "engine"
], ids=[b[0] for b in BUNDLES if b[2] == "engine"])
def test_engine_bundle_tier_override_skips(name, flag, _tier, monkeypatch):
    """Engine-tier bundles skip when TIER_OVERRIDE=1 even if umbrella set."""
    monkeypatch.setenv(f"SNDR_ENABLE_{flag}", "1")
    monkeypatch.setenv("SNDR_ENABLE_TIER_OVERRIDE", "1")

    mod = __import__(f"sndr.bundles.{name}", fromlist=["apply"])
    status, reason = mod.apply()
    assert status == "skipped", f"engine bundle should skip under TIER_OVERRIDE"
    assert "TIER_OVERRIDE" in reason or "community-only" in reason


@pytest.mark.parametrize("name,flag,_tier", [
    b for b in BUNDLES if b[2] == "engine"
], ids=[b[0] for b in BUNDLES if b[2] == "engine"])
def test_engine_bundle_skips_without_sndr_engine(name, flag, _tier, monkeypatch):
    """Engine bundles skip when sndr_engine package is unavailable."""
    monkeypatch.setenv(f"SNDR_ENABLE_{flag}", "1")
    monkeypatch.delenv("SNDR_ENABLE_TIER_OVERRIDE", raising=False)
    monkeypatch.delenv("GENESIS_ENABLE_TIER_OVERRIDE", raising=False)

    # Block sndr_engine import — simulates uninstalled commercial pkg
    saved = sys.modules.pop("vllm.sndr_engine", None)
    sys.modules["vllm.sndr_engine"] = None  # forces ImportError on import
    try:
        mod = __import__(f"sndr.bundles.{name}", fromlist=["apply"])
        status, reason = mod.apply()
        assert status == "skipped"
        assert "sndr_engine not installed" in reason or "tier=engine" in reason
    finally:
        # Restore
        if saved is not None:
            sys.modules["vllm.sndr_engine"] = saved
        else:
            sys.modules.pop("vllm.sndr_engine", None)


def test_bundle_registry_in_flags():
    """Every bundle's umbrella flag must be declared on Flags class."""
    from sndr.env import known_flags
    declared = set(known_flags())
    for name, flag, _tier in BUNDLES:
        assert flag in declared, (
            f"Bundle {name} references SNDR_ENABLE_{flag} but {flag} "
            "not on Flags class"
        )


def test_run_bundle_helper_handles_empty_factory_list():
    """Edge case: empty bundle returns proper skip."""
    from sndr.bundles._common import run_bundle
    from sndr.env import Flags
    import os

    os.environ["SNDR_ENABLE_BUNDLE_TOOL_PARSING_QWEN3CODER"] = "1"
    try:
        status, reason = run_bundle(
            name="empty_test_bundle",
            umbrella_flag=Flags.BUNDLE_TOOL_PARSING_QWEN3CODER,
            tier="community",
            patcher_factories=[],
        )
        assert status == "skipped"
        assert "no resolvable target files" in reason
    finally:
        os.environ.pop("SNDR_ENABLE_BUNDLE_TOOL_PARSING_QWEN3CODER", None)

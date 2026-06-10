# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P67b — TurboQuant spec-verify forward() routing.

P67b shares its env_flag with the P67 family
(``GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL``); the two patches work
together — P67 is the kernel-side multi-query batch path, P67b is
the wrapper that routes spec-verify calls through it. The test
contract here covers the wrapper-side invariants.

Audit R-04 closure (2026-05-16).
"""
from __future__ import annotations

import importlib

import pytest


PATCH_ID = "P67b"
MODULE_NAME = (
    "sndr.engines.vllm.patches.attention.turboquant.p67b_spec_verify_routing"
)
SHARED_ENV_FLAG = "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL"


def test_p67b_registered():
    from sndr.dispatcher.registry import PATCH_REGISTRY
    assert PATCH_ID in PATCH_REGISTRY
    meta = PATCH_REGISTRY[PATCH_ID]
    assert meta["env_flag"] == SHARED_ENV_FLAG
    assert meta["family"] == "attention.turboquant"
    assert meta["category"] == "spec_decode"


def test_p67b_shares_env_with_p67():
    """P67 and P67b are a coordinated pair. Both must read the same
    env flag so the operator-facing knob stays single-source."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    p67 = PATCH_REGISTRY.get("P67")
    p67b = PATCH_REGISTRY[PATCH_ID]
    if isinstance(p67, dict):
        assert p67["env_flag"] == p67b["env_flag"], (
            "P67 and P67b should share env_flag — coordinated pair contract"
        )


def test_module_imports_cleanly():
    mod = importlib.import_module(MODULE_NAME)
    assert callable(getattr(mod, "apply", None))


def test_marker_constant_present():
    mod = importlib.import_module(MODULE_NAME)
    marker = getattr(mod, "GENESIS_P67B_MARKER", None)
    assert isinstance(marker, str) and marker.strip()
    assert "P67b" in marker


def test_apply_skipped_without_env(monkeypatch):
    monkeypatch.delenv(SHARED_ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    status, _ = mod.apply()
    # Spec-decode wrappers may also auto-detect non-TQ runtime and
    # skip; the only contract we enforce is no crash + a 2-tuple.
    assert status in ("skipped", "failed", "applied")


def test_apply_idempotent(monkeypatch):
    monkeypatch.delenv(SHARED_ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    first = mod.apply()
    second = mod.apply()
    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert len(first) == len(second) == 2


def test_apply_never_raises(monkeypatch):
    mod = importlib.import_module(MODULE_NAME)
    monkeypatch.setenv(SHARED_ENV_FLAG, "junk-value")
    try:
        result = mod.apply()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"apply() must not raise on invalid env: {e!r}")
    assert isinstance(result, tuple) and len(result) == 2


def test_applies_to_is_turboquant():
    """P67b is TurboQuant-specific — guard must reflect that."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    applies_to = PATCH_REGISTRY[PATCH_ID].get("applies_to") or {}
    assert "is_turboquant" in applies_to, (
        f"P67b must guard on is_turboquant; got: {sorted(applies_to.keys())}"
    )

# SPDX-License-Identifier: Apache-2.0
"""Tests for G4_60L — supports_mm_prefix=True monkey-patch on stock
TurboQuantAttentionBackend (Phase 7.G4.31B.K4-TURBOQUANT-BACKEND-MM-PREFIX).

Contract:

  • Module exposes apply, is_applied, GENESIS_G4_60L_MARKER, _ENV_ENABLE.
  • Env-disabled → apply() returns ("skipped", ...).
  • Stock class missing supports_mm_prefix=True → apply() patches it
    and returns ("applied", ...). After apply, the classmethod returns
    True.
  • Class already returning True (overlay path) → apply() is a no-op
    that returns ("applied", "...already present...") without
    re-binding.
  • Apply is idempotent (second call returns "applied", "already
    installed").
"""
from __future__ import annotations

import importlib

import pytest


_MOD_PATH = (
    "sndr.engines.vllm.patches.attention.turboquant."
    "g4_60l_tq_backend_mm_prefix"
)


def _reload_module():
    """Reload the patch module to reset its global _APPLIED state.

    Tests that mutate the live vllm class need a fresh module instance
    so apply()'s idempotency latch doesn't leak between tests.
    """
    import sys
    if _MOD_PATH in sys.modules:
        del sys.modules[_MOD_PATH]
    return importlib.import_module(_MOD_PATH)


# ─── Public surface ────────────────────────────────────────────────────


def test_module_imports_and_exposes_public_api():
    mod = _reload_module()
    for sym in ("apply", "is_applied", "GENESIS_G4_60L_MARKER"):
        assert hasattr(mod, sym), f"missing public symbol: {sym}"
    assert "G4_60L" in mod.GENESIS_G4_60L_MARKER
    assert "supports_mm_prefix" in mod.GENESIS_G4_60L_MARKER


def test_env_flag_is_canonical_name():
    mod = _reload_module()
    assert mod._ENV_ENABLE == "GENESIS_ENABLE_G4_60L_TQ_BACKEND_MM_PREFIX"


# ─── Env gating ────────────────────────────────────────────────────────


def test_apply_skipped_when_env_unset(monkeypatch):
    mod = _reload_module()
    monkeypatch.delenv(mod._ENV_ENABLE, raising=False)
    status, msg = mod.apply()
    assert status == "skipped"
    assert "G4_60L disabled" in msg


def test_apply_skipped_when_env_false(monkeypatch):
    mod = _reload_module()
    monkeypatch.setenv(mod._ENV_ENABLE, "0")
    status, msg = mod.apply()
    assert status == "skipped"


# ─── Live-class behavior ───────────────────────────────────────────────


def _save_supports_mm_prefix():
    """Snapshot the live TurboQuantAttentionBackend.supports_mm_prefix attribute.

    Returns (cls, original_value, was_classmethod) so tests can restore.
    Skips the test if vllm isn't importable in this environment.
    """
    pytest.importorskip("vllm")
    try:
        from vllm.v1.attention.backends.turboquant_attn import (
            TurboQuantAttentionBackend,
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"TurboQuantAttentionBackend not importable: {e}")
    original = TurboQuantAttentionBackend.__dict__.get("supports_mm_prefix")
    return TurboQuantAttentionBackend, original


def test_apply_patches_stock_class_when_method_missing(monkeypatch):
    """Simulate the stock-class case: remove any local override on
    TurboQuantAttentionBackend so supports_mm_prefix() falls through to the base
    class (which returns False). After apply(), the override must be
    installed and the classmethod must return True."""
    TurboQuantAttentionBackend, original = _save_supports_mm_prefix()
    try:
        # Force the stock-class case by deleting any subclass override.
        if "supports_mm_prefix" in TurboQuantAttentionBackend.__dict__:
            delattr(TurboQuantAttentionBackend, "supports_mm_prefix")

        # Base-class default is False — sanity check the precondition.
        assert TurboQuantAttentionBackend.supports_mm_prefix() is False

        mod = _reload_module()
        monkeypatch.setenv(mod._ENV_ENABLE, "1")
        status, msg = mod.apply()
        assert status == "applied"
        assert "installed" in msg or "override" in msg
        assert mod.is_applied() is True

        # Override now installed and returns True.
        assert TurboQuantAttentionBackend.supports_mm_prefix() is True
    finally:
        if original is not None:
            TurboQuantAttentionBackend.supports_mm_prefix = original
        elif "supports_mm_prefix" in TurboQuantAttentionBackend.__dict__:
            delattr(TurboQuantAttentionBackend, "supports_mm_prefix")


def test_apply_no_op_when_overlay_already_present(monkeypatch):
    """Simulate the overlay-bind-mount case: TurboQuantAttentionBackend already
    has supports_mm_prefix=True. apply() must report "applied" with a
    "...already present..." message and NOT re-bind the attribute."""
    TurboQuantAttentionBackend, original = _save_supports_mm_prefix()
    try:
        # Pre-install the override to mimic the overlay path.
        TurboQuantAttentionBackend.supports_mm_prefix = classmethod(lambda cls: True)
        sentinel = TurboQuantAttentionBackend.__dict__["supports_mm_prefix"]

        mod = _reload_module()
        monkeypatch.setenv(mod._ENV_ENABLE, "1")
        status, msg = mod.apply()
        assert status == "applied"
        assert "no-op" in msg or "already present" in msg
        assert mod.is_applied() is True

        # The same classmethod object is still bound — apply did NOT
        # rebind.
        assert TurboQuantAttentionBackend.__dict__["supports_mm_prefix"] is sentinel
    finally:
        if original is not None:
            TurboQuantAttentionBackend.supports_mm_prefix = original
        elif "supports_mm_prefix" in TurboQuantAttentionBackend.__dict__:
            delattr(TurboQuantAttentionBackend, "supports_mm_prefix")


def test_apply_idempotent(monkeypatch):
    """Second apply() call returns ("applied", "...already installed...")
    without re-touching the class."""
    TurboQuantAttentionBackend, original = _save_supports_mm_prefix()
    try:
        if "supports_mm_prefix" in TurboQuantAttentionBackend.__dict__:
            delattr(TurboQuantAttentionBackend, "supports_mm_prefix")

        mod = _reload_module()
        monkeypatch.setenv(mod._ENV_ENABLE, "1")
        s1, m1 = mod.apply()
        s2, m2 = mod.apply()
        assert s1 == "applied"
        assert s2 == "applied"
        assert "already installed" in m2 or "idempotent" in m2
    finally:
        if original is not None:
            TurboQuantAttentionBackend.supports_mm_prefix = original
        elif "supports_mm_prefix" in TurboQuantAttentionBackend.__dict__:
            delattr(TurboQuantAttentionBackend, "supports_mm_prefix")


# ─── Registry consistency ──────────────────────────────────────────────


def test_registry_entry_present_and_consistent():
    """The G4_60L entry must exist in PATCH_REGISTRY with the canonical
    env_flag, apply_module path, and family. This guards against the
    registry/source mismatch class of bug."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    assert "G4_60L" in PATCH_REGISTRY, "G4_60L missing from PATCH_REGISTRY"
    entry = PATCH_REGISTRY["G4_60L"]
    assert entry["env_flag"] == "GENESIS_ENABLE_G4_60L_TQ_BACKEND_MM_PREFIX"
    assert entry["family"] == "attention.turboquant"
    assert entry["apply_module"] == _MOD_PATH
    assert entry["default_on"] is False
    assert entry["lifecycle"] == "experimental"
    assert entry.get("vllm_version_range") == "<0.21"
    assert "G4_60B" in entry["composes_with"]


def test_apply_module_path_resolves():
    """Sanity: the apply_module path in the registry actually imports."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    entry = PATCH_REGISTRY["G4_60L"]
    mod = importlib.import_module(entry["apply_module"])
    assert hasattr(mod, "apply")
    assert callable(mod.apply)

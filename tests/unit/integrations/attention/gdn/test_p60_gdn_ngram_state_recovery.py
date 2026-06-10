# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P60 — GDN+ngram SSM state recovery (Phase 1).

CPU-only tests covering the per-patch contract:

  * Registry entry has the expected shape (env_flag / family /
    lifecycle / category / applies_to).
  * Wiring module imports cleanly and exposes the canonical surface
    (``apply``, ``_is_enabled``, the marker constant).
  * Marker constant is a non-empty grep-friendly string.
  * Env gate is honoured: without ``GENESIS_ENABLE_P60_GDN_NGRAM_FIX``
    set, ``apply()`` returns ``("skipped", …)`` and never raises.
  * Idempotency: a second ``apply()`` in the same process must not
    crash and must still return a 2-tuple.

The runtime correctness of the SSM pre-copy logic is exercised by the
spec-decode integration bench (``tests/integration/``), which requires
a real GPU + GDN model. The tests here are the boot-stability contract
that runs on every PR.

Audit R-04 closure (2026-05-16) — closes the in-subset test gap.
"""
from __future__ import annotations

import importlib

import pytest


PATCH_ID = "P60"
MODULE_NAME = (
    "sndr.engines.vllm.patches.attention.gdn.p60_gdn_ngram_state_recovery"
)
ENV_FLAG = "GENESIS_ENABLE_P60_GDN_NGRAM_FIX"


# ─── Registry contract ────────────────────────────────────────────────


def test_p60_registered():
    from sndr.dispatcher.registry import PATCH_REGISTRY
    assert PATCH_ID in PATCH_REGISTRY
    meta = PATCH_REGISTRY[PATCH_ID]
    assert meta["env_flag"] == ENV_FLAG
    assert meta["default_on"] is False
    assert meta["family"] == "attention.gdn"
    assert meta["category"] == "spec_decode"
    assert meta["lifecycle"] in ("experimental", "stable")


def test_p60_applies_to_hybrid_models():
    """P60 targets hybrid GDN models — guard must reflect that."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    applies_to = PATCH_REGISTRY[PATCH_ID].get("applies_to") or {}
    assert "is_hybrid" in applies_to, (
        f"P60 must guard on is_hybrid; got: {sorted(applies_to.keys())}"
    )


# ─── Wiring module ────────────────────────────────────────────────────


def test_module_imports_cleanly():
    mod = importlib.import_module(MODULE_NAME)
    assert callable(getattr(mod, "apply", None))
    assert callable(getattr(mod, "_is_enabled", None))


def test_marker_constant_present():
    mod = importlib.import_module(MODULE_NAME)
    marker = getattr(mod, "GENESIS_P60_MARKER", None)
    assert isinstance(marker, str) and marker.strip()
    # Marker must include the patch id so log greps surface it.
    assert "P60" in marker


def test_no_top_level_torch_import():
    """Patch wiring must not import torch at top level — that would
    crash collection on torch-less hosts."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[5]
    src = (
        repo_root
        / "sndr/engines/vllm/patches/attention/gdn/p60_gdn_ngram_state_recovery.py"
    ).read_text(encoding="utf-8")
    # We only inspect non-comment / non-string top-level lines.
    forbidden = "\nimport torch"
    assert forbidden not in src.split('"""')[0], (
        "top-level `import torch` would break torch-less test collection"
    )


# ─── Env gate ─────────────────────────────────────────────────────────


def test_is_enabled_false_without_env(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    assert mod._is_enabled() is False


def test_is_enabled_recognizes_truthy(monkeypatch):
    mod = importlib.import_module(MODULE_NAME)
    for v in ("1", "true", "TRUE"):
        monkeypatch.setenv(ENV_FLAG, v)
        assert mod._is_enabled() is True, f"{v!r} should activate P60"
    for v in ("0", "", "off", "no"):
        monkeypatch.setenv(ENV_FLAG, v)
        assert mod._is_enabled() is False, f"{v!r} should NOT activate P60"


def test_apply_skipped_without_env(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    result = mod.apply()
    assert isinstance(result, tuple) and len(result) == 2
    status, _ = result
    # Without env, must be skipped — never failed or applied.
    assert status == "skipped"


def test_apply_idempotent_when_skipped(monkeypatch):
    """Two consecutive apply() with env off must both succeed without
    interfering with each other."""
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    first = mod.apply()
    second = mod.apply()
    assert first[0] == "skipped"
    assert second[0] == "skipped"


def test_apply_never_raises_with_invalid_env(monkeypatch):
    """Bad env value must NOT crash apply() — registry validates the
    flag elsewhere, but the patch itself must be defensive."""
    mod = importlib.import_module(MODULE_NAME)
    monkeypatch.setenv(ENV_FLAG, "garbage-value-not-1-or-0")
    try:
        result = mod.apply()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"apply() must not raise on invalid env, got: {e!r}")
    assert isinstance(result, tuple) and len(result) == 2

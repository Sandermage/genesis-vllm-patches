# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P60b — GDN+ngram Triton kernel offset (Phase 2).

CPU-only contract tests. Numerical correctness for the modified
causal_conv1d kernel runs on GPU as part of the spec-decode bench
(``tests/integration/``).

Audit R-04 closure (2026-05-16).
"""
from __future__ import annotations

import importlib

import pytest


PATCH_ID = "P60b"
MODULE_NAME = (
    "sndr.engines.vllm.patches.attention.gdn.p60b_gdn_ngram_triton_kernel"
)
ENV_FLAG = "GENESIS_ENABLE_P60B_TRITON_KERNEL"


def test_p60b_registered():
    from sndr.dispatcher.registry import PATCH_REGISTRY
    assert PATCH_ID in PATCH_REGISTRY
    meta = PATCH_REGISTRY[PATCH_ID]
    assert meta["env_flag"] == ENV_FLAG
    assert meta["default_on"] is False
    assert meta["family"] == "attention.gdn"
    assert meta["category"] == "spec_decode"


def test_p60b_pairs_with_p60():
    """P60b is Phase 2 of P60 — it should declare a dependency or
    document the pairing in its docstring."""
    mod = importlib.import_module(MODULE_NAME)
    src = (mod.__doc__ or "")
    assert "P60" in src, "P60b docstring must reference the P60 Phase-1 pairing"


def test_module_imports_cleanly():
    mod = importlib.import_module(MODULE_NAME)
    assert callable(getattr(mod, "apply", None))
    assert callable(getattr(mod, "_is_enabled", None))


def test_marker_constant_present():
    mod = importlib.import_module(MODULE_NAME)
    marker = getattr(mod, "GENESIS_P60B_MARKER", None)
    assert isinstance(marker, str) and marker.strip()
    assert "P60b" in marker


def test_is_enabled_false_without_env(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    assert mod._is_enabled() is False


def test_is_enabled_recognizes_truthy(monkeypatch):
    mod = importlib.import_module(MODULE_NAME)
    for v in ("1", "true", "TRUE"):
        monkeypatch.setenv(ENV_FLAG, v)
        assert mod._is_enabled() is True


def test_apply_skipped_without_env(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    status, _ = mod.apply()
    assert status == "skipped"


def test_apply_idempotent_when_skipped(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    first = mod.apply()
    second = mod.apply()
    assert first[0] == second[0] == "skipped"


def test_apply_never_raises_with_invalid_env(monkeypatch):
    mod = importlib.import_module(MODULE_NAME)
    monkeypatch.setenv(ENV_FLAG, "not-a-bool")
    try:
        result = mod.apply()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"apply() must not raise on invalid env: {e!r}")
    assert isinstance(result, tuple) and len(result) == 2


def test_clear_triton_cache_callable():
    """P60b ships a cache-clear helper — verify the signature so callers
    in apply() don't drift away from it."""
    mod = importlib.import_module(MODULE_NAME)
    assert callable(getattr(mod, "_clear_triton_cache", None))

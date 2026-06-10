# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN202 — per-layer KV tensor split (Tier 2.A enabler).

PN202 ships the per-layer KV-cache split that PN200 / PN201 / PN203
build on. CPU-only tests cover the wire-in contract; the
streaming-throughput side-effect is validated by the multi-conc bench
on GPU.

Audit R-04 closure (2026-05-16).
"""
from __future__ import annotations

import importlib

import pytest


PATCH_ID = "PN202"
MODULE_NAME = (
    "sndr.engines.vllm.patches.streaming.pn202_per_layer_kv_split"
)
ENV_FLAG = "GENESIS_ENABLE_PN202_PER_LAYER_KV_SPLIT"


def test_pn202_registered():
    from sndr.dispatcher.registry import PATCH_REGISTRY
    assert PATCH_ID in PATCH_REGISTRY
    meta = PATCH_REGISTRY[PATCH_ID]
    assert meta["env_flag"] == ENV_FLAG
    assert meta["family"] == "streaming"
    assert meta["category"] == "memory"
    assert meta["default_on"] is False


def test_applies_to_declares_pin_range():
    """PN202 modifies kv-cache shape — a pin guard is mandatory so the
    patch doesn't fire on a version where the anchor moved."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    applies_to = PATCH_REGISTRY[PATCH_ID].get("applies_to") or {}
    assert "vllm_version_range" in applies_to, (
        f"PN202 must declare vllm_version_range; got: "
        f"{sorted(applies_to.keys())}"
    )


def test_module_imports_cleanly():
    mod = importlib.import_module(MODULE_NAME)
    assert callable(getattr(mod, "apply", None))
    assert callable(getattr(mod, "_enabled", None))


def test_marker_constant_present():
    mod = importlib.import_module(MODULE_NAME)
    marker = getattr(mod, "GENESIS_MARKER", None)
    assert isinstance(marker, str) and marker.strip()
    assert "PN202" in marker


def test_enabled_false_without_env(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    assert mod._enabled() is False


def test_enabled_recognizes_truthy(monkeypatch):
    mod = importlib.import_module(MODULE_NAME)
    for v in ("1", "true", "TRUE"):
        monkeypatch.setenv(ENV_FLAG, v)
        assert mod._enabled() is True
    for v in ("0", "", "off"):
        monkeypatch.setenv(ENV_FLAG, v)
        assert mod._enabled() is False


def test_apply_skipped_without_env(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    status, reason = mod.apply()
    assert status == "skipped"
    # The skip reason must name the env flag so operators can flip it.
    assert ENV_FLAG in reason


def test_apply_idempotent_when_skipped(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    first = mod.apply()
    second = mod.apply()
    assert first[0] == second[0] == "skipped"


def test_apply_never_raises(monkeypatch):
    mod = importlib.import_module(MODULE_NAME)
    monkeypatch.setenv(ENV_FLAG, "garbage-value")
    try:
        result = mod.apply()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"apply() must not raise on invalid env: {e!r}")
    assert isinstance(result, tuple) and len(result) == 2


def test_two_part_patcher_factories():
    """PN202 declares Part A and Part B sub-patchers — both factories
    must be exposed and callable so the contract is auditable."""
    mod = importlib.import_module(MODULE_NAME)
    assert callable(getattr(mod, "_make_part_a_patcher", None))
    assert callable(getattr(mod, "_make_part_b_patcher", None))

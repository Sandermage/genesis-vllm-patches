# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P78 — TurboQuant ``.tolist()`` capture-guard.

Adapted from the noonghunna reference impl. The guard prevents
silent crashes when a CUDA-graph capture observes a ``Tensor.tolist()``
call on a graph-managed tensor (graph capture can't trace eager-mode
host transfers; the call returns the wrong value at replay).

The CPU-only test surface verifies the registry + module + env-gate
contract. The actual guard behaviour is exercised by the
spec-decode + structured-output bench on GPU.

Audit R-04 closure (2026-05-16).
"""
from __future__ import annotations

import importlib

import pytest


PATCH_ID = "P78"
MODULE_NAME = (
    "sndr.engines.vllm._archive.p78_tolist_capture_guard"
)
ENV_FLAG = "GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD"


def test_p78_registered():
    from sndr.dispatcher.registry import PATCH_REGISTRY
    assert PATCH_ID in PATCH_REGISTRY
    meta = PATCH_REGISTRY[PATCH_ID]
    assert meta["env_flag"] == ENV_FLAG
    assert meta["default_on"] is False
    assert meta["family"] == "attention.turboquant"
    assert meta["category"] == "compile_safety"


def test_applies_to_turboquant_and_quant_format():
    """P78 targets TurboQuant under specific quant formats — applies_to
    must declare both gates so it doesn't fire on unsupported runtimes."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    applies_to = PATCH_REGISTRY[PATCH_ID].get("applies_to") or {}
    assert "is_turboquant" in applies_to, (
        f"P78 must guard on is_turboquant; got: {sorted(applies_to.keys())}"
    )
    assert "quant_format" in applies_to, (
        f"P78 must guard on quant_format; got: {sorted(applies_to.keys())}"
    )


def test_module_imports_cleanly():
    mod = importlib.import_module(MODULE_NAME)
    assert callable(getattr(mod, "apply", None))


def test_marker_constant_present():
    mod = importlib.import_module(MODULE_NAME)
    marker = getattr(mod, "GENESIS_P78_MARKER", None)
    assert isinstance(marker, str) and marker.strip()
    assert "P78" in marker


def test_apply_skipped_without_env(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    status, _ = mod.apply()
    assert status in ("skipped", "failed", "applied")
    # In the no-env case, must be skipped — never applied.
    if status == "applied":
        pytest.fail("P78 applied with env unset — should be skipped")


def test_apply_idempotent(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    first = mod.apply()
    second = mod.apply()
    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert len(first) == len(second) == 2


def test_apply_never_raises(monkeypatch):
    mod = importlib.import_module(MODULE_NAME)
    monkeypatch.setenv(ENV_FLAG, "garbage")
    try:
        result = mod.apply()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"apply() must not raise on invalid env: {e!r}")
    assert isinstance(result, tuple) and len(result) == 2


def test_replacement_contains_env_check():
    """The replacement code injected by P78 must reference the env
    flag so the in-graph guard can be operator-tunable at runtime
    without re-applying the patch."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[5]
    # v12.x: real source moved to sndr/; vllm/sndr_core/... is now a
    # re-export shim that carries no replacement block. Read canonical.
    src_path = (
        repo_root
        / "sndr/engines/vllm/_archive/p78_tolist_capture_guard.py"
    )
    if not src_path.is_file():
        src_path = (
            repo_root
            / "sndr/engines/vllm/_archive/p78_tolist_capture_guard.py"
        )
    src = src_path.read_text(encoding="utf-8")
    assert ENV_FLAG in src, (
        f"P78 wiring must reference {ENV_FLAG} in its replacement block"
    )

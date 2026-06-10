# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PN122 — CUDA graph dispatch trace wire-in.

Formerly tracked as ``SPRINT26_CG_DISPATCH_TRACE``; renamed to PN122
in the 2026-05-14 audit cleanup. The patch hooks
``record_dispatch(matched)`` into the v1 cudagraph dispatcher call
site in ``gpu_model_runner.py`` so operators can capture per-request
dispatch decisions when ``GENESIS_CUDAGRAPH_DISPATCH_TRACE=1`` is set.

The runtime trace itself is operator-facing observability (no
production-traffic side-effect). These tests cover the wire-in
contract: registry / module shape / env gate / safe-skip when the
dispatcher decision says no.

Audit R-04 closure (2026-05-16).
"""
from __future__ import annotations

import importlib

import pytest


PATCH_ID = "PN122"
MODULE_NAME = (
    "sndr.engines.vllm.patches.observability."
    "pn122_sprint26_cudagraph_dispatch_trace"
)
ENV_FLAG = "GENESIS_ENABLE_PN122_CG_DISPATCH_TRACE"
LEGACY_ENV_FLAG = "GENESIS_ENABLE_SPRINT26_CG_DISPATCH_TRACE"


def test_pn122_registered():
    from sndr.dispatcher.registry import PATCH_REGISTRY
    assert PATCH_ID in PATCH_REGISTRY
    meta = PATCH_REGISTRY[PATCH_ID]
    assert meta["env_flag"] == ENV_FLAG
    assert meta["family"] == "observability"
    assert meta["category"] == "observability"
    assert meta["default_on"] is False


def test_legacy_env_documented():
    """The docstring or source must reference the pre-rename env name
    so operators with old launch scripts can grep their way back."""
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[4]
    # v12.x: real source moved to sndr/; vllm/sndr_core/... is a re-export
    # shim that carries no docstring/source markers. Read canonical.
    src_path = (
        repo_root
        / "sndr/engines/vllm/patches/observability/"
          "pn122_sprint26_cudagraph_dispatch_trace.py"
    )
    if not src_path.is_file():
        src_path = (
            repo_root
            / "sndr/engines/vllm/patches/observability/"
              "pn122_sprint26_cudagraph_dispatch_trace.py"
        )
    src = src_path.read_text(encoding="utf-8")
    assert LEGACY_ENV_FLAG in src, (
        f"PN122 wiring must document the legacy {LEGACY_ENV_FLAG} env "
        "so v7.x operators can locate the migrated patch"
    )


def test_module_imports_cleanly():
    mod = importlib.import_module(MODULE_NAME)
    assert callable(getattr(mod, "apply", None))


def test_marker_constant_present():
    mod = importlib.import_module(MODULE_NAME)
    marker = getattr(mod, "GENESIS_SPRINT26_DISPATCH_MARKER", None)
    assert isinstance(marker, str) and marker.strip()
    assert "Sprint 2.6" in marker or "Sprint26" in marker


def test_apply_skipped_without_env(monkeypatch):
    """With env unset, dispatcher.should_apply() returns False →
    apply() returns ('skipped', reason)."""
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.delenv(LEGACY_ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    status, _ = mod.apply()
    # Without env, must be skipped — never applied.
    assert status in ("skipped", "failed"), (
        f"unexpected status {status!r} when env unset"
    )


def test_apply_idempotent(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising=False)
    monkeypatch.delenv(LEGACY_ENV_FLAG, raising=False)
    mod = importlib.import_module(MODULE_NAME)
    first = mod.apply()
    second = mod.apply()
    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert len(first) == len(second) == 2


def test_apply_never_raises(monkeypatch):
    mod = importlib.import_module(MODULE_NAME)
    monkeypatch.setenv(ENV_FLAG, "junk-not-a-bool")
    try:
        result = mod.apply()
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"apply() must not raise on invalid env: {e!r}")
    assert isinstance(result, tuple) and len(result) == 2


def test_drift_markers_present():
    """The patcher must declare upstream_drift_markers so anchor drift
    after a pin bump fails cleanly (skipped) rather than corrupting
    upstream code."""
    mod = importlib.import_module(MODULE_NAME)
    patcher = mod._make_patcher() if hasattr(mod, "_make_patcher") else None
    if patcher is None:
        pytest.skip("vllm install root not resolvable (off-server test env)")
    markers = getattr(patcher, "upstream_drift_markers", None) or []
    assert markers, "PN122 patcher must declare upstream_drift_markers"

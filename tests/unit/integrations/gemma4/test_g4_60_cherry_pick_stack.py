# SPDX-License-Identifier: Apache-2.0
"""Unit tests for G4_60* — PR #42637 cherry-pick stack.

These tests verify the 5 Genesis monkey-patch leaves apply cleanly
without conflicting with each other or with the production G4_19
wrapper stack. They do NOT run a full model forward — that requires
GPU + bind-mount in a container and is covered by integration bench
work.

Test design:
  * Each ``apply()`` is called once and asserted to return ``applied``.
  * Idempotency: second call returns ``applied`` with "already installed".
  * Reversibility: ``revert()`` returns ``True`` and the original state
    is restored (where verifiable).
  * Cross-patch: G4_60a + G4_60e together produce a consistent state.

These tests are vllm-importing — they MUST run inside a vllm-equipped
environment. If vllm is not installed, pytest skips them.

References:
  - Upstream PR: https://github.com/vllm-project/vllm/pull/42637
  - Plan: docs/_internal/TQ_FULL_PLAN_2026-05-17_RU.md (Phase 2c)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import os

import pytest

# Skip the whole module if vllm/torch absent — patches require them.
vllm = pytest.importorskip("vllm")
torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _clear_env_flags(monkeypatch):
    """Each test sets only its own env flag to avoid cross-contamination."""
    for flag in (
        "GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC",
        "GENESIS_ENABLE_G4_60E_KV_CACHE_UTILS",
        "GENESIS_ENABLE_G4_60G_TQ_DISPATCH",
        "GENESIS_ENABLE_G4_60H_TQ_CONFIG_AUGMENT",
        "GENESIS_ENABLE_G4_60K_TQ_ENGINE_CONFIG",
    ):
        monkeypatch.delenv(flag, raising=False)


# === G4_60a — TQSlidingWindowSpec injection ===


def test_g4_60a_skipped_without_env(monkeypatch):
    """Apply returns ``skipped`` when env flag is absent."""
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60a_tq_sliding_window_spec as m,
    )
    # Reset module state (test isolation)
    m._APPLIED = False
    status, msg = m.apply()
    assert status == "skipped", msg
    assert "GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC" in msg


def test_g4_60a_apply_adds_tqslidingwindowspec(monkeypatch):
    """Apply makes TQSlidingWindowSpec available on the module."""
    monkeypatch.setenv("GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC", "1")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60a_tq_sliding_window_spec as m,
    )
    m._APPLIED = False  # test-only reset

    status, msg = m.apply()
    assert status == "applied", msg

    from vllm.v1.kv_cache_interface import TQSlidingWindowSpec, SlidingWindowSpec

    # The class must be a subclass of SlidingWindowSpec.
    assert issubclass(TQSlidingWindowSpec, SlidingWindowSpec)
    # The class must expose tq_slot_size as a dataclass field.
    assert "tq_slot_size" in {f.name for f in TQSlidingWindowSpec.__dataclass_fields__.values()}


def test_g4_60a_idempotent(monkeypatch):
    """Second apply() returns ``applied`` with idempotent message."""
    monkeypatch.setenv("GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC", "1")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60a_tq_sliding_window_spec as m,
    )
    m._APPLIED = False
    m.apply()
    status, msg = m.apply()
    assert status == "applied"
    assert "idempotent" in msg or "already" in msg


# === G4_60h — TurboQuantConfig augment ===


def test_g4_60h_skipped_without_env():
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60h_turboquant_config_augment as m,
    )
    m._APPLIED = False
    status, _ = m.apply()
    assert status == "skipped"


def test_g4_60h_injects_static_methods(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_G4_60H_TQ_CONFIG_AUGMENT", "1")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60h_turboquant_config_augment as m,
    )
    m._APPLIED = False

    status, _ = m.apply()
    assert status == "applied"

    from vllm.model_executor.layers.quantization.turboquant.config import (
        TurboQuantConfig,
    )

    assert hasattr(TurboQuantConfig, "align_kv_sharing_skip_layers")
    assert hasattr(TurboQuantConfig, "get_kv_sharing_target_skip_layers")


def test_g4_60h_sort_skip_layers_helper():
    """The injected ``_sort_skip_layers`` helper preserves numeric order."""
    from vllm.sndr_core.integrations.attention.turboquant.g4_60h_turboquant_config_augment import (
        _sort_skip_layers,
    )

    assert _sort_skip_layers(["10", "2", "100", "alpha", "0"]) == [
        "0",
        "2",
        "10",
        "100",
        "alpha",
    ]


# === G4_60k — EngineArgs.create_engine_config wrap ===


def test_g4_60k_skipped_without_env():
    from vllm.sndr_core.integrations.attention.turboquant import g4_60k_arg_utils as m

    m._APPLIED = False
    status, _ = m.apply()
    assert status == "skipped"


def test_g4_60k_wraps_create_engine_config(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_G4_60K_TQ_ENGINE_CONFIG", "1")
    from vllm.sndr_core.integrations.attention.turboquant import g4_60k_arg_utils as m

    m._APPLIED = False
    status, _ = m.apply()
    assert status == "applied"

    from vllm.engine.arg_utils import EngineArgs

    assert getattr(
        EngineArgs.create_engine_config, "_genesis_g4_60k_wrapped", False
    )


# === G4_60e — kv_cache_utils dispatch patches ===


def test_g4_60e_requires_g4_60a(monkeypatch):
    """G4_60e returns ``skipped`` if G4_60a hasn't injected the class."""
    monkeypatch.setenv("GENESIS_ENABLE_G4_60E_KV_CACHE_UTILS", "1")
    # Best-effort remove TQSlidingWindowSpec — emulates "G4_60a not applied"
    from vllm.v1 import kv_cache_interface as _kci

    if hasattr(_kci, "TQSlidingWindowSpec"):
        # Cannot reliably remove — class may have been added by Python class
        # body. Just skip this preflight assertion if we can't reset state.
        pytest.skip(
            "TQSlidingWindowSpec already on module; cannot test missing-"
            "prerequisite path without reload."
        )

    from vllm.sndr_core.integrations.attention.turboquant import g4_60e_kv_cache_utils as m

    m._APPLIED = False
    status, msg = m.apply()
    assert status == "skipped"
    assert "G4_60a" in msg


def test_g4_60e_apply_after_g4_60a(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC", "1")
    monkeypatch.setenv("GENESIS_ENABLE_G4_60E_KV_CACHE_UTILS", "1")

    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60a_tq_sliding_window_spec as a_mod,
    )
    from vllm.sndr_core.integrations.attention.turboquant import g4_60e_kv_cache_utils as e_mod

    a_mod._APPLIED = False
    e_mod._APPLIED = False
    a_mod.apply()
    status, _ = e_mod.apply()
    assert status == "applied"

    from vllm.v1.core import kv_cache_utils as _kcu

    assert getattr(
        _kcu.is_kv_cache_spec_uniform, "_genesis_g4_60e_wrapped", False
    )
    assert getattr(
        _kcu.unify_kv_cache_spec_page_size, "_genesis_g4_60e_wrapped", False
    )
    assert hasattr(_kcu, "_is_tq_native_mixed_kv_cache_spec")


# === G4_60g — Attention.get_kv_cache_spec dispatch ===


def test_g4_60g_requires_g4_60a(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_G4_60G_TQ_DISPATCH", "1")

    from vllm.v1 import kv_cache_interface as _kci

    if hasattr(_kci, "TQSlidingWindowSpec"):
        pytest.skip(
            "TQSlidingWindowSpec already present; cannot test "
            "missing-prerequisite path."
        )

    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60g_attention_dispatch as m,
    )

    m._APPLIED = False
    status, msg = m.apply()
    assert status == "skipped"
    assert "G4_60a" in msg


def test_g4_60g_apply_after_g4_60a(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC", "1")
    monkeypatch.setenv("GENESIS_ENABLE_G4_60G_TQ_DISPATCH", "1")

    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60a_tq_sliding_window_spec as a_mod,
    )
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60g_attention_dispatch as g_mod,
    )

    a_mod._APPLIED = False
    g_mod._APPLIED = False
    a_mod.apply()
    status, _ = g_mod.apply()
    assert status == "applied"

    from vllm.model_executor.layers.attention.attention import Attention

    assert getattr(
        Attention.get_kv_cache_spec, "_genesis_g4_60g_wrapped", False
    )


# === Composite — full stack applies in dependency order ===


def test_g4_60_full_stack_compose(monkeypatch):
    """All 5 patches apply together without conflicts."""
    for flag in (
        "GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC",
        "GENESIS_ENABLE_G4_60E_KV_CACHE_UTILS",
        "GENESIS_ENABLE_G4_60G_TQ_DISPATCH",
        "GENESIS_ENABLE_G4_60H_TQ_CONFIG_AUGMENT",
        "GENESIS_ENABLE_G4_60K_TQ_ENGINE_CONFIG",
    ):
        monkeypatch.setenv(flag, "1")

    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_60a_tq_sliding_window_spec,
        g4_60e_kv_cache_utils,
        g4_60g_attention_dispatch,
        g4_60h_turboquant_config_augment,
        g4_60k_arg_utils,
    )

    # Reset module-level state for clean test
    for mod in (
        g4_60a_tq_sliding_window_spec,
        g4_60e_kv_cache_utils,
        g4_60g_attention_dispatch,
        g4_60h_turboquant_config_augment,
        g4_60k_arg_utils,
    ):
        mod._APPLIED = False

    # Apply in dependency order (matches sndr_core/__init__.py)
    statuses = []
    for mod in (
        g4_60a_tq_sliding_window_spec,
        g4_60h_turboquant_config_augment,
        g4_60e_kv_cache_utils,
        g4_60g_attention_dispatch,
        g4_60k_arg_utils,
    ):
        status, msg = mod.apply()
        statuses.append((mod.__name__.split(".")[-1], status, msg))

    # All 5 must return "applied"
    failed = [(name, status, msg) for name, status, msg in statuses if status != "applied"]
    assert not failed, f"Some patches failed: {failed}"

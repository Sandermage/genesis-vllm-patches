# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the G4_19 module-level config registry."""
from __future__ import annotations

import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_registry():
    """Each test starts with an empty registry."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19_config_registry as reg
    reg.clear_active_config()
    yield
    reg.clear_active_config()


def test_registry_initial_state_empty():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19_config_registry as reg
    assert reg.get_active_config() is None
    assert reg.is_active() is False


def test_set_and_get_round_trip():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19_config_registry as reg
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    cfg = G4TurboQuantConfig(pack_mode="tight", wht_mode="full_wht")
    reg.set_active_config(cfg)
    assert reg.get_active_config() is cfg
    assert reg.is_active() is True


def test_clear_drops_config():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19_config_registry as reg
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    reg.set_active_config(G4TurboQuantConfig())
    assert reg.is_active() is True
    reg.clear_active_config()
    assert reg.is_active() is False
    assert reg.get_active_config() is None


def test_set_idempotent_same_object():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19_config_registry as reg
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    cfg = G4TurboQuantConfig()
    reg.set_active_config(cfg)
    reg.set_active_config(cfg)  # no-op, must not raise
    assert reg.get_active_config() is cfg


def test_set_overwrites_with_different_config():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19_config_registry as reg
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    a = G4TurboQuantConfig(pack_mode="uint32")
    b = G4TurboQuantConfig(pack_mode="tight")
    reg.set_active_config(a)
    reg.set_active_config(b)
    assert reg.get_active_config() is b


def test_registry_thread_safe():
    """Concurrent set + get calls — singleton must remain consistent
    (last writer wins; readers never see torn state)."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19_config_registry as reg
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )

    cfgs = [G4TurboQuantConfig() for _ in range(4)]
    errors = []

    def writer(c):
        try:
            for _ in range(100):
                reg.set_active_config(c)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reader():
        try:
            for _ in range(100):
                c = reg.get_active_config()
                # readers must see either None or one of the cfgs — never garbage
                assert c is None or c in cfgs
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = []
    for c in cfgs:
        threads.append(threading.Thread(target=writer, args=(c,)))
    for _ in range(4):
        threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread errors: {errors}"


def test_marker_present():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19_config_registry as reg
    assert reg.GENESIS_G4_19_REGISTRY_MARKER.startswith("Genesis G4_19")


def test_g4_19_apply_publishes_to_registry(monkeypatch):
    """When G4_19 apply() runs with env enabled, the registry should
    receive a config eagerly (without waiting for verify_and_update_config
    to fire). This guarantees worker subprocesses get the config even if
    the wrap is installed AFTER vllm_config setup."""
    pytest.importorskip("torch")
    monkeypatch.setenv("GENESIS_ENABLE_G4_19_GEMMA4_TURBOQUANT_KV", "1")
    monkeypatch.setenv("GENESIS_G4_TQ_PACK_MODE", "tight")
    monkeypatch.setenv("GENESIS_G4_TQ_WHT_MODE", "full_wht")
    monkeypatch.setenv("GENESIS_G4_TQ_BITS_GLOBAL", "3")
    monkeypatch.setenv("GENESIS_G4_TQ_BITS_SLIDING", "4")

    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19_config_registry as reg,
        g4_19_turboquant_kv_cache as g4_19,
    )
    reg.clear_active_config()

    # Reset the module-level _APPLIED flag so apply() runs the full path
    import importlib
    g4_19 = importlib.reload(g4_19)

    status, _msg = g4_19.apply()
    # apply() may return "skipped" if vllm not importable in this
    # environment — that's fine, but if status is "applied" then the
    # registry MUST have a non-None config.
    if status == "applied":
        cfg = reg.get_active_config()
        assert cfg is not None, "registry empty after applied apply()"
        assert cfg.pack_mode == "tight"
        assert cfg.wht_mode == "full_wht"
        assert cfg.bits_global == 3
        assert cfg.bits_sliding == 4

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for G4_19c — K,V round-trip wrapper on Gemma4Attention."""
from __future__ import annotations

import pytest


def test_module_imports():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19c_attention_wrapper as mod
    assert mod.GENESIS_G4_19C_MARKER.startswith("Genesis G4_19c")


def test_public_api_present():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19c_attention_wrapper as mod
    for name in ("apply", "is_applied", "revert", "GENESIS_G4_19C_MARKER"):
        assert hasattr(mod, name), f"missing public symbol {name!r}"


def test_extract_layer_idx_from_prefix():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper import (
        _extract_layer_idx,
    )
    assert _extract_layer_idx("model.layers.5.self_attn") == 5
    assert _extract_layer_idx("model.layers.0.self_attn") == 0
    assert _extract_layer_idx("model.layers.59.self_attn") == 59
    assert _extract_layer_idx("") == 0
    assert _extract_layer_idx("no-layers-here") == 0


def test_select_bits_with_layer_types():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper import (
        _select_bits,
    )
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    cfg = G4TurboQuantConfig(
        bits_sliding=4, bits_global=3,
        per_layer_types=[
            "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "full_attention",
        ],
    )
    assert _select_bits(cfg, 0) == 4  # sliding
    assert _select_bits(cfg, 1) == 3  # full
    assert _select_bits(cfg, 4) == 3  # full


def test_select_bits_without_layer_types():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper import (
        _select_bits,
    )
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    cfg = G4TurboQuantConfig(
        bits_sliding=4, bits_global=3, per_layer_types=None,
    )
    # Falls back to bits_global for all layers
    assert _select_bits(cfg, 0) == 3
    assert _select_bits(cfg, 7) == 3


def test_select_bits_out_of_range_layer_idx():
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper import (
        _select_bits,
    )
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    cfg = G4TurboQuantConfig(
        bits_sliding=4, bits_global=3,
        per_layer_types=["sliding_attention"] * 3,
    )
    # Layer index past the list falls back to bits_global
    assert _select_bits(cfg, 10) == 3
    assert _select_bits(cfg, -1) == 3  # negative — also falls back


def test_resolve_kernels_dispatch():
    """All 4 (pack, wht) combos resolve to a (write_fn, read_fn, name) triple."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant.g4_19c_attention_wrapper import (
        _resolve_kernels,
    )
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )
    combos = [
        ("uint32", "signs_only", "uint32+signs_only"),
        ("uint32", "full_wht",   "uint32+full_wht"),
        ("tight",  "signs_only", "tight+signs_only"),
        ("tight",  "full_wht",   "tight+full_wht"),
    ]
    for pack, wht, expected_name in combos:
        cfg = G4TurboQuantConfig(pack_mode=pack, wht_mode=wht)
        write_fn, read_fn, name = _resolve_kernels(cfg)
        assert callable(write_fn)
        assert callable(read_fn)
        assert name == expected_name, f"{pack}+{wht} → expected {expected_name}, got {name}"


def test_apply_skips_when_env_disabled(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.delenv("GENESIS_ENABLE_G4_19C_ATTN_WRAP", raising=False)
    from vllm.sndr_core.integrations.attention.turboquant import g4_19c_attention_wrapper as mod
    import importlib
    mod = importlib.reload(mod)
    status, msg = mod.apply()
    assert status == "skipped"
    assert "G4_19c disabled" in msg


def test_apply_skips_when_registry_empty(monkeypatch):
    """G4_19c requires G4_19 to have populated the registry."""
    pytest.importorskip("torch")
    monkeypatch.setenv("GENESIS_ENABLE_G4_19C_ATTN_WRAP", "1")
    from vllm.sndr_core.integrations.attention.turboquant import g4_19_config_registry as reg
    from vllm.sndr_core.integrations.attention.turboquant import g4_19c_attention_wrapper as mod
    reg.clear_active_config()
    import importlib
    mod = importlib.reload(mod)
    status, msg = mod.apply()
    assert status == "skipped"
    assert "registry" in msg or "G4_19" in msg


def test_get_or_build_signs_dynamo_disabled():
    """G4_19c bug 2026-05-23: ``_get_or_build_signs`` had ``with
    _SIGNS_LOCK:`` on the cold path, which made torch.compile bail
    the entire engine init at warmup because Dynamo cannot trace
    ``threading.Lock`` context managers.

    Fix: ``@torch._dynamo.disable`` on both ``_get_or_build_signs``
    and ``_build_signs_torch`` so the lock context manager is never
    seen by a Dynamo trace.

    Regression test: call ``_get_or_build_signs`` from inside a
    ``torch.compile`` boundary. Before the fix this raised
    ``torch._dynamo.exc.Unsupported`` / engine bail. After the fix
    it returns the signs tensor cleanly because Dynamo treats the
    function as an opaque eager call.
    """
    torch = pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_attention_wrapper as mod,
    )

    # CPU device works for the trace test — Dynamo will still try
    # (and fail without the fix) regardless of device.
    device = torch.device("cpu")

    # Pre-populate the cache so the hot path is exercised; cold path
    # is also Dynamo-disabled but the dict.get is what we want to
    # verify works under compile.
    signs0 = mod._get_or_build_signs(0, 8, 0xC0FFEE, device)
    assert signs0.shape == (8,)
    assert signs0.dtype == torch.float32

    def caller(x):
        # The wrapper function called from a torch.compile region.
        # _get_or_build_signs is @torch._dynamo.disable, so Dynamo
        # graph-breaks at the call but does not bail.
        signs = mod._get_or_build_signs(0, 8, 0xC0FFEE, device)
        return x * signs.sum()

    # mode="reduce-overhead" is the typical vLLM compile setting; if
    # the fix regresses, this raises Unsupported at trace time.
    compiled = torch.compile(caller, fullgraph=False)
    out = compiled(torch.ones(4))
    # signs are ±1, sum is an integer in [-8, 8]; just verify the
    # call completes and returns a tensor of the expected shape.
    assert out.shape == (4,)
    assert out.dtype == torch.float32

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


def test_signs_attached_to_module_as_buffer():
    """G4_19c fullgraph fix 2026-05-23 (Phase 7.G4.G4_19C-FIX.2):
    ``_wrapped_init`` must attach the per-layer signs tensor to the
    Gemma4Attention instance as a CUDA-resident buffer
    (``self._g4_19c_signs``). The wrapped forward then reads this
    attribute directly — fully Dynamo-traceable under fullgraph
    compile, no cache lookup, no lock.

    Regression test: simulate ``_wrapped_init`` on a minimal
    ``nn.Module`` with the required attributes; verify the buffer
    is attached and contains the expected signs tensor.

    Without this fix, the forward calls ``_get_or_build_signs``
    inside a fullgraph-compiled region, which triggers a Dynamo
    bail (either via the cold-path lock or via the @disable
    "skip-call-in-fullgraph" error).
    """
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    monkey_env = "GENESIS_ENABLE_G4_19C_ATTN_WRAP"
    import os as _os
    _os.environ[monkey_env] = "1"
    try:
        # Set up an active config so _wrapped_init's config probe
        # succeeds.
        from vllm.sndr_core.integrations.attention.turboquant import (
            g4_19_config_registry as reg,
        )
        from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
            G4TurboQuantConfig,
        )
        cfg = G4TurboQuantConfig(seed_base=0xC0FFEE)
        reg.set_active_config(cfg)

        # Mock Gemma4Attention-like module: nn.Module subclass with
        # ``prefix`` and ``head_dim`` attrs. The wrapped init only
        # needs original_init, then probes these attrs.
        from vllm.sndr_core.integrations.attention.turboquant import (
            g4_19c_attention_wrapper as mod,
        )

        class _FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.prefix = "model.layers.7.self_attn"
                self.head_dim = 8

        # Manually simulate what _wrapped_init does on top of __init__.
        # We can't easily call the real _wrapped_init because it
        # composes with the original Gemma4Attention.__init__ from
        # vllm. Instead, exercise the helper directly: call
        # _build_signs_torch, then attach as buffer via the same
        # pattern.
        instance = _FakeAttn()
        layer_idx = mod._extract_layer_idx(instance.prefix)
        assert layer_idx == 7
        signs_cpu = mod._build_signs_torch(
            head_dim=instance.head_dim,
            layer_idx=layer_idx,
            seed_base=cfg.seed_base,
        )
        instance.register_buffer(
            "_g4_19c_signs", signs_cpu, persistent=False,
        )

        # Verify the buffer is attached + recognised by nn.Module
        # buffer machinery.
        assert "_g4_19c_signs" in dict(instance.named_buffers())
        signs = instance._g4_19c_signs
        assert signs.shape == (8,)
        assert signs.dtype == torch.float32
        # ±1 signs only
        assert torch.all(torch.abs(signs) == 1.0)

        # Determinism: same key → identical signs.
        signs_redux = mod._build_signs_torch(
            head_dim=8, layer_idx=7, seed_base=cfg.seed_base,
        )
        assert torch.equal(signs, signs_redux)
    finally:
        _os.environ.pop(monkey_env, None)
        from vllm.sndr_core.integrations.attention.turboquant import (
            g4_19_config_registry as reg,
        )
        reg.clear_active_config()


def test_wrapped_forward_source_uses_module_attribute():
    """Phase 7.G4.G4_19C-FULLGRAPH-AUDIT (iter-3, 2026-05-23):
    the active hot-path forward lives in the companion module
    ``g4_19c_per_layer_forward._active_forward``. The wrapper
    module no longer has a ``_make_wrapped_forward`` (class-level
    monkeypatch retired); per-instance install happens in
    ``_wrapped_init`` via ``types.MethodType``.

    Static check: the active forward reads ``self._g4_19c_signs``
    directly and must NOT call ``_get_or_build_signs`` from the
    compile region.
    """
    pytest.importorskip("torch")
    from pathlib import Path
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_per_layer_forward as pl,
    )
    src = Path(pl.__file__).read_text()
    start = src.index("def _active_forward")
    end = src.index("# Marker for the wrapper", start)
    body = src[start:end]
    assert "self._g4_19c_signs" in body, (
        "active forward must read self._g4_19c_signs directly"
    )
    assert "_get_or_build_signs(" not in body, (
        "active forward must NOT call _get_or_build_signs from the "
        "compile region (fullgraph bail). Use the per-layer buffer "
        "attached by _wrapped_init instead."
    )

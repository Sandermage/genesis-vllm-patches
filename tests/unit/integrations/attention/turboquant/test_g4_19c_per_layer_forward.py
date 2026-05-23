# SPDX-License-Identifier: Apache-2.0
"""Tests for the iter-3 G4_19c architectural refactor — Phase
7.G4.G4_19C-FULLGRAPH-AUDIT.

Pins the contract that:

  • the active hot-path forward is in a companion file
    (g4_19c_per_layer_forward._active_forward)
  • the active forward's source is Dynamo-clean (no env reads, no
    try/except, no logging, no locks, no config lookups, no module
    mutation, no getattr-with-default)
  • make_per_layer_forward returns the unmodified original_forward
    for inactive layers and _active_forward for active layers
  • the roundtrip kernel entry is decorated with
    torch.compiler.allow_in_graph
  • setup() wires module-level _WRITE_FN / _READ_FN / _BLOCK_SIZE so
    the active forward can reach them as install-time constants
  • _decide_layer_active uses static layer properties only (registry
    state + is_kv_shared_layer + is_sliding + _FORCE_ALL_LAYERS flag)

If any of these break in a future iteration, the iter-3 architectural
invariant is broken and the wrapper risks regressing into the iter-1/
iter-2 peel-the-error pattern.
"""
from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest


# ─── Companion module shape ────────────────────────────────────────────


def test_companion_module_imports():
    """Phase 7.G4.G4_19C-FULLGRAPH-AUDIT (iter-3): the companion
    module exists and exports the contracted symbols."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_per_layer_forward as pl,
    )
    for sym in (
        "_g4_19c_roundtrip_tensor",
        "_active_forward",
        "make_per_layer_forward",
        "setup",
    ):
        assert hasattr(pl, sym), f"missing companion symbol: {sym}"


def test_setup_sets_module_constants():
    """setup() wires _WRITE_FN, _READ_FN, _BLOCK_SIZE so the active
    forward can reach them as install-time constants."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_per_layer_forward as pl,
    )

    sentinel_write = object()
    sentinel_read = object()
    pl.setup(sentinel_write, sentinel_read, 64)
    try:
        assert pl._WRITE_FN is sentinel_write
        assert pl._READ_FN is sentinel_read
        assert pl._BLOCK_SIZE == 64
    finally:
        # Restore None/default so other tests are independent.
        pl.setup(None, None, 128)


# ─── make_per_layer_forward — install-time factory ──────────────────────


def test_make_per_layer_forward_inactive_returns_original_unchanged():
    """When ``do_roundtrip=False`` the factory returns the unmodified
    original forward (eager-pass; indistinguishable from
    'G4_19c never touched this layer')."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_per_layer_forward as pl,
    )

    def _sentinel_original(self, positions, hidden_states, **kwargs):
        return "sentinel"

    result = pl.make_per_layer_forward(False, _sentinel_original)
    assert result is _sentinel_original, (
        "inactive install must return the exact original_forward "
        "callable (no wrapping, no closure)"
    )


def test_make_per_layer_forward_active_returns_active_forward():
    """When ``do_roundtrip=True`` the factory returns the shared
    ``_active_forward`` from the companion module."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_per_layer_forward as pl,
    )

    def _sentinel_original(self, positions, hidden_states, **kwargs):
        return "sentinel"

    result = pl.make_per_layer_forward(True, _sentinel_original)
    assert result is pl._active_forward, (
        "active install must return the canonical _active_forward "
        "(not a new closure per call)"
    )
    assert getattr(result, "_genesis_g4_19c_wrapped", False), (
        "_active_forward must carry the _genesis_g4_19c_wrapped marker "
        "so revert / audit code can recognise it"
    )


# ─── Dynamo-clean source invariant (regression anchor) ─────────────────


def test_active_forward_source_is_dynamo_clean():
    """The active forward body must NOT contain any Dynamo-hostile
    Python pattern. This is the single regression anchor for the
    iter-3 architectural shift — if a future fix slips an env read
    / config lookup / try-except / log call into the hot path, this
    test fails.

    The check scans the EXECUTABLE body only — the docstring is
    stripped because it intentionally enumerates the forbidden
    patterns by name as documentation. Forbidden patterns in the
    docstring are fine; forbidden patterns in actual code are not.
    """
    pytest.importorskip("torch")
    import ast
    import textwrap
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_per_layer_forward as pl,
    )

    raw = inspect.getsource(pl._active_forward)
    fn_node = ast.parse(textwrap.dedent(raw)).body[0]
    # Drop the leading docstring statement if present.
    body = fn_node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    src = "\n".join(ast.unparse(stmt) for stmt in body)

    forbidden = {
        # env reads
        "os.environ": "os.environ read in hot path — must be frozen at apply() time",
        "_env_": "any _env_* helper call — must be frozen at apply() time",
        "_env_debug": "debug env read in hot path — freeze at apply()",
        "_env_truthy_local": "env-truthy helper in hot path — freeze at apply()",
        # config / registry
        "get_active_config": "config-registry lookup in hot path — move to __init__",
        # exception flow
        "try:": "try-block in hot path — graph break under fullgraph",
        "except": "except-block in hot path — graph break under fullgraph",
        # logging
        "log.": "log call in hot path — opaque external call, freeze at apply()",
        "logger.": "log call in hot path",
        # threading
        "_SIGNS_LOCK": "threading.Lock context in hot path — unsupported by Dynamo",
        "_KERNEL_LOCK": "threading.Lock context in hot path — unsupported by Dynamo",
        # module-state mutation
        "self._genesis_g4_19c_warned": "module-state mutation in hot path — forbidden under fullgraph",
        # getattr with defaults (each attribute must be guaranteed by install)
        "getattr(self,": "getattr-with-default in hot path — make attributes mandatory in install",
        "is_sliding": "is_sliding check in hot path — must be baked into install decision",
        "is_kv_shared_layer": "kv_shared check in hot path — must be baked into install decision",
        # cold-path sign rebuild (must never reach hot path)
        "_get_or_build_signs(": "cold-path sign rebuild from hot path — sign must be a pre-attached buffer",
        "_build_signs_torch(": "sign builder in hot path — numpy + CPU work; freeze at __init__",
    }

    violations = []
    for pattern, why in forbidden.items():
        if pattern in src:
            violations.append(f"  {pattern!r}: {why}")
    assert not violations, (
        "iter-3 active forward contains Dynamo-hostile pattern(s):\n"
        + "\n".join(violations)
        + "\n\nFix one of:\n"
        + "  (a) move the decision to apply() / __init__ time (preferred)\n"
        + "  (b) wrap the helper in torch.compiler.allow_in_graph\n"
        + "  (c) install the eager-pass forward on this layer\n"
        + "DO NOT slip the pattern back into the hot path."
    )


def test_active_forward_calls_roundtrip_kernel_entry():
    """Sanity check: the active forward must reach the round-trip
    kernel entry point exactly twice (one K, one V)."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_per_layer_forward as pl,
    )
    src = inspect.getsource(pl._active_forward)
    # Two K/V calls — locate by function name.
    assert src.count("_g4_19c_roundtrip_tensor(") == 2, (
        "active forward must call _g4_19c_roundtrip_tensor exactly "
        "twice (once for K, once for V); got "
        f"{src.count('_g4_19c_roundtrip_tensor(')}"
    )


def test_active_forward_attends_to_buffer_attribute():
    """Active forward must read ``self._g4_19c_signs`` as a plain
    attribute (Dynamo-traceable tensor read), not via any lookup
    helper."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_per_layer_forward as pl,
    )
    src = inspect.getsource(pl._active_forward)
    assert "self._g4_19c_signs" in src, (
        "active forward must read self._g4_19c_signs directly"
    )


# ─── allow_in_graph decoration ──────────────────────────────────────────


def test_roundtrip_kernel_entry_is_allow_in_graph_decorated():
    """``_g4_19c_roundtrip_tensor`` must be registered with
    ``torch.compiler.allow_in_graph`` so Dynamo treats it as an
    opaque tensor op rather than tracing into it.

    Detection: ``torch._dynamo.allow_in_graph`` is the underlying
    implementation. The decorator either:
      (a) sets a sentinel attribute (older torch), OR
      (b) registers in an internal allow-list (newer torch).
    We check both pathways defensively.
    """
    pytest.importorskip("torch")
    import torch
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_per_layer_forward as pl,
    )
    fn = pl._g4_19c_roundtrip_tensor

    # Path (a): some torch versions add a sentinel.
    has_attr = (
        getattr(fn, "_torchdynamo_inline", False)
        or getattr(fn, "torchdynamo_inline", False)
        or getattr(fn, "_allow_in_graph", False)
    )

    # Path (b): newer torch keeps an internal set. Best-effort check.
    in_allowlist = False
    try:
        from torch._dynamo import trace_rules
        if hasattr(trace_rules, "_allowed_callable_ids"):
            ids = trace_rules._allowed_callable_ids.function_ids
            in_allowlist = id(fn) in ids
    except Exception:
        pass

    # If neither sentinel nor allowlist worked, at least confirm the
    # decoration syntactically appears in the source (anchor against
    # accidental removal).
    src = inspect.getsource(pl)
    has_decorator_in_source = (
        "@torch.compiler.allow_in_graph" in src
        and "def _g4_19c_roundtrip_tensor" in src
    )

    assert has_attr or in_allowlist or has_decorator_in_source, (
        "_g4_19c_roundtrip_tensor must be decorated with "
        "@torch.compiler.allow_in_graph (Dynamo otherwise treats the "
        "kernel call as a graph break under fullgraph)"
    )


# ─── Wrapper-side install logic ─────────────────────────────────────────


def test_decide_layer_active_kv_shared_returns_false():
    """_decide_layer_active must return False for KV-shared layers
    even when the registry is active."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_attention_wrapper as mod,
        g4_19_config_registry as reg,
    )
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )

    reg.set_active_config(G4TurboQuantConfig(seed_base=0xC0FFEE))
    try:
        class _FakeAttn:
            prefix = "model.layers.7.self_attn"
            head_dim = 8
            is_kv_shared_layer = True
            is_sliding = False
        assert mod._decide_layer_active(_FakeAttn()) is False
    finally:
        reg.clear_active_config()


def test_decide_layer_active_sliding_returns_false_when_not_forced():
    """Sliding-attention layers default to False (round-trip overhead
    not justified at sliding_window=1024)."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_attention_wrapper as mod,
        g4_19_config_registry as reg,
    )
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )

    reg.set_active_config(G4TurboQuantConfig(seed_base=0xC0FFEE))
    original_flag = mod._FORCE_ALL_LAYERS
    mod._FORCE_ALL_LAYERS = False
    try:
        class _FakeAttn:
            prefix = "model.layers.3.self_attn"
            head_dim = 8
            is_kv_shared_layer = False
            is_sliding = True
        assert mod._decide_layer_active(_FakeAttn()) is False
    finally:
        mod._FORCE_ALL_LAYERS = original_flag
        reg.clear_active_config()


def test_decide_layer_active_sliding_returns_true_when_forced():
    """Operator override (GENESIS_G4_19C_FORCE_ALL_LAYERS=1) overrides
    the sliding skip — frozen at apply() time."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_attention_wrapper as mod,
        g4_19_config_registry as reg,
    )
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )

    reg.set_active_config(G4TurboQuantConfig(seed_base=0xC0FFEE))
    original_flag = mod._FORCE_ALL_LAYERS
    mod._FORCE_ALL_LAYERS = True
    try:
        class _FakeAttn:
            prefix = "model.layers.3.self_attn"
            head_dim = 8
            is_kv_shared_layer = False
            is_sliding = True
        assert mod._decide_layer_active(_FakeAttn()) is True
    finally:
        mod._FORCE_ALL_LAYERS = original_flag
        reg.clear_active_config()


def test_decide_layer_active_full_attention_returns_true():
    """Full-attention non-shared layers (the production target) round-trip."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_attention_wrapper as mod,
        g4_19_config_registry as reg,
    )
    from vllm.sndr_core.integrations.attention.turboquant.kernels.g4_tq_cache import (
        G4TurboQuantConfig,
    )

    reg.set_active_config(G4TurboQuantConfig(seed_base=0xC0FFEE))
    original_flag = mod._FORCE_ALL_LAYERS
    mod._FORCE_ALL_LAYERS = False
    try:
        class _FakeAttn:
            prefix = "model.layers.0.self_attn"
            head_dim = 8
            is_kv_shared_layer = False
            is_sliding = False
        assert mod._decide_layer_active(_FakeAttn()) is True
    finally:
        mod._FORCE_ALL_LAYERS = original_flag
        reg.clear_active_config()


def test_decide_layer_active_empty_registry_returns_false():
    """No active config → no round-trip, regardless of layer type."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_attention_wrapper as mod,
        g4_19_config_registry as reg,
    )
    # Registry not populated.
    reg.clear_active_config()

    class _FakeAttn:
        prefix = "model.layers.0.self_attn"
        head_dim = 8
        is_kv_shared_layer = False
        is_sliding = False
    assert mod._decide_layer_active(_FakeAttn()) is False


# ─── apply() shape ─────────────────────────────────────────────────────


def test_apply_no_longer_class_level_forward_monkeypatches():
    """Iter-3: ``apply()`` must NOT class-level monkeypatch
    ``Gemma4Attention.forward``. The class-level forward stays the
    original; per-instance installs happen in ``_wrapped_init`` via
    ``types.MethodType``.

    Static check on the wrapper's source — confirms the
    iter-1/iter-2 ``_g4.Gemma4Attention.forward = _make_wrapped_forward(...)``
    pattern is gone.
    """
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_attention_wrapper as mod,
    )
    src = Path(mod.__file__).read_text()
    assert "Gemma4Attention.forward = " not in src, (
        "iter-3 must NOT class-level monkeypatch Gemma4Attention.forward. "
        "Per-instance install happens in _wrapped_init via types.MethodType."
    )
    assert "_make_wrapped_forward" not in src, (
        "iter-2 _make_wrapped_forward helper retired in iter-3."
    )


def test_apply_wires_companion_module_via_setup():
    """``apply()`` must wire the companion module via ``setup()`` so
    the active forward reaches its kernel constants. Static check on
    the wrapper's source."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_attention_wrapper as mod,
    )
    src = Path(mod.__file__).read_text()
    assert "_per_layer.setup(" in src, (
        "apply() must call g4_19c_per_layer_forward.setup(...) once "
        "with the resolved (write_fn, read_fn, block_size) so the "
        "active forward has its install-time constants"
    )


def test_wrapped_init_installs_per_instance_forward_via_methodtype():
    """``_wrapped_init`` must use ``types.MethodType`` to install the
    per-instance forward. Static check on the wrapper's source."""
    pytest.importorskip("torch")
    from vllm.sndr_core.integrations.attention.turboquant import (
        g4_19c_attention_wrapper as mod,
    )
    src = Path(mod.__file__).read_text()
    start = src.index("def _wrapped_init")
    end = src.index("_wrapped_init._genesis_g4_19c_init_wrapped", start)
    body = src[start:end]
    assert "types.MethodType(" in body, (
        "_wrapped_init must install the specialized forward via "
        "types.MethodType(specialized, self) — per-instance bind, "
        "no class-level monkeypatch"
    )
    assert "self.forward = " in body, (
        "_wrapped_init must assign self.forward = MethodType(...)"
    )

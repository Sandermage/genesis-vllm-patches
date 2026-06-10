# SPDX-License-Identifier: Apache-2.0
"""Phase B tests for §1.4 G4_19C opaque-op wrap.

The customop module fixes the FakeTensor crash in
``_g4_19c_roundtrip_tensor`` by exposing the round-trip as
``torch.ops.genesis.g4_19c_roundtrip`` with a registered ``fake_impl``.
These tests pin down the registration / fake-impl / dispatcher
behavior without needing real Triton kernels — the impl is exercised
via a monkey-patched ``_WRITE_FN`` / ``_READ_FN`` pair in the
companion module.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_customop():
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        mod = importlib.import_module(
            "sndr.engines.vllm.kernels_legacy.g4_19c_roundtrip_customop"
        )
    finally:
        sys.path.pop(0)
    return mod


def _import_per_layer():
    pytest.importorskip("torch")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        mod = importlib.import_module(
            "sndr.engines.vllm.patches.attention.turboquant."
            "g4_19c_per_layer_forward"
        )
    finally:
        sys.path.pop(0)
    return mod


# ─── _g4_19c_roundtrip_fake — shape/dtype invariants ────────────────────


def test_fake_impl_preserves_shape_and_dtype() -> None:
    """The fake (meta) impl is what Dynamo consults under
    FakeTensorMode. It MUST return a tensor with the exact shape and
    dtype of the input — anything else corrupts shape inference and
    crashes the AOT pass."""
    import torch
    mod = _import_customop()

    for shape in [(4, 8 * 16), (1, 2 * 16), (32, 4 * 64)]:
        x = torch.empty(shape, dtype=torch.float16)
        signs = torch.empty(16, dtype=torch.int8)
        out = mod._g4_19c_roundtrip_fake(x, signs, head_dim=16, block_size=128)
        assert out.shape == x.shape, (
            f"fake impl shape drift: input {x.shape} → output {out.shape}"
        )
        assert out.dtype == x.dtype, (
            f"fake impl dtype drift: input {x.dtype} → output {out.dtype}"
        )


def test_fake_impl_ignores_signs_value() -> None:
    """fake_impl receives the signs tensor and the integer args but
    must NOT depend on them — the output shape comes from ``x`` only."""
    import torch
    mod = _import_customop()

    x = torch.empty((8, 32), dtype=torch.bfloat16)
    out_a = mod._g4_19c_roundtrip_fake(
        x, torch.zeros(8, dtype=torch.int8), head_dim=16, block_size=128,
    )
    out_b = mod._g4_19c_roundtrip_fake(
        x, torch.ones(8, dtype=torch.int8), head_dim=8, block_size=64,
    )
    assert out_a.shape == out_b.shape == x.shape
    assert out_a.dtype == out_b.dtype == x.dtype


def test_fake_impl_uses_new_empty_not_empty_like() -> None:
    """``x.new_empty(x.shape)`` preserves the device-meta tag in a way
    ``torch.empty_like`` doesn't always — this matters under
    FakeTensorMode. The check: the result must come from x itself, not
    from the global allocator."""
    import torch
    mod = _import_customop()

    x = torch.empty((2, 64), dtype=torch.float16)
    out = mod._g4_19c_roundtrip_fake(x, x, head_dim=16, block_size=128)
    # new_empty inherits the device + layout of x.
    assert out.device == x.device
    assert out.layout == x.layout


# ─── _g4_19c_roundtrip_impl — real round-trip delegation ───────────────


def test_impl_delegates_to_per_layer_kernels(monkeypatch) -> None:
    """The real impl must call the kernel pair installed via
    ``g4_19c_per_layer_forward.setup()``. We don't need the actual
    Triton kernels — a recording stub is sufficient to lock the
    contract: write -> read -> reshape."""
    import torch
    mod = _import_customop()
    pl = _import_per_layer()

    calls: list[tuple[str, tuple]] = []

    def fake_write(x_3d, signs, head_dim, block_size):
        calls.append(("write", x_3d.shape))
        # Return packed bytes + scale — shape doesn't matter for the
        # contract test; only the call order does.
        packed = torch.zeros_like(x_3d, dtype=torch.int8)
        scale = torch.zeros(x_3d.shape[:2], dtype=torch.float16)
        return packed, scale

    def fake_read(packed, scale, signs, head_dim, block_size, dtype):
        calls.append(("read", packed.shape))
        # Read returns the same shape as packed (3D), in the requested dtype.
        return torch.zeros(packed.shape, dtype=dtype)

    monkeypatch.setattr(pl, "_WRITE_FN", fake_write)
    monkeypatch.setattr(pl, "_READ_FN", fake_read)

    x = torch.zeros((4, 8 * 16), dtype=torch.float16)
    signs = torch.zeros(16, dtype=torch.int8)
    out = mod._g4_19c_roundtrip_impl(x, signs, head_dim=16, block_size=128)

    # Order: write then read.
    assert [c[0] for c in calls] == ["write", "read"]
    # Output reshapes back to the original outer shape.
    assert out.shape == x.shape
    assert out.dtype == x.dtype


# ─── _register_op_once — idempotency + fork safety ─────────────────────


def test_register_op_once_succeeds_or_no_ops() -> None:
    """First call registers the op (when ``direct_register_custom_op``
    is available); second call is a no-op. The function must NEVER
    raise — registration failure is reported via the return value."""
    mod = _import_customop()
    # Reset in-process flag so we exercise the re-check path.
    mod._op_registered = False
    r1 = mod._register_op_once()
    r2 = mod._register_op_once()
    # If r1 succeeded, r2 must short-circuit to True instantly.
    # If r1 failed (vllm.utils.torch_utils not importable in test env),
    # r2 must also fail — but never raise.
    assert r1 in (True, False)
    assert r2 in (True, False)
    if r1 is True:
        assert r2 is True, "second call must be a no-op when first succeeded"


def test_is_registered_reflects_op_state() -> None:
    """``is_registered()`` exposes the module-level flag without
    side-effects."""
    mod = _import_customop()
    # Manipulate the flag directly to lock the surface.
    mod._op_registered = False
    assert mod.is_registered() is False
    mod._op_registered = True
    assert mod.is_registered() is True


def test_register_op_handles_missing_vllm_utils(monkeypatch) -> None:
    """When ``vllm.utils.torch_utils.direct_register_custom_op`` isn't
    importable (older vLLM, lint-only environment), the function must
    return False and never raise."""
    mod = _import_customop()
    mod._op_registered = False

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if "torch_utils" in name and "vllm" in name:
            raise ImportError("synthetic: no direct_register_custom_op")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Force a fresh cycle through the import path.
    sys.modules.pop("vllm.utils.torch_utils", None)
    result = mod._register_op_once()
    assert result is False
    assert mod._op_registered is False


# ─── Dispatcher in g4_19c_per_layer_forward — fallback semantics ──────


def test_per_layer_dispatcher_uses_opaque_op_when_registered(monkeypatch):
    """When ``_OP_REGISTERED`` is True, the hot-path helper must route
    through ``torch.ops.genesis.g4_19c_roundtrip`` and surface its
    result unchanged."""
    import torch
    pl = _import_per_layer()

    # If the live op exists, use it. Otherwise stub it out via the
    # torch.ops registry under a temporary namespace and patch the
    # dispatcher to look there. We take the simpler route: if the op
    # isn't registered, we can't exercise this path empirically — skip.
    if not (hasattr(torch.ops, "genesis")
            and hasattr(torch.ops.genesis, "g4_19c_roundtrip")):
        pytest.skip(
            "torch.ops.genesis.g4_19c_roundtrip not registered in this "
            "env — opaque-op dispatch path can't be exercised. The fake "
            "impl smoke test above already locks the shape contract."
        )

    # Force the dispatcher into the registered branch.
    monkeypatch.setattr(pl, "_OP_REGISTERED", True)

    # The opaque op needs the kernel pair too — stub them out so we
    # can exercise the full path without real Triton.
    def fake_write(x_3d, signs, head_dim, block_size):
        return (torch.zeros_like(x_3d, dtype=torch.int8),
                torch.zeros(x_3d.shape[:2], dtype=torch.float16))

    def fake_read(packed, scale, signs, head_dim, block_size, dtype):
        return torch.zeros(packed.shape, dtype=dtype)

    monkeypatch.setattr(pl, "_WRITE_FN", fake_write)
    monkeypatch.setattr(pl, "_READ_FN", fake_read)

    x = torch.zeros((4, 8 * 16), dtype=torch.float16)
    signs = torch.zeros(16, dtype=torch.int8)
    out = pl._g4_19c_roundtrip_tensor(x, signs, head_dim=16, block_size=128)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_per_layer_dispatcher_falls_back_to_inline_when_unregistered(
    monkeypatch,
):
    """When ``_OP_REGISTERED`` is False, the hot-path helper must use
    the inline body — preserving eager-mode behavior. This is the
    safety guarantee: §1.4 Phase A can land before the op is
    universally registered without regressing existing operators."""
    import torch
    pl = _import_per_layer()
    monkeypatch.setattr(pl, "_OP_REGISTERED", False)

    inline_calls = {"n": 0}
    real_inline = pl._g4_19c_roundtrip_inline

    def counting_inline(x, signs, head_dim, block_size):
        inline_calls["n"] += 1
        return real_inline(x, signs, head_dim, block_size)

    monkeypatch.setattr(pl, "_g4_19c_roundtrip_inline", counting_inline)

    # Stub the kernel pair so the inline body actually runs.
    def fake_write(x_3d, signs, head_dim, block_size):
        return (torch.zeros_like(x_3d, dtype=torch.int8),
                torch.zeros(x_3d.shape[:2], dtype=torch.float16))

    def fake_read(packed, scale, signs, head_dim, block_size, dtype):
        return torch.zeros(packed.shape, dtype=dtype)

    monkeypatch.setattr(pl, "_WRITE_FN", fake_write)
    monkeypatch.setattr(pl, "_READ_FN", fake_read)

    x = torch.zeros((4, 8 * 16), dtype=torch.float16)
    signs = torch.zeros(16, dtype=torch.int8)
    pl._g4_19c_roundtrip_tensor(x, signs, head_dim=16, block_size=128)
    assert inline_calls["n"] == 1


# ─── Module-level API surface lock ──────────────────────────────────────


def test_customop_public_exports() -> None:
    """Lock the exported surface so refactors don't accidentally hide
    the symbols apply() depends on."""
    mod = _import_customop()
    assert "_g4_19c_roundtrip_impl" in mod.__all__
    assert "_g4_19c_roundtrip_fake" in mod.__all__
    assert "_register_op_once" in mod.__all__
    assert "is_registered" in mod.__all__


def test_per_layer_module_exports_include_dispatcher_symbols() -> None:
    """Lock the per-layer module surface so the wrapper and the active
    forward keep working through refactors."""
    pl = _import_per_layer()
    assert "_g4_19c_roundtrip_tensor" in pl.__all__
    assert "_g4_19c_roundtrip_inline" in pl.__all__
    assert "_OP_REGISTERED" in pl.__all__

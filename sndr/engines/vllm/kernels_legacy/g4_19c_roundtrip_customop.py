# SPDX-License-Identifier: Apache-2.0
"""G4_19C opaque-op wrap for the TurboQuant write+read round-trip.

§1.4 Phase A of the unified plan. Fixes the
``@torch.compiler.allow_in_graph`` FakeTensor crash in the existing
``_g4_19c_roundtrip_tensor`` helper by exposing the round-trip as a
``torch.library`` custom op via the canonical
``vllm.utils.torch_utils.direct_register_custom_op`` entry point —
the same pattern PN25 (``silu_and_mul_pooled``) and P7b
(``dual_linear_parallel``) already use successfully in this tree.

Why the existing helper crashes under ``fullgraph=True``
--------------------------------------------------------
``@torch.compiler.allow_in_graph`` tells Dynamo to emit an opaque
call, but the AOT FakeTensor pass still needs an output shape. Without
a registered ``fake_impl`` (a.k.a. meta function), Dynamo falls back
to running the body against FakeTensor inputs. The body calls Triton
kernels (``_WRITE_FN`` / ``_READ_FN`` from
``g4_19c_per_layer_forward.py``), which try to read raw device pointers
from FakeTensors and crash:

    RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor,
    FunctionalTensor). If you're using torch.compile / export / fx, it
    is likely that we are erroneously tracing into a custom kernel.

The fix
-------
Register the round-trip as ``torch.ops.genesis.g4_19c_roundtrip`` with:
  * ``mutates_args=[]`` — functional op (no in-place writes).
  * ``op_func`` — the real impl that calls the Triton kernels. Runs
    only on real tensors at execution time.
  * ``fake_impl`` — pure shape/dtype propagation; never touches the
    Triton kernels. The round-trip is a quantization-noise pass that
    returns the same shape and dtype, so the fake impl is just
    ``x.new_empty(x.shape)``.

Dynamo sees the call, looks up the fake impl for shape inference,
and emits a single ``call_function`` node. The real Triton path
executes at runtime against materialised tensors — exactly what we
want.

Coexistence with the existing helper
------------------------------------
``g4_19c_per_layer_forward._g4_19c_roundtrip_tensor`` becomes a thin
dispatcher: it calls ``torch.ops.genesis.g4_19c_roundtrip`` when this
module successfully registered the op, otherwise it falls back to the
old inline body (which still works in pure eager / no-Dynamo paths).
This keeps the patch safe to land before flipping
``GENESIS_ENABLE_G4_19C_ATTN_WRAP=1`` in PROD.

Registration is fork-safe via the same pre-check pattern as PN25:
read ``torch.ops.genesis.g4_19c_roundtrip`` first; only call
``direct_register_custom_op`` when absent. Worker-spawn / re-import
cycles re-sync the local flag without raising a duplicate-name error.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch


log = logging.getLogger("genesis.kernels.g4_19c_roundtrip_customop")


_OP_NAME = "g4_19c_roundtrip"
_OP_QUALNAME = f"genesis::{_OP_NAME}"
_GENESIS_LIB: Optional["torch.library.Library"] = None
_op_registered: bool = False


def _make_genesis_lib() -> Optional["torch.library.Library"]:
    """Idempotent construction of the shared ``genesis`` namespace.

    ``FRAGMENT`` mode lets us reopen the same library across worker
    spawns and across other Genesis custom ops (PN25, P7b) that also
    register under ``genesis::`` — they share this library handle.
    """
    global _GENESIS_LIB
    if _GENESIS_LIB is not None:
        return _GENESIS_LIB
    try:
        from torch.library import Library
        _GENESIS_LIB = Library("genesis", "FRAGMENT")
        return _GENESIS_LIB
    except Exception as exc:  # pragma: no cover — torch always present here
        log.info("[G4_19C] Library construction failed: %s", exc)
        return None


def _g4_19c_roundtrip_impl(
    x: torch.Tensor,
    signs: torch.Tensor,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """Real path — the TurboQuant write+read round-trip.

    Imports the kernel pair lazily from the companion module so the
    custom-op file itself has no top-level Triton dependency. The
    kernels are populated by ``g4_19c_per_layer_forward.setup()`` once,
    at apply() time, before any layer is constructed.

    Functional contract: input ``x`` has shape ``(..., num_kv_heads *
    head_dim)``; the kernel works on ``(M, num_kv_heads, head_dim)``,
    so reshape in and back out. ``signs`` is the per-layer ±1
    random-Hadamard seed tensor (attached to the layer as
    ``self._g4_19c_signs``).
    """
    from sndr.engines.vllm.patches.attention.turboquant import (
        g4_19c_per_layer_forward as _pl,
    )
    orig_shape = x.shape
    num_kv_heads = orig_shape[-1] // head_dim
    M = x.numel() // (num_kv_heads * head_dim)
    x_3d = x.contiguous().view(M, num_kv_heads, head_dim)
    packed, scale = _pl._WRITE_FN(x_3d, signs, head_dim, block_size)
    x_rt = _pl._READ_FN(packed, scale, signs, head_dim, block_size, x.dtype)
    return x_rt.view(orig_shape)


def _g4_19c_roundtrip_fake(
    x: torch.Tensor,
    signs: torch.Tensor,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """Shape / dtype inference under FakeTensorMode.

    MUST NOT call the Triton kernels — that's the whole reason this
    custom op exists. The round-trip is a noise-injection pass that
    preserves both shape and dtype, so ``x.new_empty(x.shape)`` is the
    correct fake. Using ``new_empty`` (not ``empty_like``) keeps the
    same memory-format and meta-device tagging as the input.

    The ``signs`` / ``head_dim`` / ``block_size`` arguments are
    intentionally ignored here — they only affect the actual numeric
    transformation, not the output's shape or dtype.
    """
    return x.new_empty(x.shape)


def _register_op_once() -> bool:
    """Idempotent registration of ``torch.ops.genesis.g4_19c_roundtrip``.

    Returns True on success or if the op was already registered (e.g.
    by a parent process before fork). Returns False only on a real
    failure where ``direct_register_custom_op`` raises something other
    than a duplicate-name error.

    The pre-check via ``hasattr(torch.ops.genesis, ...)`` mirrors the
    fork-safety guard PN25 retained from v7.65 — survives worker
    spawn even though the in-process flag does not.
    """
    global _op_registered
    if _op_registered:
        return True
    try:
        if hasattr(torch.ops, "genesis") and hasattr(
            torch.ops.genesis, _OP_NAME,
        ):
            _op_registered = True
            log.info(
                "[G4_19C] op %s already globally registered — synced "
                "local flag (worker-spawn path)", _OP_QUALNAME,
            )
            return True
    except (AttributeError, RuntimeError):
        # torch.ops.genesis namespace may not exist yet — fall through.
        pass

    try:
        from vllm.utils.torch_utils import direct_register_custom_op
    except ImportError as exc:
        log.info(
            "[G4_19C] vllm.utils.torch_utils.direct_register_custom_op "
            "not available (%s) — eager fallback will be used", exc,
        )
        return False

    lib = _make_genesis_lib()
    if lib is None:
        return False

    try:
        direct_register_custom_op(
            op_name=_OP_NAME,
            op_func=_g4_19c_roundtrip_impl,
            mutates_args=[],
            fake_impl=_g4_19c_roundtrip_fake,
            target_lib=lib,
        )
        _op_registered = True
        log.info(
            "[G4_19C] registered %s via direct_register_custom_op",
            _OP_QUALNAME,
        )
        return True
    except RuntimeError as exc:
        # Most likely a concurrent fork-safe registration. Sync our
        # flag and accept the global state instead of raising.
        if "already" in str(exc).lower() or "registered" in str(exc).lower():
            _op_registered = True
            log.info(
                "[G4_19C] race-resolved duplicate registration of %s — "
                "synced local flag", _OP_QUALNAME,
            )
            return True
        log.warning(
            "[G4_19C] direct_register_custom_op failed: %s — eager "
            "fallback will be used", exc,
        )
        return False
    except Exception as exc:
        log.warning(
            "[G4_19C] direct_register_custom_op failed (%s: %s) — eager "
            "fallback will be used", type(exc).__name__, exc,
        )
        return False


def is_registered() -> bool:
    """Operator-facing helper — True if the opaque op is callable."""
    return _op_registered


__all__ = [
    "_g4_19c_roundtrip_impl",
    "_g4_19c_roundtrip_fake",
    "_register_op_once",
    "is_registered",
]

# SPDX-License-Identifier: Apache-2.0
"""G4_19c per-layer specialized forward — Phase 7.G4.G4_19C-FULLGRAPH-AUDIT.

Companion to ``g4_19c_attention_wrapper.py``. Hosts the two specialized
forward closures and the install-time factory that returns the right
one per layer.

Architectural shift (vs iter-1 / iter-2):
  • Every Python-side decision — env reads, config lookups, kernel
    resolution, sliding-skip — is made ONCE at apply() / __init__ time
    and baked in as install-time constants.
  • The HOT-PATH forward body is pure tensor ops + one
    ``allow_in_graph``-decorated kernel entry. No env reads, no Python
    config lookups, no try/except, no logging, no module mutation.
  • Each ``Gemma4Attention`` instance gets EITHER the unmodified
    original_forward (eager-pass, when the layer is inactive) OR
    ``_active_forward`` (straight-line round-trip). Selection is
    install-time via ``types.MethodType(...)`` in the wrapper's
    ``_wrapped_init``. Dynamo compiles each instance's bound forward
    independently — no per-call Python branches.

This file is intentionally minimal — its source is statically scanned
by the regression tests to assert the hot path stays Dynamo-clean.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


__all__ = [
    "_g4_19c_roundtrip_tensor",
    "_active_forward",
    "make_per_layer_forward",
    "setup",
    "_WRITE_FN",
    "_READ_FN",
    "_BLOCK_SIZE",
]


# ─── Install-time constants ─────────────────────────────────────────────
# Populated by ``setup()`` which is called from
# ``g4_19c_attention_wrapper.apply()`` ONCE, before any forward runs.
# Never re-bound at request time — the active forward closes over
# these as module-level constants.

_WRITE_FN: Optional[Callable] = None
_READ_FN: Optional[Callable] = None
_BLOCK_SIZE: int = 128


def setup(write_fn: Callable, read_fn: Callable, block_size: int) -> None:
    """Wire the kernel pair + block-size that ``_active_forward`` will
    use. Called by ``apply()`` after the registry config is resolved.

    Replacing these values after layers are constructed is supported
    in principle (the active forward reads the module globals at call
    time, not at install time), but in practice apply() runs once per
    process and these stay constant.
    """
    global _WRITE_FN, _READ_FN, _BLOCK_SIZE
    _WRITE_FN = write_fn
    _READ_FN = read_fn
    _BLOCK_SIZE = block_size


# ─── allow_in_graph kernel entry ────────────────────────────────────────
# Single Dynamo-visible entry point for the round-trip. Decorated with
# ``torch.compiler.allow_in_graph`` so the fullgraph tracer treats this
# as an opaque Tensor-returning op: it traces the call site, captures
# the input tensors, and emits a call. Actual execution at run time
# uses module-level _WRITE_FN / _READ_FN — eager-equivalent semantics
# inside a compiled graph, no graph break.

@torch.compiler.allow_in_graph
def _g4_19c_roundtrip_tensor(
    x: torch.Tensor,
    signs: torch.Tensor,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """Compress + decompress ``x`` once via the active G4-TurboQuant
    kernel pair. Returns a tensor of the same shape and dtype as ``x``
    with the quantization noise applied.

    ``x`` is expected to be ``(..., num_kv_heads * head_dim)``; the
    kernel works on ``(M, num_kv_heads, head_dim)`` so we reshape in
    and back out. ``signs`` is the per-layer ±1 random-Hadamard seed
    tensor attached to the module as ``self._g4_19c_signs``.

    This function is the ONLY Dynamo-visible Python helper called from
    the active forward. ``allow_in_graph`` declares to the tracer
    that this is an opaque Tensor-in, Tensor-out call: do not inline,
    do not graph-break, emit a call.
    """
    orig_shape = x.shape
    num_kv_heads = orig_shape[-1] // head_dim
    M = x.numel() // (num_kv_heads * head_dim)
    x_3d = x.contiguous().view(M, num_kv_heads, head_dim)
    packed, scale = _WRITE_FN(x_3d, signs, head_dim, block_size)
    x_rt = _READ_FN(packed, scale, signs, head_dim, block_size, x.dtype)
    return x_rt.view(orig_shape)


# ─── Specialized active forward ─────────────────────────────────────────
# Hot path body — pure tensor ops only. Install decision at
# __init__ time has already excluded the layers that should NOT
# round-trip (KV-shared / sliding-skip / config-inactive / signs
# pre-build failed). Those layers carry the unmodified original
# forward instead.
#
# The regression test ``test_active_forward_source_is_dynamo_clean``
# in test_g4_19c_per_layer_forward.py scans this function's source
# for env reads, try/except, logging, locks, and config lookups.
# Any reintroduction of those patterns fails the test.

def _active_forward(self, positions, hidden_states, **kwargs):
    """G4_19c active hot path. Replicates ``Gemma4Attention.forward``
    on this vLLM pin except for the K, V round-trip insertion.

    Tracer-visible operations only:
      • Module attribute reads (head_dim, num_heads, num_kv_heads, etc.)
      • Tensor.split / unflatten / flatten
      • Sub-module forward calls (qkv_proj, q_norm, k_norm, v_norm,
        rotary_emb, attn, o_proj)
      • One ``_g4_19c_roundtrip_tensor`` call per K and V

    Tracer-invisible (i.e. ABSENT) operations:
      • os.environ reads
      • get_active_config() / any Python registry lookup
      • try/except
      • logging
      • threading locks
      • module attribute writes (no ``self._foo = ...``)
      • getattr with defaults (each attribute is guaranteed by install
        decision in _wrapped_init)
    """
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split(
        [self.q_size, self.kv_size, self.kv_size], dim=-1,
    )
    q = q.unflatten(-1, (self.num_heads, self.head_dim))
    q = self.q_norm(q)
    q = q.flatten(-2, -1)

    k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
    k = self.k_norm(k)
    k = k.flatten(-2, -1)
    q, k = self.rotary_emb(positions, q, k)
    v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
    v = self.v_norm(v)
    v = v.flatten(-2, -1)

    k = _g4_19c_roundtrip_tensor(
        k, self._g4_19c_signs, self.head_dim, _BLOCK_SIZE,
    )
    v = _g4_19c_roundtrip_tensor(
        v, self._g4_19c_signs, self.head_dim, _BLOCK_SIZE,
    )

    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output


# Marker for the wrapper to recognise an already-installed active forward.
_active_forward._genesis_g4_19c_wrapped = True  # type: ignore[attr-defined]


# ─── Install-time factory ───────────────────────────────────────────────


def make_per_layer_forward(
    do_roundtrip: bool,
    original_forward: Callable,
) -> Callable:
    """Return the callable to bind on a specific Gemma4Attention
    instance via ``types.MethodType``.

    The decision is install-time. Dynamo compiles each instance's
    bound forward independently, so a layer that gets
    ``original_forward`` here is indistinguishable from a layer the
    G4_19c apply chain never touched.

    Args:
      do_roundtrip: True iff this layer should round-trip K, V at
        every forward. Decided by ``_decide_layer_active`` in the
        wrapper's ``_wrapped_init`` from static layer properties
        only (registry state, is_kv_shared_layer, is_sliding,
        sign pre-build success).
      original_forward: ``Gemma4Attention.forward`` BEFORE any
        Genesis monkeypatch, captured at apply() time.

    Returns:
      ``original_forward`` when ``do_roundtrip`` is False (eager-pass);
      ``_active_forward`` when ``do_roundtrip`` is True.
    """
    if not do_roundtrip:
        return original_forward
    return _active_forward

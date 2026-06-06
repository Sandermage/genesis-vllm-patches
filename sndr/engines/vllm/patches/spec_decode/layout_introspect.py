# SPDX-License-Identifier: Apache-2.0
"""LayoutIntrospect — backend-aware KV cache layout query helpers.

Library code. Does NOT monkey-patch anything.

Central place for all Genesis patches that need to reason about KV
cache layout. Replaces hand-rolled shape inspection scattered across
several modules (kv_contract._classify_layout, g4_74's shape[0]==2
discrimination, pn130's hardcoded warmup shape, etc.).

Three primary calls:

  ``block_dim_of(backend_cls, ...)`` — return the index of the
  num_blocks dimension in the backend's declared cache shape. Uses
  upstream's explicit ``get_kv_cache_block_dim`` introspection API
  (added in vllm PR #42095, May 2026) when available, falls back to
  shape-inspection via ``get_kv_cache_shape`` with a unique sentinel
  num_blocks value.

  ``classify_layout(cache_tensor)`` — inspect a live tensor and assign
  a ``Layout`` enum verdict. Distinguishes HND vs NHD (the upstream
  pre-#42095 vs post-#42095 5-dim shapes), TQ_PACKED_4D (TurboQuant's
  combined K+V slot layout), MAMBA_STATE (3-dim variable-shape Mamba
  state), and UNKNOWN.

  ``build_warmup_kv_cache(backend_cls, ...)`` — allocate a small zero
  KV cache matching whatever shape the backend declares. Use for
  compile warmup, kernel JIT pre-compilation, smoke tests.

Why this exists:

  The two upstream PRs that landed in vllm bf610c2f5 → 626fa9bb
  (#42095 KV layout, #43543 attn groups) made it impossible to safely
  hard-code KV cache shape assumptions in Genesis patches. PR #42095
  introduced a per-backend introspection API
  (``get_kv_cache_block_dim``); PR #43543 changed metadata-key
  construction. Genesis patches that branched on ``shape[0]==2`` vs
  ``shape[1]==2`` (e.g. G4_74) became fragile across pins. This
  library captures the polymorphic-detection logic ONCE so future
  upstream changes only need updating here, not across N callsites.

Design choices:

  * Fail-soft on missing introspection. The fallback chain
    (explicit API → shape inspection → conservative default) means a
    caller never sees an exception just from querying layout.
  * No torch import at module top level. ``classify_layout`` lazily
    imports torch only when called; ``build_warmup_kv_cache``
    requires it explicitly. Lets the family-contract tests collect
    this module under torch-less CI.
  * No vllm import at top level either. All backend introspection is
    duck-typed via ``hasattr`` so this module loads cleanly even on
    pins where the relevant classes were renamed/moved.

Backed by tests in
``tests/unit/integrations/spec_decode/test_layout_introspect.py``.

Provenance: K.1.R.R.3 (2026-05-29). Author: Sander, Odessa.
"""
from __future__ import annotations

import enum
import logging
from typing import Any

log = logging.getLogger("genesis.spec_decode.layout_introspect")


# ----------------------- Layout verdicts -----------------------

class Layout(str, enum.Enum):
    """Layout family of a KV cache tensor.

    Ordering (concrete → abstract):

      HND:
        5-D, ``(2, num_blocks, block_size, num_kv_heads, head_dim)``.
        Upstream pre-#42095 default. K and V are split on dim 0 via
        ``kv_cache.unbind(0)``.

      NHD:
        5-D, ``(num_blocks, 2, block_size, num_kv_heads, head_dim)``.
        Upstream post-#42095 default. K and V are split on dim 1 via
        ``kv_cache.unbind(1)``. Lucas Wilkinson, May 2026.

      TQ_PACKED_4D:
        4-D, ``(num_blocks, block_size, num_kv_heads,
        slot_size_aligned)``, ``dtype=uint8``. TurboQuant overlay
        packs K+V into a single byte slot per (block, position, head).
        Asymmetric K8V4 / K3V4 quantization lives here. No leading 2
        dim because K and V share a byte slot.

      MAMBA_STATE:
        3-D, ``(num_blocks, *spec_shape)``. Mamba/SSM state tensor.
        First dim is num_blocks; downstream dims encode conv state,
        temporal state, SSM state. Not subject to K/V split.

      UNKNOWN:
        Empty tensor, scalar, or shape that doesn't match any of the
        above. Callers should treat as "do not assume" and skip
        layout-dependent paths.
    """
    HND = "HND"
    NHD = "NHD"
    TQ_PACKED_4D = "TQ_PACKED_4D"
    MAMBA_STATE = "MAMBA_STATE"
    UNKNOWN = "UNKNOWN"


# Sentinel num_blocks value used for the shape-inspection fallback.
# Picked to be (a) unique enough to never collide with a real
# block_size or num_kv_heads, (b) the same magic number upstream uses
# in the base ``AttentionBackend.get_kv_cache_block_dim`` default so
# behavior matches when our fallback fires on an old pin.
_SENTINEL_NUM_BLOCKS: int = 1234567


# ----------------------- block_dim_of -----------------------

def block_dim_of(
    backend_cls: Any,
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "auto",
) -> int:
    """Return the index of the num_blocks dimension in the backend's
    KV cache shape.

    Resolution order:

      1. Backend's explicit ``get_kv_cache_block_dim`` classmethod
         (vllm PR #42095, May 2026). This is the canonical source of
         truth on post-#42095 pins.
      2. Backend's ``get_kv_cache_shape`` called with a unique
         sentinel num_blocks; the position of the sentinel in the
         returned tuple is the block dim. Matches the algorithm used
         by the upstream base class default.
      3. Conservative fallback: 1 (the pre-#42095 default; assumes
         leading-2 HND layout). Callers using this fallback should
         have already logged that the introspection failed.

    Returns:
        Integer index ``0`` or ``1`` (or higher for exotic layouts).
        ``0`` = num-blocks-first (post-#42095 NHD, TurboQuant); ``1``
        = K/V-first (pre-#42095 HND).
    """
    # 1. Explicit upstream introspection (preferred)
    if hasattr(backend_cls, "get_kv_cache_block_dim"):
        try:
            return int(backend_cls.get_kv_cache_block_dim(
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str=cache_dtype_str,
            ))
        except Exception as e:
            log.debug(
                "[layout_introspect] %s.get_kv_cache_block_dim raised %r; "
                "falling back to shape-inspection",
                getattr(backend_cls, "__name__", "<backend>"), e,
            )

    # 2. Shape-inspection sentinel trick
    if hasattr(backend_cls, "get_kv_cache_shape"):
        try:
            shape = backend_cls.get_kv_cache_shape(
                _SENTINEL_NUM_BLOCKS,
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str=cache_dtype_str,
            )
            return list(shape).index(_SENTINEL_NUM_BLOCKS)
        except Exception as e:
            log.debug(
                "[layout_introspect] %s.get_kv_cache_shape introspection "
                "raised %r; falling back to default block_dim=1",
                getattr(backend_cls, "__name__", "<backend>"), e,
            )

    # 3. Conservative fallback
    return 1


# ----------------------- classify_layout -----------------------

def classify_layout(cache_tensor: Any) -> Layout:
    """Classify a live KV cache tensor by shape + dtype.

    Pure shape/dtype inspection, no backend introspection. Use when
    you have a tensor in hand and need to know how to read it.

    The check sequence (concrete-before-generic):

      * 4-D + uint8 + last dim > 2 → ``TQ_PACKED_4D``.
        TurboQuant slot is uint8 and the slot_size dim is the last
        and always > 2 (smallest known slot is 24 bytes).

      * 5-D + ``shape[0] == 2`` → ``HND`` (pre-#42095 layout).

      * 5-D + ``shape[1] == 2`` → ``NHD`` (post-#42095 layout).

      * 3-D → ``MAMBA_STATE`` (variable per state-spec).

      * Otherwise → ``UNKNOWN``.

    Disambiguation note: when both ``shape[0] == 2`` and
    ``shape[1] == 2`` (i.e. a 2-block warmup cache), the tensor alone
    cannot distinguish HND from NHD. This routine picks HND, matching
    pre-#42095 convention. Callers with a backend handle should use
    ``block_dim_of(backend_cls, ...)`` instead, which is unambiguous
    because it uses a unique sentinel num_blocks value (1234567).

    Returns:
        ``Layout`` enum value. Never raises.
    """
    if cache_tensor is None:
        return Layout.UNKNOWN

    try:
        shape = tuple(cache_tensor.shape)
        dtype = cache_tensor.dtype
    except Exception:
        return Layout.UNKNOWN

    if len(shape) < 2:
        return Layout.UNKNOWN

    # TQ_PACKED_4D: 4-D uint8 with slot dim last.
    # Defer torch import until we actually need to compare dtype.
    if len(shape) == 4 and int(shape[3]) > 2:
        try:
            import torch
            if dtype == torch.uint8:
                return Layout.TQ_PACKED_4D
        except ImportError:
            # No torch available; fall through. The 4D+slot>2 shape
            # still strongly suggests TQ but we can't confirm dtype.
            pass

    # HND / NHD: 5-D with a 2-axis somewhere.
    if len(shape) == 5:
        if int(shape[0]) == 2:
            return Layout.HND
        if int(shape[1]) == 2:
            return Layout.NHD

    # MAMBA_STATE: 3-D, e.g. (num_blocks, conv_dim, conv_size).
    if len(shape) == 3:
        return Layout.MAMBA_STATE

    return Layout.UNKNOWN


# ----------------------- block_dim_from_tensor -----------------------

def block_dim_from_tensor(cache_tensor: Any) -> int | None:
    """Derive block dim from a live tensor's classified layout.

    Convenience wrapper around ``classify_layout``:

      * ``HND`` → 1 (num_blocks is dim 1)
      * ``NHD`` → 0 (num_blocks is dim 0)
      * ``TQ_PACKED_4D`` → 0 (num_blocks is dim 0)
      * ``MAMBA_STATE`` → 0 (num_blocks is dim 0)
      * ``UNKNOWN`` → ``None``

    Use when the caller has a tensor but not a backend class handle.
    """
    layout = classify_layout(cache_tensor)
    if layout == Layout.HND:
        return 1
    if layout in (Layout.NHD, Layout.TQ_PACKED_4D, Layout.MAMBA_STATE):
        return 0
    return None


# ----------------------- build_warmup_kv_cache -----------------------

def build_warmup_kv_cache(
    backend_cls: Any,
    *,
    num_blocks: int = 2,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "auto",
    dtype: Any = None,
    device: str = "cuda",
) -> Any:
    """Allocate a zero KV cache tensor matching the backend's declared
    shape.

    Use for compile warmup, kernel JIT pre-compilation, smoke
    benches. Lets a Genesis patch construct a warmup cache without
    hardcoding shape — if upstream changes the shape (e.g. layout
    flip in PR #42095), the warmup tensor follows automatically.

    Args:
        backend_cls: Class with a ``get_kv_cache_shape(num_blocks,
            block_size, num_kv_heads, head_size, cache_dtype_str)``
            classmethod. Both upstream FlashAttention and the Genesis
            TurboQuant overlay expose this.
        num_blocks: Tiny number for warmup. Default 2 — enough to
            exercise the block-table indexing without burning memory.
        block_size: Tokens per block. Match the model's configured
            value (typically 16 for vllm v1).
        num_kv_heads: Per-TP-rank num_kv_heads.
        head_size: Per-head dim. For TurboQuant this is the model's
            real head_dim, NOT the padded slot size — the backend's
            ``get_kv_cache_shape`` translates internally.
        cache_dtype_str: ``"auto"`` for native, or a TurboQuant variant
            (``"turboquant_k8v4"``, ``"turboquant_4bit_nc"``, etc.).
            Passed through to ``get_kv_cache_shape``.
        dtype: torch dtype to allocate with. When ``None``, auto-pick:
            ``torch.uint8`` for any TurboQuant variant (packed bytes),
            ``torch.bfloat16`` otherwise.
        device: Allocation device. Default ``"cuda"``.

    Returns:
        A ``torch.Tensor`` of zeros with the backend's declared shape.

    Raises:
        RuntimeError: backend has no ``get_kv_cache_shape`` method.
        ImportError: torch not importable.
    """
    if not hasattr(backend_cls, "get_kv_cache_shape"):
        raise RuntimeError(
            f"{getattr(backend_cls, '__name__', '<backend>')} has no "
            "get_kv_cache_shape method; cannot build layout-aware "
            "warmup tensor."
        )

    import torch

    shape = backend_cls.get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        cache_dtype_str=cache_dtype_str,
    )

    if dtype is None:
        if str(cache_dtype_str).startswith("turboquant_"):
            dtype = torch.uint8
        else:
            dtype = torch.bfloat16

    return torch.zeros(shape, dtype=dtype, device=device)


__all__ = [
    "Layout",
    "block_dim_of",
    "classify_layout",
    "block_dim_from_tensor",
    "build_warmup_kv_cache",
]

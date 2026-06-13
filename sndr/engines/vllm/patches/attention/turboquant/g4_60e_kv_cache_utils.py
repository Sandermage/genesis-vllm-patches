# SPDX-License-Identifier: Apache-2.0
"""G4_60e — patch ``vllm.v1.core.kv_cache_utils`` for TQ/native mixed layouts.

================================================================
PROBLEM
================================================================

After G4_60a + G4_60g, Gemma 4 with ``--kv-cache-dtype turboquant_*`` can
produce a heterogeneous ``dict[str, KVCacheSpec]`` mixing:

  * ``TQFullAttentionSpec``   (full layers, head_dim=512, TQ-compressed)
  * ``TQSlidingWindowSpec``   (sliding layers, head_dim=256, TQ-compressed)
  * ``FullAttentionSpec``     (skip-layers, uncompressed bf16)
  * ``SlidingWindowSpec``     (skip-layers, uncompressed bf16)

The dev371 ``kv_cache_utils.py`` cannot route such mixes:

  1. ``is_kv_cache_spec_uniform`` returns True if first spec's ``merge()``
     succeeds — but with PR #42637's tightened TQ ``merge()`` (G4_60a),
     mixed lists raise ``AssertionError`` only when actually merged. The
     pre-check at line 854 needs to detect mixed types *before* merge to
     route to the hybrid path.

  2. ``unify_kv_cache_spec_page_size`` resizes via ``block_size`` ratio
     only — TQ specs cannot be resized this way (their ``real_page_size_
     bytes`` is ``block_size × num_kv_heads × tq_slot_size``, not the
     standard ``head_size × dtype`` formula).

  3. No predicate exists to recognize "mixed TQ + native" as a valid
     supported layout vs unsupported chimera (MLA mixed with TQ, etc.).

  4. The top-level dispatch in ``get_kv_cache_groups`` has no branch to
     route mixed TQ+native through the hybrid manager.

Additionally (2026-06-11 reconciliation, pin g303916e93):

  5. ``MambaSpec`` pages are state-shape determined and do NOT scale
     with ``block_size``. The pin's unify scales block_size anyway and
     dies on a bare ``AssertionError`` (#43626 class) the moment a layer
     with a larger page — e.g. a dense bf16 drafter on a hybrid GDN
     model (our 35B/27B Qwen3.6 stacks) — joins the spec dict.
     Upstream fix: OPEN vllm#45207.

  6. Non-divisible attention pages (e.g. 192-dim target KV heads vs
     128-dim DFlash drafter heads = 3:2 page ratio) hit the
     ``NotImplementedError`` boot-fail even though the physical page can
     simply be padded. Upstream fix: OPEN vllm#45181.

  7. The pin's worker-side padded-page reshape (``gpu/attn_utils.py``
     ``_reshape_kv_cache`` and legacy ``GPUModelRunner._reshape_kv_
     cache_tensors``) assumes physical dim 0 is the block index and
     leaves the K/V-dim stride at its contiguous value — wrong block
     dim is strided for ``(2, num_blocks, ...)`` layouts and the V half
     of a padded page can be read from the padding tail. Upstream fix:
     OPEN vllm#45181 ``_reshape_attention_kv_cache``.

================================================================
FIX (PR #42637 cherry-pick + #45207/#45181 fold)
================================================================

Five monkey-patches and one new function:

  1. **Replace** ``is_kv_cache_spec_uniform`` — add early-out for
     ``TQFullAttentionSpec×FullAttentionSpec`` and
     ``TQSlidingWindowSpec×SlidingWindowSpec`` co-presence (returns False).

  2. **Replace** ``unify_kv_cache_spec_page_size`` — reconciled ladder:
       a. page already at max          → keep;
       b. TQ spec                       → pad via ``page_size_padded``
          (TQ slot layout is kernel-bound; block-size resize forbidden);
       c. ``MambaSpec``                 → pad (vllm#45207 — page is
          state-shape determined, block-size resize is a silent no-op);
       d. divisible attention page      → block-size scaling (upstream
          original path, with #45207's actionable assert message);
       e. non-divisible ``AttentionSpec`` → pad (vllm#45181 generic
          fallback — supersedes the old G4_60e non-TQ passthrough);
       f. anything else                 → actionable NotImplementedError
          naming the layer and both page sizes.

  3. **Inject** ``_is_tq_native_mixed_kv_cache_spec`` predicate that
     identifies the supported mixed layout (TQ + plain Full/Sliding only;
     not MLA, Mamba, ChunkedLocal).

  4. **Wrap** ``get_kv_cache_groups`` to add the mixed-route branch
     before the existing ``is_kv_cache_spec_uniform`` check. When the
     predicate matches, route through
     ``unify_kv_cache_spec_page_size + _get_kv_cache_groups_uniform_
     page_size``.

  5. **Wrap** modular-runner ``vllm.v1.worker.gpu.attn_utils.
     _reshape_kv_cache`` with a padded-view post-correction that
     rebuilds every padded-attention layer view via the vendored
     ``_reshape_attention_kv_cache`` (vllm#45181 stride hardening:
     num_blocks dim detection + explicit K/V-dim stride). The helper is
     injected onto the module under its upstream name so a future pin
     that already carries #45181 is detected and both reshape patches
     self-skip (retirement hook).

  6. **Wrap** legacy-runner ``GPUModelRunner._reshape_kv_cache_tensors``
     with the same post-correction.

================================================================
DEPENDENCIES
================================================================

  * **G4_60a** required (defines ``TQSlidingWindowSpec``). The predicate
    references the symbol from ``vllm.v1.kv_cache_interface``.

  * Compatible with **G4_60g** but not strictly required — G4_60e is a
    no-op when no TQ specs are present in the spec dict.

================================================================
SCOPE
================================================================

Active only when ``GENESIS_ENABLE_G4_60E_KV_CACHE_UTILS=1``. Touches
4 symbols on ``vllm.v1.core.kv_cache_utils`` plus the two worker-side
reshape entry points (boot-time only, not hot-path). The mixed-route
branch is only taken when the predicate matches (TQ + native
co-present); all-TQ and pure-native flows go through unchanged paths.
The reshape post-correction only rebuilds views for layers whose spec
carries ``page_size_padded`` — unpadded models see the original result
object unchanged.

Known limitation (matches upstream #45181): padded pages on
``(2, num_blocks, ...)`` layouts combined with a hybrid Mamba model are
left on the original view (the in-place hybrid restride already ran);
a warning is logged. Not reachable on our backends (FlashAttention,
Triton, TQ — all block-dim-0 layouts).

================================================================
REFERENCES
================================================================

  * Upstream PR: https://github.com/vllm-project/vllm/pull/42637
  * Upstream lines (PR #42637 HEAD ``fdeb14981``):
    ``vllm/v1/core/kv_cache_utils.py``
      - ``is_kv_cache_spec_uniform``        lines 854-881
      - ``unify_kv_cache_spec_page_size``   lines 1019-1063
      - ``_is_tq_native_mixed_kv_cache_spec`` lines 1484-1512
      - dispatch update                     lines 1696-1706
  * Folded 2026-06-11 (both OPEN, NOT in pin g303916e93 — verified
    against /private/tmp/candidate_pin_current/vllm):
      - https://github.com/vllm-project/vllm/pull/45207 (MambaSpec
        padding elif in ``unify_kv_cache_spec_page_size``)
      - https://github.com/vllm-project/vllm/pull/45181 (generic
        AttentionSpec padding fallback + ``_reshape_attention_kv_cache``
        stride hardening in ``vllm/v1/worker/gpu/attn_utils.py`` and
        ``vllm/v1/worker/gpu_model_runner.py``)
  * Companion patches: G4_60a (TQSlidingWindowSpec), G4_60h (slot_size).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import functools
import logging
import os

log = logging.getLogger("genesis.turboquant.g4_60e_kv_cache_utils")

GENESIS_G4_60E_MARKER = (
    "Genesis G4_60e kv_cache_utils.py patches: "
    "is_kv_cache_spec_uniform + unify_kv_cache_spec_page_size + "
    "_is_tq_native_mixed_kv_cache_spec + get_kv_cache_groups dispatch "
    "(PR #42637 cherry-pick) + Mamba page padding (vllm#45207 fold) + "
    "generic AttentionSpec padding fallback and "
    "_reshape_attention_kv_cache stride hardening (vllm#45181 fold)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60E_KV_CACHE_UTILS"
_APPLIED = False
_ORIGINAL_IS_UNIFORM = None
_ORIGINAL_UNIFY = None
_ORIGINAL_GET_GROUPS = None
_ORIGINAL_RESHAPE_MODULAR = None
_ORIGINAL_RESHAPE_LEGACY = None
_PATCHED_ATTN_UTILS_MOD = None
_PATCHED_RUNNER_CLS = None


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _is_tq_native_mixed_kv_cache_spec(kv_cache_spec):
    """Return whether the spec mixes TurboQuant and native attention storage.

    Cherry-picked from PR #42637 lines 1484-1512 verbatim.

    Keep this predicate intentionally narrow. The special mixed path is
    only for TQ attention specs plus exact native full/sliding attention
    specs. All-TQ, pure-native, MLA, Mamba, and chunked-local layouts
    must keep their existing routing.
    """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        SlidingWindowSpec,
        TQFullAttentionSpec,
        TQSlidingWindowSpec,
    )

    supported_spec_types = {
        FullAttentionSpec,
        SlidingWindowSpec,
        TQFullAttentionSpec,
        TQSlidingWindowSpec,
    }
    specs = list(kv_cache_spec.values())
    has_tq = any(
        type(spec) in (TQFullAttentionSpec, TQSlidingWindowSpec)
        for spec in specs
    )
    has_native = any(
        type(spec) in (FullAttentionSpec, SlidingWindowSpec)
        for spec in specs
    )
    return (
        has_tq
        and has_native
        and all(type(spec) in supported_spec_types for spec in specs)
        and any(isinstance(spec, SlidingWindowSpec) for spec in specs)
    )


def _contiguous_strides(shape):
    """Row-major contiguous strides for ``shape`` (pure-Python equivalent
    of ``torch.empty(shape).stride()``)."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _padded_attention_view_strides(
    unpermuted_kv_cache_shape,
    kv_cache_stride_order,
    num_blocks,
    page_stride,
):
    """Compute the strided-view geometry for a padded attention page.

    Pure-Python core of vllm#45181's ``_reshape_attention_kv_cache``
    (kept torch-free so it is unit-testable on torch-less hosts):

      * permute the backend shape by its stride order;
      * detect the num_blocks dim on the UNPERMUTED shape, preferring
        dims 0/1 (upstream parity — avoids matching unrelated dims such
        as num_kv_heads that happen to equal num_blocks);
      * stride the block dim by the PADDED page;
      * if a FlashAttention-style explicit K/V dim (extent 2) is
        adjacent to the block dim, stride it by half the UNPADDED page
        so the V half is not read from the padding tail.

    Returns ``(permuted_shape, strides, inv_order)``.
    """
    kv_cache_shape = tuple(
        unpermuted_kv_cache_shape[i] for i in kv_cache_stride_order
    )
    inv_order = [
        kv_cache_stride_order.index(i)
        for i in range(len(kv_cache_stride_order))
    ]

    if unpermuted_kv_cache_shape[0] == num_blocks:
        unpermuted_num_blocks_dim = 0
    elif (
        len(unpermuted_kv_cache_shape) > 1
        and unpermuted_kv_cache_shape[1] == num_blocks
    ):
        unpermuted_num_blocks_dim = 1
    else:
        try:
            unpermuted_num_blocks_dim = unpermuted_kv_cache_shape.index(
                num_blocks
            )
        except ValueError as e:
            raise ValueError(
                f"num_blocks={num_blocks} not present in KV cache shape "
                f"{unpermuted_kv_cache_shape}; cannot build a padded "
                "strided view"
            ) from e
    num_blocks_dim = inv_order[unpermuted_num_blocks_dim]

    strides = _contiguous_strides(kv_cache_shape)
    unpadded_page_stride = strides[num_blocks_dim]
    strides[num_blocks_dim] = page_stride

    # FlashAttention-style layouts have an explicit K/V dimension
    # adjacent to the block dimension. Avoid matching unrelated size-2
    # dims like head_size or num_kv_heads (vllm#45181 parity).
    kv_dim = None
    for dim in (
        unpermuted_num_blocks_dim - 1,
        unpermuted_num_blocks_dim + 1,
    ):
        if (
            0 <= dim < len(unpermuted_kv_cache_shape)
            and unpermuted_kv_cache_shape[dim] == 2
        ):
            kv_dim = dim
            break
    if kv_dim is not None:
        strides[inv_order[kv_dim]] = unpadded_page_stride // 2

    return kv_cache_shape, tuple(strides), inv_order


def _reshape_attention_kv_cache(
    kv_tensor,
    kv_cache_spec,
    unpermuted_kv_cache_shape,
    kv_cache_stride_order,
    num_blocks,
):
    """Vendored from OPEN vllm#45181 (``vllm/v1/worker/gpu/attn_utils.py``).

    Builds the per-layer KV cache view; for padded pages it uses the
    hardened strided-view geometry from
    ``_padded_attention_view_strides`` instead of the pin's
    dim-0-assuming inline code.
    """
    import torch
    from vllm.utils.torch_utils import get_dtype_size

    if kv_cache_spec.page_size_padded is not None:
        dtype_size = get_dtype_size(kv_cache_spec.dtype)
        page_stride = kv_cache_spec.page_size_bytes // dtype_size
        kv_cache_shape, strides, inv_order = _padded_attention_view_strides(
            unpermuted_kv_cache_shape,
            kv_cache_stride_order,
            num_blocks,
            page_stride,
        )
        kv_cache = torch.as_strided(
            kv_tensor,
            size=kv_cache_shape,
            stride=strides,
        )
    else:
        kv_cache_shape = tuple(
            unpermuted_kv_cache_shape[i] for i in kv_cache_stride_order
        )
        inv_order = [
            kv_cache_stride_order.index(i)
            for i in range(len(kv_cache_stride_order))
        ]
        # No padding — safe to use a contiguous view.
        kv_cache = kv_tensor.view(kv_cache_shape)

    return kv_cache.permute(*inv_order)


_reshape_attention_kv_cache._genesis_g4_60e_injected = True  # type: ignore[attr-defined]


def _backend_block_dim(backend, kernel_block_size, spec, cache_dtype) -> int:
    """Best-effort ``get_kv_cache_block_dim`` probe (0 when unknown)."""
    get_block_dim = getattr(backend, "get_kv_cache_block_dim", None)
    if get_block_dim is None:
        return 0
    try:
        return int(
            get_block_dim(
                kernel_block_size,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=cache_dtype,
            )
        )
    except Exception:  # noqa: BLE001 — probe only, never fatal at boot
        return 0


def _post_correct_padded_attention_views(
    kv_caches,
    groups,
    kv_cache_raw_tensors,
    kernel_block_sizes,
    cache_dtype,
    skip_layer_names,
    legacy,
):
    """Rebuild padded-attention layer views with the hardened geometry.

    Runs AFTER the original reshape so unpadded models pay nothing and
    the original function keeps full ownership of Mamba/sharing/hybrid
    handling. Returns True when at least one view was rebuilt.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec

    changed = False
    has_mamba = any(
        isinstance(group.kv_cache_spec, MambaSpec) for group in groups
    )
    for group in groups:
        if group.kv_cache_group_id >= len(kernel_block_sizes):
            # Trailing group for layers without KV cache.
            continue
        spec = group.kv_cache_spec
        if not isinstance(spec, AttentionSpec):
            continue
        if getattr(spec, "page_size_padded", None) is None:
            continue

        if legacy:
            # Mirror GPUModelRunner._reshape_kv_cache_tensors (pin
            # g303916e93 lines 7060-7090).
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            num_blocks_per_kv_block = spec.block_size // kernel_block_size
            if spec.storage_block_size != spec.block_size:
                shape_block_size = spec.storage_block_size
            else:
                shape_block_size = kernel_block_size
        else:
            # Mirror gpu/attn_utils._reshape_kv_cache (pin g303916e93
            # lines 183-210).
            if spec.storage_block_size != spec.block_size:
                kernel_block_size = spec.storage_block_size
            else:
                kernel_block_size = kernel_block_sizes[
                    group.kv_cache_group_id
                ]
            num_blocks_per_kv_block = (
                spec.storage_block_size // kernel_block_size
            )
            shape_block_size = kernel_block_size

        for layer_name in group.layer_names:
            if (
                layer_name in skip_layer_names
                or layer_name not in kv_cache_raw_tensors
            ):
                continue
            raw_tensor = kv_cache_raw_tensors[layer_name]
            num_blocks = raw_tensor.numel() // spec.page_size_bytes
            kernel_num_blocks = num_blocks * num_blocks_per_kv_block
            unpermuted_shape = group.backend.get_kv_cache_shape(
                kernel_num_blocks,
                shape_block_size,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=cache_dtype,
            )
            try:
                stride_order = group.backend.get_kv_cache_stride_order()
                assert len(stride_order) == len(unpermuted_shape)
            except (AttributeError, NotImplementedError):
                stride_order = tuple(range(len(unpermuted_shape)))

            if (
                has_mamba
                and _backend_block_dim(
                    group.backend, shape_block_size, spec, cache_dtype
                )
                == 1
            ):
                # The hybrid attention/Mamba layout pass has already
                # restrided this view in place; rebuilding it here would
                # discard that. Padded (2, num_blocks, ...) layouts are
                # not supported with hybrid models (matches upstream
                # #45181 coverage) — keep the original view, warn loud.
                log.warning(
                    "[G4_60e] layer %s: padded page on a "
                    "(2, num_blocks, ...) layout with a hybrid Mamba "
                    "model — keeping the original view (unsupported "
                    "combination, see vllm#45181).",
                    layer_name,
                )
                continue

            kv_caches[layer_name] = _reshape_attention_kv_cache(
                raw_tensor.view(spec.dtype),
                spec,
                unpermuted_shape,
                stride_order,
                kernel_num_blocks,
            )
            changed = True
    return changed


def apply() -> tuple[str, str]:
    """Patch kv_cache_utils for TQ/native mixed-layout dispatch."""
    global _APPLIED, _ORIGINAL_IS_UNIFORM, _ORIGINAL_UNIFY, _ORIGINAL_GET_GROUPS
    global _ORIGINAL_RESHAPE_MODULAR, _ORIGINAL_RESHAPE_LEGACY
    global _PATCHED_ATTN_UTILS_MOD, _PATCHED_RUNNER_CLS

    if not _env_enabled():
        return "skipped", (
            f"G4_60e disabled (set {_ENV_ENABLE}=1 to enable TQ/native "
            "mixed-layout dispatch — PR #42637 cherry-pick + "
            "#45207/#45181 fold)"
        )

    if _APPLIED:
        return "applied", "G4_60e already installed (idempotent)"

    # G4_60a prerequisite — TQSlidingWindowSpec must be on the interface
    # module BEFORE we wrap functions that reference it.
    try:
        from vllm.v1.kv_cache_interface import TQSlidingWindowSpec  # noqa: F401
    except ImportError:
        return "skipped", (
            "G4_60a prerequisite not applied: TQSlidingWindowSpec missing "
            "on vllm.v1.kv_cache_interface. Enable "
            "GENESIS_ENABLE_G4_60A_TQ_SLIDING_SPEC=1 first."
        )

    try:
        from vllm.v1.core import kv_cache_utils as _kcu
    except ImportError as e:
        return "skipped", f"vllm.v1.core.kv_cache_utils not importable: {e}"

    # === Patch 1: is_kv_cache_spec_uniform ===
    _ORIGINAL_IS_UNIFORM = _kcu.is_kv_cache_spec_uniform

    def _patched_is_kv_cache_spec_uniform(kv_cache_spec):
        """Whether all layers have the same KV cache spec, with TQ/native
        detection (PR #42637 lines 872-875)."""
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            SlidingWindowSpec,
            TQFullAttentionSpec,
            TQSlidingWindowSpec,
        )

        if not kv_cache_spec:
            return True
        spec_types = {type(spec) for spec in kv_cache_spec.values()}
        if (
            TQFullAttentionSpec in spec_types and FullAttentionSpec in spec_types
        ) or (
            TQSlidingWindowSpec in spec_types and SlidingWindowSpec in spec_types
        ):
            return False
        try:
            kv_cache_spec_values = list(kv_cache_spec.values())
            _ = kv_cache_spec_values[0].merge(kv_cache_spec_values)
        except AssertionError:
            return False
        return True

    _patched_is_kv_cache_spec_uniform._genesis_g4_60e_wrapped = True  # type: ignore[attr-defined]
    _kcu.is_kv_cache_spec_uniform = _patched_is_kv_cache_spec_uniform

    # === Patch 2: unify_kv_cache_spec_page_size (reconciled ladder) ===
    _ORIGINAL_UNIFY = _kcu.unify_kv_cache_spec_page_size

    def _patched_unify_kv_cache_spec_page_size(kv_cache_spec):
        """Unify page sizes with the reconciled padding ladder.

        PR #42637 lines 1019-1063 (TQ branch) + vllm#45207 (MambaSpec
        padding) + vllm#45181 (generic AttentionSpec padding fallback).
        """
        from dataclasses import replace

        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            MambaSpec,
            TQFullAttentionSpec,
            TQSlidingWindowSpec,
        )

        page_sizes = {
            layer.page_size_bytes for layer in kv_cache_spec.values()
        }
        if len(page_sizes) <= 1:
            return kv_cache_spec

        max_page_size = max(page_sizes)
        tq_spec_types = (TQFullAttentionSpec, TQSlidingWindowSpec)
        new_kv_cache_spec: dict = {}
        for layer_name, layer_spec in kv_cache_spec.items():
            if layer_spec.page_size_bytes == max_page_size:
                new_kv_cache_spec[layer_name] = layer_spec
                continue

            if isinstance(layer_spec, tq_spec_types):
                # TQ specs cannot be resized via block_size — the TQ
                # slot layout is kernel-bound; they use the
                # page_size_padded field on the AttentionSpec base.
                padded_spec = replace(
                    layer_spec, page_size_padded=max_page_size
                )
                assert padded_spec.page_size_bytes == max_page_size, (
                    f"layer {layer_name}: TQ padding failed — "
                    f"{padded_spec.page_size_bytes=} != {max_page_size=}"
                )
                new_kv_cache_spec[layer_name] = padded_spec
                continue

            if isinstance(layer_spec, MambaSpec):
                # vllm#45207: MambaSpec's page size is determined by its
                # state shapes and does not scale with block_size, so pad
                # the page instead. This is the same padding mechanism
                # the platform uses to align Mamba pages with the main
                # model's attention page size; it is needed here when
                # another layer (e.g. from a draft model) has a larger
                # page than the already-aligned Mamba page.
                padded_spec = replace(
                    layer_spec, page_size_padded=max_page_size
                )
                assert padded_spec.page_size_bytes == max_page_size, (
                    f"layer {layer_name}: Mamba padding failed — "
                    f"{padded_spec.page_size_bytes=} != {max_page_size=}"
                )
                new_kv_cache_spec[layer_name] = padded_spec
                continue

            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                # Upstream original path: scale the logical block size.
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                resized_spec = replace(
                    layer_spec, block_size=new_block_size
                )
                assert resized_spec.page_size_bytes == max_page_size, (
                    f"layer {layer_name}: page size "
                    f"{resized_spec.page_size_bytes} after scaling "
                    f"block_size to {new_block_size} does not match the "
                    f"maximum page size {max_page_size}"
                )
                new_kv_cache_spec[layer_name] = resized_spec
            elif isinstance(layer_spec, AttentionSpec):
                # vllm#45181: a smaller attention page that does not
                # evenly divide the maximum keeps its logical block size
                # and pads its physical page instead (DFlash drafters
                # with smaller head sizes, TQ-vs-bf16 mixes, etc.).
                padded_spec = replace(
                    layer_spec, page_size_padded=max_page_size
                )
                assert padded_spec.page_size_bytes == max_page_size, (
                    f"layer {layer_name}: attention padding failed — "
                    f"{padded_spec.page_size_bytes=} != {max_page_size=}"
                )
                new_kv_cache_spec[layer_name] = padded_spec
            else:
                # Runtime guard mirroring upstream's behavior when a
                # layer's page size is incompatible with the unified
                # max. Not a stub; raising is the correct outcome.
                # audit-no-stub: allow
                raise NotImplementedError(
                    f"The page size of layer {layer_name} "
                    f"({layer_page_size} bytes) is not a divisor of the "
                    f"maximum page size ({max_page_size} bytes) and the "
                    "KV cache spec cannot be padded. Cannot unify."
                )
        return new_kv_cache_spec

    _patched_unify_kv_cache_spec_page_size._genesis_g4_60e_wrapped = True  # type: ignore[attr-defined]
    _kcu.unify_kv_cache_spec_page_size = _patched_unify_kv_cache_spec_page_size

    # === Patch 3: inject _is_tq_native_mixed_kv_cache_spec predicate ===
    _kcu._is_tq_native_mixed_kv_cache_spec = _is_tq_native_mixed_kv_cache_spec

    # === Patch 4: wrap get_kv_cache_groups dispatch ===
    if hasattr(_kcu, "get_kv_cache_groups"):
        _ORIGINAL_GET_GROUPS = _kcu.get_kv_cache_groups

        def _patched_get_kv_cache_groups(vllm_config, kv_cache_spec):
            """Route TQ/native mixed layouts through hybrid manager.

            Inserted branch from PR #42637 lines 1696-1706. Falls through
            to original implementation for non-mixed cases.
            """
            from vllm.v1.core.kv_cache_utils import (
                is_kv_cache_type_attention_free,
            )

            if is_kv_cache_type_attention_free(kv_cache_spec):
                return []

            disable_hybrid = getattr(
                vllm_config.scheduler_config,
                "disable_hybrid_kv_cache_manager",
                False,
            )
            if (
                not disable_hybrid
                and _is_tq_native_mixed_kv_cache_spec(kv_cache_spec)
            ):
                # TQ+native mixed layouts need to preserve hybrid block
                # manager semantics. See PR #42637 line 1700 comment.
                kv_cache_spec = _patched_unify_kv_cache_spec_page_size(
                    kv_cache_spec
                )
                return _kcu._get_kv_cache_groups_uniform_page_size(
                    kv_cache_spec
                )

            return _ORIGINAL_GET_GROUPS(vllm_config, kv_cache_spec)

        _patched_get_kv_cache_groups._genesis_g4_60e_wrapped = True  # type: ignore[attr-defined]
        _kcu.get_kv_cache_groups = _patched_get_kv_cache_groups
    else:
        log.warning(
            "[G4_60e] get_kv_cache_groups not found — dispatch wrap skipped"
        )

    # === Patches 5+6: worker-side padded-view stride hardening ===
    # (vllm#45181 fold). Non-fatal: older pins / partial trees without
    # the worker modules still get the kv_cache_utils patches above.
    try:
        from vllm.v1.worker.gpu import attn_utils as _attn_utils
    except ImportError as e:
        _attn_utils = None
        log.info(
            "[G4_60e] vllm.v1.worker.gpu.attn_utils not importable (%s) — "
            "modular reshape hardening skipped.", e,
        )
    try:
        from vllm.v1.worker import gpu_model_runner as _gmr
    except ImportError as e:
        _gmr = None
        log.info(
            "[G4_60e] vllm.v1.worker.gpu_model_runner not importable (%s) "
            "— legacy reshape hardening skipped.", e,
        )

    native_helper = getattr(
        _attn_utils, "_reshape_attention_kv_cache", None
    ) if _attn_utils is not None else None
    upstream_45181_merged = native_helper is not None and not getattr(
        native_helper, "_genesis_g4_60e_injected", False
    )
    if upstream_45181_merged:
        log.info(
            "[G4_60e] upstream vllm#45181 merged form detected "
            "(_reshape_attention_kv_cache present natively) — reshape "
            "patches 5/6 self-skip. Retirement review due for the "
            "#45181 portion of G4_60e."
        )

    if (
        _attn_utils is not None
        and not upstream_45181_merged
        and hasattr(_attn_utils, "_reshape_kv_cache")
    ):
        # === Patch 5: modular-runner _reshape_kv_cache wrap ===
        _ORIGINAL_RESHAPE_MODULAR = _attn_utils._reshape_kv_cache
        _PATCHED_ATTN_UTILS_MOD = _attn_utils

        @functools.wraps(_ORIGINAL_RESHAPE_MODULAR)
        def _patched_reshape_kv_cache(
            attn_groups,
            kv_cache_raw_tensors,
            cache_dtype,
            kernel_block_sizes,
            shared_kv_cache_layers,
        ):
            kv_caches = _ORIGINAL_RESHAPE_MODULAR(
                attn_groups,
                kv_cache_raw_tensors,
                cache_dtype,
                kernel_block_sizes,
                shared_kv_cache_layers,
            )
            groups = list(attn_groups)
            changed = _post_correct_padded_attention_views(
                kv_caches,
                groups,
                kv_cache_raw_tensors,
                kernel_block_sizes,
                cache_dtype,
                set(shared_kv_cache_layers),
                legacy=False,
            )
            if changed:
                # Re-alias shared layers: the original mapped them to
                # the now-replaced target views.
                for layer_name, target in shared_kv_cache_layers.items():
                    if target in kv_caches:
                        kv_caches[layer_name] = kv_caches[target]
            return kv_caches

        _patched_reshape_kv_cache._genesis_g4_60e_wrapped = True  # type: ignore[attr-defined]
        _attn_utils._reshape_kv_cache = _patched_reshape_kv_cache
        # Upstream-parity helper name — merged-form detection hook for
        # future pins, and a single shared implementation for both
        # runner paths (mirrors #45181's gpu_model_runner import).
        _attn_utils._reshape_attention_kv_cache = _reshape_attention_kv_cache

    if _gmr is not None and not upstream_45181_merged and hasattr(
        getattr(_gmr, "GPUModelRunner", None), "_reshape_kv_cache_tensors"
    ):
        # === Patch 6: legacy-runner _reshape_kv_cache_tensors wrap ===
        _PATCHED_RUNNER_CLS = _gmr.GPUModelRunner
        _ORIGINAL_RESHAPE_LEGACY = (
            _PATCHED_RUNNER_CLS._reshape_kv_cache_tensors
        )

        @functools.wraps(_ORIGINAL_RESHAPE_LEGACY)
        def _patched_reshape_kv_cache_tensors(
            self, kv_cache_raw_tensors, kernel_block_sizes
        ):
            kv_caches = _ORIGINAL_RESHAPE_LEGACY(
                self, kv_cache_raw_tensors, kernel_block_sizes
            )
            groups = list(self._kv_cache_spec_attn_group_iterator())
            cache_dtype = getattr(
                getattr(self, "cache_config", None), "cache_dtype", "auto"
            )
            skip_layers = set(
                getattr(self, "runner_only_attn_layers", ()) or ()
            )
            _post_correct_padded_attention_views(
                kv_caches,
                groups,
                kv_cache_raw_tensors,
                kernel_block_sizes,
                cache_dtype,
                skip_layers,
                legacy=True,
            )
            return kv_caches

        _patched_reshape_kv_cache_tensors._genesis_g4_60e_wrapped = True  # type: ignore[attr-defined]
        _PATCHED_RUNNER_CLS._reshape_kv_cache_tensors = (
            _patched_reshape_kv_cache_tensors
        )

    _APPLIED = True
    log.info(
        "[G4_60e] kv_cache_utils patches installed: is_kv_cache_spec_uniform "
        "+ unify_kv_cache_spec_page_size (TQ/Mamba/AttentionSpec padding "
        "ladder, #45207+#45181 fold) + _is_tq_native_mixed_kv_cache_spec "
        "+ get_kv_cache_groups dispatch + worker reshape stride hardening "
        "(modular=%s, legacy=%s).",
        _ORIGINAL_RESHAPE_MODULAR is not None,
        _ORIGINAL_RESHAPE_LEGACY is not None,
    )
    return "applied", (
        "G4_60e installed: TQ/native mixed-layout routing active "
        "(+ Mamba page padding #45207, AttentionSpec padding fallback "
        "and reshape stride hardening #45181)."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_IS_UNIFORM, _ORIGINAL_UNIFY, _ORIGINAL_GET_GROUPS
    global _ORIGINAL_RESHAPE_MODULAR, _ORIGINAL_RESHAPE_LEGACY
    global _PATCHED_ATTN_UTILS_MOD, _PATCHED_RUNNER_CLS
    if not _APPLIED:
        return False
    try:
        from vllm.v1.core import kv_cache_utils as _kcu

        if _ORIGINAL_IS_UNIFORM is not None:
            _kcu.is_kv_cache_spec_uniform = _ORIGINAL_IS_UNIFORM
        if _ORIGINAL_UNIFY is not None:
            _kcu.unify_kv_cache_spec_page_size = _ORIGINAL_UNIFY
        if _ORIGINAL_GET_GROUPS is not None:
            _kcu.get_kv_cache_groups = _ORIGINAL_GET_GROUPS
        if hasattr(_kcu, "_is_tq_native_mixed_kv_cache_spec"):
            delattr(_kcu, "_is_tq_native_mixed_kv_cache_spec")

        if (
            _PATCHED_ATTN_UTILS_MOD is not None
            and _ORIGINAL_RESHAPE_MODULAR is not None
        ):
            _PATCHED_ATTN_UTILS_MOD._reshape_kv_cache = (
                _ORIGINAL_RESHAPE_MODULAR
            )
            injected = getattr(
                _PATCHED_ATTN_UTILS_MOD,
                "_reshape_attention_kv_cache",
                None,
            )
            if injected is not None and getattr(
                injected, "_genesis_g4_60e_injected", False
            ):
                delattr(
                    _PATCHED_ATTN_UTILS_MOD, "_reshape_attention_kv_cache"
                )
        if (
            _PATCHED_RUNNER_CLS is not None
            and _ORIGINAL_RESHAPE_LEGACY is not None
        ):
            _PATCHED_RUNNER_CLS._reshape_kv_cache_tensors = (
                _ORIGINAL_RESHAPE_LEGACY
            )
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_IS_UNIFORM = None
    _ORIGINAL_UNIFY = None
    _ORIGINAL_GET_GROUPS = None
    _ORIGINAL_RESHAPE_MODULAR = None
    _ORIGINAL_RESHAPE_LEGACY = None
    _PATCHED_ATTN_UTILS_MOD = None
    _PATCHED_RUNNER_CLS = None
    return True


__all__ = [
    "GENESIS_G4_60E_MARKER",
    "apply",
    "is_applied",
    "revert",
]

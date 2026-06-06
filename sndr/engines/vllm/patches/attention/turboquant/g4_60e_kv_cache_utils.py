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

================================================================
FIX (PR #42637 cherry-pick)
================================================================

Three monkey-patches and one new function:

  1. **Replace** ``is_kv_cache_spec_uniform`` — add early-out for
     ``TQFullAttentionSpec×FullAttentionSpec`` and
     ``TQSlidingWindowSpec×SlidingWindowSpec`` co-presence (returns False).

  2. **Replace** ``unify_kv_cache_spec_page_size`` — for TQ specs that
     would otherwise need ``block_size`` resize, use
     ``page_size_padded`` field instead. Falls through to original
     ``block_size`` ratio for non-TQ specs.

  3. **Inject** ``_is_tq_native_mixed_kv_cache_spec`` predicate that
     identifies the supported mixed layout (TQ + plain Full/Sliding only;
     not MLA, Mamba, ChunkedLocal).

  4. **Wrap** ``get_kv_cache_groups`` to add the mixed-route branch
     before the existing ``is_kv_cache_spec_uniform`` check. When the
     predicate matches, route through
     ``unify_kv_cache_spec_page_size + _get_kv_cache_groups_uniform_
     page_size``.

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
4 symbols on a single module. The mixed-route branch is only taken
when the predicate matches (TQ + native co-present); all-TQ and
pure-native flows go through unchanged paths.

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
  * Companion patches: G4_60a (TQSlidingWindowSpec), G4_60h (slot_size).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_60e_kv_cache_utils")

GENESIS_G4_60E_MARKER = (
    "Genesis G4_60e kv_cache_utils.py patches: "
    "is_kv_cache_spec_uniform + unify_kv_cache_spec_page_size + "
    "_is_tq_native_mixed_kv_cache_spec + get_kv_cache_groups dispatch "
    "(PR #42637 cherry-pick)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_60E_KV_CACHE_UTILS"
_APPLIED = False
_ORIGINAL_IS_UNIFORM = None
_ORIGINAL_UNIFY = None
_ORIGINAL_GET_GROUPS = None


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


def apply() -> tuple[str, str]:
    """Patch kv_cache_utils for TQ/native mixed-layout dispatch."""
    global _APPLIED, _ORIGINAL_IS_UNIFORM, _ORIGINAL_UNIFY, _ORIGINAL_GET_GROUPS

    if not _env_enabled():
        return "skipped", (
            f"G4_60e disabled (set {_ENV_ENABLE}=1 to enable TQ/native "
            "mixed-layout dispatch — PR #42637 cherry-pick)"
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

    # === Patch 2: unify_kv_cache_spec_page_size ===
    _ORIGINAL_UNIFY = _kcu.unify_kv_cache_spec_page_size

    def _patched_unify_kv_cache_spec_page_size(kv_cache_spec):
        """Unify page size with TQ-aware padding (PR #42637 lines 1019-1063)."""
        from dataclasses import replace

        from vllm.v1.kv_cache_interface import (
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
                # TQ specs cannot be resized via block_size — they use
                # the page_size_padded field on the AttentionSpec base.
                padded_spec = replace(
                    layer_spec, page_size_padded=max_page_size
                )
                assert padded_spec.page_size_bytes == max_page_size, (
                    f"TQ padding failed: {padded_spec.page_size_bytes=} != "
                    f"{max_page_size=}"
                )
                new_kv_cache_spec[layer_name] = padded_spec
                continue

            # Non-TQ path: block_size ratio resize (verbatim original).
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size != 0:
                # Runtime guard mirroring upstream's behavior when a
                # layer's page size is incompatible with the unified
                # max. Not a stub; raising is the correct outcome.
                # audit-no-stub: allow
                raise NotImplementedError(
                    "The page size of the layer is not divisible by the "
                    "maximum page size. Cannot unify by adjusting "
                    "block_size."
                )
            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            resized_spec = replace(layer_spec, block_size=new_block_size)
            assert resized_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = resized_spec
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

    _APPLIED = True
    log.info(
        "[G4_60e] kv_cache_utils patches installed: is_kv_cache_spec_uniform "
        "+ unify_kv_cache_spec_page_size + _is_tq_native_mixed_kv_cache_spec "
        "+ get_kv_cache_groups dispatch."
    )
    return "applied", (
        "G4_60e installed: TQ/native mixed-layout routing active."
    )


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_IS_UNIFORM, _ORIGINAL_UNIFY, _ORIGINAL_GET_GROUPS
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
    except Exception:  # noqa: BLE001
        return False
    _APPLIED = False
    _ORIGINAL_IS_UNIFORM = None
    _ORIGINAL_UNIFY = None
    _ORIGINAL_GET_GROUPS = None
    return True


__all__ = [
    "GENESIS_G4_60E_MARKER",
    "apply",
    "is_applied",
    "revert",
]

# SPDX-License-Identifier: Apache-2.0
"""G4_74 — Drafter HND layout enforcement post-reshape (PN263 fix).

================================================================
PROBLEM (PN262 confirmed)
================================================================

After G4_71 (drafter impl→FlashAttn) + G4_72 (drafter spec→native) +
G4_73 (skip drafter profile dummy_run) + upstream
``SpeculativeConfig.attention_backend=FLASH_ATTN``, K=2 still crashed
at runtime in ``flash_attn.py:744``::

    key_cache, value_cache = kv_cache.unbind(0)
    ValueError: too many values to unpack (expected 2)

Fixed PN262 args index (kv_cache is args[4], not args[3]) captured the
actual drafter kv_cache shape::

    drafter sliding layer:  shape=(4, 2, 16, 8, 256)
                            stride=(65536, 32768, 2048, 256, 1)
    drafter full layer:     shape=(4, 2, 32, 2, 512)
                            stride=(65536, 32768, 1024, 512, 1)
    ndim=5  dtype=bf16  contiguous=True

That's NHD layout — ``(num_blocks=4, 2, block_size, num_kv_heads,
head_dim)``. FlashAttn at line 744 expects HND layout —
``(2, num_blocks, block_size, num_kv_heads, head_dim)`` — so
``kv_cache.unbind(0)`` cleanly splits into ``key_cache`` and
``value_cache``. NHD's leading dim is ``num_blocks`` (4) — unbind(0)
returns 4 tensors, fails to unpack into 2.

Diagnostics confirmed not aliasing (``kv_sharing_target=None``), not
``VLLM_KV_CACHE_LAYOUT`` env, not PN259c (A/B identical with PN259c=0).
Path A (``--speculative-config '{"attention_backend":"FLASH_ATTN"}'``)
gave bit-identical NHD shape — upstream's attention_backend field
controls draft ``attention_config.backend`` but does NOT propagate to
the physical kv_cache layout decided by
``GPUModelRunner._reshape_kv_cache_tensors``.

================================================================
FIX
================================================================

Wrap ``GPUModelRunner._reshape_kv_cache_tensors``. After the original
call returns the ``kv_caches`` dict, for each layer whose name starts
with ``"draft_model."``::

  * If shape is already HND ``(2, num_blocks, ...)`` → no-op.
  * If shape is NHD ``(num_blocks, 2, ...)`` → replace with
    ``kv_caches[layer_name] = kv_cache.transpose(0, 1).contiguous()``
    to materialize the HND-layout tensor.
  * Any other 5-D shape → fail-fast with full context, so the operator
    knows the assumed mapping is wrong on a different config.

The mutated dict propagates through the remaining lines of
``initialize_kv_cache_tensors`` (cross-layer share lookup,
``bind_kv_cache``) so attention context delivers the HND tensor to
FlashAttn forward.

================================================================
WHY DRAFTER-ONLY
================================================================

Target (TQ) layers MUST stay in their TQ layout — that's the contract
TurboQuant Triton kernels expect. Touching their shape would crash
TQ decode kernels and break the entire engine. Only the drafter
layers (G4_71 → FlashAttn impl) need HND.

================================================================
WHY POST-RESHAPE, NOT INSIDE
================================================================

Rewriting the inside of ``_reshape_kv_cache_tensors`` (which is 100+
lines of upstream logic with MLA / Mamba / kernel-block sizing edge
cases) is far more invasive than a single ``.transpose(0, 1).contiguous()``
post-hook. The post-hook also runs BEFORE
``bind_kv_cache(kv_caches, static_forward_context, self.kv_caches, ...)``
so the static forward context picks up the HND tensor — that's the
critical timing requirement.

================================================================
ENV FLAG
================================================================

  GENESIS_ENABLE_G4_74_DRAFTER_HND_LAYOUT=1   (opt-in)
  GENESIS_G4_74_DRAFTER_PREFIX=draft_model.   (override prefix)

================================================================
ACCEPTANCE GATES
================================================================

  1. K=2 boot — server up.
  2. K=2 first prompt — PN262 trace MUST show ``shape[0] == 2`` for
     drafter; no flash_attn.py:744 unbind(0) ValueError; no PN261-A
     RuntimeError; no cudaErrorIllegalAddress.
  3. PN248 acceptance trace — ``accepted_per_req > 0``.
  4. K=4 — same checks once K=2 is clean.

================================================================
RISKS
================================================================

  * Memory: ``.contiguous()`` of the transpose materializes a new
    tensor temporarily holding both old + new (transient ~32 MiB per
    drafter layer × 4 layers = ~128 MiB on each TP rank during boot).
    Drafter is small; trivial vs 31B target.
  * Timing: the wrap MUST replace ``kv_caches[layer_name]`` in the
    returned dict BEFORE bind_kv_cache runs. Achieved by post-call
    mutation of the returned dict.
  * If a future pin uses ``allocate_uniform_kv_caches`` fast path
    instead of ``_reshape_kv_cache_tensors`` (requires kv_transfer_group
    + matching FlashAttn stride_order), G4_74 silently no-ops on that
    path. We log apply status so it's visible.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.g4_74_drafter_hnd_layout")

GENESIS_G4_74_MARKER = (
    "Genesis G4_74 Drafter HND layout enforcement post-reshape "
    "(PN263 fix for NHD-shaped FlashAttn kv_cache on Gemma 4 MTP)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_74_DRAFTER_HND_LAYOUT"
_ENV_PREFIX = "GENESIS_G4_74_DRAFTER_PREFIX"
_ENV_MAX_BLOCKS = "GENESIS_G4_74_DRAFTER_MAX_BLOCKS"
_APPLIED = False
_ORIGINAL_RESHAPE = None
_CONVERT_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _drafter_prefix() -> str:
    return os.environ.get(_ENV_PREFIX, "draft_model.").strip()


def _drafter_max_blocks() -> int:
    """0 means no cap (full transpose+contiguous). >0 means allocate
    a fresh zero HND tensor with min(source_num_blocks, cap) blocks.
    """
    raw = os.environ.get(_ENV_MAX_BLOCKS, "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def apply() -> tuple[str, str]:
    """Install drafter-only HND-layout post-bind rebinding.

    PIVOT (2026-05-19 K=2 gate): wrapping _reshape_kv_cache_tensors did
    not intercept drafter because drafter is NOT in that function's
    output. Gemma4Proposer._setup_gemma4_kv_sharing makes drafter ALIAS
    a target layer's kv_cache (via kv_sharing_target_layer_name). The
    aliased target tensor is NHD-laid-out, so drafter inherits NHD.

    Solution: wrap GPUModelRunner.initialize_kv_cache_tensors. AFTER its
    original call (which includes bind_kv_cache that sets each
    Attention.kv_cache via aliasing), iterate
    self.compilation_config.static_forward_context for drafter Attention
    layers. For each drafter that has a 5-D NHD kv_cache, replace it
    with a freshly-allocated HND tensor via
    ``kv_cache.transpose(0, 1).contiguous()``. This BREAKS the alias to
    target (memory is independent) — drafter now has its own bf16 HND
    cache. The transpose+contiguous copy preserves data at allocation
    time (which is zero for newly-allocated cache).

    Trade-off: drafter no longer shares context with target. The Gemma4
    kv_sharing was an inference-time acceleration to give drafter cold
    access to target's accumulated KV. With acceptance currently at 0%
    (per H8 notes), this trade is acceptable — it unblocks the layout
    crash and lets us measure whether drafter even produces useful
    drafts on its own.
    """
    global _APPLIED, _ORIGINAL_RESHAPE

    if not _env_enabled():
        return "skipped", (
            f"G4_74 disabled (set {_ENV_ENABLE}=1 to force HND layout on "
            "drafter kv_cache after initialize_kv_cache_tensors — PN263 "
            "fix for FlashAttn unbind(0) on NHD-shared drafter cache)"
        )

    if _APPLIED:
        return "applied", "G4_74 already installed (idempotent)"

    log.warning("[G4_74] apply() entered — beginning import phase")

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_74] SKIP: GPUModelRunner not importable: %s", e)
        return "skipped", f"GPUModelRunner not importable: {e!r}"

    if not hasattr(GPUModelRunner, "initialize_kv_cache_tensors"):
        return "skipped", "GPUModelRunner.initialize_kv_cache_tensors missing on this pin"

    original_init_tensors = GPUModelRunner.initialize_kv_cache_tensors
    if getattr(original_init_tensors, "_genesis_g4_74_wrapped", False):
        _APPLIED = True
        return "applied", "GPUModelRunner.initialize_kv_cache_tensors already wrapped"
    _ORIGINAL_RESHAPE = original_init_tensors

    drafter_prefix = _drafter_prefix()
    drafter_max_blocks = _drafter_max_blocks()
    log.warning(
        "[G4_74] import phase OK — drafter_prefix=%r max_blocks=%d "
        "(0=no cap); about to wrap "
        "GPUModelRunner.initialize_kv_cache_tensors (post-bind hook)",
        drafter_prefix, drafter_max_blocks,
    )

    def _wrapped_init_tensors(self, kv_cache_config, kernel_block_sizes):
        """Post-call: break drafter kv_sharing and give it own HND cache."""
        kv_caches = original_init_tensors(self, kv_cache_config, kernel_block_sizes)

        # After bind_kv_cache, each Attention.kv_cache is set. For drafter
        # layers aliased to a target tensor, replace with an independent
        # HND-laid-out copy.
        try:
            fwd_ctx = self.compilation_config.static_forward_context
            ctx_items = list(fwd_ctx.items())
        except Exception as _e:
            log.warning(
                "[G4_74] could not access static_forward_context: %s; "
                "drafter HND layout not enforced",
                _e,
            )
            return kv_caches

        for layer_name, attn_layer in ctx_items:
            if not (isinstance(layer_name, str)
                    and layer_name.startswith(drafter_prefix)):
                continue

            # G4_75 marker: this drafter layer was rerouted to a
            # backend (typically TRITON_ATTN) that uses NHD layout
            # natively. Skip HND conversion for it.
            if getattr(attn_layer, "_genesis_g4_75_drafter_triton", False):
                _CONVERT_COUNT[0] += 1
                if _CONVERT_COUNT[0] <= 12:
                    log.warning(
                        "[G4_74] drafter layer=%r marked by G4_75 "
                        "(triton/native-NHD backend) — skip HND "
                        "conversion (count=%d)",
                        layer_name, _CONVERT_COUNT[0],
                    )
                continue

            kv_cache = getattr(attn_layer, "kv_cache", None)
            if kv_cache is None:
                continue

            try:
                ndim = int(kv_cache.dim())
                shape = tuple(kv_cache.shape)
                stride_before = tuple(kv_cache.stride())
                contig_before = bool(kv_cache.is_contiguous())
                data_ptr_before = int(kv_cache.data_ptr())
            except Exception as _e:
                log.warning(
                    "[G4_74] introspection failed on drafter kv_cache "
                    "layer=%r: %s; skipping",
                    layer_name, _e,
                )
                continue

            if ndim == 0 or (ndim == 1 and shape == (0,)):
                # Empty placeholder (drafter not yet bound — unexpected at
                # this point, but skip rather than crash).
                continue

            if ndim != 5:
                raise RuntimeError(
                    f"[G4_74] drafter layer {layer_name!r} has unexpected "
                    f"ndim={ndim} (expected 5); shape={shape} "
                    f"stride={stride_before} dtype={kv_cache.dtype} "
                    f"contig={contig_before}. Disable G4_74 to bypass; "
                    f"investigate the allocator before re-enabling."
                )

            if shape[0] == 2:
                _CONVERT_COUNT[0] += 1
                if _CONVERT_COUNT[0] <= 12:
                    log.warning(
                        "[G4_74] drafter layer=%r already HND "
                        "shape=%s — no-op (count=%d)",
                        layer_name, shape, _CONVERT_COUNT[0],
                    )
                continue

            if shape[1] == 2:
                # NHD layout aliased from target.
                # If drafter_max_blocks > 0, allocate a FRESH zero HND
                # tensor capped at max_blocks (drafter doesn't need the
                # target's full num_blocks; it only needs enough for the
                # current sequence + K-step lookahead). This dramatically
                # reduces memory vs full transpose+contiguous of a
                # full-size target tensor.
                # If drafter_max_blocks == 0, full transpose+contiguous.
                import torch as _torch
                source_num_blocks = int(shape[0])
                cap = drafter_max_blocks if drafter_max_blocks > 0 else source_num_blocks
                effective_num_blocks = min(source_num_blocks, cap)

                if drafter_max_blocks > 0 and effective_num_blocks < source_num_blocks:
                    # Capped path — fresh zero allocation.
                    new_hnd_shape = (2, effective_num_blocks) + tuple(shape[2:])
                    new_kv_cache = _torch.zeros(
                        new_hnd_shape,
                        dtype=kv_cache.dtype,
                        device=kv_cache.device,
                    )
                else:
                    # Full transpose+contiguous.
                    new_kv_cache = kv_cache.transpose(0, 1).contiguous()

                attn_layer.kv_cache = new_kv_cache

                # Also patch the kv_caches dict if drafter is present there
                # (covers any consumer that re-reads the dict).
                if layer_name in kv_caches:
                    kv_caches[layer_name] = new_kv_cache

                _CONVERT_COUNT[0] += 1
                if _CONVERT_COUNT[0] <= 12:
                    log.warning(
                        "[G4_74] drafter layer=%r NHD->HND (alias broken): "
                        "before shape=%s stride=%s data_ptr=0x%x -> "
                        "after shape=%s stride=%s data_ptr=0x%x "
                        "contig=%s capped_blocks=%d (count=%d)",
                        layer_name,
                        shape, stride_before, data_ptr_before,
                        tuple(new_kv_cache.shape),
                        tuple(new_kv_cache.stride()),
                        int(new_kv_cache.data_ptr()),
                        bool(new_kv_cache.is_contiguous()),
                        effective_num_blocks,
                        _CONVERT_COUNT[0],
                    )
                elif _CONVERT_COUNT[0] == 13:
                    log.warning(
                        "[G4_74] further drafter NHD->HND logs suppressed (> 12)"
                    )
                continue

            raise RuntimeError(
                f"[G4_74] drafter layer {layer_name!r} has 5-D shape "
                f"{shape} with neither shape[0]==2 nor shape[1]==2; "
                f"stride={stride_before} dtype={kv_cache.dtype} "
                f"contig={contig_before}. FlashAttn expects "
                f"shape[0]==2 (HND). Cannot determine intended axis for "
                f"transpose. Disable G4_74 and investigate the allocator."
            )

        return kv_caches

    _wrapped_init_tensors._genesis_g4_74_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner.initialize_kv_cache_tensors = _wrapped_init_tensors  # type: ignore[method-assign]
    _APPLIED = True

    log.warning(
        "[G4_74] INSTALLED: GPUModelRunner.initialize_kv_cache_tensors wrapped; "
        "drafter layers (prefix=%r) with NHD layout will have alias broken "
        "and replaced with independent HND tensor after bind_kv_cache.",
        drafter_prefix,
    )
    return "applied", (
        f"G4_74 installed: drafter (prefix {drafter_prefix!r}) NHD->HND "
        f"post-bind rebinding active (breaks gemma4 kv_sharing alias)."
    )


def is_applied() -> bool:
    return _APPLIED


def convert_count() -> int:
    return _CONVERT_COUNT[0]


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_RESHAPE
    if not _APPLIED or _ORIGINAL_RESHAPE is None:
        return False
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        GPUModelRunner._reshape_kv_cache_tensors = _ORIGINAL_RESHAPE  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_RESHAPE = None
    return True


__all__ = [
    "GENESIS_G4_74_MARKER",
    "apply",
    "is_applied",
    "convert_count",
    "revert",
]

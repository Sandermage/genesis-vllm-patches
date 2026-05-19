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
_APPLIED = False
_ORIGINAL_RESHAPE = None
_CONVERT_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _drafter_prefix() -> str:
    return os.environ.get(_ENV_PREFIX, "draft_model.").strip()


def apply() -> tuple[str, str]:
    """Install drafter-only HND-layout post-reshape rebinding."""
    global _APPLIED, _ORIGINAL_RESHAPE

    if not _env_enabled():
        return "skipped", (
            f"G4_74 disabled (set {_ENV_ENABLE}=1 to force HND layout on "
            "drafter kv_cache after _reshape_kv_cache_tensors — PN263 fix "
            "for FlashAttn unbind(0) on NHD-shaped drafter cache)"
        )

    if _APPLIED:
        return "applied", "G4_74 already installed (idempotent)"

    log.warning("[G4_74] apply() entered — beginning import phase")

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        log.warning("[G4_74] SKIP: GPUModelRunner not importable: %s", e)
        return "skipped", f"GPUModelRunner not importable: {e!r}"

    if not hasattr(GPUModelRunner, "_reshape_kv_cache_tensors"):
        return "skipped", "GPUModelRunner._reshape_kv_cache_tensors missing on this pin"

    original_reshape = GPUModelRunner._reshape_kv_cache_tensors
    if getattr(original_reshape, "_genesis_g4_74_wrapped", False):
        _APPLIED = True
        return "applied", "GPUModelRunner._reshape_kv_cache_tensors already wrapped"
    _ORIGINAL_RESHAPE = original_reshape

    drafter_prefix = _drafter_prefix()
    log.warning(
        "[G4_74] import phase OK — drafter_prefix=%r; about to wrap "
        "GPUModelRunner._reshape_kv_cache_tensors",
        drafter_prefix,
    )

    def _wrapped_reshape(self, kv_cache_raw_tensors, kernel_block_sizes):
        """Post-call: transpose drafter kv_cache from NHD to HND."""
        kv_caches = original_reshape(self, kv_cache_raw_tensors, kernel_block_sizes)

        # Iterate keys in a list copy because we mutate the dict.
        try:
            items = list(kv_caches.items())
        except Exception as _e:
            log.warning(
                "[G4_74] result is not a dict-like (%s); cannot post-fix",
                _e,
            )
            return kv_caches

        for layer_name, kv_cache in items:
            if not (isinstance(layer_name, str)
                    and layer_name.startswith(drafter_prefix)):
                continue

            try:
                ndim = int(kv_cache.dim())
                shape = tuple(kv_cache.shape)
                stride_before = tuple(kv_cache.stride())
                contig_before = bool(kv_cache.is_contiguous())
            except Exception as _e:
                log.warning(
                    "[G4_74] introspection failed on drafter kv_cache "
                    "layer=%r: %s; skipping",
                    layer_name, _e,
                )
                continue

            if ndim != 5:
                # Non-5D drafter cache is unexpected for FlashAttn path; fail
                # fast so the operator sees the actual shape rather than a
                # downstream surprise.
                raise RuntimeError(
                    f"[G4_74] drafter layer {layer_name!r} has unexpected "
                    f"ndim={ndim} (expected 5); shape={shape} stride={stride_before} "
                    f"dtype={kv_cache.dtype} contig={contig_before}. "
                    f"Disable G4_74 (GENESIS_ENABLE_G4_74_DRAFTER_HND_LAYOUT=0) "
                    f"to bypass; investigate the allocator before re-enabling."
                )

            if shape[0] == 2:
                # Already HND. No-op.
                _CONVERT_COUNT[0] += 1
                if _CONVERT_COUNT[0] <= 12:
                    log.warning(
                        "[G4_74] drafter layer=%r already HND "
                        "shape=%s — no-op (count=%d)",
                        layer_name, shape, _CONVERT_COUNT[0],
                    )
                continue

            if shape[1] == 2:
                # NHD layout — transpose to HND.
                new_kv_cache = kv_cache.transpose(0, 1).contiguous()
                kv_caches[layer_name] = new_kv_cache
                _CONVERT_COUNT[0] += 1
                if _CONVERT_COUNT[0] <= 12:
                    log.warning(
                        "[G4_74] drafter layer=%r NHD->HND: "
                        "before shape=%s stride=%s contig=%s -> "
                        "after shape=%s stride=%s contig=%s (count=%d)",
                        layer_name,
                        shape, stride_before, contig_before,
                        tuple(new_kv_cache.shape),
                        tuple(new_kv_cache.stride()),
                        bool(new_kv_cache.is_contiguous()),
                        _CONVERT_COUNT[0],
                    )
                elif _CONVERT_COUNT[0] == 13:
                    log.warning(
                        "[G4_74] further drafter NHD->HND logs suppressed (> 12)"
                    )
                continue

            # Neither axis 0 nor axis 1 has size 2 — unexpected; fail fast.
            raise RuntimeError(
                f"[G4_74] drafter layer {layer_name!r} has 5-D shape "
                f"{shape} with neither shape[0]==2 nor shape[1]==2; "
                f"stride={stride_before} dtype={kv_cache.dtype} "
                f"contig={contig_before}. FlashAttn expects "
                f"shape[0]==2 (HND, k/v stack at axis 0). Cannot "
                f"determine intended axis for transpose. Disable G4_74 "
                f"and investigate the allocator."
            )

        return kv_caches

    _wrapped_reshape._genesis_g4_74_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner._reshape_kv_cache_tensors = _wrapped_reshape  # type: ignore[method-assign]
    _APPLIED = True

    log.warning(
        "[G4_74] INSTALLED: GPUModelRunner._reshape_kv_cache_tensors wrapped; "
        "drafter layers (prefix=%r) with shape[1]==2 will be transposed to "
        "HND layout before bind_kv_cache.",
        drafter_prefix,
    )
    return "applied", (
        f"G4_74 installed: drafter (prefix {drafter_prefix!r}) kv_cache "
        f"NHD->HND post-reshape rebinding active."
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

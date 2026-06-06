# SPDX-License-Identifier: Apache-2.0
"""G4_69 — Per-layer native attention backend for skip-listed TurboQuant layers.

================================================================
PROBLEM
================================================================

Gemma 4 + TurboQuant + MTP correctness fallback (P65 + PN256 + G4_68)
fixes target output but leaves drafter acceptance at 0%. PN258 SELF-
ORACLE causally proved rejection sampler + target verifier rows 0..K-1
are wired correctly. The non-TQ MTP control (H8 Option 3) showed
drafter recovers ~62% acceptance on the same prompt when KV cache is
NOT TurboQuant-compressed. Conclusion: drafter degrades because it
reads KV-sharing target layers (58, 59 on Gemma 4 31B AWQ) whose K/V
are TurboQuant-compressed and quantization noise corrupts drafter's
context.

The intended remedy is to exclude exactly those layers from TQ via
`cache_config.kv_cache_dtype_skip_layers`. G4_60K's PN247 hook already
plumbs `GENESIS_G4_TQ_FORCE_SKIP_LAYERS=58,59` into the cache config,
and G4_60G correctly returns native `FullAttentionSpec` for those
layers. The cache-spec side of the route is fine.

The remaining route bug is at the ATTENTION-IMPL dispatch:

  1. Launcher passes `--attention-backend TURBOQUANT`.
  2. `vllm_config.attention_config.backend = TURBOQUANT` for the whole
     model.
  3. `Attention.__init__` for every layer calls `get_attn_backend(
     head_size, dtype, kv_cache_dtype, ...)` where `kv_cache_dtype` is
     per-layer ("turboquant_4bit_nc" for non-skipped, "auto" for
     skipped).
  4. `_cached_get_attn_backend(backend=TURBOQUANT, attn_selector_config
     =...)` dispatches to `CudaPlatform.get_attn_backend_cls(
     selected_backend=TURBOQUANT, ...)`.
  5. With `selected_backend != None`, the dispatcher requires
     validation to pass and returns the selected backend unconditionally
     — no per-layer fall-through.
  6. G4_32 bypasses `TurboQuantAttentionBackend.validate_configuration`
     unconditionally for legitimate reasons (kv_cache_dtype coercion
     bug on this pin), so even "auto"-dtype skipped layers validate
     successfully.
  7. The dispatcher returns TURBOQUANT for skipped layers too.
  8. `TurboQuantAttentionImpl.__init__` runs for those layers with
     `kv_cache_dtype="auto"` and crashes at
     `TurboQuantConfig.from_cache_dtype("auto", head_size)` with
     `ValueError: Unknown TurboQuant cache dtype: 'auto'`.

================================================================
FIX
================================================================

Wrap `CudaPlatform.get_attn_backend_cls` at the dispatch boundary.
When the call presents `selected_backend = TURBOQUANT` AND
`attn_selector_config.kv_cache_dtype = "auto"`, the dispatcher MUST
NOT return TURBOQUANT for that layer. The clean substitution is to
clear `selected_backend = None` for just that call, which lets the
existing auto-priority list (FLASH_ATTN -> FLASHINFER -> ...) pick the
right native backend for the skip-listed layer. The same dispatcher
call for non-skipped layers (kv_cache_dtype="turboquant_*") is
unaffected and TURBOQUANT continues to be selected for them.

The fix is route-correct (matches the overlay's stated intent that
"layers that fall back to native dtype via kv_cache_dtype_skip_layers
get their own standard-shaped cache allocation"), capture-safe (boot
only), and quality-preserving (G4_32's TQ-dtype validation bypass is
not touched; only the "auto" branch is rerouted).

================================================================
INTERACTION WITH OTHER PATCHES
================================================================

  * G4_32 stays installed. It bypasses validation for legitimate
    TQ dtype dispatch.
  * G4_60K stays installed. It populates
    `cache_config.kv_cache_dtype_skip_layers` from
    `GENESIS_G4_TQ_FORCE_SKIP_LAYERS`.
  * G4_60G stays installed. It returns native `FullAttentionSpec` for
    skip-listed layers' KV cache shape.
  * G4_60E stays installed for hybrid TQ/native KV cache handling.
  * P65 + PN256 + G4_68 (correctness fallback) remain operational for
    layers that ARE TQ-quantized.

The boot log will show:
  `[G4_69] dispatched layer with kv_cache_dtype=auto away from
   TurboQuant backend (selected_backend cleared for fall-through)`

================================================================
ENV FLAG
================================================================

  GENESIS_ENABLE_G4_69_SKIP_LAYERS_NATIVE_BACKEND=1   (opt-in)

When unset: original behavior (Attention.__init__ would crash for
skipped layers under --attention-backend TURBOQUANT).

When set without `cache_config.kv_cache_dtype_skip_layers`: no-op.
The wrap only activates when both TURBOQUANT is selected AND the
per-layer kv_cache_dtype is "auto".

================================================================
ACCEPTANCE GATE
================================================================

Boot succeeds with `GENESIS_G4_TQ_FORCE_SKIP_LAYERS=58,59` AND
`--attention-backend TURBOQUANT`. Log shows G4_69 fired for at least
the listed layers, and TurboQuant runtime path is taken for all
other layers. Subsequent prompt run shows `accepted_per_req > 0`
under PN248 trace.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.turboquant.g4_69_skip_layers_native_backend")

GENESIS_G4_69_MARKER = (
    "Genesis G4_69 per-layer native attention backend for skip-listed "
    "TurboQuant layers (substitutes selected_backend=None when "
    "kv_cache_dtype='auto' under --attention-backend TURBOQUANT)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_69_SKIP_LAYERS_NATIVE_BACKEND"
_APPLIED = False
_ORIGINAL = None
_REROUTE_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Install per-layer backend reroute on CudaPlatform.get_attn_backend_cls."""
    global _APPLIED, _ORIGINAL

    if not _env_enabled():
        return "skipped", (
            f"G4_69 disabled (set {_ENV_ENABLE}=1 to enable per-layer "
            f"native attention backend dispatch for skip-listed layers)"
        )

    if _APPLIED:
        return "applied", "G4_69 already installed (idempotent)"

    try:
        from vllm.platforms.cuda import CudaPlatformBase
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
    except ImportError as e:
        return "skipped", (
            f"vllm.platforms.cuda or v1.attention.backends.registry not "
            f"importable: {e}; G4_69 is no-op on this pin"
        )

    original = CudaPlatformBase.get_attn_backend_cls
    if getattr(original, "_genesis_g4_69_wrapped", False):
        _APPLIED = True
        return "applied", "G4_69 already wrapped (idempotent)"
    _ORIGINAL = original

    def _wrapped_get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads=None,
    ):
        """Route non-TQ-dtype layers away from TURBOQUANT backend.

        The intent: any layer whose kv_cache_dtype does NOT start with
        "turboquant_" cannot be served by the TurboQuant Triton decode
        kernel — it would walk out-of-bounds on a native KV cache. So
        for those layers the global `--attention-backend TURBOQUANT`
        MUST be substituted by `selected_backend=None`, letting the
        dispatcher's auto-priority list pick a native backend
        (FLASH_ATTN on Ampere with bf16 + head_size 256/512).

        Three cases that match the reroute:
          - `kv_cache_dtype == "auto"`  — target skip-listed layers
            (set by Attention.__init__ via
            `cache_config.kv_cache_dtype_skip_layers`).
          - `kv_cache_dtype is None`    — drafter / spec-decode layers
            whose loader did not propagate the TQ dtype.
          - `kv_cache_dtype` is a non-TQ string (e.g. "fp8") — defensive.

        Non-TQ branches:
          - skip when `selected_backend != AttentionBackendEnum.TURBOQUANT`
            (caller didn't ask for TQ — nothing to substitute).
        TQ branches (kv_cache_dtype starts with "turboquant_"):
          - keep TURBOQUANT — the kernel is appropriate.

        PN261-C extended the original gate (`auto` only) to also catch
        drafter Attention objects that go through this dispatcher with
        a non-TQ kv_cache_dtype. PN260 trace proved drafter's
        TurboQuantAttentionImpl was launching the TQ decode kernel on
        a native bf16 KV cache, producing CUDA illegal address.
        """
        _kv_dt = getattr(attn_selector_config, "kv_cache_dtype", None)
        _is_tq_dtype = isinstance(_kv_dt, str) and _kv_dt.startswith(
            "turboquant_"
        )
        if (
            selected_backend == AttentionBackendEnum.TURBOQUANT
            and not _is_tq_dtype
        ):
            _REROUTE_COUNT[0] += 1
            if _REROUTE_COUNT[0] <= 8:
                # Log first few rerouted layers (avoid log flood).
                log.warning(
                    "[G4_69] non-TQ kv_cache_dtype detected with "
                    "selected_backend=TURBOQUANT — clearing selected "
                    "backend for this layer so auto-priority dispatcher "
                    "picks a native attention backend. (call #%d, "
                    "kv_cache_dtype=%r, head_size=%s, num_heads=%s)",
                    _REROUTE_COUNT[0],
                    _kv_dt,
                    getattr(attn_selector_config, "head_size", "?"),
                    num_heads,
                )
            elif _REROUTE_COUNT[0] == 9:
                log.warning(
                    "[G4_69] further reroute logs suppressed (count > 8)"
                )
            selected_backend = None

        return original.__func__(
            cls,
            selected_backend,
            attn_selector_config,
            num_heads=num_heads,
        )

    _wrapped_get_attn_backend_cls._genesis_g4_69_wrapped = True  # type: ignore[attr-defined]
    CudaPlatformBase.get_attn_backend_cls = classmethod(
        _wrapped_get_attn_backend_cls
    )
    _APPLIED = True

    log.info(
        "[G4_69] installed: CudaPlatformBase.get_attn_backend_cls now "
        "reroutes kv_cache_dtype='auto' away from explicit TURBOQUANT "
        "selection. Skip-listed layers will get native attention impl."
    )
    return "applied", (
        "G4_69 installed: per-layer native backend dispatch for "
        "skip-listed TurboQuant layers. Reroute count will be logged "
        "for first 4 dispatch calls."
    )


def is_applied() -> bool:
    return _APPLIED


def reroute_count() -> int:
    return _REROUTE_COUNT[0]


def revert() -> bool:
    global _APPLIED, _ORIGINAL
    if not _APPLIED or _ORIGINAL is None:
        return False
    try:
        from vllm.platforms.cuda import CudaPlatformBase
        CudaPlatformBase.get_attn_backend_cls = _ORIGINAL
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL = None
    return True


__all__ = [
    "GENESIS_G4_69_MARKER",
    "apply",
    "is_applied",
    "reroute_count",
    "revert",
]

# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 39a — FLA `chunk_scaled_dot_kkt_fwd` persistent A pool.

Replaces per-call `torch.empty(B, T, H, BT, fp32)` with a persistent
pool via `FlaKktBufferManager.acquire`. Monkey-patches
`vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt.chunk_scaled_dot_kkt_fwd`
at module level.

Rationale: the GDN chunked-prefill path inside the AOT-compiled model
calls this function once per GDN-bearing layer per chunk. Each alloc is
16 MiB on our config (Qwen3.6 B=1 T≤4096 H=16 BT=64 fp32) but the
N_layers-fold churn saturates the allocator at the yaml=0.93/0.94
boundary with dev134 memory accounting. A single shared pool removes
the churn entirely.

Compatibility
-------------
- NVIDIA CUDA SM 8.0+: wiring applied.
- AMD / CPU / pre-Ampere: wiring skipped, fallback in manager
  (`acquire` returns fresh `torch.empty` when `should_apply()` is
  False).
- Upstream drift: this module's `apply()` dry-imports the target and
  logs skip if the symbol isn't present (future rename or removal).

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
Status: v7.3 implementation
"""

# Legacy auto-apply note (audit 2026-05-11): registry env_flag
# `GENESIS_LEGACY_P39A` is synthetic — flag exists for registry/audit
# coherence but has no runtime effect. Patch applies unconditionally
# via dispatcher's legacy auto-apply path (`is_legacy_active` in
# vllm/sndr_core/dispatcher/decision.py). See registry.py "Legacy
# patches" section (~line 2083) for full context.

from __future__ import annotations

import logging
from typing import Any

from vllm.sndr_core.detection.guards import is_nvidia_cuda, is_sm_at_least
# v11.1.0 P3.3: surface the FLA KKT persistent A pool through
# PersistentBufferRegistry so operators can `sndr patches show
# buffer_registry` and see this pool listed. Byte-equivalent — the
# actual torch.empty() still happens inside FlaKktBufferManager.acquire
# (allocate-once-keep-forever, pointer-stable via the
# reserve-before-cudagraph pattern). The registry hook only exposes
# the pool name; tensor storage ownership is unchanged.
from vllm.sndr_core.runtime.persistent_buffer_registry import (
    PersistentBufferRegistry,
    POOL_FLA_KKT_PERSISTENT_A,
)

log = logging.getLogger("genesis.wiring.p39a_fla_kkt")

_GENESIS_P39A_MARKER_ATTR = "_genesis_p39a_wrapped"


def ensure_pool_registered() -> None:
    """Idempotent registry hook — exposes POOL_FLA_KKT_PERSISTENT_A in
    PersistentBufferRegistry for operator visibility. No allocation,
    no behavior change.

    The real FLA KKT `A` tensor (B, T, H, BT, fp32) is owned by
    vllm.sndr_core.kernels.fla_kkt_buffer.FlaKktBufferManager via the
    reserve-before-cudagraph pattern (P39b). Its allocation semantics
    are GROW-IN-PLACE + SLICE-ON-ACQUIRE keyed by (H, BT, device, dtype)
    — variable first two dims (B, T) → fixed last two — which matches
    PersistentSlicePool exactly.

    v11.3.0 bug fix: this was previously calling `get_pool()` which
    creates a BufferPool (free-list acquire/release semantics, wrong
    pool type for P39a). Switched to `get_slice_pool()` which matches
    the actual allocation pattern. Operator-visibility only — does
    not change any runtime allocation behavior; the actual storage
    still lives in FlaKktBufferManager via GPB.
    """
    PersistentBufferRegistry().get_slice_pool(POOL_FLA_KKT_PERSISTENT_A)

# Module paths we target. Primary + candidates for future renames.
_CANDIDATE_MODULE_PATHS = (
    "vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt",
)
_FN_NAME = "chunk_scaled_dot_kkt_fwd"


def should_apply() -> bool:
    if not is_nvidia_cuda():
        return False
    if not is_sm_at_least(8, 0):
        return False
    return True


def _import_target() -> tuple[Any, Any] | None:
    """Return (module, original_fn) or None on failure."""
    import importlib
    for modpath in _CANDIDATE_MODULE_PATHS:
        try:
            mod = importlib.import_module(modpath)
        except ImportError:
            continue
        except Exception as e:
            log.warning("[Genesis P39a] import %s: %s", modpath, e)
            continue
        fn = getattr(mod, _FN_NAME, None)
        if fn is not None:
            return mod, fn
    return None


def apply() -> tuple[str, str]:
    """Rebind `chunk_scaled_dot_kkt_fwd` to the pooled version.

    Never raises. Returns (status, reason).
    """
    if not should_apply():
        return "skipped", "platform: NVIDIA SM 8.0+ required"

    # P53 (v7.9): Hybrid-active dispatch gate. chunk_scaled_dot_kkt_fwd is
    # FLA-GDN only. Pure-attention models may not even have the FLA module
    # imported — the target-import check below would skip, but we log
    # the dispatch reason up-front.
    try:
        from vllm.sndr_core.detection.model_detect import is_hybrid_model, log_skip
        if not is_hybrid_model():
            log_skip(
                "P39a FLA chunk_scaled_dot_kkt pool",
                "pure-attention model (no GDN chunked-prefill)",
            )
            return "skipped", "P53 dispatch: model has no hybrid linear-attention layers"
    except Exception as e:
        log.debug("[Genesis P39a] model_detect probe failed (proceeding): %s", e)

    target = _import_target()
    if target is None:
        return "skipped", (
            f"FLA module {_CANDIDATE_MODULE_PATHS[0]!r} or symbol "
            f"{_FN_NAME!r} not available (not an FLA-GDN build)"
        )
    mod, original = target

    # P49 interface contract check (v7.8): our replacement calls
    # `mod.chunk_scaled_dot_kkt_fwd_kernel`, `mod.FLA_CHUNK_SIZE`,
    # and `mod.prepare_chunk_indices`. If upstream renamed ANY of
    # these, we bail rather than calling into a missing symbol at
    # first forward.
    #
    # Note: Triton `@triton.jit`-decorated kernels are `JITFunction`
    # instances that are NOT `callable()` in the Python sense (you
    # invoke via `kernel[grid](*args)`). So for the kernel symbol we
    # use `required_attrs={...: ANY}` (presence check) instead of
    # `required_methods` (callable check). For `chunk_scaled_dot_kkt_fwd`
    # (the regular Python wrapper) and `prepare_chunk_indices` (also
    # plain Python), `required_methods` works fine.
    try:
        from vllm.sndr_core.runtime.interface_guard import (
            validate_impl, ANY,
        )
        validate_impl(
            mod,
            role="FLA chunk_scaled_dot_kkt module (P39a)",
            required_attrs={
                "chunk_scaled_dot_kkt_fwd_kernel": ANY,  # Triton JIT
                "FLA_CHUNK_SIZE": int,
            },
            required_methods=[
                "chunk_scaled_dot_kkt_fwd",
                "prepare_chunk_indices",
            ],
        )
    except Exception as e:
        if "GenesisInterfaceMismatch" in type(e).__name__:
            return "skipped", f"P49 interface drift: {e}"

    if getattr(original, _GENESIS_P39A_MARKER_ATTR, False):
        return "applied", "already wrapped (idempotent)"

    try:
        from vllm.sndr_core.kernels.fla_kkt_buffer import FlaKktBufferManager
    except Exception as e:
        return "failed", f"kernel import failed: {e}"

    # v11.1.0 P3.3: expose the pool name in the registry — no allocation,
    # purely operator-visibility surface.
    try:
        ensure_pool_registered()
    except Exception as e:
        log.debug("[P39a] registry pool registration failed (proceeding): %s", e)

    # P39b: resolve `max_num_batched_tokens` + `max_num_seqs` ONCE at
    # apply time so the pool can be grown to its final size on the very
    # first call (profiler-visible, pointer-stable — no CUDA-graph
    # invalidation from later pool-swap on growth). This is the
    # reserve-before-cudagraph pattern from upstream PR #40798,
    # adapted to our class-based manager.
    _MAX_T_HINT: list[int] = [4096]  # mutable box (closure-captured)
    _MAX_B_HINT: list[int] = [2]     # mutable box
    try:
        from vllm.config import get_current_vllm_config
        _cfg = get_current_vllm_config()
        _scheduler_cfg = getattr(_cfg, "scheduler_config", None)
        if _scheduler_cfg is not None:
            _mb = getattr(_scheduler_cfg, "max_num_batched_tokens", None)
            _ms = getattr(_scheduler_cfg, "max_num_seqs", None)
            if _mb:
                _MAX_T_HINT[0] = int(_mb)
            if _ms:
                _MAX_B_HINT[0] = int(_ms)
            log.info(
                "[Genesis P39b] resolved warmup hints max_T=%d max_B=%d "
                "from current vllm_config (pool will grow to these sizes "
                "on first call → no pointer-swap later)",
                _MAX_T_HINT[0], _MAX_B_HINT[0],
            )
        else:
            log.info(
                "[Genesis P39b] vllm_config unavailable at apply time — "
                "falling back to defaults (max_T=%d max_B=%d). Pool may "
                "pointer-swap once on first large call.",
                _MAX_T_HINT[0], _MAX_B_HINT[0],
            )
    except Exception as e:
        # Non-fatal — pool will auto-grow lazily as before.
        log.info(
            "[Genesis P39b] vllm_config fetch failed (%s); defaults used",
            e,
        )

    # Env override — operators can force a specific max if they know
    # their config better than the auto-detection.
    import os
    _env_t = os.environ.get("GENESIS_FLA_KKT_MAX_T", "")
    if _env_t.isdigit() and int(_env_t) > 0:
        _MAX_T_HINT[0] = int(_env_t)
        log.info(
            "[Genesis P39b] GENESIS_FLA_KKT_MAX_T env override → max_T=%d",
            _MAX_T_HINT[0],
        )
    _env_b = os.environ.get("GENESIS_FLA_KKT_MAX_B", "")
    if _env_b.isdigit() and int(_env_b) > 0:
        _MAX_B_HINT[0] = int(_env_b)
        log.info(
            "[Genesis P39b] GENESIS_FLA_KKT_MAX_B env override → max_B=%d",
            _MAX_B_HINT[0],
        )

    def _genesis_pooled_chunk_scaled_dot_kkt_fwd(
        k,
        g=None,
        beta=None,
        cu_seqlens=None,
        chunk_indices=None,
        chunk_size=None,
        output_dtype=None,
    ):
        """Signature-compatible drop-in around the original.

        Replaces the `A = torch.empty(B, T, H, BT, ...)` line with a
        pooled acquire + same Triton kernel call. Everything else
        (heuristics, autotune, store layout) is untouched because we
        pass a same-shape same-stride view.

        P39b (reserve-before-cudagraph): `max_T` and `max_B` are passed
        on every call so the pool grows to its final size on the FIRST
        call (typically at profile_run with small batch) — afterwards
        all calls reuse the same buffer pointer, eliminating any risk
        of CUDA-graph invalidation from pool pointer-swap on growth.
        """
        import triton
        import torch

        # Resolve defaults by asking the module (in case upstream bumps
        # FLA_CHUNK_SIZE or changes output dtype default).
        if chunk_size is None:
            try:
                chunk_size = mod.FLA_CHUNK_SIZE
            except AttributeError:
                chunk_size = 64
        if output_dtype is None:
            output_dtype = torch.float32

        B, T, Hg, K = k.shape
        H = beta.shape[-1]
        BT = chunk_size
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = mod.prepare_chunk_indices(cu_seqlens, BT)
        NT = (
            triton.cdiv(T, BT)
            if cu_seqlens is None
            else len(chunk_indices)
        )

        # POOLED acquire — P39a core + P39b pre-sizing hints
        A = FlaKktBufferManager.acquire(
            B=B, T=T, H=H, BT=BT,
            device=k.device, dtype=output_dtype,
            max_T=_MAX_T_HINT[0],
            max_B=_MAX_B_HINT[0],
        )

        mod.chunk_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
            k=k,
            g=g,
            beta=beta,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            Hg=Hg,
            K=K,
            BT=BT,
        )
        return A

    # Marker + preserve the original so revert can restore it.
    setattr(
        _genesis_pooled_chunk_scaled_dot_kkt_fwd,
        _GENESIS_P39A_MARKER_ATTR, True,
    )
    setattr(
        _genesis_pooled_chunk_scaled_dot_kkt_fwd,
        "_genesis_p39a_original", original,
    )

    setattr(mod, _FN_NAME, _genesis_pooled_chunk_scaled_dot_kkt_fwd)

    # ALSO rebind on any already-imported callers. FLA internal code
    # typically imports via `from .chunk_scaled_dot_kkt import
    # chunk_scaled_dot_kkt_fwd` — those modules will retain the ORIGINAL
    # reference. To fix, we walk the chunk_delta_h importer.
    # However: callers inside the AOT-compiled model path resolve the
    # symbol from the `mod` namespace at call time when accessed as
    # attribute. Most FLA internal calls DO `from ... import ...` →
    # they capture the original. To cover both, we also rebind the
    # symbol inside `vllm.model_executor.layers.fla.ops.chunk_delta_h`
    # and siblings if they imported it.
    import sys as _sys
    rebound_callers = []
    fla_ops_prefix = "vllm.model_executor.layers.fla.ops"
    for mod_name, caller_mod in list(_sys.modules.items()):
        if caller_mod is None:
            continue
        if not mod_name.startswith(fla_ops_prefix):
            continue
        if mod_name == _CANDIDATE_MODULE_PATHS[0]:
            continue
        existing = getattr(caller_mod, _FN_NAME, None)
        if existing is original:
            try:
                setattr(
                    caller_mod, _FN_NAME,
                    _genesis_pooled_chunk_scaled_dot_kkt_fwd,
                )
                rebound_callers.append(mod_name)
            except Exception as e:
                log.debug(
                    "[Genesis P39a] couldn't rebind in %s: %s",
                    mod_name, e,
                )

    log.info(
        "[Genesis P39a] rebound %s.%s (+%d caller mods: %s)",
        _CANDIDATE_MODULE_PATHS[0], _FN_NAME,
        len(rebound_callers), rebound_callers,
    )
    return "applied", (
        f"module-level fn replaced ({len(rebound_callers)} caller "
        f"module(s) also rebound — pool shared across GDN layers)"
    )


def is_applied() -> bool:
    target = _import_target()
    if target is None:
        return False
    _mod, fn = target
    return getattr(fn, _GENESIS_P39A_MARKER_ATTR, False)


def revert() -> bool:
    """Restore the original function. For tests only."""
    target = _import_target()
    if target is None:
        return False
    mod, fn = target
    if not getattr(fn, _GENESIS_P39A_MARKER_ATTR, False):
        return False
    original = getattr(fn, "_genesis_p39a_original", None)
    if original is None:
        return False
    setattr(mod, _FN_NAME, original)
    # Restore in caller mods too
    import sys as _sys
    for mod_name, caller_mod in _sys.modules.items():
        if caller_mod is None or mod_name == _CANDIDATE_MODULE_PATHS[0]:
            continue
        existing = getattr(caller_mod, _FN_NAME, None)
        if existing is fn:
            setattr(caller_mod, _FN_NAME, original)
    return True

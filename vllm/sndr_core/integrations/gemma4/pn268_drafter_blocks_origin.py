# SPDX-License-Identifier: Apache-2.0
"""PN268 — Drafter num_blocks origin trace (G4_78-CAP-0 probe).

================================================================
PROBLEM
================================================================

PN267 trace showed drafter[0..2] kv_cache has only 4 blocks at
runtime — way below the sliding window minimum (32 blocks for
1024-token visible context). But target[58]/[59] have 5354 blocks.
With G4_76 disabling kv_sharing, drafter is supposed to be a fully
independent group in `kv_cache_config.kv_cache_groups`, and the KV
planner should give it a fair share of memory.

We need to know WHERE the 4-block figure comes from:

  (a) profile_cudagraph_memory's minimal_config (min_blocks = 4 ≈
      max_cudagraph_capture_size) — and the real-config call NEVER
      re-allocates drafter, so the profile-time tensor lingers
  (b) KV planner gives drafter only 4 blocks at real config (e.g.,
      because drafter's group has tiny page_size_padded or unusual
      allocation rule)
  (c) G4_74 transpose pivots on the profile-time tensor and
      semantically downgrades the real-config one
  (d) Drafter is not in real config's kv_cache_groups at all (gets
      skipped by allocator entirely after profile cleanup)

PN262-B captured one snapshot but didn't trace EVERY call. PN268
fixes that.

================================================================
TRACE STRATEGY
================================================================

Wrap `GPUModelRunner.initialize_kv_cache_tensors`. At every call:

  Pre-call:
    - call_idx (1, 2, ...)
    - kv_cache_config.kv_cache_groups: per group (gid, spec class,
      spec.num_kv_heads, spec.head_size, spec.block_size,
      spec.dtype, page_size_bytes, layer_names with drafter flag)
    - kv_cache_config.kv_cache_tensors: per tensor (size, shared_by)
    - kv_cache_config.num_blocks

  Post-call (after bind):
    - for each drafter Attention in static_forward_context:
      - kv_cache shape/stride/dtype/contig/data_ptr
      - inferred num_blocks (shape[1] for HND, shape[0] for NHD)

This runs UNCONDITIONALLY on every call (profile minimal, real
config, etc.) so we can see what changes between calls.

================================================================
ENV
================================================================

  GENESIS_ENABLE_PN268_DRAFTER_BLOCKS_TRACE=1

================================================================
ACCEPTANCE
================================================================

After one container boot, the log answers:

  Q1. Is drafter included in real-config kv_cache_config.kv_cache_groups?
  Q2. What num_blocks does the planner request for the drafter group?
  Q3. What's the actual drafter shape AFTER real-config bind?
  Q4. Is the profile→real config transition correctly re-allocating
      drafter, or is it stuck on the profile-minimal tensor?

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("genesis.gemma4.pn268_drafter_blocks_origin")

GENESIS_PN268_MARKER = "Genesis PN268 drafter num_blocks origin trace"

_ENV_ENABLE = "GENESIS_ENABLE_PN268_DRAFTER_BLOCKS_TRACE"
_APPLIED = False
_ORIGINAL_INIT_TENSORS = None
_CALL_COUNT = [0]

DRAFTER_PREFIX = "draft_model."


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _safe(value: Any, default: str = "<?>") -> str:
    try:
        return repr(value)
    except Exception:
        return default


def _spec_summary(spec: Any) -> str:
    if spec is None:
        return "<None>"
    parts = []
    parts.append(f"class={type(spec).__qualname__}")
    for attr in ("num_kv_heads", "head_size", "head_size_v", "block_size",
                 "dtype", "page_size_bytes", "sliding_window",
                 "page_size_padded", "kv_quant_mode"):
        if hasattr(spec, attr):
            try:
                v = getattr(spec, attr)
                parts.append(f"{attr}={v}")
            except Exception:
                parts.append(f"{attr}=<err>")
    return " ".join(parts)


def apply() -> tuple[str, str]:
    global _APPLIED, _ORIGINAL_INIT_TENSORS

    if not _env_enabled():
        return "skipped", f"PN268 disabled (set {_ENV_ENABLE}=1)"

    if _APPLIED:
        return "applied", "PN268 already installed"

    log.warning("[PN268] apply() entered")

    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001
        log.warning("[PN268] SKIP: GPUModelRunner not importable: %s", e)
        return "skipped", f"GPUModelRunner not importable: {e!r}"

    if not hasattr(GPUModelRunner, "initialize_kv_cache_tensors"):
        return "skipped", "GPUModelRunner.initialize_kv_cache_tensors missing"

    original = GPUModelRunner.initialize_kv_cache_tensors
    if getattr(original, "_genesis_pn268_wrapped", False):
        _APPLIED = True
        return "applied", "initialize_kv_cache_tensors already wrapped"
    _ORIGINAL_INIT_TENSORS = original

    def _wrapped(self, kv_cache_config, kernel_block_sizes):
        _CALL_COUNT[0] += 1
        call_idx = _CALL_COUNT[0]

        # --- Pre-call dump
        try:
            groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
            tensors = getattr(kv_cache_config, "kv_cache_tensors", None) or []
            num_blocks = getattr(kv_cache_config, "num_blocks", "?")
            log.warning(
                "[PN268/pre call#%d] kv_cache_config.num_blocks=%s "
                "n_groups=%d n_tensors=%d kernel_block_sizes=%s",
                call_idx, num_blocks, len(groups), len(tensors),
                _safe(kernel_block_sizes),
            )
            for gid, grp in enumerate(groups):
                spec = getattr(grp, "kv_cache_spec", None)
                layer_names = list(getattr(grp, "layer_names", []) or [])
                drafter_count = sum(
                    1 for n in layer_names
                    if isinstance(n, str) and n.startswith(DRAFTER_PREFIX)
                )
                log.warning(
                    "[PN268/pre call#%d] group gid=%d %s n_layers=%d "
                    "drafter_count=%d sample_layers=%s",
                    call_idx, gid, _spec_summary(spec), len(layer_names),
                    drafter_count, layer_names[:3] + (
                        ["..."] + layer_names[-2:] if len(layer_names) > 5 else []
                    ),
                )
            for ti, t in enumerate(tensors):
                size = getattr(t, "size", "?")
                shared_by = list(getattr(t, "shared_by", []) or [])
                drafter_in_tensor = any(
                    isinstance(n, str) and n.startswith(DRAFTER_PREFIX)
                    for n in shared_by
                )
                log.warning(
                    "[PN268/pre call#%d] tensor[%d] size=%s drafter_in_shared=%s "
                    "shared_by=%s",
                    call_idx, ti, size, drafter_in_tensor,
                    shared_by[:4] + (
                        ["..."] + shared_by[-2:] if len(shared_by) > 6 else []
                    ),
                )
        except Exception as _e:  # noqa: BLE001
            log.warning("[PN268/pre call#%d] introspection failed: %s",
                        call_idx, _e)

        # --- Call original
        result = original(self, kv_cache_config, kernel_block_sizes)

        # --- Post-call dump of drafter Attention kv_caches
        try:
            fwd_ctx = self.compilation_config.static_forward_context
            drafter_items = [
                (n, a) for n, a in fwd_ctx.items()
                if isinstance(n, str) and n.startswith(DRAFTER_PREFIX)
            ]
            log.warning(
                "[PN268/post call#%d] n_drafter_attns=%d",
                call_idx, len(drafter_items),
            )
            for name, attn in sorted(drafter_items):
                kv = getattr(attn, "kv_cache", None)
                if kv is None:
                    log.warning(
                        "[PN268/post call#%d] %s: kv_cache=<None>",
                        call_idx, name,
                    )
                    continue
                try:
                    shape = tuple(kv.shape)
                    stride = tuple(kv.stride())
                    inferred_num_blocks = (
                        shape[1] if (len(shape) >= 2 and shape[0] == 2)
                        else shape[0] if len(shape) >= 1 else "?"
                    )
                    log.warning(
                        "[PN268/post call#%d] %s: shape=%s stride=%s "
                        "dtype=%s contig=%s ndim=%d numel=%d "
                        "inferred_num_blocks=%s data_ptr=0x%x",
                        call_idx, name, shape, stride, kv.dtype,
                        bool(kv.is_contiguous()), int(kv.dim()),
                        int(kv.numel()), inferred_num_blocks,
                        int(kv.data_ptr()),
                    )
                except Exception as _e:  # noqa: BLE001
                    log.warning(
                        "[PN268/post call#%d] %s: introspection failed: %s",
                        call_idx, name, _e,
                    )
        except Exception as _e:  # noqa: BLE001
            log.warning("[PN268/post call#%d] outer introspection failed: %s",
                        call_idx, _e)

        return result

    _wrapped._genesis_pn268_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner.initialize_kv_cache_tensors = _wrapped  # type: ignore[method-assign]
    _APPLIED = True
    log.warning(
        "[PN268] INSTALLED: GPUModelRunner.initialize_kv_cache_tensors wrapped — "
        "every call logs kv_cache_config groups/tensors + drafter "
        "post-bind shapes."
    )
    return "applied", "PN268 installed (trace-only)"


def is_applied() -> bool:
    return _APPLIED


def revert() -> bool:
    global _APPLIED, _ORIGINAL_INIT_TENSORS
    if not _APPLIED or _ORIGINAL_INIT_TENSORS is None:
        return False
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        GPUModelRunner.initialize_kv_cache_tensors = _ORIGINAL_INIT_TENSORS  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_INIT_TENSORS = None
    return True


__all__ = ["GENESIS_PN268_MARKER", "apply", "is_applied", "revert"]

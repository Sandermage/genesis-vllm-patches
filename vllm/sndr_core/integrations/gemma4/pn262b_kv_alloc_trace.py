# SPDX-License-Identifier: Apache-2.0
"""PN262-B — KV cache allocator/reshape/proposer-init diagnostic trace.

================================================================
PROBLEM (PN262-A finding)
================================================================

PN262-A fail-fast at FlashAttn.forward showed::

    shape=(8192, 8, 256)  stride=(2048, 256, 1)  dtype=torch.bfloat16
    contiguous=True       ndim=3                 kv_sharing=None
    VLLM_KV_CACHE_LAYOUT='<unset>'

Identical with PN259c ON or OFF. So:

  * Not a view/transpose (contiguous=True).
  * Not aliasing (kv_sharing=None).
  * Not the global layout env.
  * Not PN259c (A/B identical).

The wrong shape comes from ``attn_backend.get_kv_cache_shape()`` in
``GpuModelRunner._reshape_kv_cache_tensors`` (line ~6846) — where the
``attn_backend`` is the GROUP's backend, not per-layer. Even though
G4_71 forced FlashAttn impl per-layer and G4_72 forced native spec
per-layer, the *AttentionGroup* into which drafter layers were sorted
at config-build time may still carry a TurboQuant backend AND/OR a
TQ-flavored ``kv_cache_spec``, and that group-level state drives the
physical shape.

SpecDecodeBaseProposer.initialize_attn_backend (line ~1535) adds a second
risk: it iterates ``self._draft_attn_layer_names``, calls
``all_attn_layers[layer_name].get_attn_backend()`` (per-layer, honors
G4_71), but uses ``layer_kv_cache_spec = kv_cache_spec`` from the
GROUP — only doing per-layer lookup when the group spec is a
``UniformTypeKVCacheSpecs`` instance. If drafter's group spec is the
old TQ spec (or a plain non-uniform TQ spec), the metadata builder is
created with the wrong contract.

================================================================
FIX (DIAGNOSTIC ONLY)
================================================================

Wraps two functions and logs the disambiguating state. No behaviour
change.

  1. ``GpuModelRunner._reshape_kv_cache_tensors`` — pre-call: iterate
     ``self.kv_cache_config.kv_cache_groups`` and dump each group's
     id / layer_names / spec class / backend class. Post-call: for
     every drafter layer in the returned ``kv_caches`` dict, log
     final ``shape``/``stride``/``dtype``/``is_contiguous``/``data_ptr``.

  2. ``SpecDecodeBaseProposer.initialize_attn_backend`` — pre-call: log the
     selected ``kv_cache_gid``, group spec class, group layer_names.
     Post-call: log each ``AttentionGroup`` in ``self.draft_attn_groups``
     with backend.full_cls_name(), kv_cache_spec class,
     layer_names, kv_cache_group_id.

Together these cover all four spots the wrong shape can originate.

================================================================
ENV FLAGS
================================================================

  GENESIS_ENABLE_PN262B_KV_ALLOC_TRACE=1   (opt-in)
  GENESIS_PN262B_PREFIX=draft_model.       (drafter prefix filter)

================================================================
ACCEPTANCE
================================================================

  Gate: K=2 boot with PN262-A trace OFF (or fail-fast OFF), PN262-B
  ON. Expected output:

    [PN262-B/reshape:groups] gid=0 backend=<...> spec=<...> layers=[...]
    [PN262-B/reshape:groups] gid=1 backend=<...> spec=<...> layers=[...]
    ...
    [PN262-B/reshape:drafter] layer=draft_model.layers.0.... shape=(...)
    [PN262-B/proposer:init] kv_cache_gid=<int> group_spec=<class>
                            group_layers=[...]
    [PN262-B/proposer:groups] backend_full=<...> spec=<...>
                              layers=[...] group_id=<int>

After analyzing these lines we know exactly which of:
  (e1) wrong group backend
  (e2) wrong group spec
  (e3) UniformTypeKVCacheSpecs missing per-layer entries for drafter
to fix.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("genesis.gemma4.pn262b_kv_alloc_trace")

GENESIS_PN262B_MARKER = (
    "Genesis PN262-B KV cache allocator/reshape/proposer-init diagnostic "
    "trace (D-3 deep-dive locator)"
)

_ENV_ENABLE = "GENESIS_ENABLE_PN262B_KV_ALLOC_TRACE"
_ENV_PREFIX = "GENESIS_PN262B_PREFIX"
_APPLIED = False
_ORIGINAL_RESHAPE = None
_ORIGINAL_PROPOSER_INIT = None
_LOG_COUNT = [0]


def _env_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _drafter_prefix() -> str:
    return os.environ.get(_ENV_PREFIX, "draft_model.").strip()


def _safe_cls(obj) -> str:
    if obj is None:
        return "<None>"
    try:
        cls = type(obj)
        return f"{cls.__module__}.{cls.__qualname__}"
    except Exception:
        return f"<{type(obj).__name__}>"


def _backend_full_name(attn_backend) -> str:
    if attn_backend is None:
        return "<None>"
    try:
        return attn_backend.full_cls_name()
    except Exception:
        try:
            return f"{attn_backend.__module__}.{attn_backend.__qualname__}"
        except Exception:
            return repr(attn_backend)


def _truncate_layers(names, limit: int = 6) -> str:
    try:
        names = list(names)
    except Exception:
        return repr(names)
    if len(names) <= limit:
        return repr(names)
    return repr(names[:limit]) + f" ...+{len(names) - limit}"


def _emit(line: str) -> None:
    log.warning(line)
    _LOG_COUNT[0] += 1


def apply() -> tuple[str, str]:
    """Install diagnostic wraps on reshape + proposer init."""
    global _APPLIED, _ORIGINAL_RESHAPE, _ORIGINAL_PROPOSER_INIT

    if not _env_enabled():
        return "skipped", (
            f"PN262-B disabled (set {_ENV_ENABLE}=1 to trace kv_cache "
            "group / proposer init / reshape for D-3 root-cause)"
        )

    if _APPLIED:
        return "applied", "PN262-B already installed (idempotent)"

    # --- Import targets ---
    log.warning("[PN262-B] apply() entered — beginning import phase")
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as e:  # noqa: BLE001 — any error during this import is fatal for the patch
        try:
            from vllm.v1.worker.gpu_model_runner import (  # type: ignore[no-redef]
                GpuModelRunner as GPUModelRunner,
            )
        except Exception as e2:  # noqa: BLE001
            log.warning(
                "[PN262-B] SKIP: GPUModelRunner not importable: "
                "first=%s second=%s", e, e2,
            )
            return "skipped", (
                f"GPUModelRunner not importable from "
                f"vllm.v1.worker.gpu_model_runner: {e!r}"
            )

    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
    except Exception as e:  # noqa: BLE001
        log.warning(
            "[PN262-B] SKIP: SpecDecodeBaseProposer not importable: %s", e,
        )
        return "skipped", (
            f"SpecDecodeBaseProposer not importable from "
            f"vllm.v1.spec_decode.llm_base_proposer: {e!r}"
        )

    log.warning(
        "[PN262-B] import phase OK — GPUModelRunner=%s SpecDecodeBaseProposer=%s",
        GPUModelRunner.__name__, SpecDecodeBaseProposer.__name__,
    )

    drafter_prefix = _drafter_prefix()

    # ----------------------------------------------------------------
    # Wrap GPUModelRunner._reshape_kv_cache_tensors
    # ----------------------------------------------------------------
    if not hasattr(GPUModelRunner, "_reshape_kv_cache_tensors"):
        return "skipped", (
            "GPUModelRunner._reshape_kv_cache_tensors not present "
            "(pin signature changed?)"
        )

    original_reshape = GPUModelRunner._reshape_kv_cache_tensors
    if getattr(original_reshape, "_genesis_pn262b_wrapped", False):
        _APPLIED = True
        return "applied", "PN262-B already wrapped on reshape (idempotent)"
    _ORIGINAL_RESHAPE = original_reshape

    def _wrapped_reshape(self, kv_cache_raw_tensors, kernel_block_sizes):
        # --- Pre-call: dump kv_cache_config groups ---
        try:
            cfg = getattr(self, "kv_cache_config", None)
            groups = getattr(cfg, "kv_cache_groups", None) if cfg else None
        except Exception:
            groups = None

        if groups is not None:
            _emit(
                f"[PN262-B/reshape:pre] num_groups={len(groups)} "
                f"len(kernel_block_sizes)={len(kernel_block_sizes)} "
                f"raw_tensor_layers={len(kv_cache_raw_tensors)}"
            )
            for gid, grp in enumerate(groups):
                spec_cls = _safe_cls(getattr(grp, "kv_cache_spec", None))
                backend_cls = _backend_full_name(getattr(grp, "backend", None))
                layer_names = getattr(grp, "layer_names", []) or []
                # Highlight drafter-containing groups.
                drafter_count = sum(
                    1 for n in layer_names
                    if isinstance(n, str) and n.startswith(drafter_prefix)
                )
                _emit(
                    f"[PN262-B/reshape:groups] gid={gid} "
                    f"spec={spec_cls} backend={backend_cls} "
                    f"n_layers={len(layer_names)} "
                    f"drafter_layers={drafter_count} "
                    f"layer_names={_truncate_layers(layer_names)}"
                )
        else:
            _emit(
                "[PN262-B/reshape:pre] kv_cache_config or .kv_cache_groups "
                "not accessible on self"
            )

        # --- Pre-call: per-drafter raw_tensor info ---
        drafter_raw = [
            (n, t) for n, t in kv_cache_raw_tensors.items()
            if isinstance(n, str) and n.startswith(drafter_prefix)
        ]
        for layer_name, raw_t in drafter_raw[:8]:
            try:
                _emit(
                    f"[PN262-B/reshape:raw] layer={layer_name!r} "
                    f"raw_shape={tuple(raw_t.shape)} "
                    f"raw_numel={raw_t.numel()} "
                    f"raw_dtype={raw_t.dtype} "
                    f"raw_contig={raw_t.is_contiguous()}"
                )
            except Exception as e:
                _emit(
                    f"[PN262-B/reshape:raw] layer={layer_name!r} "
                    f"raw introspection failed: {e}"
                )

        # --- Call original reshape (may raise) ---
        result = original_reshape(self, kv_cache_raw_tensors, kernel_block_sizes)

        # --- Post-call: per-drafter final kv_cache info ---
        try:
            drafter_out = [
                (n, t) for n, t in result.items()
                if isinstance(n, str) and n.startswith(drafter_prefix)
            ]
            for layer_name, kv_cache in drafter_out[:8]:
                try:
                    _emit(
                        f"[PN262-B/reshape:final] layer={layer_name!r} "
                        f"shape={tuple(kv_cache.shape)} "
                        f"stride={tuple(kv_cache.stride())} "
                        f"dtype={kv_cache.dtype} "
                        f"contig={kv_cache.is_contiguous()} "
                        f"ndim={kv_cache.dim()} "
                        f"data_ptr=0x{kv_cache.data_ptr():x}"
                    )
                except Exception as e:
                    _emit(
                        f"[PN262-B/reshape:final] layer={layer_name!r} "
                        f"introspection failed: {e}"
                    )
        except Exception as e:
            _emit(f"[PN262-B/reshape:final] result iteration failed: {e}")

        return result

    _wrapped_reshape._genesis_pn262b_wrapped = True  # type: ignore[attr-defined]
    GPUModelRunner._reshape_kv_cache_tensors = _wrapped_reshape  # type: ignore[method-assign]

    # ----------------------------------------------------------------
    # Wrap SpecDecodeBaseProposer.initialize_attn_backend
    # ----------------------------------------------------------------
    if hasattr(SpecDecodeBaseProposer, "initialize_attn_backend"):
        original_proposer_init = SpecDecodeBaseProposer.initialize_attn_backend
        if not getattr(original_proposer_init, "_genesis_pn262b_wrapped", False):
            _ORIGINAL_PROPOSER_INIT = original_proposer_init

            def _wrapped_proposer_init(self, kv_cache_config, kernel_block_sizes=None):
                # --- Pre-call ---
                try:
                    draft_layers = getattr(self, "_draft_attn_layer_names", None)
                    n_layers = len(draft_layers) if draft_layers else 0
                    n_groups = len(kv_cache_config.kv_cache_groups)
                    _emit(
                        f"[PN262-B/proposer:pre] n_draft_layers={n_layers} "
                        f"n_kv_cache_groups={n_groups} "
                        f"kernel_block_sizes={kernel_block_sizes!r}"
                    )
                    # Find which group(s) the draft layers will match.
                    for gid, grp in enumerate(kv_cache_config.kv_cache_groups):
                        grp_layers = set(getattr(grp, "layer_names", []) or [])
                        hits = (
                            (draft_layers or set()) & grp_layers
                            if isinstance(draft_layers, (set, frozenset))
                            else set(draft_layers or []) & grp_layers
                        )
                        if hits:
                            _emit(
                                f"[PN262-B/proposer:match] draft layers "
                                f"match group gid={gid} hit_count={len(hits)} "
                                f"group_spec={_safe_cls(grp.kv_cache_spec)} "
                                f"group_backend={_backend_full_name(grp.backend)} "
                                f"sample_hits={_truncate_layers(sorted(hits), 4)}"
                            )
                except Exception as e:
                    _emit(f"[PN262-B/proposer:pre] introspection failed: {e}")

                # --- Call original ---
                result = original_proposer_init(self, kv_cache_config, kernel_block_sizes)

                # --- Post-call ---
                try:
                    groups = getattr(self, "draft_attn_groups", None) or []
                    _emit(
                        f"[PN262-B/proposer:post] kv_cache_gid="
                        f"{getattr(self, 'kv_cache_gid', '<missing>')} "
                        f"block_size={getattr(self, 'block_size', '<missing>')} "
                        f"n_draft_attn_groups={len(groups)}"
                    )
                    for i, grp in enumerate(groups):
                        _emit(
                            f"[PN262-B/proposer:groups] idx={i} "
                            f"backend_full={_backend_full_name(grp.backend)} "
                            f"spec={_safe_cls(getattr(grp, 'kv_cache_spec', None))} "
                            f"group_id={getattr(grp, 'kv_cache_group_id', '?')} "
                            f"layers={_truncate_layers(grp.layer_names)}"
                        )
                except Exception as e:
                    _emit(f"[PN262-B/proposer:post] introspection failed: {e}")

                return result

            _wrapped_proposer_init._genesis_pn262b_wrapped = True  # type: ignore[attr-defined]
            SpecDecodeBaseProposer.initialize_attn_backend = _wrapped_proposer_init  # type: ignore[method-assign]

    _APPLIED = True
    log.warning(
        "[PN262-B] INSTALLED: _reshape_kv_cache_tensors + "
        "initialize_attn_backend wrapped with diagnostic trace "
        "(drafter prefix %r)",
        drafter_prefix,
    )
    return "applied", (
        f"PN262-B installed: trace on GpuModelRunner._reshape_kv_cache_tensors "
        f"+ SpecDecodeBaseProposer.initialize_attn_backend; drafter prefix "
        f"{drafter_prefix!r}"
    )


def is_applied() -> bool:
    return _APPLIED


def log_count() -> int:
    return _LOG_COUNT[0]


def revert() -> bool:
    """Best-effort revert (test isolation only)."""
    global _APPLIED, _ORIGINAL_RESHAPE, _ORIGINAL_PROPOSER_INIT
    if not _APPLIED:
        return False
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        if _ORIGINAL_RESHAPE is not None:
            GPUModelRunner._reshape_kv_cache_tensors = _ORIGINAL_RESHAPE  # type: ignore[method-assign]
    except ImportError:
        return False
    try:
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
        if _ORIGINAL_PROPOSER_INIT is not None:
            SpecDecodeBaseProposer.initialize_attn_backend = _ORIGINAL_PROPOSER_INIT  # type: ignore[method-assign]
    except ImportError:
        return False
    _APPLIED = False
    _ORIGINAL_RESHAPE = None
    _ORIGINAL_PROPOSER_INIT = None
    return True


__all__ = [
    "GENESIS_PN262B_MARKER",
    "apply",
    "is_applied",
    "log_count",
    "revert",
]

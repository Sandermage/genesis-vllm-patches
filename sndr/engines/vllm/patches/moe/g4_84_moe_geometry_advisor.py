# SPDX-License-Identifier: Apache-2.0
"""G4_84 — generic MoE-geometry advisor + wna16 tuned-config provider.

================================================================
WHAT IT FIXES (the structural problem we hit on Gemma-4-26B-A4B)
================================================================

A WNA16 int4 MoE layer can only use the fast tensor-core **Marlin** MoE
kernel when ``intermediate_size_per_partition % max(64, group_size) == 0``
(vLLM ``check_moe_marlin_supports_layer``). Gemma-4-26B-A4B has
``moe_intermediate_size = 704``; at TP=2 the per-shard intermediate is
``704 / 2 = 352``, and ``352 % 64 = 32 != 0`` for ANY group_size. So
Marlin is structurally rejected and the layer silently falls back to
``moe_wna16_gemm`` — a CUDA-core (NOT tensor-core), memory-bound kernel
on batch=1 decode. The operator gets NO warning; the model just runs
~1.5-1.85x slower than it would on a Marlin-eligible geometry
(measured precedent: vllm#36095, E=128 — 47 vs 87 tok/s).

This is NOT a 26B bug — it is a GENERIC trap for any MoE whose sharded
intermediate is not a multiple of 64. Different models hit it with
different geometries; we had no mechanism to detect it or to supply a
tuned config for the fallback kernel. G4_84 adds both:

  1. ADVISOR (always-on diagnostic): at MoE-method selection time, detect
     the int4-Marlin-ineligible geometry and emit a LOUD one-line warning
     naming the model's shape and the recommended remedy (FP8/int8 quant,
     which vLLM's own Gemma4 recipe recommends for small-expert MoE).
     Turns a silent 1.5x slowdown into a visible, actionable signal.

  2. CONFIG-PROVIDER (opt-in accelerator): inject Genesis-tuned
     ``moe_wna16`` block configs for known shapes so the fallback kernel
     stops running on vLLM's default ("Using default MoE config.
     Performance might be sub-optimal!"). The table is keyed by
     (E, N, dtype) and is fail-open — unknown shapes pass through to
     vLLM's lookup unchanged. Generic across models; extend the table as
     each model is rig-swept.

================================================================
SAFETY MODEL
================================================================

* env_flag: GENESIS_ENABLE_G4_84_MOE_GEOMETRY_ADVISOR (default_on True —
  the advisor is a pure log; the provider only fires for table-listed
  shapes, so default-on is safe).
* mechanism: wraps vllm.model_executor.layers.fused_moe.fused_moe.
  get_moe_configs (config-provider) and logs once per ineligible layer
  (advisor). Never changes numerics — only which block-tile schedule the
  SAME kernel uses, and only for shapes whose tuned config we verified.
* applies_to: any MoE arch (generic). No-op when no MoE layers exist.
* the provider table is EMPTY by default until a rig sweep lands a
  verified entry; the advisor works immediately and is the load-bearing
  value of this patch.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ..model_compat.gemma4._gemma4_detect import env_truthy

log = logging.getLogger("genesis.moe.g4_84_geometry_advisor")

GENESIS_G4_84_MARKER = (
    "Genesis G4_84 MoE-geometry advisor + wna16 tuned-config provider v1 "
    "(detect int4-Marlin-ineligible geometry; supply tuned moe_wna16 config)"
)

_ENV_FLAG = "GENESIS_ENABLE_G4_84_MOE_GEOMETRY_ADVISOR"

# Genesis-tuned moe_wna16 block configs, keyed by (E, N, dtype_str).
# N is the per-shard intermediate (size_n). Each value is a dict with the
# vLLM tuned-config schema: {"BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K",
# "GROUP_SIZE_M","num_warps","num_stages"}. Empty until a rig sweep lands
# a bit-verified entry (tools/triton_gemm_sweep / benchmark_moe). Unknown
# shapes fall through to vLLM's own lookup (fail-open).
_GENESIS_MOE_WNA16_CONFIGS: dict[tuple[int, int, str], dict[str, int]] = {
    # (E, N, dtype): {tuned block config}
    # e.g. (128, 352, "int4_w4a16"): {...}  # Gemma-4-26B-A4B TP=2 — pending rig sweep
}

# State for revert / idempotency.
_ORIGINAL_GET_MOE_CONFIGS: Optional[Callable[..., Any]] = None
_WARNED_SHAPES: set[tuple[int, int, str]] = set()


def marlin_moe_marginal(intermediate_per_partition: int, group_size: int) -> bool:
    """Return True if this WNA16 MoE geometry is Marlin-INELIGIBLE.

    Mirrors vLLM ``check_moe_marlin_supports_layer``: Marlin needs
    ``intermediate_per_partition % max(64, group_size) == 0``. When that
    fails the layer falls back to the slow CUDA-core ``moe_wna16`` kernel.
    """
    divisor = max(64, group_size if group_size and group_size > 0 else 64)
    return (intermediate_per_partition % divisor) != 0


def _advise_if_ineligible(E: int, N: int, dtype: str, group_size: int) -> None:
    """Emit one loud warning per ineligible (E, N, dtype) shape."""
    key = (int(E), int(N), str(dtype))
    if key in _WARNED_SHAPES:
        return
    _WARNED_SHAPES.add(key)
    if "int4" in dtype and marlin_moe_marginal(N, group_size):
        log.warning(
            "[G4_84] MoE geometry E=%d N=%d (%s, group_size=%s) is "
            "MARLIN-INELIGIBLE (N %% max(64,gs) != 0) -> falls back to the "
            "slow CUDA-core moe_wna16 kernel (~1.5-1.85x slower on decode). "
            "REMEDY: serve an FP8-Dynamic or int8 W8A16 checkpoint of this "
            "model (vLLM's own Gemma4 recipe recommends int8/FP8 for "
            "small-expert MoE), or lower TP so the per-shard intermediate "
            "becomes a multiple of 64.",
            E, N, dtype, group_size,
        )


def apply() -> bool:
    """Wrap get_moe_configs: advise on ineligible geometry + inject our
    tuned config for table-listed shapes. Returns True on success."""
    global _ORIGINAL_GET_MOE_CONFIGS
    if not env_truthy(_ENV_FLAG, default=True):
        return False
    try:
        from vllm.model_executor.layers.fused_moe import fused_moe as _fm
    except Exception as e:  # pragma: no cover - import guard
        log.info("[G4_84] fused_moe not importable: %s", e)
        return False

    if getattr(_fm.get_moe_configs, "_genesis_g4_84", False):
        return True  # already wrapped (idempotent)

    _ORIGINAL_GET_MOE_CONFIGS = _fm.get_moe_configs
    original = _ORIGINAL_GET_MOE_CONFIGS

    def _guarded_get_moe_configs(E: int, N: int, dtype, *args, **kwargs):
        # dtype may be a str ("int4_w4a16") or None depending on call site.
        dtype_str = str(dtype) if dtype is not None else ""
        # Advisor: warn once if this is an int4-Marlin-ineligible shape.
        # group_size is not passed here; advise on dtype+geometry heuristically.
        if "int4" in dtype_str:
            _advise_if_ineligible(E, N, dtype_str, group_size=128)
        # Provider: inject Genesis-tuned config for known shapes.
        key = (int(E), int(N), dtype_str)
        tuned = _GENESIS_MOE_WNA16_CONFIGS.get(key)
        if tuned is not None:
            log.info("[G4_84] using Genesis-tuned moe_wna16 config for "
                     "E=%d N=%d %s", E, N, dtype_str)
            # vLLM keys configs by M-bucket; return a single-entry mapping
            # honoured by get_moe_configs' caller (nearest-M lookup).
            return {0: dict(tuned)}
        return original(E, N, dtype, *args, **kwargs)

    _guarded_get_moe_configs._genesis_g4_84 = True  # type: ignore[attr-defined]
    _fm.get_moe_configs = _guarded_get_moe_configs
    log.info("[G4_84] installed: MoE-geometry advisor + wna16 config-provider "
             "(table entries: %d)", len(_GENESIS_MOE_WNA16_CONFIGS))
    return True


def is_applied() -> bool:
    try:
        from vllm.model_executor.layers.fused_moe import fused_moe as _fm
    except Exception:
        return False
    return bool(getattr(_fm.get_moe_configs, "_genesis_g4_84", False))


def revert() -> bool:
    global _ORIGINAL_GET_MOE_CONFIGS
    if _ORIGINAL_GET_MOE_CONFIGS is None:
        return False
    try:
        from vllm.model_executor.layers.fused_moe import fused_moe as _fm
        _fm.get_moe_configs = _ORIGINAL_GET_MOE_CONFIGS
        _ORIGINAL_GET_MOE_CONFIGS = None
        _WARNED_SHAPES.clear()
        return True
    except Exception:
        return False

# SPDX-License-Identifier: Apache-2.0
"""PN299C — arch-aware NUM_WARPS cap on FLA ``layernorm_guard.py``.

Last remaining ``num_warps = 8`` site in ``vllm/model_executor/layers/
fla/ops/`` after PN298 + PN299 + PN299B coverage. Different shape from
the autotune list-comprehension prunes: this one is a **runtime
heuristic** (``num_warps = min(max(BLOCK_N // 256, 1), 8)``), so it
needs a numeric-cap replacement instead of a config-filter wrap.

For Qwen3.6-A3B (hidden 5120 → BLOCK_N = 8192 after ``next_power_of_2``)
the heuristic picks ``num_warps = min(max(8192 // 256, 1), 8) =
min(32, 8) = 8`` on every layer-norm call. With 30 GDN layers and
periodic LN inside MoE forward, this kernel fires per layer per token —
hot path. On SM 8.6 (RTX A5000, 100 KB shared) 8 warps spill registers
and force an eviction-recompile loop in autotune.

The fix is a 1-line cap: replace the hardcoded ``8`` with the same env
var PN296 auto-sets (``GENESIS_TRITON_AUTOTUNE_MAX_WARPS``, default 4
on Ampere). On Hopper+ where 8 warps fit comfortably, the env stays at
``'8'`` and the heuristic behaves identically to upstream.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Audit lineage: kernels loop iteration, 2026-06-08.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn299c_fla_layernorm_guard")

GENESIS_PN299C_MARKER = (
    "Genesis PN299C layernorm_guard arch warps (SM 8.6 cap) v1"
)


# Single anchor — the upstream heuristic literal. The leading 4-space
# indent matches the function body of ``_layer_norm_fwd``. The
# replacement reads the env var that PN296 auto-sets; in absence of
# PN296 (non-Ampere host), the env defaults to ``'8'`` and behaviour
# is identical to upstream.
PN299C_LN_OLD = (
    "    # heuristics for number of warps\n"
    "    num_warps = min(max(BLOCK_N // 256, 1), 8)\n"
)
PN299C_LN_NEW = (
    "    # [Genesis PN299C] arch-aware cap on the upstream heuristic.\n"
    "    # PN296 auto-sets GENESIS_TRITON_AUTOTUNE_MAX_WARPS=4 on SM 8.6\n"
    "    # (RTX A5000 / 3090) where 8-warp configs spill 100 KB shared/SM.\n"
    "    # Without the env (non-Ampere host), default '8' = upstream\n"
    "    # behaviour. Heuristic + cap are evaluated per call; no JIT churn.\n"
    "    num_warps = min(\n"
    "        max(BLOCK_N // 256, 1),\n"
    "        int(__import__('os').environ.get('GENESIS_TRITON_AUTOTUNE_MAX_WARPS', '8')),\n"
    "    )\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN299C", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply PN299C — cap the FLA layernorm_guard runtime num_warps heuristic."""
    if _env_disabled():
        return "skipped", "PN299C disabled via GENESIS_DISABLE_PN299C=1"

    target = resolve_vllm_file("model_executor/layers/fla/ops/layernorm_guard.py")
    if target is None:
        return "skipped", (
            "PN299C: layernorm_guard.py not found in vllm install — "
            "pin may predate FLA layernorm guard or have a different layout"
        )

    patcher = TextPatcher(
        patch_name=(
            "PN299C model_executor/layers/fla/ops/layernorm_guard.py — "
            "arch-aware NUM_WARPS heuristic cap"
        ),
        target_file=str(target),
        marker=GENESIS_PN299C_MARKER,
        sub_patches=[
            TextPatch(
                name="pn299c_ln_num_warps_cap",
                anchor=PN299C_LN_OLD,
                replacement=PN299C_LN_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["[Genesis PN299C", "GENESIS_TRITON_AUTOTUNE_MAX_WARPS"],
    )

    try:
        result, failure = patcher.apply()
    except Exception as e:
        log.warning("[PN299C] apply() raised %s — leaving upstream heuristic", e)
        return "skipped", f"PN299C raised at apply: {e!r}"

    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "anchor drift / not eligible"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "skipped", f"PN299C: {reason}{detail}"

    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "failed", f"PN299C: {reason}{detail}"

    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN299C idempotent: marker already present"

    return "applied", (
        "PN299C applied: layernorm_guard.py num_warps heuristic now reads "
        "GENESIS_TRITON_AUTOTUNE_MAX_WARPS (PN296 auto-sets =4 on SM 8.6). "
        "Closes the last num_warps=8 site in vllm/model_executor/layers/fla/ops/."
    )


def is_applied() -> bool:
    target = resolve_vllm_file("model_executor/layers/fla/ops/layernorm_guard.py")
    if target is None:
        return False
    try:
        return GENESIS_PN299C_MARKER in target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

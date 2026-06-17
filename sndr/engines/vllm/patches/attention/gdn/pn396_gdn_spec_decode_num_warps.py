# SPDX-License-Identifier: Apache-2.0
"""PN396 — GDN spec-decode recurrent kernel num_warps 4 -> 1 (SM 8.6).

RETIRED 2026-06-17 (lifecycle=retired) — tested-negative dead-end. The anchor
still resolves on 0.23.1, but the A/B on 2026-06-14 proved num_warps=1 REGRESSES
everywhere vs upstream 4 (code -4.2%, tool_call -10.8%, short_chat -5.5%). Kept
capped <0.23.0 AND retired so the refuted 4-vs-1 result is not re-investigated.

TESTED-NEGATIVE — DO NOT ENABLE (default OFF). A/B 2026-06-14 on dev491 (PROD
35B, chat-matrix n=5): num_warps=1 REGRESSED vs upstream 4 — code -4.2%,
thinking -2.6/-3.8%, short_chat -5.5%, tool_call -10.8% (multi_turn flat). The
"1 warp avoids the cross-warp reduction shuffle" hypothesis below was refuted:
4 warps win because the extra threads hide per-token q/k/v global-load latency,
and this gating+varlen kernel differs from the pure-recurrent fused_recurrent
siblings. Kept as a documented dead-end (anchor + dispatch retained, env off)
so the 4-vs-1 question is not re-opened. The original (refuted) rationale:

The dominant decode kernel under MTP K=3 on hybrid_gdn_moe is
``fused_sigmoid_gating_delta_rule_update_kernel`` (30 of 41 layers — every
GDN/Mamba layer routes its spec-decode step through it, IS_VARLEN=True). Its
launcher (``fla/ops/fused_sigmoid_gating.py``) hardcodes ``num_warps = 4``,
while every sibling recurrent kernel in the same family
(``fla/ops/fused_recurrent.py:199,439``) uses ``num_warps = 1`` for the
identical ``[BV=32, BK=128]`` fp32 tile.

Why 1 is correct for this kernel on our shape (V=128 -> BV=min(npo2(V),32)=32,
K=128 -> BK=128, so the state tile is [32,128]):

  * The hot per-token work is two reductions over BK=128 —
    ``b_v = -tl.sum(b_h * b_k[None,:], 1)`` and ``b_o = tl.sum(b_h * b_q, 1)``
    (fused_sigmoid_gating.py:148,152) — executed every token.
  * With num_warps=1 (32 threads) Triton maps the 32 state rows one-per-thread,
    so each BK=128 reduction is INTRA-thread (sequential, no shuffle) and the
    recurrent b_h state stays in that thread's registers across the T loop.
  * With num_warps=4 (128 threads) the 32 rows are split across 4 threads each,
    so every per-token reduction becomes a 4-way CROSS-WARP shuffle/shared-mem
    reduction — pure overhead on a kernel whose arithmetic is already
    register-resident. num_warps=1 also packs more (sequence, head) blocks per
    SM (higher occupancy on the 28-SM A5000).
  * The gating (softplus / exp(A_log) / sigmoid) is SCALAR per head (not
    V-vectorized: ``p_a = a + bos*HV + i_hv``), so warp count does not help it.

This is a LAUNCH-PARAMETER change only — output is bit-identical (verify with a
fixed greedy decode replay). Opt-in (default OFF) pending the decode-TPOT A/B;
``GENESIS_DISABLE_PN396=1`` force-reverts even when enabled.

Anchor: ``    num_stages = 3\n    num_warps = 4`` is unique to
fused_sigmoid_gating.py within fla/ops (fused_recurrent already ships =1).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn396_gdn_spec_decode_num_warps")

_ENV_ENABLE = "GENESIS_ENABLE_PN396_GDN_SPEC_DECODE_WARPS"
_ENV_DISABLE = "GENESIS_DISABLE_PN396"
_TARGET_REL = "model_executor/layers/fla/ops/fused_sigmoid_gating.py"

GENESIS_PN396_MARKER = "[Genesis PN396] GDN spec-decode recurrent num_warps 4->1"

PN396_ANCHOR_OLD = (
    "    assert NK == 1, \"NK > 1 is not supported yet\"\n"
    "    num_stages = 3\n"
    "    num_warps = 4\n"
)
PN396_ANCHOR_NEW = (
    "    assert NK == 1, \"NK > 1 is not supported yet\"\n"
    "    num_stages = 3\n"
    "    # [Genesis PN396] GDN spec-decode recurrent num_warps 4->1 — the\n"
    "    # [BV=32,BK=128] state tile maps one-row-per-thread at 1 warp, making\n"
    "    # the per-token tl.sum over BK intra-thread (no cross-warp shuffle);\n"
    "    # matches fused_recurrent.py siblings (num_warps=1). Bit-exact launch\n"
    "    # param. GENESIS_DISABLE_PN396=1 reverts.\n"
    "    num_warps = 1\n"
)


def _env_enabled() -> bool:
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return False
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN396 fla/ops/fused_sigmoid_gating.py — GDN spec-decode "
            "recurrent num_warps 4->1 (SM 8.6 row-per-thread reduction)"
        ),
        target_file=str(target),
        marker=GENESIS_PN396_MARKER,
        sub_patches=[
            TextPatch(
                name="pn396_gdn_spec_decode_num_warps",
                anchor=PN396_ANCHOR_OLD,
                replacement=PN396_ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=["[Genesis PN396"],
    )


def apply() -> tuple[str, str]:
    """Text-patch the GDN spec-decode recurrent kernel launch num_warps."""
    if not _env_enabled():
        return "skipped", (
            f"PN396 disabled (set {_ENV_ENABLE}=1 to drop the GDN spec-decode "
            f"recurrent kernel num_warps 4->1 on SM 8.6 — A/B pending)"
        )

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", (
            "PN396: fla/ops/fused_sigmoid_gating.py not found in the vllm "
            "install — pin may predate the fused-sigmoid-gating GDN kernel"
        )

    try:
        result, failure = patcher.apply()
    except Exception as e:  # never raise out of an apply hook
        log.warning("[PN396] apply() raised %s — leaving upstream num_warps", e)
        return "skipped", f"PN396 raised at apply: {e!r}"

    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "anchor drift / not eligible"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "skipped", (
            f"PN396: {reason}{detail}. num_warps NOT overridden — upstream 4 "
            "remains (kernel may have been retuned upstream)."
        )
    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "failed", f"PN396: {reason}{detail}"
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN396 idempotent: marker already present (num_warps=1)."

    return "applied", (
        "PN396 applied: GDN spec-decode recurrent kernel num_warps 4->1 "
        "(intra-thread BK reduction; matches fused_recurrent siblings)."
    )


def is_applied() -> bool:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return False
    try:
        return GENESIS_PN396_MARKER in target.read_text(encoding="utf-8")
    except OSError:
        return False

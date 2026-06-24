# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN400 — restore the is_sym qzeros guard for symmetric AutoRound/
GPTQ Marlin MoE (backport of vllm#45656; fixes the regression from vllm#43409).

RETIRED 2026-06-24 (pin bump dev148 -> dev301) — superseded by vllm#45656,
whose NVIDIA auto_gptq.py half is native in dev301 (the anchor drifted per the
dev301 anchor-SOT regen). Registry lifecycle=retired, capped <0.23.1rc1.dev301.
The transform below is UNCHANGED and still applies on dev148 (the previous/
rollback pin, where the #43409 regression is live); on dev301+ the version cap
+ the post-fix anchor self-skip it. Do not delete the module while dev148
rollback is possible.

================================================================
ROOT CAUSE — a CORRECTNESS regression that landed IN our pin
================================================================

vllm#43409 (merged 2026-06-12, IN dev148) removed the
`if not self.quant_config.is_sym else None` guard around the MoE zero-points
in ``AutoGPTQMoEMethod.get_fused_moe_quant_config`` so the CPU
``fused_experts_cpu`` path could receive synthesized zero points for
symmetric models. On NVIDIA GPUs this regressed *symmetric* (is_sym=True)
AutoRound / GPTQ Marlin MoE: the method now passes the (meaningless)
``w13_qzeros`` / ``w2_qzeros`` tensors to the Marlin kernel, which consumes
them and produces INCORRECT expert outputs.

vllm#45656 ("Restore is_sym guard for zp in GPTQ/CT MoE", merged 2026-06-18
16:20Z) fixes it by gating the zero-points behind
``use_zp = not is_sym or backend == CPU``. That fix landed ~12h AFTER the
dev148 base commit (b4c80ec0f @ 2026-06-18 04:18Z) so it is NOT in our pin.

================================================================
WHY OUR 27B IS HIT (CONFIRMED)
================================================================

``Qwen3.6-27B-int4-AutoRound`` is W4A16, group_size=128, **sym=True**,
packing ``auto_round:auto_gptq`` (verified live on the checkpoint config.json:
quant_method=auto-round, sym=True, bits=4). The auto_gptq packing routes it
through ``AutoGPTQMoEMethod`` (FILE auto_gptq.py) — the exact #43409-broken
path — on NVIDIA A5000. So on dev148 the 27B MoE expert outputs are wrong.

================================================================
FIX (text-patch, 1 anchored sub-patch in 1 file)
================================================================

``auto_gptq.py`` ``get_fused_moe_quant_config``: insert
``_genesis_pn400_use_zp = not self.quant_config.is_sym`` before the
``gptq_marlin_moe_quant_config(...)`` return and gate w1_zp/w2_zp on it. For
symmetric models this passes ``None`` (the pre-#43409 behaviour) so the Marlin
kernel does not consume meaningless qzeros.

DELIBERATE SIMPLIFICATION vs #45656: the upstream ``or backend == CPU`` clause
is dropped — this rig is NVIDIA-only, so the CPU ``fused_experts_cpu`` path is
dead here and the extra ``WNA16MoEBackend`` import (a second anchor / drift
surface / NameError-if-half-applied risk) is avoided. If CPU inference is ever
added, restore the full clause + import. We also intentionally cover ONLY the
auto_gptq path (our 27B's path); the compressed-tensors twin
(``CompressedTensorsWNA16MarlinMoEMethod``, file 2 of #45656) is a different
checkpoint format we do not run — add a PN400B sibling if a compressed-tensors
WNA16 Marlin MoE model enters the rotation.

================================================================
SAFETY MODEL
================================================================

- Anchor SELF-GATES to exactly the broken pin window: the unconditional
  ``w1_zp=getattr(layer, "w13_qzeros", None),`` text exists ONLY on pins that
  have #43409 and not #45656. Pre-#43409 the line was
  ``... if not is_sym else None``; post-#45656 it is ``... if use_zp else
  None``. Both differ from our anchor -> TextPatcher self-skips -> no
  double-apply.
- Default OFF (env-gated ``GENESIS_ENABLE_PN400=1``, dispatcher-applied) +
  ``applies_to.quant_format`` scoped to autoround/gptq formats, so it never
  touches the FP8 35B. Symmetric-only effect: for is_sym=False (asymmetric)
  models the guard evaluates to True -> unchanged behaviour.
- Idempotent via marker; drift-aware on the upstream-fixed form.

STATUS: lifecycle=experimental — NEEDS 27B END-TO-END VALIDATION. The text
transform is unit-tested (fixture: dev148-broken -> fixed), but the semantic
"27B output is now correct" check requires a 27B greedy A/B on the rig
(dev148 vs dev148+PN400), which displaces the 35B PROD — operator-gated.

Author backport: Genesis /loop upstream sweep 2026-06-20.
Original fix: vllm#45656 (regression: vllm#43409).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn400_marlin_moe_sym_zp_guard")


GENESIS_PN400_MARKER = "Genesis PN400 Marlin MoE is_sym zp guard (vllm#45656)"


PN400_ANCHOR = (
    "        return gptq_marlin_moe_quant_config(\n"
    "            w1_scale=layer.w13_scales,\n"
    "            w2_scale=layer.w2_scales,\n"
    "            weight_bits=self.quant_config.weight_bits,\n"
    "            group_size=self.quant_config.group_size,\n"
    '            w1_zp=getattr(layer, "w13_qzeros", None),\n'
    '            w2_zp=getattr(layer, "w2_qzeros", None),\n'
)

PN400_REPLACE = (
    "        # [Genesis PN400 vllm#45656 backport] Restore the is_sym qzeros\n"
    "        # guard vllm#43409 removed. For symmetric (sym=True) AutoRound/GPTQ\n"
    "        # Marlin MoE on NVIDIA, feeding the meaningless qzeros to the Marlin\n"
    "        # kernel produces INCORRECT expert outputs. Our 27B\n"
    "        # (Qwen3.6-27B-int4-AutoRound, sym=True, packing auto_round:auto_gptq)\n"
    "        # is on exactly this path. CPU-backend clause from #45656 dropped —\n"
    "        # this rig is NVIDIA-only (CPU fused_experts_cpu path is dead here).\n"
    "        _genesis_pn400_use_zp = not self.quant_config.is_sym\n"
    "        return gptq_marlin_moe_quant_config(\n"
    "            w1_scale=layer.w13_scales,\n"
    "            w2_scale=layer.w2_scales,\n"
    "            weight_bits=self.quant_config.weight_bits,\n"
    "            group_size=self.quant_config.group_size,\n"
    '            w1_zp=getattr(layer, "w13_qzeros", None) if _genesis_pn400_use_zp else None,\n'
    '            w2_zp=getattr(layer, "w2_qzeros", None) if _genesis_pn400_use_zp else None,\n'
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/quantization/auto_gptq.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN400 auto_gptq.py — is_sym zp guard for Marlin MoE (vllm#45656)",
        target_file=str(target),
        marker=GENESIS_PN400_MARKER,
        sub_patches=[
            TextPatch(
                name="pn400_marlin_moe_sym_zp_guard",
                anchor=PN400_ANCHOR,
                replacement=PN400_REPLACE,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN400",
            # Upstream-merge / pre-#43409 detection: if the file already gates
            # the zp (the #45656 `if use_zp else None` or the pre-#43409
            # `if not is_sym else None`), our unconditional anchor cannot match
            # anyway — flag the gated form so the skip reason is honest.
            'w1_zp=getattr(layer, "w13_qzeros", None) if',
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN400 — restore the is_sym qzeros guard for symmetric Marlin MoE."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN400")
    log_decision("PN400", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "auto_gptq.py not found (pre-rename pin or non-GPTQ build)"
    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"

    with open(patcher.target_file) as f:
        content = f.read()

    if GENESIS_PN400_MARKER in content:
        return "applied", "idempotent (marker present)"

    # Drift / upstream-merge detection: if the zp is already is_sym-gated
    # (upstream #45656 landed, or a pre-#43409 build), our unconditional anchor
    # cannot match — surface that honestly instead of a bare anchor-miss.
    for marker in patcher.upstream_drift_markers:
        if marker.startswith("[Genesis"):
            continue
        if marker in content:
            return (
                "skipped",
                f"upstream drift: {marker!r} present — the zp is already "
                "is_sym-gated (vllm#45656 merged or pre-#43409 build); PN400 "
                "not needed on this pin",
            )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return (
        "applied",
        "PN400 applied: auto_gptq.py get_fused_moe_quant_config now gates "
        "w1_zp/w2_zp on `not is_sym`, so symmetric AutoRound/GPTQ Marlin MoE no "
        "longer feeds meaningless qzeros to the Marlin kernel (restores "
        "pre-#43409 correctness; vllm#45656 backport).",
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return GENESIS_PN400_MARKER in f.read()
    except Exception:
        return False


def revert() -> tuple[str, str]:
    return (
        "skipped",
        "PN400 text-patch revert not supported in-place; redeploy a fresh "
        "container or `git checkout` auto_gptq.py",
    )

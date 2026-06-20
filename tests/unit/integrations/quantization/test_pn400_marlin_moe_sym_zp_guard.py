# SPDX-License-Identifier: Apache-2.0
"""TDD for PN400 — restore is_sym qzeros guard for symmetric Marlin MoE
(vllm#45656 backport). Validates the text transform against a fixture that
reproduces the exact dev148 (post-#43409, pre-#45656) broken state, without
depending on a live vLLM install.
"""
from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path

from sndr.engines.vllm.patches.quantization import (
    pn400_marlin_moe_sym_zp_guard as pn400,
)
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

# Exact dev148 method body (post-#43409 broken form) — the anchor verbatim
# plus the surrounding lines the patch leaves untouched.
_DEV148_BROKEN = (
    "class AutoGPTQMoEMethod:\n"
    "    def get_fused_moe_quant_config(self, layer):\n"
    "        from vllm.x import gptq_marlin_moe_quant_config\n"
    + pn400.PN400_ANCHOR
    + '            w1_bias=getattr(layer, "w13_bias", None),\n'
    '            w2_bias=getattr(layer, "w2_bias", None),\n'
    "        )\n"
)


def _apply(content: str) -> tuple[TextPatchResult, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        patcher = TextPatcher(
            patch_name="PN400 test",
            target_file=path,
            marker=pn400.GENESIS_PN400_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn400_marlin_moe_sym_zp_guard",
                    anchor=pn400.PN400_ANCHOR,
                    replacement=pn400.PN400_REPLACE,
                    required=True,
                ),
            ],
        )
        result, _failure = patcher.apply()
        out = Path(path).read_text(encoding="utf-8")
        return result, out
    finally:
        os.unlink(path)


def test_pn400_gates_zp_on_not_is_sym():
    result, out = _apply(_DEV148_BROKEN)
    assert result in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT), result
    # the use_zp guard is inserted before the return
    assert "_genesis_pn400_use_zp = not self.quant_config.is_sym" in out
    # both zp args are now gated on it
    assert (
        'w1_zp=getattr(layer, "w13_qzeros", None) if _genesis_pn400_use_zp else None'
        in out
    )
    assert (
        'w2_zp=getattr(layer, "w2_qzeros", None) if _genesis_pn400_use_zp else None'
        in out
    )
    # marker present so is_applied() / idempotency detects it
    assert pn400.GENESIS_PN400_MARKER in out
    # the bare (broken) unconditional zp lines are gone
    assert 'w1_zp=getattr(layer, "w13_qzeros", None),\n' not in out
    assert 'w2_zp=getattr(layer, "w2_qzeros", None),\n' not in out
    # untouched lines survive
    assert 'w1_bias=getattr(layer, "w13_bias", None),' in out
    # the transformed source is valid Python
    ast.parse(out)


def test_pn400_self_gates_on_already_fixed_pin():
    # post-#45656 form: zp already gated -> our unconditional anchor must NOT
    # match -> SKIPPED, no double-apply, no marker injected.
    fixed = _DEV148_BROKEN.replace(
        'w1_zp=getattr(layer, "w13_qzeros", None),',
        'w1_zp=getattr(layer, "w13_qzeros", None) if use_zp else None,',
    ).replace(
        'w2_zp=getattr(layer, "w2_qzeros", None),',
        'w2_zp=getattr(layer, "w2_qzeros", None) if use_zp else None,',
    )
    result, out = _apply(fixed)
    assert result == TextPatchResult.SKIPPED, result
    assert pn400.GENESIS_PN400_MARKER not in out


def test_pn400_idempotent_reapply():
    result1, out1 = _apply(_DEV148_BROKEN)
    assert result1 in (TextPatchResult.APPLIED, TextPatchResult.IDEMPOTENT)
    # re-applying to the already-patched output must not match the bare anchor
    result2, out2 = _apply(out1)
    assert result2 in (
        TextPatchResult.IDEMPOTENT,
        TextPatchResult.SKIPPED,
    ), result2

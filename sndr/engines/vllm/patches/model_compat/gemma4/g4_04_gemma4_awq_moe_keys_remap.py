# SPDX-License-Identifier: Apache-2.0
"""G4_04 — vendor vllm#40886: AWQ compressed-tensors MoE key remap.

================================================================
WHAT IT FIXES
================================================================

The ``cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`` Gemma 4 26B MoE checkpoint
stores expert weights with ``_packed`` (int32) and ``_scale`` (bf16)
suffixes (compressed-tensors pack-quantized format). vLLM's
``_weight_iterator`` in ``gemma4.py`` renames base keys (``experts.gate_up_proj``
→ ``moe.gate_up_proj``) but has no handling for those AWQ-specific
suffix variants, so weight load fails with KeyError.

The upstream fix is vllm#40886 (OPEN as of 2026-05-17): 23 lines added
to ``_weight_iterator`` to remap 4 packed/scale key types. Zero
deletions, no new imports — surgical addition.

This patch vendors that fix via a Genesis TextPatcher so AWQ MoE Gemma 4
checkpoints load on operator-side **today**, without waiting for
upstream merge. When upstream merges, the anchor falls out (the surrounding
context changes), and Genesis records the patch as ``superseded_by``.

================================================================
WHY TEXTPATCHER (NOT MONKEY-PATCH)
================================================================

``_weight_iterator`` is a generator function defined inside
``Gemma4ForCausalLM.load_weights``. There's no clean rebind point —
the function captures local state (use_k_eq_v, k_eq_v_layer_indices,
config, etc.) from the enclosing scope. Monkey-patching would require
re-writing ~80 lines of the parent method.

TextPatcher is the correct mechanism: surgical anchor → replacement
at the exact line where the new MoE key handling should sit.

================================================================
SAFETY MODEL
================================================================

* default_on: True
* env_flag: GENESIS_ENABLE_G4_04_GEMMA4_AWQ_MOE_KEYS_REMAP
* applies_to:
    - architecture: gemma4
    - quantization: compressed-tensors (AWQ pack-quantized)
* superseded_by: when vllm#40886 merges (anchor drift detects it)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/pull/40886
  * Diff (saved locally): sndr_private/research/gemma4/vllm_prs/pr_40886.diff
"""
from __future__ import annotations

import logging

from vllm.sndr_core.core import TextPatch, TextPatcher
from vllm.sndr_core.detection.guards import resolve_vllm_file

from ._gemma4_detect import env_truthy

log = logging.getLogger("genesis.gemma4.g4_04_awq_moe_keys_remap")

GENESIS_G4_04_MARKER = (
    "Genesis G4_04 gemma4 AWQ compressed-tensors MoE keys remap v1 "
    "(vendors vllm#40886 — unblocks cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_04_GEMMA4_AWQ_MOE_KEYS_REMAP"


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


# Anchor exactly matches the pre-fix code (the comment + the float-path "if"
# that immediately follows it). The replacement inserts the 4 new key cases
# BEFORE the original "if" line, leaving the original code path intact for
# unquantized checkpoints.

_ANCHOR = (
    "                # No transpose needed: checkpoint orientation already\n"
    "                # matches FusedMoE's expected layout.\n"
    "                if \"moe.gate_up_proj\" in name and weight.dim() == 3:"
)

_REPLACEMENT = (
    "                # No transpose needed: checkpoint orientation already\n"
    "                # matches FusedMoE's expected layout.\n"
    "                # === Genesis G4_04 (vendor vllm#40886) START ===\n"
    "                # AWQ compressed-tensors pack-quantized MoE key remap.\n"
    "                # Handles 4 expert-weight key types not covered by the\n"
    "                # float-explosion branch below.\n"
    "                if \"moe.gate_up_proj_packed\" in name and weight.dim() == 3:\n"
    "                    mid = weight.size(1) // 2\n"
    "                    for e in range(weight.size(0)):\n"
    "                        base = name.replace(\"moe.\", f\"moe.experts.{e}.\")\n"
    "                        yield (base.replace(\"gate_up_proj_packed\", \"gate_proj.weight_packed\"),\n"
    "                               weight[e, :mid])\n"
    "                        yield (base.replace(\"gate_up_proj_packed\", \"up_proj.weight_packed\"),\n"
    "                               weight[e, mid:])\n"
    "                    continue\n"
    "                if \"moe.gate_up_proj_scale\" in name and weight.dim() == 3:\n"
    "                    mid = weight.size(1) // 2\n"
    "                    for e in range(weight.size(0)):\n"
    "                        base = name.replace(\"moe.\", f\"moe.experts.{e}.\")\n"
    "                        yield (base.replace(\"gate_up_proj_scale\", \"gate_proj.weight_scale\"),\n"
    "                               weight[e, :mid])\n"
    "                        yield (base.replace(\"gate_up_proj_scale\", \"up_proj.weight_scale\"),\n"
    "                               weight[e, mid:])\n"
    "                    continue\n"
    "                if \"moe.down_proj_packed\" in name and weight.dim() == 3:\n"
    "                    for e in range(weight.size(0)):\n"
    "                        yield (name.replace(\"moe.\", f\"moe.experts.{e}.\")\n"
    "                                   .replace(\"down_proj_packed\", \"down_proj.weight_packed\"),\n"
    "                               weight[e])\n"
    "                    continue\n"
    "                if \"moe.down_proj_scale\" in name and weight.dim() == 3:\n"
    "                    for e in range(weight.size(0)):\n"
    "                        yield (name.replace(\"moe.\", f\"moe.experts.{e}.\")\n"
    "                                   .replace(\"down_proj_scale\", \"down_proj.weight_scale\"),\n"
    "                               weight[e])\n"
    "                    continue\n"
    "                # === Genesis G4_04 END ===\n"
    "                if \"moe.gate_up_proj\" in name and weight.dim() == 3:"
)


def _make_patcher() -> TextPatcher | None:
    # K.1.R.R.6 (2026-05-29): fixed path prefix — resolve_vllm_file expects
    # path relative to vllm package root (no "vllm/" prefix). Bug found via
    # gemma4 boot log audit showing this patch silently skipping despite env.
    target_file = resolve_vllm_file("model_executor/models/gemma4.py")
    if target_file is None:
        return None
    return TextPatcher(
        patch_name="G4_04 gemma4 AWQ MoE keys remap",
        target_file=target_file,
        marker=GENESIS_G4_04_MARKER,
        sub_patches=[
            TextPatch(
                name="awq_moe_keys_remap",
                anchor=_ANCHOR,
                replacement=_REPLACEMENT,
                required=True,
                upstream_merged_markers=[
                    # When upstream merges #40886 these strings appear:
                    "moe.gate_up_proj_packed",
                    "moe.down_proj_packed",
                ],
                on_upstream_merge="skip_silently",
            ),
        ],
        upstream_drift_markers=[
            "Genesis G4_04",  # idempotency on re-apply
        ],
    )


def apply() -> tuple[str, str]:
    if not _env_enabled():
        return "skipped", (
            f"G4_04 disabled (set {_ENV_ENABLE}=1 to vendor vllm#40886 AWQ MoE "
            "key remap into gemma4.py — unblocks AWQ-4bit Gemma 4 MoE)"
        )

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/model_executor/models/gemma4.py not resolvable in this pin"

    result, failure = patcher.apply()
    from vllm.sndr_core.core import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "G4_04 applied: vllm#40886 AWQ compressed-tensors MoE keys remap "
            "vendored into gemma4.py. cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit will "
            "now load. Will silently skip once upstream #40886 merges."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    from vllm.sndr_core.core import marker_present_in_target
    patcher = _make_patcher()
    if patcher is None:
        return False
    return marker_present_in_target(patcher)


# ════════════════════════════════════════════════════════════════════════
# Build-time manifest registration (P2.1 Site Map — STABLE ratchet)
# ════════════════════════════════════════════════════════════════════════
#
# `register_for_manifest()` is called by scripts/build_anchor_manifest.py
# at BUILD TIME (not runtime) to enroll G4_04's sub-patcher into the
# anchor-offset manifest. It constructs the same TextPatcher object
# pointed at the PRISTINE FIXTURE under tests/legacy/pristine_fixtures/
# (gemma4.py extracted at vllm 0.20.2rc1.dev338+gbf0d2dc6d) so it works
# without a live vllm install. Runtime apply() above is unaffected — it
# still uses resolve_vllm_file() to find the live install.


def _make_patcher_for_fixture(
    name: str, fixture_path, *, patch_id: str,
) -> TextPatcher:
    """Build a TextPatcher targeting a pristine fixture file (build mode)."""
    return TextPatcher(
        patch_name=name,
        target_file=str(fixture_path),
        marker=GENESIS_G4_04_MARKER,
        sub_patches=[
            TextPatch(
                name="awq_moe_keys_remap",
                anchor=_ANCHOR,
                replacement=_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=["Genesis G4_04"],
        patch_id=patch_id,
    )


def register_for_manifest(*, pristine_root) -> None:
    """Register G4_04's sub-patcher into the Site Map registry, using
    the pristine `gemma4.py` fixture under `pristine_root`.

    Called by `scripts/build_anchor_manifest.py` and by
    `tests/unit/infra/conftest.py` (STABLE ratchet seed). Idempotent:
    re-registering with the same patcher is a no-op.
    """
    from vllm.sndr_core.wiring.patcher_registry import register_text_patcher

    register_text_patcher(
        "G4_04.Sub-1",
        _make_patcher_for_fixture(
            "G4_04 Sub-1 gemma4.py (build mode)",
            pristine_root / "gemma4.py",
            patch_id="G4_04.Sub-1",
        ),
    )


__all__ = [
    "GENESIS_G4_04_MARKER", "apply", "is_applied", "register_for_manifest",
]

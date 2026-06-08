# SPDX-License-Identifier: Apache-2.0
"""G4_05 — vendor vllm#42069: DFlash drafter backend autoselect for Gemma 4.

================================================================
WHAT IT FIXES
================================================================

When the spec-decode target is Gemma 4, ``Gemma4Config.verify_and_update_config``
force-locks ``attention_config.backend`` to ``TRITON_ATTN`` to prevent
mixed-backend numerical divergence within the target's intra-forward
(sliding + global attention layers). That lock propagates to the
DFlash drafter via ``DFlashProposer._create_draft_vllm_config`` —
but DFlash uses **non-causal** attention which TRITON_ATTN rejects:

    ValueError: Selected backend AttentionBackendEnum.TRITON_ATTN is not valid
                for this configuration.
                Reason: ['non-causal attention not supported']

Result: Gemma 4 + DFlash is structurally unrunnable upstream today,
even on hardware that has working non-causal backends (Hopper, etc.).

Upstream fix vllm#42069 (OPEN as of 2026-05-17) is a 1-line addition
(``backend=None``) so the drafter's autoselect path picks a non-causal-
capable backend.

This patch vendors that 1-line fix via TextPatcher so DFlash works
**today** for Hopper / Blackwell operators, and is harmless for Ampere
operators (G4_03 still refuses non-causal drafters there unless G4_10
is enabled).

================================================================
SAFETY MODEL
================================================================

* default_on: True (1-line, no harm on platforms where DFlash is already
  blocked elsewhere — G4_03 catches Ampere)
* env_flag: GENESIS_ENABLE_G4_05_GEMMA4_DFLASH_BACKEND_AUTOSELECT
* superseded_by: vllm#42069 when it merges

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/pull/42069 (OPEN)
  * Diff (saved locally): sndr_private/research/gemma4/vllm_prs/pr_42069.diff
"""
from __future__ import annotations

from sndr.kernel import TextPatch, TextPatcher
from sndr.engines.vllm.detection.guards import resolve_vllm_file

from sndr.engines.vllm.patches.model_compat.gemma4._gemma4_detect import env_truthy

GENESIS_G4_05_MARKER = (
    "Genesis G4_05 gemma4 DFlash drafter backend=None autoselect v1 "
    "(vendors vllm#42069)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_05_GEMMA4_DFLASH_BACKEND_AUTOSELECT"


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


# The original method has this signature:
#
#     @override
#     def _create_draft_vllm_config(self) -> VllmConfig:
#         base = super()._create_draft_vllm_config()
#         return replace(
#             base,
#             attention_config=replace(
#                 base.attention_config,
#                 use_non_causal=True,
#             ),
#         )
#
# We patch the inner replace() to also clear backend=None.

_ANCHOR = (
    "        return replace(\n"
    "            base,\n"
    "            attention_config=replace(\n"
    "                base.attention_config,\n"
    "                use_non_causal=True,\n"
    "            ),\n"
    "        )"
)

_REPLACEMENT = (
    "        # === Genesis G4_05 (vendor vllm#42069) START ===\n"
    "        # Clear the drafter's backend lock so autoselect picks a\n"
    "        # non-causal-capable attention backend (FLEX_ATTENTION, FLASHINFER,\n"
    "        # or Genesis G4_10 Triton kernel on Ampere if registered).\n"
    "        # The target's TRITON_ATTN lock remains untouched — only the\n"
    "        # drafter is affected, which is correct because target and drafter\n"
    "        # have independent KV caches and forwards.\n"
    "        return replace(\n"
    "            base,\n"
    "            attention_config=replace(\n"
    "                base.attention_config,\n"
    "                use_non_causal=True,\n"
    "                backend=None,\n"
    "            ),\n"
    "        )\n"
    "        # === Genesis G4_05 END ==="
)


def _make_patcher() -> TextPatcher | None:
    # K.1.R.R.6 (2026-05-29): fixed path prefix — resolve_vllm_file expects
    # path relative to vllm package root (no "vllm/" prefix). Bug found via
    # gemma4 boot log audit. Patch is retired but anchor lookup still runs;
    # without fix, prints "not resolvable in this pin" warning unnecessarily.
    target_file = resolve_vllm_file("v1/spec_decode/dflash.py")
    if target_file is None:
        return None
    return TextPatcher(
        patch_name="G4_05 gemma4 DFlash backend autoselect",
        target_file=target_file,
        marker=GENESIS_G4_05_MARKER,
        sub_patches=[
            TextPatch(
                name="dflash_backend_autoselect",
                anchor=_ANCHOR,
                replacement=_REPLACEMENT,
                required=True,
                upstream_merged_markers=[
                    "backend=None,",  # presence in this region == upstream merged
                ],
                on_upstream_merge="skip_silently",
            ),
        ],
        upstream_drift_markers=["Genesis G4_05"],
    )


def apply() -> tuple[str, str]:
    if not _env_enabled():
        return "skipped", (
            f"G4_05 disabled (set {_ENV_ENABLE}=1 to vendor vllm#42069 — "
            "lets DFlash drafter autoselect a working backend on non-Ampere)"
        )
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/spec_decode/dflash.py not resolvable in this pin"
    result, failure = patcher.apply()
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "G4_05 applied: DFlash drafter now autoselects a non-causal-capable "
            "backend on Gemma 4 targets (vllm#42069 vendored). On Ampere SM 8.6 "
            "the backend matrix wall still blocks unless G4_10 Triton kernel is "
            "registered."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    from sndr.kernel import marker_present_in_target
    patcher = _make_patcher()
    if patcher is None:
        return False
    return marker_present_in_target(patcher)


__all__ = ["GENESIS_G4_05_MARKER", "apply", "is_applied"]

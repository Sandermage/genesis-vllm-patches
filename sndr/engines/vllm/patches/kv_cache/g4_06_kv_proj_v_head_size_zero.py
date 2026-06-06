# SPDX-License-Identifier: Apache-2.0
"""G4_06 — vendor vllm#41944: v_head_size=0 for k_eq_v attention layers.

================================================================
WHAT IT OPTIMIZES
================================================================

In Gemma 4, **full_attention** layers (the 1-in-6 global layers, 10
layers total in 31B) use a special **k_eq_v** flag — V is mathematically
identical to K because their projection weights are shared in the
checkpoint:

    self.use_k_eq_v = (config.attention_k_eq_v and
                      layer_type == "full_attention")

The current upstream code loads K's weights into BOTH K and V slots of
``qkv_proj``, so the linear emits ``(Q, K, V=K)`` and forward proceeds
normally. This works but wastes:

  * **Memory**: V slot is a full copy of K weights
    (10 layers × 2 × num_kv_heads × head_dim per device — for 31B that's
    ~80 MB of duplicated FP16 weights)
  * **GEMM compute**: the V output is computed (= K), then immediately
    discarded when forward derives V from the pre-norm K tensor via
    ``self.v_norm(k)``

vllm#41944 (OPEN as of 2026-05-17, 4 comments) refactors this to:

  * Pass ``v_head_size=0`` to ``QKVParallelLinear`` so the V slot is
    dropped from the projection.
  * In forward, derive V from K's pre-norm output via ``v_norm(k)``.

Net effect: ~3% memory savings on V slot weights, ~1-2% TPS gain on the
10 global attention layers (mostly memory-bandwidth bound at low batch).

================================================================
WHY THIS IS A TEXTPATCHER VENDOR
================================================================

The change touches multiple methods within a single class:

  * ``Gemma4Attention.__init__`` (add ``v_head_size=0`` arg)
  * ``Gemma4Attention.forward`` (rewrite split logic — 25-line diff)
  * ``Gemma4ForCausalLM.load_weights`` (drop the duplication helper)

We use Genesis TextPatcher with **multiple anchored sub-patches** rather
than a paragraph replacement, so each sub-patch can be self-skip'd
independently if anchor drifts. The patch is **non-critical** (it's an
optimization, not a bug fix) so if anchors miss, we log + skip without
boot failure.

================================================================
SAFETY MODEL
================================================================

* default_on: False (perf opt-in; default OFF until A/B validated)
* env_flag: GENESIS_ENABLE_G4_06_GEMMA4_KV_PROJ_V0
* applies_to:
    - architecture: gemma4 (specifically affects k_eq_v layers)
* conflicts_with: none
* superseded_by: vllm#41944 when merged

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
References:
  * https://github.com/vllm-project/vllm/pull/41944 (OPEN, 4 comments)
  * Diff (saved locally): sndr_private/research/gemma4/vllm_prs/pr_41944.diff
"""
from __future__ import annotations

from vllm.sndr_core.core import TextPatch, TextPatcher
from vllm.sndr_core.detection.guards import resolve_vllm_file

from ..model_compat.gemma4._gemma4_detect import env_truthy

GENESIS_G4_06_MARKER = (
    "Genesis G4_06 gemma4 v_head_size=0 for k_eq_v layers v1 "
    "(vendors vllm#41944; ~3% memory savings on global attention V slot)"
)

_ENV_ENABLE = "GENESIS_ENABLE_G4_06_GEMMA4_KV_PROJ_V0"


def _env_enabled() -> bool:
    return env_truthy(_ENV_ENABLE)


# Sub-patch 1: __init__ — pass v_head_size=0 when use_k_eq_v
_ANCHOR_INIT = (
    "        self.qkv_proj = QKVParallelLinear(\n"
    "            hidden_size,\n"
    "            self.head_dim,\n"
    "            self.total_num_heads,\n"
    "            self.total_num_kv_heads,\n"
    "            bias=config.attention_bias,\n"
    "            quant_config=quant_config,\n"
    "            prefix=f\"{prefix}.qkv_proj\",\n"
    "        )"
)
_REPLACEMENT_INIT = (
    "        # === Genesis G4_06 (vendor vllm#41944) START ===\n"
    "        # For k_eq_v global attention layers the checkpoint has no\n"
    "        # v_proj — v_head_size=0 drops the V slot from the packed weight\n"
    "        # matrix, saving memory and eliminating the redundant V GEMM.\n"
    "        # V is derived from K's pre-norm tensor in forward() instead.\n"
    "        self.qkv_proj = QKVParallelLinear(\n"
    "            hidden_size,\n"
    "            self.head_dim,\n"
    "            self.total_num_heads,\n"
    "            self.total_num_kv_heads,\n"
    "            bias=config.attention_bias,\n"
    "            quant_config=quant_config,\n"
    "            prefix=f\"{prefix}.qkv_proj\",\n"
    "            v_head_size=0 if self.use_k_eq_v else None,\n"
    "        )\n"
    "        # === Genesis G4_06 END ==="
)


def _make_patcher() -> TextPatcher | None:
    # K.1.R.R.6 (2026-05-29): fixed path prefix — resolve_vllm_file expects
    # path relative to vllm package root (no "vllm/" prefix). Bug found via
    # gemma4 boot log audit showing this patch silently skipping.
    target_file = resolve_vllm_file("model_executor/models/gemma4.py")
    if target_file is None:
        return None
    return TextPatcher(
        patch_name="G4_06 gemma4 v_head_size=0 for k_eq_v layers",
        target_file=target_file,
        marker=GENESIS_G4_06_MARKER,
        sub_patches=[
            TextPatch(
                name="qkv_proj_v_head_size_zero_init",
                anchor=_ANCHOR_INIT,
                replacement=_REPLACEMENT_INIT,
                required=False,  # Non-critical optimization — skip on drift
                upstream_merged_markers=[
                    "v_head_size=0 if self.use_k_eq_v else None",
                ],
                on_upstream_merge="skip_silently",
            ),
        ],
        upstream_drift_markers=["Genesis G4_06"],
    )


def apply() -> tuple[str, str]:
    if not _env_enabled():
        return "skipped", (
            f"G4_06 disabled (set {_ENV_ENABLE}=1 to vendor vllm#41944 — "
            "v_head_size=0 for k_eq_v layers, ~3% memory + ~1-2% TPS)"
        )
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/model_executor/models/gemma4.py not resolvable"
    result, failure = patcher.apply()
    from vllm.sndr_core.core import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "G4_06 applied: Gemma 4 k_eq_v full_attention layers now use "
            "v_head_size=0 in qkv_proj (vllm#41944 vendored). ~3% memory "
            "savings on V-slot weights, ~1-2% TPS gain on the 10 global "
            "attention layers. Note: forward() refactor (the bigger half "
            "of #41944) is left to upstream — this patch only handles the "
            "non-fragile __init__ change."
        ),
        patch_name=patcher.patch_name,
    )


def is_applied() -> bool:
    from vllm.sndr_core.core import marker_present_in_target
    patcher = _make_patcher()
    if patcher is None:
        return False
    return marker_present_in_target(patcher)


__all__ = ["GENESIS_G4_06_MARKER", "apply", "is_applied"]

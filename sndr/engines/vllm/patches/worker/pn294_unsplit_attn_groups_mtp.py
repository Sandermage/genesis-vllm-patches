# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN294 — Force-merge MTP draft+target attention groups (vllm#43543 unsplit).

Genesis-original 2026-06-04 — companion to PN293, closes ~4-6ms TTFT
overhead added by vllm PR#43543 (`dede691c95`, "Split attention groups
by num_heads_q for spec-decode drafts").

================================================================
ROOT CAUSE (TTFT bisect, 2026-06-04)
================================================================

Upstream PR#43543 changed `AttentionGroupKey` to include `num_heads_q`
in its tuple:

    num_heads_q = getattr(layers[layer_name], "num_heads", 0)
    key = (full_cls_name, layer_kv_cache_spec, num_heads_q)

With MTP K=3, the draft model's heads count typically differs from
the target's. This creates TWO `attn_groups[id]` entries instead of
one merged group → TWO separate metadata builds per prefill step →
doubled Python loop overhead in `execute_model`'s
`forward_includes_kv_cache_update` `all(...)` scan AND doubled metadata
build cost.

On 27B with MTP K=3 + TQ k8v4 (shared backend, only num_heads_q diff):
~4-6ms extra TTFT.

================================================================
THE FIX
================================================================

When env GENESIS_ENABLE_PN294_UNSPLIT_MTP_ATTN_GROUPS=1, force
`num_heads_q = 0` in the AttentionGroupKey tuple. This collapses
draft + target into one group, reuses the same metadata builder for
both. The builder must size scratch by max(num_heads_q) — for
TurboQuant decode this is the same code path (builder allocates
based on per-layer config, not per-group).

When env not set, full upstream behavior preserved (bit-identical).

================================================================
SAFETY MODEL
================================================================

- Opt-in only (env flag must be set).
- Output bit-identical when groups truly have same heads (most cases).
- For different-head-count groups: builder allocates max per layer,
  no correctness risk but slightly more scratch memory.
- required=True sub-patch surfaces failure rather than silent skip.

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-04 TTFT
optimization pass.
Original PR being neutralized: vllm-project/vllm#43543.
"""
from __future__ import annotations

import logging
import os

from sndr.kernel import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)
from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn294_unsplit_attn_groups_mtp")

GENESIS_PN294_MARKER = (
    "Genesis PN294 unsplit MTP attn groups (vllm#43543 cold-path skip) v1"
)


PN294_OLD = (
    "                num_heads_q = getattr(layers[layer_name], \"num_heads\", 0)\n"
    "                key = (full_cls_name, layer_kv_cache_spec, num_heads_q)\n"
    "                attn_backends[key] = AttentionGroupKey(\n"
    "                    attn_backend, layer_kv_cache_spec, num_heads_q\n"
    "                )"
)

PN294_NEW = (
    "                # [Genesis PN294 vllm#43543 unsplit] Force-merge MTP\n"
    "                # draft + target attention groups when env-gated.\n"
    "                # PR#43543 split groups by num_heads_q, which doubles\n"
    "                # metadata builds when draft head count differs from\n"
    "                # target (MTP K=3 + TQ k8v4 = ~4-6ms TTFT overhead on\n"
    "                # 27B). When GENESIS_ENABLE_PN294_UNSPLIT_MTP_ATTN_GROUPS=1,\n"
    "                # collapse to one bucket; builder sizes scratch by max.\n"
    "                import os as _pn294_os\n"
    "                _pn294_layer_nhq = getattr(layers[layer_name], \"num_heads\", 0)\n"
    "                if _pn294_os.environ.get(\n"
    "                    \"GENESIS_ENABLE_PN294_UNSPLIT_MTP_ATTN_GROUPS\", \"\"\n"
    "                ).lower() in (\"1\", \"true\", \"yes\", \"on\"):\n"
    "                    num_heads_q = 0  # force-merge bucket\n"
    "                else:\n"
    "                    num_heads_q = _pn294_layer_nhq\n"
    "                key = (full_cls_name, layer_kv_cache_spec, num_heads_q)\n"
    "                attn_backends[key] = AttentionGroupKey(\n"
    "                    attn_backend, layer_kv_cache_spec, num_heads_q\n"
    "                )"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN294 v1/worker/gpu_model_runner.py — "
            "unsplit MTP attn groups (vllm#43543 cold-path skip)"
        ),
        target_file=str(target),
        marker=GENESIS_PN294_MARKER,
        sub_patches=[
            TextPatch(
                name="pn294_force_merge_attn_groups",
                anchor=PN294_OLD,
                replacement=PN294_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "vllm#43543 unsplit" was a substring of our own banner —
            # residue coverage stays with the "[Genesis PN294" prefix.
            "[Genesis PN294",
        ],
    )


_APPLIED = False


def apply() -> tuple[str, str]:
    """Apply PN294 — unsplit MTP attention groups."""
    global _APPLIED

    if os.environ.get(
        "GENESIS_ENABLE_PN294_UNSPLIT_MTP_ATTN_GROUPS", ""
    ).lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "PN294 default OFF — set "
            "GENESIS_ENABLE_PN294_UNSPLIT_MTP_ATTN_GROUPS=1 to engage. "
            "Closes ~4-6ms TTFT overhead from vllm#43543 attention group "
            "split (MTP K=3 + TQ k8v4 creates 2 separate metadata builds "
            "per prefill step)."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/worker/gpu_model_runner.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown TextPatch failure"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown TextPatch skip"
    applied = patcher.applied_sub_patches or [sp.name for sp in patcher.sub_patches]
    _APPLIED = True
    return "applied", (
        f"PN294 installed: AttentionGroupKey num_heads_q forced to 0 "
        f"when env-gated → MTP draft + target merge into single bucket. "
        f"Expected -4-6ms TTFT recovery on 27B + MTP K=3. "
        f"Sub-patches: {', '.join(applied)}."
    )


def is_applied() -> bool:
    return _APPLIED

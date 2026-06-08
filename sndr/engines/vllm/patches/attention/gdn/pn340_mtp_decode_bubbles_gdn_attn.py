# SPDX-License-Identifier: Apache-2.0
"""PN340 — vendor open PR vllm#43955 (gdn_attn.py portion).

Open upstream PR `vllm-project/vllm#43955`_ by ``Nekofish-L``: *[Perf]
Reduce MTP decode bubbles for Qwen3.5 hybrid models*. The PR identifies
that the GDN backend's metadata-build hot path launches a tiny CUDA
kernel (``torch.arange``) on every call, plus does CPU-mask indexing
(``block_table_tensor[spec_sequence_masks_cpu, ...]``) when a simple
forward slice would suffice because spec rows are already compacted to
the front of the batch and padded rows live at the back.

This file vendors the gdn_attn.py portion of the PR (the simpler half —
the gpu_model_runner.py portion lives in a separate Genesis patch if
we choose to vendor it). Three text-patches::

  1. ``__init__``  — add ``self.spec_token_arange`` preallocated buffer
                     right after ``self.spec_token_indx`` allocation
  2. ``build()``   — replace dynamic ``torch.arange`` + CPU-mask
                     indexing with slice into the preallocated buffer
  3. ``build()``   — conditional ``copy_``: skip when ``spec_token_indx``
                     already points at ``self.spec_token_arange``
                     (no-op copy elimination)

Expected impact:
  * Per metadata build: 1 fewer tiny kernel launch + 1 avoided index
    kernel + 1 avoided device-to-device copy (when arange path active).
  * Fires every time the GDN backend builds metadata for spec-decode
    batches (every step on hybrid + MTP K=3 path).
  * Author's profile screenshots show measurably smaller inter-step
    gaps in the Qwen3.5 trace.

Compat with our stack:
  * Direct hit — Qwen3.6-A3B FP8 + TQ k8v4 + MTP K=3 uses this exact
    code path on every decode step.
  * Anchors stable across PN125 / PN286 / PN204 — same file the dual-
    stream input projection touches.

Risk: medium-low. Anchor fragility on subsequent pin bumps is the
usual concern (this is an open PR; upstream may rebase before merge).
The three sub-patches are independent (``required=False`` each), so
partial anchor drift drops only the unmatched sub.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#43955 (open as of 2026-06-09).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn340_mtp_decode_bubbles_gdn_attn")

GENESIS_PN340_MARKER = (
    "Genesis PN340 vendor of vllm#43955 (MTP decode bubbles, gdn_attn.py) v1"
)


# ─── Sub-patch 1: add spec_token_arange preallocated buffer ────────────
# Anchor on the unique sequence ``self.spec_token_indx`` block followed
# by ``self.non_spec_token_indx``.
PN340_INIT_OLD = (
    "        self.spec_token_indx: torch.Tensor = torch.empty(\n"
    "            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),\n"
    "            dtype=torch.int32,\n"
    "            device=device,\n"
    "        )\n"
    "        self.non_spec_token_indx: torch.Tensor = torch.empty(\n"
)
PN340_INIT_NEW = (
    "        self.spec_token_indx: torch.Tensor = torch.empty(\n"
    "            (self.decode_cudagraph_max_bs * (self.num_spec + 1),),\n"
    "            dtype=torch.int32,\n"
    "            device=device,\n"
    "        )\n"
    "        # [Genesis PN340 vendor of vllm#43955] preallocated arange buffer.\n"
    "        # Lets build() avoid launching a tiny torch.arange CUDA kernel on\n"
    "        # every metadata build — slice into this static buffer instead.\n"
    "        self.spec_token_arange: torch.Tensor = torch.arange(\n"
    "            self.decode_cudagraph_max_bs * (self.num_spec + 1),\n"
    "            dtype=torch.int32,\n"
    "            device=device,\n"
    "        )\n"
    "        self.non_spec_token_indx: torch.Tensor = torch.empty(\n"
)


# ─── Sub-patch 2: build() — replace dynamic arange + CPU-mask indexing ─
# Anchor on the unique combination of ``torch.arange(spec_token_size,``
# and the ``spec_sequence_masks_cpu, : self.num_spec + 1`` mask index.
PN340_BUILD_OLD = (
    "                spec_token_indx = torch.arange(\n"
    "                    spec_token_size,\n"
    "                    dtype=torch.int32,\n"
    "                    device=query_start_loc.device,\n"
    "                )\n"
    "                non_spec_token_indx = torch.empty(\n"
    "                    0, dtype=torch.int32, device=query_start_loc.device\n"
    "                )\n"
    "                # Filter by spec_sequence_masks to exclude padded sequences\n"
    "                spec_state_indices_tensor = block_table_tensor[\n"
    "                    spec_sequence_masks_cpu, : self.num_spec + 1\n"
    "                ]\n"
)
PN340_BUILD_NEW = (
    "                # [Genesis PN340 vendor of vllm#43955] avoid tiny CUDA\n"
    "                # kernel launches on every metadata build. Spec rows are\n"
    "                # compacted to the front of the batch; padded rows live\n"
    "                # at the back — slice instead of mask-index.\n"
    "                spec_token_indx = self.spec_token_arange[:spec_token_size]\n"
    "                non_spec_token_indx = self.non_spec_token_indx[:0]\n"
    "                spec_state_indices_tensor = block_table_tensor[\n"
    "                    :num_spec_decodes, : self.num_spec + 1\n"
    "                ]\n"
)


# ─── Sub-patch 3: build() — conditional copy_ for spec_token_indx ──────
# When ``spec_token_indx`` is already the preallocated ``spec_token_arange``
# (sub-patch 2 path), the ``copy_`` is a redundant device-to-device copy.
# Skip it. Also gate the non_spec_token_indx copy on numel > 0.
PN340_COPY_OLD = (
    "            assert non_spec_token_indx is not None and spec_token_indx is not None\n"
    "            self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(\n"
    "                non_spec_token_indx, non_blocking=True\n"
    "            )\n"
    "            non_spec_token_indx = self.non_spec_token_indx[\n"
    "                : non_spec_token_indx.size(0)\n"
    "            ]\n"
    "\n"
    "            self.spec_token_indx[: spec_token_indx.size(0)].copy_(\n"
    "                spec_token_indx, non_blocking=True\n"
    "            )\n"
    "            spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]\n"
)
PN340_COPY_NEW = (
    "            assert non_spec_token_indx is not None and spec_token_indx is not None\n"
    "            # [Genesis PN340 vendor of vllm#43955] skip no-op copies.\n"
    "            if non_spec_token_indx.numel() > 0:\n"
    "                self.non_spec_token_indx[: non_spec_token_indx.size(0)].copy_(\n"
    "                    non_spec_token_indx, non_blocking=True\n"
    "                )\n"
    "                non_spec_token_indx = self.non_spec_token_indx[\n"
    "                    : non_spec_token_indx.size(0)\n"
    "                ]\n"
    "\n"
    "            if spec_token_indx.data_ptr() == self.spec_token_arange.data_ptr():\n"
    "                # Already the static arange buffer — no copy needed.\n"
    "                spec_token_indx = self.spec_token_arange[: spec_token_indx.size(0)]\n"
    "            else:\n"
    "                self.spec_token_indx[: spec_token_indx.size(0)].copy_(\n"
    "                    spec_token_indx, non_blocking=True\n"
    "                )\n"
    "                spec_token_indx = self.spec_token_indx[: spec_token_indx.size(0)]\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN340", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply PN340 — vendor PR vllm#43955 gdn_attn.py portion."""
    if _env_disabled():
        return "skipped", "PN340 disabled via GENESIS_DISABLE_PN340=1"

    target = resolve_vllm_file("v1/attention/backends/gdn_attn.py")
    if target is None:
        return "skipped", (
            "PN340: gdn_attn.py not found in vllm install — pin may "
            "predate the v1 GDN backend or have a different layout"
        )

    patcher = TextPatcher(
        patch_name=(
            "PN340 v1/attention/backends/gdn_attn.py — vendor vllm#43955 "
            "(MTP decode bubbles, slice instead of arange+mask)"
        ),
        target_file=str(target),
        marker=GENESIS_PN340_MARKER,
        sub_patches=[
            TextPatch(
                name="pn340_init_spec_token_arange",
                anchor=PN340_INIT_OLD,
                replacement=PN340_INIT_NEW,
                required=False,
            ),
            TextPatch(
                name="pn340_build_slice_instead_of_mask",
                anchor=PN340_BUILD_OLD,
                replacement=PN340_BUILD_NEW,
                required=False,
            ),
            TextPatch(
                name="pn340_build_conditional_copy",
                anchor=PN340_COPY_OLD,
                replacement=PN340_COPY_NEW,
                required=False,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN340",
            # Upstream marker if vllm#43955 lands:
            "self.spec_token_arange",
        ],
    )

    try:
        result, failure = patcher.apply()
    except Exception as e:
        log.warning("[PN340] apply() raised %s — leaving upstream gdn_attn", e)
        return "skipped", f"PN340 raised at apply: {e!r}"

    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "anchor drift / not eligible"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "skipped", f"PN340: {reason}{detail}"

    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "failed", f"PN340: {reason}{detail}"

    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN340 idempotent: marker already present (or upstream merged)"

    applied = ", ".join(patcher.applied_sub_patches) or "(unknown)"
    return "applied", (
        f"PN340 applied: gdn_attn.py MTP decode-bubble reduction via "
        f"sub-patches [{applied}]. Closes per-step torch.arange CUDA "
        f"launches + avoidable CPU-mask indexing on hybrid GDN + MTP "
        f"K=3 hot path. Vendor of OPEN PR vllm#43955 (Nekofish-L)."
    )


def is_applied() -> bool:
    target = resolve_vllm_file("v1/attention/backends/gdn_attn.py")
    if target is None:
        return False
    try:
        return GENESIS_PN340_MARKER in target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

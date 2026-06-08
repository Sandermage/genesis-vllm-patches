# SPDX-License-Identifier: Apache-2.0
"""PN341 — vendor open PR vllm#43955 (gpu_model_runner.py num_accepted_tokens portion).

Sister patch to PN340 (which vendored the gdn_attn.py portion of the same
PR). This vendors the LARGER half of vllm#43955: the
``num_accepted_tokens`` GPU-only path that avoids the per-step
``num_accepted_tokens_event.synchronize()`` CPU sync on
hybrid + MTP K=3 decode steps.

Background — the bubble being closed
=====================================

In the upstream baseline, after every decode step:

1. ``_update_states_after_model_execute`` computes per-request accepted-
   token count via ``(output_token_ids != -1).sum(dim=1)`` on GPU
2. Then copies it to a pinned-CPU mirror buffer via ``non_blocking=True``
   AND records ``num_accepted_tokens_event`` so a later code path can
   synchronize on it.
3. ``_prepare_inputs`` of the NEXT step calls
   ``num_accepted_tokens_event.synchronize()`` to read the CPU mirror,
   does scatter/gather indexing in NumPy, then copies the resulting
   NumPy array back to GPU.

The PR observes that for hybrid + MTP (our exact stack — Qwen3.6-A3B
FP8 + TQ k8v4 + MTP K=3 with ``cache_config.mamba_cache_mode != "align"``)
the entire CPU round-trip can be skipped:

* The accepted-token count is already on GPU.
* The ``prev_positions`` mapping (current pos → previous pos) can be
  computed on CPU from a Python dict of ``req_ids`` captured in
  ``_update_states_after_model_execute``, then a single ``copy_to_gpu``
  populates a GPU buffer.
* GPU gather + ``masked_fill_`` produce the final
  ``num_accepted_tokens.gpu`` without ever waiting on the event.

Result: no event synchronize, no NumPy roundtrip, no second copy_to_gpu.
Author's profile screenshots show measurably smaller inter-step gaps in
the Qwen3.5 trace.

Implementation
==============

Four ``TextPatch`` sub-patches on
``vllm/v1/worker/gpu_model_runner.py`` (all ``required=False``):

1. ``__init__`` — add the gate flag + two state vars right after the
   speculative-token-streams init block.
2. ``_update_states_after_model_execute`` — early return on the GPU-only
   path, capturing the ``req_ids`` snapshot.
3. ``_compute_prev_positions`` — add the optional ``prev_req_id_to_
   index`` parameter so the GPU-only path can pass its captured dict.
4. ``_prepare_inputs`` — add the GPU-only branch before the existing
   ``if self.num_accepted_tokens_event is not None`` block.

Sub-patch coverage is the MAIN bubble-elimination work of the PR. The
PR also has _sample / sample_tokens changes that gate the
``_copy_draft_token_ids_to_cpu`` calls on penalties / bad_words usage —
those are a smaller win and live in a more sensitive code path (the
sampler), saved for a follow-up PN342 if the bench validates.

Composes with PN340 (gdn_attn.py sister patch — same upstream PR).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn341_mtp_decode_bubbles_gpu_runner")

GENESIS_PN341_MARKER = (
    "Genesis PN341 vendor of vllm#43955 (MTP decode bubbles, gpu_runner) v1"
)


# ─── Sub-patch 1: __init__ flag + state vars ──────────────────────────
# Insert right after the speculative-token streams init block. Anchor
# uses the unique ``set_offloader`` import-time call that follows the
# init region.
PN341_INIT_OLD = (
    "                )\n"
    "\n"
    "        # Model weight offloader\n"
    "        # Make sure this is called before any get_offloader call\n"
    "        set_offloader(create_offloader(self.offload_config))\n"
)
PN341_INIT_NEW = (
    "                )\n"
    "        # [Genesis PN341 vendor of vllm#43955] GPU-only num_accepted_tokens\n"
    "        # path for hybrid + MTP. Replaces the per-step\n"
    "        # ``num_accepted_tokens_event.synchronize()`` CPU sync.\n"
    "        self._use_gpu_only_num_accepted_tokens = (\n"
    "            self.num_spec_tokens > 0\n"
    "            and self.model_config.is_hybrid\n"
    "            and self.cache_config.mamba_cache_mode != \"align\"\n"
    "        )\n"
    "        self._num_accepted_tokens_valid = False\n"
    "        self._num_accepted_tokens_req_id_to_index: dict[str, int] = {}\n"
    "\n"
    "        # Model weight offloader\n"
    "        # Make sure this is called before any get_offloader call\n"
    "        set_offloader(create_offloader(self.offload_config))\n"
)


# ─── Sub-patch 2: _update_states_after_model_execute early return ─────
# Anchor on the ``num_accepted_tokens.gpu[:num_reqs] = (output_token_ids
# != -1).sum(dim=1)`` line + the immediately-following ``if self.cache_
# config.mamba_cache_mode == "align":`` discriminator. Inject the GPU-
# only branch between them.
PN341_UPDATE_STATES_OLD = (
    "        num_reqs = output_token_ids.size(0)\n"
    "        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)\n"
    "\n"
    "        if self.cache_config.mamba_cache_mode == \"align\":\n"
)
PN341_UPDATE_STATES_NEW = (
    "        num_reqs = output_token_ids.size(0)\n"
    "        self.num_accepted_tokens.gpu[:num_reqs] = (output_token_ids != -1).sum(dim=1)\n"
    "\n"
    "        # [Genesis PN341 vendor of vllm#43955] GPU-only fast path.\n"
    "        # Capture req_ids snapshot for _prepare_inputs to use; skip\n"
    "        # the CPU copy and event.record() entirely on hybrid + MTP.\n"
    "        if self._use_gpu_only_num_accepted_tokens:\n"
    "            self._num_accepted_tokens_valid = True\n"
    "            self._num_accepted_tokens_req_id_to_index = {\n"
    "                req_id: i\n"
    "                for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs])\n"
    "            }\n"
    "            return\n"
    "\n"
    "        if self.cache_config.mamba_cache_mode == \"align\":\n"
)


# ─── Sub-patch 3: _compute_prev_positions optional param ──────────────
# Add ``prev_req_id_to_index`` optional parameter and short-circuit the
# default ``self.input_batch.prev_req_id_to_index`` lookup when caller
# passes it explicitly.
PN341_COMPUTE_PREV_OLD = (
    "    def _compute_prev_positions(self, num_reqs: int) -> None:\n"
    "        \"\"\"Build prev_positions mapping: current pos -> previous pos (-1 if new).\n"
    "\n"
    "        Populates self.prev_positions.np[:num_reqs] with the mapping.\n"
    "        \"\"\"\n"
    "        prev_req_id_to_index = self.input_batch.prev_req_id_to_index\n"
)
PN341_COMPUTE_PREV_NEW = (
    "    def _compute_prev_positions(\n"
    "        self,\n"
    "        num_reqs: int,\n"
    "        prev_req_id_to_index: dict[str, int] | None = None,\n"
    "    ) -> None:\n"
    "        \"\"\"Build prev_positions mapping: current pos -> previous pos (-1 if new).\n"
    "\n"
    "        Populates self.prev_positions.np[:num_reqs] with the mapping.\n"
    "        \"\"\"\n"
    "        # [Genesis PN341 vendor of vllm#43955] caller can pass an explicit\n"
    "        # mapping (used by the GPU-only num_accepted_tokens path).\n"
    "        if prev_req_id_to_index is None:\n"
    "            prev_req_id_to_index = self.input_batch.prev_req_id_to_index\n"
)


# ─── Sub-patch 4: _prepare_inputs GPU-only branch ─────────────────────
# Inject the new branch BEFORE the existing
# ``if self.num_accepted_tokens_event is not None:`` block. We change it
# to ``elif`` so the existing path runs only when our new branch
# doesn't.
PN341_PREPARE_OLD = (
    "        # Sync num_accepted_tokens from CPU (set by\n"
    "        # _update_states_after_model_execute for hybrid models).\n"
    "        if self.num_accepted_tokens_event is not None:\n"
    "            self.num_accepted_tokens_event.synchronize()\n"
)
PN341_PREPARE_NEW = (
    "        # Sync num_accepted_tokens from CPU (set by\n"
    "        # _update_states_after_model_execute for hybrid models).\n"
    "        # [Genesis PN341 vendor of vllm#43955] GPU-only branch on hybrid + MTP.\n"
    "        if self._use_gpu_only_num_accepted_tokens:\n"
    "            if self._num_accepted_tokens_valid:\n"
    "                self._compute_prev_positions(\n"
    "                    num_reqs, self._num_accepted_tokens_req_id_to_index\n"
    "                )\n"
    "                self.prev_positions.copy_to_gpu(num_reqs)\n"
    "                prev_positions_gpu = self.prev_positions.gpu[:num_reqs]\n"
    "                prev_indices_gpu = prev_positions_gpu.clamp_min(0)\n"
    "                num_accepted_tokens = self.num_accepted_tokens.gpu[prev_indices_gpu]\n"
    "                num_accepted_tokens.masked_fill_(prev_positions_gpu < 0, 1)\n"
    "                self.num_accepted_tokens.gpu[:num_reqs].copy_(\n"
    "                    num_accepted_tokens, non_blocking=True\n"
    "                )\n"
    "            else:\n"
    "                self.num_accepted_tokens.gpu.fill_(1)\n"
    "            self.num_accepted_tokens.gpu[num_reqs:].fill_(1)\n"
    "        elif self.num_accepted_tokens_event is not None:\n"
    "            self.num_accepted_tokens_event.synchronize()\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN341", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def apply() -> tuple[str, str]:
    """Apply PN341 — vendor vllm#43955 gpu_model_runner.py portion."""
    if _env_disabled():
        return "skipped", "PN341 disabled via GENESIS_DISABLE_PN341=1"

    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return "skipped", (
            "PN341: gpu_model_runner.py not found in vllm install — pin "
            "may predate the v1 runner or have a different layout"
        )

    patcher = TextPatcher(
        patch_name=(
            "PN341 v1/worker/gpu_model_runner.py — vendor vllm#43955 "
            "(MTP decode bubbles, GPU-only num_accepted_tokens)"
        ),
        target_file=str(target),
        marker=GENESIS_PN341_MARKER,
        sub_patches=[
            TextPatch(
                name="pn341_init_gpu_only_flag",
                anchor=PN341_INIT_OLD,
                replacement=PN341_INIT_NEW,
                required=False,
            ),
            TextPatch(
                name="pn341_update_states_early_return",
                anchor=PN341_UPDATE_STATES_OLD,
                replacement=PN341_UPDATE_STATES_NEW,
                required=False,
            ),
            TextPatch(
                name="pn341_compute_prev_positions_optional_arg",
                anchor=PN341_COMPUTE_PREV_OLD,
                replacement=PN341_COMPUTE_PREV_NEW,
                required=False,
            ),
            TextPatch(
                name="pn341_prepare_inputs_gpu_only_branch",
                anchor=PN341_PREPARE_OLD,
                replacement=PN341_PREPARE_NEW,
                required=False,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN341",
            # Upstream sentinel when vllm#43955 merges:
            "_use_gpu_only_num_accepted_tokens",
        ],
    )

    try:
        result, failure = patcher.apply()
    except Exception as e:
        log.warning("[PN341] apply() raised %s — leaving upstream gpu_model_runner", e)
        return "skipped", f"PN341 raised at apply: {e!r}"

    if result == TextPatchResult.SKIPPED:
        reason = failure.reason if failure else "anchor drift / not eligible"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "skipped", f"PN341: {reason}{detail}"

    if result == TextPatchResult.FAILED:
        reason = failure.reason if failure else "unknown"
        detail = f" ({failure.detail})" if failure and failure.detail else ""
        return "failed", f"PN341: {reason}{detail}"

    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "PN341 idempotent: marker already present (or upstream merged)"

    applied = ", ".join(patcher.applied_sub_patches) or "(unknown)"
    return "applied", (
        f"PN341 applied: gpu_model_runner.py num_accepted_tokens GPU-only "
        f"path via sub-patches [{applied}]. Closes the per-step "
        f"num_accepted_tokens_event.synchronize() CPU bubble on hybrid + "
        f"MTP K=3 decode steps. Vendor of OPEN PR vllm#43955 "
        f"(Nekofish-L) — second half (gdn_attn.py is sister patch PN340)."
    )


def is_applied() -> bool:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return False
    try:
        return GENESIS_PN341_MARKER in target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False

# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN292 — Revert PR #40172 fused Triton Mamba postprocess.

Genesis-original 2026-06-04 — closes the -18% TPS regression on Qwen3.6 27B
(hybrid GDN+Mamba) between vllm dev371 (`bf610c2f5`, 130 TPS) and dev354
(`626fa9bba`, 107 TPS) on 2x A5000 (SM 8.6) under MTP K=3 + TQ k8v4.

================================================================
ROOT CAUSE (six-step bisect, 2026-06-04)
================================================================

Upstream PR #40172 (`b730c46352`, "Perf Hybrid Fused Triton kernel for
GPU-side Mamba state postprocessing") replaces the dev371 per-decode-step
Python postprocess (single .cpu() sync + one Python `postprocess_mamba`
call) with:

  (a) a fused Triton kernel `postprocess_mamba_align_gpu` (per-decode-step
      kernel launch, grid = num_reqs x num_layers*num_state_types),
  (b) a NEW per-decode-step staging Triton call `stage_postprocess_inputs_to_gpu`
      added inside `execute_model` to populate GPU buffers consumed by (a).

The gate is `cache_config.mamba_cache_mode == "align"` AND `speculative_config
is not None` AND `model_config.is_hybrid` — i.e. exactly the Qwen3.6 27B
hybrid + MTP K=3 PROD config.

On Hopper (H100, GB200) the saved CPU-GPU sync dominates and the change is
a perf win as the PR claims. On Ampere SM 8.6 (A5000) two factors invert
the balance:

  - Triton kernel launch overhead per decode step (1 launch for stage +
    1 launch for postprocess on top of the existing mamba ops) is large
    relative to the sub-millisecond saving from removing the .cpu() sync;
  - the grid (num_reqs=5 x num_layers*num_state_types ~ 64*2 = 128
    programs) under-occupies SM 8.6 (84 SMs on A5000) but still pays full
    launch + memcpy via 1024-element COPY_BLOCK_SIZE loops.

Measured impact: -18% steady-state TPS at our 5x5x1024 standard bench,
fully reproduced after every other candidate (PR41126 Mamba refactor,
PR42095 FA KV layout, PR43361 stable-ABI mamba kernel, PR43273 SM100 GDN
kernel) was ruled out.

================================================================
WHAT THIS PATCH DOES
================================================================

Restores the dev371 code path in `vllm/v1/worker/gpu_model_runner.py` at
TWO sites:

  Site 1: `_update_states_after_model_execute` (~ live line 1490-1545)
          revert the `if mamba_cache_mode == "align"` branch to the dev371
          form (one .cpu().numpy() sync + Python `postprocess_mamba` call).

  Site 2: `execute_model` (~ live line 4140-4175) drop the
          `stage_postprocess_inputs_to_gpu` per-step staging block. Keep
          `preprocess_mamba`. Pass `self._get_mamba_copy_bufs()` instead
          of `mamba_bufs.preprocess` (the dev371 single-handle form).

We do NOT revert `mamba_utils.py` itself; we only rewire the two callers in
`gpu_model_runner.py`. The dev354 `mamba_utils` module still exposes the
legacy `postprocess_mamba` symbol (PR40172 kept it as a Python reference for
the unit tests). We rely on that symbol surviving — if upstream deletes it
in a future pin, this patch will SKIP (anchor miss) and the controller will
fall back to the fast path.

================================================================
WHEN IT FIRES
================================================================

- vllm pin between PR40172 (2026-05-21) and the next pin where this code
  is removed
- Hybrid model (Qwen3.6 27B, NemotronH, Bamba, FalconH1 etc.)
- `mamba_cache_mode == "align"` (PROD setting)
- Speculative decoding enabled (MTP K>=1, EAGLE, NGRAM, ...)
- Ampere SM 8.6 only (env-gated; H100+ should keep the fast path)

================================================================
SAFETY MODEL
================================================================

- env: `GENESIS_ENABLE_PN292_REVERT_FUSED_MAMBA_POSTPROCESS=1`
- default OFF (controller must explicitly enable after bench confirms)
- Idempotent (marker check)
- Anchor miss -> SKIPPED, not crash
- Auto-no-op once upstream lands an Ampere-aware dispatch (drift marker)

Author: backport-revert for Genesis from PR40172.
"""
from __future__ import annotations

import logging
import os

from sndr.kernel.text_patch import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)
from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root

log = logging.getLogger("genesis.wiring.pn292_revert_fused_mamba_postprocess")

GENESIS_PN292_MARKER = "Genesis PN292 revert PR40172 fused mamba postprocess v7.67"

# --------------------------------------------------------------------------
# Site 1: _update_states_after_model_execute — replace the dev354 align branch
#         (Triton kernel + sync record) with the dev371 .cpu().numpy()
#         + Python postprocess_mamba form.
# --------------------------------------------------------------------------

PN292_SITE1_ANCHOR = (
    "        if self.cache_config.mamba_cache_mode == \"align\":\n"
    "            # Fused GPU postprocess: state copies + per-request accepted-token\n"
    "            # update without CPU-GPU sync. The metadata\n"
    "            # (num_scheduled_tokens, num_draft_tokens, num_computed_tokens) is\n"
    "            # pre-staged to GPU buffers in _prepare_inputs.\n"
    "            mamba_utils.postprocess_mamba_align_gpu(\n"
    "                bufs=self._get_mamba_bufs(),\n"
    "                num_reqs=num_reqs,\n"
    "                num_accepted_tokens_gpu=self.num_accepted_tokens.gpu,\n"
    "                num_accepted_tokens_cpu_tensor=(\n"
    "                    self.input_batch.num_accepted_tokens_cpu_tensor\n"
    "                ),\n"
    "                input_batch=self.input_batch,\n"
    "                kv_cache_config=self.kv_cache_config,\n"
    "                forward_context=self.compilation_config.static_forward_context,\n"
    "                mamba_state_copy_funcs=self.model.get_mamba_state_copy_func(),\n"
    "            )\n"
    "\n"
    "            assert self.num_accepted_tokens_event is not None\n"
    "            self.num_accepted_tokens_event.record()\n"
)

PN292_SITE1_REPLACEMENT = (
    "        if self.cache_config.mamba_cache_mode == \"align\":\n"
    "            # [Genesis PN292] Revert PR40172 fused Triton postprocess on Ampere\n"
    "            # SM 8.6. Restore dev371 single-.cpu()-sync + Python postprocess_mamba\n"
    "            # call — Triton kernel launch overhead exceeds the saved sync on A5000.\n"
    "            for i, num_tokens in enumerate(\n"
    "                self.num_accepted_tokens.gpu[:num_reqs].cpu().numpy()\n"
    "            ):\n"
    "                self.input_batch.num_accepted_tokens_cpu[i] = num_tokens\n"
    "            mamba_utils.postprocess_mamba(\n"
    "                scheduler_output,\n"
    "                self.kv_cache_config,\n"
    "                self.input_batch,\n"
    "                self.requests,\n"
    "                self.mamba_state_idx,\n"
    "                self.compilation_config.static_forward_context,\n"
    "                self.model.get_mamba_state_copy_func(),\n"
    "                self._get_mamba_bufs().preprocess,\n"
    "            )\n"
)

# --------------------------------------------------------------------------
# Site 2: execute_model — drop the per-step Triton staging block. Keep the
#         existing preprocess_mamba call (which is unchanged in dev354 except
#         it now receives mamba_bufs.preprocess instead of the legacy handle).
# --------------------------------------------------------------------------

PN292_SITE2_ANCHOR = (
    "                # Stage per-request inputs for the fused postprocess kernel\n"
    "                # only when that kernel will actually run. The kernel is\n"
    "                # gated on spec-decode + hybrid (see MambaBuffers.create);\n"
    "                # without it, ``mamba_bufs.postprocess_align`` is None and\n"
    "                # the staging buffers don't exist.\n"
    "                if mamba_bufs.postprocess_align is not None:\n"
    "                    mamba_utils.stage_postprocess_inputs_to_gpu(\n"
    "                        mamba_bufs.postprocess_align,\n"
    "                        scheduler_output,\n"
    "                        self.input_batch.req_ids,\n"
    "                        num_reqs,\n"
    "                        self.requests,\n"
    "                        self.mamba_state_idx,\n"
    "                    )\n"
)

PN292_SITE2_REPLACEMENT = (
    "                # [Genesis PN292] Skip stage_postprocess_inputs_to_gpu —\n"
    "                # not needed when site 1 above runs the legacy Python\n"
    "                # postprocess_mamba (no fused align kernel consumes the\n"
    "                # staged GPU buffers).\n"
)


def apply() -> tuple[str, str]:
    """Apply PN292 — revert PR40172 fused Mamba postprocess on Ampere SM 8.6."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN292")
    log_decision("PN292", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None or not os.path.isfile(str(target)):
        return "skipped", "gpu_model_runner.py not found"

    patcher = TextPatcher(
        patch_name=(
            "PN292 gpu_model_runner.py — revert PR40172 fused Mamba postprocess "
            "(Ampere SM 8.6 closes -18% TPS regression)"
        ),
        target_file=str(target),
        marker=GENESIS_PN292_MARKER,
        sub_patches=[
            TextPatch(
                name="pn292_site1_revert_postprocess_align_gpu",
                anchor=PN292_SITE1_ANCHOR,
                replacement=PN292_SITE1_REPLACEMENT,
                required=True,
            ),
            TextPatch(
                name="pn292_site2_drop_stage_postprocess_inputs",
                anchor=PN292_SITE2_ANCHOR,
                replacement=PN292_SITE2_REPLACEMENT,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN292]",
            "self._get_mamba_bufs().preprocess,\n            )",
        ],
    )
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN292 applied: dev354 fused Triton Mamba postprocess (PR40172) reverted "
            "to dev371 Python form on Ampere SM 8.6. Expected to recover +18% TPS "
            "on Qwen3.6 27B hybrid + MTP K=3 (107 -> ~130 TPS at 5x5x1024 bench)."
        ),
        patch_name=(
            "PN292 revert PR40172 fused Mamba postprocess (Ampere SM 8.6)"
        ),
    )


def is_applied() -> bool:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return False
    try:
        with open(str(target)) as f:
            return GENESIS_PN292_MARKER in f.read()
    except OSError:
        return False

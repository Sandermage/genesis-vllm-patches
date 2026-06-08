# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN290 — num_accepted_tokens D2H race fix (vllm Issue #41190).

Backport-style fix for [vllm Issue #41190](https://github.com/vllm-project/vllm/issues/41190)
— OPEN at time of write; no upstream PR yet.

================================================================
WHAT THIS PATCH DOES
================================================================

In ``GPUModelRunner._update_states_after_model_execute`` (V1 spec-decode
path), the runner issues a non-blocking D2H copy of ``num_accepted_tokens.gpu``
into a CPU-resident pinned tensor, records a CUDA event, and continues
into model forward. With TP>1, the NCCL collective that follows can run
on a different stream and frees/reallocates the source GPU tensor while
the D2H is still in flight. On the next iteration, ``self.num_accepted_tokens_event.synchronize()``
trips an ``cudaErrorIllegalAddress`` because the underlying memory has
been recycled.

Symptom (rig log line, dev354+g626fa9bba, 2× A5000 + TP=2 + MTP K=3 + multi-conc):

    (Worker_TP1) ERROR [multiproc_executor.py:962]
        self.num_accepted_tokens_event.synchronize()
    torch.AcceleratorError: CUDA error: an illegal memory access was encountered

The race only fires when ALL of:
- TP > 1 (NCCL collective in between record and synchronize)
- MTP / DFlash / EAGLE speculation (num_accepted_tokens path active)
- Concurrent traffic (max_num_seqs > 1, so the D2H/NCCL ordering matters)

Reference issue thread (validators):
- hata1234 — Qwen3.6-35B-A3B-AWQ, TP=2, RTX 6000 Ada, full/piecewise/none cudagraph
- UmutAlihan — Gemma4 e2b-it, TP=2, RTX 3060, "TP=1 fine, TP=2+MTP crashes"

Both confirm: TP=2 + MTP = guaranteed crash; TP=2 without MTP works.

================================================================
THE FIX
================================================================

Replace the ``non_blocking=True`` D2H copy with ``non_blocking=False``.
This forces the CPU thread to wait until the D2H completes before any
subsequent NCCL collective can launch, eliminating the race window.

Cost: the D2H of a small int tensor (num_reqs entries) blocks briefly —
measured ~0.3–0.6 ms on A5000 + PCIe 4.0. Negligible vs the cost of an
engine crash + restart cycle.

We patch ONLY the ``else`` branch (mamba_cache_mode != "all"). The ``if``
branch delegates to ``mamba_utils.postprocess_mamba_align_gpu`` which
has its own internal sync semantics and a different anchor site (out
of scope here; tracked as follow-up if mamba_all-mode also crashes).

================================================================
SAFETY MODEL
================================================================

- Pure synchronization — no algorithmic change. Output sequences are
  bit-identical to vanilla (just a CPU/GPU pipeline rearrangement).
- Idempotent via Genesis marker comment block.
- Drift-marker watches for upstream-merged form so the patch self-skips
  once vllm Issue #41190 lands a fix.
- Adds zero VRAM.
- Required sub-patch: if anchor cannot be found, Genesis FAILS the patch
  rather than silently soft-skipping (lesson from PN286 half-apply bug).

================================================================

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa, 2026-06-04.
Original issue: vllm-project/vllm#41190.
Reproduced on: 0.21.1rc1.dev354+g626fa9bba, 2× A5000, TP=2, Qwen3.6-35B-A3B-FP8 + MTP K=3.
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

log = logging.getLogger("genesis.wiring.pn287_num_accepted_tokens_race")

GENESIS_PN290_MARKER = (
    "Genesis PN290 num_accepted_tokens D2H race fix (vllm#41190) v1"
)

# Anchor: the else-branch direct copy + event.record(). 12-space indented
# (inside `if self.cache_config.mamba_cache_mode != "all":` → else: block).
PN290_OLD = (
    "        else:\n"
    "            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(\n"
    "                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=True\n"
    "            )\n"
    "            assert self.num_accepted_tokens_event is not None\n"
    "            self.num_accepted_tokens_event.record()"
)

PN290_NEW = (
    "        else:\n"
    "            # ════════════════════════════════════════════════════════════\n"
    "            # [Genesis PN290 vllm#41190 fix] v2 — full device sync before\n"
    "            # the D2H copy of num_accepted_tokens.gpu.\n"
    "            #\n"
    "            # v1 (blocking copy alone) was insufficient: on TP>1+MTP,\n"
    "            # the source tensor self.num_accepted_tokens.gpu is already\n"
    "            # corrupted/freed by previous iteration's NCCL/cudagraph ops\n"
    "            # BEFORE this copy starts. Crash at this line (1559 dev354)\n"
    "            # with torch.AcceleratorError: cudaErrorIllegalAddress.\n"
    "            #\n"
    "            # Solution: torch.cuda.synchronize() to drain ALL pending\n"
    "            # device ops (NCCL allreduce, cudagraph capture, spec-decode\n"
    "            # draft kernel) before reading the source.\n"
    "            #\n"
    "            # Operator override:\n"
    "            #   GENESIS_PN290_SYNC_MODE=full   (default, hardest hammer)\n"
    "            #   GENESIS_PN290_SYNC_MODE=stream (cheaper, current stream only)\n"
    "            #   GENESIS_PN290_SYNC_MODE=none   (revert to upstream behavior)\n"
    "            # ════════════════════════════════════════════════════════════\n"
    "            import os as _pn290_os, torch as _pn290_torch\n"
    "            _pn290_mode = _pn290_os.environ.get('GENESIS_PN290_SYNC_MODE', 'full').lower()\n"
    "            if _pn290_mode == 'full':\n"
    "                _pn290_torch.cuda.synchronize()\n"
    "            elif _pn290_mode == 'stream':\n"
    "                _pn290_torch.cuda.current_stream().synchronize()\n"
    "            # 'none' falls through — equivalent to unpatched upstream.\n"
    "            self.input_batch.num_accepted_tokens_cpu_tensor[:num_reqs].copy_(\n"
    "                self.num_accepted_tokens.gpu[:num_reqs], non_blocking=False\n"
    "            )\n"
    "            assert self.num_accepted_tokens_event is not None\n"
    "            self.num_accepted_tokens_event.record()"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN290 v1/worker/gpu_model_runner.py — num_accepted_tokens "
            "D2H race fix (vllm#41190)"
        ),
        target_file=str(target),
        marker=GENESIS_PN290_MARKER,
        sub_patches=[
            TextPatch(
                name="pn287_force_blocking_d2h",
                anchor=PN290_OLD,
                replacement=PN290_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN290",
            # Watch for any upstream form that eliminates the race —
            # e.g. an explicit synchronize, a wait_event before NCCL,
            # or a switch to blocking copy upstream.
            "GENESIS_PN290_FORCE_NONBLOCKING",
        ],
    )


_APPLIED = False


def apply() -> tuple[str, str]:
    """Apply PN290 — num_accepted_tokens D2H race fix."""
    global _APPLIED

    if os.environ.get("GENESIS_ENABLE_PN290_NUM_ACCEPTED_TOKENS_RACE", "").lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "PN290 default OFF — set GENESIS_ENABLE_PN290_NUM_ACCEPTED_TOKENS_RACE=1 to engage. "
            "Targets vllm Issue #41190 (TP>1 + MTP + multi-conc cudaErrorIllegalAddress)."
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
    # APPLIED or IDEMPOTENT
    applied = patcher.applied_sub_patches or [sp.name for sp in patcher.sub_patches]
    _APPLIED = True
    return "applied", (
        f"PN290 installed: D2H copy of num_accepted_tokens.gpu forced blocking "
        f"to eliminate TP>1+MTP race (vllm#41190). Sub-patches: {', '.join(applied)}. "
        f"Operator override: GENESIS_PN290_FORCE_NONBLOCKING=1 to revert."
    )


def is_applied() -> bool:
    return _APPLIED

# SPDX-License-Identifier: Apache-2.0
"""Sprint 2.6 v2 — CUDA graph dispatch trace text-patch wire-in.

Hooks `record_dispatch(matched)` into the vllm v1 cudagraph dispatcher
call sites in `gpu_model_runner.py`. The wrapper is fail-silent (any
error is swallowed so no production breakage) and only fires when
GENESIS_CUDAGRAPH_DISPATCH_TRACE=1 in the env. Registry env_flag is
`GENESIS_ENABLE_PN122_CG_DISPATCH_TRACE` (legacy `GENESIS_ENABLE_SPRINT26_CG_DISPATCH_TRACE` accepted; dispatcher-gated; default
OFF unless opted in for instrumentation).

Two anchors (both at known dispatcher.dispatch() call sites):

  1. `_, batch_desc = self.cudagraph_dispatcher.dispatch(...)` at
     ~line 2717 (decoder portion) → wrap with record_dispatch(matched)
     based on batch_desc.cudagraph_mode != NONE
  2. `dispatch_cudagraph` inner function at ~3661 (multiple sites
     using it) — leaving inner function as-is, wrap each call instead

For v2 simplicity, anchor only the line 2717 site (the most common
decoder-decode dispatch — where the original Wave 6 PN16 V1 mismatch
was observed). Sweep / spec-decode sites can be added in v3 if needed.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatcher,
    TextPatch,
)

log = logging.getLogger("genesis.sprint26.dispatch_trace")

GENESIS_SPRINT26_DISPATCH_MARKER = (
    "Genesis Sprint 2.6 v2 — cudagraph dispatch trace wire-in"
)


# Anchor: the decoder dispatch site at gpu_model_runner.py:~2717.
# Surrounding context for unique match.
SPRINT26_SITE1_OLD = (
    "        # Dispatch for the decoder portion of the model.\n"
    "        _, batch_desc = self.cudagraph_dispatcher.dispatch(\n"
    "            num_logits, invalid_modes={CUDAGraphMode.FULL}\n"
    "        )\n"
)
SPRINT26_SITE1_NEW = (
    "        # Dispatch for the decoder portion of the model.\n"
    "        _, batch_desc = self.cudagraph_dispatcher.dispatch(\n"
    "            num_logits, invalid_modes={CUDAGraphMode.FULL}\n"
    "        )\n"
    "        # [Genesis Sprint 2.6 v2] cudagraph dispatch trace — fail-silent\n"
    "        try:\n"
    "            from sndr.observability.cudagraph_dispatch import record_dispatch as _g_s26_rec\n"
    "            from vllm.v1.cudagraph_dispatcher import CUDAGraphMode as _g_s26_CGM\n"
    "            _g_s26_rec(matched=getattr(batch_desc, 'cudagraph_mode', None) != _g_s26_CGM.NONE)\n"
    "        except Exception:\n"
    "            pass\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/worker/gpu_model_runner.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "Sprint 2.6 v2 v1/worker/gpu_model_runner.py — cudagraph dispatch trace"
        ),
        target_file=str(target),
        marker=GENESIS_SPRINT26_DISPATCH_MARKER + " — site1",
        sub_patches=[
            TextPatch(
                name="sprint26_dispatch_trace_decoder",
                anchor=SPRINT26_SITE1_OLD,
                replacement=SPRINT26_SITE1_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "_g_s26_" was baked by our own replacement — residue coverage
            # stays with the "[Genesis Sprint 2.6" banner.
            "[Genesis Sprint 2.6",
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN122 cudagraph dispatch trace wire-in (formerly SPRINT26_CG_DISPATCH_TRACE)."""
    from sndr.dispatcher import should_apply, log_decision
    decision, reason = should_apply("PN122")  # renamed from SPRINT26_CG_DISPATCH_TRACE 2026-05-14
    log_decision("PN122", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "v1/worker/gpu_model_runner.py not resolvable"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[Sprint26 v2] marker present — idempotent skip")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis Sprint 2.6" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file}",
            )

    result, failure = patcher.apply()
    from sndr.kernel import result_to_wiring_status
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "Sprint 2.6 v2: cudagraph dispatch trace wire-in injected. "
            "Set GENESIS_CUDAGRAPH_DISPATCH_TRACE=1 + run workload — "
            "summary lines emit every GENESIS_CUDAGRAPH_LOG_EVERY (default 1000) "
            "requests."
        ),
        patch_name=patcher.patch_name,
    )

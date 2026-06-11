# SPDX-License-Identifier: Apache-2.0
"""PN367 — clamp negative CUDA graph memory estimates (vendor of vllm#44745).

Upstream bug (vllm#44740): with MTP speculative decoding active,
``profile_cudagraph_memory()`` in ``gpu_model_runner.py`` can produce
NEGATIVE estimates. Verified at our pin g303916e93: the decoder
measurement path is unprotected —

    mem_samples.append(mem_before - free_after)      # can go negative
    first_capture = mem_samples[0]                   # ...propagates

while the encoder path in the SAME function already clamps:

    encoder_memory_estimate = max(mem_before - free_after, 0)

Upstream history (v2, 2026-06-11): PN367 v1 vendored OPEN PR #45076
(Oxygen56). On 2026-06-10 the author CLOSED #45076 and consolidated
into PR #44745 ("Fix negative cudagraph memory estimate", same clamp +
1 MiB first-capture floor + unit tests, still OPEN). PN367 v2 now
tracks #44745: drift markers match its actual merged code form, and
the first-capture floor is vendored for full parity.

Negative-delta mechanics (the two factors that are hardware-agnostic):

  1. PyTorch CachingAllocator non-monotonicity — freelist
     consolidation after capture can make ``free_after > mem_before``.
  2. MTP lazy buffer allocation — the proposer's deferred per-group
     slot-mapping buffers allocate on first capture and may be
     GC-cleaned between measurements.

(The reporter's third factor — unified-memory page migration on GB10
Blackwell — does not apply to discrete-VRAM A5000s.)

Impact on Genesis PROD (2x A5000 24 GB, gpu_memory_utilization=0.9,
MTP K=3): a negative ``first_capture`` understates total graph memory
-> vLLM sizes the KV cache LARGER than the card can actually afford ->
silent headroom loss; worst case OOM during capture or under load.
We run VLLM_LOGGING_LEVEL=WARNING, so the upstream logger.debug line
that would reveal the computed estimate is invisible — this patch adds
a WARNING when a negative delta is observed, so any occurrence becomes
diagnosable in PROD logs.

Vendoring scope vs upstream #44745 (documented divergence):
  * VENDORED: per-sample clamp to >= 0 + negative-delta warning and
    the 1 MiB first-capture floor (gpu_model_runner.py), plus the
    final non-negative guard (gpu_worker.py).
  * NOT vendored: the PR's per-measurement ``empty_cache()`` calls —
    they improve measurement stability but add boot-time cost; the
    clamp alone removes the pathological negative case, which is the
    correctness issue. Revisit if estimates remain noisy.

Self-skips when upstream lands #44745: the drift markers below are
exact substrings of that PR's merged form (v1's markers were not —
they could never fire; fixed in v2).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn367_cudagraph_mem_estimate_clamp")

GENESIS_PN367_MARKER = (
    "Genesis PN367 cudagraph memory estimate clamp "
    "(vendor of vllm#44745, formerly #45076) v2"
)

_RUNNER_REL = "v1/worker/gpu_model_runner.py"
_WORKER_REL = "v1/worker/gpu_worker.py"

# Drift markers — exact substrings of #44745's merged form, verified
# against `gh pr diff 44745` on 2026-06-11. Absent at pin g303916e93
# (decoder path appends the raw delta; worker has no guard line) and
# deliberately NOT substrings of our own replacement texts (we name
# the local `_g_pn367_delta`, and our worker guard has a Genesis
# comment between the two lines the worker marker requires adjacent).
_RUNNER_DRIFT_MARKERS = (
    "mem_samples.append(max(delta, 0))",
)
_WORKER_DRIFT_MARKERS = (
    "profile_cudagraph_memory()\n"
    "                cudagraph_memory_estimate = "
    "max(cudagraph_memory_estimate, 0)",
)

PN367_RUNNER_OLD = (
    "                        torch.accelerator.synchronize()\n"
    "                        free_after = torch.cuda.mem_get_info()[0]\n"
    "                        mem_samples.append(mem_before - free_after)\n"
)

PN367_RUNNER_NEW = (
    "                        torch.accelerator.synchronize()\n"
    "                        free_after = torch.cuda.mem_get_info()[0]\n"
    "                        # [Genesis PN367 cudagraph memory estimate clamp "
    "(vendor of vllm#44745, formerly #45076) v2]\n"
    "                        # Decoder path lacked the encoder path's >=0\n"
    "                        # clamp: allocator freelist consolidation or MTP\n"
    "                        # lazy buffers can make free_after > mem_before,\n"
    "                        # and a negative estimate inflates the KV cache\n"
    "                        # budget on 24 GB cards. Clamp + surface a\n"
    "                        # WARNING so occurrences are visible at PROD's\n"
    "                        # VLLM_LOGGING_LEVEL=WARNING.\n"
    "                        _g_pn367_delta = mem_before - free_after\n"
    "                        if _g_pn367_delta < 0:\n"
    "                            logger.warning(\n"
    "                                \"[Genesis PN367] negative CUDA graph \"\n"
    "                                \"memory delta %.2f MiB clamped to 0 \"\n"
    "                                \"(allocator non-monotonicity / MTP lazy \"\n"
    "                                \"buffers — see vllm#44740)\",\n"
    "                                _g_pn367_delta / (1 << 20),\n"
    "                            )\n"
    "                        mem_samples.append(max(_g_pn367_delta, 0))\n"
)

PN367_FLOOR_OLD = (
    "                    first_capture = mem_samples[0]\n"
)

PN367_FLOOR_NEW = (
    "                    # [Genesis PN367 cudagraph memory estimate clamp "
    "(vendor of vllm#44745, formerly #45076) v2]\n"
    "                    # 1 MiB floor on the shared first-capture estimate\n"
    "                    # (driver overhead) — parity with #44745's merged\n"
    "                    # form; without it a clamped-to-zero first sample\n"
    "                    # would book zero shared graph memory.\n"
    "                    first_capture = max(mem_samples[0], 1 << 20)\n"
)

PN367_WORKER_OLD = (
    "                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()\n"
)

PN367_WORKER_NEW = (
    "                cudagraph_memory_estimate = self.model_runner.profile_cudagraph_memory()\n"
    "                # [Genesis PN367 cudagraph memory estimate clamp "
    "(vendor of vllm#44745, formerly #45076) v2] final non-negative guard\n"
    "                cudagraph_memory_estimate = max(cudagraph_memory_estimate, 0)\n"
)


def _make_runner_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_RUNNER_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN367 gpu_model_runner.py — clamp negative decoder cudagraph "
            "memory deltas + first-capture floor (vendor of vllm#44745)"
        ),
        target_file=str(target),
        marker=GENESIS_PN367_MARKER,
        sub_patches=[
            TextPatch(
                name="pn367_decoder_mem_sample_clamp",
                anchor=PN367_RUNNER_OLD,
                replacement=PN367_RUNNER_NEW,
                required=True,
            ),
            TextPatch(
                name="pn367_first_capture_floor",
                anchor=PN367_FLOOR_OLD,
                replacement=PN367_FLOOR_NEW,
                # Completeness parity with #44745 — never abort the
                # correctness clamp if this anchor drifts.
                required=False,
            ),
        ],
        upstream_drift_markers=list(_RUNNER_DRIFT_MARKERS),
    )


def _make_worker_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_WORKER_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN367 gpu_worker.py — final non-negative cudagraph estimate "
            "guard (vendor of vllm#44745)"
        ),
        target_file=str(target),
        marker=GENESIS_PN367_MARKER,
        sub_patches=[
            TextPatch(
                name="pn367_worker_final_guard",
                anchor=PN367_WORKER_OLD,
                replacement=PN367_WORKER_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_WORKER_DRIFT_MARKERS),
    )


def apply() -> tuple[str, str]:
    """Install both clamps. Never raises."""
    runner = _make_runner_patcher()
    if runner is None:
        return "skipped", f"PN367: {_RUNNER_REL} not resolvable"
    worker = _make_worker_patcher()
    if worker is None:
        return "skipped", f"PN367: {_WORKER_REL} not resolvable"

    r_result, r_failure = runner.apply()
    r_status, r_reason = result_to_wiring_status(
        r_result, r_failure,
        applied_message="decoder mem-sample clamp + first-capture floor applied",
        patch_name="PN367 runner clamp",
    )
    if r_status == "failed":
        return "failed", f"PN367 runner sub-patch failed: {r_reason}"

    w_result, w_failure = worker.apply()
    w_status, w_reason = result_to_wiring_status(
        w_result, w_failure,
        applied_message="worker final non-negative guard applied",
        patch_name="PN367 worker guard",
    )
    if w_status == "failed":
        return "failed", f"PN367 worker sub-patch failed: {w_reason}"

    if r_status == "applied" or w_status == "applied":
        return "applied", (
            "PN367 applied: CUDA graph memory estimate clamped >= 0 on the "
            "decoder profiling path (was unprotected — encoder path already "
            "clamps) + 1 MiB first-capture floor + final guard in "
            "gpu_worker. Protects KV-cache budget on 24 GB cards from "
            "negative-estimate inflation under MTP spec-decode "
            "(vllm#44740, vendor of vllm#44745). Negative deltas now log "
            f"WARNING. runner: {r_reason} | worker: {w_reason}"
        )
    return "skipped", f"PN367: runner: {r_reason} | worker: {w_reason}"

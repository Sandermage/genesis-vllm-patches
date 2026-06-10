# SPDX-License-Identifier: Apache-2.0
"""PN352 — route top-k=8 moe_sum through a Genesis Triton kernel.

STATUS 2026-06-10: PARKED — stream-race crash under async engine load.
=======================================================================
Standalone kernel test passes (all shapes incl. the PROD decode shape,
fp32-accumulate numerics verified vs torch reference). In-engine the
first real decode dies with CUDA illegal memory access async-reported
at rejection_sampler; with CUDA_LAUNCH_BLOCKING=1 the same request
SUCCEEDS -> the failure is a stream race (kernel launched on the
current torch stream races the engine's producer/consumer streams
around the fused_experts region), not a kernel-body bug.

Park decision per the 3-diagnostic-restart rule: lever value -1-3 %
TPOT does not justify multi-hour mixed-capture stream-semantics
debugging right now. To resume: investigate which stream
dispatch_fused_moe_kernel records on during PIECEWISE capture vs what
triton uses at eager launch, and pin the launch to the same stream
(torch.cuda.stream context from the captured region) before retrying.

Keep GENESIS_DISABLE_PN352_INSTALL=1 on PROD until then.


Genesis counterpart of OPEN vllm PR #44557 (xyang16). See
``sndr.engines.vllm.kernels_legacy.pn352_moe_sum_topk`` for the kernel
and the full design rationale.

Why a text-patch at the call site (not a monkey-patch on ops.moe_sum)
---------------------------------------------------------------------
``vllm._custom_ops.moe_sum`` is re-exported and referenced through
several module aliases; a setattr there is fragile across vLLM's op
registration. The single hot call site in
``fused_moe.py::fused_experts_impl`` is a stable anchor and keeps the
change visible in the file for other patches' anchor planning.

Env gating
----------
``GENESIS_ENABLE_PN352=1`` enables the Triton branch at runtime. The
text is installed with a runtime conditional (same pattern as PN204 /
PN365): with the env unset the patched block is bit-equivalent to
upstream. NOTE the stale-residue lesson (journal 2026-06-10): if this
patch is later retired, REVERT the text, do not just flip the env.

Numerics
--------
fp32 accumulation, identical to ATen's acc_type for half/bf16 inputs.
Output dtype unchanged. Token-level bit-equivalence with the fallback
is not guaranteed (different reduction order) but is within the same
tolerance class as the upstream CUDA kernels for top-k 2/3/4.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn352_moe_sum_topk8")

GENESIS_PN352_MARKER = (
    "Genesis PN352 triton moe_sum for unsupported topk "
    "(counterpart of OPEN vllm#44557) v1"
)

_TARGET_REL = "model_executor/layers/fused_moe/fused_moe.py"
_ENV_FLAG = "GENESIS_ENABLE_PN352"

# Upstream drift markers: when vllm#44557 (or an equivalent) lands, the
# compiled op stops falling back for topk=8 and this patch self-skips.
_UPSTREAM_DRIFT_MARKERS = (
    "moe_sum_kernel<scalar_t, 8>",
    "genesis_pn352",  # never overwrite our own text
)

PN352_OLD = (
    "    ops.moe_sum(\n"
    "        intermediate_cache3.view(*intermediate_cache3.size()),\n"
    "        out_hidden_states,\n"
    "    )\n"
)

PN352_NEW = (
    "    # [Genesis PN352 triton moe_sum for unsupported topk "
    "(counterpart of OPEN vllm#44557) v1]\n"
    "    # _moe_C.moe_sum covers topk 2/3/4 only; other topk values\n"
    "    # (Qwen3.6-A3B uses 8) fall back to at::sum_out — a generic\n"
    "    # TensorIterator reduce. Route those through the Genesis Triton\n"
    "    # kernel instead. Env-gated; unset env = upstream behavior.\n"
    "    import os as _g_pn352_os\n"
    "    _g_pn352_on = _g_pn352_os.environ.get(\n"
    "        \"GENESIS_ENABLE_PN352\", \"0\"\n"
    "    ).strip().lower() in (\"1\", \"true\", \"yes\", \"on\")\n"
    "    _g_pn352_done = False\n"
    "    if _g_pn352_on and intermediate_cache3.size(1) not in (2, 3, 4):\n"
    "        from sndr.engines.vllm.kernels_legacy.pn352_moe_sum_topk import (\n"
    "            moe_sum_topk as _g_pn352_sum,\n"
    "        )\n"
    "        _g_pn352_done = _g_pn352_sum(\n"
    "            intermediate_cache3.view(*intermediate_cache3.size()),\n"
    "            out_hidden_states,\n"
    "        )\n"
    "    if not _g_pn352_done:\n"
    "        ops.moe_sum(\n"
    "            intermediate_cache3.view(*intermediate_cache3.size()),\n"
    "            out_hidden_states,\n"
    "        )\n"
)


def _enabled_install() -> bool:
    """Install the text whenever not explicitly disabled.

    The runtime branch is env-gated inside the text itself, so install
    is safe-by-default; GENESIS_DISABLE_PN352_INSTALL=1 skips even the
    text install for emergency hygiene.
    """
    return os.environ.get(
        "GENESIS_DISABLE_PN352_INSTALL", "0",
    ).strip().lower() not in ("1", "true", "yes", "on")


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN352 fused_moe.py — triton moe_sum for unsupported topk "
            "(counterpart of OPEN vllm#44557)"
        ),
        target_file=str(target),
        marker=GENESIS_PN352_MARKER,
        sub_patches=[
            TextPatch(
                name="pn352_moe_sum_call_site",
                anchor=PN352_OLD,
                replacement=PN352_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=_UPSTREAM_DRIFT_MARKERS,
    )


def apply() -> tuple[str, str]:
    """Install the env-gated Triton moe_sum branch. Never raises."""
    if not _enabled_install():
        return "skipped", (
            "PN352 text install disabled via GENESIS_DISABLE_PN352_INSTALL"
        )
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN352: target {_TARGET_REL} not resolvable"
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN352 applied: fused_experts moe_sum call site now routes "
            "topk not in (2,3,4) through the Genesis Triton kernel when "
            "GENESIS_ENABLE_PN352=1 (Qwen3.6-A3B topk=8 -> skips the "
            "at::sum_out fallback; PR author measures ~-700 us/decode "
            "step on a 40-layer topk=8 MoE, est -1-3 % decode TPOT). "
            "Env unset -> bit-equivalent upstream behavior."
        ),
        patch_name="PN352 triton moe_sum topk8",
    )

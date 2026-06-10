# SPDX-License-Identifier: Apache-2.0
"""PN368 — Marlin MoE w13 reduce-mode wire: env-gated atomic-add.

Genesis-original wire of upstream's OWN dense-path reduce-mode heuristic
(``should_use_atomic_add_reduce``, marlin_utils.py) into the MoE Marlin
w13 GEMM, where upstream hardcodes ``use_atomic_add=False,
use_fp32_reduce=True``.

Why (verified at pin g303916e93 / 0.22.1rc1.dev259, live container
vllm-qwen3.6-35b-balanced-k3, 2026-06-10)
------------------------------------------------------------------
Qwen3.6-35B-A3B-FP8 MoE runs MarlinExperts on SM 8.6 (TritonExperts is
excluded: ``supports_fp8()`` is False on Ampere). Both
``ops.moe_wna16_marlin_gemm`` call sites in
``model_executor/layers/fused_moe/experts/marlin_moe.py`` hardcode
``use_atomic_add=False`` — while the dense Marlin path
(``apply_gptq_marlin_linear`` / ``apply_awq_marlin_linear``) routes the
exact same decision through ``should_use_atomic_add_reduce(m, n, k,
device, dtype)``. For the w13 GEMM on our deployment the heuristic
APPROVES atomic-add:

  - n = w13_num_shards * N = 512 < 2048   (per-rank, TP=2)
  - k = K = 2048 >= 2048
  - device.type == "cuda"
  - VLLM_MARLIN_USE_ATOMIC_ADD=1          (set in PROD launcher)
  - dtype float16 (container runs --dtype float16; the sm8x refusal in
    the heuristic applies ONLY to bfloat16)

The w2 GEMM (n=K=2048, k=N=256) FAILS the heuristic (n >= 2048, k <
2048) — upstream's hardcoded False is correct there, so v1 deliberately
does NOT touch the w2 call site.

atomic_add / fp32_reduce mutual exclusion — VERIFIED, not assumed
------------------------------------------------------------------
The dense path passes BOTH flags independently (``use_atomic_add=
<heuristic>, use_fp32_reduce=USE_FP32_REDUCE_DEFAULT`` where the
default is True — marlin_utils.py L36/L594/L658 at this pin). The
kernel resolves the conflict in favor of atomic-add:

  - host side (csrc/moe/marlin_moe_wna16/ops.cu L692): the fp32
    global-reduce buffer ``c_tmp`` is allocated only under
    ``if (use_fp32_reduce && !use_atomic_add)``;
  - device side (csrc/moe/marlin_moe_wna16/marlin_template.h
    L2162-2165): ``use_fp32_reduce`` is consulted only inside the
    ``if (slice_count > 1 && !use_atomic_add)`` global-reduce branch.

So mirroring dense exactly means: wire the heuristic into
``use_atomic_add`` and leave ``use_fp32_reduce=True`` untouched. That
is what this patch does — the text diff flips a single kwarg.

Why the heuristic is replicated inline (not imported)
------------------------------------------------------------------
Two reasons:

1. Drift-marker hygiene: ``should_use_atomic_add_reduce`` appearing in
   marlin_moe.py is this patch's obsolescence signal (upstream wiring
   its own heuristic into the MoE path). Importing the function by name
   into the patched text would put the marker string into the file we
   monitor, false-triggering the daily drift watcher on our own text.
2. The heuristic body at this pin never consults ``m`` (despite taking
   it as a parameter), so the replicated branch needs only the n / k /
   device / dtype facts available at the call site. The inline copy
   approves ONLY float16 — a strict subset of upstream's approvals on
   every arch (upstream refuses bfloat16 on sm8x only; we never run
   the MoE Marlin path in bfloat16 on PROD anyway).

Env gating
----------
``GENESIS_ENABLE_PN368_MARLIN_MOE_ATOMIC_ADD=1`` enables the wire at
runtime; it additionally requires upstream's own
``VLLM_MARLIN_USE_ATOMIC_ADD=1`` opt-in (both read ONCE at module
import of the patched file, never per call). With either env unset the
helper returns False and the GEMM receives ``use_atomic_add=False,
use_fp32_reduce=True`` — bit-identical to upstream. Text install is
always-on (same pattern as PN352 / PN204 / PN365);
``GENESIS_DISABLE_PN368_INSTALL=1`` skips even the text for hygiene.
NOTE the stale-residue lesson (journal 2026-06-10): if this patch is
later retired, REVERT the text, do not just flip the env.

Observability
-------------
First enabled hot-path call logs the resolved reduce mode once
(``[Genesis PN368] ... use_atomic_add=...``) so ON-vs-OFF firing is
verifiable in docker logs (iron rule #3).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, result_to_wiring_status

log = logging.getLogger("genesis.wiring.pn368_marlin_moe_atomic_add_wire")

GENESIS_PN368_MARKER = (
    "Genesis PN368 Marlin MoE w13 atomic-add reduce-mode wire v1"
)

_TARGET_REL = "model_executor/layers/fused_moe/experts/marlin_moe.py"
_ENV_FLAG = "GENESIS_ENABLE_PN368_MARLIN_MOE_ATOMIC_ADD"

# Upstream drift markers: if upstream wires its own dense-path heuristic
# (should_use_atomic_add_reduce) into marlin_moe.py, this patch is
# obsolete and self-skips. NOTE: the inserted text deliberately avoids
# both strings (heuristic replicated inline, helper names use the
# _g_pn368_ prefix) so the markers can never fire on our own text.
_UPSTREAM_DRIFT_MARKERS = (
    "should_use_atomic_add_reduce",
    "genesis_pn368",  # never overwrite our own text
)

# Sub-patch 1: module-level helper, inserted right after the import
# block. Env flags are resolved ONCE at import time of the patched
# module; the helper itself is branch-only (no env / no import on the
# hot path).
PN368_IMPORTS_OLD = (
    "from vllm.platforms import current_platform\n"
    "from vllm.scalar_type import ScalarType, scalar_types\n"
)

PN368_IMPORTS_NEW = (
    "from vllm.platforms import current_platform\n"
    "from vllm.scalar_type import ScalarType, scalar_types\n"
    "\n"
    "# [Genesis PN368 Marlin MoE w13 atomic-add reduce-mode wire v1]\n"
    "# Upstream hardcodes use_atomic_add=False at both\n"
    "# moe_wna16_marlin_gemm call sites while the dense Marlin path\n"
    "# (apply_gptq_marlin_linear in marlin_utils.py) routes the same\n"
    "# decision through its reduce-mode heuristic. The heuristic's\n"
    "# approved branch is replicated inline below (it ignores m at this\n"
    "# pin; n/k/device/dtype are available at the call site). Approving\n"
    "# only float16 is a strict subset of upstream's approvals on every\n"
    "# arch (sm8x lacks native bfloat16 atomicAdd). Env flags are read\n"
    "# ONCE here; both unset/0 -> the helper returns False and the GEMM\n"
    "# args are bit-identical to upstream (False/True reduce-mode pair).\n"
    "import os as _g_pn368_os\n"
    "\n"
    "import vllm.envs as _g_pn368_envs\n"
    "\n"
    "_g_pn368_enabled = _g_pn368_os.environ.get(\n"
    "    \"GENESIS_ENABLE_PN368_MARLIN_MOE_ATOMIC_ADD\", \"0\"\n"
    ").strip().lower() in (\"1\", \"true\", \"yes\", \"on\") and bool(\n"
    "    _g_pn368_envs.VLLM_MARLIN_USE_ATOMIC_ADD\n"
    ")\n"
    "_g_pn368_logged = False\n"
    "\n"
    "\n"
    "def _g_pn368_use_atomic_add(\n"
    "    n: int, k: int, device: torch.device, dtype: torch.dtype\n"
    ") -> bool:\n"
    "    \"\"\"Reduce-mode decision for the w13 MoE Marlin GEMM.\n"
    "\n"
    "    Mirrors the approved branch of the dense-path heuristic in\n"
    "    marlin_utils.py: atomic-add only for small-n / large-k GEMMs on\n"
    "    CUDA, float16 only. Kernel-side, use_fp32_reduce is consulted\n"
    "    only when use_atomic_add is False (marlin_template.h global-\n"
    "    reduce branch), so the caller keeps use_fp32_reduce=True --\n"
    "    exactly like the dense path does.\n"
    "    \"\"\"\n"
    "    global _g_pn368_logged\n"
    "    if not _g_pn368_enabled:\n"
    "        return False\n"
    "    use_atomic = (\n"
    "        n < 2048\n"
    "        and k >= 2048\n"
    "        and device.type == \"cuda\"\n"
    "        and dtype == torch.float16\n"
    "    )\n"
    "    if not _g_pn368_logged:\n"
    "        _g_pn368_logged = True\n"
    "        import logging as _g_pn368_logging\n"
    "        _g_pn368_logging.getLogger(__name__).info(\n"
    "            \"[Genesis PN368] w13 Marlin MoE GEMM reduce mode \"\n"
    "            \"resolved: use_atomic_add=%s (n=%s, k=%s, device=%s, \"\n"
    "            \"dtype=%s)\",\n"
    "            use_atomic, n, k, device.type, dtype,\n"
    "        )\n"
    "    return use_atomic\n"
)

# Sub-patch 2: the w13 GEMM call site (ANCHOR A). The size_m/size_n
# lines are part of the anchor for uniqueness — the bare
# use_atomic_add/use_fp32_reduce pair appears at BOTH GEMM call sites
# (w13 and w2). The w2 site (size_m=M * num_topk, size_n=K) fails the
# heuristic (n >= 2048, k < 2048) and is deliberately NOT modified.
PN368_W13_OLD = (
    "        mul_topk_weights=apply_router_weight_on_input,\n"
    "        b_q_type=quant_type,\n"
    "        size_m=M,\n"
    "        size_n=w13_num_shards * N,\n"
    "        size_k=K,\n"
    "        is_k_full=is_k_full,\n"
    "        use_atomic_add=False,\n"
    "        use_fp32_reduce=True,\n"
    "        is_zp_float=False,\n"
)

PN368_W13_NEW = (
    "        mul_topk_weights=apply_router_weight_on_input,\n"
    "        b_q_type=quant_type,\n"
    "        size_m=M,\n"
    "        size_n=w13_num_shards * N,\n"
    "        size_k=K,\n"
    "        is_k_full=is_k_full,\n"
    "        # [Genesis PN368] env-gated atomic-add for the w13 GEMM.\n"
    "        # use_fp32_reduce stays True like the upstream dense path:\n"
    "        # the kernel ignores it when atomic add engages (fp32 c_tmp\n"
    "        # buffer is not even allocated -- ops.cu host check).\n"
    "        use_atomic_add=_g_pn368_use_atomic_add(\n"
    "            w13_num_shards * N, K, hidden_states.device, hidden_states.dtype\n"
    "        ),\n"
    "        use_fp32_reduce=True,\n"
    "        is_zp_float=False,\n"
)


def _enabled_install() -> bool:
    """Install the text whenever not explicitly disabled.

    The runtime branch is env-gated inside the text itself (read once
    at module import of the patched file), so install is safe-by-
    default; GENESIS_DISABLE_PN368_INSTALL=1 skips even the text
    install for emergency hygiene.
    """
    return os.environ.get(
        "GENESIS_DISABLE_PN368_INSTALL", "0",
    ).strip().lower() not in ("1", "true", "yes", "on")


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file(_TARGET_REL)
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN368 marlin_moe.py — w13 reduce-mode wire "
            "(env-gated atomic-add, dense-path heuristic parity)"
        ),
        target_file=str(target),
        marker=GENESIS_PN368_MARKER,
        sub_patches=[
            TextPatch(
                name="pn368_module_helper",
                anchor=PN368_IMPORTS_OLD,
                replacement=PN368_IMPORTS_NEW,
                required=True,
            ),
            TextPatch(
                name="pn368_w13_reduce_mode",
                anchor=PN368_W13_OLD,
                replacement=PN368_W13_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=list(_UPSTREAM_DRIFT_MARKERS),
    )


def apply() -> tuple[str, str]:
    """Install the env-gated w13 atomic-add wire. Never raises."""
    if not _enabled_install():
        return "skipped", (
            "PN368 text install disabled via GENESIS_DISABLE_PN368_INSTALL"
        )
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", f"PN368: target {_TARGET_REL} not resolvable"
    result, failure = patcher.apply()
    return result_to_wiring_status(
        result,
        failure,
        applied_message=(
            "PN368 applied: w13 moe_wna16_marlin_gemm reduce mode now "
            "routes through the dense-path heuristic (inline replica) "
            "when GENESIS_ENABLE_PN368_MARLIN_MOE_ATOMIC_ADD=1 AND "
            "VLLM_MARLIN_USE_ATOMIC_ADD=1 AND dtype is float16 "
            "(w13 on 35B-A3B: n=512<2048, k=2048>=2048 -> atomic-add "
            "engages; w2 fails the heuristic and stays untouched). "
            "use_fp32_reduce=True kept — kernel ignores it under "
            "atomic add. Env unset -> bit-identical upstream behavior."
        ),
        patch_name="PN368 marlin moe w13 atomic-add wire",
    )

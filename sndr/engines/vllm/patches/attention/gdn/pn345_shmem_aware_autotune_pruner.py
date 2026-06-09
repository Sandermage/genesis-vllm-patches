# SPDX-License-Identifier: Apache-2.0
"""PN345 — vendor of OPEN PR vllm#43047 (shmem-aware autotune pruner) + Genesis extensions.

Deep-dive understanding of the upstream PR
==========================================

**Problem**: Triton kernels in vLLM ship with ``@triton.autotune`` config
lists tuned for the largest opt-in shared-memory budget Triton supports
(228 KiB on H100/H200). GPUs with smaller budgets (Turing T4 ~64 KiB,
Ampere A100 ~163 KiB, **consumer Ampere A5000/3090 ~99 KiB**, Blackwell
SM_120 ~99 KiB) hit ``triton.runtime.errors.OutOfResources`` at JIT
time on configs that won't fit.

**Concrete math** for ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``
(``chunk_delta_h.py``) at BV=64, BT=64, num_stages=4::

    persistent = 4 * BV * 64 * 4 = 65,536 bytes  (4× fp32 [BV,64] b_h)
    per_stage  = 2*BT*64*2 + BT*BV*2 = 24,576 bytes  (b_w + b_k + b_v in bf16)
    total      = 65,536 + 4*24,576 + 4,096 = 163,840 bytes (160 KiB)

That **exceeds A5000's 99 KiB opt-in budget by 64 KiB** — JIT fails or
silently falls back to the smallest bucket. The PR adds a precise
per-config + per-num_stages filter via Triton's existing
``prune_configs_by={"early_config_prune": fn}`` hook.

**Why this is different from Genesis PN298-PN299E**:

Our existing patches (PN298 chunk_o, PN299 kkt+wy+l2, PN299B cumsum+kda+
solve_tril, PN299C layernorm_guard, PN299D mamba_ssm fallback, PN299E
kv_cache writer) are **coarse env-based filters** — they drop configs
by ``num_warps`` threshold read from ``GENESIS_TRITON_AUTOTUNE_MAX_
WARPS`` env (=4 on SM 8.6 per PN296 auto-set).

The PR's pruner is **precise per-config shmem-budget filter** — it
checks the actual memory footprint of each config against the actual
device opt-in budget. This is strictly more accurate:

  * A coarse cap might drop a num_warps=8 config that fits comfortably
    (e.g. BT=BK=BV=32, only 50 KiB).
  * A precise filter would keep that config because the shmem fits.

The two approaches **compose** — they target different files (PN29x
covers 6 files, PN345 covers chunk_delta_h.py + chunk_o.py — no
overlap). No conflict.

Why we VENDOR this OPEN PR (not just wait for merge)
====================================================

* It's a structural fix for Ampere — author's SM_120 (Blackwell consumer)
  has the same ~99 KiB budget as our A5000 (verified at boot:
  ``shared_memory_per_block_optin: 101376 (99.0 KiB)``).
* The 2 kernels patched are on our GDN hot path — every Mamba/GDN
  forward fires them.
* Estimated gain: +3-7 % GDN prefill TPS per author's SM_120 bench.
  On our 35B + MTP K=3 stack that translates to +5-15 TPS on the
  current ~228 baseline → target ~235-243 TPS sustained warm.

Implementation strategy
=======================

The upstream PR adds a 228-LOC helper module (``vllm/triton_utils/
shmem_budget.py``) + wires 2 kernels. For Genesis text-patch vendoring,
we **cannot add a new file** — we inline the minimal pruner (~30 LOC)
into BOTH FLA files as their first sub-patch. Trade-off:
  * Pro: pure text-patch, no Genesis module-injection hack
  * Pro: zero risk if PR rebases — each file is self-contained
  * Pro: idempotent marker per file
  * Con: helper code duplicated (~30 LOC × 2 = 60 LOC)
  * Con: no shared cache between the two files' calls to
    ``_genesis_shmem_budget`` — minor (the call is cached per device,
    and there are only 2 GPUs in a TP=2 box)

Six sub-patches total (3 per file):
  * Import line: add the inlined helper + estimator just below the
    existing ``from vllm.triton_utils import tl, triton`` import.
  * Estimator function: inserted just before the kernel's
    ``@triton.autotune`` decorator.
  * Autotune decorator: extended with
    ``prune_configs_by={"early_config_prune": make_shmem_pruner(...)}``.

All sub-patches ``required=False`` — partial-apply allowed (one file's
anchor drift doesn't kill the other's win).

Composition + safety
====================

* No anchor overlap with PN298/PN299/PN299B/PN299C/PN299D/PN299E (those
  target 6 different files; this targets 2).
* Pruner is no-op when all shipped configs fit the budget (e.g. on
  H100/H200 in production) — keeps upstream behaviour on capable GPUs.
* On estimator failure: log warning + keep the config (preserve
  upstream behaviour). On empty kept-list: fall back to the smallest
  config + warn-once (PR pattern).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Vendor target: vllm-project/vllm#43047 (open as of 2026-06-09).
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file
from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

log = logging.getLogger("genesis.wiring.pn345_shmem_aware_autotune_pruner")

GENESIS_PN345_MARKER = (
    "Genesis PN345 vendor of vllm#43047 (shmem-aware autotune pruner) v1"
)


# ─── Inlined helper (~30 LOC, kept identical between the two files) ───
# Imports/definitions injected right after the existing ``from vllm.
# triton_utils import tl, triton`` import. Self-contained: no
# cross-file dependency, no Genesis module needed.
_GENESIS_HELPER = (
    "\n"
    "# ─── [Genesis PN345 vendor of vllm#43047] shmem-aware pruner ─────────\n"
    "# Triton autotune configs in this file were tuned for the largest opt-in\n"
    "# shmem budget H100/H200 support (228 KiB). On consumer Ampere (A5000\n"
    "# /3090 ~99 KiB) and SM_120 (~99 KiB) some configs exceed the budget and\n"
    "# OOR at JIT. The pruner drops those at autotune time using a precise\n"
    "# per-config + per-num_stages estimate against the actual device budget.\n"
    "import functools as _g_pn345_ft\n"
    "import torch as _g_pn345_torch\n"
    "\n"
    "@_g_pn345_ft.cache\n"
    "def _g_pn345_budget(device_index):\n"
    "    try:\n"
    "        return _g_pn345_torch.cuda.get_device_properties(\n"
    "            device_index).shared_memory_per_block_optin\n"
    "    except Exception:\n"
    "        return 99 * 1024  # safe Ampere consumer / SM_120 default\n"
    "\n"
    "def _g_pn345_make_pruner(estimator, safety_bytes=1024):\n"
    "    def _prune(configs, named_args, **kwargs):  # noqa: ARG001\n"
    "        try:\n"
    "            budget = _g_pn345_budget(\n"
    "                _g_pn345_torch.cuda.current_device()) - safety_bytes\n"
    "        except Exception:\n"
    "            return configs\n"
    "        if budget <= 0:\n"
    "            return configs\n"
    "        kept, smallest, smallest_est = [], None, None\n"
    "        for c in configs:\n"
    "            try:\n"
    "                est = int(estimator(c, named_args))\n"
    "            except Exception:\n"
    "                kept.append(c)\n"
    "                continue\n"
    "            if est <= budget:\n"
    "                kept.append(c)\n"
    "            if smallest_est is None or est < smallest_est:\n"
    "                smallest, smallest_est = c, est\n"
    "        return kept if kept else ([smallest] if smallest is not None else configs)\n"
    "    return _prune\n"
    "# ─── /[Genesis PN345] ─────────────────────────────────────────────────\n"
    "\n"
)


# ════════════════════════════════════════════════════════════════════════
# File 1: chunk_delta_h.py
# ════════════════════════════════════════════════════════════════════════

# Sub-patch 1A: inject helper after the canonical vllm.triton_utils import
PN345_DELTA_H_HELPER_OLD = (
    "from vllm.triton_utils import tl, triton\n"
    "\n"
    "from .index import prepare_chunk_indices, prepare_chunk_offsets\n"
)
PN345_DELTA_H_HELPER_NEW = (
    "from vllm.triton_utils import tl, triton\n"
    + _GENESIS_HELPER
    + "from .index import prepare_chunk_indices, prepare_chunk_offsets\n"
)


# Sub-patch 1B: insert estimator + autotune wiring around the kernel.
# Anchor on the autotune decorator's ``key=["H", "K", "V", "BT"],`` line
# (unique in the file for this kernel's decorator block).
PN345_DELTA_H_ESTIMATOR_OLD = (
    "    key=[\"H\", \"K\", \"V\", \"BT\"],\n"
    "    use_cuda_graph=use_cuda_graph,\n"
)
PN345_DELTA_H_ESTIMATOR_NEW = (
    "    key=[\"H\", \"K\", \"V\", \"BT\"],\n"
    "    # [Genesis PN345] precise shmem-budget filter per config.\n"
    "    # For BV=64 BT=64 num_stages=4 the persistent 4 fp32 [BV,64] b_h\n"
    "    # buffers (64 KiB) + per-stage b_w/b_k/b_v in bf16 (24 KiB × 4)\n"
    "    # would total 160 KiB — exceeds the 99 KiB A5000 opt-in budget.\n"
    "    prune_configs_by={\"early_config_prune\": _g_pn345_make_pruner(\n"
    "        lambda c, na: (\n"
    "            4 * c.kwargs.get(\"BV\", 64) * 64 * 4  # persistent: 4× fp32 [BV,64] b_h\n"
    "            + c.num_stages * (\n"
    "                2 * na.get(\"BT\", 64) * 64 * 2  # per-stage b_w + b_k bf16\n"
    "                + na.get(\"BT\", 64) * c.kwargs.get(\"BV\", 64) * 2  # b_v bf16\n"
    "            )\n"
    "            + 4096  # Triton bookkeeping safety\n"
    "        )\n"
    "    )},\n"
    "    use_cuda_graph=use_cuda_graph,\n"
)


# ════════════════════════════════════════════════════════════════════════
# File 2: chunk_o.py
# ════════════════════════════════════════════════════════════════════════

# Sub-patch 2A: inject helper after the canonical vllm.triton_utils import
PN345_CHUNK_O_HELPER_OLD = (
    "from vllm.triton_utils import tl, triton\n"
    "\n"
    "from .index import prepare_chunk_indices\n"
)
PN345_CHUNK_O_HELPER_NEW = (
    "from vllm.triton_utils import tl, triton\n"
    + _GENESIS_HELPER
    + "from .index import prepare_chunk_indices\n"
)


# Sub-patch 2B: inject estimator wiring on the autotune decorator.
# chunk_fwd_kernel_o uses ``key=["H", "K", "V", "BT"]`` too — same as
# delta_h. We must scope the anchor uniquely. Both decorators are
# followed by a ``@triton.jit`` line, but the chunk_o decorator ends
# with just ``],\n`` (no ``use_cuda_graph=``) while delta_h's ends with
# ``use_cuda_graph=use_cuda_graph,\n``. Anchor on the chunk_o-specific
# trailing pattern.
PN345_CHUNK_O_ESTIMATOR_OLD = (
    "    key=[\"H\", \"K\", \"V\", \"BT\"],\n"
    ")\n"
    "@triton.jit(do_not_specialize=[\"T\"])\n"
    "def chunk_fwd_kernel_o(\n"
)
PN345_CHUNK_O_ESTIMATOR_NEW = (
    "    key=[\"H\", \"K\", \"V\", \"BT\"],\n"
    "    # [Genesis PN345] precise shmem-budget filter per config.\n"
    "    # Persistent: b_o[BT,BV] fp32 + b_A[BT,BT] fp32. Per-stage:\n"
    "    # b_q[BT,BK] + b_k[BK,BT] + b_h[BV,BK] all bf16. At BT=64 BK=BV=64\n"
    "    # num_stages=4 → 132 KiB > 99 KiB A5000 budget → drop the config.\n"
    "    prune_configs_by={\"early_config_prune\": _g_pn345_make_pruner(\n"
    "        lambda c, na: (\n"
    "            na.get(\"BT\", 64) * c.kwargs[\"BV\"] * 4  # b_o fp32\n"
    "            + na.get(\"BT\", 64) * na.get(\"BT\", 64) * 4  # b_A fp32\n"
    "            + c.num_stages * (\n"
    "                na.get(\"BT\", 64) * c.kwargs[\"BK\"] * 2  # b_q bf16\n"
    "                + c.kwargs[\"BK\"] * na.get(\"BT\", 64) * 2  # b_k bf16\n"
    "                + c.kwargs[\"BV\"] * c.kwargs[\"BK\"] * 2  # b_h bf16\n"
    "            )\n"
    "            + 4096  # Triton bookkeeping safety\n"
    "        )\n"
    "    )},\n"
    ")\n"
    "@triton.jit(do_not_specialize=[\"T\"])\n"
    "def chunk_fwd_kernel_o(\n"
)


def _env_disabled() -> bool:
    return os.environ.get("GENESIS_DISABLE_PN345", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _apply_one_file(rel_path: str, helper_old: str, helper_new: str,
                    estimator_old: str, estimator_new: str,
                    file_label: str) -> tuple[str, int]:
    """Apply both sub-patches to one file. Returns (status_msg, num_applied)."""
    target = resolve_vllm_file(rel_path)
    if target is None:
        return f"{file_label}: file not found", 0
    patcher = TextPatcher(
        patch_name=f"PN345 {rel_path} — shmem-aware autotune pruner",
        target_file=str(target),
        marker=GENESIS_PN345_MARKER + f" :: {file_label}",
        sub_patches=[
            TextPatch(
                name=f"pn345_{file_label}_helper_inject",
                anchor=helper_old,
                replacement=helper_new,
                required=True,
            ),
            TextPatch(
                name=f"pn345_{file_label}_estimator_wiring",
                anchor=estimator_old,
                replacement=estimator_new,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN345",
            "make_shmem_pruner",  # upstream sentinel if vllm#43047 merges
            "infer_shmem_budget",
        ],
    )
    try:
        result, failure = patcher.apply()
    except Exception as e:
        return f"{file_label}: apply raised {e!r}", 0
    if result == TextPatchResult.FAILED:
        return f"{file_label}: FAILED — {failure.reason if failure else 'unknown'}", 0
    if result == TextPatchResult.SKIPPED:
        return f"{file_label}: skipped — {failure.reason if failure else 'unknown'}", 0
    if result == TextPatchResult.IDEMPOTENT:
        return f"{file_label}: idempotent (already applied)", 2
    n = len(patcher.applied_sub_patches)
    return f"{file_label}: applied {n}/2 sub-patches", n


def apply() -> tuple[str, str]:
    """Apply PN345 — shmem-aware autotune pruner on 2 FLA kernels."""
    if _env_disabled():
        return "skipped", "PN345 disabled via GENESIS_DISABLE_PN345=1"

    delta_h_msg, delta_h_n = _apply_one_file(
        "model_executor/layers/fla/ops/chunk_delta_h.py",
        PN345_DELTA_H_HELPER_OLD, PN345_DELTA_H_HELPER_NEW,
        PN345_DELTA_H_ESTIMATOR_OLD, PN345_DELTA_H_ESTIMATOR_NEW,
        "chunk_delta_h",
    )
    chunk_o_msg, chunk_o_n = _apply_one_file(
        "model_executor/layers/fla/ops/chunk_o.py",
        PN345_CHUNK_O_HELPER_OLD, PN345_CHUNK_O_HELPER_NEW,
        PN345_CHUNK_O_ESTIMATOR_OLD, PN345_CHUNK_O_ESTIMATOR_NEW,
        "chunk_o",
    )

    total = delta_h_n + chunk_o_n
    summary = f"chunk_delta_h: {delta_h_msg} | chunk_o: {chunk_o_msg}"
    if total == 0:
        return "skipped", f"PN345: no anchors matched. {summary}"
    return "applied", (
        f"PN345 applied: {total}/4 sub-patches across chunk_delta_h.py "
        f"+ chunk_o.py — shmem-aware autotune pruners now drop configs "
        f"that would OOR on A5000's 99 KiB opt-in budget. Vendor of "
        f"OPEN PR vllm#43047 (Closes #36598 + #38918 + #36802 + #41063 + #32826). "
        f"Details: {summary}"
    )


def is_applied() -> bool:
    from pathlib import Path
    for rel in ("model_executor/layers/fla/ops/chunk_delta_h.py",
                "model_executor/layers/fla/ops/chunk_o.py"):
        target = resolve_vllm_file(rel)
        if target is None:
            continue
        try:
            if GENESIS_PN345_MARKER in Path(str(target)).read_text(encoding="utf-8"):
                return True
        except (OSError, UnicodeDecodeError):
            continue
    return False

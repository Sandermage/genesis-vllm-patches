# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 105 — TQ K/V dequant kernel num_stages=3 (lmdeploy sm86 hint).

Per 2026-04-28 cross-engine grail research: lmdeploy uses
`num_warps=4, num_stages=3` for Ampere SM 8.6 prefill kernels with
D≤128. Our TQ K/V dequant kernel `_tq_full_dequant_kv` (in
turboquant_attn.py::_continuation_prefill) currently uses num_warps=4
without explicit num_stages — Triton defaults (likely num_stages=2).

Adding `num_stages=3` enables 3-stage async-copy pipeline on Ampere,
which can hide DRAM latency on KV reads during long-context prefill.

Predicted gain: +0.5-1% on long-context prefill workloads.
Risk: low — single line addition; if shared mem exceeds, Triton fails
to compile and we'd see boot error (revert immediately).

Status: opt-in via `GENESIS_ENABLE_P105=1`. Default OFF.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Reference: 2026-04-28 grail kernel research dump (lmdeploy sm86 config).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.p105_tq_dequant_num_stages")

GENESIS_P105_MARKER = (
    "Genesis P105 TQ K/V dequant num_stages=3 (lmdeploy sm86 hint) v7.62.25"
)


P105_OLD = (
    "            FP8_E4B15=_use_fp8_e4b15(device.index or 0),\n"
    "            num_warps=4,\n"
    "        )\n"
)


P105_NEW = (
    "            FP8_E4B15=_use_fp8_e4b15(device.index or 0),\n"
    "            num_warps=4,\n"
    "            num_stages=3,  # [Genesis P105] lmdeploy sm86 hint — 3-stage async pipe\n"
    "        )\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P105 turboquant_attn.py — TQ K/V dequant num_stages=3",
        target_file=str(target),
        marker=GENESIS_P105_MARKER,
        sub_patches=[
            TextPatch(
                name="p105_tq_dequant_num_stages_3",
                anchor=P105_OLD,
                replacement=P105_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P105",
            "num_stages=3,  # [Genesis P105]",
        ],
    )


def apply() -> tuple[str, str]:
    from vllm._genesis.dispatcher import log_decision, should_apply

    decision, reason = should_apply("P105")
    log_decision("P105", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "turboquant_attn.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m.startswith("[Genesis"):
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file}",
            )

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return (
        "applied",
        "P105 v7.62.25 applied: TQ K/V dequant kernel now uses num_stages=3 "
        "(was Triton default 2). Predicted +0.5-1% prefill on long ctx via "
        "deeper async-copy pipeline. Auto-noop if Triton can't fit shmem."
    )


def is_applied() -> bool:
    if vllm_install_root() is None:
        return False
    patcher = _make_patcher()
    if patcher is None:
        return False
    try:
        with open(patcher.target_file) as f:
            return patcher.marker in f.read()
    except Exception:
        return False

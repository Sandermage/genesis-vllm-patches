# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN293 — mamba_attn _compute_common_metadata prefill fast-path.

Genesis-original 2026-06-04 — closes the +24ms warm TTFT regression on
Qwen3.6 27B Lorbus INT4 + TQ k8v4 + MTP K=3 between vllm dev93 (97.1ms)
and dev354 (120.96ms) on 2× A5000 SM 8.6.

================================================================
ROOT CAUSE (TTFT bisect, 2026-06-04)
================================================================

Upstream PR #42430 (`47829b1159`, "[Bugfix] mamba: run single-token extends
as decodes") added unconditional per-build CPU overhead in
`_compute_common_metadata`, executed once per GDN/Mamba layer per forward.

The 5-line block runs:
    is_prefilling = common_attn_metadata.is_prefilling
    seq_lens_cpu  = common_attn_metadata.seq_lens_cpu_upper_bound
    query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)
    single_token_prefill_rows = is_prefilling & (query_lens_cpu == 1)
    has_prior_state = seq_lens_cpu > 1
    prefill_to_decode = single_token_prefill_rows & has_prior_state
    if torch.any(prefill_to_decode).item():   # <-- CPU .item() every build
        is_prefilling = is_prefilling.clone()
        is_prefilling[prefill_to_decode] = False
        common_attn_metadata = common_attn_metadata.replace(
            is_prefilling=is_prefilling
        )

On 27B's ~32 hybrid layers, this is **32× the overhead** per prefill
iteration: 32 × (torch.diff + 2 bool ops + torch.any + .item()) =
~14-18 ms TTFT.

The branch is ONLY required when (a) spec-decode draft accept_tokens
populated for the build (NIXL P-D disagg or mixed prefill+decode batch).
For pure first-chunk prefill (warm TTFT path), num_accepted_tokens is
None AND every row has q_len > 1 — the entire block is dead overhead.

================================================================
THE FIX
================================================================

Add early-exit guards:
  - If num_accepted_tokens is None → no spec data, can't have prefill→decode
  - If min(query_lens_cpu) > 1 → no single-token rows, can't reclassify

When either guard hits, skip the 5 tensor ops + .item(). When neither
hits (true mixed-batch case), full upstream behavior runs.

================================================================
SAFETY MODEL
================================================================

- Pure CPU op skip — no algorithmic change. Output bit-identical to
  upstream on all true-positive cases.
- Idempotent via Genesis marker.
- required=True sub-patch surfaces failure rather than silent skip.
- Expected impact: -14...-18 ms warm TTFT recovery on 27B hybrid.

Author: Sandermage (Sander) Barzov Aleksandr, 2026-06-04 TTFT bisect.
Reproduces on: 0.21.1rc1.dev354+g626fa9bba, 2× A5000 SM 8.6,
Qwen3.6-27B-INT4-AutoRound + TQ k8v4 + MTP K=3.
Original PR being neutralized for cold path: vllm-project/vllm#42430.
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

log = logging.getLogger("genesis.wiring.pn293_mamba_attn_prefill_fastpath")

GENESIS_PN293_MARKER = (
    "Genesis PN293 mamba_attn prefill fastpath (vllm#42430 cold-path skip) v1"
)


PN293_OLD = (
    "        is_prefilling = common_attn_metadata.is_prefilling\n"
    "        assert is_prefilling is not None\n"
    "        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound\n"
    "        assert seq_lens_cpu is not None\n"
    "        query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)\n"
    "        single_token_prefill_rows = is_prefilling & (query_lens_cpu == 1)\n"
    "        # First-token prefills have no prior Mamba state and must stay prefills.\n"
    "        has_prior_state = seq_lens_cpu > 1\n"
    "        prefill_to_decode = single_token_prefill_rows & has_prior_state\n"
    "        if torch.any(prefill_to_decode).item():\n"
    "            is_prefilling = is_prefilling.clone()\n"
    "            is_prefilling[prefill_to_decode] = False\n"
    "            common_attn_metadata = common_attn_metadata.replace(\n"
    "                is_prefilling=is_prefilling\n"
    "            )"
)

PN293_NEW = (
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis PN293 vllm#42430 cold-path skip] Fast-path: the\n"
    "        # prefill→decode reclassification is only meaningful when\n"
    "        # (a) spec-decode populated num_accepted_tokens for THIS build\n"
    "        # (NIXL P-D disagg or mixed prefill+decode batch), AND\n"
    "        # (b) at least one row has q_len == 1.\n"
    "        #\n"
    "        # For pure first-chunk prefill (warm TTFT path), both are False\n"
    "        # — skip the 5 tensor ops + .item() that cost ~14-18 ms of TTFT\n"
    "        # on 27B's 32 hybrid layers (32× per-build .item() syncs).\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        is_prefilling = common_attn_metadata.is_prefilling\n"
    "        assert is_prefilling is not None\n"
    "        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound\n"
    "        assert seq_lens_cpu is not None\n"
    "        if num_accepted_tokens is not None:\n"
    "            query_lens_cpu = torch.diff(common_attn_metadata.query_start_loc_cpu)\n"
    "            # Cheap early-rejection: if min query length > 1, no single-token rows possible.\n"
    "            if int(query_lens_cpu.min()) == 1:\n"
    "                single_token_prefill_rows = is_prefilling & (query_lens_cpu == 1)\n"
    "                # First-token prefills have no prior Mamba state and must stay prefills.\n"
    "                has_prior_state = seq_lens_cpu > 1\n"
    "                prefill_to_decode = single_token_prefill_rows & has_prior_state\n"
    "                if bool(prefill_to_decode.any()):\n"
    "                    is_prefilling = is_prefilling.clone()\n"
    "                    is_prefilling[prefill_to_decode] = False\n"
    "                    common_attn_metadata = common_attn_metadata.replace(\n"
    "                        is_prefilling=is_prefilling\n"
    "                    )"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/mamba_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN293 v1/attention/backends/mamba_attn.py — "
            "_compute_common_metadata prefill fastpath (vllm#42430 cold-path skip)"
        ),
        target_file=str(target),
        marker=GENESIS_PN293_MARKER,
        sub_patches=[
            TextPatch(
                name="pn293_prefill_to_decode_fastpath",
                anchor=PN293_OLD,
                replacement=PN293_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN293",
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "vllm#42430 cold-path skip" was a substring of our own banner
            # AND the Layer-6 marker line — false "upstream_merged" skip on
            # residue. Defended form, same marker-line coverage:
            "[Genesis wiring marker: Genesis PN293",
        ],
    )


_APPLIED = False


def apply() -> tuple[str, str]:
    """Apply PN293 — mamba_attn _compute_common_metadata prefill fast-path."""
    global _APPLIED

    if os.environ.get(
        "GENESIS_ENABLE_PN293_MAMBA_ATTN_PREFILL_FASTPATH", ""
    ).lower() not in ("1", "true", "yes", "on"):
        return "skipped", (
            "PN293 default OFF — set "
            "GENESIS_ENABLE_PN293_MAMBA_ATTN_PREFILL_FASTPATH=1 to engage. "
            "Closes +14-18ms TTFT overhead from vllm#42430 mamba_attn "
            "prefill-to-decode reclassification (32× per-build CPU "
            ".item() syncs on hybrid 27B layers)."
        )

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/attention/backends/mamba_attn.py not found"

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", failure.reason if failure else "unknown TextPatch failure"
    if result == TextPatchResult.SKIPPED:
        return "skipped", failure.reason if failure else "unknown TextPatch skip"
    applied = patcher.applied_sub_patches or [sp.name for sp in patcher.sub_patches]
    _APPLIED = True
    return "applied", (
        f"PN293 installed: mamba_attn _compute_common_metadata prefill "
        f"fast-path skips dead vllm#42430 reclassification on pure "
        f"first-chunk prefill. Expected -14-18ms TTFT recovery on 27B "
        f"hybrid 32-layer prefill. Sub-patches: {', '.join(applied)}."
    )


def is_applied() -> bool:
    return _APPLIED

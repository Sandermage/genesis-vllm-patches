# SPDX-License-Identifier: Apache-2.0
"""Wiring for PN59 — streaming GDN orchestrator (Variant D Phase 2).

Text-patches `vllm/model_executor/layers/fla/ops/chunk.py` to redirect
`chunk_gated_delta_rule_fwd` through Genesis's window-iterative
`streaming_chunk_gated_delta_rule_fwd` driver when eligible.

Eliminates Cliff 2b multi-turn OOM by replacing the `(B, NT, H, V, K)`
single-allocation peak (805 MiB at T=64K Genesis 27B Lorbus shapes)
with shape-keyed scratch pool of `(B, WINDOW_NT, H, V, K)` (~3-12 MiB).

Independent confirmation (issue #20, 2026-05-05) — noonghunna:
"the limitation is the triton kernel for cliff 2; doesn't appear with
llama.cpp" — exact materialization pattern this fix removes.

Architecture
------------
PN59 is a **single-anchor text patch** on the body of
`chunk_gated_delta_rule_fwd` orchestrator. Replacement wraps the
entire function body in a runtime dispatcher:

  - If GENESIS_ENABLE_PN59_STREAMING_GDN=1 AND eligible (single-seq,
    long-T, NVIDIA CUDA) → call `streaming_chunk_gated_delta_rule_fwd`
    (Genesis-managed window-iterative driver)
  - Otherwise → run vanilla code unchanged (zero-regression contract)

Strict no-regression: any failure in streaming path → fall through
to vanilla path with WARNING log.

Compatibility
-------------
- **PN50** GDN proj fusion — operates BEFORE chunk_gated_delta_rule
  (in gdn_linear_attn.py); orthogonal, no conflict
- **PN54** GDN contiguous dedup — operates on ssm_state read; orthogonal
- **P103** FLA Cliff2 chunked — operates AT outer-orchestrator level;
  PN59 supersedes when both ON (auto-fallthrough handles)
- **PN26b** sparse-V — non-GDN attention path; orthogonal

Default OFF until live A/B prod-validates on 27B Lorbus.

Author: Sandermage 2026-05-05, Variant D Phase 2.
Phase 1 numerical proof: tests/integration/test_streaming_gdn_numerical.py
Cross-engine references: llama.cpp ssm-scan.cu (register-streaming),
  Mamba2 ssd_combined (3-stage chunk split), FLA RFC #485.
"""
from __future__ import annotations

import logging
import os

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.kernel import (
    TextPatch,
    TextPatcher,
    TextPatchResult,
)

log = logging.getLogger("genesis.wiring.pn59_streaming_gdn")

GENESIS_PN59_MARKER = "Genesis PN59 streaming GDN orchestrator (Variant D Phase 2)_v11.3.0_hotpath"


def _is_enabled() -> bool:
    return os.environ.get(
        "GENESIS_ENABLE_PN59_STREAMING_GDN", ""
    ).strip().lower() in ("1", "true", "yes", "on")


# Anchor on the entire `chunk_gated_delta_rule_fwd` function body.
# Pristine upstream + post-Genesis state both have this signature
# unchanged (Genesis P103 modifies a different orchestrator wrap).
ANCHOR_OLD = (
    "def chunk_gated_delta_rule_fwd(\n"
    "    q: torch.Tensor,\n"
    "    k: torch.Tensor,\n"
    "    v: torch.Tensor,\n"
    "    g: torch.Tensor,\n"
    "    beta: torch.Tensor,\n"
    "    scale: float,\n"
    "    initial_state: torch.Tensor,\n"
    "    output_final_state: bool,\n"
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    core_attn_out: torch.Tensor | None = None,\n"
    "):\n"
)

ANCHOR_NEW = (
    "def chunk_gated_delta_rule_fwd(\n"
    "    q: torch.Tensor,\n"
    "    k: torch.Tensor,\n"
    "    v: torch.Tensor,\n"
    "    g: torch.Tensor,\n"
    "    beta: torch.Tensor,\n"
    "    scale: float,\n"
    "    initial_state: torch.Tensor,\n"
    "    output_final_state: bool,\n"
    "    cu_seqlens: torch.Tensor | None = None,\n"
    "    chunk_indices: torch.Tensor | None = None,\n"
    "    chunk_offsets: torch.Tensor | None = None,\n"
    "    core_attn_out: torch.Tensor | None = None,\n"
    "):\n"
    "    # [Genesis PN59 v2 Variant D Phase 2 — hot-path optimized]\n"
    "    # Streaming-GDN dispatch — cache the streaming fn callable in\n"
    "    # upstream-file globals (this runs per chunk_gated_delta_rule_fwd\n"
    "    # call — high frequency on GDN/Mamba-heavy models).\n"
    "    _genesis_pn59_streaming = globals().get('_GENESIS_PN59_streaming_fn')\n"
    "    if _genesis_pn59_streaming is None:\n"
    "        try:\n"
    "            from sndr.engines.vllm.kernels_legacy.streaming_gdn_driver import (\n"
    "                streaming_chunk_gated_delta_rule_fwd as _genesis_pn59_streaming,\n"
    "            )\n"
    "        except Exception:\n"
    "            _genesis_pn59_streaming = False\n"
    "        globals()['_GENESIS_PN59_streaming_fn'] = _genesis_pn59_streaming\n"
    "    try:\n"
    "        if not _genesis_pn59_streaming:\n"
    "            raise ImportError('PN59 streaming driver not available')\n"
    "        return _genesis_pn59_streaming(\n"
    "            q=q, k=k, v=v, g=g, beta=beta, scale=scale,\n"
    "            initial_state=initial_state,\n"
    "            output_final_state=output_final_state,\n"
    "            cu_seqlens=cu_seqlens,\n"
    "            chunk_indices=chunk_indices,\n"
    "            chunk_offsets=chunk_offsets,\n"
    "            chunk_local_cumsum=chunk_local_cumsum,\n"
    "            chunk_scaled_dot_kkt_fwd=chunk_scaled_dot_kkt_fwd,\n"
    "            solve_tril=solve_tril,\n"
    "            recompute_w_u_fwd=recompute_w_u_fwd,\n"
    "            chunk_gated_delta_rule_fwd_h=chunk_gated_delta_rule_fwd_h,\n"
    "            chunk_fwd_o=chunk_fwd_o,\n"
    "            SUPPRESS_LEVEL=SUPPRESS_LEVEL,\n"
    "        )\n"
    "    except Exception as _genesis_pn59_err:\n"
    "        # club-3090#22 fix 2026-05-07: surface bypass cause for diagnostics.\n"
    "        # Original silent `except: pass` masked PN59 runtime errors → operators\n"
    "        # only saw downstream OOM with no link back to PN59. WARN-level lets\n"
    "        # genesis.kernels.streaming_gdn_driver log subscribers see what failed.\n"
    "        # Defensive: fall through to vanilla original code below regardless.\n"
    "        try:\n"
    "            import logging as _genesis_pn59_log\n"
    "            _genesis_pn59_log.getLogger(\n"
    "                \"genesis.kernels.streaming_gdn_driver\"\n"
    "            ).warning(\n"
    "                \"[PN59] dispatch failed (%s: %s) — vanilla fallback engaged. \"\n"
    "                \"Set GENESIS_PN59_DEBUG=1 for traceback.\",\n"
    "                type(_genesis_pn59_err).__name__,\n"
    "                str(_genesis_pn59_err)[:200],\n"
    "            )\n"
    "        except Exception:\n"
    "            pass  # logger import failed — preserve original silent semantics\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("model_executor/layers/fla/ops/chunk.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="PN59 streaming GDN orchestrator (Variant D Phase 2)",
        target_file=str(target),
        marker=GENESIS_PN59_MARKER,
        sub_patches=[
            TextPatch(
                name="pn59_chunk_orchestrator_dispatch",
                anchor=ANCHOR_OLD,
                replacement=ANCHOR_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            # If upstream FLA lands #485 (Songlin Yang memory_efficient flag),
            # signature changes → drift detected → SKIP cleanly
            "memory_efficient",
            "streaming_window_chunks",
            # Self-collision lint (triage plan §6 2026-06-11): former entry
            # "streaming_chunk_gated_delta_rule_fwd" is our own
            # orchestrator function called by the replacement text — false
            # "upstream_merged" skip on residue. An upstream equivalent is
            # still caught by the two markers above + anchor mismatch.
            # 2026-05-06: vllm PR #41824 (Kermit-C) adds `ssm_state_indices`
            # parameter to chunk_gated_delta_rule_fwd for in-place SSM state
            # access (eliminates gather/scatter). PN59's anchor matches the
            # OLD signature. When #41824 lands in our pin, the new param
            # appears in the function body → drift fires → PN59 SKIPs cleanly.
            # At that point, PN59 needs re-anchoring to the new signature
            # (or evaluation whether streaming-GDN is still needed given the
            # in-place fix changes the memory pressure profile).
            "ssm_state_indices",
        ],
    )


def apply() -> tuple[str, str]:
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN59")
    log_decision("PN59", decision, reason)
    if not decision:
        return "skipped", reason
    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"
    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "fla/ops/chunk.py not found"
    result, failure = patcher.apply()
    if result == TextPatchResult.APPLIED:
        return (
            "applied",
            "PN59 applied: streaming-GDN dispatcher inserted; runtime "
            "engages when single-seq long-T (eliminates Cliff 2b OOM)",
        )
    if result == TextPatchResult.IDEMPOTENT:
        return "applied", "already applied (idempotent)"
    if result == TextPatchResult.SKIPPED:
        msg = failure.reason if failure else "anchor not found"
        return ("skipped",
                f"{msg} — likely upstream merged #485-style fix or signature drift")
    return "failed", failure.reason if failure else "unknown failure"

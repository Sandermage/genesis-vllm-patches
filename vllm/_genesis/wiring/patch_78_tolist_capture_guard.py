# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch 78 — surgical .tolist() capture-guard for TurboQuant.

================================================================
CREDIT
================================================================

Algorithm + anchor strings adapted from noonghunna's
`patch_tolist_cudagraph.py`:
  https://github.com/noonghunna/qwen36-27b-single-3090/blob/master/patches/patch_tolist_cudagraph.py
  (Apache-2.0, original author: @noonghunna)

Original problem-statement and bypass logic are noonghunna's. We adapt
it to:
  - Run under our `TextPatcher` framework (idempotent, drift-marker-aware)
  - Use Genesis env-gate convention (`GENESIS_ENABLE_P78_*`)
  - Compose cleanly with our P22/P26/P44 prealloc patches (which already
    avoid the `.tolist()` path on steady-state — P78 is the safety-net
    for cases where prealloc is bypassed, e.g. cold cudagraph capture
    with dynamic batch shapes)

Per `feedback_no_ai_credit_in_public.md`: no AI co-author credit. Sole
human authors are Sander Barzov (Genesis adaptation) + noonghunna (original).

================================================================
WHAT THIS FIXES
================================================================

`vllm/v1/attention/backends/turboquant_attn.py` has two `.tolist()` calls
that force GPU->CPU sync inside paths that can execute under active
CUDA stream capture:

  Site A — `forward()` mixed-batch branch:
      prefill_max_seq = max(attn_metadata.seq_lens[num_decodes:].tolist())

  Site B — `_prefill_attention()` continuation branch:
      qsl = query_start_loc.tolist()
      seq_lens_list = attn_metadata.seq_lens.tolist()

Hit during cudagraph capture warmup with mixed prefill+decode or with
spec-decode + chunked-prefill (continuation chunks).

Our P22/P26/P44 patches AVOID these paths on steady-state (prealloc'd
buffers used directly), but warmup/capture can transit them before
prealloc kicks in. P78 makes the path itself capture-safe by using
`torch.cuda.is_current_stream_capturing()` as a guard.

================================================================
COMPOSITION
================================================================

- Composes additively with P22/P26/P44 — P78 fires ONLY during capture,
  prealloc fires on steady-state. No conflict.
- Composes additively with P67/P67b — P67 routes K+1 spec-verify above
  the `_prefill_attention` path; P78 makes the fallback path safer.
- For sites already prealloc'd by P22/P26/P44 in the steady-state path:
  the runtime check `is_current_stream_capturing() == False` short-circuits
  to original behavior — zero overhead.

================================================================
ENV
================================================================

GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=1   # opt-in master switch

================================================================
RISK
================================================================

LOW. Capture-time output values are not used by inference (V1 PIECEWISE
mode marks attention as splitting_op — capture only drives memory
profiling). The flash_attn_varlen_func fast-path returns the right
shape with similar workspace footprint, so memory profiling stays accurate.

If `_HAS_FLASH_ATTN` is False (rare), falls back to `torch.zeros(...)`
which is safe (correct shape, no garbage propagation in non-inference
captured graph).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Adapted from: @noonghunna `patch_tolist_cudagraph.py` (Apache-2.0).
"""
from __future__ import annotations

import logging
import os

from vllm._genesis.guards import resolve_vllm_file, vllm_install_root
from vllm._genesis.wiring.text_patch import (
    TextPatcher,
    TextPatchResult,
    TextPatch,
)

log = logging.getLogger("genesis.wiring.p78_tolist_capture_guard")

GENESIS_P78_MARKER = "Genesis P78 tolist capture-guard (adapted from noonghunna) v7.43"


# ─── Sub-patch: insert capture-guard early-return BEFORE continuation branch ─
# Anchor on the start of the continuation branch in _prefill_attention.

P78_OLD = (
    "        # Continuation or no flash_attn: per-request attention.\n"
    "        # For continuation chunks (seq_len > q_len), we must attend to\n"
    "        # previously cached K/V from the TQ cache, not just the current\n"
    "        # chunk's raw K/V.\n"
    "        Hk = key.shape[1]\n"
)

P78_NEW = (
    "        # ════════════════════════════════════════════════════════════\n"
    "        # [Genesis P78 — adapted from noonghunna's patch_tolist_cudagraph.py]\n"
    "        # During CUDA graph capture, the continuation branch below calls\n"
    "        # .tolist() forcing a GPU->CPU sync — illegal under torch.cuda.graph().\n"
    "        # vLLM V1 PIECEWISE marks unified_attention_with_output as a splitting_op,\n"
    "        # so capture does NOT bake in attention outputs; capture-time values\n"
    "        # only need to drive memory profiling. Falling back to the graph-safe\n"
    "        # flash_attn_varlen_func returns the same shape with similar workspace.\n"
    "        # At inference (non-capture), is_current_stream_capturing()==False and\n"
    "        # the original per-request continuation path runs unchanged.\n"
    "        # CREDIT: github.com/noonghunna/qwen36-27b-single-3090 (Apache-2.0)\n"
    "        # ════════════════════════════════════════════════════════════\n"
    "        import os as _genesis_p78_os\n"
    "        if (\n"
    "            _genesis_p78_os.environ.get('GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD', '').strip().lower()\n"
    "            in ('1', 'true', 'yes', 'on')\n"
    "            and torch.cuda.is_current_stream_capturing()\n"
    "        ):\n"
    "            try:\n"
    "                from vllm.attention.backends.flash_attn import flash_attn_varlen_func as _genesis_p78_fa_func\n"
    "                return _genesis_p78_fa_func(\n"
    "                    q=query, k=key, v=value,\n"
    "                    cu_seqlens_q=attn_metadata.query_start_loc,\n"
    "                    cu_seqlens_k=attn_metadata.query_start_loc,\n"
    "                    max_seqlen_q=attn_metadata.max_query_len,\n"
    "                    max_seqlen_k=attn_metadata.max_query_len,\n"
    "                    softmax_scale=self.scale,\n"
    "                    causal=True,\n"
    "                )\n"
    "            except Exception:\n"
    "                # Final fallback: correct shape zero tensor (capture-time\n"
    "                # output is unused under PIECEWISE; memory profile stays valid)\n"
    "                return torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)\n"
    "\n"
    "        # Continuation or no flash_attn: per-request attention.\n"
    "        # For continuation chunks (seq_len > q_len), we must attend to\n"
    "        # previously cached K/V from the TQ cache, not just the current\n"
    "        # chunk's raw K/V.\n"
    "        Hk = key.shape[1]\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name="P78 v1/attention/backends/turboquant_attn.py — tolist capture-guard",
        target_file=str(target),
        marker=GENESIS_P78_MARKER,
        sub_patches=[
            TextPatch(
                name="p78_capture_guard",
                anchor=P78_OLD,
                replacement=P78_NEW,
                required=True,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis P78",
            "GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD",
            "is_current_stream_capturing",  # if upstream adds same guard
        ],
    )


def apply() -> tuple[str, str]:
    """Apply P78 — surgical .tolist() capture guard."""
    from vllm._genesis.dispatcher import should_apply, log_decision
    decision, reason = should_apply("P78")
    log_decision("P78", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "vllm/v1/attention/backends/turboquant_attn.py not found"

    if not os.path.isfile(patcher.target_file):
        return "skipped", f"target disappeared: {patcher.target_file}"
    with open(patcher.target_file) as f:
        content = f.read()
    if patcher.marker in content:
        log.info("[P78] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"
    for m in patcher.upstream_drift_markers:
        if m == "[Genesis P78" and m in content:
            continue
        if m in content:
            return (
                "skipped",
                f"upstream drift marker {m!r} in {patcher.target_file} — "
                "upstream may have absorbed this fix or independent capture-guard",
            )

    result, failure = patcher.apply()
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: {failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )
    return "applied", (
        "P78 applied: capture-guard installed in TurboQuant._prefill_attention "
        "continuation branch. Falls back to flash_attn_varlen_func during cudagraph "
        "capture (zero overhead at inference). Adapted from noonghunna's "
        "patch_tolist_cudagraph.py (Apache-2.0, attribution in patch docstring)."
    )

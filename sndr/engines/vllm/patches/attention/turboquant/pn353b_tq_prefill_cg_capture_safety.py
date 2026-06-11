# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch PN353B — TurboQuant prefill CUDA-graph capture safety.

Backport of OPEN upstream vllm#43747 (`oneraghavan`, 2026-05-15):
  ``[Bugfix][TurboQuant] Fix CUDA graph capture crash with spec-decode +
   chunked-prefill (#40807)``
  Closes vllm Issue #40807 (noonghunna, 2026-04-24).

================================================================
WHAT THIS PATCH DOES
================================================================

Root cause: TurboQuant's `_prefill_attention` continuation branch
calls `.tolist()` on GPU tensors (`query_start_loc`, `seq_lens`),
which is illegal during CUDA-graph capture. This crashes engine init
when TurboQuant KV-cache (`turboquant_k8v4`, `turboquant_4bit_nc`,
`turboquant_3bit_nc`) is combined with `--speculative-config
method=mtp` and `--enable-chunked-prefill` — EXACTLY our PROD config
(35B-A3B FP8 + MTP K=3 + TQ k8v4).

The PR applies three coupled fixes:

  1) Downgrade `_cudagraph_support` from `UNIFORM_BATCH` to
     `UNIFORM_SINGLE_TOKEN_DECODE`. This tells the compilation
     framework "do NOT full-capture spec-decode K+1 verify batches".
     Spec-decode K+1 gets PIECEWISE capture instead, so the
     continuation branch never executes under torch.cuda.graph().

  2) `build_for_cudagraph_capture` always populates CPU-resident
     `seq_lens_cpu` and `query_start_loc_cpu` so even if the
     continuation branch is reached, it reads from CPU tensors
     (where `.tolist()` is a Python op, NOT a CUDA sync).

  3) `_prefill_attention` continuation branch adds a
     defense-in-depth early-return when
     `torch.cuda.is_current_stream_capturing()` is True — returns
     a zero tensor of the correct shape (capture-time output is
     unused under PIECEWISE; memory profile stays valid).

================================================================
RELATIONSHIP TO EXISTING GENESIS PATCHES
================================================================

P65 (TurboQuant spec-decode CG downgrade):
  - P65 is DEFAULT OFF and conflicts here. P65 replaces the same
    ClassVar line with a CLASSMETHOD `get_cudagraph_support` that
    conditionally downgrades. PR #43747 just hard-flips the ClassVar.
  - Decision: PN353B's ClassVar anchor only matches the STOCK line
    `_cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH`.
    If P65 has been applied first, the anchor is consumed and PN353B
    skips (drift-marker `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`
    triggers). If P65 is OFF (current default), PN353B applies cleanly.
  - PN353B and P65 both achieve the same effect for spec-decode (force
    PIECEWISE). PN353B is the UNCONDITIONAL upstream-aligned fix; P65
    is the SUPERSEDED conditional Genesis-original. With PN353B,
    P65 becomes redundant.

P78 (tolist capture-guard, Sites C/D/E):
  - P78 already added `prefill_max_seq_cpu: int` field, pre-computes
    it in Builder.build() from CPU seq_lens, reads it in forward().
    P78 closes the FORWARD-side .tolist() (Site E).
  - P78 does NOT cover the `_prefill_attention` continuation branch
    (Sites B's flash_attn_varlen capture-guard is gated behind env
    `GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD` which is OFF by default).
  - PN353B's anchor #2 (build_for_cudagraph_capture CPU-copies) lives
    in a DIFFERENT METHOD from P78 (P78 modifies build(); PN353B
    modifies build_for_cudagraph_capture()). No anchor collision.
  - PN353B's anchor #3 (continuation early-return) targets the same
    region as P78 Site B but with DIFFERENT anchor text:
      * P78 Site B anchor: "Continuation or no flash_attn: per-request"
        and SKIP_LIST early-return uses flash_attn_varlen_func
      * PN353B anchor: "For continuation chunks (seq_len > q_len)"
        and inserts `torch.zeros(N, Hq, D, ...)` early-return
    These can NOT both apply — anchor text differs but covers same
    semantic region. Strategy: PN353B uses a slightly later anchor
    that survives P78 Site B's insertion if both are applied.

PN116 (TurboQuant prefill max_seq_len fallback fix):
  - Independent code path (forward()'s mixed-batch branch); no overlap.

P101 (TurboQuant continuation 64-token slicing):
  - P101 ONLY modifies the use_decode_continuation block deeper inside
    the continuation branch. PN353B's early-return at the TOP of the
    continuation branch fires BEFORE P101's region. No conflict.

PN118 (workspace fallback): independent code path. No conflict.

================================================================
SAFETY MODEL
================================================================

Risk: LOW-MEDIUM. The ClassVar downgrade (anchor #1) has a real
performance side-effect: spec-decode K+1 batches lose FULL cudagraph
capture and fall to PIECEWISE. On Ampere small-batch single-stream
this is ~5-8% TPS hit. BUT — without this fix our config crashes
outright during CUDA-graph warmup (verified bug present at line 211
in container with vllm pin dev259+g303916e93). The "crash"
outcome is strictly worse than the "5-8% perf hit" outcome.

Idempotent via Genesis marker at file head.
Drift retreat: required=True on anchor #1 only (downgrade); anchors
#2 and #3 are required=False (defense-in-depth; CPU-copy + zero-return
are nice-to-have safety nets on top of the downgrade).

================================================================

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Original: oneraghavan — vllm#43747 (OPEN). Closes #40807.
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

log = logging.getLogger("genesis.wiring.pn353b_tq_prefill_cg_capture_safety")

GENESIS_PN353B_MARKER = (
    "Genesis PN353B TQ prefill CG capture safety "
    "(backport: vllm#43747 closes vllm#40807) v1"
)


# ─────────────────────────────────────────────────────────────────────
# ANCHOR #1 — _cudagraph_support ClassVar downgrade.
# ─────────────────────────────────────────────────────────────────────

PN353B_ANCHOR_CG_OLD = (
    "    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH\n"
)

PN353B_ANCHOR_CG_NEW = (
    "    # [Genesis PN353B — backport of vllm#43747 closing vllm#40807]\n"
    "    # Downgrade from UNIFORM_BATCH to UNIFORM_SINGLE_TOKEN_DECODE so\n"
    "    # spec-decode K+1 verify batches don't get full CUDA-graph capture\n"
    "    # (which would hit _prefill_attention continuation branch and crash\n"
    "    # on its .tolist() GPU→CPU sync). Cost: ~5-8 %% TPS on Ampere small\n"
    "    # batch. Without it: engine init crashes on TQ + MTP + chunked\n"
    "    # prefill (our PROD config).\n"
    "    _cudagraph_support: ClassVar[AttentionCGSupport] = (\n"
    "        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE\n"
    "    )\n"
)


# ─────────────────────────────────────────────────────────────────────
# ANCHOR #2 — build_for_cudagraph_capture: always populate CPU copies.
# vllm pin dev259+g303916e93 current shape:
#
#     def build_for_cudagraph_capture(
#         self, common_attn_metadata: CommonAttentionMetadata
#     ) -> TurboQuantMetadata:
#         attn_metadata = self.build(0, common_attn_metadata)
#         # Set seq_lens to 1 so CUDA graph capture is fast
#         # (real seq_lens are filled at replay time).
#         attn_metadata.seq_lens.fill_(1)
#         return attn_metadata
# ─────────────────────────────────────────────────────────────────────

PN353B_ANCHOR_CGBUILD_OLD = (
    "        # Set seq_lens to 1 so CUDA graph capture is fast\n"
    "        # (real seq_lens are filled at replay time).\n"
    "        attn_metadata.seq_lens.fill_(1)\n"
    "        return attn_metadata\n"
)

PN353B_ANCHOR_CGBUILD_NEW = (
    "        # Set seq_lens to 1 so CUDA graph capture is fast\n"
    "        # (real seq_lens are filled at replay time).\n"
    "        attn_metadata.seq_lens.fill_(1)\n"
    "\n"
    "        # [Genesis PN353B — backport of vllm#43747]\n"
    "        # Always populate CPU-resident copies so the continuation-prefill\n"
    "        # path never falls back to .tolist() on GPU tensors during CUDA\n"
    "        # graph capture (which is illegal and crashes engine init).\n"
    "        # Sibling protection to the _cudagraph_support downgrade above.\n"
    "        if attn_metadata.seq_lens_cpu is None:\n"
    "            import torch as _genesis_pn353b_torch\n"
    "            attn_metadata.seq_lens_cpu = _genesis_pn353b_torch.ones(\n"
    "                attn_metadata.seq_lens.shape[0],\n"
    "                dtype=attn_metadata.seq_lens.dtype,\n"
    "                device=\"cpu\",\n"
    "            )\n"
    "        if attn_metadata.query_start_loc_cpu is None:\n"
    "            attn_metadata.query_start_loc_cpu = (\n"
    "                attn_metadata.query_start_loc.to(\"cpu\")\n"
    "            )\n"
    "        return attn_metadata\n"
)


# ─────────────────────────────────────────────────────────────────────
# ANCHOR #3 — _prefill_attention continuation branch capture-guard.
# Insert early-return BEFORE the `Hk = key.shape[1]` line at the start
# of the continuation branch.
#
# NOTE: This anchor overlaps with P78 Site B's same region. P78 Site B
# inserts a FLASH_ATTN_VARLEN early-return guarded by GENESIS_ENABLE_P78
# env. If P78 Site B has been applied, the original `Hk = key.shape[1]`
# is preceded by a long block of P78 code, so our shorter anchor still
# matches (anchor is the LAST 4 lines of the continuation-branch
# header, which P78 preserves intact for forwards-compatibility).
# If P78's flash_attn early-return fires, we never reach our zero-return
# anyway — both are no-ops at inference time (only fire during capture).
# ─────────────────────────────────────────────────────────────────────

PN353B_ANCHOR_PREFILL_OLD = (
    "        # For continuation chunks (seq_len > q_len), we must attend to\n"
    "        # previously cached K/V from the TQ cache, not just the current\n"
    "        # chunk's raw K/V.\n"
    "        Hk = key.shape[1]\n"
)

PN353B_ANCHOR_PREFILL_NEW = (
    "        # For continuation chunks (seq_len > q_len), we must attend to\n"
    "        # previously cached K/V from the TQ cache, not just the current\n"
    "        # chunk's raw K/V.\n"
    "        # [Genesis PN353B — backport of vllm#43747 defense-in-depth]\n"
    "        # If we somehow reach this continuation path during CUDA-graph\n"
    "        # capture (e.g. spec-decode warmup shapes if anchor #1 downgrade\n"
    "        # didn't take effect), return zeros instead of crashing on\n"
    "        # .tolist() GPU→CPU sync. Capture-time output is unused under\n"
    "        # PIECEWISE, so this preserves memory-profiling accuracy.\n"
    "        import torch as _genesis_pn353b_torch\n"
    "        if _genesis_pn353b_torch.cuda.is_current_stream_capturing():\n"
    "            return _genesis_pn353b_torch.zeros(\n"
    "                N, Hq, D, device=query.device, dtype=query.dtype\n"
    "            )\n"
    "        Hk = key.shape[1]\n"
)


def _make_patcher() -> TextPatcher | None:
    target = resolve_vllm_file("v1/attention/backends/turboquant_attn.py")
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN353B turboquant_attn.py — prefill CUDA-graph capture safety "
            "(backport vllm#43747)"
        ),
        target_file=str(target),
        marker=GENESIS_PN353B_MARKER,
        sub_patches=[
            # Anchor #1: hard-required ClassVar downgrade. Without this
            # the downstream defense-in-depth doesn't have the upstream
            # PIECEWISE switch and crash will still hit on some shapes.
            TextPatch(
                name="pn353b_cudagraph_support_downgrade",
                anchor=PN353B_ANCHOR_CG_OLD,
                replacement=PN353B_ANCHOR_CG_NEW,
                required=True,
            ),
            # Anchor #2: CPU-copies safety net. Required=False because
            # the .build() in newer pins already populates them; this
            # is a belt-and-suspenders for build_for_cudagraph_capture.
            TextPatch(
                name="pn353b_build_for_cg_capture_cpu_copies",
                anchor=PN353B_ANCHOR_CGBUILD_OLD,
                replacement=PN353B_ANCHOR_CGBUILD_NEW,
                required=False,
            ),
            # Anchor #3: continuation early-return defense. Required=False
            # — overlaps semantically with P78 Site B but P78 is opt-in
            # and uses a different code path. Both can fire safely.
            TextPatch(
                name="pn353b_prefill_continuation_capture_guard",
                anchor=PN353B_ANCHOR_PREFILL_OLD,
                replacement=PN353B_ANCHOR_PREFILL_NEW,
                required=False,
            ),
        ],
        upstream_drift_markers=[
            "[Genesis PN353B",
            # Self-collision lint (triage plan §6 2026-06-11): former
            # entries "UNIFORM_SINGLE_TOKEN_DECODE" /
            # "is_current_stream_capturing" are vllm#43747 strings baked
            # verbatim by our own backport replacement — they cannot
            # distinguish a real upstream merge from our residue (false
            # "upstream_merged" skip, PN369 class). Real-merge detection
            # via required-anchor mismatch (Layer 5) + preflight deep-diff.
        ],
    )


def apply() -> tuple[str, str]:
    """Apply PN353B — TQ prefill CUDA-graph capture safety."""
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN353B")
    log_decision("PN353B", decision, reason)
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
        log.info("[PN353B] marker present — skip (idempotent)")
        return "applied", "idempotent (marker present)"

    # Drift detection — skip if upstream/P65 already shipped equivalent.
    # NOTE: anchor #3 references `is_current_stream_capturing` which is
    # also present in P67 (line 1018 in container) and P78 Site B. To
    # avoid false-positive drift, only treat `UNIFORM_SINGLE_TOKEN_DECODE`
    # as a hard skip marker (since it would mean the downgrade landed
    # via upstream merge or P65 v2).
    if (
        "UNIFORM_SINGLE_TOKEN_DECODE" in content
        and "_cudagraph_support" in content
        and "UNIFORM_BATCH" not in (
            # Has the stock declaration been replaced?
            content.split("class TurboQuantMetadataBuilder")[1]
            if "class TurboQuantMetadataBuilder" in content else ""
        )
    ):
        return (
            "skipped",
            "TurboQuantMetadataBuilder _cudagraph_support already downgraded "
            "(upstream merge or P65 v2 applied). PN353B redundant.",
        )

    result, failure = patcher.apply()
    if result == TextPatchResult.SKIPPED:
        _r = failure.reason if failure else "anchor drift / not eligible"
        _d = f" ({failure.detail})" if (failure and failure.detail) else ""
        return "skipped", f"{patcher.patch_name}: {_r}{_d}"
    if result == TextPatchResult.FAILED:
        return "failed", (
            f"{patcher.patch_name}: "
            f"{failure.reason if failure else 'unknown'} "
            f"({failure.detail if failure else ''})"
        )

    return (
        "applied",
        "PN353B applied: TurboQuantMetadataBuilder._cudagraph_support "
        "downgraded to UNIFORM_SINGLE_TOKEN_DECODE; "
        "build_for_cudagraph_capture always populates CPU-resident "
        "seq_lens_cpu/query_start_loc_cpu; _prefill_attention continuation "
        "branch adds capture-stream early-return. Closes vllm#40807 "
        "(engine-init crash on TQ + MTP + chunked-prefill — our PROD config). "
        "Conflicts with P65 (default OFF); supersedes if both present."
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

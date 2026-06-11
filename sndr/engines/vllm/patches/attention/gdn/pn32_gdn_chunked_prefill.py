# SPDX-License-Identifier: Apache-2.0
"""Wiring for Patch N32 v3 — GDN _forward_core chunked prefill (Cliff 2 fix).

================================================================
v3 RE-ANCHOR + PN79 COMPOSITION (2026-06-11)
================================================================

Pin `0.22.1rc1.dev259+g303916e93` moved the target twice:

1. Upstream #41126 split the monolithic
   `model_executor/layers/mamba/gdn_linear_attn.py` into per-model
   files under `model_executor/layers/mamba/gdn/`; the Qwen prefill
   branch now lives in `gdn/qwen_gdn_linear_attn.py`
   (`QwenGatedDeltaNetAttention._forward_core`).
2. Upstream #44700 reworked mixed-batch handling: decodes are peeled
   off the non-spec batch and the prefill chunk kernel consumes
   builder-precomputed `attn_metadata.prefill_query_start_loc` /
   `prefill_state_indices` / `prefill_has_initial_state` instead of
   the v2-era `non_spec_query_start_loc` /
   `non_spec_state_indices_tensor` / `has_initial_state` locals.

The v3 anchor is the new `# 2.3` prefill block, pristine
`gdn/qwen_gdn_linear_attn.py` lines 1503-1532 (quoted verbatim in
`PN32_BLOCK_HEADER` + `_PN32_PRECOMPUTED_LINES` + `PN32_PRISTINE_TAIL`
below), ending at the `# Init cache` persist line.

Replacement updates vs v2:

- single-sequence detection via
  `attn_metadata.prefill_query_start_loc.shape[0] == 2` (one [0, T]
  entry; the builder peels decodes off, so this is the prefill-only
  cu_seqlens);
- final-state persist into `ssm_state[prefill_state_indices]`;
- NEW cutedsl bypass guard: `ChunkGatedDeltaRule.forward_cutedsl`
  asserts `chunk_indices is not None` / `chunk_offsets is not None`
  (pristine `qwen_gdn_linear_attn.py` lines 398-400) while the
  chunked path passes `None` for both (Triton/FLA recomputes them
  from the chunk-local cu_seqlens; flashinfer ignores them). Chunking
  therefore engages only when `self.gdn_prefill_backend != 'cutedsl'`
  (attribute set at pristine line 555, already consulted the same way
  at line 1143).

================================================================
COMPOSITION WITH PN79 (MANDATORY — PN79 sub-patch 3C is PROD-applied)
================================================================

PN79 sub-patch 3C (`pn79_inplace_ssm_state.py`,
`ANCHOR_3C_PREFILL_INPLACE_OLD`) anchors on the IDENTICAL pristine
block (lines 1509-1532). PN32 v3 composes via the action plan's
option (a): apply-order dependency + a post-PN79 anchor variant.

- PN32 carries TWO anchor variants with required-at-least-one
  semantics (both `required=False`; the TextPatcher kernel returns
  SKIPPED `no_applicable_sub_patches` when every sub-patch misses):

  * pristine-shaped variant (`PN32_ANCHOR`) — matches an untouched
    pristine file (PN79 disabled);
  * post-PN79-shaped variant (`PN32_ANCHOR_POST_PN79`) — matches the
    file AFTER PN79 3C applied. Assembled from PN79's own
    `ANCHOR_3C_PREFILL_INPLACE_NEW` constant (chain convention, same
    as PN365 importing PN50's ANCHOR_NEW) so the two modules cannot
    silently diverge.

- APPLY-ORDER DEPENDENCY: PN79 must apply BEFORE PN32. The boot
  dispatch sequence already guarantees this (PN79 at
  `sndr/apply/_per_patch_dispatch.py` ~line 2782, PN32 at ~line 4270;
  live boot log lines 113 vs 219). The REVERSE order is forbidden: a
  PN32-patched file no longer contains PN79's required 3C anchor, so
  PN79's whole multi-file transaction would abort. Registry owner:
  see the REGISTRY DECLARATIONS section below.

- Semantics when both are applied: the non-chunked path keeps PN79's
  backend-gated in-place state logic verbatim (re-indented under the
  bypass `else:`). The chunked path (opt-in, single-seq long-prefill
  only) uses upstream gather/scatter semantics — chunk chaining needs
  the gathered per-call state, and the per-sequence gather for a
  single sequence is a (1, H, V, K) copy, negligible next to the
  buffers this patch exists to shrink.

================================================================
REGISTRY DECLARATIONS (registry.py is owned by another agent —
declared here, to be mirrored into the PN32 entry)
================================================================

- `composes_with`: add "PN79" (post-PN79 anchor variant; PN79 first)
  and keep the recommended "P103" pairing documented in `credit`.
- `requires_patches`: stays `[]` — PN32 does NOT require PN79 (the
  pristine variant covers PN79-disabled deployments); the dependency
  is ORDER-ONLY when both are enabled.
- `conflicts_with`: keep ["P28", "PN108"] unchanged.
- Entry comment: "PN32 must stay AFTER PN79 in the apply_all dispatch
  order — PN32 v3 carries a post-PN79 anchor variant; PN79's required
  3C anchor dies on a PN32-patched file."

================================================================
v7.69 v2 DESIGN (still the core of the chunked path)
================================================================

The v7.65 PN32 v1 chunked at the WRONG level. It patched the outer
`GatedDeltaNetAttention.forward_cuda` and sliced inputs before calling
`torch.ops.vllm.gdn_attention_core`; the inner FLA call still received
full-prompt cu_seqlens/chunk metadata, so the kernel allocated
full-size buffers regardless. Cross-rig evidence: noonghunna
2026-05-02 reported v1 OOMing EARLIER (30K) than baseline (50-60K) —
club-3090#19 finding 3.

v2 (and v3) instead wrap the prefill branch's
`self.chunk_gated_delta_rule(...)` call with a chunk-aware loop:

1. Detect single-sequence prefill (cu_seqlens shape [2]).
   Multi-sequence prefill bypasses to original (correct chunking
   across sequence boundaries needs inner state-cache management not
   exposed at this layer).
2. For T > THRESHOLD (default 16384), split into chunks of CHUNK_SIZE
   (default 8192). Per chunk: slice query/key/value/g/beta along T
   (dim=1; shape is `(1, T, H, D)` after unsqueeze in the caller),
   build chunk-local `cu_seqlens=[0, chunk_len]`, pass
   `chunk_indices=None, chunk_offsets=None`, thread `initial_state`
   via the prior chunk's `last_recurrent_state`,
   `output_final_state=True` always.
3. Concatenate chunk outputs along dim=1.
4. Persist the final chunk's `last_recurrent_state` into
   `ssm_state[prefill_state_indices]` exactly as the original.

================================================================
COMPOSITION WITH P103 (RECOMMENDED — both default OFF)
================================================================

PN32 chunks at the OUTER FLA boundary (the call to
`chunk_gated_delta_rule`). P103 chunks INSIDE FLA's
`chunk_gated_delta_rule_fwd` (the kernel orchestrator), splitting the
inner `h = k.new_empty(B, NT, H, V, K)` allocation. The two are
COMPLEMENTARY, not redundant:

- **PN32 alone**: reduces the OUTPUT buffer size per call. Reduces
  transient peak inside one layer.
- **P103 alone**: reduces the INNER `h` tensor inside the FLA
  orchestrator.
- **PN32 + P103**: PN32 calls FLA with chunk-sized inputs, FLA's
  wrapped orchestrator further splits if the chunk is still >
  P103's MAX_T. Best memory profile.

Recommended for single-24GB-GPU users hitting Cliff 2:

    GENESIS_ENABLE_P103=1                              # required
    GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1          # recommended
    GENESIS_PN32_GDN_CHUNK_SIZE=8192                   # default
    GENESIS_PN32_GDN_CHUNK_THRESHOLD=16384             # default
    GENESIS_FLA_FWD_H_MAX_T=16384                      # P103 default

================================================================
DEPENDENCIES
================================================================

- **Hard requirement**: NONE (PN32 v3 functions standalone — chunks
  the outer call independently).
- **Ordering**: PN79 (when enabled) must apply BEFORE PN32 — see the
  PN79 composition section above.
- **Strong recommendation**: enable P103 simultaneously for very long
  contexts (>200K).
- **Conflict**: P28 (legacy persistent buffer pool) and PN108 (GDN
  fused_recurrent prefill backend switch) — both modify overlapping
  code paths. Single-sequence prefill is a precondition of the
  chunked path itself (multi-seq bypasses).

================================================================
THRESHOLD SEMANTICS
================================================================

The chunked path fires only when ALL hold:

  GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL truthy (re-checked at
      runtime, so a patched file behaves vanilla when the env is off)
  AND single-sequence prefill (prefill cu_seqlens shape == [2])
  AND T > GENESIS_PN32_GDN_CHUNK_THRESHOLD (default 16384)
  AND GENESIS_PN32_GDN_CHUNK_SIZE > 0 (default 8192; garbage-proof)
  AND self.gdn_prefill_backend != 'cutedsl'

Multi-sequence prefill (continuous batching of N short prompts
totaling > THRESHOLD) bypasses to original — correct chunking across
sequence boundaries requires inner state-cache management not exposed
at this layer.

================================================================
SAFETY MODEL
================================================================

- Default OFF (opt-in via `GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1`)
- Pure text-patch on `_forward_core` (idempotent via marker)
- Single-sequence prefill only — multi-sequence bypasses (no risk of
  wrong cross-sequence state mixing)
- Drift-aware: if upstream rewrites the `# 2.3` block again, neither
  variant matches → SKIPPED `no_applicable_sub_patches`, source stays
  vanilla
- Numerical correctness: chained `last_recurrent_state` propagation
  preserves recurrent state across chunks (same mechanism FLA uses
  internally for chunk_indices/chunk_offsets)

================================================================
HISTORY
================================================================

- v7.65 v1 (2026-05-01): patched `forward_cuda` outer — wrong level,
  didn't propagate cu_seqlens to the inner FLA call (metadata
  mismatch); empirically OOM'd EARLIER on club-3090 cross-rig.
- v7.69 v2 (2026-05-02): rewritten to chunk `_forward_core` directly,
  with chunk-local cu_seqlens and threaded initial_state.
- v3 (2026-06-11): re-anchored on the post-#41126/#44700 `# 2.3`
  block; prefill_* metadata fields; cutedsl bypass guard; dual anchor
  variants for PN79 composition.

Author: Sandermage(Sander) Barzov Aleksandr, Ukraine, Odessa.
Reporter: noonghunna (CLIFF2_INVESTIGATION_20260430.md +
                       club-3090#19 cross-rig finding 3, 2026-05-02).
"""
from __future__ import annotations

import logging

from sndr.engines.vllm.detection.guards import resolve_vllm_file, vllm_install_root
from sndr.engines.vllm.patches.attention.gdn.pn79_inplace_ssm_state import (
    ANCHOR_3C_PREFILL_INPLACE_NEW as _PN79_3C_NEW,
)
from sndr.kernel import (
    TextPatch,
    TextPatcher,
    result_to_wiring_status,
)

log = logging.getLogger("genesis.wiring.pN32_gdn_chunked_prefill")

GENESIS_PN32_MARKER = (
    "Genesis PN32 v3 GDN _forward_core chunked-prefill (Cliff 2 fix)"
)

# Drift markers must stay disjoint from everything this patch emits
# (self-collision lint, action plan section 6). The single marker below
# detects stale v2-patched residue (e.g. a half-migrated tree) so v3
# never stacks on top of a v2 body; it is NOT emitted by v3.
PN32_UPSTREAM_DRIFT_MARKERS = [
    "[Genesis PN32 v2 v7.69 chunked-prefill]",
]


def _indent_block(block: str, pad: str = "    ") -> str:
    """Re-indent a replacement block by `pad` (blank lines untouched)."""
    return "".join(
        (pad + line) if line.strip() else line
        for line in block.splitlines(keepends=True)
    )


# ─── Anchor building blocks — pristine gdn/qwen_gdn_linear_attn.py ──
# Pin 0.22.1rc1.dev259+g303916e93. All three constants quote pristine
# lines 1503-1532 verbatim (verified count==1 against both the live
# pristine tree and tests/legacy/pristine_fixtures/qwen_gdn_linear_attn.py,
# md5 194c57a13156fe2f1105064a483de989).

# Pristine lines 1503-1508 — the `# 2.3` block header. PN79 3C does NOT
# include these lines in its anchor; prepending them keeps both PN32
# variants unique AND mutually exclusive with each other.
PN32_BLOCK_HEADER = (
    "        # 2.3: Process the remaining part (prefill chunk, or non-spec decode-only)\n"
    "        if attn_metadata.num_prefills > 0:\n"
    "            # State indices, initial-state mask and cu_seqlens for the chunk\n"
    "            # kernel are precomputed by the metadata builder (the prefill tail\n"
    "            # when decodes are peeled off, else the full non-spec batch), so they\n"
    "            # don't need to be re-derived per layer.\n"
)

# Pristine lines 1509-1512 — builder-precomputed metadata reads +
# asserts. Identical in the pristine block and in PN79's 3C NEW text,
# so both replacements keep them verbatim ahead of the PN32 insertion.
_PN32_PRECOMPUTED_LINES = (
    "            prefill_state_indices = attn_metadata.prefill_state_indices\n"
    "            prefill_has_initial_state = attn_metadata.prefill_has_initial_state\n"
    "            assert prefill_state_indices is not None\n"
    "            assert prefill_has_initial_state is not None\n"
)

# Pristine lines 1513-1532 — gather, FLA call, `# Init cache` persist.
# Reused verbatim (re-indented) as the pristine variant's bypass path.
PN32_PRISTINE_TAIL = (
    "            initial_state = ssm_state[prefill_state_indices]\n"
    "            initial_state[~prefill_has_initial_state, ...] = 0\n"
    "            (\n"
    "                core_attn_out_non_spec,\n"
    "                last_recurrent_state,\n"
    "            ) = self.chunk_gated_delta_rule(\n"
    "                q=query_non_spec,\n"
    "                k=key_non_spec,\n"
    "                v=value_non_spec,\n"
    "                g=g_non_spec,\n"
    "                beta=beta_non_spec,\n"
    "                initial_state=initial_state,\n"
    "                output_final_state=True,\n"
    "                cu_seqlens=attn_metadata.prefill_query_start_loc,\n"
    "                chunk_indices=attn_metadata.chunk_indices,\n"
    "                chunk_offsets=attn_metadata.chunk_offsets,\n"
    "                use_qk_l2norm_in_kernel=False,\n"
    "            )\n"
    "            # Init cache\n"
    "            ssm_state[prefill_state_indices] = last_recurrent_state.to(ssm_state.dtype)\n"
)

# ─── PN32 insertion: chunk decision + chunked path (shared) ─────────

_PN32_DECISION_BLOCK = (
    "            # [Genesis PN32 v3 chunked-prefill] Cliff 2 fix: chunk the\n"
    "            # FLA call so core_attn_out_non_spec is allocated per-chunk\n"
    "            # (not full-prompt). Composes with P103 (P103 chunks INSIDE\n"
    "            # chunk_gated_delta_rule_fwd's h tensor). Single-sequence\n"
    "            # prefill only: prefill cu_seqlens shape [2] == one [0, T]\n"
    "            # entry; multi-sequence bypasses to the original path.\n"
    "            # cutedsl bypass: ChunkGatedDeltaRule.forward_cutedsl asserts\n"
    "            # chunk_indices/chunk_offsets are not None (pristine lines\n"
    "            # 398-400 on pin 0.22.1rc1.dev259+g303916e93) while the\n"
    "            # chunked path passes None for both (Triton/FLA recomputes\n"
    "            # them from chunk-local cu_seqlens; flashinfer ignores them).\n"
    "            import os as _genesis_pn32_os\n"
    "            _genesis_pn32_enabled = _genesis_pn32_os.environ.get(\n"
    "                'GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL', ''\n"
    "            ).strip().lower() in ('1', 'true', 'yes', 'on')\n"
    "            try:\n"
    "                _genesis_pn32_threshold = int(\n"
    "                    _genesis_pn32_os.environ.get(\n"
    "                        'GENESIS_PN32_GDN_CHUNK_THRESHOLD', '16384'\n"
    "                    )\n"
    "                )\n"
    "            except (ValueError, TypeError):\n"
    "                _genesis_pn32_threshold = 16384\n"
    "            try:\n"
    "                _genesis_pn32_chunk_size = int(\n"
    "                    _genesis_pn32_os.environ.get(\n"
    "                        'GENESIS_PN32_GDN_CHUNK_SIZE', '8192'\n"
    "                    )\n"
    "                )\n"
    "            except (ValueError, TypeError):\n"
    "                _genesis_pn32_chunk_size = 8192\n"
    "\n"
    "            # Single-sequence detection: prefill cu_seqlens shape [2] =\n"
    "            # one [0, T] entry. Multi-seq has shape [N+1] for N>1.\n"
    "            _genesis_pn32_T_full = int(query_non_spec.shape[1])\n"
    "            _genesis_pn32_is_single_seq = (\n"
    "                attn_metadata.prefill_query_start_loc is not None\n"
    "                and attn_metadata.prefill_query_start_loc.shape[0] == 2\n"
    "            )\n"
    "            _genesis_pn32_should_chunk = (\n"
    "                _genesis_pn32_enabled\n"
    "                and _genesis_pn32_is_single_seq\n"
    "                and _genesis_pn32_chunk_size > 0\n"
    "                and _genesis_pn32_T_full > _genesis_pn32_threshold\n"
    "                and self.gdn_prefill_backend != 'cutedsl'\n"
    "            )\n"
    "\n"
)

_PN32_CHUNK_BRANCH = (
    "            if _genesis_pn32_should_chunk:\n"
    "                # ─── Chunked path: split the FLA call along T ───\n"
    "                # Upstream gather/scatter semantics: chunk chaining\n"
    "                # threads the gathered per-call state, then persists\n"
    "                # the final state once.\n"
    "                initial_state = ssm_state[prefill_state_indices]\n"
    "                initial_state[~prefill_has_initial_state, ...] = 0\n"
    "                _genesis_pn32_chunks = []\n"
    "                _genesis_pn32_state = initial_state\n"
    "                _genesis_pn32_last_state = None\n"
    "                for _genesis_pn32_start in range(\n"
    "                    0, _genesis_pn32_T_full, _genesis_pn32_chunk_size\n"
    "                ):\n"
    "                    _genesis_pn32_end = min(\n"
    "                        _genesis_pn32_start + _genesis_pn32_chunk_size,\n"
    "                        _genesis_pn32_T_full,\n"
    "                    )\n"
    "                    _genesis_pn32_chunk_len = (\n"
    "                        _genesis_pn32_end - _genesis_pn32_start\n"
    "                    )\n"
    "                    # Slice along T dim (dim=1) — shape (1, T, H, D)\n"
    "                    _genesis_pn32_q_chunk = query_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    _genesis_pn32_k_chunk = key_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    _genesis_pn32_v_chunk = value_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    _genesis_pn32_g_chunk = g_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    _genesis_pn32_beta_chunk = beta_non_spec[\n"
    "                        :, _genesis_pn32_start:_genesis_pn32_end\n"
    "                    ]\n"
    "                    # Chunk-local cu_seqlens — single-seq, length = chunk_len\n"
    "                    _genesis_pn32_chunk_cu_seqlens = torch.tensor(\n"
    "                        [0, _genesis_pn32_chunk_len],\n"
    "                        device=query_non_spec.device,\n"
    "                        dtype=attn_metadata.prefill_query_start_loc.dtype,\n"
    "                    )\n"
    "                    # FLA call on chunk; output_final_state=True for chaining\n"
    "                    (\n"
    "                        _genesis_pn32_o_chunk,\n"
    "                        _genesis_pn32_last_state,\n"
    "                    ) = self.chunk_gated_delta_rule(\n"
    "                        q=_genesis_pn32_q_chunk,\n"
    "                        k=_genesis_pn32_k_chunk,\n"
    "                        v=_genesis_pn32_v_chunk,\n"
    "                        g=_genesis_pn32_g_chunk,\n"
    "                        beta=_genesis_pn32_beta_chunk,\n"
    "                        initial_state=_genesis_pn32_state,\n"
    "                        output_final_state=True,\n"
    "                        cu_seqlens=_genesis_pn32_chunk_cu_seqlens,\n"
    "                        chunk_indices=None,\n"
    "                        chunk_offsets=None,\n"
    "                        use_qk_l2norm_in_kernel=False,\n"
    "                    )\n"
    "                    _genesis_pn32_chunks.append(_genesis_pn32_o_chunk)\n"
    "                    # Thread state for next chunk\n"
    "                    _genesis_pn32_state = _genesis_pn32_last_state\n"
    "                    # Free chunk references (allocator can reuse)\n"
    "                    del (\n"
    "                        _genesis_pn32_q_chunk,\n"
    "                        _genesis_pn32_k_chunk,\n"
    "                        _genesis_pn32_v_chunk,\n"
    "                        _genesis_pn32_g_chunk,\n"
    "                        _genesis_pn32_beta_chunk,\n"
    "                    )\n"
    "\n"
    "                core_attn_out_non_spec = torch.cat(\n"
    "                    _genesis_pn32_chunks, dim=1\n"
    "                )\n"
    "                last_recurrent_state = _genesis_pn32_last_state\n"
    "                del _genesis_pn32_chunks\n"
    "                # Init cache (chunked path: explicit scatter persist)\n"
    "                ssm_state[prefill_state_indices] = last_recurrent_state.to(\n"
    "                    ssm_state.dtype\n"
    "                )\n"
)

# ─── Variant 1: pristine-shaped (PN79 disabled) ─────────────────────

PN32_ANCHOR = PN32_BLOCK_HEADER + _PN32_PRECOMPUTED_LINES + PN32_PRISTINE_TAIL

PN32_REPLACEMENT = (
    PN32_BLOCK_HEADER
    + _PN32_PRECOMPUTED_LINES
    + _PN32_DECISION_BLOCK
    + _PN32_CHUNK_BRANCH
    + "            else:\n"
    + _indent_block(PN32_PRISTINE_TAIL)
)

# ─── Variant 2: post-PN79-shaped (PN79 3C applied first) ────────────
#
# The anchor is the `# 2.3` header + PN79's 3C NEW text exactly as PN79
# writes it. The replacement keeps PN79's backend-gated in-place logic
# as the bypass path: split PN79's NEW text after the (shared)
# precomputed/assert lines, re-indent the tail under `else:`.

_PN79_TAIL_SPLIT_POINT = "            assert prefill_has_initial_state is not None\n"

PN32_ANCHOR_POST_PN79 = PN32_BLOCK_HEADER + _PN79_3C_NEW

if _PN79_3C_NEW.count(_PN79_TAIL_SPLIT_POINT) == 1:
    _PN79_3C_TAIL = _PN79_3C_NEW.split(_PN79_TAIL_SPLIT_POINT, 1)[1]
    PN32_REPLACEMENT_POST_PN79 = (
        PN32_BLOCK_HEADER
        + _PN32_PRECOMPUTED_LINES
        + _PN32_DECISION_BLOCK
        + _PN32_CHUNK_BRANCH
        + "            else:\n"
        + _indent_block(_PN79_3C_TAIL)
    )
else:
    # PN79's 3C NEW text changed shape — the post-PN79 replacement can
    # no longer be assembled safely. Disable the variant with a
    # never-matching sentinel (the pristine variant still works for
    # PN79-disabled deployments) and fail loudly in the unit test
    # (test_post_pn79_anchor_built_from_pn79_constant).
    log.warning(
        "[PN32 v3] PN79 ANCHOR_3C_PREFILL_INPLACE_NEW no longer contains "
        "the expected precomputed/assert prefix exactly once — disabling "
        "the post-PN79 anchor variant. Re-verify PN32/PN79 composition."
    )
    PN32_ANCHOR_POST_PN79 = (
        "# [Genesis PN32 v3 sentinel — post-PN79 variant disabled, "
        "PN79 3C NEW drifted]\n"
    )
    PN32_REPLACEMENT_POST_PN79 = PN32_ANCHOR_POST_PN79


def build_sub_patches() -> list[TextPatch]:
    """The two anchor variants, required-at-least-one semantics.

    Both `required=False`: the kernel soft-skips the variant whose
    anchor is absent and returns SKIPPED `no_applicable_sub_patches`
    only when BOTH miss. The variants are mutually exclusive by
    construction (the post-PN79 anchor contains PN79's `[Genesis PN79`
    comments, absent from pristine; PN79's apply destroys the pristine
    block) — verified in
    tests/unit/integrations/attention/gdn/test_pn32_gdn_chunked_prefill_v3.py.
    """
    return [
        TextPatch(
            name="pN32_v3_forward_core_chunked_prefill_pristine",
            anchor=PN32_ANCHOR,
            replacement=PN32_REPLACEMENT,
            required=False,
        ),
        TextPatch(
            name="pN32_v3_forward_core_chunked_prefill_post_pn79",
            anchor=PN32_ANCHOR_POST_PN79,
            replacement=PN32_REPLACEMENT_POST_PN79,
            required=False,
        ),
    ]


def _make_patcher() -> TextPatcher | None:
    # v3 (2026-06-11): target ONLY the per-model file introduced by the
    # upstream gdn/ split (#41126). No fallback to the retired
    # monolithic `mamba/gdn_linear_attn.py` — the v3 anchors are
    # derived from the new file and are structurally absent from the
    # monolith (same call as PN79's K.2 re-anchor).
    target = resolve_vllm_file(
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    if target is None:
        return None
    return TextPatcher(
        patch_name=(
            "PN32 v3 model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py — "
            "_forward_core chunked-prefill (Cliff 2 fix)"
        ),
        target_file=str(target),
        marker=GENESIS_PN32_MARKER,
        sub_patches=build_sub_patches(),
        upstream_drift_markers=list(PN32_UPSTREAM_DRIFT_MARKERS),
    )


def apply() -> tuple[str, str]:
    """Apply PN32 v3 — _forward_core chunked-prefill (text-patch).

    v3 (2026-06-11) re-anchors v2 on the post-#41126/#44700 `# 2.3`
    prefill block and composes with PN79 sub-patch 3C via dual anchor
    variants (apply-order: PN79 first). v2 itself superseded v1, which
    chunked at the wrong level (forward_cuda outer) and OOM'd EARLIER
    than baseline on club-3090 cross-rig.
    """
    from sndr.dispatcher import log_decision, should_apply

    decision, reason = should_apply("PN32")
    log_decision("PN32", decision, reason)
    if not decision:
        return "skipped", reason

    if vllm_install_root() is None:
        return "skipped", "vllm install root not discoverable"

    patcher = _make_patcher()
    if patcher is None:
        return "skipped", "gdn/qwen_gdn_linear_attn.py not resolvable"

    result, failure = patcher.apply()
    return result_to_wiring_status(
        result, failure,
        applied_message=(
            "PN32 v3 applied: GDN _forward_core prefill branch now uses "
            "single-seq chunked path for long prompts (>16K tokens "
            "default). Chunks query/key/value/g/beta along T, builds "
            "chunk-local cu_seqlens, threads initial_state via "
            "last_recurrent_state, persists into "
            "ssm_state[prefill_state_indices]. Multi-seq and the cutedsl "
            "prefill backend bypass to original. Default OFF — opt-in "
            "via GENESIS_ENABLE_PN32_GDN_CHUNKED_PREFILL=1. Composes "
            "with PN79 (post-PN79 anchor variant; PN79 applies first) "
            "and with P103 for full Cliff 2 coverage on single-24GB-GPU."
        ),
        patch_name="PN32 v3 GDN _forward_core chunked-prefill",
    )

# R1 (TQ decode WorkspaceManager memo) — TESTED-NEGATIVE on dev491

**Date**: 2026-06-15
**Model**: Qwen3.6-27B-int4-AutoRound, TQ k8v4, MTP K=3, 2× A5000 TP=2, dev491 (1033ffac2)
**Question**: recover the 27B single-stream decode (operator target 146+ TPS; current ~125).

## Hypothesis (from the cross-model perf workflow, ranked R1, "high confidence")

On 2026-06-08 (commit 5c20b51f) P98's `p98_decode_workspace_revert` sub-patch was
retired because PN118 (backport of vllm#42551) rewrote `_decode_attention` to route
decode scratch through `WorkspaceManager.try_get_simultaneous(...)`. P99 only memoizes
`get_simultaneous`, **not** `try_get_simultaneous`, so the decode hot path pays the full
per-token Python indirection (`_compute_bytes×3 + round_up×3 + accumulate + lock-check +
3× view/reshape`) × TQ full-attn layers × (1+MTP-K). The workflow estimated this at
**~17%** based on P98's original docstring measurement (200→167 TPS), and proposed
memoizing `try_get_simultaneous`.

## What I built

Folded a success-path memo into PN118's injected `try_get_simultaneous` (boot-order-robust
vs a separate P99 sub-patch anchoring on PN118-injected text), keyed on
`(shapes, ubatch, workspace.data_ptr)`, caching **only** the success path (never the
locked-undersized `None`, preserving PN118's keep-serving contract), gated by
`GENESIS_P99_TRY_MEMO` (default ON; `0` for A/B). A standalone harness verified all four
invariants: miss→hit identical, None-never-cached, realloc-invalidates, env-off transparent.

## Canonical A/B (genesis_chat_matrix_bench, n=3, same image, env toggle)

| variant       | OFF (memo off) | ON (memo on) | Δ wall TPS |
|---------------|---------------:|-------------:|-----------:|
| thinking_off  | **126.5**      | 124.9        | −1.6       |
| thinking_on   | 122.7          | 123.6        | +0.9       |
| code_gen      | 113.7          | 108.0        | −5.7       |
| multi_turn    | 116.1          | 116.6        | +0.5       |
| long_gen      | 93.3           | 95.3         | +2.0       |

All within bench noise (n=3, CV ~5–8%); thinking_off — the canonical single-stream decode
variant — is even marginally *worse* ON. The boot log confirmed the memo applied and (by
construction) hits the cache in steady state. **No recovery toward 146.**

## Verdict: REFUTED

The per-token TQ-workspace Python cost the memo eliminates is **negligible** vs the CUDA
kernel time on this workload, and the memo's own per-call overhead (hasattr + dict.get +
`dbo_current_ubatch_id()` + tuple-key build) roughly cancels the saving. The workflow's
~17% estimate came from a **pre-#40941** measurement of the heavier `get_simultaneous`
indirection; dev491's `try_get_simultaneous` is much lighter, so the figure does not
translate. This mirrors PN396 (num_warps) and PN352B (moe_sum) — static mechanism analysis
necessary but not sufficient; the bench gate is decisive.

**Action**: reverted the memo from PN118 (no neutral dead weight in a PROD patch). Kept the
P98 boot-log telemetry fix (R3) — the old "+15-25% decode recovery" string was a stale
pre-#40941 claim regardless of this result. Re-test candidate only if a higher-concurrency
(max_num_seqs=8) workload ever shows the workspace resolution count dominating.

## Where the real decode time goes (next: R4)

The workflow's secondary finding is now the leading hypothesis: under MTP K=3,
`gdn_attn.py` reclassifies pure 1-token decode rows to prefill (`num_spec_decodes>0 →
num_decodes=0`), so decode runs the slow `fused_sigmoid_gating_delta_rule_update` spec
recurrent kernel on 30/41 GDN layers, and the new upstream packed-recurrent-decode fast
path (PR#36596, gated on `spec_sequence_masks is None`) is **unreachable**. R4 measures the
MTP-off ceiling to test whether that is the real 146→125 lever. NOTE: skill bench reference
lists the 27B single-stream canonical at **120 TPS** — current ~125 is at/above that, so the
146 may be a variant-peak or an older-pin number; R4 (and, if needed, a dev259 pin A/B)
settles it.

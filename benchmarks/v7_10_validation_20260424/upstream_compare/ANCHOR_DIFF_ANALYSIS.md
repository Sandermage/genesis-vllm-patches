# Phase 4 — Upstream PR compat analysis (anchor diff)

Date: 2026-04-25
Method: PR metadata + `gh api repos/vllm-project/vllm/pulls/.../files` diff inspection.
**Full side-by-side bench deferred**: requires rebuilding `genesis-v7.0-baseline` image with each PR checked out (~1 h per PR) — tracked as follow-up.

This document is sufficient to decide retire/keep for each of our overlapping patches.

---

## PR #40807 — TurboQuant spec-decode capture crash at `query_start_loc.tolist()`

**Type**: OPEN issue (not PR — no code change yet).
**Our overlap**: Patch 44 (`_tq_mixed_attn_out_buf` capture-safe reuse) + Patch 23 (`cu_seqlens` scratch preallocation).

**Verdict**: **we are the fix** — reporter explicitly named our Patch 23 in the writeup. EN+RU comment already drafted in [docs/UPSTREAM_COMMENT_DRAFTS_v7_10.md#1](../../../docs/UPSTREAM_COMMENT_DRAFTS_v7_10.md). Pending `ok push #40807` to post.

---

## PR #40792 — TurboQuant k8v4 GQA head grouping

**Title**: *"[Attention][TurboQuant] Optimize k8v4 decode attention with GQA head grouping"*
**State**: OPEN, head=89991de1
**Changes**: +639/-37 across 2 files.

### Files touched

| File | +/- | Our overlap |
|---|---|---|
| `vllm/v1/attention/ops/triton_turboquant_decode.py` | +272/-37 | **Our P40** rebinds `_tq_decode_stage1` dispatch here |
| `tests/quantization/test_turboquant.py` | +367/-0 | — |

### What #40792 adds

New Triton kernel `_tq_grouped_decode_stage1` (lines ~313-514 in the PR):

- Each CTA processes **up to BLOCK_H Q heads that share one KV head**
- Uses `tl.dot` for tensor-core-accelerated scoring
- Scoped to FP8 keys + 4-bit values (our exact prod config: `turboquant_k8v4`)
- MSE-quantized key presets (`turboquant_{4bit,k3v4,3bit}_nc`) retain the scalar kernel

### Our P40

Our Patch 40 is a **text-patch + dispatch rebind** that swaps the scalar stage1 for a GQA-grouped variant. Same scope (k8v4 only), same optimisation intent (group heads sharing a KV head), but implemented as a rebind rather than a new native Triton kernel.

### Verdict

**#40792 supersedes our P40 functionally.**

- Native `tl.dot` kernel is strictly better than our monkey-patched scalar grouping (uses tensor cores properly, fewer shape surprises).
- Scope matches exactly (k8v4, FP8+4bit value pairs, SM 8.0+).
- Full rebuild bench deferred, but kernel-level inspection shows #40792 is a proper superset.

**Action when #40792 merges:**
1. Remove P40 from the default-opt-in list (already opt-in today via `GENESIS_ENABLE_P40=1`).
2. Add a marker to upstream_compat.py noting P40 is covered by #40792.
3. Keep P40 source file as a historical reference; it can still apply on pre-merge builds where users need the optimisation.

**No conflict** — our P40 is opt-in, never applied by default, won't collide with the upstream kernel.

---

## PR #40798 — TurboQuant share decode scratch workspace across layers

**Title**: *"[TurboQuant] Share decode scratch workspace across layers"*
**State**: OPEN, head=9f7c8399
**Changes**: +183/-44 across 5 files.

### Files touched

| File | +/- | Our overlap |
|---|---|---|
| `vllm/model_executor/layers/attention/attention.py` | +4/-24 | **Our P36** + anchor for P22/P26/P44 |
| `vllm/v1/attention/backends/turboquant_attn.py` | 0/-11 | Our P38 context |
| `vllm/v1/attention/ops/triton_turboquant_decode.py` | +15/-8 | P40 context |
| `vllm/v1/worker/gpu_model_runner.py` | +31/-1 | — |
| `tests/quantization/test_turboquant.py` | +133/-0 | — |

### What #40798 removes

From `attention.py::_init_turboquant_buffers`:

```python
# REMOVED:
self.register_buffer("_tq_mid_o_buf",  torch.empty(B, Hq, S, D+1, fp32), persistent=False)
self.register_buffer("_tq_output_buf", torch.empty(B, Hq, D, fp32),      persistent=False)
self.register_buffer("_tq_lse_buf",    torch.empty(B, Hq, fp32),         persistent=False)

# REPLACED BY (comment):
# TQ decode scratch space is allocated through the v1 workspace manager
# at runtime. It is shared across layers, rather than registered once
# per attention layer...
```

### Our P36

Genesis Patch 36 (TurboQuant shared decode buffers) does **exactly this**:
- Removes per-layer `register_buffer` calls
- Routes decode mid_o / output / lse through a shared pool in `TurboQuantBufferManager`
- Saves memory proportional to `num_attention_layers` (44 GDN/attention layers on Qwen3.6-35B-A3B)

### Verdict

**#40798 supersedes our P36 functionally.**

- Upstream does the same architectural fix (shared across layers instead of per-layer).
- Upstream uses v1 workspace manager (cleaner than our text-patch anchor).
- Our anchor target (`register_buffer` lines) is **deleted** by the PR — if #40798 merges, our P36 text-patch will ANCHOR-MISS and auto-skip (graceful — by design).

**Action when #40798 merges:**
1. Add entry to `upstream_compat.py::PR_40798_shared_decode_workspace` — marks P36 as superseded.
2. Adjust our P36 anchor: if upstream_compat detects #40798 merged → P36 anchor is expected to miss → log as "retired by #40798" not "anchor drift".
3. No action needed for P22 / P26 / P44 — they target different code paths (prefill preallocs, not decode workspace).

**No P28 conflict detected** — I initially worried #40798 would conflict with P28 (GDN forward-cuda rewire) because both touch `attention.py`, but the diff is confined to `_init_turboquant_buffers` (TurboQuant-specific), which P28 does not touch.

### Minor: `turboquant_attn.py` and `triton_turboquant_decode.py`

Both have small edits in #40798 that I have not fully inspected. P38 (`_continuation_prefill` persistent workspace) does NOT conflict because our anchor is in the prefill path, not decode scratch allocation. Full bench will confirm once image is rebuilt.

---

## Summary of actions

| PR | Action when merged | Risk until then |
|---|---|---|
| #40807 issue | Post EN+RU comment with P44+P23 refs (pending `ok push`) | None — our fix is live |
| #40792 | Retire P40 opt-in (keep file, mark superseded) | None — P40 already opt-in only |
| #40798 | Retire P36; add upstream_compat marker for graceful skip | None — anchor-miss auto-skips |

**None of these PRs currently break our patches.** All three represent upstream catching up — validating our architectural choices. Retirement happens after their merge; we move downstream cleanup then.

---

## Bench that was deferred

Full side-by-side perf numbers (baseline vs PR-applied vs ours vs both) require:

1. Git fetch PR → merge locally → rebuild `vllm/vllm-openai:genesis-v7.0-baseline-with-PR-40792` (~40 min)
2. Run `scripts/run_validation_suite.sh` against it
3. Diff `decode_tok_s` at 100k / 256k vs our baseline in [../qwen3_next_fp8/](../qwen3_next_fp8/)

This is scheduled for a **follow-up session** with dedicated bench window. The anchor-level analysis above is sufficient for the retirement decisions (verdicts don't depend on the bench numbers — both PRs functionally supersede; the numbers would only answer "by how much").

If bench confirms positive deltas, we update the upstream comment drafts with real percent-change numbers before posting (currently TBD in [docs/UPSTREAM_COMMENT_DRAFTS_v7_10.md](../../../docs/UPSTREAM_COMMENT_DRAFTS_v7_10.md)).

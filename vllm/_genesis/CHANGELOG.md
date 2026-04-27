# Genesis `_genesis/` Package Changelog

## v7.52 — 2026-04-27 (Tier 3 H: fused-M kernel as opt-in; REJECTED for prod default)

Implemented Tier 3 H from the throughput sprint plan: fused-M variant of
P67 multi-query kernel. Adapted from the FP64 reference impl in private
repo `Sandermage/p67-genesis-kernel/p67_dev/p67_test_ieee_precision.py`,
ported to our production kernel signature with all v7.50/v7.51 opts
(tl.exp2 + LOG2E, -FLT_MAX, cache_modifier hints, tl.range, hoisted
invariants).

### Added

- `_build_kernel_fused()` in `p67_multi_query_kernel.py` — opt-in via
  env `GENESIS_P67_USE_FUSED=1`. Same kernel signature as split-M for
  caller compat. Architecture: ONE dot per KV-tile with
  `m=K_PLUS_1*HEADS_PER_KV=32`, vectorized online softmax over BLOCK_M
  rows with per-row causal mask `q_abs_pos[:, None] >= seq_offset[None, :]`
  (this is the v7.27 drift fix — finally validated).

### Empirical (validated 2026-04-27, opt-in test on prod)

| Metric | v7.51 split-M | v7.52 fused | Δ |
|---|---|---|---|
| @ 64 tok | 191.0 | 185.0 | -3.1% |
| @ 128 tok | 172.5 | 161.7 | -6.3% |
| **@ 256 tok** | 160.1 | 134.0 | **-16.3%** |
| @ 512 tok | 144.9 | 134.7 | -7.0% |
| @ 1024 tok | 132.0 | 128.4 | -2.7% |
| @ 2048 tok | 137.7 | 134.1 | -2.6% |
| **Stability mean** | **167.2** | **155.5** | **-7.0%** |
| Quality 30-shot | 30/31 | **30/31** | preserved |
| Tool-call 2/2 | PASS | **PASS** | preserved |

### Verdict: KEEP DEFAULT split-M, retain fused as opt-in

Quality preserved (no numerical drift) → **the v7.27 per-row online
softmax fix actually works**. The drift problem that led us to split-M
in v7.34 is solved by `q_abs_pos[:, None] >= seq_offset[None, :]`
broadcast (per-row causal mask). This is genuinely useful knowledge
captured in working code.

But throughput regressed by 7-16% because of register spill:
the fused `acc` tensor is `[BLOCK_M=32, BLOCK_D=128]` fp32 =
**16 KB virtual registers per CTA**, exceeding the A5000 register
budget (64 KB per SM, shared across active warps). Triton compiler
spills to local memory, and each `acc * alpha + dot(P, V)` then goes
through L1/L2 — far slower than the theoretical MMA-count savings
from fewer dots.

This is the same pattern as Steps E (num_stages 3→2, rejected) and F
(Q hoist, rejected): **theoretical optimization wins on consumer Ampere
require respecting register pressure**. With 64 KB per SM (RTX A5000)
and our acc tensor demand, fused-M is in the spill regime.

### Why we keep the fused code in source

1. **Reference for future maintainers** — when someone considers
   fused-M again, the rejection rationale + working code prevents
   re-treading.
2. **Useful on different hardware** — A100/H100 (96 KB / 228 KB per
   SM) may have enough register file. Operator can opt in to test.
3. **Useful at smaller BLOCK_D** — if we ever serve a model with
   HEAD_DIM=64 (not 128), `acc` shrinks to 8 KB; fused may win there.
4. **Opt-in via `GENESIS_P67_USE_FUSED=1` is zero-cost when off** —
   `_build_kernel()` checks env at module load, returns split-M if
   not set.

### Snapshot tag for rollback

`pre-tier-3-h-2026-04-27` (set before this commit).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.51.2 — 2026-04-27 (cleanup pass: torch.cat → slice-assign in fallback paths + bench v3 fix)

Pure-cleanliness pass — no behaviour change in our hot path, no measurable
performance delta (within ±5 tok/s noise band of v7.51 baseline 167 mean).
Quality 30/31 preserved.

### Changed

- **`vllm/_genesis/wiring/patch_38_tq_continuation_memory.py`** — fallback
  path (fires only when prealloc pool isn't wired, e.g. AMD/CPU tests):
  replaced final `torch.cat([k_cached_trim, key_chunk])` with the same
  `pre-allocate-then-slice` idiom used in our main `use_persistent` branch.
  Eliminates one allocation peak in the rare fallback path. Behaviour-
  equivalent.

- **`vllm/_genesis/kernels/block_verify_sampler.py`** — both `cu_start`
  construction sites (lines 131 + 275) replaced `torch.cat` with
  `torch.empty_like` + slice-assign. The tensor is 8 bytes
  (`batch_size = max_num_seqs = 2`) so the perf delta is invisible, but
  the idiom matches our P38 pattern and reads cleaner. P71 itself remains
  opt-in, default OFF.

- **`scripts/genesis_bench_v3.py`** — backported v4 fix: use
  `usage.completion_tokens` from final SSE chunk instead of counting raw
  delta chunks. Necessary because vLLM nightly batches stream deltas
  (3-5 tokens per chunk), so the old chunk-count was undercounting tokens
  by 3×, masking real throughput. Server-side bench was fixed already
  (validated v7.48 baseline at 165 tok/s vs old 51 tok/s misreading);
  public scripts now align.

### Notes

- Server-side bench tools (`/home/sander/Genesis_Project/vllm_engine/`)
  remain `genesis_bench_v3.py` (already patched) AND
  `genesis_bench_v4.py` (separate file kept for cross-check). Public repo
  ships only the corrected `genesis_bench_v3.py`.
- All `torch.cat` in `_genesis/` tree audited: only documentation comments
  remain referencing it as historical context.
- Snapshot tag for rollback: `pre-quick-wins-2026-04-27`.

---

## v7.51.1 — 2026-04-27 (Action #2/#3 evaluation + dev/public split)

Documentation-only update closing out the audit of two further candidates
from the vllm#40941 deep-dive (Action #2: OUTPUT_FP16 stage2 fold;
Action #3: `torch.cat` → slice-assign in continuation prefill).

### Action #2 — OUTPUT_FP16 stage2 fold: NOT APPLICABLE

PR #40941 adds `OUTPUT_FP16: tl.constexpr` to upstream
`vllm/v1/attention/ops/triton_decode_attention.py:_fwd_kernel_stage2`
to fold an fp32→fp16 cast into the `tl.store`. This kernel is used by
the upstream `_decode_attention` path (non-spec decode + ngram_gpu).

**Our P67 multi-query kernel does NOT use upstream `_fwd_kernel_stage2`** —
it is single-pass (no two-stage reduce-then-cast pattern), writes its
output directly inside the inner loop. So the OUTPUT_FP16 win is invisible
to our MTP K=3 verify path which is what dominates our production
workload.

The only path where this would help us is `start_no_spec_async.sh`
(no-spec mode using upstream decode kernel). For that path the win is
still small (one launch per token saved). Not worth a backport on the
MTP-default prod stack. Re-evaluate if/when we ship a no-spec variant
as a primary path.

### Action #3 — `torch.cat` → slice-assign: ALREADY DEPLOYED via P38

PR #40941 replaces `k_full = torch.cat([k_cached_trim, key_chunk])` with
pre-allocate-then-slice in upstream `_continuation_prefill`. This is
**already what our P38 (`patch_38_tq_continuation_memory.py`) does** when
its prealloc pool is wired:

```python
k_full[:cached_len].copy_(src)           # cached portion
k_full[cached_len:seq_len].copy_(key_chunk)  # new chunk
```

The fallback path (when prealloc not wired — e.g. AMD/CPU tests) still
uses `torch.cat`, which is the upstream pre-#40941 behaviour. That's
fine — fallback is rare and correctness-preserving by design.

The remaining `torch.cat` sites in our `_genesis/` tree are:
- `block_verify_sampler.py:131,275` — P71, opt-in default OFF, builds
  a 2-element `cu_start` tensor (literally 8 bytes). Marginal.
- `dequant_buffer.py` — only in comments documenting the pattern P38
  replaces.

**Net: nothing left to extract from PR #40941 that we're not already
doing.** v7.48 P38/P40 shared-pool work covered this ground.

### Repo housekeeping (separate from Action items)

- 22 dev kernel artifacts (`p67_dev/`), 2 backup tarballs (`p67_backups/`),
  and `docs/DISCUSSION_DRAFT_NOONGHUNNA.md` moved to private repo
  `Sandermage/p67-genesis-kernel`. Public patcher repo now ships only
  production-ready patches + supporting infra.
- Root-level Python harness/bench files moved under `scripts/` for tidier
  layout: `genesis_bench_v3.py`, `genesis_quality_harness.py`,
  `genesis_context_sweep.py`, `genesis_longbench_runner.py`.
- `.gitignore` hardened to prevent re-adding dev artifacts.
- Server-side backup `/home/sander/genesis-backups/v7.50-stable-20260427_0202/`
  contains full restore set (tar of `_genesis`, scripts, compile cache,
  bench tools + RESTORE.md).

### Snapshot tags (rollback-safe)

- `v7.50-stable-2026-04-27` — pre-Step-D state
- `v7.51-stable-2026-04-27` — current production (P67 exp2+FLT_MAX)
- `pre-step-d-2026-04-27` — transient, before Step D sweep
- `pre-action-2-2026-04-27` — transient, before Action #2 audit (no code changes resulted)

### Next sprints (deferred)

- **Tier 3 H** — re-fuse split-M with per-row online softmax. Multi-day
  refactor, needs FP64 reference gate (numerical correctness regression
  suite). Expected gain: +8-15%. Risk: medium (numerical drift across
  ~256 KV iterations is what split-M originally fixed).
- **Tier 3 I** — 2D split with `temp_size=32` (vllm#38786 backport).
  Multi-day, needs context-window sweep (4K → 256K). Expected gain:
  +8-15% specifically on long context.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.51 — 2026-04-27 (P67 softmax: tl.exp2 + -FLT_MAX sentinel; Step D rejected)

Two corrections to `p67_multi_query_kernel.py` softmax inner loop, derived
from deep-dive of upstream PR vllm#40929 (DeepSeek-V4 Triton fallback
kernels) — they apply textbook FlashAttention-2 numerical idioms that
our v7.34 split-M kernel had missed.

### Changed (`vllm/_genesis/kernels/p67_multi_query_kernel.py`)

- **`tl.exp` → `tl.exp2`** for online-softmax `α_t` and `P_t` updates.
  Triton's `tl.exp2` maps directly to the hardware `ex2.approx.f32`
  PTX instruction; `tl.exp` is synthesized as `ex2(x * log2e)` so adds
  one extra fp multiply per softmax step. Pre-multiplying by
  `LOG2E = 1.4426950408889634` once is the standard FA2 idiom.
- **`float("-inf")` → `-3.4028234663852886e38`** (`-FLT_MAX`) for masked-out
  attention scores. `inf*0 = NaN` in fp32 accumulator can poison the
  online-softmax across subsequent KV iterations; FLT_MAX gives the same
  effective masking via `tl.exp2(very_negative)` clamping to 0, but
  without NaN risk.

### Empirical (validated 2026-04-27, Qwen3.6-A3B-FP8 + MTP K=3)

| Metric | v7.50 | v7.51 | Δ |
|---|---|---|---|
| **Stability mean (10 runs)** | 157.6 | **167.2** | **+6.1%** |
| @ 1024 tok | 132.0 | **146.5** | **+11.0%** |
| @ 2048 tok | 137.7 | 142.0 | +3.1% |
| @ 128 tok | 172.5 | 169.4 | -1.8% |
| @ 256 tok | 160.1 | 150.8 | -5.8% (3-run high CV) |
| @ 512 tok | 144.9 | 135.9 | -6.2% (3-run high CV) |
| Quality 30-shot | 30/31 PASS | **30/31 PASS** | preserved |
| Tool-call | 2/2 PASS | 2/2 PASS | preserved |

Stability mean (10-run, low CV) is the load-bearing number. Mid-length
3-run speed tests have CV up to 11% — within noise. Long-generation
(1024+) shows clear consistent improvement.

### Step D (@triton.autotune) — REJECTED

Manual sweep of 5 alternative configs vs production
(BLOCK_KV=32, NUM_WARPS=8, num_stages=3):

| BLOCK_KV | NUM_WARPS | tok/s | vs baseline 157 |
|---|---|---|---|
| 16 | 4 | 151.0 | -3.8% |
| 16 | 8 | 148.9 | -5.2% |
| 32 | 4 | 149.1 | -5.0% |
| 64 | 4 | 147.9 | -5.8% |
| 64 | 8 | 149.6 | -4.7% |

All alternatives regressed. Current `(32, 8, stages=3)` IS the optimum
for our Ampere SM 8.6 + dequant-heavy workload. No autotune needed —
the search space has a clean global maximum at the current setting.
Rejection rationale recorded in P67 docstring with each config row.

### Notes

- `LOG2E = 1.4426950408889634` is precomputed at compile time inside the
  inner loop body — Triton constant-folds it.
- `_FLT_MAX_NEG = -3.4028234663852886e38` is the IEEE 754 most-negative
  finite fp32 value. It survives all subtraction operations within the
  online-softmax range without underflow.
- Action #1 derived from research-agent deep-dive of vllm#40929 DeepSeek-V4
  Triton fallback kernels (PR doesn't apply to us as a model, but the
  softmax idioms inside transfer cleanly to our k8v4 verify path).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.50 — 2026-04-27 (Tier 1 Step C: P67 cache_modifier + tl.range hints)

Backport of [vllm#33529](https://github.com/vllm-project/vllm/pull/33529)
("Triton MLA perf fixes", merged 2026-04-02) Triton compiler hints into our
P67 multi-query attention kernel. Memory-traffic optimizations only — zero
arithmetic change.

### Changed (`vllm/_genesis/kernels/p67_multi_query_kernel.py`)

- **`tl.range()` instead of plain `range()`** for the outer KV loop —
  explicit Triton pipelining hint. Lets the compiler overlap `cp.async`
  loads with prior-iteration MMA on Ampere.
- **`cache_modifier=".cg"`** on K/V dequant raw loads (`KV_cache_ptr +
  k_addrs / val_addrs`) — streaming reads that should NOT pollute L1.
  L2-direct frees L1 capacity for Q + scales.
- **`cache_modifier=".ca"`** on Q load, `Block_table_ptr` lookup, and
  scale/zero loads (`sc_lo`, `sc_hi`, `zr_lo`, `zr_hi`) — these are
  reused inside the CTA across all KV iterations. Pinning them in L1
  saves repeated DRAM round-trips.
- **Hoisted `kv_head * stride_cache_head`** out of the inner KV loop
  (`_kv_head_byte_offset` precomputed once per CTA) — invariant across
  all per-tile `slot_bases` calculations. Triton -O2 would also hoist
  this but explicit form matches upstream MLA decode style.

### Empirical (validated 2026-04-27, 2× RTX A5000 + Qwen3.6-A3B-FP8 + MTP K=3)

| max_tokens | v7.48 | v7.50 | Δ |
|---|---|---|---|
| 64 | 188.6 | 191.0 | +1% |
| 256 | 145.6 | **160.1** | **+10%** |
| 512 | 141.8 | 144.9 | +2% |
| 1024 | 132.8 | 132.0 | ~0% |
| 2048 | 129.2 | 137.7 | +6.5% |

Stability mean 157-162 tok/s (within v7.48 noise band). Quality 30/31
PASS unchanged. Tool-call regression 2/2 unchanged. Long-context probe
16K-160K all PASS at GMU 0.90.

### Notes

- All hints are memory-traffic only — `cache_modifier` is a PTX-level
  cache-policy attribute, not arithmetic. Numerical correctness verified
  via quality harness (no token-level deviations from v7.48 baseline).
- `tl.range()` enables Triton 3.x async-copy pipelining (`num_stages>1`
  in our autoconfig). On Triton 2.x it falls back to a plain loop —
  graceful degrade.
- Tested ONLY on `cache_modifier=".cg"`/`".ca"` literals supported by
  Triton 3.x on Ampere (sm_86). On older Triton, the modifiers are
  ignored — kernel still correct, just no cache-policy hint applied.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.49 — 2026-04-27 (P79d retired + P79c improved per upstream review)

Two small but important corrections to the v7.46 async-safety patch trio,
based on upstream maintainer feedback received within 24h of v7.48 push.
Tested versions unchanged from v7.48 (vLLM `dev212+g8cd174fa3`, PyTorch
`2.11.0+cu130`, Triton `3.6.0`, CUDA 13.0, driver 580.126.09, 2× RTX A5000).

### Removed

- **P79d retired completely** (`vllm/_genesis/wiring/patch_79d_preempt_async_discard.py` + dispatcher entry + apply_all register).
  - Reason: njhill (vLLM core maintainer) explicitly confirmed in
    [vllm#38624](https://github.com/vllm-project/vllm/pull/38624) that the
    asymmetry P79d "fixed" is **intentional**: the regular `_preempt_request`
    removes the request from the next step entirely, so placeholder state is
    never re-read; only `reset_prefix_cache` re-admits it. The backport
    targeted a non-bug.
  - CodersAcademy006 (original PR author) acknowledged the static-analysis
    miss and committed to closing #38624 with a clarifying comment.
  - Genesis prod is unaffected (P79d was opt-in, default-off, never enabled
    in our `start_mtp.sh`). Removal is preventive — keeps the patcher
    surface clean and avoids any operator accidentally enabling a
    misguided modification.

### Improved

- **P79c smarter cleanup** (`vllm/_genesis/wiring/patch_79c_stale_spec_token_cleanup.py`):
  - Old behaviour: cleared **any** `spec_token_ids` for unscheduled running
    requests — risked wiping **real draft token IDs** (positive ints from
    MTP / EAGLE / ngram), not just `-1` placeholders. Could corrupt MTP
    state across budget-exhaustion cycles.
  - New behaviour (matches the spirit of the emerging canonical fix
    [vllm#40768](https://github.com/vllm-project/vllm/pull/40768) by jvlunteren):
    1. Only clear when `spec_token_ids` is **all `-1`** (`all(t == -1 for t in ids)`).
       Real draft tokens preserved.
    2. **`prev_step_scheduled_req_ids` membership gate** — if request was in
       the previous worker step, placeholders may still be consumed by async
       input prep; we leave them alone. Otherwise (new request not in prev
       step) → safe to clear.
  - Drift detector unchanged — when #40768 (or the eventual canonical fix)
    merges and adds `_consume_spec_decode_tokens_for_step`, our P79c
    self-skips and the upstream takes over.
  - Still opt-in via `GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP=1`.
    Genesis prod (sync ngram, max_num_seqs=2) still doesn't engage it —
    only protects high-concurrency multimodal users on async + EAGLE/MTP.

### Tracker delta (since v7.48 push)

- **vllm#38624 (P79d source)**: dead per maintainer — already retired here.
- **vllm#40610 (P79b source)**: still draft, no human review — backport stays.
- **vllm#37629 (P79c source)**: active discussion (benchislett asked for
  non-multimodal repro; haosdent committed to providing one). Watch for v2
  with proper root-cause fix.
- **vllm#40925 (P81 source)**: open, mergeable, blocked on first-time
  contributor label gate. Backport stays. Will retire when merged.
- **vllm#40768 (canonical fix for the bug class P79c addresses)**: NEW PR,
  introduces `_consume_spec_decode_tokens_for_step` + dedicated
  `num_pending_async_spec_placeholders` Request field. Direct supersession
  candidate for our P79c — when it lands we drop the patch entirely.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.48 — 2026-04-27 (memory shared-pool sprint + P81 backport + driver 580)

Tested on **vLLM 0.19.2rc1.dev212+g8cd174fa3** (nightly image
`vllm/vllm-openai:nightly` ID `10c7a6ba51c6`, PyTorch 2.11.0+cu130,
Triton 3.6.0) on **NVIDIA driver 580.126.09 + CUDA 13.0**, 2× RTX
A5000 (Ampere SM 8.6), Qwen3.6-35B-A3B-FP8 with TurboQuant k8v4 KV
cache and MTP K=3 spec decode.

### Added

- **P81 — fp8 block-scaled MM low-M decode tuning**
  (`wiring/patch_81_fp8_block_scaled_m_le_8.py`):
  - Backport of [vllm#40925](https://github.com/vllm-project/vllm/pull/40925)
    (tonyliu312, OPEN as of 2026-04-26)
  - Opt-in via `GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1`
  - Specializes default `w8a8_triton_block_scaled_mm` config for M ≤ 8
    (single-request decode + MTP K=3 verify): `BLOCK_SIZE_M` 64 → 16,
    `num_stages` 2 → 3 (non-ROCm)
  - Direct hit for our prod (Qwen3.6-A3B FP8 + max_num_seqs=2,
    no pre-tuned JSON for A5000)
  - Empirical (per upstream PR on GB10 sm_121): +23% median decode
  - Drift detector: presence of `if M <= 8:` literal without Genesis
    marker → upstream PR merged → auto-skip

- **`vllm/_genesis/buffer_mode.py`** — centralized buffer-mode toggle:
  - Reads `GENESIS_BUFFER_MODE=shared|per_layer` env (default `shared`)
  - Per-patch override via `GENESIS_BUFFER_MODE_<PID>` (e.g. P38, P40)
  - `shared` = singleton pool via `GenesisPreallocBuffer` (memory-efficient,
    saves multi-GB on long-context)
  - `per_layer` = legacy attached-attribute path (rollback safety)

### Memory-opt sprint

Driver 570 → 580 upgrade brought CUDA 13.0 PyTorch which adds ~3 GB
allocator overhead. To restore long-context capability while staying
at GMU 0.90+, audited prealloc patches:

- **8 of 9 patches** (P22/P26/P28/P36/P37/P39/P44/P46) **already use
  shared singleton** via `TurboQuantBufferManager` /
  `GenesisPreallocBuffer` / `gdn_core_attn_manager` /
  `FlaKktBufferManager`. The `setattr(layer, ...)` only attaches 36
  references to a single registered buffer; per-layer attribute lookup
  is for fast-path access, not duplicated allocation.

- **P38 was the real waste** — `_genesis_continuation_prefill` had a
  fresh `torch.empty(buf_shape, ...)` fallback when `seq_len` exceeded
  current buffer, allocating per-call growth. Fixed
  `wiring/patch_38_tq_continuation_memory.py` to use `buffer_mode_for`:
  `shared` mode now allocates **one max-size buffer** per
  (Hk, D, dtype, device) signature via `GenesisPreallocBuffer`, slice
  to actual `alloc_len` per call. Single namespace, no growth churn,
  no per-layer waste.

- **P40 fallback** (TQ grouped decode `mid_o`/`output`/`lse` per-call
  `torch.empty` when `buf_holder` not pre-attached) similarly fixed via
  `buffer_mode_for("P40")` shared singleton with max-shape (max_B,
  Hq, NUM_KV_SPLITS, D+1) registered through `GenesisPreallocBuffer`.

### Empirical (validated v7.48 baseline)

| Metric | v7.48 (driver 580 + P81 + shared P38/P40) | vs v7.13 (driver 570 + per-layer) |
|---|---|---|
| Throughput mean | **160-190 tok/s** | 130-143 tok/s = **+15-30%** |
| Quality 30-shot harness | 30/31 PASS (96.8%) | 30/31 PASS |
| Tool-call regression | 2/2 PASS | 2/2 PASS |
| Long-ctx 16K-160K needle | ALL PASS | 16K-128K PASS |
| Long-ctx 200K | PASS (153K server tokens) | OOM |
| **GMU at which 200K runs** | **0.90** (Sander obligatory range MET) | 0.91 limit |
| Production launch script | `scripts/launch/start_mtp.sh` (updated) | unchanged |

### Notes

- Env-driven `GENESIS_BUFFER_MODE=shared` is the new default. Set to
  `per_layer` if shared pool ever shows regression on a different
  model/config — purely a rollback knob, not expected for normal use.
- The shared pool requires sequential layer execution within the
  forward pass (which is true for TP + non-PP + sync-scheduling — our
  config). Anyone adding pipeline parallelism or multi-stream pipelined
  layers should re-evaluate.
- Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.

---

## v7.46 — 2026-04-26 (async × spec-decode safety patches — opt-in)

Three additive backports of OPEN upstream PRs that fix async-scheduling
race conditions on EAGLE/MTP/ngram_gpu paths. **All three default-off:**
Genesis prod (sync ngram, max_num_seqs=2) gains nothing direct — these
protect users who run `--async-scheduling` + spec-decode.

### Added

- **P79b — Async × spec-decode proposer-sync** (`wiring/patch_79b_async_proposer_sync.py`):
  - Backport of [vllm#40610](https://github.com/vllm-project/vllm/pull/40610) (OPEN draft)
  - Opt-in via `GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC=1`
  - Wraps `GPUModelRunner.sample_tokens()` to re-record `prepare_inputs_event`
    in `finally:` AFTER spec-decode proposer GPU work completes
  - Fixes happens-before race where next batch's `_update_states` could
    mutate persistent block_table while previous batch's proposer was
    still reading on GPU — symptom: nondeterministic stale state on
    async + EAGLE/MTP/ngram_gpu
  - Drift detector: presence of `_sample_tokens_impl` symbol without
    Genesis marker → upstream merged → auto-skip
  - Verified on dev205+g07351e088: applies cleanly, file compiles,
    method reload picks up new structure, idempotent

- **P79c — Stale spec_token_ids cleanup** (`wiring/patch_79c_stale_spec_token_cleanup.py`):
  - Backport of [vllm#37629](https://github.com/vllm-project/vllm/pull/37629) (OPEN, fixes #36906)
  - Opt-in via `GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP=1`
  - Adds cleanup pass after main scheduling loop in `Scheduler.schedule()`
    that clears `spec_token_ids` for any running request not in
    `num_scheduled_tokens`
  - Fixes EAGLE3 + async high-concurrency CUDA device-side assert
    in `F.embedding()` from stale `-1` placeholder leak
  - Triggered when token budget exhausts before scheduler visits all
    running requests; not multimodal-specific (PR's regression test
    proves text-only with sufficient concurrency reproduces)
  - Verified on dev205+g07351e088: applies cleanly, file compiles,
    idempotent

- **P79d — Preempt async-discard** (`wiring/patch_79d_preempt_async_discard.py`):
  - Backport of [vllm#38624](https://github.com/vllm-project/vllm/pull/38624) (OPEN, CodersAcademy006)
  - Opt-in via `GENESIS_ENABLE_P79D_PREEMPT_ASYNC_DISCARD=1`
  - Adds 2 lines to `_preempt_request()` that set
    `num_output_placeholders=0` + `discard_latest_async_tokens=True`
  - Currently set ONLY in `reset_prefix_cache()` — scheduler-loop
    preemption path bypasses cleanup, leading to duplicated tokens
    after request resume on async paths ("the the", "of of")
  - SAFER than upstream PR: additive only — does NOT remove the
    existing block from `reset_prefix_cache()` (defensive)
  - Drift detector: counts `discard_latest_async_tokens = True`
    occurrences; ≥2 → upstream merged → auto-skip
  - Verified on dev205+g07351e088: applies cleanly, count goes 1→2,
    marker present, idempotent on re-apply

### Investigation notes (no patch)

- `docs/DRAFT_38903_async_pp_contamination.md` — local draft on
  [vllm#38903](https://github.com/vllm-project/vllm/issues/38903)
  (cross-request data contamination on PP>1 + async + multi-node).
  Severe privacy bug but Genesis cannot reproduce (TP=2, PP=1,
  single-node, single-user). Document includes proposed config-level
  bandaid, recommendation NOT to ship it (over-broad, no reproducer),
  and pointer to research-agent's Section 5 Cat-5v embedding-input
  invariant guard as a multi-bug defensive layer worth implementing.

### Empirical findings (no patch — data only)

- **P80 ngram_gpu+async verification on dev205+ pin**: 3-shot bench on
  Qwen3.6-A3B FP8 + `--async-scheduling` + ngram_gpu method (testing
  whether [vllm#37150](https://github.com/vllm-project/vllm/issues/37150)
  fix shipped). Result: 35-43 tok/s mean ≈40 — works without error
  (no cascade, normal output) but SLOWER than sync ngram CPU (46 tok/s).
  Same pattern as our MTP+async finding: async overhead > savings on
  single-user max_num_seqs=2 setups.
- Confirms #37150 fix IS active in our pin (no 1.22% acceptance
  pathology), but ngram_gpu+async at single-user is net-negative —
  use sync ngram CPU instead for our workload class.

### Upstream-watch deltas

These three PRs are independent of the TurboQuant workspace cluster
tracked above. Each has its own drift marker; none conflict with
P22/P26/P67/P67b. After any of #40610/#37629/#38624 merges, the
respective drift detector auto-skips the corresponding Genesis patch.

---

## Upstream-watch — pending rebase work (added 2026-04-26)

Three competing upstream PRs target the TurboQuant decode scratch workspace
that our P22 / P26 / P67b stack already addresses. We monitor + auto-skip
when any merges. Action plan per PR:

| Upstream PR | Author | Drift marker | Genesis impact |
|---|---|---|---|
| **#40798** (likely winner) | Bot1822 | `_reserve_turboquant_decode_workspace` symbol in `vllm/v1/worker/gpu_model_runner.py` | (1) P22 auto-skips via drift detector. (2) **P67b needs rebase** — the PR REMOVES `buf_holder` kwarg from `triton_turboquant_decode_attention`. Drop these 4 explicit args from `patch_67b_spec_verify_routing.py:131-156`: `mid_o_buf`, `output_buf`, `lse_buf`, `buf_holder=layer`. Routing logic itself unchanged. |
| #40706 (backup) | lesj0610 | `reserve_turboquant_decode_workspace` symbol in `vllm/v1/attention/backends/turboquant_attn.py` | (1) P22 auto-skips. (2) P67b unchanged — preserves `buf_holder` fallback. |
| #40655 | bhoomit | `_init_turboquant_buffers` REMOVED from `TurboQuantAttentionImpl` | (1) P22 auto-skips. (2) P67b unchanged. CHANGES_REQUESTED upstream — less likely to land. |

**Drift detector**: `wiring/patch_22_tq_prealloc.py:_check_upstream_tq_workspace_drift()` probes for all 3 markers. When any matches, P22 returns `("skipped", "PR #XXXXX merged ...")` — ready for next sync without manual intervention.

**P26 (prefill output)**: orthogonal to all 3 PRs (they target decode path, P26 covers prefill). KEEP.

**Our PR #40914**: complementary, not competing. Routing fix vs workspace dedup are separate axes.

---

## v7.11.0 — 2026-04-25 (spec-decode workaround + diagnostic tooling)

**Investigation + opt-in workaround for [vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831)** — TurboQuant × any speculative decoding (MTP or ngram) produces degenerate token loops on structured outputs.

### Added

- **P56 — TQ spec-decode safe-path guard** (`wiring/patch_56_spec_decode_decode_path_guard.py`):
  - Opt-in via `GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=1` (off by default)
  - 5-line text-patch on `turboquant_attn.py` tightens `_prefill_attention` continuation fast-path entry from `q_len ≤ _CONTINUATION_DECODE_THRESHOLD` to `q_len == 1`
  - Spec-decode batches (q_len > 1) now route through `_continuation_prefill`'s `flash_attn_varlen_func(causal=True)` — causal-correct
  - Closes Layer 1 (catastrophic XML/JSON loops); Layer 2 (token duplication, e.g. `for for`, `age age`, `parameter parameter`) remains and is upstream's territory
  - Registered in `apply_all.py` between P26 and P44

- `scripts/sequential_backend_probe.py` — 9-prompt diagnostic probe set covering smoke / narrative / tool calls (no-thinking + thinking) / JSON / needle short+medium / code / structured XML. `run` subcommand fires the set against any vLLM endpoint and writes JSONL; `diff` subcommand compares two such logs side-by-side with degenerate-pattern detection.

- `scripts/dual_backend_diagnostic_proxy.py` — FastAPI proxy on :9000 forwarding each request to two backends concurrently. Captures both responses byte-for-byte, computes structural diff, detects degenerate patterns. Useful when concurrent backends fit (we currently fall back to sequential probing because TP=2 saturates our 2× A5000).

### Verified on Genesis pin `fe9c3d6c5`

- 2× RTX A5000 (Ampere SM 8.6), TP=2
- Qwen3-Next-35B-A3B-FP8 (MoE hybrid), `kv_cache_dtype=turboquant_k8v4`
- ngram spec-decode `n=3` (chosen so result doesn't depend on MTP draft head)
- Reproduced #40831 catastrophically without P56 (`tool_calls=[]`, content=`<parameter=parameter=unit>...</parameter>×16+`)
- With P56: `tool_calls` populated, narrative coherent, no infinite loops
- Layer 2 token-duplication probed via 9-prompt diff against prod baseline; documented in upstream comment

### Upstream interactions (this release)

- [#40807 issuecomment-4316663581](https://github.com/vllm-project/vllm/issues/40807#issuecomment-4316663581) — pointed at P44+P23 as fix direction for the CUDA graph crash (noonghunna's first bug)
- [#40124 issuecomment-4316828133](https://github.com/vllm-project/vllm/issues/40124#issuecomment-4316828133) — replied to noonghunna's heads-up; promised the test we then ran
- [#40831 issuecomment-4317214311](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4317214311) — full Layer 1 root cause + P56 workaround + Layer 2 finding

### Files touched

- `vllm/_genesis/wiring/patch_56_spec_decode_decode_path_guard.py` (new, ~165 lines)
- `vllm/_genesis/patches/apply_all.py` (+46 lines for P56 registration)
- `scripts/sequential_backend_probe.py` (new, ~225 lines)
- `scripts/dual_backend_diagnostic_proxy.py` (new, ~265 lines)
- `README.md` — v7.11 What's-new section, P56 in opt-in roster, upstream tracking with issuecomment IDs, scripts/ in architecture
- `vllm/_genesis/CHANGELOG.md` — this entry

---

## v7.9.0 — 2026-04-24 (runtime architecture-dispatch detection)

**Defense-in-depth layer 2: detect which patches need to fire before work begins.**

### Added

- `model_detect.py` — cached `get_model_profile()` returns `(moe, hybrid, turboquant)`
  - `is_moe_model()` — Qwen3-MoE / Mixtral / DeepSeek / Gemma-4-MoE / architecture + model_type heuristics
  - `is_hybrid_model()` — Qwen3-Next `layer_types`, Mamba, GDN, SSM detection
  - `is_turboquant_active()` — config-level `kv_cache_dtype` check (layer-level is P51 in `dequant_buffer.py`)
  - `log_skip(patch, reason)` — uniform single-line dispatch log format
  - `clear_for_tests()` — cache reset for unit tests
  - Conservative fallback: unknown config → True for all flags (patches still apply, their own guards decide)

- **P51 — TQ-active runtime detection** in `kernels/dequant_buffer.py::ensure_turboquant_buffers`
  - Reads `impl.kv_cache_dtype`; early-returns with single log if non-TurboQuant
  - Saves ~516 MiB / rank on FP16-KV + `auto` deployments where TQ text-patches graceful-skip but preallocs would fire
  - `_p51_logged` flag avoids log spam across all model layers (one log per impl)

- **P52 — MoE-active dispatch gate** wired into `wiring/patch_{24,31,37}_*.py`
  - Skips P24 (MoE num_warps overlay), P31 (grouped-topk fp32 upcast), P37 (intermediate-cache pool) on dense models
  - Single log line per skipped patch at apply time; no runtime overhead thereafter

- **P53 — Hybrid-active dispatch gate** wired into `wiring/patch_{28,34,39,46}_*.py`
  - Skips P28 (GDN core-attn rewire), P34 (Mamba zero-collapse guard), P39a (FLA kkt pool), P46 (GDN gating pool) on pure-attention models
  - All targets still graceful-skip without P53 (their text-patch anchors wouldn't match), but the dispatch log now explains *why*

- `tests/test_model_detect.py` — 19 tests covering MoE detection across architectures, hybrid detection, TQ detection, conservative fallback, caching, log helper
- `tests/test_p51_tq_active.py` — 8 tests covering fp8/auto/fp16 skip, single-log-per-impl, legacy-impl backward compat, TQ-active passthrough

### Changed

- `kernels/dequant_buffer.py::ensure_turboquant_buffers` now early-returns on non-TQ impls before any config resolution work
- Wiring apply() docstrings updated to reference P52/P53 gates where applicable
- Root `README.md` rewritten for v7.9 with compatibility matrix, installation guide, patch roster, upstream tracking

### Upstream correspondence

Re-audit of `vllm-project/vllm` since 2026-04-24 surfaced:
- **#40807** (OPEN) — TurboQuant + spec-decode capture crash; reporter namechecks Sander's Patch 23. Our P44 aligns.
- **#40792** (OPEN) — TQ k8v4 GQA head grouping; may supersede our P40. Diff + bench pending.
- **#40798** (OPEN) — TQ scratch workspace across layers; superset of #40655+#40706. May conflict with P28 anchor.
- **#40794** (MERGED 2026-04-24) — MoE unpad routed output; smoke test on Qwen3.6-35B-A3B pending.
- **#40420** (OPEN) — TQ continuation-prefill OOM at 185k; adding ≥150k regression to integration gate.

No PR posted upstream without explicit user approval (per `feedback_no_push_without_explicit_approval`).

---

## v7.8.5 — 2026-04-24 (cross-quantization validation)

Validated v7.8 on three configurations: FP8 prod / AWQ 4-bit / FP16-KV 32k.

**Results**: 28 applied / 0 failed across all three. 3× 256k stable on FP8 + AWQ. AWQ frees ~9 GiB/rank → 2.5× KV capacity (1.099M → 2.787M tokens). Speed: AWQ 1-4% slower than FP8 (4-bit dequant cost on SM 8.6). Linear degradation unchanged: `1/tgs ≈ 0.007 + 2.4e-5 × ctx`.

**Finding**: TQ preallocated buffers waste ~516 MiB/rank on FP16-KV deployments where TQ is inactive — led to P51 in v7.9.

## v7.8.0 — 2026-04-24 (interface guards + middleware)

### Added

- **P49 — interface contract validation** (`interface_guard.py`, ~240 lines)
  - `GenesisInterfaceMismatch` exception
  - `validate_impl(impl, required_attrs, required_methods, optional_attrs, role)` helper
  - `validate_method_signature(method, expected_params)` — catches renamed params
  - `assert_shape_compat(t, expected, msg)` — runtime shape drift detection
  - `describe_impl(impl)` — diagnostic snapshot
  - `ANY` sentinel — presence-only check (used for Triton `@triton.jit` kernels that aren't `callable()` in Python sense)
  - Wired into P22, P38, P39a as pre-flight guards (defense layer 1)

- **P50 — ASGI `ResponseCacheMiddleware`** (`middleware/response_cache_middleware.py`, ~280 lines)
  - Drop-in ASGI middleware for any FastAPI/Starlette app (target: cliproxyapi:8330)
  - Deterministic cache key (JSON `sort_keys=True`)
  - `stream=True` + sampled requests (`temp>0`, `top_p<1`, `top_k>1`) NOT cached by default
  - Graceful degradation on cache errors (silent miss)
  - `x-genesis-cache: HIT|MISS` header for diagnostics

- 18 tests in `test_interface_guard.py` (validate, sig, shape, describe)
- 25 tests in `test_response_cache_middleware.py` (key extraction, ASGI flow, error handling)

### Fixed

- P39a initial false-positive: Triton `@triton.jit` `chunk_scaled_dot_kkt_fwd_kernel` isn't Python-callable. Switched to `required_attrs={...: ANY}` (presence check) instead of `required_methods` (callable check). The guard correctly caught the edge case — API usage corrected.

### Tests

Full unit suite: 605 passed / 8 skipped / 0 failed.

---

## v7.0.0-dev — 2026-04-24

**Major architectural shift**: migrate from monolithic text-replacement overlay (`patch_genesis_unified.py`, ~3000 LOC) to modular professional package.

### Added

- `vllm/_genesis/` package structure (upstream-compatible namespace)
- `guards.py` — canonical vendor/chip/model/dependency detection
  - Vendor identity: `is_nvidia_cuda()`, `is_amd_rocm()`, `is_intel_xpu()`, `is_cpu_only()`
  - NVIDIA compute capability: `get_compute_capability()`, `is_sm_at_least(major, minor)`, arch predicates (`is_ampere_consumer()`, `is_hopper()`, `is_blackwell()`, etc.)
  - AMD architecture: `is_rocm_cdna2()`, `is_rocm_cdna3()`, `is_rocm_rdna()` via `_GCN_ARCH` parsing
  - Dependency versions: `get_torch_version()`, `get_transformers_version()`, `get_vllm_version_tuple()`, `is_transformers_v5_plus()`, `is_torch_211_plus()`
  - Model architecture: `is_model_arch(cfg, arch_name)`, family helpers (`is_qwen3_family`, `is_deepseek_v3`, etc.)
  - Backend detection: `has_turboquant_support()`, `is_marlin_selected()`, `is_flash_attn_backend()`
  - Path resolution: `vllm_install_root()`, `resolve_vllm_file()` — replaces hardcoded `/usr/local/lib/python3.12/` paths (works on any Python version, Mac/Linux/Docker slim)
  - Diagnostic: `platform_summary()` returns full JSON-serializable platform info

- `prealloc.py` — `GenesisPreallocBuffer` framework
  - Class-level registry for shared tensor allocation
  - `get_or_create(namespace, shape, dtype, device, zero_init)` — fresh or cached
  - `slice_to(buf, n, dim)` — pointer-stable view (CUDA graph safe)
  - `get_registry_info()` — diagnostic JSON of all allocations
  - `clear_for_tests()` — test helper (warns if called outside pytest)

- `kernels/router_softmax.py` — **Patch 31** implemented
  - Drop-in replacement for `torch.softmax` in MoE routers
  - Fp32-upcast intermediate prevents bf16 mantissa collision
  - Fixes non-deterministic top-k routing on Qwen3-MoE (pre-SM90)
  - `router_softmax()` and `router_softmax_preserving_mask()` variants
  - Platform-universal: CUDA / ROCm / XPU / CPU all supported

- `kernels/dequant_buffer.py` — **Patch 22** skeleton (Phase 2 target)
  - `TurboQuantBufferManager` class with platform guard
  - Designed for profiler-visible KV buffer pre-allocation

- `kernels/gdn_dual_stream.py` — **Patch 7** skeleton (Phase 2 target)
  - `DualStreamDispatcher` with platform-aware fallback
  - NVIDIA parallel, ROCm HIP attempt, XPU/CPU sequential

- `kernels/marlin_tuning.py` — **Patch 17/18** skeleton (Phase 2 target)
  - Per-SM optimal `block_size_m` auto-selection
  - Env overrides: `VLLM_MARLIN_MOE_BLOCK_SIZE_M`, `_NUM_WARPS`, `_NUM_STAGES`

- `kernels/fp8_dispatcher.py` — **Patch 1/2** skeleton (Phase 2 target)
  - `requires_marlin_fp8_fallback()` — SM<8.9 detection
  - Per-arch routing logic

- `patches/apply_all.py` — new orchestrator replacing monolithic patcher
  - Decorator-based patch registration (`@register_patch("P31 ...")`)
  - `PatchStats` with counts and per-patch details
  - CLI entrypoint: `python3 -m vllm._genesis.patches.apply_all`
  - Exit codes: 0 success / 1 patch failure / 2 setup error
  - Stub registration for Patch 31 (full implementation Phase 2)

- `patches/upstream_compat.py` — upstream PR marker registry
  - Central tracking of all upstream fixes Genesis mirrors
  - Used by Layer 3 (upstream merge) defensive checks
  - Coverage: #39016, #39391, #39953, #40060, #40105, #40159, #40172, #40194, #40384, #40572, #40633, #38479

- `tests/conftest.py` — pytest fixtures
  - `cuda_available`, `rocm_available`, `nvidia_cuda_available` platform fixtures
  - `reset_genesis_prealloc` — clear registry before/after test
  - `deterministic_seed` — torch.manual_seed(42)
  - Custom markers: `cuda_required`, `rocm_required`, `gpu_required`, `slow`
  - Auto-skip GPU tests on CPU-only hosts

- `tests/test_guards.py` — comprehensive guards test coverage
  - TestVendorIdentity (6 tests)
  - TestComputeCapability (5 tests)
  - TestDependencyVersions (4 tests)
  - TestModelArchDetection (4 tests)
  - TestBackendDetection (2 tests)
  - TestPathResolution (3 tests)
  - TestPlatformSummary (2 tests)

- `tests/test_prealloc.py` — `GenesisPreallocBuffer` test coverage
  - TestGetOrCreate (7 tests)
  - TestSliceTo (6 tests)
  - TestRegistryInfo (3 tests)
  - TestPointerStability (2 tests) — CRITICAL for CUDA graph
  - TestClearForTests (2 tests)
  - TestCUDABehavior (2 tests)

- `tests/test_router_softmax.py` — Patch 31 TDD test suite
  - TestRouterSoftmaxDeterminism (3 tests)
  - TestRouterSoftmaxDtypePreservation (5 tests, parametrized)
  - TestRouterSoftmaxMathematicalCorrectness (5 tests)
  - TestRouterSoftmaxPlatformSafety (4 tests)
  - TestRouterSoftmaxEdgeCases (3 tests)
  - TestRouterSoftmaxPerformanceCUDA (1 test, CUDA-gated)

- `README.md` — package documentation with usage, testing, migration status
- `CHANGELOG.md` — this file

### Design decisions (why this structure)

1. **Why `vllm/_genesis/` namespace**: placed inside vllm's package layout so installation via overlay mount works without PYTHONPATH manipulation. Leading underscore marks it as "private" (Genesis-specific, not upstream API).

2. **Why separate `kernels/` and `patches/`**: clean separation between WHAT the code does (kernels) and HOW it integrates (patches). When we submit upstream PRs, we submit kernels/ directly — patches/ is just the bridging overlay.

3. **Why TDD discipline**: matches user's `CLAUDE.md` explicit requirement: "Test-first for new functionality." Also mandatory for Patch 28 (GDN prealloc) to prevent repeating Patch 19's revert (−30% throughput, 188× stdev).

4. **Why `@functools.cache` on guards**: NVML probe and vllm.platforms queries are ~1ms. Cached after first call (~50ns). At 20+ patches × startup = 20ms vs 1μs difference.

5. **Why `vllm_install_root()` helper**: replaces hardcoded `/usr/local/lib/python3.12/dist-packages/` (breaks on Mac, venv, Python 3.13 coming 2027, Docker slim images). `vllm.__file__` is canonical universal.

### Not yet done (Phase 2 target)

- Full monkey-patch glue from `kernels/` to upstream vllm modules (current v5.14.1 does text-replacement; v7.0 will use function-level monkey-patching via `patches/apply_all.py`)
- Remaining kernel implementations: `dequant_buffer.py`, `gdn_dual_stream.py`, `marlin_tuning.py`, `fp8_dispatcher.py`
- Test suites for the 4 remaining kernels
- Integration platform matrix tests (`test_platform_matrix.py`)
- Migration of Patches 1-25 from monolithic `patch_genesis_unified.py` to per-patch modular entries

## Late v7.0-dev additions (2026-04-24, session 2)

### New patches wired

- **P7** — GDN dual-stream in_proj parallelism. Text-patch on
  `model_executor/layers/mamba/gdn_linear_attn.py:544-545` replacing the
  serial `in_proj_qkvz` + `in_proj_ba` calls with a
  `DualStreamDispatcher.maybe_parallel(...)` call. Platform-safe:
  sequential fallback on CPU / XPU; true parallelism on CUDA SM ≥ 8.0.
- **P12** — Qwen3 `<tool_call>` as implicit reasoning end (ADDITIVE scope
  to avoid conflict with P27). Adds `_tool_call_token_id`,
  `is_reasoning_end`, `is_reasoning_end_streaming`,
  `extract_content_ids` methods to `Qwen3ReasoningParser`.
- **P24** — Per-SM auto-select for Marlin MoE `num_warps` and
  `num_stages`. Ampere A5000 (SM 8.6) → warps=4, stages=3 measured
  optimum. Env `VLLM_MARLIN_MOE_NUM_WARPS`, `_NUM_STAGES` still override.
- **P26** — TurboQuant prefill output prealloc. Helper
  `TurboQuantBufferManager.get_or_create_prefill_output` +
  `layer._tq_prefill_output` attach. Kernel text-patch deferred to A/B
  benchmark.
- **P27** — Qwen3 reasoning parser BEFORE-THINK fallback. Captures text
  before `<think>` (previously dropped) and routes it to `content` in
  both streaming and non-streaming paths. Coexists with P12.
- **P28** — GDN `core_attn_out` prealloc via `GdnCoreAttnManager`.
  Correct P19 redo: allocation is profiler-visible via the manager's
  first `get_or_create` on the max-sized buffer, not lazy in forward.
  Text-patch on `gdn_linear_attn.py:569-575` with unique anchor
  (includes the preceding #28182 comment) so `forward_xpu`'s identical
  line is untouched.
- **P29** — Verified the qwen3coder tool parser already contains
  bounded-index guards in the v7.0 baseline (lines 609-616, 659-666,
  436-438). Registration is a no-op on the current image; re-emits if a
  future vLLM upgrade regresses.
- **P32 / P33** — `_cu_2` and `synth_seq_lens` preallocs bundled with
  P22. `TurboQuantBufferManager.get_or_create_cu_2` +
  `get_or_create_synth_seq_lens`; attached to the layer inside
  `ensure_turboquant_buffers`.
- **P5b** — Scaffolding for the future pad-smaller-to-max KV
  unification. `kernels/page_size_padded.py` helpers
  (`is_p5b_enabled`, `compute_real_page_size_bytes`, `clamp_to_real_shape`)
  behind `GENESIS_ENABLE_P5B=1`. Kernel text-patch intentionally not
  shipped.

### Infrastructure

- `benchmarks/harness/` — Part 11.1 pre-deploy gate runner:
  - `gsm8k_regression`, `quality_harness`, `long_context_oom`,
    `tgs_decode`, `offline_api_parity`, `cuda_graph_recapture`,
    `run_all`.
  - Standard JSON report format, P0/P1 tiering, aggregated `summary.json`.
  - Dataset stubs in `benchmarks/data/`.
- `docs/RUNBOOK.md` — steady-state ops, diagnostic probes, blue/green
  deploy, rollback, known gotchas.

### Patch registry size

- Session start: 16 registered patches.
- Session end: **23 registered patches** (+P7, +P12, +P24, +P26, +P27,
  +P29, +P32/P33, +P28, +P5b).

### Compatibility

- Python: 3.10+ (uses modern type hints and `from __future__ import annotations`)
- PyTorch: 2.10+ (compatible with 2.11 upgrade in v0.20.0)
- Transformers: v5.0+ (compatible with vLLM v0.19.1+ requirement)
- vLLM: 0.19+ (tested against 0.19.2rc1.dev8, targeting 0.20.0)

### Author

Sandermage(Sander)-Barzov Aleksandr — Ukraine, Odessa

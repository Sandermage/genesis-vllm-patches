# Genesis vLLM Patches — v7.51

**Runtime patches for [vLLM](https://github.com/vllm-project/vllm) — long-context Qwen3-class inference on NVIDIA Ampere, with TurboQuant k8v4 KV-cache and 256k context.**

> **Production-validated stack (v7.48, 2026-04-27): 160-190 tok/s MTP / 134 tok/s no-spec async / on 2× RTX A5000 (Ampere SM 8.6, 48GB VRAM total).** Long-context corrected — 16K to 200K all PASS needle test, 240K processed without OOM. Zero hallucinations / cascades / garbage in 30/31 harness tests.

## Tested versions (v7.48 baseline)

| Component | Version |
|---|---|
| **Genesis patcher** | `v7.48` |
| **vLLM** | `0.19.2rc1.dev212+g8cd174fa3` (image `vllm/vllm-openai:nightly` ID `10c7a6ba51c6`) |
| **PyTorch** | `2.11.0+cu130` |
| **Triton** | `3.6.0` |
| **CUDA** | `13.0` |
| **NVIDIA driver** | `≥ 580.126.09` ⚠️ **REQUIRED** (driver 570 puts PyTorch in compat fallback ≈ 3× slower decode) |
| **Hardware** | 2× NVIDIA RTX A5000 (Ampere SM 8.6), TP=2 |
| **Model** | Qwen3.6-35B-A3B-FP8 + TurboQuant k8v4 KV cache |
| **OS** | Ubuntu 24.04.4 LTS, kernel 6.8 |

## What's new in v7.48 (2026-04-27 — memory shared-pool sprint + driver 580 + P81)

After upgrading host NVIDIA driver from `570` → `580.126.09` (required because vLLM nightly bumped PyTorch to `2.11+cu130`), three changes landed:

1. **P81 backport of [vllm#40925](https://github.com/vllm-project/vllm/pull/40925)** — `w8a8_triton_block_scaled_mm` low-M (M≤8) decode tuning: `BLOCK_SIZE_M` 64→16, `num_stages` 2→3. Direct hit for Qwen3.6-A3B FP8 + max_num_seqs=2 (M=1 typical, M=4 for MTP K=3 verify). Empirical +23% median decode on GB10 (per upstream PR).

2. **`vllm/_genesis/buffer_mode.py`** — env-driven toggle `GENESIS_BUFFER_MODE=shared|per_layer` (default `shared`) + per-patch override. Makes shared singleton vs legacy per-layer attribute path operator-controllable.

3. **P38/P40 shared-pool fix** — both had per-call `torch.empty` fallback paths that defeated singleton intent on long-context (P38 at growth boundary, P40 when `buf_holder` not pre-attached). Fixed via `GenesisPreallocBuffer.get_or_create()` with **single max-size namespace per (Hk, D, dtype, device)** signature — one buffer reused across all 36 attention layers via slicing.

| Metric | v7.13 (driver 570 + per-layer) | v7.48 (driver 580 + shared P38/P40 + P81) |
|---|---|---|
| Throughput mean | 130-143 tok/s | **160-190 tok/s** (+15-30%) |
| Quality 30-shot | 30/31 PASS | 30/31 PASS |
| Tool-call regression | 2/2 PASS | 2/2 PASS |
| Long-ctx 16K-160K | PASS | PASS |
| Long-ctx 200K | OOM | **PASS** (153K server tokens) |
| GMU at which 200K runs | 0.91 (limit) | **0.90** (Sander obligatory range MET) |

Other patches audited (P22/P26/P28/P36/P37/P39/P44/P46) — all already use shared singleton through `TurboQuantBufferManager` / `GenesisPreallocBuffer` / `gdn_core_attn_manager` / `FlaKktBufferManager`. The `setattr(layer, ...)` only attaches 36 references to a single registered buffer, not duplicated allocations.

See [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) v7.48 entry for full changelog.

---

## v7.45 (2026-04-26 — gemini bot review fix)

After opening upstream draft PR [vllm#40914](https://github.com/vllm-project/vllm/pull/40914) for the K+1 spec-verify routing fix, **gemini-code-assist** flagged a critical issue in code review: the new routing path was not forwarding the cached decode buffers (`mid_o_buf`, `lse_buf`, `buf_holder=layer`) to `triton_turboquant_decode_attention`. Without these, the kernel allocates fresh tensors on every call — defeating the very cudagraph replay this PR aims to restore.

The bot was right and the catch was important — `_decode_attention` next door already does this pattern correctly; we just hadn't matched the full call signature when copying the dispatch.

**Fix applied** (commit [`aec8535`](https://github.com/Sandermage/genesis-vllm-patches/commit/aec8535) on `vllm/_genesis/wiring/patch_67b_spec_verify_routing.py`):

- Forward all 5 decode-buffer parameters (`mid_o_buf`, `output_buf`, `lse_buf`, `buf_holder=layer`, `max_num_kv_splits`) to the routing call
- Marker bumped to `v7.45_buf_reuse_fix` to force reapply

**Empirical result** (12 runs, MTP, free-form, 2× A5000 + Qwen3.6-35B-A3B-FP8):

| Config | Mean tok/s | std | CV | max |
|---|---|---|---|---|
| v7.40 baseline | 128.3 | 7.3 | 5.7% | 139 |
| v7.42 full-stack (pre-fix) | 127.09 | 8.37 | 6.6% | 140 |
| **v7.45 (with buf-reuse fix)** | **130.68** | **6.59** | **5.0%** | **141** |

**+2.6% TPS over baseline + lowest CV** (5.0%) measured across the entire test cycle. Long-context still 4/4 PASS (180K, 216K, 237K, 252K). Tool-call 3/3 PASS. Quality and multi-needle results match the established baseline noise floor (model behavior, not regressions).

This is a textbook example of why opening Draft PRs early is valuable — external review (even from a bot) caught a meaningful issue + the fix made the code both more correct AND faster.

---

## What's new in v7.42-v7.43 (2026-04-26 — almost 24-hour push)

This release closes the spec-decode method comparison loop and adds 4 new patches addressing concrete production gaps. After **methodical empirical comparison of all 4 vLLM-supported speculative-decode methods** on our hardware, deep audit of upstream PRs / forks / academic papers, and reverse-engineering of two distinct buffer-overflow bug classes — we ship:

| Patch | Source | Effect | Status |
|---|---|---|---|
| **P71** | Backport of [vllm#40819](https://github.com/vllm-project/vllm/pull/40819) (Z. Golpayegani draft) + Sun et al. arXiv 2403.10444 + 2 critical bug-fixes from gemini-code-assist review | Block-verify rejection sampler. Strictly ≥ per-token rule (Sun 2024 §4 theorem). Bug-fixes: SHARED `u` per request (PR used per-position), `denom==0 → 1.0` ACCEPT (PR returned 0.0 — rejected perfect drafts). | opt-in (default OFF), MTP-only |
| **P72** | Genesis-original | Cap `profile_run` M to 4096 → unblocks `--max-num-batched-tokens > 4096` on MoE (Dynamo fake-tensor mismatch in moe_align symbolic shape). | opt-in |
| **P73** | Genesis-original | Central `prealloc_budget.py` resolver. Wired into P28 / P26 / P37 / P44 — eliminates 4 hardcoded `4096` constants that caused regression on chunk > 4096. | library |
| **P74** | Genesis-original (companion to P72) | Auto-set `SchedulerConfig.long_prefill_token_threshold = GENESIS_PREALLOC_TOKEN_BUDGET` so prefill chunks never exceed prealloc budget. **Zero VRAM cost** — decodes still get full multi-seq parallelism. | opt-in |
| **P75** | Genesis-enabler of [vllm#25784](https://github.com/vllm-project/vllm/pull/25784) (Arctic Inference Suffix Decoding, MERGED 2025-11-03, present in pin) | Auto-swap `method=ngram → method=suffix` when `GENESIS_ENABLE_P75_SUFFIX_DECODING=1`. Per arXiv 2411.04975: dynamic K speculation per step, per-prompt suffix tree. **Empirically: +32% over historic ngram on tool-call workload (max 175 tok/s).** | opt-in (requires `arctic-inference`) |
| **P77** | Genesis-original (port of SGLang `adaptive_spec_params.py` Apache-2.0 + Nightjar arXiv 2512.22420 auto-disable extension) | Adaptive ngram K controller with EMA + hysteresis + auto-disable to K=0 on `accept_rate < 30%`. Fixes free-form ngram pathology where K=3 wastes 4 forward passes per accepted token. | opt-in |

### Empirical comparison of all spec-decode methods (Qwen3.6-35B-A3B-FP8, 2× A5000)

12-run benchmarks per config, mean ± std:

| Method | Free-form (tok/s) | Tool-call (tok/s) | Quality | CV | Verdict |
|---|---|---|---|---|---|
| **MTP (default)** | **127.0** | (not tested) | 5/5 PASS | 4.7% | **BEST overall, prod default** |
| ngram CPU (numba) | 46.6 | 75 (v7.13 historic) | 5/5 PASS | 4.4% | Fallback |
| ngram_gpu | 43.6 | 45.4 | 4/5 PASS | 3-13% | **NO GAIN** (V1 stale-data residual; PR #40704 V2 rewrite needed for high-conc; dtype bug #37150 already fixed via PR #37246 in our pin) |
| **suffix decoding (P75)** | 45.9 | **99.0 mean (max 175!)** | 4/5 PASS | 16-36% | **WIN for tool-call workload (+32%)** |

### Long-context verification (P74 chunk-clamp, batched=8192)

| Context (tokens) | Result | Latency |
|---|---|---|
| 14,640 | PASS | 2.5s |
| 73,040 | PASS | 9.4s |
| 146,040 | PASS | 23.2s |
| 180,037 | PASS | 49s (cold) |
| 216,037 | PASS | 17.4s (warm prefix-cache) |
| **252,037** | **PASS** | **10.5s ← max stable** |
| 266,050 | HTTP 400 (max_model_len=262144 cap, enforced cleanly — no OOM) |

### Story log — what those 24 hours produced

This release is the result of a near-continuous research+coding session that started from one user question ("can we improve ngram speed?") and ended with a fully-validated multi-method spec-decode stack. The chronology:

1. **Discovery phase** — confirmed all 4 vLLM spec-decode methods (mtp / ngram / ngram_gpu / suffix) end-to-end on our hardware. Burned the myth that `ngram_gpu` is broken in our pin (it's not — bug #37150 was fixed via PR #37246 merged 2026-03-17, our pin is from 2026-04-25). The remaining `ngram_gpu` underperformance is V1 stale-data race (PR #40704 V2 rewrite, not text-patchable today).
2. **Crash investigation** — discovered P28 `core_attn_out` buffer overflow when `--max-num-batched-tokens > 4096` (P72 unblock exposed it on long-context). Root-caused as 4 hardcoded `4096` constants across our prealloc patches (P28/P26/P37/P44). Built P73 central resolver + P74 auto chunk-clamp to fix without VRAM cost.
3. **PR backport phase** — implemented vllm#40819 (block-verify) as P71 with **2 gemini-flagged critical bugs fixed before port** (shared u, denom==0 acceptance). Avoided the regression trap of porting broken code.
4. **Method comparison phase** — empirically benchmarked all 4 methods on free-form + tool-call workloads. Discovered Suffix Decoding wins decisively on tool-call (+32% over ngram historic, peak 175 tok/s).
5. **Adaptive control phase** — built P77 controller (port of SGLang EMA + hysteresis logic + Nightjar auto-disable) as backstop for ngram-only deployments where MTP isn't available.
6. **Cross-rig validation** — analysed [@noonghunna/qwen36-dual-3090](https://github.com/noonghunna/qwen36-dual-3090) (RTX 3090 cousin rig) for cross-validation; identified `patch_tolist_cudagraph.py` as a candidate complementary patch (P78, with attribution).

The full discovery loop — research → bug → fix → benchmark → rinse — happened in a tight cycle. Every patch landed has empirical evidence behind it, every regression was caught **before** it would have hit prod (P28 overflow regression caught by long-context test gate). The discipline rule "push to private repo BEFORE every prod deploy" held for all v7.42-v7.43 changes (commits `c2beaf0` v7.42, `9416c77` v7.43 on `Sandermage/p67-genesis-kernel`).

### Final production recommendation

```bash
# Free-form / general-purpose (most users):
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
# 127 tok/s mean

# Tool-call / agentic-heavy workload:
GENESIS_ENABLE_P75_SUFFIX_DECODING=1 \
--speculative-config '{"method":"ngram","num_speculative_tokens":3}'  # P75 auto-swaps to suffix
# 99 tok/s mean, peak 175

# Ngram-only deployment (no MTP available) — opt-in adaptive K:
GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K=1 \
--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_min":8}'
# Auto-tunes K based on workload acceptance rate
```

---

## P67/P67b — the kernel in depth

P67 (and its companion routing patch P67b) is the central performance + correctness contribution of Genesis. This section explains the **what / why / how** because the bare patch entry above doesn't do it justice.

### What it fixes (the bug)

[vllm#40880](https://github.com/vllm-project/vllm/issues/40880), reported by @noonghunna, observed degenerate output (`<tool_call>\n<tool_call>...` cascades) on Qwen3.6-MoE under MTP num_speculative_tokens=3 + FULL_AND_PIECEWISE cudagraph + TurboQuant k8v4 KV cache.

Root cause we identified:
1. MTP verify pass produces **uniform-query batches** with `max_query_len = K+1 = 4` (one verify batch per request)
2. Each request has prior cached KV in TurboQuant compressed form (`max_seq_len > max_query_len`)
3. The default `_prefill_attention` continuation branch in vLLM reads `query_start_loc.tolist()` — a GPU→CPU sync **incompatible with active CUDA stream capture**
4. To work around the sync issue, vLLM previously took a "first-chunk" path that used `cu_seqlens_k = cu_seqlens_q` — meaning the captured kernel attended to **only the current chunk's K/V**, ignoring all prior cached KV
5. Drafter and verifier both then converge on the same high-bias special token (e.g. `<tool_call>`) because both see "no context" — output cascades

### What it does (the fix)

P67/P67b adds a **dispatch branch in `TurboQuantAttentionImpl.forward()`** that:

1. **Detects** uniform K+1 spec-verify batches via the predicate:
   - `is_prefill = True`
   - `num_decodes = 0`
   - `1 < max_query_len ≤ 16`
   - `max_seq_len > max_query_len` (have prior cache)
   - `N (total tokens) % max_query_len == 0` (uniform batch)
2. **Routes** those batches through `triton_turboquant_decode_attention` via the same `synth_seq_lens` trick `_continuation_prefill` uses internally:
   ```
   synth_seq_lens[req*K1+i] = base_seq_lens[req] - K1 + 1 + i
   synth_block_table[req*K1+i] = block_table[req]
   ```
   All tensor ops on GPU — cudagraph-safe.
3. **Falls through** to upstream code on any non-eligible batch (zero overhead at inference for non-spec-decode workloads)

Two implementation variants ship:
- **P67** (`vllm/_genesis/kernels/p67_multi_query_kernel.py` ~470 LOC): Genesis-original Triton kernel that handles K+1 verify directly. Faster on small K1 + short prior cache; can drift on long context (>4096 prior).
- **P67b** (`vllm/_genesis/wiring/patch_67b_spec_verify_routing.py`): Thin routing wrapper that uses upstream's proven `triton_turboquant_decode_attention`. Drift-free, slightly slower on small K1. **This is what we recommend in production** (set `GENESIS_P67_USE_UPSTREAM=1`).

### Why this is the proper fix vs the prior workarounds

| Approach | TPS | Correctness | CG mode |
|---|---|---|---|
| `cudagraph_mode=NONE` (no spec-decode CG) | -30% | OK | None |
| Genesis P65 (PIECEWISE downgrade) | -20% (vs full CG potential) | OK | PIECEWISE |
| @noonghunna `patch_tolist_cudagraph.py` | OK at inference | safe at capture | FULL_AND_PIECEWISE |
| **Genesis P67/P67b** | **baseline + 32%** | OK + verified clean | **FULL_AND_PIECEWISE** |

P67/P67b restores `FULL_AND_PIECEWISE` capture for spec-decode (the fastest mode) while fixing the correctness bug. Empirical: 75.6 tok/s vs 57.2 tok/s baseline-with-P65 on Qwen3.6-35B-A3B-FP8 + MTP=3 + 2× A5000.

### How to enable

```bash
GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1   # master switch (enables both P67 + P67b wiring)
GENESIS_P67_USE_UPSTREAM=1                   # use upstream kernel via P67b routing (recommended)
GENESIS_P67_NUM_KV_SPLITS=32                 # multi-CTA parallelism for upstream kernel
GENESIS_P67_MAX_PRIOR_LEN=4096               # P67-only: threshold above which fall through to upstream (drift safety)
```

When `P67_USE_UPSTREAM=1` (default), only the routing logic fires — P67's custom Triton kernel is dormant, used only as a fallback if the upstream call raises.

### Upstream PR

The conservative routing-only version (P67b approach) was submitted as draft PR upstream:
**[vllm#40914 — TurboQuant K+1 spec-verify routing (fixes #40880)](https://github.com/vllm-project/vllm/pull/40914)** (Draft, awaiting cross-arch validation).

If/when merged, our P67/P67b will auto-no-op via drift markers and operators can drop the env flags.

### Full kernel + wiring source

- `vllm/_genesis/kernels/p67_multi_query_kernel.py` — Triton kernel + Python wrapper
- `vllm/_genesis/wiring/patch_67_tq_multi_query_kernel.py` — text-patch installing the wrapper hook
- `vllm/_genesis/wiring/patch_67b_spec_verify_routing.py` — text-patch installing the upstream routing branch in `TurboQuantAttentionImpl.forward()`

---

## Tests

The test suite lives in `vllm/_genesis/tests/`:

| Test file | What it verifies | How to run |
|---|---|---|
| `test_p71_block_verify.py` | P71 bug-fixes from gemini review (shared `u`, `denom==0 → 1.0`), Triton-PyTorch parity, greedy short-circuit, smoke test for unbiasedness | `pytest vllm/_genesis/tests/test_p71_block_verify.py -v` (requires CUDA) |
| `test_gdn_core_attn_manager.py` | P28 GDN buffer registry — `should_apply` per arch, env override, get-or-create caching, multi-shape registry | `pytest vllm/_genesis/tests/test_gdn_core_attn_manager.py` (CPU-only) |
| `test_config_detect.py` | Patch dispatcher decisions per model architecture (P51/P52/P53 logic) | `pytest vllm/_genesis/tests/test_config_detect.py` (CPU-only) |
| `test_response_cache.py` | P41/P50 ASGI middleware response cache | `pytest vllm/_genesis/tests/test_response_cache.py` |
| `test_v7_14_15_audit.py` | Anchor-presence audit for v7.14/v7.15 patches + drift-safety + dispatcher↔apply_all consistency | `pytest vllm/_genesis/tests/test_v7_14_15_audit.py` (some require live vLLM source) |

### Empirical / integration tests

These run end-to-end against a live vLLM server:

| Script | What it tests | Hardware needed |
|---|---|---|
| `scripts/genesis_bench_v3.py` | Throughput benchmark (mean + std over N runs, multiple prompts, MTP/ngram/suffix variants) | GPU + booted server |
| `scripts/genesis_quality_harness.py` | Quality regression: cascades / garbage / repetition detection over N prompts | GPU + booted server |
| `scripts/genesis_longbench_runner.py` | Long-context recall (single needle + multi-needle, ladder from 1K to 250K tokens) | GPU + booted server |
| `scripts/genesis_context_sweep.py` | Context-window sweep with bisection to find OOM threshold | GPU + booted server |

Validation matrix run for the v7.42-v7.43 release (per the empirical table in the v7.42 section above):

- **Quality**: 5/5 PASS on free-form (`quality_check.sh`)
- **Tool-call**: 3/3 PASS (`tool_call_check.sh`) on MTP, P75, baseline ngram
- **Long-context recall**: PASS at 14K, 73K, 146K, 216K, **252K** tokens
- **Stability**: CV 4.7% over 12 runs (best config: P67b + MTP)
- **Boot crashes**: 0 across all method variants
- **Hidden runtime errors**: 0 in container logs

### Running tests against a live server

The test container helper scripts on the prod server:

```bash
# Free-form throughput
/tmp/bench_compare.sh 8000 12 my-bench-name

# Tool-call quality
/tmp/quality_check.sh 8000

# Long-context (single 6-digit needle, max_tokens cap to avoid thinking eat-up)
python3 /tmp/max_ctx_probe.py
```

Their bash sources are reproducible from the public `benchmarks/` dir or the repo root scripts.

---

## Older changelog

> 34+ active patches (P64-P70 added across v7.14/v7.15, P63 deprecated). Zero vLLM source modifications beyond container startup. Stability matrix verified to GMU 0.93 with full patch set.
> Defense-in-depth interface guards (P49). ASGI response-cache middleware (P50).
> Runtime architecture-dispatch detection (P51 / P52 / P53 — v7.9).
> Cross-model validated on FP8 / AWQ 4-bit / FP16-KV configurations.
>
> ⚠️ **Test matrix is 2× RTX A5000 (Ampere SM 8.6) + NVIDIA drivers 570+.**
> Patches are written defensively for AMD ROCm / Intel XPU / Hopper / Blackwell / FP8-native paths — they graceful-skip on platform mismatch rather than crash.
> Bug reports from other hardware are very welcome.
> Optional support / hardware sponsorship — see [SPONSORS.md](SPONSORS.md).

---

## What's new in v7.15 (2026-04-25 late evening)

**P65 v2 refactor + P70 auto-strict-ngram + audit suite + GMU stability matrix verified to 0.93.**

Tightening pass on top of v7.14: P65 was reworked from a blunt ClassVar override into a context-aware classmethod that only downgrades cudagraph support when speculative_config is active (no longer penalizes operators who don't use spec-decode). P70 added as engine-level enforcement of the empirical `prompt_lookup_min ≥ 8` finding from #40875. Audit test suite added with anchor-presence checks for all v7.14/v7.15 new patches plus drift-safety checks for legacy P14/P28/P38/P39a. Full GMU stability sweep run on production-mirror test container (256K context, warm compile cache) — verified safe to GMU 0.93 with the full patch set; GMU 0.94 OOMs due to extra workspace footprint from P64-P70.

| New patch | Source | Files | Effect |
|---|---|---|---|
| **P65 v2** | Genesis-original (refactor of v7.14 P65) | turboquant_attn.py | Replaces blunt ClassVar override with context-aware `get_cudagraph_support` classmethod. For non-spec-decode setups: keeps `UNIFORM_BATCH` (full caps). For spec-decode setups: returns `UNIFORM_SINGLE_TOKEN_DECODE` (downgrade only when needed). Same effective behavior for spec-decode prod, but avoids unnecessary downgrade for users without spec-decode. |
| **P70** | Genesis-original (mirror vllm#40875 enforcement) | config/speculative.py | Auto-bump ngram `prompt_lookup_min` and `prompt_lookup_max` to ≥8 when method is "ngram"/"ngram_gpu" and env `GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1`. Engine-level (per-request override is not architecturally possible — `speculative_config` is engine-level). |
| **test_v7_14_15_audit.py** | Genesis-internal | tests/ | Anchor-presence checks for all v7.14/v7.15 new patches + drift-safety checks for legacy P14/P28/P38/P39a + dispatcher registry consistency + apply_all↔dispatcher consistency. 4/4 dispatcher tests pass; 9 anchor tests skip gracefully when pinned vLLM source is not present in the test environment. |

### v7.15 stability matrix — GMU sweep (2× RTX A5000 24 GB, 256K context, all v7.14+v7.15 patches enabled, ngram strict, warm compile cache)

Tested with full Genesis v7.14+v7.15 patch set: P58/P59 disabled (research artifacts), P60+P60b+P61+P61b+P62+P64+P65v2+P66+P68+P69+P70 enabled, plus all legacy Genesis patches.

| GMU | Status | Boot time | Total GPU used (both cards) | Tool-call test (n=2) |
|---|---|---|---|---|
| 0.90 | ✅ HEALTHY | 126 s | 46,039 MiB | 2/2 |
| 0.91 | ✅ HEALTHY | 126 s | 46,519 MiB | 2/2 |
| 0.915 | ✅ HEALTHY | 125 s | 46,759 MiB | 2/2 |
| 0.92 | ✅ HEALTHY | 126 s | 46,999 MiB | 2/2 |
| 0.925 | ✅ HEALTHY | 126 s | 47,239 MiB | 2/2 |
| 0.93 | ✅ HEALTHY | 126 s | 47,479 MiB | 2/2 |
| 0.94 | ❌ OOM (both cold + warm cache) | 110 s (failure) | — | — |

**Stability ceiling on 2× A5000 with full v7.15 patch set: GMU = 0.93.** OOM at 0.94 due to additional workspace footprint of P64-P70 (long-ctx middleware, capture-size filter, cudagraph downgrade extras).

For comparison, prod stack (without P64-P70 patches enabled — just core v7.13 set) successfully runs at GMU = 0.94. If you need 0.94+ headroom, selectively disable the optional v7.14/v7.15 patches that aren't load-bearing for your workload.

GPU usage scales linearly: ~240 MiB per +0.005 GMU (mostly KV-cache budget growth).

**Sweet spot for full feature set: GMU = 0.92** (1.4 GiB safety margin per card, 38/38 regression suite passes, all v7.14/v7.15 patches active).

---

## What's new in v7.14 (2026-04-25 evening)

**MTP × TurboQuant × FULL cudagraph root cause located + fix landed. Long-context tool-call adherence layer added. 38/38 (100%) regression suite + 18/18 (100%) extended ladder up to 250K characters.**

After @noonghunna opened [vllm#40880](https://github.com/vllm-project/vllm/issues/40880) (a separate bug class from the v7.13 ngram path), Genesis ran a full investigation cycle. Walked back one wrong hypothesis (P63 — MTP layers turned out to use `layer_type="full_attention"`, not GDN, so a GDN-side fix was the wrong layer), then identified the actual root cause: `TurboQuantAttentionImpl._prefill_attention` cudagraph capture bypass treats continuation prefill batches (q_len < seq_len) as first-chunk prefill (`cu_seqlens_k = cu_seqlens_q`), so the captured kernel attends only to the current chunk and ignores all prior cached KV. Drafter and verifier converge on the same high-bias special token (`<tool_call>`) → cascade output.

| New patch | Source | Author | Files | Effect |
|---|---|---|---|---|
| **P64** | [vllm#39598](https://github.com/vllm-project/vllm/pull/39598) backport | **kotori-yan** | qwen3coder_tool_parser.py + serving.py | Streaming tool-call early-return removal — fixes empty `tool_calls` when MTP/spec-decode bundles last parameter + `</function>` in single delta. Plus widens safety-net trigger condition. |
| **P65** | Genesis-original (root cause for [#40880](https://github.com/vllm-project/vllm/issues/40880)) | Genesis | turboquant_attn.py | Downgrade TurboQuant `_cudagraph_support` from `UNIFORM_BATCH` to `UNIFORM_SINGLE_TOKEN_DECODE` when speculative_config is active. vLLM auto-flips `cudagraph_mode` from `FULL_AND_PIECEWISE` to `PIECEWISE`, so spec-verify K+1 batches fall to eager (correct per-request continuation path). 1-token decode batches retain piecewise capture. |
| **P66** | mirror of [vllm#23679](https://github.com/vllm-project/vllm/pull/23679) (fhl2000, closed/stale) | Genesis | config/vllm.py | Filter `cudagraph_capture_sizes` to sizes divisible by `1 + num_speculative_tokens` when spec-decode is active. Eliminates 12 of 16 wasteful captures (e.g., for MTP n=3: keep `[4, 8, 12, 16]`, drop `[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]`). Boot 2.5× faster (3 min vs 7-8 min on prod). Less peak GPU memory during warmup. |
| **P68** | Genesis-original | Genesis | serving.py + middleware/long_ctx_tool_adherence.py | Auto-upgrade `tool_choice` from `"auto"` to `"required"` when prompt > threshold AND tools provided. Triggers vLLM's tool-format enforcement. |
| **P69** | Genesis-original (informed by [Liu et al. 2023 "Lost in the Middle"](https://arxiv.org/abs/2307.03172)) | Genesis | serving.py + middleware/long_ctx_tool_adherence.py | Append explicit format reminder to last user message when prompt > threshold. Mitigates LLM long-context format-decay (`<tool_call>` markers replaced by JSON-text/refusals/hallucinations). Threshold via `GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS` (default 8000 chars ≈ 2K tokens). |
| **P63 (deprecated)** | Genesis hypothesis disproven 2026-04-25 | Genesis | gdn_attn.py (kept in tree for archival) | Originally hypothesized that MTP drafter forward needs `spec_decode_src_indices` recovery via `GDNAttentionMetadataBuilder.build_for_drafting`. Wrong layer — MTP module uses `layer_type="full_attention"`. Marked `deprecated:True` in the registry; never fires for MTP runtime. |

### Empirical results — Genesis test rig (Qwen3.6-35B-A3B-FP8, 2× A5000, TurboQuant k8v4)

**7-test regression suite** (short tool-call, multi-tool, plain chat, needle, long+tool, streaming, burst):

| Config | Pass rate |
|---|---|
| baseline (no v7.14 patches, MTP n=3) | 5/38 (13%) |
| + P64 + P65 + P66 | 37/38 (97%) |
| **+ P64 + P65 + P66 + P68 + P69** | **38/38 (100%)** |

**Extended long-context ladder** (1K → 250K characters, MTP n=3 + tool-call):

| Prompt size (chars) | Approx tokens | Pass rate | Generation TPS |
|---:|---:|---:|---:|
| 1,000 | 193 | 2/2 | 43.6 |
| 4,000 | 700 | 2/2 | 43.6 |
| 16,000 | 2,740 | 2/2 | 41.1 |
| 32,000 | 5,460 | 2/2 | 37.2 |
| 50,000 | 9,162 | 2/2 | 28.6 |
| 64,000 | 10,900 | 2/2 | 35.8 |
| 100,000 | 17,945 | 2/2 | 18.3 |
| 128,000 | 21,780 | 2/2 | 30.9 |
| 150,000 | 26,729 | 2/2 | 17.1 |
| 200,000 | 34,020 | 2/2 | 29.0 |
| 250,000 | 44,297 | 2/2 | 14.3 |

**18/18 PASS** at every tested size from 1K to 250K characters. Throughput degradation is healthy attention-scaling (~30% drop from peak 43.6 → 30 TPS over the 200× context expansion, with second-run speedups from prefix-caching). All within `--max-model-len 65536` window of the test container; full 256K context window in production retains the same scaling.

### Boot-time impact

| Config | Boot to ready |
|---|---|
| Without P66 (16 capture sizes) | ~7-8 minutes |
| **With P66 (4 divisible capture sizes)** | **~3 minutes** (2.5× faster) |

### Walked-back hypothesis: P63

P63 was drafted on the assumption that the MTP drafter forward path goes through `GDNAttentionMetadataBuilder.build_for_drafting()` and needs an analogous `spec_decode_src_indices` recovery to what [#40738](https://github.com/vllm-project/vllm/pull/40738) fixed for the main-model decode path. After implementing and testing P63 (and a more aggressive P63b variant that always passes a synthetic `num_accepted_tokens`), neither helped — both stayed at 0/38 on the tool-call suite. Reading `vllm/model_executor/models/qwen3_5_mtp.py:97-104` confirmed: MTP layers are constructed with `layer_type="full_attention"` (Qwen3NextAttention), not `linear_attention` (GatedDeltaNet). DEBUG trace confirmed: P63's `build_for_drafting` log line never fires per request. P63 stays in the tree as a research artifact (`deprecated: True` in the dispatcher registry) but is a no-op for MTP setups. May still be relevant for hypothetical eagle/draft_model setups that use a separate drafter model with hybrid layers — none verified.

### What v7.14 does NOT cover

- **Custom multi-query TurboQuant kernel for spec-decode (P67)** — design documented in `Genesis_Doc/P67_KERNEL_DESIGN_RU.md`, not implemented. Would extend the P40 grouped decode kernel from `[Hq, D]` per-request to `[K+1, Hq, D]` per-request, allowing FULL cudagraph for spec-verify batches. Substantial Triton work (10-15 hours, needs GPU dev cycle); deferred. Without P67, P65 forces spec-verify to PIECEWISE which is a workaround not a proper fix.
- **Long-context tool-call format adherence is a model-level limitation**, not an engine bug (verified by plain text generation test 3/3 OK at same context where tool-call test was 0/3). P68+P69 mitigate the symptom; the underlying model-format-decay issue remains. Other Qwen3-class models or future model versions may have improved format adherence at long context.

### Production guidance

For Qwen3-Next family + MTP + TurboQuant + tool calls:

1. Apply P64 + P65 + P66 (engine-level fixes) — required for correctness with MTP
2. Apply P68 + P69 if you need long-context (>2K token) tool-call workflows — model adherence layer
3. Default thresholds work for our prod (Qwen3.6-35B-A3B-FP8, 2× A5000); tune `GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS` for your setup
4. P58 / P59 / P63 stay opt-in or `deprecated:True` — research artifacts only

For Qwen3-Next family + ngram (no MTP) + TurboQuant + tool calls:

The v7.13 strict-ngram config (`prompt_lookup_min=8`) plus P64/P65/P66 still gives 100% clean tool-call rate on short prompts. P68+P69 recommended if your prompts include long context with tools. P65 is a no-op for ngram (no spec-verify K+1 batches) but the cudagraph_mode auto-downgrade is harmless.

### Cross-rig / cross-model notes

- Independent multi-rig confirmation of the v7.13 + #40875 strict-ngram config came from [@noonghunna's Probe 9](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4319911691) on Qwen3.6-27B-int4-AutoRound + 1× RTX 3090 + `turboquant_3bit_nc`. The v7.14 P65 root cause and patch tree are from Genesis rig (35B + 2× A5000 + k8v4); empirical confirmation on another rig is welcome.

---

## What's new in v7.13 (2026-04-25)

**Spec-decode + structured output bug class — root cause located + multi-layer fix backported.**

After deep investigation of the noonghunna #40831 + #40807 thread + parallel community reports (#39273 / #34650 / #36138), located the actual root cause class for tool-call output corruption (`<<`, `parameter=parameter`, `<argname>` patterns) on hybrid GDN models with ngram speculative decoding.

| New patch | Source | Author | Files | Effect |
|---|---|---|---|---|
| **P58** | vllm#40768 backport | @z1ying | scheduler.py + async_scheduler.py + request.py | Async-scheduler `-1` placeholder gating (opt-in research artifact; empirically NOT our bug for ngram method since vLLM auto-disables async_scheduling) |
| **P59** | vllm#39055 backport | @ZenoAFfectionate | qwen3_reasoning_parser.py | Promotes `<tool_call>` XML out of `<think>` reasoning into content (opt-in research) |
| **P60** | vllm#40738 backport (Phase 1) | @tdoublep, @bhaktatejas922 (#39273) | gdn_attn.py + gdn_linear_attn.py + gpu_model_runner.py | SSM state pre-copy from accepted block. **+23-40% clean tool-call rate** (43-60% vs 20% baseline) |
| **P60b** | vllm#40738 backport (Phase 2) | @tdoublep | causal_conv1d.py (Triton) + gdn_linear_attn.py | Conv state Triton kernel offset. **Combined with P60: 70% clean** (3.5× baseline) |
| **P61** | vllm#40783 backport (slice) | @ExtReMLapin | qwen3_reasoning_parser.py | Multi-tool first-occurrence fix — preserves all tool calls in agentic flows |
| **P61b** | vllm#40783 backport (streaming slice) | @ExtReMLapin | qwen3_reasoning_parser.py | Defensive overlap guard for partial `<tool_call>` tags during streaming |
| **P62** | vllm#36138 backport | @sfbemerk, @cicirori (#34650) | structured_output/__init__.py + scheduler.py | Reasoning-aware grammar acceptance — fixes grammar bypass when `</think>` arrives in spec-decode batch |

### Empirical results (2026-04-25)

| Config | Clean rate | Notes |
|---|---|---|
| baseline (no Genesis) | ~20% | bug pattern present |
| Genesis v7.11 (P56 only) | ~20% | P56 was workaround attempt — disproven |
| + P58 | ~20% | wrong code path (ngram → async_scheduling auto-disabled) |
| + P59 | 40% | helps when `<tool_call>` nested inside `<think>` |
| + cudagraph_mode=NONE | 40% | partial, tradeoff: ~25% TPS loss |
| **+ P60** | **43-60%** | SSM pre-copy main fix |
| **+ P60 + P60b** | **70%** (test) / **53%** (prod sample) | Triton kernel offset complete fix |
| + P60 + P60b + P61 | 50% (prod) | P61 = multi-tool fix, marginal on single-tool reproducer |
| **+ P60 + P60b + P61 + P61b + P62** | **53-56%** (prod) | Full v7.13 — best reachable with current patches |
| no spec-decode | 100% | reference |

### v7.13 BREAKTHROUGH (2026-04-25 evening) — config-only fix achieves 100% clean

After deep token-level investigation, **the structural ceiling formula was identified**:

```
clean_rate ≈ (per_token_accept_rate)^num_speculative_tokens
0.8^3 ≈ 51%   ← matches our 53% empirical measurement with default ngram
0.8^1 = 80%   ← matches 65% (n=20, statistical variance)
```

The mechanism: ngram with default `prompt_lookup_min=2` finds spurious 2-token suffix matches in the system prompt's tool definitions, drafting wrong tokens that get accepted by the rejection sampler.

**Config-level fix (no code changes):**

```bash
--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":10,"prompt_lookup_min":8}'
```

Setting `prompt_lookup_min=8` requires ngram to find an 8-token suffix match — this almost never matches spurious system-prompt template fragments, only true natural repetitions. Result: ngram drafts almost nothing on tool-call requests, close to no-spec correctness with whatever speedup is achievable on natural repetitions.

**Empirical results with strict ngram config:**

| Test | Clean rate | n |
|---|---|---|
| Single-query Tokyo | **100%** | 30/30 |
| Multi-query diverse | **96%** | 24/25 |

This is the working production configuration as of v7.13.

#### Independent multi-rig confirmation + scope notes (post-deploy 2026-04-25)

After we deployed v7.13 + opened upstream bug report [vllm#40875](https://github.com/vllm-project/vllm/issues/40875), @noonghunna independently re-tested the v7.13 patch tree on a different rig + different model family member: **Qwen3.6-27B dense hybrid (Lorbus int4-AutoRound) on 1× RTX 3090 with `turboquant_3bit_nc` KV**. Detailed Probe 9 results in [vllm#40831 thread](https://github.com/vllm-project/vllm/issues/40831). Net findings:

| Spec method (with v7.13 backports + cudagraph ON) | Status |
|---|---|
| ngram + `prompt_lookup_min=8` | ✓ all 7 tests clean (independent confirmation of #40875) |
| MTP n=3 | ✗ tool calls still empty + first-token truncation at 10K — **separate bug class not covered by v7.13** |

So v7.13 cleanly closes the **ngram path** for Qwen3-Next family models when `prompt_lookup_min=8` is set. The **MTP path** remains an open bug class and will be tracked in a separate upstream issue (to be opened by the contributor with the active reproducer rig).

#### Memory footprint note (when adopting v7.13)

Genesis v7.13 added several pre-allocation patches (P28 GDN core_attn_out + P38 TQ continuation_prefill workspace + P39a FLA persistent A pool, primarily) which expand the steady-state workspace footprint. Pre-v7.13 setups using `gpu_memory_utilization=0.95+` may need to **drop GMU by 1-2 percentage points** to leave room for the new workspaces — otherwise long-context sessions (≥10K tokens) can hit OOM in the KV budget that previously fit.

Specific tuning numbers depend on `max_num_seqs × num_speculative_tokens × layer_count`; we have not benchmarked exact values yet. If you hit OOM after upgrading to v7.13, dropping GMU from 0.97 → 0.94-0.95 is a safe first cut.



### Genesis Dispatcher v2

New `vllm/_genesis/dispatcher.py` provides:

- `should_apply(patch_id) -> (bool, reason)` — unified gate combining env-flag + `config_detect.recommend()` + `model_detect`
- `dump_apply_matrix() / log_apply_matrix()` — single condensed startup summary instead of scattered INFO lines
- CLI: `python3 -m vllm._genesis.dispatcher` — diagnostic table

Built on the existing `model_detect.py` (P52/P53 dispatch) + `config_detect.py` (v7.12 runtime probes) layers. Adds P58/P59/P60/P60b/P61 to the recommendation registry.

### Pin bump

| Component | v7.9 | v7.13 |
|---|---|---|
| Test base | `dev134+gfe9c3d6c5` (2026-04-23) | `dev205+g07351e088` (2026-04-25) |
| Prod | `dev8+g4b7f5ea1a` (2026-04-19, legacy v5.12 monolith) | `dev205+g07351e088` (2026-04-25, modular `_genesis/` plugin) |
| Patcher | v5.12 monolith | Modular plugin + Dispatcher v2 |

## What's new in v7.9 (2026-04-24)

**Runtime architecture-dispatch detection — three defense-in-depth guards** that answer "did this patch even need to fire?" before doing any work.

| Patch | Role | Gate |
|---|---|---|
| **P51 — TQ-active runtime detection** | `ensure_turboquant_buffers` | skip preallocs if `impl.kv_cache_dtype != turboquant_*` |
| **P52 — MoE-active detection** | P24 / P31 / P37 apply() | skip on dense models (no `num_experts` / no `*MoE*` arch) |
| **P53 — Hybrid-active detection** | P28 / P34 / P39a / P46 apply() | skip on pure-attention models (no Mamba / GDN / linear-attn) |

The central `vllm/_genesis/model_detect.py` probes `hf_config` once per process and returns a cached profile `(moe, hybrid, turboquant)`. Each patch's `apply()` consults it and either engages or logs a clean single-line skip. Conservative fallback: unknown or unavailable config → True for all flags (patch still attempts; its own call-site guards decide).

**Cross-model validation** (Config 1 FP8 prod / Config 2 AWQ 4-bit / Config 3 FP16-KV):
- 28 applied / 4 skipped / 0 failed (identical dispatch across all three)
- 3× 256k stable on FP8 + AWQ
- AWQ frees ~9 GiB/rank → 2.5× more KV capacity (KV tokens 1.099M → 2.787M)
- Speed: AWQ 1–4% slower than FP8 (4-bit dequant cost on SM 8.6)

See [`docs/REPORT_v7_1_INTEGRATION_20260424_RU.md`](docs/REPORT_v7_1_INTEGRATION_20260424_RU.md) §13 (v7.8) + §14 (v7.8.5) + §15 (v7.9) for full figures.

---

## Compatibility matrix

### vLLM version

| Component | Pin | Notes |
|---|---|---|
| Integration baseline | `v0.19.2rc1.dev134+gfe9c3d6c5` | Patches authored against this SHA |
| Upstream branch | `main` | Rebased periodically; all patches graceful-skip on anchor drift |
| Minimum supported | `v0.19.2rc1.dev8+` (P8a) | Older builds: anchors may not match |
| Known-broken | — | No incompatibilities at time of v7.9 release |

### Hardware

| GPU class | SM | FP8 | TurboQuant | Status |
|---|---|---|---|---|
| RTX A5000 / A6000 / 4090 | 8.6 | ✅ via Marlin | ✅ | **tested** (2× A5000 is the prod target) |
| RTX 3090 / 3090 Ti | 8.6 | ✅ via Marlin | ✅ | should work; untested — reports welcome |
| A100 / H100 | 8.0 / 9.0 | ✅ native | ✅ | graceful — patches platform-guard |
| Blackwell (B200) | 10.0+ | ✅ native | ✅ (FA4 path) | graceful — patches platform-guard |
| AMD ROCm CDNA3 | — | — | — | graceful-skip |
| Intel XPU | — | — | — | graceful-skip |
| CPU-only | — | — | — | graceful-skip |

### Models

| Model | Architecture | Status |
|---|---|---|
| Qwen3.6-35B-A3B-FP8 | Qwen3-Next MoE + hybrid linear-attn | **prod** — full 256k |
| Qwen3.6-35B-A3B-AWQ | Qwen3-Next MoE + hybrid linear-attn + 4-bit weights | **validated** — 3× 256k, 2.5× KV capacity vs FP8 |
| Qwen3-32B dense (planned) | Qwen3 dense attention | graceful-skip cross-model test (v7.9) |
| Gemma 4 26B MoE (planned) | Gemma 4 MoE | cross-arch test (v7.9 follow-up) |

---

## Installation / deployment

### Option 1 — docker compose (recommended)

```yaml
# docker-compose.yml (minimal example — see docker-compose.example.yml)
services:
  vllm-server:
    image: vllm/vllm-openai:nightly   # or pinned commit image
    volumes:
      - ./vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro
      - ./docker/genesis-entrypoint.sh:/opt/genesis-entrypoint.sh:ro
    entrypoint: ["/opt/genesis-entrypoint.sh"]
    command: [
      "--model", "Qwen/Qwen3-Next-35B-A3B-FP8",
      "--kv-cache-dtype", "turboquant_k8v4",
      "--max-model-len", "262144",
      "--tensor-parallel-size", "2",
      # ...
    ]
    environment:
      # Default 28-patch set (opt-in features off):
      - GENESIS_ENABLE=1
      # Opt-in features — set to 1 to enable:
      - GENESIS_ENABLE_P5B=0      # Page-size pad-smaller allocator
      - GENESIS_ENABLE_P7B=0      # GDN dual-stream custom-op
      - GENESIS_ENABLE_P37=0      # MoE intermediate-cache text-patch
      - GENESIS_ENABLE_P40=0      # TQ grouped-decode (watch upstream #40792)
      - GENESIS_ENABLE_P41=0      # ResponseCacheMiddleware (needs P50 wired)
      # ResponseCache backend (P50):
      - GENESIS_RESPONSE_CACHE=redis://cache:6379/0
```

The entrypoint runs `python -m vllm._genesis.patches.apply_all` before `vllm serve`. All patches are idempotent per container filesystem layer.

### Option 2 — manual

```bash
# Install vLLM at the integration baseline
pip install vllm==0.19.2rc1.dev134

# Drop the Genesis package into the vLLM install
cp -r vllm/_genesis /path/to/site-packages/vllm/

# Apply patches once
python -m vllm._genesis.patches.apply_all

# Start vLLM normally
vllm serve ...
```

### Verify apply-time dispatch

On a healthy boot you should see lines like:

```
[INFO genesis.apply_all] Genesis v7.9 — 28 applied, 4 skipped, 0 failed
[INFO genesis.model_detect] profile resolved: model_type=qwen3_next_moe moe=True hybrid=True turboquant=True (kv=turboquant_k8v4)
[INFO genesis.wiring.p28_gdn_core_attn] forward_cuda patched + __init__ wrapped
[INFO genesis.prealloc] [P51 TQ-active] ...   (absent on turboquant_k8v4 — P51 only fires when non-TQ)
```

On a dense-model boot:

```
[INFO genesis.model_detect] profile resolved: model_type=qwen3 moe=False hybrid=False turboquant=False (kv=fp16)
[INFO genesis.model_detect] [Genesis v7.9 dispatch] P24 MoE num_warps/num_stages overlay skipped — dense model (no fused_moe dispatch)
[INFO genesis.model_detect] [Genesis v7.9 dispatch] P28 GDN core-attn forward rewire skipped — pure-attention model (no GDN)
# ...etc
```

---

## Patch roster (v7.9)

### 28 patches applied by default

| # | Area | What it does |
|---|---|---|
| P3 | TurboQuant | bf16 cast for Ampere non-FP8 fallback |
| P4 | TurboQuant | Hybrid reporting guard |
| P5 | Block table | Page size reporting |
| P6 | TurboQuant | Block-size align |
| P8 | KV cache | Hybrid reporting |
| P12 | Chat | Tool-call reasoning extraction |
| P14 | Scheduler | Block table zero-fill |
| P15 | Chat | Qwen3 None/null reasoning guard |
| P17 | MoE | Marlin bsm=8 (A5000 SM 8.6) |
| P18 | MoE | Marlin num_warps overlay |
| P22 | TurboQuant | Shared dequant pre-allocation |
| P23 | TurboQuant | cu_seqlens scratch reuse |
| P26 | TurboQuant | Prefill output buffer reuse |
| P27 | Chat | Reasoning-before-think ordering |
| P28 | GDN | Core-attn forward-cuda rewire |
| P31 | MoE | Grouped-topk fp32 softmax upcast |
| P32 | TurboQuant | Second-hop cu_seqlens |
| P33 | TurboQuant | Synthetic seq_lens mirror |
| P34 | Hybrid | Mamba zero-collapse deadlock guard |
| P36 | TurboQuant | Shared decode buffer |
| P38 | TurboQuant | Continuation 4-D K/V memory |
| P39a | GDN | FLA `chunk_scaled_dot_kkt` persistent A pool |
| P42 | Chat | — (reserved) |
| P44 | TurboQuant | Mixed-attn-out capture-safe reuse |
| P46 | GDN | fused_gdn_gating buffer pool |
| **P49** | **Infra** | **Interface-contract validation (v7.8)** |
| **P50** | **Infra** | **ASGI ResponseCacheMiddleware for cliproxyapi (v7.8)** |
| **P51** | **Dispatch** | **TQ-active runtime detection (v7.9)** |
| **P52** | **Dispatch** | **MoE-active dispatch gate (v7.9)** |
| **P53** | **Dispatch** | **Hybrid-active dispatch gate (v7.9)** |

### 4 opt-in patches

| # | Env var | What it enables |
|---|---|---|
| P5b | `GENESIS_ENABLE_P5B=1` | page-size pad-smaller allocator |
| P7b | `GENESIS_ENABLE_P7B=1` | GDN dual-stream custom-op (+8% decode on hybrid) |
| P40 | `GENESIS_ENABLE_P40=1` | TurboQuant grouped-decode (watch upstream #40792) |
| P41 | `GENESIS_ENABLE_P41=1` | ResponseCache + P50 middleware (needs a backend) |

### Retired / shelved

| Status | Patch | Reason |
|---|---|---|
| ❌ retired | P43 → P28 | covered by P28 rewire |
| ❌ retired | P47 → P38 | covered by P38 4-D memory patch |
| ❌ dropped | swap-space | upstream removed the knob |
| ❌ dropped | LMCache | no measurable gain on our workload |
| 💤 shelved | P7 (eager-only) | Torch.compile conflict, deferred |
| 💤 shelved | P41b semantic cache | hallucination risk — **permanently excluded** |
| 💤 shelved | P45 qkv concat | byte-copy stride risk |
| 💤 shelved | P48 conv-view | corruption risk |
| 👁 monitor | P40 retire | upstream PR [#40792](https://github.com/vllm-project/vllm/pull/40792) |
| 👁 monitor | P36 retire | upstream PR [#40798](https://github.com/vllm-project/vllm/pull/40798) |

---

## Testing

Two gates run separately and independently.

### Unit gate (CPU-only, no GPU required)

```bash
cd <repo-root>
./validate_unit.sh
# or:
pytest vllm/_genesis/tests/ -v -m 'not cuda_required and not gpu_required'
```

Full unit suite passes: **605 + 43 (v7.8) + ~40 (v7.9 model_detect + P51) = ~688 tests**. No network, no CUDA.

### Integration gate (GPU required)

```bash
cd <repo-root>
./validate_integration.sh      # brings up docker-compose.integration.yml
# For cross-quantization verification:
docker compose -f docker-compose.integration-awq.yml up -d
docker compose -f docker-compose.integration-fp16kv.yml up -d
```

Integration checks: 3× 256k prefill+decode, speed (t/s @ 100k), KV tokens, all applied/skipped/failed counters.

### Manual smoke (production container)

```bash
# Verify model_detect profile:
docker exec vllm-prod python -c "from vllm._genesis.model_detect import get_model_profile; import json; print(json.dumps(get_model_profile(), indent=2))"

# Verify applied patches:
docker exec vllm-prod python -m vllm._genesis.patches.apply_all --verify
```

---

## Upstream status tracking (as of 2026-04-24)

Items we track / monitor:

| PR / Issue | Status | Impact |
|---|---|---|
| [#40807 — TurboQuant+spec-decode capture crash](https://github.com/vllm-project/vllm/issues/40807) | OPEN | Reporter namechecks Sander's Patch 23; our P44 aligns. Comment drafted, not yet posted. |
| [#40792 — TQ k8v4 GQA head grouping](https://github.com/vllm-project/vllm/pull/40792) | OPEN | May supersede our P40. Diff + bench pending. |
| [#40798 — TQ scratch workspace across layers](https://github.com/vllm-project/vllm/pull/40798) | OPEN | Superset of #40655+#40706. May conflict with P28 anchor. |
| [#40794 — MoE unpad routed output](https://github.com/vllm-project/vllm/pull/40794) | **MERGED 2026-04-24** | Smoke test pending on Qwen3.6-35B-A3B. |
| [#40420 — TurboQuant continuation-prefill OOM at 185K](https://github.com/vllm-project/vllm/issues/40420) | OPEN | Our P22 covers adjacent class. Adding ≥150k regression test. |
| [#40172 — Fused Mamba postprocess (+15-17% decode)](https://github.com/vllm-project/vllm/pull/40172) | OPEN | On merge: drop P25 guard. |
| [#40384 — Exclude O(1) Mamba groups](https://github.com/vllm-project/vllm/pull/40384) | OPEN | Sander co-author credit. On merge: drop Patch 9. |

Three-source truth: every patch is verified against the tagged release, `main` HEAD, and nightly image before each deploy.

---

## Architecture

```
vllm/_genesis/
├── __init__.py               Public API
├── guards.py                 Canonical vendor/chip/model detection
├── interface_guard.py        v7.8 — GenesisInterfaceMismatch + validate_impl
├── model_detect.py           v7.9 — MoE/hybrid/TQ dispatch probing
├── memory_metrics.py         Per-pool reporting (snapshot + humanize)
├── prealloc.py               GenesisPreallocBuffer framework
├── kernels/                  Pure-python drop-in replacements
│   ├── dequant_buffer.py     P22/P26/P32/P33/P36 + v7.9 P51 hook
│   ├── fla_kkt_buffer.py     P39a pool
│   ├── fp8_dispatcher.py     Ampere FP8 routing
│   ├── gdn_core_attn_manager.py  P28 buffer
│   ├── gdn_dual_stream.py    P7 opt-in
│   ├── gdn_gating_buffer.py  P46 pool
│   ├── marlin_tuning.py      P17/P18 bsm+num_warps
│   ├── moe_intermediate_cache.py  P37 pool
│   ├── page_size_padded.py   P5/P5b
│   ├── router_softmax.py     P31 fp32 upcast
│   ├── tq_continuation_prefill.py  P22/P23/P26
│   ├── tq_decode_tune.py     P36/P40
│   └── tq_grouped_decode.py  P40 opt-in
├── middleware/
│   ├── __init__.py
│   └── response_cache_middleware.py  P50 ASGI cache
├── patches/
│   ├── apply_all.py          Orchestration entrypoint
│   └── upstream_compat.py    PR marker registry
├── wiring/                   Attach-to-vLLM layer
│   ├── __init__.py
│   ├── rebind.py             Attribute rebind registry
│   ├── text_patch.py         Text anchor replacement (idempotent)
│   ├── patch_22_tq_prealloc.py
│   ├── patch_24_moe_tune.py         (P52 gated)
│   ├── patch_28_gdn_core_attn.py    (P53 gated)
│   ├── patch_31_router_softmax.py   (P52 gated)
│   ├── patch_34_mamba_deadlock_guard.py  (P53 gated)
│   ├── patch_37_moe_intermediate_cache.py  (P52 gated)
│   ├── patch_38_tq_continuation_memory.py
│   ├── patch_39_fla_kkt_buffer.py   (P53 gated)
│   ├── patch_46_gdn_gating_buffers.py  (P53 gated)
│   └── ...
└── tests/                    pytest TDD suite
```

---

## Contributing

PRs welcome. Bug reports from hardware we don't test (3090, 4090, A6000, A100, H100, ROCm, XPU) are especially valuable.

Note on MoE tuning: on Ampere with FP8 block quantization, vLLM selects **MARLIN**, not Triton, for MoE. Before tuning, check the startup log:

```
Using MARLIN Fp8 MoE backend out of potential backends: [...]
```

If MARLIN, the only runtime lever is `block_size_m` (Patch 17).

---

## Attribution

Genesis draws on and credits prior work:

- **DeepSeek-V3 team** — fp32 router upcast pattern (basis for P31)
- **@JartX** — TurboQuant author, `JartX/vllm#11` FP16 rotation (P20 prerequisite)
- **@jhsmith409** — endorsed Genesis Ampere investigation, pre-approved P22
- **@ZJY0516** — hybrid prefix cache design clarifications
- **@vibhavagarwal5** — collaborative PR scope guidance
- **@youkaichao** — memory profiler invariants documentation
- **vLLM core team** (@WoosukKwon, @zhuohan123, @robertgshaw2-redhat, @bnellnm) — responsive community, educational codebase

Per-kernel attribution lives in each module's docstring.

---

## Author

**Sandermage(Sander)-Barzov Aleksandr**
Ukraine, Odessa
GitHub: [@Sandermage](https://github.com/Sandermage)
Project: [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches)

---

## License

Apache-2.0 — see `LICENSE`.

---

*Genesis vLLM Master Plan v7.0 / integration v7.1–v7.9.*
*Canonical reference: `Genesis_Doc/common/GENESIS_VLLM_MASTER_PLAN_v7.0_20260424.md`.*
*Integration report: `docs/REPORT_v7_1_INTEGRATION_20260424_RU.md`.*

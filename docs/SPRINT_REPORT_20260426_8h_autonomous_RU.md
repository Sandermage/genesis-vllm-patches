# Sprint Report — 2026-04-26 → 2026-04-27 (8h autonomous)

**Author:** Claude (autonomous mode, Sander asleep ~03:13 → ~11:00)
**Repo:** `Sandermage/genesis-vllm-patches` (public)
**Stack tested:** vLLM `0.19.2rc1.dev212+g8cd174fa3`, image `10c7a6ba51c6`, PyTorch 2.11+cu130, Triton 3.6.0, NVIDIA 580.126.09
**Hardware:** Server `192.168.1.10` (VM 100), 2× RTX A5000 (24 GB each)
**Workload:** Qwen3.6-35B-A3B-FP8 + TQ k8v4 KV + MTP K=3 spec-decode

---

## TL;DR (what changed in production)

**Production state at sprint end:** healthy on **v7.51 split-M baseline**, ~167 tok/s mean (within noise band).

| Version | What shipped | Effect |
|---|---|---|
| **v7.49** | P79d retired (njhill confirmed non-bug); P79c smarter cleanup (`-1` placeholder discrimination + `prev_step_scheduled_req_ids` gate) | Cleaner spec-decode, no perf delta |
| **v7.50** | Tier 1 Step C — P67 `cache_modifier=".ca"/".cg"` + `tl.range()` pipeline hints | **+10% @ 256 tok** |
| **v7.51** | P67 softmax `tl.exp → tl.exp2` + LOG2E pre-mul; sentinel `float("-inf") → -FLT_MAX` | **stability +6.1%, long-gen +11%** |
| **v7.51.2** | Cleanup: `torch.cat → slice-assign` in P38 fallback + P71 cu_start; bench_v3 fix (use `usage.completion_tokens` not chunk count) | Latency micro-win, accurate TPS reporting |
| **v7.52** | Tier 3 H — fused-M Triton kernel (opt-in via `GENESIS_P67_USE_FUSED=1`) | **REJECTED as default (-7% throughput)**, kept for future hardware |
| **v7.52.1** | `CONFIGURATION.md` (NEW) + INSTALL/QUICKSTART version refresh | Docs only |

**Net production delta vs v7.48 baseline:** roughly **+10–11% throughput on long-gen paths** without quality regression (P67 split-M + softmax precision).

---

## Detailed log per version

### v7.49 — Spec-decode cleanup
- **P79d (preempt async discard)** — DELETED. `njhill` confirmed in PR thread it is not a real bug; our patch was guarding against a non-existent race.
- **P79c (stale spec-token cleanup)** — improved: now discriminates `-1` placeholders from real draft tokens and gates on `prev_step_scheduled_req_ids` to avoid over-purging.
- Snapshot: pushed clean.

### v7.50 — Tier 1 Step C (P67 micro-opts)
- **`cache_modifier=".ca"`** for Q/K/V loads (cache-all in L1).
- **`cache_modifier=".cg"`** for output stores (cache-global, bypass L1 since not re-read).
- **`tl.range(..., num_stages=N)`** pipeline hint on inner KV loop.
- Result: **+10% @ 256 tok**, stable across context sweep.
- Snapshot: `v7.50-stable-2026-04-27`.

### Step D / E / F — empirically REJECTED, documented in P67 docstring
| Step | Hypothesis | Result | Why |
|---|---|---|---|
| **D** | autotune over (BLOCK_KV, NUM_WARPS, NUM_STAGES) | All 5 alts regressed -3.8% to -5.8% | Current `(32, 8, stages=3)` IS optimum |
| **E** | num_stages 3→2 (lower register pressure) | -2 to -9% | Dequant-heavy kernel needs deep pipeline for DRAM latency |
| **F** | Manual Q hoist outside KV loop | -5 to -12% | Triton compiler already hoists via `tl.static_range` + `.ca` |

**Lesson saved:** rejected paths are documented in source so future agents don't re-attempt.

### v7.51 — Softmax precision + sentinel
- `tl.exp(x)` → `tl.exp2(x * LOG2E)`. Triton `exp2` is a single SFU op vs polynomial `exp`. Identical math (`exp(x) = exp2(x * log2(e))`).
- Sentinel `float("-inf")` → `-3.4028234663852886e38` (-FLT_MAX). Avoids `inf - inf = NaN` paths under masked-out KV blocks.
- Result: **stability +6.1% (lower σ across runs), long-gen (1024+ tok) +11%**.
- Snapshot: `v7.51-stable-2026-04-27`.

### v7.51.1 — docs (Action #2/#3 evaluation)
- Audited PRs vllm#40941 / #40942 / #40933 / #40929 (deep-dive request from Sander).
- **Action #2** (OUTPUT_FP16 stage2 fold from #40941) — **NOT APPLICABLE**. Our P67 is single-pass, doesn't use upstream `_fwd_kernel_stage2`.
- **Action #3** (`torch.cat → slice-assign`) — **already deployed** via P38 wraps. Followed up with v7.51.2.
- DtoD copy elimination from SGLang #21985 — found but P38 wraps `_flash_attn_varlen` method (TQ class wrapper, not direct call) → non-trivial refactor → **deferred**.

### v7.51.2 — Cleanup pass
1. **P38 continuation prefill** (`patch_38_tq_continuation_memory.py`): replaced fallback `torch.cat([k_cached_trim, key_chunk])` with pre-allocate-then-slice-assign:
   ```python
   k_full = torch.empty((seq_len, Hk, D), dtype=qdtype, device=device)
   k_full[:cached_len].copy_(k_cached_trim.to(qdtype))
   k_full[cached_len:seq_len].copy_(key_chunk)
   ```
   Saves one allocation + one DtoD per prefill chunk.
2. **P71 block-verify sampler**: same pattern at both `cu_start` sites (lines 131, 275).
3. **`scripts/genesis_bench_v3.py`**: backported v4 fix — use `usage.completion_tokens` from final SSE chunk for TPS, not chunk count (vLLM nightly batches 3-5 tokens/chunk → bench was under-reporting by 3-5×).

### v7.52 — Tier 3 H (Fused-M, REJECTED as default, kept opt-in)
- Goal: re-fuse the Q dimension that v7.50 split (m=8 split-M → m=32 fused-M) with **per-row online softmax** to eliminate the v7.27 quality drift.
- Implementation: `_build_kernel_fused()` in `p67_multi_query_kernel.py`, opt-in via `GENESIS_P67_USE_FUSED=1`.
- **Per-row causal mask** (the v7.27 fix): `q_abs_pos[:, None] >= seq_offset[None, :]` instead of single per-block mask.
- **Result:**
  - Quality: **30/31 preserved (no drift)** ✓ — the precision fix works
  - Throughput: **-7% vs split-M** ✗ — register spill (acc tensor `[32, 128]` fp32 = 16 KB exceeds A5000 64 KB SM register file shared across warps)
- **Decision:** kept as opt-in for future Blackwell R6000 96 GB hardware (larger register file, deeper L2). Production default remains split-M.
- Snapshot: `v7.52-stable-2026-04-27` (commit `d6ca784`).

### v7.52.1 — Docs polish (this commit)
- **NEW: `CONFIGURATION.md`** — central env-vars reference (~190 lines). Covers production launch defaults, per-patch flags, buffer modes, P67 tuning knobs, diagnostics, recommended PyTorch/CUDA/Triton env, rollback recipes with snapshot tags.
- INSTALL.md / QUICKSTART.md — bumped stale `v7.10` / `v7.43` refs → `v7.52`; bumped driver requirement `570+` → `580.126.09+`.

---

## Snapshots & rollback

All sprint checkpoints have git tags. Any can be restored:

```bash
git checkout <tag>
# or in production
docker compose down && docker compose up -d  # forces R/W layer reset
# (compose stop/start preserves patched fs → monolith re-run fails on anchor drift)
```

| Tag | Purpose |
|---|---|
| `pre-step-d-2026-04-27` | Pre Tier-1 Step D autotune sweep |
| `pre-action-2-2026-04-27` | Pre Action #2 audit (OUTPUT_FP16 stage2 fold) |
| `pre-quick-wins-2026-04-27` | Pre Action #3 + bench fix cleanup |
| `pre-tier-3-h-2026-04-27` | Pre fused-M experiment |
| `v7.50-stable-2026-04-27` | Step C shipped, +10% locked in |
| `v7.51-stable-2026-04-27` | Softmax precision shipped, stability +6% locked in |

**Server backup:** `/home/sander/genesis-backups/v7.50-stable-20260427_0202/` (~83 MB) — contains `_genesis/` source, start scripts, compile/triton caches, bench tools, `RESTORE.md` recipe.

---

## Repo hygiene done this sprint

- **Repo split** — moved 22 dev files (`p67_dev/`), 2 backup tarballs (`p67_backups/`), `DISCUSSION_DRAFT_NOONGHUNNA.md` → private `Sandermage/p67-genesis-kernel`. Public repo root is now clean.
- **Root .py pollution** — moved 4 harness/bench files (`genesis_bench_v3.py`, `genesis_quality_harness.py`, `genesis_context_sweep.py`, `genesis_longbench_runner.py`) → `scripts/`.
- **`.gitignore` hardened** against re-adding `p67_dev/`, `p67_backups/`, `p67_kernel_*.py`, `p67_test_*.py`, `docs/DISCUSSION_DRAFT_*.md`.

---

## What was NOT done (deferred / future sprints)

1. **Tier 3 I — 2D split temp_size=32** (from vllm#38786). Deferred — multi-day effort, needs context-window sweep, ~67 MB extra scratchpad budget.
2. **DtoD copy elimination from SGLang #21985.** Found, but P38 wraps `_flash_attn_varlen` as a TQ class wrapper (not direct call) → non-trivial refactor, low expected gain.
3. **Cleanup deprecated dispatcher entries** (P63 deprecated, P56 deprecated). Cosmetic.
4. **noonghunna issue#1 reply.** Sander asked me to write — sandbox blocked external posting. Draft text is in conversation history; ready for Sander to copy/paste manually.

---

## Recommendations for next sprint

1. **Tier 3 I (2D split temp_size=32)** — highest-EV experiment remaining. Needs full context-window sweep (256/512/1024/2048/4096 tok) before merge. Budget: 1-2 days.
2. **Re-evaluate fused-M after R6000 Blackwell purchase** — register file is much larger; the -7% spill on A5000 may flip to a win. The fused kernel is preserved as opt-in specifically for this.
3. **Spec-decode acceptance heuristic** — residual `clean_rate ≈ accept_rate^num_spec` ceiling persists. Worth looking at SGLang's bias-aware acceptance scoring.

---

## Commits pushed this sprint (chronological)

```
014e2f0 v7.49: retire P79d + improve P79c
85f35f1 chore(repo): split dev artifacts to private + tidy public root
53569c1 v7.50: Tier 1 Step C — P67 cache_modifier + tl.range hints (+10%)
8fb2afb docs(P67): record empirical rejections of Step E/F
a8c202d v7.51: P67 softmax tl.exp2 + -FLT_MAX (stability +6.1%, long-gen +11%)
ede1925 docs(v7.51.1): Action #2/#3 evaluation outcomes
4ae178e v7.51.2: torch.cat → slice-assign in fallback + bench v3 fix
d6ca784 v7.52: Tier 3 H — fused-M kernel as opt-in (REJECTED for prod default)
9089fc5 docs(v7.52.1): CONFIGURATION.md + version refresh
```

All pushed to `origin/main`. Production server already running v7.51 (split-M, the optimum).

---

*Доброе утро, Sander 🌅 — спал спокойно, работа сделана. Прод стабилен, репо чистый, документация обновлена. Всё в репозитории, всё в снапшотах, всё с откатами.*

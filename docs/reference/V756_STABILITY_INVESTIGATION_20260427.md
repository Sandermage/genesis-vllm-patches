# v756 stability investigation — 2026-04-27

**Status:** LIVE REPRODUCER COMPLETE — crash captured, bisect designed
**Priority:** Medium (unlocks prefix-cache benefit if solvable)
**Risk:** PROD swap done (~5 min); next bisect needs another go.

---

## Symptom recap

- v756 = `--enable-prefix-caching --mamba-cache-mode=align` + NO spec-decode
- Bench result: hit_tokens=2768 on 5K identical prompts (prefix-cache **works**)
- TTFT improvement: 1654ms → 382ms (-77%) on cache HIT
- **BUT** engine crashed under sustained-load bench. "Worker_TP0 died" — exact stack/cause not captured (no saved log)
- Decode TPS: lower than v748 because no MTP K=3 multi-tokenoutput

## Hypotheses ruled out

### Hypothesis A — PR #39064 NULL_BLOCK_ID guard regression (REJECTED)

PR #39064 (merged 2026-04-11, BEFORE our pin `07351e088` of 2026-04-25)
fixed `state_idx < 0` → `state_idx <= 0` guards in
`fla/ops/fused_sigmoid_gating.py` + `fused_recurrent.py` so they catch
the new `NULL_BLOCK_ID=0` (in addition to old `PAD_SLOT_ID=-1`). Old
guards caused IMA crashes on hybrid Qwen3.5-35B-A3B under sustained
concurrency.

**Verified rejection:**
- Both upstream files at our pin contain correct `<= 0` / `> 0` guards
  (lines 114, 158, 296 of `fused_recurrent.py`; line 114 of
  `fused_sigmoid_gating.py`).
- Genesis custom Triton kernels (`p67_multi_query_kernel.py`,
  `tq_grouped_decode.py`, `block_verify_sampler.py`) do NOT touch
  `state_idx`/`NULL_BLOCK_ID`/`PAD_SLOT_ID` at all — they're pure
  spec-decode-verify and Q dequant, no GDN state machinery.

The #39064 fix is correctly in our base. Not the cause.

## Leading hypotheses (to test with live reproducer)

### Hypothesis B — Issue #40707 `_mamba_block_aligned_split` truncation under chunked prefill (LEADING)

Issue [vllm#40707](https://github.com/vllm-project/vllm/issues/40707)
(fanghao566, 2026-04-25, OPEN) reports a scheduling **deadlock** in
`_mamba_block_aligned_split` on hybrid Mamba models under multi-image
multimodal + chunked prefill on Qwen3.5-35B-A3B.

**Code site** in our pinned vLLM,
[`vllm/v1/core/sched/scheduler.py:335`](https://github.com/vllm-project/vllm/blob/07351e088/vllm/v1/core/sched/scheduler.py#L335):

```python
num_new_tokens = num_new_tokens // block_size * block_size
```

When `num_new_tokens < block_size` (typical with `block_size=2048+`
on hybrid + chunked prefill), this truncates to 0. Scheduler then has
nothing to run for that request → tries to schedule again → may get
the same chunk → potential infinite loop.

**Why this fits v756:**
- v756 uses `--enable-chunked-prefill`
- Mamba align on our model gives `block_size` ≈ 2048-2832 (per
  prior investigation)
- `--max-num-batched-tokens=8192` allows multi-request batches
- Sustained load = many requests in flight = high probability of
  hitting a chunk where `num_new_tokens < block_size`
- We don't do multi-image (Sander's reproducer is text-only) — but
  the trigger (small chunks under align) is the same

**Difference from #40707:** that issue describes a **hang**, not a
crash. v756 was reported as "Worker_TP0 died". That could be:
- Container watchdog killed the apparently-frozen process (looks
  like a crash to the user)
- A different bug fires after the deadlock
- Crash predates the deadlock entirely

**Cannot confirm without live trace.**

### Hypothesis C — Mamba state eviction race under high turnover

vLLM v1 hybrid prefix-cache uses
[`MambaManager`](https://github.com/vllm-project/vllm/blob/07351e088/vllm/v1/core/single_type_kv_cache_manager.py)
which reuses pre-allocated state slots via a free-list. Under
sustained load with many distinct prompts, slots churn rapidly. Any
race between:
- `cache_blocks()` registering a hash → slot mapping
- `free_blocks()` returning the slot to free-list
- `find_longest_cache_hit()` reading the hash → slot map

…could cause a worker to read a stale slot → garbage GDN state →
either silent corruption or crash. Our P85 fine-shadow patch adds
shadow-hash entries that mirror the same real slots, which *narrows*
the race window but doesn't eliminate it.

**No upstream issue exactly matches this** (closest is #40624 which
is about miss rate, not crash). Hypothesis C is speculative; needs
trace.

### Hypothesis D — TQ k8v4 `_continuation_prefill` OOM at sustained tail (LOW LIKELIHOOD)

Issue [vllm#40420](https://github.com/vllm-project/vllm/issues/40420)
reports TQ continuation_prefill OOM-kills the engine at ~185K
tokens. v756's bench was on 5K identical prompts (well below 185K),
so this is unlikely the trigger — but a memory-leak pattern over
sustained throughput could accumulate.

## Recommended live reproducer (NEEDS SANDER GO)

**Cost:** ~10 min PROD downtime.

1. `docker stop vllm-server-mtp-test` (PROD v748 down)
2. `bash /home/sander/launch_scripts/test/start_v756_align_no_spec.sh`
   - Wait for boot (~2 min)
3. `python3 /home/sander/Genesis_Project/vllm_engine/genesis_bench_v4.py
     --host localhost --port 8000 --label v756_repro
     --speed-runs 5 --stability-n 50` (sustained-load test)
4. **Capture immediately:**
   - `docker logs vllm-server-mtp-test 2>&1 | tail -200 > /tmp/v756_crash.log`
   - `dmesg | tail -100 > /tmp/v756_dmesg.log`
   - `nvidia-smi > /tmp/v756_nvsmi.log`
5. Restart PROD: `bash /home/sander/launch_scripts/current/start_v748_p82_prod.sh`

Once we have the actual stack:
- If matches Hypothesis B (deadlock signature): backport upstream
  fix once #40707 has one, OR add Genesis guard against
  `num_new_tokens=0` after split (small text-patch in scheduler.py).
- If matches Hypothesis C (race): harder — likely architectural.
- If matches Hypothesis D (OOM): tune `gpu_memory_utilization` or
  add `VLLM_ATTENTION_BACKEND` toggle.
- If something else: write up + post upstream issue (we already
  closed 24-day silence on #38898; opening a new one would extend
  the community-engagement chain).

## Live reproducer outcome (2026-04-27 17:45 UTC, ~5 min PROD downtime)

Sander green-lit the swap. Bench: 5 speed-runs + 50 stability + 30
bursts × 5 stress = 200 total requests on `genesis_bench_v4.py`.

**Crash hit at burst 21/30** (~150 requests in). Container died, all
remaining requests returned 500. Confirmed:

- `nvidia-smi` post-crash: 1 MiB used on each A5000 (engine fully gone)
- `dmesg`: **NVIDIA Xid 43** (Reset Channel Verif Error) on BOTH
  GPUs (PCI 01:00.0 + 02:00.0) → unrecoverable channel reset

### Real error trace (smoking gun)

Saved at `docs/reference/v756_crash_20260427.log` (300 lines).

Root error at line 1 of the stack:

```
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111: operator():
  block: [0,0,0], thread: [0..127,0,0]
  Assertion `-sizes[i] <= index && index < sizes[i]
             && "index out of bounds"` failed.
```

Entire warp (128 threads of block 0, then block 3) hit the assertion
simultaneously → CUDA `device-side assert triggered` → Xid 43 →
worker dead.

### Python surface

```
gpu_model_runner.py:4197  sample_tokens(grammar_output)
gpu_model_runner.py:3351  _sample(logits, spec_decode_metadata)
gpu_input_batch.py:1013   self.async_copy_ready_event.synchronize()
                          → torch.AcceleratorError: CUDA error:
                            device-side assert triggered
```

**Surface ≠ root cause.** `synchronize()` is the first sync point where
a prior async CUDA error surfaces. The bad-index op happened earlier in
the same forward pass (sample / copy / scatter step). Without
`CUDA_LAUNCH_BLOCKING=1` the stack is asynchronous and points at the
sync, not the producer.

### Hypothesis matching

| Hyp | Match | Evidence |
|-----|-------|----------|
| A: PR #39064 NULL_BLOCK_ID guard | ❌ | upstream guards correct, our kernels don't touch state_idx |
| B: #40707 `_mamba_block_aligned_split` deadlock | ❌ partial | bug truncates to 0 → would HANG, not bad-index crash |
| C: Mamba state eviction race | ⚠️ possible | sustained burst churn matches; not directly proven |
| **D: torch tensor index out-of-bounds (NEW)** | ✅ confirmed | IndexKernel.cu:111 is exactly this |

The actual symptom is **D** — a pure PyTorch tensor index op
(`tensor[idx]`, `gather`, `scatter`, or `index_select`) was called
with an index outside the tensor's valid range. This is a Python /
metadata bug, NOT a Triton kernel guard issue.

### Genesis-specific suspects

What v756 enables that v748 does not:
- `--enable-prefix-caching`
- `--mamba-cache-mode align`
- P83 (MTP keep-last-cached-block guard)
- P84 (`hash_block_size` override → 16)
- P85 (hybrid fine-shadow prefix cache, MambaManager scale-factor
  shadow entries)

P83 + P84 + P85 add fine-grained shadow hash entries with
`scale_factor = block_size / hash_block_size`. With block_size≈2048+
and hash_block_size=16, scale_factor ≈ 128. Each `cache_blocks()`
call inserts ~128 shadow entries pointing to the same real KV slot.
Under burst churn (5 concurrent reqs, 30 bursts = constant
preempt+resume, with `max-num-seqs=2` causing 2.5× oversubscription),
a slot can be freed and reused while shadow entries are still mapped
→ subsequent lookup returns a stale slot index → tensor indexing
goes out of bounds.

This is the same shape as **Hypothesis C**, now with a concrete
mechanism: Genesis P85's shadow eviction-safety verify is per-call,
but the `update_async_output_token_ids` codepath uses indices
captured at scheduling time (one or more steps earlier) — those
indices may be stale by the time the async sample completes.

## Bisect plan (next session, ~10 min downtime)

To prove or disprove the Genesis-side hypothesis we need ONE more
reproducer run with all Genesis cache patches OFF:

```bash
# v756-ascetic: align mode but no Genesis cache patches
GENESIS_ENABLE_P83=0 GENESIS_ENABLE_P84=0 GENESIS_ENABLE_P85=0 \
  bash /home/sander/launch_scripts/test/start_v756_align_no_spec.sh
```

Expected outcomes:

| Result | Interpretation | Action |
|--------|----------------|--------|
| Crashes same way | Upstream bug: hybrid prefix-cache + chunked-prefill + async-sample race | Open vllm#issue with our minimal repro; PROD stays cache-OFF |
| Stable | Genesis P83/P84/P85 introduces the race | Disable P83/P84/P85 by default in dispatcher; document as opt-in research only; deploy v756 (or v756-ascetic) as secondary endpoint with cache enabled |
| Different crash | Different bug class | Triage individually |

Optional: capture `CUDA_LAUNCH_BLOCKING=1` for ~10× slower bench but
exact stack pointing at the bad-index operator.

## Production decision (for now)

**PROD stays on v748** (cache OFF + MTP K=3 + P82 t=0.3) — unchanged.

v756 in any current form is unsafe for sustained-load deployment.
Single-user homelab DOES NOT trigger this (max_num_seqs=2 + burst-of-1
behavior at usage time), so cache wins from align mode would be real
ONLY for workloads we don't actually have.

The bisect is worth doing for upstream value (close another silent
issue, like #38898), but Genesis PROD optimization-by-cache is
**lower priority than spec-decode improvements** at our actual usage
profile.

## Pickup checklist (next bisect window)

- [ ] Confirm Sander OK with another ~10 min PROD downtime
- [ ] Edit `start_v756_align_no_spec.sh` to set
      `GENESIS_ENABLE_P83=0 GENESIS_ENABLE_P84=0 GENESIS_ENABLE_P85=0`
- [ ] Stop PROD, launch v756-ascetic
- [ ] Run `genesis_bench_v4.py` with same args (5 speed-runs + 50
      stability + 30 bursts × 5 stress)
- [ ] If crash: capture log → `v756_ascetic_crash_<date>.log`
- [ ] If stable: post upstream issue with Genesis-side cause +
      proposal for P85 shadow-cache TTL/refresh
- [ ] Restart PROD v748

---

## BISECT RESULT (2026-04-27 21:03 UTC, 5 min PROD downtime)

Sander green-lit Variant A. Bisect executed. **Result: bug is
upstream, NOT Genesis-introduced.**

### Setup

Created `start_v756_ascetic_bisect.sh` from `start_v756_align_no_spec.sh`
with three env flags flipped:

```bash
GENESIS_ENABLE_P83=0   # was 1
GENESIS_ENABLE_P84=0   # was 1
GENESIS_ENABLE_P85=0   # was 1
GENESIS_P83_DEBUG=0
GENESIS_P85_DEBUG=0
```

Boot logs confirmed all three patches SKIPPED via dispatcher
(`opt-in only — set GENESIS_ENABLE_PXX=1 to engage`).

Same bench command for fair comparison:

```bash
genesis_bench_v4.py --speed-runs 5 --stability-n 50 \
                    --stress-bursts 30 --stress-per-burst 5
```

### Outcome — IDENTICAL crash signature

Saved at `docs/reference/v756_ascetic_crash_20260427.log` (218 lines).

**GPU level (`dmesg`):**
```
NVRM: Xid (PCI:0000:02:00): 43, pid=2381615, name=python3
NVRM: Xid (PCI:0000:01:00): 43, pid=2381614, name=python3
```
→ Same Xid 43 on both A5000s.

**PyTorch level:**
```
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111:
  Assertion `index out of bounds` failed.
  block: [3,0,0], thread: [0..127,0,0]
```
→ Identical `IndexKernel.cu:111` assert, identical block coordinates,
identical warp-wide failure pattern.

**Python stack:**
```
gpu_worker.py:750         sample_tokens(grammar_output)
gpu_model_runner.py:4197  sample_tokens
gpu_model_runner.py:3351  _sample → input_batch.update_async_output_token_ids()
gpu_input_batch.py:1013   self.async_copy_ready_event.synchronize()
                          → torch.AcceleratorError: device-side assert triggered
```
→ Same async-sync surface, line-for-line identical to v756-with-patches.

### Comparison table

| Metric | v756 (with P83/P84/P85) | v756-ascetic (without) |
|--------|------------------------|------------------------|
| Crash class | IndexKernel.cu:111 OOB | **Same** IndexKernel.cu:111 OOB |
| GPU error | Xid 43 on both A5000s | **Same** Xid 43 on both A5000s |
| Python surface | gpu_input_batch.py:1013 sync | **Same** gpu_input_batch.py:1013 sync |
| Stress success | 0/150 | 0/150 |
| Stability success | 0/50 | 0/50 |
| Speed test | 0 tok/s on all sizes | 0 tok/s on all sizes |
| Burst at first ERR | 21/30 visible (started failing earlier) | 11/30 visible (started failing earlier) |

The slight difference in "first visible ERR burst" is meaningless —
in both runs ALL 200 requests failed (speed + stability + stress
together). The bench runs sequentially: speed → context → stability
→ stress. By the time the printed ERR lines appear in stress, the
engine had already died during speed-test or stability.

### Conclusion

**Hypothesis D-Genesis (P83/P84/P85 race) — REJECTED.**

The crash reproduces with identical signature when our cache
patches are completely disabled. The bug lives in the upstream
combination of:
- `--enable-prefix-caching`
- `--mamba-cache-mode align`
- `--enable-chunked-prefill`
- async scheduling (default in v1)
- sustained burst load (5 concurrent ÷ max-num-seqs=2 = 2.5×
  oversubscription)

Genesis P83/P84/P85 may *amplify* the race window (more shadow
entries → more stale-index opportunities) but they are NOT required
to trigger the crash.

## What we will NOT do (per Sander's rule 2026-04-27)

> "не пишем в ошибки без точных данных и перепроверок с тестами"

We have:
- ✅ Two reproducible crashes with identical signature
- ✅ Exact bench command + exact launch script + exact pin commit
- ✅ Bisected one variable (Genesis cache patches)

We do NOT have:
- ❌ Bisect of remaining variables (`--enforce-eager`,
  `--enable-chunked-prefill`, `--async-scheduling`)
- ❌ Exact bad-index op location (need `CUDA_LAUNCH_BLOCKING=1`)
- ❌ Confirmation the bug exists on a clean `vllm/vllm-openai:nightly`
  WITHOUT any Genesis monkey-patching at all

Until those are checked, **NO upstream post**.

## Triple-confirm narrowing plan (next windows)

To prove the bug is purely upstream and pinpoint the exact code
path, three more bisect runs (each ~10 min PROD downtime):

### Run B1 — `--enforce-eager` (no CUDA graphs)

Adds to launch: `--enforce-eager`. If stable → bug is in CUDA
graph capture/replay path with hybrid+cache+async. Known workaround
class (#39064 family).

### Run B2 — `CUDA_LAUNCH_BLOCKING=1`

Adds `-e CUDA_LAUNCH_BLOCKING=1` to env. Bench runs ~5-10× slower
but the Python stack will point at the EXACT operator that fired
the bad index. Without this, "device-side assert" stack is
asynchronous and points at the sync, not the producer.

### Run B3 — vanilla nightly without ANY Genesis monkey-patching

Mount nothing from `_genesis/`, no env flags, just the upstream
nightly + align mode. If still crashes → bug is purely upstream
of Genesis. If stable → some Genesis text-patch (P34? P5? P14?)
in our 43-applied set is contributing.

Each run independently bounded; each takes 10 min PROD downtime.

## Production decision (post-bisect)

PROD stays on **v748** (cache OFF + MTP K=3 + P82 t=0.3) —
unchanged, healthy.

v756 in any form is unsafe for sustained-load deployment. For our
single-user homelab profile this doesn't matter (max_num_seqs=2 +
1-request-at-a-time = no oversubscription = bug doesn't trigger).
But this rules out v756 as a multi-user secondary endpoint.

## Memory + status

- Bisect cost: ~15 min total PROD downtime (5 + 10 across 2 swaps)
- Two reproducer logs preserved at `docs/reference/v756_*.log`
- v756-ascetic launch script preserved on server at
  `/home/sander/launch_scripts/test/start_v756_ascetic_bisect.sh`
- No upstream posts made (per rule)
- PROD healthy

---

## TRIPLE-CONFIRM RESULTS (2026-04-27 night, ~50 min PROD downtime total)

Sander green-lit overnight bisect. Four narrowing runs executed.

### Bisect matrix

| Run | Config | Speed/Stability/Stress | Verdict |
|---|---|---|---|
| v756 (orig) | TQ k8v4 + cache + Genesis P83/P84/P85=1 | crash @ burst 21 | crash |
| v756-ascetic | TQ k8v4 + cache + P83/P84/P85=0 | crash @ burst 11 | crash |
| **B3-alt-2** | vanilla nightly (no Genesis) + auto kv | **5/5 + 50/50 + 150/150** | **PASS** |
| **B4** | Genesis-ascetic + auto kv (no TQ) | **5/5 + 50/50 + 150/150** | **PASS** |
| **B2** (CUDA_LAUNCH_BLOCKING=1) | TQ k8v4 + cache + P83/P84/P85=0 | crash | crash + EXACT stack |
| B1 | TQ k8v4 + cache + `--enforce-eager` | (in progress at write time) | TBD |

### Trigger ISOLATED

**`--kv-cache-dtype turboquant_k8v4`** combined with
**`--enable-prefix-caching --mamba-cache-mode align`** under
sustained burst load is the trigger.

- B3-alt-2 (vanilla + auto kv): PASS — disproves "purely upstream
  hybrid+cache+chunked+async race" (auto kv with same flags works).
- B4 (Genesis text-patches + auto kv): PASS — disproves "Genesis
  text-patches cause it" (43 active patches, no crash).
- v756-ascetic (Genesis-only-cache-OFF + TQ k8v4 + cache): crashes —
  same as v756. Genesis cache patches (P83/P84/P85) NOT the cause.

### EXACT failure point (from B2 with CUDA_LAUNCH_BLOCKING=1)

```python
# vllm/v1/worker/gpu_model_runner.py:4099 (in execute_model)
sample_hidden_states = hidden_states[logits_indices]
                       ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: device-side assert triggered
```

The bad-index op is **`hidden_states[logits_indices]`** —
`logits_indices` contains an index >= `hidden_states.shape[0]`.

Computation upstream (line 2039 for non-spec-decode path):
```python
logits_indices = query_start_loc[1:] - 1
```

`query_start_loc` is the cumulative sum of token counts per request.
Indices point at the LAST token of each request. For these to be
valid, `hidden_states.shape[0]` must equal total scheduled tokens.

### Hypothesis (still requires direct kernel-level proof)

Under TQ k8v4 + chunked-prefill + prefix-cache + sustained burst:
- TurboQuant `_continuation_prefill` path concatenates dequanted-cached
  K/V with new K/V chunks for the GQA causal attention
- Under burst, multiple requests in flight with varying cache-hit
  prefix sizes
- A subtle off-by-one between `query_start_loc` (computed from
  scheduled tokens) and actual `hidden_states` rows produced by the
  TQ-augmented forward pass causes overflow once the pattern of
  cache hits + chunk boundaries hits a specific configuration

Could also be:
- Genesis P67 (multi-query kernel for spec-verify K+1) hook on
  `_prefill_attention` — but no spec-decode in v756, so this should
  NOT fire. Need to disprove with B5 (P67=0).
- Genesis external_probe patches (`tolist_cudagraph_fix`,
  `PR40074`) directly text-patch turboquant_attn.py at ~5 sites.
  Could affect output shape under specific conditions.

### What we will do (per Sander rule)

> "только нечего не пишем в ошибки без точных данных и
> перепроверок с тестами"

Have:
- ✅ Five reproductions (v756, v756-ascetic ×2, B2 with sync mode)
- ✅ One disproven hypothesis (Genesis cache patches)
- ✅ One disproven hypothesis (purely upstream)
- ✅ Exact line + exact op `hidden_states[logits_indices]`
- ✅ Trigger components: TQ k8v4 + cache + chunked + burst

Still need before drafting upstream issue:
- ❓ B1 result (--enforce-eager) — workaround?
- ❓ B5 (Genesis P67/P67b OFF) — Genesis kernel hook contribution?
- ❓ B6 (skip external_probe patches) — external probe contribution?
- ❓ B7 (TQ k8v4 + cache + bench WITHOUT burst, just sequential) —
  is burst required or is it sustained sequential too?

These remaining bisects narrow the patch surface. Until B5/B6
disprove Genesis involvement, we cannot honestly claim "purely
upstream TQ k8v4 bug" to the maintainers.

### Production reality

PROD v748 (cache OFF + MTP K=3 + P82) continues unchanged. Single-user
homelab profile (max_num_seqs=2, real concurrency=1) does not trigger
this race regardless of root cause. The only deployment value of v756
would be multi-user serving with high prompt repetition — not our
profile. So this investigation is for upstream-engagement value
(closing another silent bug like #38898 we recently helped close)
rather than direct PROD improvement.


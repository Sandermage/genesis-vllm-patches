# 2026-06-09 — Bench reproducibility playbook (PN362 / vllm#42425)

**Author**: Sander (Sandermage / Aleksandr Barzov), Odessa, Ukraine.
**Pin**: `0.22.1rc1.dev259+g303916e93` on `192.168.1.10`, 2× RTX A5000 SM 8.6.
**Trigger**: operator complaint "199 vs 228 = regression" on 2026-06-09 turned
out to be **Triton autotune variance between container restarts**, not a real
code regression.

## TL;DR

Every regression claim from now on goes through a **strict A/B harness** with
`VLLM_TRITON_FORCE_FIRST_CONFIG=1` set on **both sides**. This removes Triton
autotuner non-determinism — the same noise source that produced the false
alarm — so the only remaining delta between A and B is real code, real
config, or thermals.

**Hard targets**:
* 5-run cross-bench CV of `decode_TPOT_ms` < **2 %** when force-first is ON.
* 5-run cross-bench CV of `wall_TPS` < **3 %** when force-first is ON.
* Anything above those is *not* a regression candidate — it is noise; rerun.

## Why Triton autotune is the dominant variance source on our stack

1. **vLLM ships ~12 Triton-autotuned kernels on our hot path**: FLA
   `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o`,
   `chunk_scaled_dot_kkt_fwd_kernel`, the GDN-attn reductions, the unified
   attention prefill kernel, the KV-cache writer, MoE-grouped GEMM, plus a
   few RMSNorm/softmax variants.
2. `@triton.autotune` benchmarks every candidate config on the **first call
   per `key`** and caches the winner. Benchmarking is **timing-driven**, so
   GPU contention, ambient temp, even DRAM refresh aliasing can flip the
   winner between two otherwise-identical container restarts.
3. A flipped winner means a **different reduction order** → numerically
   different outputs → measurably different accept rates under MTP K=3, and
   measurably different per-token latency.
4. Cached results are **persisted on disk** in our `/root/.triton/cache`
   bind-mount. "Just restart and rerun" doesn't reset state; rebuilding the
   image does, but inconsistently.

This is **exactly** what the upstream PR vllm#42425 (Francesco Fusco) was
written for. The author's debug session that motivated the PR was the
**GDN prefill + MTP non-determinism** in PR #40172 — the same kernel family
that fires on every Qwen3.6-35B-A3B request we serve.

## The PN362 vendor and what it does

* PN362 inlines the upstream 107-LOC `force_first_config.install()` helper
  into `vllm/env_override.py` as a single text-patch sub-patch.
* Gated by `VLLM_TRITON_FORCE_FIRST_CONFIG` env var. Default off — no PROD
  behaviour change unless the env var is set on the server container.
* When set, `triton.runtime.autotuner.Autotuner.run` is replaced. New
  behaviour: walk candidate configs in **declaration order**, pick the
  first one that does not raise `OutOfResources` /
  `CompileTimeAssertionFailure` / `PTXASError`. Cache the picked index per
  `(autotuner, key)` so subsequent calls stay deterministic.
* One INFO log line per unique kernel records the picked config — visible
  in the container logs as `[triton-autotune-disabled] kernel=...`.

**Composition with PN345 (shmem-aware autotune pruner)**:

| Layer | When it runs | What it does |
|---|---|---|
| `@triton.autotune` decorator | module import | builds the full candidate config list |
| **PN345** `early_config_prune` | decorator-time | drops configs whose shmem footprint > A5000's 99 KiB budget |
| Triton `Autotuner` instantiation | first kernel call | uses the surviving (PN345-pruned) configs |
| **PN362** `Autotuner.run` | every kernel call | picks first surviving config, caches the index |

PN345 + PN362 are **strictly complementary**: PN345 makes the candidate
list *correct* for our GPU; PN362 makes the *pick* deterministic.

## The A/B reproducibility protocol

### Server-side setup (one-time per pin)

1. Apply PN362 to the container's vllm install:
   ```bash
   ssh sander@192.168.1.10 'docker exec vllm-qwen3.6-35b-balanced-k3 \
       env GENESIS_ENABLE_PN362=1 python3 -m sndr.apply.unified \
           --patches PN362 --apply'
   ```
   (Or include PN362 in the standard apply set if regression A/B is a
   recurring activity for that pin.)

2. Restart the container with the env var set:
   ```bash
   # add to the start-script env block:
   -e VLLM_TRITON_FORCE_FIRST_CONFIG=1 \
   ```
   First server start will be slower than usual on the GDN prefill path
   (force-first walks past OOR configs at runtime; without PN345 it can
   walk past several). With PN345 applied, the first surviving config is
   shmem-safe by construction, so the walk is a single call.

3. Verify the patch lit up by tailing the container log on first request:
   ```
   [triton-autotune-disabled] kernel=chunk_gated_delta_rule_fwd_kernel_h_blockdim64 configs=12 picked_index=0 picked=BV: 32, num_warps: 2, ...
   ```

### Bench-side protocol (every A/B run)

1. Run baseline ("A"), capture JSON:
   ```bash
   python3 tools/genesis_full_bench.py \
       --url http://localhost:8102/v1 \
       --model qwen3.6-35b-a3b \
       --api-key genesis-local \
       --triton-force-first \
       --tag baseline_$(date +%Y-%m-%d) \
       --out tools/bench_results/$(date +%Y-%m-%d)_baseline.json
   ```

2. Apply the candidate patch. Restart server (same env var still set).

3. Run "B", capture JSON:
   ```bash
   python3 tools/genesis_full_bench.py \
       --url http://localhost:8102/v1 \
       --model qwen3.6-35b-a3b \
       --api-key genesis-local \
       --triton-force-first \
       --tag candidate_$(date +%Y-%m-%d) \
       --out tools/bench_results/$(date +%Y-%m-%d)_candidate.json
   ```

4. Compare the JSON files. Pay attention to:
   - `block4_stability.cross_run_CV_TPOT` — must be < 2 % on **both** sides.
     If higher, **rerun** — your bench is too noisy to draw conclusions.
   - `env.vllm_triton_force_first_config_env` — sanity-check it reads `"1"`
     on both reports.
   - `block1_per_request_decode.decode_TPOT_ms.mean` delta between A and B.
     Anything < `3 × std` of either side is **noise**, not a regression.

### Server-side check on noise floor

Run the same bench **5 times back-to-back** against the **same** server
(no patch change) with `--triton-force-first`. Cross-run CV of TPOT must
land under 2 %. If it doesn't:
* check `nvidia-smi` for thermal throttling;
* check for co-tenant containers stealing the GPU (`docker ps`);
* check `tools/bench_results/` for prior runs at radically different
  ambient temps;
* only after those: investigate whether PN345 is also applied (without
  it, PN362 might still pick a different shmem-aware config if some other
  patch shifts the candidate list).

## When NOT to enable PN362

* **PROD steady-state serving.** Picking the autotuned winner is faster
  than picking the first-valid config. Force-first deliberately picks a
  safer (smaller-tile, lower-warp) config first — that is often slower in
  steady state.
* **Bench runs whose purpose is to measure raw peak throughput** — peak
  throughput tracking is what the autotune winner exists for; turning it
  off hides the actual upper bound.

## When to enable PN362

* **All A/B regression hunts** (the whole point of this playbook).
* **Determinism investigations** — flaky accept rates, flaky logits,
  flaky tool-call ordering.
* **Numerical-correctness audits** — same input, same output.
* **Reproducing operator-reported flakes** — first reach for PN362 before
  reaching for a rebuild.

## Decision tree

```
Operator reports "TPS dropped from X to Y between restarts"
    │
    ├─ Did the bench run with VLLM_TRITON_FORCE_FIRST_CONFIG=1 ?
    │       NO  → this is autotune variance; rerun with PN362; STOP.
    │       YES → continue.
    │
    ├─ Is cross_run_CV_TPOT < 2% on the rerun ?
    │       NO  → still noisy; check thermals, container co-tenancy.
    │       YES → continue.
    │
    ├─ Is the delta > 3 × std of either side ?
    │       NO  → this is within-noise; not a regression.
    │       YES → real delta; bisect via patch on/off A/B.
```

## File ledger

* Genesis patch: `sndr/engines/vllm/patches/kernels/pn362_triton_force_first_config.py`
* Registry: `sndr/dispatcher/registry.py` (entry `"PN362"`)
* Dispatcher: `sndr/apply/_per_patch_dispatch.py` (`apply_patch_N362_*`)
* Bench harness: `tools/genesis_full_bench.py` (`--triton-force-first` flag)
* Upstream tracker: vllm-project/vllm#42425 (OPEN as of 2026-06-09)
* Sister patches: PN345 (shmem-aware pruner, vendor of #43047), PN340/PN341
  (MTP decode bubbles, vendor of #43955).

## Post-merge unwind

When vllm#42425 merges to upstream and lands in our pin:
* `sndr/engines/vllm/patches/kernels/pn362_triton_force_first_config.py::apply`
  auto-detects the upstream file `vllm/triton_utils/force_first_config.py`
  and returns `skipped` — no work needed.
* `tools/genesis_full_bench.py --triton-force-first` continues to work
  unchanged (it just sets the env var; upstream's gate reads the same
  name).
* Update PN362's registry `lifecycle` from `experimental` → `retired`,
  add to the RetiredPatchSpec stub list.

## Reference: PR #42425 mechanism quick-card

| Item | Value |
|---|---|
| File added | `vllm/triton_utils/force_first_config.py` (107 LOC) |
| Env-read site | `vllm/env_override.py` (17 LOC append) |
| Envs registration | `vllm/envs.py` (9 LOC) |
| Tests | `tests/test_force_first_config.py` (93 LOC, CPU-only) |
| Net diff | +226 / -0 LOC, 4 files |
| Patch site | `triton.runtime.autotuner.Autotuner.run` (the **runtime** method, not the **decorator**) |
| Invalid errors caught | `OutOfResources`, `CompileTimeAssertionFailure`, `PTXASError` |
| Cache key | `(id(autotuner), tuple(key_vals))` |
| Cache hit | reuses picked index, no walk |
| Cache miss | walks `range(len(configs))` in declaration order |
| All-invalid | raises `RuntimeError("No valid config ...")` with original exception chained |
| Per-kernel log | one INFO line per unique `base_fn.__name__` |
| Default | **off** — strict opt-in |
| Differs from PR #34648 | this patches **runtime method** (catches third-party libs); #34648 patches **decorator** (import-order dependent) |

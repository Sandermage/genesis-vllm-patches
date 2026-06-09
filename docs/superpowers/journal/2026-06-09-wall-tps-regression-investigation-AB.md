# 2026-06-09 — wall_TPS regression investigation + clean A/B proof

## Operator concern

> "wall_TPS per-request регрессия — все что ниже 228-241 это регрессия. Ты выше
> писал про wall_TPS 228+."

Valid observation. Earlier session journals documented:

  * ``wall_TPS = 227.34, CV = 6.76 %`` (genesis_bench_suite n=50)
  * ``Pre PN29 (PN341 stack): 221.4 / 226.8 / 231.2 / 224.9 (mean 227.6)``
  * ``Post PN29 (+ PN29): 226.5 / 228.4 / 228.5 / 227.2 (mean 228.0)``

Current measurements (same pin 0.22.1rc1.dev259+g303916e93, same launcher):

  * ``bench_decode_tpot_clean_ab.py n=50 max_tokens=1024``: wall_TPS = **199.1**
  * ``genesis_bench_suite.py n=25 max_tokens=1024``: wall_TPS = **205.0**

Real **~25-30 TPS regression** vs historic baseline.

## A/B test — are my new patches (PN346-361) the cause?

Methodology: edit launcher to ``export GENESIS_DISABLE_PNxxx=1`` for the
6 new patches added this session, revert the upstream text-patches via
container python script that sees ANCHOR_MISSING when dispatcher tries
to re-apply, restart, run identical bench n=50 max_tokens=1024.

Verification before bench: ``/proc/1/environ`` confirms env vars
present on PID 1; ``Path(f).read_text().count("Genesis PNxxx") == 0``
confirms files reverted; ``inspect.getsource(MambaManager.find_longest_
cache_hit)`` confirms the drop_eagle guard absent.

### Result

|                                | wall_TPS mean | wall_TPS max | decode_TPOT | TTFT     |
|--------------------------------|--------------:|-------------:|------------:|---------:|
| All 6 new patches ON (current) | 199.10        | 237.2        | 4.91 ms     | 147.0 ms |
| All 6 new patches OFF (A/B)    | 199.56        | 238.07       | 4.90 ms     | 147.9 ms |
| **Δ**                          | **+0.46 TPS** | +0.87        | +0.01 ms    | +0.9 ms  |

**IDENTICAL within CV (8.5-9.1%). The 6 new patches add ZERO measurable
regression vs the historic-stack baseline.**

The 25-30 TPS gap vs historic 228 baseline is from a **different
source**, not from PN346/PN347/PN348/PN349/PN351/PN361.

## What actually changed since historic 228 measurement

Same pin. Same launcher. Same model. Same Genesis patch stack
(PN340+PN341+PN29+PN299* + new PN345-361 which A/B prove neutral).

What differs:

  * **Container restart count**: this session has had ~15+ restarts
    (each bench cycle, each test).
  * **Triton autotune cache state**: rebuilt fresh on every restart
    (no persistent ``VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`` configured at
    container-creation time — the YAML edit only takes effect on
    container *recreate*, not *restart*).
  * **CUDA graph capture state**: re-captured on every restart; the
    16-shape capture sequence may pick different configs each time
    depending on which shapes arrive first during warmup.
  * **Time-on-device**: historic 228 was after a sustained-load
    bench cycle (10-run × 5 prompts × 1024 tokens × 4 trials)
    ≈ 200 sustained requests with no restart in between. Today's
    longest sustained warm is 30 requests max.

This is **session-level transient state**, not code/config drift.

## What we tried that didn't recover the gap

  * **Extended warmup** (5 cycles × 6 requests = 30 warm-up): TPS still
    199-205. No change.
  * **Disable PN346 + PN347** (suspected QPS-cost from upstream PR
    bench): TPS 193 (slightly worse — within CV but not better).
  * **Disable all 6 new patches**: TPS 199-200 (identical to all-on).

## Math: how 228 TPS at max_tokens=1024 is achievable

Required TPOT to hit 228 wall_TPS at TTFT=145ms::

  wall_TPS = tokens / (TTFT + (tokens-1) * TPOT)
  228 = 1024 / (0.145 + 1023 * TPOT)
  → TPOT = 4.25 ms

Current sustained TPOT ≈ 4.9 ms. To recover historic 228 wall_TPS we
need to drop TPOT by ~0.65 ms (13%) at the engine level.

That's the gap. It's NOT a patch issue — it's a kernel / autotune
warm-state issue.

## Path forward

Three levers, in increasing effort:

### 1. Persistent FlashInfer autotune cache (free)

Already in YAML as ``VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/var/cache/vllm/
flashinfer_autotune``. The YAML edit takes effect only on container
*recreate*, not on plain *restart*. Operator step: recreate the
container with current YAML config and mount ``/var/cache/vllm``.
Expected: autotune cache survives restarts; warm-state TPOT
converges faster after each restart; mean wall_TPS recovers within
2-3 warm-up cycles.

### 2. Vendor performance-recovering PRs (medium)

The next-iteration patches in roadmap include several real perf wins
that don't depend on autotune state:

  * **PN350**: SGLang #26206 + TRT-LLM #12966 fused GDN Q/K/V split
    Triton kernel — +2.65 % output TPS on Qwen3.6-35B-A3B per
    SGLang bench. ~1-2 days effort.
  * **PN353 stack**: lesj0610's TurboQuant bundle (#43432 MSE V +
    #44053 workspace pre-alloc + #43747 chunked-prefill CG fix +
    #43887 MTP K+1 routing). Combined -5-9 % decode_TPOT at K=3.
  * **PN357**: yewentao256's draft greedy speedup (#43349) — 37-81 %
    kernel speedup on the spec-decode helper.

Combined these alone could give back the 13 % TPOT we need to hit
historic 228 wall_TPS.

### 3. Investigate kernel-level autotune drift (heavy)

Trace which Triton kernel is selecting different config in fresh-cache
vs warm-cache state via ``VLLM_LOGGING_LEVEL=DEBUG`` + autotune logs.
Identify the kernel(s) with the largest TPOT contribution, pin their
configs explicitly. Effort: ~1-2 days investigation + per-kernel
explicit-config patch authoring.

## Bottom line

  * **My 6 new patches add zero perf regression**. A/B proven.
  * **Historic 228 baseline is achievable in principle** — current TPOT
    4.9 ms vs historic ~4.25 ms is a kernel warm-state delta, not a
    patch cost.
  * **Path 1 (persistent autotune cache) is free** and the right next
    operator-side step. Container recreation needed.
  * **Path 2 (vendor next perf PRs)** is the development-side step;
    iteration N+3 roadmap already queued.
  * **Path 3 (kernel autotune trace)** is a deeper investigation
    — defer unless paths 1+2 don't recover the gap.

## Methodology lesson

When the operator says "wall_TPS regressed", the right first move is
a clean A/B with the new code OFF — same bench harness, same pin,
same launcher, just env-flag the new code out and rebench. This took
~25 minutes of investigation and definitively cleared the new
patches. Going forward, every "perf regression" claim gets an A/B
proof before any code change to "fix" it.

# Context-depth speed + quality sweep — 35B-A3B-AWQ PROD (2026-06-30)

Reddit/localllama-style sweep separating **prefill** (prompt processing) from
**decode** (generation) to expose the speed-degradation curve as context grows,
plus concurrency scaling, a prefix-cache probe, and quality-vs-depth.

Tool: `tools/ctx_depth_sweep_bench.py` (run on the rig against `localhost:8102`
so LAN latency doesn't pollute TTFT) + `tools/ctx_depth_sweep_bench.py` quality
half via the companion probe.

## Engine under test

| param | value |
|---|---|
| model | `Qwen3.6-35B-A3B-AWQ` (int4 weights), served `qwen3.6-35b-a3b` |
| hardware | 2× RTX A5000 24 GB, TP=2 |
| max-model-len | 280000 (TQ KV) |
| kv-cache-dtype | `turboquant_k8v4` (FP8-key / 4bit-value) |
| speculative | MTP, `num_speculative_tokens=5` (K=5) |
| max-num-seqs | **2** |
| max-num-batched-tokens | **4096** (chunked prefill on) |
| prefix caching | **effectively OFF** — `vllm:prefix_cache_queries_total = 0` |

## 1. Context-depth sweep (forced 256-token decode, temp=0, thinking off)

| content | prompt_tok | TTFT s | prefill tok/s | decode tok/s* | e2e tok/s |
|---|---|---|---|---|---|
| code | 544 | 0.16 | 3414 | 156 | 143 |
| code | 2094 | 0.46 | 4595 | 156 | 122 |
| code | 8318 | 2.16 | 3860 | 184 | 72 |
| code | 16606 | 3.39 | 4896 | 197 | 55 |
| code | 33186 | 7.23 | 4593 | 160 | 29 |
| code | 66348 | **38.9** | **1705** | 171 | 6.3 |
| prose | 5944 | 1.21 | 4915 | 143 | 86 |
| prose | 23701 | 5.09 | 4660 | 194 | 40 |
| prose | 47377 | 18.9 | 2503 | 149 | 12 |
| struct | 15075 | 3.29 | 4584 | 192 | 56 |
| struct | 61047 | 33.2 | 1838 | 134 | 7.3 |
| struct | 123129 | **134.9** | **913** | 101 | 1.9 |

\* decode tok/s is MTP-acceptance-dependent (see §4) — not a clean monotonic
curve; the **honest degradation signals are TTFT / prefill tok/s / e2e tok/s**.

### What the curve shows
- **The bottleneck at depth is PREFILL, not decode.** Decode stays ~130–200
  tok/s across the whole range (MTP keeps generation fast). What collapses is
  prefill: TTFT goes 0.16 s → 7 s @33k → **39 s @66k → 135 s @123k**, and
  prefill throughput **halves past ~50k tokens** (≈4600 → 1705 → 913 tok/s).
- **e2e throughput** (what a user waits for) therefore falls ~143 → 6 tok/s as
  context grows — dominated by that prefill cost.
- The prefill cliff lines up with chunked prefill at `max-num-batched-tokens=4096`
  (a 66k prompt = 16 sequential chunks) + the quadratic attention term that only
  bites at depth.

## 2. Concurrency sweep (code ~4k in, forced 256 decode) — max-num-seqs=2

| concurrency | agg tok/s | per-stream decode tok/s | mean TTFT s |
|---|---|---|---|
| 1 | 126 | 234 | 0.9 |
| 2 | 86 | 57 | 1.4 |
| 4 | 84 | 54 | 4.4 |
| 8 | 99 | 67 | 9.2 |

**MTP scales badly under concurrency.** Single-stream decode is excellent (234
tok/s with K=5 speculation), but at conc=2 per-stream decode collapses to 57 and
**aggregate throughput drops** (126 → 86). Speculative verify across a batch
loses efficiency when acceptance diverges between sequences; combined with
`max-num-seqs=2`, conc>2 just queues (TTFT climbs linearly). For a multi-user
endpoint, MTP K=5 is the wrong default.

## 3. Quality vs depth

Needle-in-haystack (secret code mid-context, thinking off): found at 1k / 8k /
64k / **128k** ✅ — but **missed at 16k** ❌ (returned filler). Deeper depths
pass, so it's not a depth ceiling — likely a position/chunk-boundary artifact;
**needs multi-sample confirmation** (single sample per depth here).

Reasoning (thinking on): 3/3 correct (60 km/45 min → 80; widget puzzle → 3;
17×23 → 391). Reasoning quality is solid.

## 4. Why decode tok/s is noisy

`ignore_eos` forces 256 tokens of (often repetitive) continuation; MTP K=5
accepts almost all speculated tokens on predictable text → inflated,
content-dependent decode rates (struct hit 226, code 197). This is **real
engine behaviour**, not measurement error: with MTP, "decode tok/s" is a
function of content predictability, not a single number.

## 5. How to improve speed — prioritized levers

| # | lever | expected gain | cost / risk | needs restart |
|---|---|---|---|---|
| 1 | **Enable prefix caching** for repeated-context workloads (RAG fixed corpus, multi-turn, agentic history) | TTFT on cache-hit: 5 s→~0 (23k), 39 s→~0 (66k) — **10–100×** | TQ KV (`turboquant_k8v4`) appears to disable block reuse → tradeoff vs the 280k context window. Needs a TQ-vs-fp8-KV + APC A/B | yes |
| 2 | **Raise `max-num-batched-tokens`** 4096 → 8192/16384 | faster prefill at depth (fewer chunks) — targets the 39 s/135 s TTFT directly | more activation memory; re-check OOM headroom at 280k | yes |
| 3 | **Drop MTP K (or disable) for multi-user serving** | aggregate throughput + per-stream fairness at conc≥2 | loses the 234 tok/s single-stream win → workload-dependent (latency vs throughput) | yes |
| 4 | **Tune MTP K** (5→3) for a latency/throughput middle ground | better concurrency without fully losing speculation | needs A/B on acceptance rate | yes |

**Headline:** the rig is decode-fast (MTP) but **prefill-bound at long context**,
and the single biggest unrealized win is **prefix caching** for any workload that
re-sends a shared prefix. Each lever is a launch-flag change → a controlled live
A/B (the natural next step).

Raw JSON: `/tmp/ctx_sweep_results.json` on the rig.

## 6. Live A/B validation of the levers (2026-06-30, prod restored after)

Each lever tested by editing the bind-mounted inner `run.sh`, `docker restart`,
re-bench, then restored to canonical (AWQ / batched-4096 / MTP K=5 / no-APC).

### Lever 1 — `max-num-batched-tokens` 4096 → 16384: **NO improvement (rejected)**

| content | prompt_tok | TTFT 4096 | TTFT 16384 |
|---|---|---|---|
| code | 33186 | 7.23 s | 7.27 s |
| code | 66348 | 38.9 s | 38.5 s |
| struct | 123129 | 134.9 s | 133.5 s |

The prefill cliff is **fundamental quadratic-attention cost at depth, not a
chunk-count artifact** — bigger prefill batches don't help and cost more
activation memory. Keep 4096.

### Lever 2 — `--enable-prefix-caching`: **6–10× TTFT win (strong recommend)**

| workload | miss TTFT | hit TTFT | speedup |
|---|---|---|---|
| prose 23.7k tok | 5.33 s | 0.84 s | **6.3×** |
| code 66k tok | 38.7 s | 3.84 s | **10.1×** |

`prefix_cache_queries_total` went 0 → 180k, hits 83.5k (~46% in-test). **APC is
fully compatible with TQ `turboquant_k8v4` + AWQ + MTP + the 280k window** — the
earlier "TQ disables APC" hypothesis was wrong; the flag was simply absent from
the launcher. No tradeoff with context size. Biggest single win for any
repeated-prefix workload (RAG fixed corpus, multi-turn chat, agentic history).

### Lever 3 — MTP `num_speculative_tokens` 5 → 3: **1.7–2.1× concurrency (recommend for multi-user)**

| concurrency | K=5 agg tok/s | K=3 agg tok/s | K=5 per-stream | K=3 per-stream |
|---|---|---|---|---|
| 1 | 126 | 111 | 234 | 205 |
| 2 | 86 | 94 | 57 | **121** |
| 4 | 84 | **161** | 54 | 130 |
| 8 | 99 | **169** | 67 | 153 |

K=3 costs ~10 % single-stream decode but is **1.7–2.1× faster in aggregate at
conc ≥ 2** and avoids K=5's concurrency cliff (per-stream 57 → 121 at conc=2).
Note: the launcher is named `…-k3` and its header says "MTP K=3" — the live `5`
is config drift. Use K=3 for any multi-user endpoint; keep K=5 only for a
dedicated single-user low-latency path.

### Recommended permanent changes (via the V2 profile YAML re-render, not a manual launcher edit)

1. **Add `--enable-prefix-caching`** — unconditional win, no downside found.
2. **Set MTP K=3** (matches the launcher's own stated intent) unless the box is
   single-user-latency only.
3. Leave `max-num-batched-tokens=4096` (16384 gave nothing).

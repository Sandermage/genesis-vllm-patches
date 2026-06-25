# Troubleshooting — cliffs, OOM, recipes, rollback

This is the single operator-facing reference for "something is wrong,
what do I do". It consolidates:

- **Quick triage** — decision tree to land in the right section.
- **Named cliffs** (1–8) — regime boundaries that flip vLLM/Genesis
  from working well to silently working badly.
- **OOM recipes** — single-card 24 GB long-context hardening,
  multi-turn soak survival.
- **Operational cookbook** — 10 named symptom→cause→workaround→fix
  scenarios from community feedback (noonghunna/club-3090) and our own
  operational history.
- **Rollback playbook** — named R-XYZ procedures with revert + smoke
  commands for every major V2 / community-SDK feature.

If your problem isn't here, please open an issue with a reproducer.
Cliffs that aren't documented hurt every operator after you.

## Quick triage

| Symptom | Jump to |
| --- | --- |
| Boot loop, `[FAIL]` lines in `docker logs` | [R/W layer trap](#9-container-rw-layer-trap-on-compose-stopstart) |
| OOM at long context (>50K tokens) | [Cliff 1](#cliff-1-fa2-softmax_lse-over-allocation), [Cliff 2](#cliff-2-gdn-fwd_h-blow-up), [OOM recipes](#oom-recipes) |
| Garbage tokens / tool-call cascade (27B + TQ) | [Cliff 3](#cliff-3-turboquant--spec-verify-k1--full-cudagraph), [Cliff 4](#cliff-4-non-power-of-2-gqa--p67) |
| TPS dropped after vLLM pin bump | [Cliff 8](#cliff-8-anchor-drift-on-vllm-pin-bumps) |
| Prefix-cache + MTP crash | [Recipe 6](#6-turboquant--spec-decode--prefix-caching-crash) |
| Driver / CUDA / NCCL mismatch | `sndr doctor`, then [`INSTALL.md`](INSTALL.md) |
| V2 alias resolution broken | [R-001](#r-001--v2-alias-resolution-broken) |
| `sndr memory explain` mis-predicts | [R-004](#r-004--sndr-memory-explain-mis-predicting-oom) |
| Want to roll back the whole release | [Rollback playbook](#rollback-playbook) |

## Named cliffs

A "cliff" is a regime boundary where vLLM (or Genesis) goes from
working well to working badly — sometimes silently. We catalogue them
so the next operator hits a known mitigation, not a debugging session.

### Cliff 1: FA2 softmax_lse over-allocation

**Mechanism.** vLLM's GPU model runner sets
`attn_metadata.max_seq_len = max_model_len` during cudagraph capture.
FlashAttention-2 allocates
`softmax_lse[num_seqs, num_heads, max_seqlen_k]` sized by that
ceiling, even when the actual batch only needs a fraction. At long
contexts (> 50K tokens, `max_model_len=256K`), this wastes
50–100 MiB per capture region.

**Impact.** OOM earlier than expected on long-context workloads. On a
24 GB card running 27B INT4 at 256K context, the over-allocation alone
can be the difference between booting and OOM.

**Fix.** **PN17** — FA2 lse runtime clamp (Genesis-original,
2026-04-30, response to noonghunna #11). Patches FA2 to use the
actual `seq_lens.max()` at runtime instead of `max_model_len` during
capture.

**Related — PN19 — scoped max-split cudagraph init.** Datacenter
Ampere / Hopper / Blackwell only. Frees 200–500 MiB during model load
on H100 / B100. **Does NOT transfer cleanly to Ampere consumer:**
noonghunna 2026-05-01 confirmed PN19 costs ~120 MiB KV pool on
single-3090 24 GB. At 218K context + 0.985 mem-util, engine init
fails with `KV cache memory available 3.4 GiB, estimated maximum
model length 206400`. Disable PN19 on 24 GB consumer cards running
long context.

**Refs.** `integrations/attention/flash/pn17_fa2_softmax_lse_clamp.py`,
`integrations/_retired/pn19_scoped_max_split.py`, noonghunna #11,
club-3090 #19.

### Cliff 2: GDN fwd_h blow-up

**Mechanism.** `chunk_gated_delta_rule_fwd_h` allocates an
intermediate `h` tensor sized `(B, NT, H, V, K)`, where `NT` is the
number of chunks along the sequence dimension. At T=64K on
Qwen3.6-27B (H=32, V=K=128), this is ~805 MiB just for `h` — for a
single prompt.

**Impact.** Single-prompt long-context generation (> 50K tokens) OOMs
on 24 GB cards even when KV cache itself fits comfortably.

**Fix.** **P103** — chunked fwd_h + fwd_o orchestrator. Splits the
chunk dimension into sub-batches, materialises `h` per sub-batch,
runs `fwd_o`, discards before the next iteration. Saves ~600 MiB of
headroom at 64K, more at longer contexts.

**Refs.** `integrations/attention/gdn/p103_fla_cliff2_chunked.py`. See
also P60 / P60b for related GDN spec-decode corruption fixes.

### Cliff 2b: GDN multi-turn soak OOM (continuous 5×5 ramp)

**Mechanism.** noonghunna residency analysis: PN12 pools stay flat at
~137 MiB across turns (Genesis-side memory is clean). Growth lives in
the PyTorch caching allocator + vLLM internal state — per turn,
`total_reserved +1400 MiB`, `total_alloc +590 MiB`,
`fragmentation +810 MiB`, `free −1402 MiB`. After 4–5 turns the
reserved+fragmented budget exceeds free → OOM in the next
`chunk_fwd_o` `empty_like(v)` allocation.

**Mitigations (in order):**

1. Lower `--gpu-memory-utilization` to 0.85 (give allocator headroom).
2. Drop `--max-model-len` below the cliff (e.g. 96K vs 180K).
3. Use `fp8_e5m2` KV instead of `turboquant_k8v4` (K activation peak
   < V activation peak on the soak path; see
   [TQ vs fp8_e5m2 trade-off](#tq-k8v4-vs-fp8_e5m2-trade-off)).
4. Disable MTP for high-cliff sessions (MTP K=3 adds ~600 MiB / draft
   step).
5. Force `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6`
   (already in 35B PROD and DFlash scripts; add to 27B too).
6. Restart the engine every ~50 multi-turn requests under stress.

**Long-term.** PN59 streaming-GDN refactor (noonghunna #19) — the
only mitigation that survives continuous soak.

### Cliff 3: TurboQuant + spec-verify K+1 + FULL cudagraph

**Mechanism.** `TurboQuantAttentionImpl._prefill_attention` treats
spec-verify K+1 batches as first-chunk prefill — it sets
`cu_seqlens_k = cu_seqlens_q`, ignoring already-cached KV. When this
path is captured into a FULL cudagraph, the captured kernel launch
ignores cached KV unconditionally even at runtime.

**Impact.** Tool-call cascades on 27B + TQ k8v4 + FULL cudagraph: the
model emits `<tool_call><tool_call>...` infinitely. Looks like a
tool-call parser bug; root cause is attention.

**Fix.** **P67** — Genesis-original multi-query Triton kernel.
Replaces upstream's `_prefill_attention` for the K+1 verify case with
a kernel that correctly attends to cached KV. The earlier P65 (switch
to PIECEWISE cudagraph) is a workaround costing ~5–8% TPS; P67 is the
proper fix and gains TPS instead of losing it.

**Refs.** `kernels/p67_multi_query_kernel.py`,
`integrations/attention/turboquant/p67_tq_multi_query_kernel.py`.
See the P67 entry in [`PATCHES.md`](PATCHES.md) for the Inf/NaN→0
sanitized variant.

### Cliff 4: non-power-of-2 GQA + P67

**Mechanism.** Triton's `tl.arange` requires power-of-2 dimensions.
P67's original kernel used `tl.arange(0, HEADS_PER_KV)` for the
query-head dimension. On Qwen3.6-27B with GQA=24/4,
`HEADS_PER_KV=6` — not a power of 2 — so the kernel fails to compile.
Without compile success P67 falls through to the upstream broken
path, and you're back at Cliff 3 (garbage tokens under FULL
cudagraph).

**Impact.** 27B with TQ k8v4 + FULL cudagraph silently emits garbage.
The fall-through is logged but easy to miss in a long boot log.

**Fix.** **P67 v7.63.x non-pow-2 generalisation** —
`BLOCK_QH = next_power_of_2(HEADS_PER_KV)` and a
`lane_valid = (lane_id < HEADS_PER_KV)` mask that writes only valid
lanes. Negligible perf cost (a couple of masked stores), full
correctness on GQA=6.

**Validated** on 35B (GQA=8, pow-2) and 27B (GQA=6, non-pow-2).

### Cliff 5: ngram strict prompt_lookup_min=8 underperforms MTP on prose

**Mechanism.** Strict ngram requires an 8-token sequence to appear
in the prompt before speculating. On code completion or tool-use-
heavy workloads (structured + repetitive prompts) acceptance stays
high. On free-form prose an 8-token literal match almost never
appears — ngram falls back to single-token decode and you lose the
speculation entirely.

**Impact.** 27 TPS on 27B creative-writing workload with strict
ngram vs. 87–100 TPS with MTP K=3.

**Fix.** Configuration, not a patch:

- **General workloads:** MTP K=3 if available, or ngram with
  `prompt_lookup_min=2, prompt_lookup_max=5` (the loose default).
- **Tool-use-heavy workloads:** strict ngram (`min=8, max=8`) lifts
  tool-call clean rate from 56% to 100% on a single-query benchmark.
  Use it only when tool-call quality matters more than prose
  throughput.

### Cliff 6: MoE backend regression on v0.20+ for non-FP8

**Mechanism.** vLLM v0.20 refactored MoE dispatch into
`PluggableLayer` / `DefaultMoERunner`. The new abstraction adds a
per-step CPU dispatch overhead. FP8 paths take a fast path that
bypasses most of it; non-FP8 (BF16, AWQ) hits the full dispatch
cost.

**Impact.** −19% TPS on Mixtral-class BF16 MoE on v0.20+ vs v0.19.
Reported upstream as vLLM #41306.

**Mitigation.** `--moe-backend=triton` flag or
`VLLM_MOE_BACKEND=triton` env var. Forces the older Triton MoE path
that doesn't go through the new dispatcher. Affects only non-FP8 MoE
on v0.20+; FP8 MoE (Qwen3.6-35B-A3B) is unaffected.

### Cliff 7: DFlash + 24 GB single-card OOM at > 80K context

**Mechanism.** DFlash speculative decoding co-resides a small drafter
model (typically 2B BF16) with the main model. On 24 GB running
35B-A3B-FP8 + DFlash 2B BF16:

- Main model FP8: ~17 GB
- DFlash drafter BF16: ~4 GB
- Activation + KV cache: rest

TurboQuant KV is not currently supported with DFlash on Ampere (the
draft path doesn't go through the TQ KV reader). So KV stays in
`auto` / `fp8_e5m2` — capacity-limited. At > 80K context the KV
cache pushes you past 24 GB.

**Mitigation.**

- vLLM PR #40898 (SWA for DFlash, pending merge) limits drafter
  context.
- vLLM PR #40849 (FP8 draft inheritance, pending merge) lets the
  drafter share the main model's FP8 cache.
- `num_speculative_tokens=4` (more aggressive verification reduces
  effective KV pressure).
- Accept the 80K ceiling on 24 GB — 48 GB cards (A6000, R6000 Pro)
  don't hit this.

### Cliff 8: anchor drift on vLLM pin bumps

**Mechanism.** Genesis text-patches anchor on verbatim upstream code.
When upstream renames a variable, refactors a function, or even
changes whitespace, the anchor no longer matches. TextPatcher logs
`INFO: anchor not found, sub-patch skipped` and moves on. If
`required=True` the whole patch is marked `failed`. If
`required=False` the patch reports `applied` despite a sub-patch
missing.

**Impact.** Operator pulls a new vLLM pin, restarts, sees
`[GENESIS] APPLY` for all expected patches in the boot log, and
assumes everything works. In reality a sub-patch silently skipped and
the bug it was guarding against is back.

**Mitigation.**

- Grep the patched file for the marker string
  (`# [Genesis wiring marker: Genesis PNN ...]`).
- Watch `partial_apply_warnings` — TextPatcher hardening planned to
  surface these in the boot summary.
- Run anchor-presence tests before bumping a pin.
- Pin vLLM commits in your launch script; don't float on `main`.

**Refs.** `sndr/kernel/text_patch.py`,
[`PATCHES.md`](PATCHES.md) for the currently-tested pin.

## OOM recipes

### Single-card 24 GB long-context (60K+) — the noonghunna club-3090 #22 recipe

**Symptom.** vLLM 0.20.2 + Genesis Wave 8+ on 1× RTX 3090 24 GB,
Qwen3.6-27B-int4-AutoRound + hybrid GDN + chunked-prefill, 60K-token
single-shot prompt → `OutOfMemoryError: tried to allocate 50 MiB,
56 MiB free` at `chunk_o.py:161 o = torch.empty_like(v)`.

**Root cause.** At `gpu_memory_utilization=0.93` the KV pool eats
22.4 GiB, leaving ~1.6 GiB headroom for activations. FLA
`chunk_gated_delta_rule_fwd_h` allocates `(B, NT, H, V, K)` h-tensor
= **1.37 GiB at T=60K** which doesn't fit. PyTorch caching allocator
fragments under repeat 1.37 GiB alloc-free cycles → "50 MiB
requested, 56 MiB free" symptom.

**The recipe** (env-only, no code changes — Level 1 mitigation):

```bash
# 1. Cliff 2 chunking patch — split T-dim 60K → 4×16K
export GENESIS_ENABLE_P103=1
export GENESIS_FLA_FWD_H_MAX_T=16384

# 2. Streaming-GDN window-iterative driver
export GENESIS_ENABLE_PN59_STREAMING_GDN=1

# 3. PyTorch allocator hardening (0.6 → 0.85 stops needless GC churn)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.85"

# 4. vllm serve flags — lower gpu_memory_utilization 0.93 → 0.85
vllm serve ... \
    --gpu-memory-utilization 0.85 \         # +1.9 GiB activation headroom
    --max-num-batched-tokens 2048 \         # halves Marlin workspace peak
    --max-num-seqs 1 \                      # halves worst-case KV reservation
    ...
```

**Effect.** h-tensor peak 1.37 GiB → **365 MiB** (P103) and headroom
1.6 GiB → **3.5 GiB**. Real KV at 60K single-stream is ~3 GiB out of
~20.5 GiB pool — the pool is over-provisioned for the workload, the
0.05 we give back is "paper" capacity not real.

The `bare_metal_27b_int4_TQ_k8v4_single_card.sh` reference launch
script ships with this recipe baked in.

### Quick-reference matrix

| Card / VRAM | Workload | Recipe |
| --- | --- | --- |
| 1× 3090 24 GB | Short ctx (≤ 8K), code completion | TQ k8v4 + util 0.92 + `max_num_seqs=4` |
| 1× 3090 24 GB | Long ctx (60–180K), multi-turn | `fp8_e5m2` KV + util 0.85 + PN35 ON |
| 2× 3080 20 GB (TP=2) | Long ctx > 90K | `fp8_e5m2` (TQ k8v4 OOMs at 90K — club-3090 #47) |
| 1× 3090 WSL2 | Any | util 0.85 (vGPU / Xwayland eats ~3.6 GB / card — club-3090 #32) |
| 2× A5000 24 GB (TP=2) | All PROD presets | TQ k8v4 + util 0.90 + MTP K=3 stable |

### TQ k8v4 vs `fp8_e5m2` trade-off

| Property | TQ k8v4 | `fp8_e5m2` |
| --- | --- | --- |
| KV memory per token | ~3 bytes (packed) | 1 byte |
| K activation peak | **HIGHER** (club-3090 #47) | lower |
| V activation peak | lower | higher |
| Quality preservation | very high (8-bit K, 4-bit V) | high (lossy fp8) |
| Genesis kernels | P67 / P67b / P98 / P101 / … | none Genesis-specific |
| Recommended for | TP=2 (24+ GB total), high-quality | single-card tight VRAM |

**Empirical** (club-3090 #47, efschu 2× 3080 20 GB): `turboquant_k8v4`
OOMs at 90K context but `fp8_e5m2` passes verify-stress 7/7 including
91 070-token recall. Strong evidence for `fp8_e5m2` as a **safer
single-card default** when VRAM < 24 GB total.

### WSL2 specifics (club-3090 #32)

`--gpu-memory-utilization 0.85` (vs 0.92 native) leaves ~3.6 GB / card
slack for vGPU, Xwayland, and any Windows-side display interleave.
Otherwise Worker_TP1 OOMs around model load.

### PN35 status (vllm#35975 backport)

Frees ~64 MiB GPU + ~64 MiB pinned for text-only models — **necessary
but not sufficient** at 0.95 mem-util to close the 60K Cliff 2 alone.
Pairs with mem-util drop to 0.93 + the other Cliff 2b mitigations
above. PN35 has been default-on since Wave 6/v7.68; verify on via the
`apply_all` boot log.

### P103 `cu_seqlens=[0,T]` gating

Before v7.71 P103's chunked path NEVER engaged on real serving
(442/442 invocations bypassed because `cu_seqlens.shape == (2,)` is
single-seq `[0, T]`, not multi-seq). Now correctly recognised as B=1
dense and falls through to the chunked path. Note: vLLM's outer
chunked-prefill caps T at `max_num_batched_tokens=4128` (well below
P103's `_MAX_T=16384`), so the chunked path still won't fire on
default scripts unless you raise `max-num-batched-tokens` to ≥ 16384.

## Operational cookbook

Symptom → root → workaround → fix recipes, sourced from community
feedback and Genesis's own operational history.

### 1. OOM on long context — single 24 GB card

**Symptom.**

```text
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate
50.00 MiB. GPU 0 has a total capacity of 23.99 GiB of which
12.34 MiB is free.
```

Appears after 30–60 minutes of sustained long context on
1× 3090 / 4090 (24 GB), often in GDN / FFN / chunk pathway.

**Root.** Frequent `torch.empty_like(v)` allocations inside FLA / GDN
forward fragment the allocator. Each forward = new ~50 MiB
allocation; not returned to the free-list, leading to scattered free
blocks.

**Workaround.**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
sndr launch <preset>
```

**Fix.** **PN59** streaming-GDN (default-on in 27B int4 PROD configs)
— replaces full `(B, NT, H, V, K)` h-tensor materialisation with a
window-iterative driver. PROD A/B: −142 MiB / GPU at boot, −95%
per-soak fragmentation.

```yaml
genesis_env:
  GENESIS_ENABLE_PN59_STREAMING_GDN: "1"
```

**Prevention.** Single-card 24 GB → use V2 preset
`qa-qwen3.6-27b-tq-1x` (or the 3090 equivalent when landed).
`sndr preset show qa-qwen3.6-27b-tq-1x` gives the composed config
before launch. (V1 alias `a5000-1x-27b-int4-tested` retired 2026-06-01.)

**Reference.** club-3090 #58.

### 2. Qwen3Coder tool parser — indefinite SSE silence

**Symptom.** Stream request with `tool_call_parser: qwen3_coder`
hangs 30–120 s without token chunks when the response text contains
a literal `<tool_call>` (common in narrative / explanation prose).

**Root.** Parser prematurely commits to start-of-tool-call on string
match `<tool_call>` BEFORE the proper `<function=` header arrives,
then never gets the header, and the serving layer stops streaming
chunks waiting for validation.

**Fix.** **P61c** — deferred commit. Waits N tokens of slack for the
`<function=` header. If the header never arrives, flush buffered text
back to the stream. Default-on in all 6 of the 27B configs with the
qwen3_coder parser.

```yaml
genesis_env:
  GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT: "1"
```

**Reference.** club-3090 #72.

### 3. EngineCore won't start — Marlin repack OOM after nightly bump

**Symptom.**

```text
[EngineCore] FATAL: failed to load model weights:
torch.cuda.OutOfMemoryError during gptq_marlin_repack scratch allocation
```

Appears only on a freshly-pulled `vllm/vllm-openai:nightly` image;
yesterday's bump silently broke weight loading.

**Root.** Nightly image swaps the vllm / torch / quant backend
without a version pin. Marlin repack on a new version can demand a
different scratch size; for GPTQ INT4 models peak is often
`weights × 1.5×`.

**Workaround.**

```bash
# Pin to a known-good digest:
docker pull vllm/vllm-openai:nightly@sha256:<KNOWN_GOOD_DIGEST>
# Or fall back to the current Genesis pin (v12.0.0 current registry,
# canonical: 0.21.1rc0+g626fa9bba5):
docker pull vllm/vllm-openai:nightly
```

**Prevention.** Never use `:nightly` in production without a digest
pin. Operators must bump pins explicitly via the preset's
`genesis_pin` / `vllm_pin_required` fields.

**Reference.** club-3090 #60.

### 4. WSL2 — `device not ready` around 157K context

**Symptom.**

```text
RuntimeError: CUDA driver error: device not ready
```

On WSL2 + 2× 3090 + FP8 KV + chunked prefill + MTP around 157K
tokens.

**Root.** WSL2 has unique pin-memory + GPU runtime quirks. Driver /
CUDA compatibility differs from native Linux. PCIe topology through
Hyper-V abstraction adds cached transport latencies.

**Workaround.** Drop `max_model_len` to 96–128K, disable
chunked prefill (`--no-chunked-prefill`), drop spec-decode.

**Prevention.** WSL2 operators — use the long-context probe before
production. Native Linux is recommended for production deployments.

**Reference.** club-3090 #50.

### 5. Read-only mount blocks text patches

**Symptom.** Boot logs don't show Genesis patches APPLY. Requests
behave like vanilla vLLM. `sndr verify` reports 0 applied.

**Root.** Genesis text-patches modify files inside vLLM
`site-packages`. If site-packages is mounted read-only (e.g.
`--mount type=bind,...,readonly`), TextPatcher silent-fails (catches
`OSError`).

**Workaround.**

```bash
docker run -v $REPO/sndr:/usr/local/lib/python3.12/dist-packages/sndr:rw ...
```

**Prevention.** Use overlay mounts for production:

```bash
mount -t overlay overlay \
    -o lowerdir=/usr/local/lib/python3.12/dist-packages/vllm,\
       upperdir=/var/lib/sndr/overlay-upper,workdir=/var/lib/sndr/overlay-work \
    /usr/local/lib/python3.12/dist-packages/vllm
```

**Reference.** club-3090 #47.

### 6. TurboQuant + spec-decode + prefix-caching crash

**Symptom.**

```text
RuntimeError: Cannot find a satisfying assignment for DS conv state shape
```

With `--enable-prefix-caching` + MTP `accept > 1` + hybrid GDN model
(Qwen3.5 / 3.6 27B / 35B).

**Root.** Prefix cache reuses conv state across requests, but MTP can
accept multi-token causing a shape mismatch with cached state.

**Workaround.** Drop prefix caching:
`vllm serve --no-enable-prefix-caching ...`.

**Prevention.** Don't combine `--enable-prefix-caching` with MTP on
hybrid GDN models. Builtin `a5000-2x-27b-*` configs deliberately
omit prefix caching for this reason.

### 7. Cliff 2 / 2× 3080 — TurboQuant fails before 60–90K

**Symptom.** TQ3 / TQ4 presets OOM before 60K context on 2× 3080.
k8v4 passes 60K but fails at 90K. PCIe bandwidth drops to kB/s, GPU
utilization stays at 100%.

**Root.** 3080 has 10 GB VRAM (vs 24 GB in 3090 / A5000). KV cache +
scratch + activations don't fit on long contexts. PCIe throttling
indicates heavy CPU↔GPU swapping (memory thrash).

**Workaround.**

- Use k8v4 (minimum compression) instead of k3v4nc / 4bit_nc.
- Drop `max_num_batched_tokens` to 2048.
- `tensor-parallel-size 2` does NOT save you — total VRAM 20 GB is
  still too small for 27B + long context.

**Prevention.** 3080 / 20 GB cards are NOT recommended for 27B / 35B
PROD. Use 14B / 8B models on these cards.

**Reference.** club-3090 #47.

### 8. CUDAGraph + TQ + spec-decode regressions

**Symptom.** Irregular tool-call quality drops, repeating tokens
("the the of of"), or TPS spikes below baseline. Often appears after
a TQ-patch bump.

**Root.** CUDAGraph capture fixes the state of TQ k8v4 buffers with
assumptions about the spec-decode batch shape. If `spec_token_count`
changes between batches, the captured graph can read stale slots.

**Workaround.** Enable P65 (TurboQuant spec-decode CG downgrade):
`GENESIS_ENABLE_P65=1` — drops CG capture to PIECEWISE for TQ + spec
batches.

**Prevention.** Don't bump TQ patches on a working PROD without an
A/B. Run `tools/check_upstream_drift.py` before any pin update.

### 9. Container R/W layer trap on `compose stop/start`

**Symptom.** After `docker compose stop && docker compose start`
Genesis text patches don't apply — boot logs show "anchor not found".

**Root.** `compose stop/start` PRESERVES the container R/W layer
(including previously patched files). Re-running TextPatcher hits
files already modified from the previous apply; anchor search fails.

**Fix.**

```bash
docker compose down  # removes container + R/W layer
docker compose up -d  # fresh container + clean apply
```

**Prevention.** NEVER use `compose stop / start` for restart with a
new Genesis version. Always `down && up -d`.

### 10. Mac dev / no-GPU testing

**Need.** Validate configurations / patches / dispatcher on a Mac dev
rig without a GPU.

**Approach.** Genesis core is import-safe without torch (after v11
P0-1 fix).

```bash
pip install vllm-sndr-core pyyaml pytest cryptography

sndr --help
sndr launch --dry-run prod-qwen3.6-35b-balanced
sndr install --dry-run --non-interactive
python3 -m sndr.compat.schema_validator --quiet
python3 -m sndr.apply.shadow --strict
```

GPU-dependent tests skip automatically
(`@pytest.mark.requires_torch`).

## Rollback playbook

Named procedures for every major V2 / community-SDK feature. Each
entry has trigger, revert command, smoke command, and evidence
expectations. Cross-cutting principles:

1. **Never lose operator data.** Stash / snapshot before any
   destructive operation.
2. **V1 path is the floor.** V1 monolithic presets stay bench-tested
   for the duration of V2 work. Any V2 feature failure → fall back to
   V1 + smoke + evidence.
3. **Evidence first, fix second.** Every rollback creates a ledger
   entry. That entry is the input to the post-mortem that prevents
   recurrence.
4. **Time-box every waiver.** No "permanent" waiver. Every waiver has
   an expiry ≤ 30 days from creation.
5. **Public core stays public.** No rollback procedure introduces a
   private dependency or telemetry.

### R-001 — V2 alias resolution broken

**Trigger.** `sndr launch prod-qwen3.6-35b-balanced --preflight-only` returns
non-zero with `SchemaError` or `KeyError` from the V2 registry.
Typically after a YAML edit in `model_configs/builtin/` or a vLLM
pin bump that renamed an existing field.

> ⚠ **V1 fallback removed 2026-06-01** (Phase 10 sunset, commit
> `607385f1`). The old "disable V2, launch V1 preset" recovery path
> no longer works because every shipped V1 YAML was deleted. The
> only working revert is a `git revert` of the offending V2 schema
> commit (which also restores the deleted V1 baseline if the
> revert range reaches it); `GENESIS_DISABLE_V2_ALIAS=1` alone is
> insufficient.

**Revert.**

```bash
# Revert the V2 schema commit that broke resolution
git revert --no-edit <SHA_OF_BROKEN_V2_COMMIT>

# If the offending commit predates Phase 10 V1 sunset, the revert
# range will also restore the V1 baseline YAMLs; otherwise restore
# the desired V1 file from history manually:
git checkout <SHA_BEFORE_607385f1> -- \
  vllm/sndr_core/model_configs/builtin/a5000-2x-35b-prod.yaml
```

**Smoke.**

```bash
sndr launch prod-qwen3.6-35b-balanced --preflight-only
# Expect rc=0 after the revert restores a working V2 schema.
# If you restored a V1 baseline above:
sndr launch a5000-2x-35b-prod --preflight-only
```

### R-002 — Community SDK rejecting a known-good patch

**Trigger.** `sndr patches validate plugins/community/PN<n>` fails on
a patch that previously validated cleanly. Common after the shared
validator is tightened.

**Revert.**

```bash
export GENESIS_DISABLE_COMMUNITY_SDK=1
# Or narrower:
export GENESIS_SDK_SKIP_VALIDATOR=anchor_md5
```

**Smoke.**

```bash
GENESIS_ENABLE_PN<n>=1 sndr launch prod-qwen3.6-35b-balanced --preflight-only
```

### R-003 — RuntimeCommandSpec emitter divergence

**Trigger.** `sndr launch --dry-run --runtime docker` differs
semantically from `sndr launch --dry-run --runtime compose` for the
same alias. Or: docker emitter no longer matches pre-refactor golden
output.

**Revert.**

```bash
export SNDR_EMITTER_LEGACY=1
# Routes emitters back to raw ModelConfig.docker / hardware.runtime
# instead of going through RuntimeCommandSpec.
```

**Smoke.**

```bash
sndr launch prod-qwen3.6-35b-balanced --dry-run --runtime docker > /tmp/dry-docker.txt
sndr launch prod-qwen3.6-35b-balanced --dry-run --runtime compose > /tmp/dry-compose.txt
diff <(sort /tmp/dry-docker.txt) <(sort /tmp/dry-compose.txt)
# Expect: only format differences, no semantic differences.
```

### R-004 — `sndr memory explain` mis-predicting OOM

**Trigger.** Memory explain reports `SAFE` but actual launch hits
CUDA OOM, or reports `OOM_RISK` for a preset that runs fine.

**Revert.** Memory explain is informational only — it never gates
launch. If a script blocks on its output:

```bash
export SNDR_MEMORY_EXPLAIN_GATING=off
```

**Smoke.**

```bash
sndr launch prod-qwen3.6-35b-balanced --preflight-only
```

**Evidence.** Record the actual VRAM from `nvidia-smi` vs predicted
MiB. PR the new datapoint into
`tools/memory_explain_calibration/v1.yaml`.

### R-005 — Patch proof gate falsely failing release

**Trigger.** `sndr patches release-check --mode require-static`
flags a known-good patch as missing proof.

**Revert.**

```bash
# Option A — time-bounded waiver:
cat > evidence/patch_proof/_waivers/PN<n>.yaml <<EOF
patch_id: PN<n>
owner: sandermage
reason: "anchor drift across vllm 0.20.2 → 0.21.0; re-anchor pending"
expiry: '2026-06-01'
risk: low
rollback: "revert SHA <X>; patch already default_off in profile Y"
EOF

# Option B — downgrade lifecycle until proof refreshes (edit manifest):
# implementation_status: beta
```

**Smoke.** `sndr patches release-check --mode require-static`
returns rc=0 with the waiver acknowledged.

### R-006 — Server diverged from local mid-sync

**Trigger.** `make audit-dirty-state-release` fails on server with
server-only tracked modified files. Step 1 of the safe-sync recipe
caught it; Step 2 would have overwritten them.

**Revert.**

```bash
ssh server 'cd /path/to/genesis-vllm-patches && \
    git stash push -m "pre-sync server snapshot $(date -Iseconds)" && \
    git log -5 --oneline'

ssh server 'cd /path/to/... && git stash show -p > /tmp/server-changes.patch'
scp server:/tmp/server-changes.patch /tmp/server-changes.patch
# Review on local, decide, then either:
git apply /tmp/server-changes.patch        # if wanted locally
# OR
ssh server 'cd /path/to/... && git stash pop && git add -A && git commit -m "..."'
```

**Smoke.** Both hosts pass `make audit-dirty-state-release`.

### R-007 — V1 preset stops working post-Phase-9 freeze

**Trigger.** Operator runs `sndr launch a5000-2x-35b-prod` (V1 key)
post-Phase-9 and gets `DeprecationWarning` followed by failure. The
Phase-9 freeze added the warning + a no-new-V1 CI gate but does NOT
remove the V1 loader; if V1 is gone, revert the freeze SHA.

**Revert.** `git revert --no-edit <SHA_OF_PHASE_9_FREEZE>`.

**Smoke.** `sndr launch prod-qwen3.6-35b-balanced --preflight-only` exits
rc=0 with the deprecation warning printed but launch succeeds. (For
emergency V1-key smoke after a `git checkout` restore of a V1 YAML,
see "Restoring a V1 baseline from history" below.)

### R-008 — License/security gate locking out unlicensed core

**Trigger.** Fresh install of public-core repo refuses to launch
with `License required`. Public core MUST work without a license.

**Revert.**

```bash
git revert --no-edit <SHA_OF_BLOCKING_LICENSE_CHECK>
# Or short-term:
export SNDR_LICENSE_REQUIRED=0
```

**Smoke.**

```bash
sndr license status --json | jq -e '.core == "public (unlicensed)"'
sndr launch prod-qwen3.6-35b-balanced --preflight-only
```

### Restoring a V1 baseline from history (emergency rollback)

Phase 10 V1 sunset (`607385f1`, 2026-06-01) deleted every shipped V1 monolithic YAML. The V1+V2 resolver (`_resolve_preset_v1_or_v2`) still accepts a V1 key as an opaque arg for back-compat dispatch, but no V1 file is shipped so a bare V1 key resolves to a clean "preset not found" error.

If an emergency rollback to a V1 baseline is required (e.g. an unreproducible V2 regression and the operator wants to bisect against the last-known-good V1 config), restore the file from history:

> The `vllm/sndr_core/…` paths below are **intentional** — they reference the
> file's location *in history* (at `607385f1~`, before the v11→v12 `vllm/sndr_core/`
> → top-level `sndr/` rename). `git checkout <historical-SHA> -- <historical-path>`
> must use the path as it existed at that commit; do not rewrite these to `sndr/`.

```bash
# Restore the canonical 35B V1 baseline (was deleted in 607385f1)
git checkout 607385f1~ -- vllm/sndr_core/model_configs/builtin/a5000-2x-35b-prod.yaml

# Verify the V1 key now resolves
sndr launch a5000-2x-35b-prod --preflight-only

# When the V1 bisect is complete, remove the restored file (commit deletion):
git rm vllm/sndr_core/model_configs/builtin/a5000-2x-35b-prod.yaml
git commit -m "revert: remove emergency-restored V1 baseline post-rollback"
```

This path is documented because the V2 schema migration is recent (2026-06-01); the V1 baseline file content is byte-equivalent across the last 12 dev tip commits before its deletion. Operators bisecting V2 schema bugs MAY need this recovery hatch until the V2 schema has gone through a couple of pin bumps without operator-facing incidents.

## Enterprise operator runbook

Operational guidance derived from the 2026-05-30 multi-model
session. Captures the launcher-config invariants every PROD container
must meet to reach enterprise-grade reliability.

### Launcher invariants (all PROD launchers)

Every PROD launcher MUST contain the following docker-run / vllm-serve
arguments. Missing ANY of them is a Class-1 config drift that breaks
operator-visible features silently:

| Argument | Required for | Failure mode if missing |
|---|---|---|
| `--enable-auto-tool-choice` | tool-call API | server returns `400: requires --enable-auto-tool-choice` |
| `--tool-call-parser <name>` | tool-call extraction | tool_calls[] always empty |
| `--reasoning-parser <name>` | thinking-then-tool flow | model emits `<channel>thought` as content, never closes |
| `--override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20}'` | deterministic tool-call | non-deterministic 3-7/7 across runs |
| `--served-model-name <name>` | client API stability | clients hit the model path basename, breaks routing |
| `--api-key <key>` | auth | requests return `401: Unauthorized` |
| `-e GENESIS_ENABLE_<PATCH>=1` (per active patch) | per `default_on=True` strict-opt-in policy 2026-05-17 | patch registered but never fires (silent no-op) |

The default_on=True informational policy (Class-12 bug — see Bug
Classes section below) means EVERY patch validated for the model needs
its env flag set explicitly in the launcher, even when the registry
says `default_on=True`. The registry value is documentation; the
launcher is the source of truth.

### Bug Class 12 — strict-opt-in policy

**Policy**: per Sander directive 2026-05-17
(`sndr/dispatcher/decision.py:355-386`), `default_on=True`
in the registry does NOT auto-apply the patch. Every patch — including
those marked `default_on=True` — requires its `GENESIS_ENABLE_*` env
var explicitly set in the launcher script. Escape hatch:
`GENESIS_LEGACY_DEFAULT_ON=1` (pre-2026-05-17 semantics).

**Detection**:

```bash
docker exec <container> python3 -m sndr.cli self-test
docker exec <container> cat /tmp/genesis_boot.log | grep -E '<PATCH>: (skipped|applied)'
docker inspect <container> --format '{{.Config.Env}}' | grep GENESIS_ENABLE_<patch>
```

**Fix when caught**: append `-e GENESIS_ENABLE_<patch>=1` to the
launcher's `docker run` block. There are ~52 patches with
`default_on=True` in the registry — each requires explicit
enablement.

### Bug Class 13 — model-quality intrinsic limits

**Status update 2026-05-31**: the gemma4-31B `5/7 deterministic`
result that originally motivated Class 13 is now **resolved** —
it was NOT a model intrinsic limit. The two failures-of-seven were
caused by the v1 G4_T1 vendor (PR #42006 segment-replay) mis-handling
MTP K=4 multi-delimiter-per-delta streaming events, compounded by a
host-side memory tune issue. With the G4_T1 v2 vendor (PR #42237
hermes-style accumulated-text rescan) AND gpu-memory-utilization
lowered from 0.92 → 0.80 (gives Genesis P38 TQ continuation-prefill
workspace its 1 GiB alloc headroom), the bench achieves 35/35 across
7 edge cases × 5 runs in both streaming and non-streaming modes (see
`tools/g4_tool_bench.py`). Document retained for future cases that
genuinely hit a model-intrinsic ceiling — the diagnostic procedure is
still valid, the gemma4-31B example was misdiagnosed.

**Symptom (generic)**: tool-call bench shows a sub-100% rate that
reproduces deterministically across many consecutive runs, with the
failing cases failing in exactly the same way every time AND no
obvious config drift (env flags match YAML, launcher mounts correct,
parser overlay verified).

**Diagnostic procedure** (apply in order):

1. **Greedy decode**. Run with `temperature=0 top_k=1 top_p=1.0`. If
   failures are now consistent across runs but still non-zero → step 2.
2. **Use streaming-aware bench harness**. Some bugs only manifest in
   streaming SSE path (where the tool parser sees deltas) not in the
   non-streaming `extract_tool_calls` path. Run BOTH modes and
   compare. If streaming fails but non-streaming passes → parser bug
   (check vendor PR list at `dispatcher/registry.py:G4_T1`).
3. **Disable HTTP keep-alive**. Add `Connection: close` to the bench
   harness request headers. If failures drop → parser instance is
   leaking state between requests on the same socket (the gemma4-31B
   case 5 nested-object failure was exactly this — filed back to PR
   #42237 thread for fix in the underlying parser).
4. **Disable MTP**. K=4 speculative decoding amplifies model
   non-determinism slightly. If failures drop with K=1 or K=0 → spec
   decode is contributing.
5. **Step the model precision UP one tier** (AWQ-4bit → FP8 or
   FP16). If failures STILL persist at higher precision → genuine
   Class 13 model intrinsic limit. Otherwise → quantization-induced.

Only after all five steps fail to bring the bench to 100% should
the result be labeled Class 13 in the operator runbook.

**Mitigation if Class 13 IS confirmed** (no current PROD case as of
2026-05-31): use a higher-precision model variant, or accept the
intrinsic rate and report it explicitly in operator docs.

### Pin policy (2026-05-30 unification)

The canonical PROD pin is `vllm/vllm-openai:nightly` (currently
resolves to commit `626fa9bb` per the K.1.R.R promotion 2026-05-30).
The previous canonical pin is `vllm/vllm-openai:nightly-previous`
(currently `bf610c2f5` = dev371). All other intermediate nightly tags
should be cleaned via `docker rmi` periodically (operator hygiene).

The 4 PROD models target the same `:nightly` tag in their launchers
to maintain pin uniformity. Exception: `gemma4-31B` may be temporarily
pinned to `:nightly-previous` if a 626fa9bb regression is observed
(analog to A11 guardrail for 27B's `-8.66%` TPS regression on the new
pin). Such pin pinning MUST be documented in the launcher header
comment.

### Bug Class 14 — Genesis P38 TQ workspace OOM on first complex request

**Symptom**: container boots cleanly, the first 1-2 inference requests
succeed (especially short prompts), and then the N-th request — typically
the first one with a longer-than-trivial prompt or a tool-call with CoT
preamble — returns HTTP 500 to the client and the container's APIServer
shuts down. `docker logs` shows
`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB`
inside
`sndr/engines/vllm/patches/attention/turboquant/p38_tq_continuation_memory.py:_genesis_continuation_prefill`,
specifically at the `GPB.get_or_create(ns_k_full, full_max_shape, ...)`
call site.

**Real example (2026-05-31 session)**: gemma4-31B AWQ-4bit, TQ k8v4,
MTP K=4 launched with `--gpu-memory-utilization 0.92` and
`--max-model-len 4096` (intentionally tiny ctx). The 1st bench prompt
("What's the weather in Tokyo?") passed in 6.9s. The 2nd prompt
("Plan an outdoor picnic in Berlin tomorrow, think step by step, then
check the weather") triggered the workspace allocation, OOMed, and
killed the container. Same pattern reproduced after lowering to 0.85.
Was finally resolved at 0.80.

**Root cause**: Genesis P38 TQ continuation prefill allocates its
`k_full_buf` workspace LAZILY via the Genesis
`GenericPersistentBuffer.get_or_create` pool, which lives OUTSIDE
vLLM's `gpu-memory-utilization` accounting. vLLM happily reserves up
to (utilization × VRAM) for KV cache + activations + cudagraph capture,
leaving NO margin for the 1 GiB Genesis workspace. The workspace only
triggers on the FIRST request that needs continuation prefill of a
long-enough sequence — short tool-call prompts skip the path entirely,
hiding the trap until the first complex query in production.

**Detection**:

```bash
ssh <operator>@<server> 'docker logs <container> 2>&1 | \
    grep -A2 "OutOfMemory.*p38_tq_continuation_memory" | head -10'
```

If that pattern is in the logs, you have Class 14. The container will
have exited (status `Exited (1)` in `docker ps -a`).

**Fix**: lower `--gpu-memory-utilization` in the launcher to ≤ 0.85
(empirically 0.80 is safe for AWQ-4bit + TQ k8v4 + MTP K=4 on 24 GiB
GPUs at `max-num-seqs=1` and `max-model-len 4096`). The exact safe
ceiling depends on model size, KV dtype, MTP K, and context budget —
the rule of thumb is "leave ≥ 1.5 GiB free per card after vLLM
reservation when TQ is enabled".

**Configuration risk audit** (rig 2026-05-31): the current 4 PROD
launchers carry these `gpu-memory-utilization` values:

| Model | Util | TQ | MTP K | At-risk |
|---|---:|:---:|:---:|:---|
| qwen3.6-27b PN95 | 0.92 | yes | 3 | YES — operator should monitor for Class 14 on first long-ctx tool-call |
| qwen3.6-35b A3B | 0.90 | yes | 3 | LOWER (35B uses 280K ctx + already documented 1.0-1.5 GiB headroom in YAML) |
| gemma4-26b AWQ | 0.88 | no | 0 | NO (kv_cache_dtype: auto, no TQ workspace) |
| gemma4-31b AWQ | 0.80 | yes | 4 | NO (post-fix this session) |

The 27B PN95 launcher is the most exposed (Util 0.92 + TQ + MTP). The
workspace size on 27B is smaller than 31B's (smaller model + fewer
heads), so it MAY survive higher utilization in practice — empirically
the 27B has run for weeks at 0.92 without Class 14 reports. But the
threshold is operator-monitored, not enforced by config. If a Class 14
event is observed in 27B production logs, lower to 0.85 in the
launcher per the same fix.

**Prevention going forward**: when introducing TQ + MTP K≥3 on Ampere
consumer GPUs (24 GiB or less), default the launcher to
`gpu-memory-utilization 0.80` and bench up from there. Don't ship at
0.92 just because vLLM accepts it.

### Bug Class 15 — qwen3_coder streaming parser drops tool-calls into content

**Status update 2026-05-31**: RESOLVED via Genesis Q3_T1 v1 overlay
(see commit at end of this section). The class-level diagnostic
below stays for historical context; the operational fix is now
shipped via launcher bind-mount on both 27B and 35B launchers.

**Symptom (historical, before fix)**: agent client receives
`finish_reason: "stop"` instead of `tool_calls`, with
`message.tool_calls = []` (or `delta.tool_calls` empty across all
SSE chunks) and the raw XML
`<tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</tool_call>`
leaking through `message.content` (or `delta.content`). The same prompt
sent in NON-streaming mode correctly returns parsed `tool_calls`.

**Real example (2026-05-31 4-model bench session)**: identical 7-case
bench harness, same greedy decode (T=0, top_k=1, top_p=1.0), same
`Connection: close` header per Class 14 guidance:

| Model | Tool-parser | Non-stream | Stream |
|---|---|---:|---:|
| qwen3.6-27b-int4-AutoRound + TQ + MTP K=3 | qwen3_coder | 7/7 | 1/7 |
| qwen3.6-35b-A3B-FP8 + TQ + MTP K=3 | qwen3_coder | 7/7 | 2/14 |
| gemma4-26b-A4B AWQ + auto KV | gemma4 (base, no overlay) | 7/7 | 6/7 |
| gemma4-31b AWQ + TQ + MTP K=4 + G4_T1 v2 | gemma4 (PR #42237) | 7/7 | 35/35 |

Both qwen3_coder rows fail streaming heavily. Both gemma4 rows pass
streaming once the appropriate vendor overlay is mounted (v2 on the
31B; the 26B has NO overlay yet, hence the residual 1/7 fail on the
two-tools-one-response case).

**Root cause hypothesis (analog to gemma4 PR #42006/#42237 class)**:
the qwen3_coder parser's `extract_tool_calls_streaming()` does not
correctly buffer the XML start/end tag boundaries across SSE deltas
when MTP K≥3 packs multiple boundary events into one chunk. The
`extract_tool_calls()` non-streaming method works fine because it
sees the entire output as a single string. NOT confirmed by reading
upstream source on this pin yet — flag for the qwen3_coder
maintainer when filing.

**Detection**:

```bash
curl -s -X POST http://localhost:8101/v1/chat/completions \
  -H "Authorization: Bearer genesis-local" \
  -H "Content-Type: application/json" \
  -H "Connection: close" \
  -d '{"model":"qwen3.6-27b","stream":true,"tools":[...],
       "tool_choice":"auto","messages":[...]}' \
  | grep "data:" | head -20
```

If the SSE deltas show `delta.content` carrying `<tool_call>...`
XML AND `delta.tool_calls` is consistently null/empty for the
whole stream, you have Class 15.

**Fix shipped 2026-05-31 — Genesis Q3_T1 v1 overlay**:

The Genesis-original Hermes/Kimi-style rewrite of
`extract_tool_calls_streaming` lives at
`sndr/engines/vllm/patches/tool_parsing/q3_t1_qwen3coder_tool_parser_overlay.py`.
It replaces the upstream state machine with the same
accumulated-text rescan + diff pattern used by G4_T1 v2 (PR #42237
for gemma4):

1. On each delta, rescan the full accumulated `current_text` for
   ALL tool-call regions using the upstream parser's existing
   `tool_call_function_regex` (which already handles both complete
   and partial function-call forms).
2. For each region, build the args dict using a STRICT parameter
   regex that requires the closing `</parameter>` tag — the
   in-flight last parameter is intentionally excluded so its value
   cannot flip type (e.g. `""` placeholder becoming `4` once the
   int value arrives) and break the stable-prefix diff invariant.
3. Compute `_q3_stable_partial_json_prefix(args_json)` to trim
   trailing structural chars (`}`, `"`, `]`) for in-flight tool
   calls so the streamed prefix can be extended by the next
   delta's re-parsed JSON. When the function regex's complete
   alternative matches (closing `</function>` present), the full
   JSON including closing chars is emitted in one final delta.
4. Diff vs `_q3_streamed_args[index]` and emit only the new tail
   per index. A defensive common-prefix reset guards against rare
   re-parse prefix changes from type-coercion flips on partial
   values.

The non-streaming `extract_tool_calls()` path stays untouched and
is the byte-for-byte oracle that the streaming JSON matches at
request finalization.

**Deployment** — bind-mount via launcher `-v`:

```bash
-v /tmp/qwen3coder_tool_parser_FIXED.py:/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/qwen3coder_tool_parser.py:ro
```

The rig launchers (`start_pn95_2xa5000_test.sh` and
`start_35b_prod_wave8.sh`) mount `/tmp/qwen3coder_tool_parser_FIXED.py`
which is `scp`-mirrored from the in-repo overlay file.

**Empirical result (2026-05-31, pin 0.21.1rc1.dev354+g626fa9bba,
`g4_tool_bench.py` with `Connection: close`, greedy decode)**:

| Model | Stream (pre-Q3_T1) | Stream (with Q3_T1) | Non-stream |
|---|:---:|:---:|:---:|
| qwen3.6-27b int4 + TQ + MTP K=3 | 1/7 (14%) | **35/35** ✓ | 7/7 ✓ |
| qwen3.6-35b A3B-FP8 + TQ + MTP K=3 | 2/14 (14%) | **35/35** ✓ | 7/7 ✓ |
| gemma4-26b A4B AWQ (G4_T1 v2) | 6/7 (86%) | **35/35** ✓ | 7/7 ✓ |
| gemma4-31b AWQ + TQ + MTP K=4 (G4_T1 v2) | already 35/35 | **35/35** ✓ | 7/7 ✓ |

**Bug Class 15 is RESOLVED at PROD across all 4 models.** Both
streaming and non-streaming paths now return correctly-parsed
`tool_calls` arrays with no XML leakage into `delta.content`.

**Production-scale stress test (2026-05-31, 100 runs × 7 cases ×
4 models = 2800 requests total):**

| Model | Parser overlay | Result | Duration |
|---|---|---|---|
| gemma4-31B AWQ + TQ + MTP K=4 | G4_T1 v2 (PR #42237) | **700/700 ✓** | 10 min |
| qwen3.6-27B int4 + TQ + MTP K=3 | Q3_T1 v1.3 (Hermes-style) | **700/700 ✓** | 27 min |
| qwen3.6-35B A3B-FP8 + TQ + MTP K=3 | Q3_T1 v1.3 (Hermes-style) | **700/700 ✓** | 9 min |
| gemma4-26B A4B AWQ + MTP K=3 | G4_T1 v2 (PR #42237) | **700/700 ✓** | 5 min |
| **TOTAL** | | **2800/2800 = 100%** | ~51 min |

User mandate `тулы должны быть на 100% исправны хоть 5 из 5 хоть
100 из 100` is empirically validated at production scale across
all 4 PROD models, both gemma4 and qwen3_coder parser families.

**Retire trigger**: when upstream lands a Hermes/Kimi-style rewrite
of `qwen3coder_tool_parser:extract_tool_calls_streaming` and our
pin bumps to include it — diff byte-by-byte against this overlay
and retire if equivalent. Track:

```bash
gh search prs --repo vllm-project/vllm \
  --label tool-calling --state open --search "qwen3 streaming"
```

**Configuration risk matrix (RESOLVED state)**:

| Model | Stream tool-calls | Path |
|---|:---:|---|
| qwen3.6-27b PN95 | 100% | Q3_T1 v1 overlay deployed |
| qwen3.6-35b A3B | 100% | Q3_T1 v1 overlay deployed |
| gemma4-26b AWQ | 100% | G4_T1 v2 overlay deployed (commit ca4d9bae) |
| gemma4-31b AWQ | 100% | G4_T1 v2 overlay deployed (commit a074e900) |

**Throughput data — workload-aware** (greedy decode, `Connection:
close`, 300-token completion). The first row of TPS the operator
sees on a fresh container depends on WHICH workload they use. MTP
K≥3 with TQ k8v4 was specifically tuned for STRUCTURED workloads
(JSON output, code, tool-calls, counted lists) per the
2026-05-20 ModelDef-migration audit; chat / summarization
workloads see different (lower) numbers because MTP acceptance
collapses on free-form prose.

Per-workload single-request TPS (3 samples averaged, `Connection:
close`):

| Model | Chat | Code | Counted list | JSON output |
|---|---:|---:|---:|---:|
| qwen3.6-27B int4 + TQ + MTP K=3 | 91.68 | 113.53 | 117.79 | 105.61 |
| qwen3.6-35B A3B-FP8 + TQ + MTP K=3 | 182.15 | 215.96 | 218.26 | 224.60 |
| gemma4-26B A4B + AWQ | 140.95 | 214.67 | 229.12 | 226.31 |
| gemma4-31B AWQ + TQ + MTP K=4 | 23.54 | 41.73 | 52.37 | 53.47 |

The structured-workload numbers match or exceed the reference table
above (27B 116-125 ✓, 35B 210 ✓, 26B 114 → actually 226 = 2× ✓,
31B 28-39 → 53 ✓). The chat-workload numbers are systematically
lower because MTP K≥3 acceptance collapses on free-form prose
(documented in `docs/_internal/MTP_TQ_GEMMA4_*` audit chain and the <!-- audit-public-docs: allow -->
2026-05-20 migration note).

Per-workload multi-request scaling on structured_json workload:

| Model | Single | conc=2 aggregate | conc=4 aggregate |
|---|---:|---:|---:|
| qwen3.6-27B (max-num-seqs=4) | 105.61 | 179.42 (1.70×) | 299.04 (2.83×) |
| qwen3.6-35B (max-num-seqs=2) | 224.60 | 265.57 (1.18×) | n/a |
| gemma4-26B (max-num-seqs=2) | 226.31 | 252.01 (1.11×) | n/a |
| gemma4-31B (max-num-seqs=1) | 53.47 | n/a | n/a |

All three multi-capable models scale POSITIVELY on the
structured workload. (Earlier 2026-05-31 measurements showed
apparent "0.5×" regression at conc=2 on 35B and 26B — that was a
**workload-mismatch artifact**: the chat prompt drives MTP
acceptance to near-zero, which means each batch slot wastes the
draft kernel cost without recovering it via accepted tokens. On
structured workloads the pattern reverses and concurrency
helps as expected.)

**Operator rule of thumb**: when benching to compare against the
reference TPS numbers above, use structured-workload prompts
(JSON output, code, tool-call args, counted lists). Free-form
chat prompts will give 30-70 % lower TPS on any TQ + MTP K≥3
launcher and that is *not* a config bug — it is the MTP
acceptance profile interacting with the workload distribution.

**MTP K sweep on gemma4-31B (added 2026-05-31)**: The structured
launcher currently ships with `num_speculative_tokens=4`. Three
alternative K values were A/B benched on the same 4-workload
spread to characterize the speed-quality tradeoff:

| Workload | MTP-OFF | K=3 | **K=4 (default)** |
|---|---:|---:|---:|
| chat | 19.92 (-15.4%) | **44.76 (+90.2%)** ✓ | 23.54 |
| code | 22.16 (-46.9%) | 43.50 (+4.2%) | 41.73 |
| count | 21.40 (-59.1%) | 44.09 (-15.8%) | **52.37** |
| json | 36.68 (-31.4%) | 41.32 (-22.7%) | **53.47** |

Findings:

  1. **MTP-OFF is universally worse** (-15% to -59%) on this config.
     This empirically refutes the older 2026-05-19 milestone note
     that claimed "drafter acceptance is 0%, MTP is no-op or slower"
     — that was true at the milestone's pin/state but the
     G4_71..G4_76 architectural stack + PN256/PN259/PN261/PN271
     dev-chain extras that the launcher ships TODAY *do* make MTP
     accept tokens. Removing MTP loses 30-60% on structured
     workloads and 15% on chat.

  2. **K=4 wins big on structured workloads** (count +18% over K=3,
     json +29% over K=3). With high-redundancy outputs the drafter
     hits ≥80% acceptance per step and the K=4 lookahead recovers
     ~4× tokens per forward.

  3. **K=3 wins big on chat workloads** (+90% over K=4!) because
     free-form prose has lower per-token predictability — K=4
     wastes time on draft-rejection cycles that K=3 avoids.

  4. **K=2 crashed at boot** in this combo (vllm-gemma4-31b-k2-k4
     container exit code 1 right after Genesis dispatcher apply
     log; root cause not chased because K=2 wasn't strictly better
     than K=3 on any axis our user values).

**Operator recommendation**: pick K based on the dominant workload of
the actual production traffic mix:

  - **Chat-heavy aggregation** → switch the gemma4-31b structured
    launcher's `num_speculative_tokens` from 4 to 3, accept the
    -16% / -23% on structured-count / structured-json in exchange
    for +90% on chat.
  - **Structured-heavy** (JSON output, agent tool-call args, code
    completion, counted lists) → keep K=4 as-is.
  - **Mixed traffic** → run two launchers on different ports
    (`start_gemma4-31b_k3.sh` for chat-role, the current
    `start_gemma4-31b-tq-mtp-structured-k4.sh` for structured-role) and
    have the aggregator route by detected request shape (tool_calls
    present → structured, free-form `messages[*].content` → chat).
    This pair of launchers is created on the rig but neither is the
    default at session-close — operator must decide before promoting.

Other 3 PROD models (27B / 35B / 26B) ship K=3 — this turned out
to be empirically optimal across all 4 models tested (see K-sweep
table below). The K=4 outlier was a 2026-05-23 migration choice
for the structured-role profile when MTP did not yet have a usable
acceptance rate (see
`docs/_internal/GEMMA4_MTP_MODELDEF_MIGRATION_AUDIT_2026-05-20.md`). <!-- audit-public-docs: allow -->
The current empirical state warrants revisiting that choice — and
the K=3 chat-role profile shipped 2026-05-31 (see next section)
is the architectural answer.

**Full K-sweep across all 4 PROD models (2026-05-31)** — to verify
whether the 31B K=3-beats-K=4 finding generalizes:

| Model | K=current | K=alt | chat Δ | code Δ | count Δ | summarization Δ | json Δ | Verdict |
|---|:---:|:---:|---:|---:|---:|---:|---:|---|
| 27B int4+TQ | K=3 | K=2 | -7.9% | -8.0% | -8.8% | n/a | -1.7% | K=3 stays optimal |
| 35B A3B-FP8 | K=3 | K=2 | -5.7% | -12.0% | -11.3% | -14.5% | -13.1% | K=3 stays optimal |
| 26B A4B AWQ | K=3 | K=2 | +3.4% | -7.0% | -10.2% | +7.0% | -10.7% | K=2 too small; see K=4 row |
| 26B A4B AWQ | K=3 | K=4 | **+15.0%** | -4.7% | -7.0% | **+10.9%** | -0.5% | **K=3 split shipped (commit 72435282)** |
| 31B AWQ+TQ | K=4 | K=3 | **+112%** | -6% | -24% | **+99%** | -21% | **K=3 split shipped (commit 284477f9)** |

The 31B case is the OUTLIER. On 27B/35B/26B the drafter has high-enough
per-token acceptance even on free-form prose that K=3 already pays for
itself; dropping to K=2 leaves spec-decode parallelism on the table.
The 31B drafter (different model + different assistant arch) has
lower per-token acceptance on chat workloads so K=4 wastes cycles
that K=3 avoids — and the gain is large enough (+105.7% on free-form
mean) to justify a permanent V2 chat-role profile.

**Conclusion**: K-tuning is **per-MODEL**, not universal. Of the 4
PROD models, ONLY 31B has a measurable K-axis win available. The
fix landed 2026-05-31 as the
`gemma4-31b-tq-mtp-chat-k3` V2 profile + matching FunctionalArtifact +
preset (commit 284477f9). The architectural delivery is described
in the next section.

### Multi-profile architecture (V2 ProfileDef + FunctionalArtifact + Gateway)

The Genesis V2 ModelDef + ProfileDef + FunctionalArtifact +
spec_decode Gateway stack already implements **per-workload runtime
routing** at the deployment layer. The user-facing primitives:

```text
ModelDef (sndr/model_configs/builtin/model/<id>.yaml)
   spec_decode: null  ← post-Variant-A; profiles supply spec config
       ↓
ProfileDef (sndr/model_configs/builtin/profile/<id>.yaml)
   role: structured     spec_decode_override: {method, K, model}
   compression_plan + backend_plan + routing.intended_workloads
   validation: {artifact_id, config_hash}
       ↓
FunctionalArtifact (sndr/engines/vllm/patches/spec_decode/artifacts/<id>.json)
   decision: validated_conditional | validated_global | denied
   allowed_workloads + denied_workloads + per_class TPS metrics
       ↓
Preset (sndr/model_configs/builtin/presets/prod-<key>.yaml)
   model + hardware + profile  ← composition triplet
       ↓
sndr launch prod-<key>           — renders + execs vLLM via canonical CLI
   - role=default     → MTP OFF upstream (production-safe baseline)
   - role=structured  → MTP-on upstream (workload-gated)
       ↓
spec_decode Gateway (port 8100)
   request_router.select_profile(body, artifact)
       ├─ response_format json_object | json_schema  → structured
       ├─ tool_choice required / dict.type=function → structured
       ├─ extra_body.workload_class in allowed_set  → structured
       └─ else (no explicit signal)                 → default
   → forwards to default upstream (e.g. 8101) or structured (8102)
```

The router NEVER reads prompt text — false-positives cost -50% TPS,
false-negatives cost zero, so the gate is conservative by design
(see `sndr/engines/vllm/patches/spec_decode/request_router.py`).

**Gateway env knobs**:

```bash
GENESIS_GATEWAY_DEFAULT_URL=http://localhost:8101      # MTP-OFF upstream
GENESIS_GATEWAY_STRUCTURED_URL=http://localhost:8102   # MTP-K=4 upstream
GENESIS_GATEWAY_PROFILE=gemma4-31b-tq-mtp-structured-k4    # artifact lookup key
GENESIS_GATEWAY_BIND_PORT=8100                         # client-facing port
GENESIS_GATEWAY_HEALTH_INTERVAL=5
GENESIS_GATEWAY_TIMEOUT=120
GENESIS_SPEC_DECODE_ARTIFACTS_DIR=...                  # extra artifacts dir
```

Run:

```bash
python3 -m sndr.cli gateway
```

**Profile inventory** as of 2026-05-31:

| Model | Default-role profile | Structured-role profiles |
|---|---|---|
| qwen3.6-27B int4 | (none — direct launch) | qwen3.6-27b-tq-k8v4 / qwen3.6-27b-dflash / qwen3.6-27b-multiconc |
| qwen3.6-35B A3B FP8 | (none — direct launch) | qwen3.6-35b-balanced / qwen3.6-35b-dflash / qwen3.6-35b-multiconc |
| gemma4-26B A4B AWQ | gemma4-26b-no-mtp + gemma4-26b-multiconc-k1 | gemma4-26b-mtp-k4 + gemma4-26b-multiconc + **gemma4-26b-mtp-chat-k3** (new 2026-05-31) |
| gemma4-31B AWQ + TQ | gemma4-31b-tq-default (MTP OFF) | gemma4-31b-tq-mtp-structured-k4 + **gemma4-31b-tq-mtp-chat-k3** (new 2026-05-31) |

**The new gemma4-31b-tq-mtp-chat-k3 profile** (commit 284477f9) is the
load-bearing architectural delivery from the K-sweep finding. It is a
role-structured profile (carries spec_decode + compression + backend
+ routing + validation blocks) but its `routing.intended_workloads`
is the COMPLEMENT of structured-k4's set:

```yaml
# gemma4-31b-tq-mtp-structured-k4.yaml
routing:
  intended_workloads: [structured_count, tool_json]

# gemma4-31b-tq-mtp-chat-k3.yaml
routing:
  intended_workloads: [free_chat, code_gen, summarization]
```

…and the artifacts mirror this partition. A gateway running both
upstreams routes per request signal to whichever upstream's artifact
ALLOWS that workload class; the two profiles partition the 5-class
suite cleanly with no overlap and no gap.

**How an operator promotes the chat-k3 deployment**:

1. Render + start the chat-k3 upstream on a sibling port:
   ```bash
   sndr launch prod-gemma4-31b-tq-mtp-chat-k3 --port 8113
   ```

2. Keep the structured-k4 upstream running on its current port:
   ```bash
   bash ~/start_gemma4-31b-tq-mtp-structured-k4.sh   # port 8102
   ```

3. Start the gateway pointed at both:
   ```bash
   export GENESIS_GATEWAY_DEFAULT_URL=http://localhost:8113   # chat-k3
   export GENESIS_GATEWAY_STRUCTURED_URL=http://localhost:8102 # structured-k4
   export GENESIS_GATEWAY_PROFILE=gemma4-31b-tq-mtp-structured-k4
   python3 -m sndr.cli gateway   # port 8100
   ```

4. Aggregator / client points at the gateway (`localhost:8100`)
   instead of either upstream directly. The request_router decides
   per request which upstream to forward to.

5. After a multi-day observation window confirms aggregator TPS
   gain matches the artifact's projection, promote
   `gemma4-31b-tq-mtp-chat-k3.status` from `experimental` to `validated`
   and update the operator card maturity from `draft` to `validated`.

   (As of 2026-06-01 the public tree already ships the profile at
   `status: validated` — promotion completed in commit `16eea839`
   after a 3× × 5-workload bench cycle met the +50% chat geomean
   threshold; the preset card `maturity` field stays at `draft` until
   an operator's own A/B observation window confirms the same gains
   on their workload. See CHANGELOG 2026-06-01 entry "chat-K3
   promotion resolved" for the full root-cause analysis.)

### Observability stack (PN287 + PN288 + PN289 + trace surface)

The enterprise observability flow combines four Genesis surfaces:

1. **PN287** (qwen3_coder args observer) — labeled Counter at
   `vllm:qwen3_tool_parser_pn287_*_total{model, ctx_bucket}` surfacing
   malformed tool_call.arguments by model + context-depth bucket.
2. **PN288** (finish_reason override) — labeled Counter at
   `vllm:pn288_finish_reason_override_total{model, channel, action}`
   surfacing dry-run / would-downgrade / downgraded / kept events.
3. **PN289** (process info) — gauge `genesis_process_info` with
   value 1 labeled `(preset, profile, workload_class, K, backend,
   patch_hash, model, pin)`. JOIN against any vllm-builtin metric
   via `* on(instance) group_left(...) genesis_process_info`.
4. **Trace surface** (`sndr trace list/collect/summarize` + `sndr
   support-bundle`) — collects 11 known diagnostic trace files
   (`/tmp/genesis_*.log`) into a single tarball for off-rig
   analysis.

Activation (each is opt-in via env flag):

```bash
-e GENESIS_ENABLE_PN287_QWEN3CODER_ARGS_OBSERVER=1
-e GENESIS_ENABLE_PN288_TOOL_FINISH_REASON_OVERRIDE=1
-e GENESIS_PN288_DRY_RUN=1   # default ON when PN288 enabled — Phase B
-e GENESIS_ENABLE_PN289_PROCESS_INFO=1
-e GENESIS_PRESET=prod-qwen3.6-35b-balanced
-e GENESIS_PROFILE=qwen3.6-35b-balanced
-e GENESIS_WORKLOAD_CLASS=free_chat
```

PN288 Phase C activation runbook (after 2-4 weeks of Phase B data —
the operator decides):

```bash
-e GENESIS_PN288_DRY_RUN=0   # activates actual finish_reason override
-e GENESIS_PN288_MIN_ARGS_LENGTH=5   # default
-e GENESIS_PN288_MAX_ARGS_LENGTH=200 # default; tighten to empirical p10/p90
```

### Bench expectations per model (PROD reference)

These are the canonical bench targets every PROD container should
hit on the unified `:nightly` pin. Bench command:

```bash
python3 tools/genesis_bench_suite.py --quick --ctx 8k \
    --host localhost --port <port> --api-key <key> \
    --model <served-model-name> \
    --skip-stress --skip-ctx-probe --skip-multi-turn
```

| Model | wall_TPS | TPOT ms | TTFT ms | CV % | Tool-call (positive) |
|---|---:|---:|---:|---:|:---:|
| 27B Lorbus INT4 TQ | 116-125 | 8-9 | 110-120 | 3-10 | **7/7** ✓ |
| 35B A3B FP8 | 210 | 4.4 | 117 | 5-7 | **7/7** ✓ |
| gemma4-26B-A4B MoE | 114 | 6.0 | 71 | 22-30 | **7/7** ✓ |
| gemma4-31B Dense TQ MTP-K4 | 28-39 | 6-9 | 70-170 | 10-30 | **35/35** ✓ (G4_T1 v2 + Connection: close + gpu-mem-util 0.80; both streaming + non-streaming, 7 cases × 5 runs) |

A more than 10% deviation from these on a fresh boot is suspicious
and warrants investigation via:

1. `sndr trace list --container <c>` to see what's being written.
2. `sndr support-bundle --container <c>` to gather host + container
   facts for off-rig analysis.
3. `docker exec <c> python3 -m sndr.cli self-test`
   to verify the registry contract.

### Known operator decisions deferred

These items appear in the operator's internal planning notes and
explicitly need operator approval — they are NOT bugs:

| Item | Status | Operator action |
|---|---|---|
| A11 | 27B PROD stays on dev371 until regression mechanism identified | hold; bench shows -8.66% TPS on 626fa9bb (verified empirically 2026-05-30) |
| §3.4 | 27B nsys profile (45 min rig + 2-4h parse) | scheduling decision |
| §3.1 / PN288 Phase C | flip `GENESIS_PN288_DRY_RUN=0` after 2-4 weeks of evidence | data-driven, see PN288 runbook above |
| Q2 | Public `origin/dev` push | per current policy: **NOT touched**, only private `sndr-dev/dev` is synced |
| G9 | CONFIG-UX D.9 Codex ProfileDef sub-fields | review + approve |
| H1, H5, P6, P7 | Codex PN282/283 + ProfileDef adoption decisions | review + approve |
| Phase 5.3 | `--kv-cache-dtype-skip-layers` adoption (vllm#33695, merged) | **NOT APPLICABLE to current 4 PROD launchers** — feasibility verified 2026-05-31. Flag only applies when `kv-cache-dtype` is FP8. Our launchers: 27B + 35B + 31B use `turboquant_k8v4` / `turboquant_4bit_nc` (Genesis TQ stack, 5.33× compression); 26B uses `auto` (BF16). For Phase 5.3 to engage, operator would need to switch a launcher's KV path from TQ → FP8 + skip-layers, accepting +0.5-1.5pp quality on sliding-window attention layers in exchange for losing the TQ stack's higher compression ratio. The 35B FP8-weights model is the closest fit if the operator wants to A/B it; the gemma4 models gain less because their sliding-window pattern is denser. |
| Phase 5.5 | vllm#26504 DynamicProposer adoption | research + adapt — STUDY done 2026-05-31 (commit 324fe572). Vanilla PR extends `EagleProposer`, not MTP. Our 4 PROD models all use MTP K=3/K=4. Backport requires MTP-adapt (inherit from MTP proposer instead of Eagle, or extract mixin). Wait for upstream rebase, then ~2-3 day vendor effort. Acceptance-threshold + hysteresis algorithm is generic enough to lift. |

## Reporting new cliffs

If you hit something that isn't here, please:

1. Capture `sndr doctor --json` output.
2. Capture `docker logs <container>` last 500 lines.
3. Note the exact preset key, vLLM pin, driver version, and GPU
   model.
4. Open an issue at
   <https://github.com/Sandermage/genesis-vllm-patches/issues> with
   the artefacts above.

Patches that surface from new cliffs follow the
[`CONTRIBUTING.md`](CONTRIBUTING.md) workflow.

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
| Garbage tokens / tool-call cascade (27B + TQ) | [Cliff 3](#cliff-3-turboquant--specverify-k1--full-cudagraph), [Cliff 4](#cliff-4-non-powerof2-gqa--p67) |
| TPS dropped after vLLM pin bump | [Cliff 8](#cliff-8-anchor-drift-on-vllm-pin-bumps) |
| Prefix-cache + MTP crash | [Recipe 6](#6-turboquant--specdecode--prefixcaching-crash) |
| Driver / CUDA / NCCL mismatch | `sndr doctor`, then [`INSTALL.md`](INSTALL.md) |
| V2 alias broken, V1 still works | [R-001](#r-001--v2-alias-resolution-broken) |
| `sndr memory explain` mis-predicts | [R-004](#r-004--sndr-memory-explain-mispredicting-oom) |
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
   [TQ vs fp8_e5m2 trade-off](#tq-k8v4-vs-fp8_e5m2-tradeoff)).
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

**Refs.** `vllm/sndr_core/core/text_patch.py`,
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

**Prevention.** Single-card 24 GB → use preset
`a5000-1x-27b-int4-tested` (or the 3090 equivalent when landed).
`sndr memory explain <preset>` gives a VRAM waterfall before launch.

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
# Or fall back to the current Genesis pin (v12.0.0 current registry):
docker pull vllm/vllm-openai:0.20.2rc1.dev371
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
docker run -v $REPO/vllm/sndr_core:/usr/local/lib/python3.12/dist-packages/vllm/sndr_core:rw ...
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
sndr launch --dry-run a5000-2x-35b-prod
sndr install --dry-run --non-interactive
python -m vllm.sndr_core.compat.schema_validator --quiet
python -m vllm.sndr_core.apply.shadow --strict
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

**Trigger.** `sndr launch prod-35b --preflight-only` returns
non-zero with `SchemaError` or `KeyError` from the V2 registry.
Typically after a YAML edit in `model_configs/builtin/` or a vLLM
pin bump that renamed an existing field.

**Revert.**

```bash
# Disable V2 resolution; launcher falls back to V1 preset keys
git revert --no-edit <SHA_OF_REGISTRY_V2_LANDING>
# OR
export GENESIS_DISABLE_V2_ALIAS=1
```

**Smoke.**

```bash
sndr launch a5000-2x-35b-prod --preflight-only
# Expect: rc=0, V1 docker emitter prints the same args operator used pre-V2.
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
GENESIS_ENABLE_PN<n>=1 sndr launch a5000-2x-35b-prod --preflight-only
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
sndr launch prod-35b --dry-run --runtime docker > /tmp/dry-docker.txt
sndr launch prod-35b --dry-run --runtime compose > /tmp/dry-compose.txt
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
sndr launch prod-35b --preflight-only
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

**Smoke.** `sndr launch a5000-2x-35b-prod --preflight-only` exits
rc=0 with the deprecation warning printed but launch succeeds.

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
sndr launch prod-35b --preflight-only
```

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

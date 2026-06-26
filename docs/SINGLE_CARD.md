# Single card (3090 / 4090 / A5000) — what fits, what doesn't, and the honest escape hatches

> **TL;DR.** Genesis is tuned and validated for **2× RTX A5000 (24 GB, SM 8.6),
> TP=2**. On a **single 24 GB card**, vLLM is fine for short / single-shot
> workloads but **not reliable above ~50K context** and **not safe under
> accumulating agentic traffic** — our **Cliff 2b is open** on consumer Ampere
> ([genesis #22](https://github.com/Sandermage/genesis-vllm-patches/issues/22)).
> If you have one card and an agentic coding client, use one of the **escape
> hatches** below (llama.cpp MTP, or ik_llama two-stage) — they are slower per
> token but cliff-immune, and the community has measured them honestly.

This page is **для людей** — for the consumer-GPU operator who landed here with
one 3090/4090, not a homelab with two A5000s. It is deliberately honest about
what our PROD presets are NOT, and routes you to paths that actually work today.

---

## First: ask the rig before you boot

Every Genesis prod preset now carries a machine-readable hardware envelope
(`card.hardware_fit` — see [CONFIGS.md](CONFIGS.md)). Project any preset against
your card **before** you waste a 2-minute cold start on a config that can't fit:

```bash
# Against your live rig (reads nvidia-smi):
sndr preflight prod-qwen3.6-35b-balanced

# Or model a single 3090 without one plugged in:
sndr preflight prod-qwen3.6-35b-balanced --fake-gpus "RTX 3090:24576:8.6"
```

On a single 24 GB card the 2× prod presets answer plainly:

```
preflight: prod-qwen3.6-35b-balanced
  rig:       fake (1 GPU(s), 24 GB/GPU, sm_8.6)
  ────────────────────────────────────────────────────────────────
  ✗ gpu_count        FAIL  need >= 2 GPU(s) · have 1 GPU(s)
  ✓ vram             PASS  need >= 21 GB/GPU · have 24 GB/GPU
  ✓ cuda_capability  PASS  need sm_8.6+ · have sm_8.6
  ────────────────────────────────────────────────────────────────
  VERDICT: CANNOT RUN
```

The 2× presets need both cards because TP=2 splits the failing GDN kernel's
working set across them — which is exactly what takes Cliff 2 / Cliff 2b off the
table (see below). On one card that split doesn't happen, so the prod presets
are correctly refused.

> `sndr preflight` is the Genesis analogue of club-3090's `scripts/preflight.sh`
> (`preflight_compose_hardware`), adapted to our preset model and extended with an
> engine-pin dimension. Credit: noonghunna's club-3090 (see bottom).

---

## Why single-card vLLM is unreliable above ~50K (the cliffs, briefly)

The full forensics live in [TROUBLESHOOTING.md → Named cliffs](TROUBLESHOOTING.md#named-cliffs).
The two that bite single-card users:

| | Cliff 2 (single-prompt) | **Cliff 2b (accumulating multi-turn)** ⭐ |
|---|---|---|
| Trigger | one prompt > ~50–60K tokens | ~21–26K **accumulated** context over 4–5 agentic turns |
| Where | `chunk_gated_delta_rule_fwd` → `chunk_o.py` `torch.empty_like(v)` | same kernel, fires earlier under multi-turn KV pressure |
| Why | FLA's `(B, NT, H, V, K)` h-tensor (~1.37 GiB at 60K) + KV pool > 24 GB | per-turn allocator reserve/fragmentation growth crosses the free budget |
| Status on consumer Ampere | mitigated by env recipe, not closed | **OPEN** — [genesis #22](https://github.com/Sandermage/genesis-vllm-patches/issues/22) |

**Cliff 2b is the one to respect.** If your workload is **hermes / OpenHands /
OpenCode / Cline / Roo / Aider / Cursor with retained context**, a single-card
vLLM session degrades to 0 throughput / 500s after a handful of turns —
regardless of which single-card vLLM config you pick. Mem-util tuning, MTP-off,
and `max-num-batched-tokens` adjustments **do not close it** (all tested).

> ⚠️ **PN59 does not close Cliff 2b on consumer Ampere (yet).** Genesis ships
> `GENESIS_ENABLE_PN59_STREAMING_GDN` (streaming-GDN window-iterative driver) and
> it survives continuous soak on the **2× A5000** rig, but on single-card 24 GB
> its eligibility gate rejects the `chunk_indices`/`chunk_offsets`-populated path
> that vLLM's mandatory `--max-num-batched-tokens` always sets — so it falls back
> to the vanilla path and OOMs at the same site. Tracked with a reproducer and
> fix proposals in [genesis #22](https://github.com/Sandermage/genesis-vllm-patches/issues/22)
> (cross-rig finding originally surfaced by club-3090). Until that lands, the
> escape hatches below are the recommendation, not a single-card vLLM tweak.

### Why TP=2 escapes it (and why you can't fake it on one card)

`chunk_gated_delta_rule_fwd` holds a ~500 MiB simultaneous live-tensor set
(`q/k/v/u/v_new/o/w/A/Ai/h`) at `T=4128`. With TP=2 the head dimension is
sharded, so **each card sees half** (~250 MiB) AND half the weights AND half the
KV. Per-card peak stays under 24 GB even with accumulated multi-turn context.
One card has nowhere to put the other half — there is no config knob that
recovers the headroom.

---

## The escape hatches (one card, blessed by club-3090, measured honestly)

These are **different engines** — different GDN kernels, different memory
allocators — so the vLLM cliff simply doesn't exist on them. You trade decode
speed for the model never falling over at depth. All numbers below are
**community-measured on a single RTX 3090 (SM 8.6, PCIe, ~230 W)**; treat them as
order-of-magnitude, not a Genesis-attested bench.

| Path | Engine | Max ctx (1× 24 GB) | Narr / Code TPS | Why it survives | When to pick |
|---|---|---|---|---|---|
| **llama.cpp + MTP** ⭐ | llama.cpp mainline (MTP merged [ggml-org/llama.cpp#22673](https://github.com/ggml-org/llama.cpp/pull/22673)) | ~200K (`-ub 512`) | ~51 / ~60 | hand-written CUDA DeltaNet + ggml flat allocator (no caching-allocator fragmentation) | bulletproof default, IDE agents, long multi-turn |
| **llama.cpp + MTP + vision** | llama.cpp + mmproj | ~49K (mmproj eats headroom) | ~57 / ~66 | same; drop `-ub 1024 → 512` to trade ~10% TPS for ~4× ctx | screenshot-debugging / vision review |
| **ik_llama + IQ4_KS + two-stage** ⭐ | ik_llama.cpp (advanced-quant fork) | ~200K | ~59 / ~98 (code +35% vs MTP-only) | two-stage **ngram + MTP `n_max=4`**: ngram drafts the repetitive code structure, MTP drafts the rest; leanest VRAM | code-heavy single-card, max speed |

**The two knobs that matter on these paths** (surfaced in club-3090's testing):

- **`-ub` (micro-batch / `UBATCH_SIZE`) does two jobs at once.** It caps the
  per-pass activation peak (cliff survival) AND eats into the KV-cache budget
  (context ceiling). `-ub 1024` is the speed-leaning default; **`-ub 512` frees
  ~1 GB to the KV pool** — worth ~4× more context for context-heavy workloads
  (especially with mmproj loaded). Smaller `-ub` = slower but longer + safer.
- **ik_llama two-stage `n_max=4`** (ngram+MTP) is the code-throughput winner: the
  second-stage MTP fills the gaps ngram can't, and `n_max=4` is the empirical
  sweet spot before draft-verify overhead eats the acceptance gain.

> ⚠️ **`CTX_SIZE` "boots ≠ fills".** A llama.cpp config can *boot* and pre-reserve
> a 262K KV pool yet only *fill* to ~125K before the flash-attention transient
> scratch at high fill OOMs. The max-safe single-card value is **~200K**
> (`CTX_SIZE=200000`), above which the config advertises more context than it can
> actually use. Source: club-3090's #200 ceiling-ladder probe.

> ⚠️ **Needle/NIAH certifies retrieval, not KV-quant *quality*.** A `verify-stress`
> 7/7 (including a 91K needle) does **not** mean an aggressive KV quant
> (`q4_0` / 3-bit / fp8) is tail-safe for JSON / closing-braces / tool calls —
> the tail is exactly where quantization breaks structure. For code/agent
> workloads prefer **≥ `q5_0` / `q4_1`** (asymmetric K/V — K is the sensitive
> cache) and treat aggressive KV quant as a context-push trade, not free quality.

---

## If you insist on single-card vLLM (short / single-shot only)

vLLM on one card is genuinely fine when context **does not accumulate**:
single-shot RAG, simple chat, batch processing, short code completion. For those:

| Workload | Recipe |
|---|---|
| Short ctx (≤ 8K), code completion | TQ k8v4 + `--gpu-memory-utilization 0.92` + `--max-num-seqs 4` |
| Long single-shot ctx (60–180K) | **`fp8_e5m2` KV** + util 0.85 + [PN35](TROUBLESHOOTING.md#pn35-status-vllm35975-backport) ON (NOT `turboquant_k8v4` — its K-activation peak OOMs earlier on tight VRAM, club-3090 #47) |

The full env-only **single-card 60K+ recipe** (P103 T-chunking + PN59 +
allocator hardening + the right `vllm serve` flags) lives in
[TROUBLESHOOTING.md → Single-card 24 GB long-context recipe](TROUBLESHOOTING.md#single-card-24-gb-long-context-60k--the-noonghunna-club-3090-22-recipe).
It buys headroom but does **not** make accumulating agentic traffic safe —
that's Cliff 2b, and the honest answer there is still an escape hatch or a
second card.

**Do not** reach for `turboquant_k8v4` as your single-card long-context default.
On < 24 GB-effective the safer choice is `fp8_e5m2` (1 byte/token, lower K
activation peak) — see the
[TQ k8v4 vs fp8_e5m2 trade-off](TROUBLESHOOTING.md#tq-k8v4-vs-fp8_e5m2-trade-off).

---

## 4090 / 5090 notes

- **RTX 4090 (24 GB, SM 8.9)** clears every SM gate our presets declare (8.9 >
  8.6) and `sndr preflight` will PASS it on capability + VRAM — but the
  single-card cliff story is identical to the 3090 (same 24 GB envelope, same
  GDN kernel). Headless 4090 has a slightly tighter usable ctx ceiling than a
  headless 3090 in practice. Use the escape hatches for agentic workloads.
- **RTX 5090 (32 GB, SM 12.0)** — the extra 8 GB genuinely unlocks single-card
  configs that don't fit on 24 GB. SM 12.0 clears the gate. This is the one
  consumer card where single-card vLLM long-context becomes reasonable; still
  validate your own ctx ceiling (`sndr preflight` checks fit, not the cliff).

---

## Two cards changes everything

If you can get a **second 24 GB card** (any topology — NVLink is optional, PCIe
works), the prod presets light up: `sndr preflight <preset>` flips to **CAN
RUN**, TP=2 splits the GDN kernel, and Cliff 2 / Cliff 2b are off the table.
That's the rig Genesis is actually tuned for. See [CONFIGS.md](CONFIGS.md) for
the dual-card prod presets and [MODELS.md](MODELS.md) for the served models.

---

## Credit

The single-card playbook, the escape-hatch measurements (llama.cpp MTP `-ub`
behaviour, ik_llama two-stage ngram+MTP `n_max=4`, the `CTX_SIZE` boots-≠-fills
and needle-≠-quality caveats), and the `preflight` convention this page builds
on all come from **[noonghunna's club-3090](https://github.com/noonghunna/club-3090)**
— a community project doing the honest, cross-rig consumer-GPU testing that
Genesis (a 2× A5000 patch tree) doesn't cover first-hand. Cliff 2b's cross-rig
reproducer and the PN59 single-card finding came from that collaboration
([genesis #22](https://github.com/Sandermage/genesis-vllm-patches/issues/22)).
See [CREDITS.md](CREDITS.md) for the full attribution ledger.

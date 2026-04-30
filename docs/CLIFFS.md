# Known Issues and Behavioral Cliffs

A "cliff" is a regime boundary where vLLM (or Genesis) goes from working well to working badly — sometimes silently. This document catalogs the cliffs we've found, what causes them, what we do about them, and what an operator should watch for.

If you hit something that isn't here, please open an issue with a reproducer. Cliffs that aren't documented hurt every operator after you.

For glossary terms (TQ, MTP, GDN, FA2, etc.) see [GLOSSARY.md](GLOSSARY.md). For the full patch catalog, see [PATCHES.md](../PATCHES.md).

---

## Cliff 1: FA2 softmax_lse over-allocation at long context

**Mechanism**

vLLM's GPU model runner sets `attn_metadata.max_seq_len = max_model_len` during cudagraph capture. FlashAttention-2 then allocates `softmax_lse[num_seqs, num_heads, max_seqlen_k]` sized by that ceiling, even when the actual batch only needs a fraction of it. At long contexts (>50K tokens, `max_model_len=256K`), this wastes 50-100 MiB per capture region.

**Impact**

You hit OOM earlier than you should on long-context workloads. On a 24 GB card running 27B INT4 at 256K context, the over-allocation alone can be the difference between booting and OOM.

**Fix**

**PN17 — FA2 lse runtime clamp.** Genesis-original, 2026-04-30, in response to noonghunna Issue #11. Patches FA2 to use the actual `seq_lens.max()` at runtime instead of `max_model_len` during capture.

**Refs**

- `vllm/_genesis/wiring/perf_hotfix/patch_n17_fa2_softmax_lse_clamp.py`
- noonghunna Issue #11 (cross-engine derivative)

---

## Cliff 2: GDN fwd_h tensor blow-up on single-prompt long context

**Mechanism**

`chunk_gated_delta_rule_fwd_h` allocates an intermediate `h` tensor sized `(B, NT, H, V, K)`, where `NT` is the number of chunks along the sequence dimension. At T=64K on Qwen3.6-27B (H=32, V=K=128), this is ~805 MiB just for `h` — for a single prompt.

**Impact**

Single-prompt long-context generation (>50K tokens) OOMs on 24 GB cards even when KV cache itself fits comfortably.

**Fix**

**P103 — chunked fwd_h + fwd_o orchestrator.** Splits the chunk dimension into sub-batches, materializes `h` for each sub-batch, runs `fwd_o`, and discards before moving on. Saves ~600 MiB of headroom at 64K, more at longer contexts.

**Refs**

- `vllm/_genesis/wiring/hybrid/patch_103_fla_cliff2_chunked.py`
- See also: P60, P60b for related GDN spec-decode corruption fixes

---

## Cliff 3: TurboQuant + spec-verify K+1 + FULL cudagraph → garbage tokens

**Mechanism**

`TurboQuantAttentionImpl._prefill_attention` treats spec-verify K+1 batches as first-chunk prefill — it sets `cu_seqlens_k = cu_seqlens_q`, ignoring already-cached KV from prior steps. When this code path is captured into a FULL cudagraph, the captured kernel launch ignores cached KV unconditionally, even at runtime.

**Impact**

Tool-call cascades on 27B + TQ k8v4 + FULL cudagraph: the model emits `<tool_call><tool_call>...` infinitely. Looks like a tool-call parser bug; root cause is attention.

**Fix**

**P67 — Genesis-original multi-query Triton kernel.** Replaces upstream's `_prefill_attention` for the K+1 verify case with a kernel that correctly attends to cached KV. The earlier P65 (switch to PIECEWISE cudagraph) is a workaround that costs ~5-8% TPS; P67 is the proper fix and gains TPS instead of losing it.

**Refs**

- `vllm/_genesis/kernels/p67_multi_query_kernel.py`
- `vllm/_genesis/wiring/spec_decode/patch_67_tq_multi_query_kernel.py`
- See [PATCHES.md](../PATCHES.md) P67 entry for sanitized variant (Inf/NaN→0 in K/V dequant)

---

## Cliff 4: Non-power-of-2 GQA + P67

**Mechanism**

Triton's `tl.arange` requires power-of-2 dimensions. P67's kernel uses `tl.arange(0, HEADS_PER_KV)` for the query-head dimension. On Qwen3.6-27B with GQA=24/4, `HEADS_PER_KV=6` — not a power of 2 — so the kernel fails to compile. Without compile success, P67 falls through to the upstream broken path, and you're back at Cliff 3 (garbage tokens under FULL cudagraph).

**Impact**

27B with TQ k8v4 + FULL cudagraph silently emits garbage. The fall-through is logged but easy to miss in a long boot log.

**Fix**

**P67 v7.63.x non-pow-2 generalization.** Uses `BLOCK_QH = next_power_of_2(HEADS_PER_KV)` and a `lane_valid = (lane_id < HEADS_PER_KV)` mask to write only valid lanes. Negligible perf cost (a couple of masked stores), full correctness on GQA=6.

**Refs**

- `vllm/_genesis/kernels/p67_multi_query_kernel.py` — see `BLOCK_QH` derivation
- Validated on 35B (GQA=8, pow-2) and 27B (GQA=6, non-pow-2)

---

## Cliff 5: ngram strict prompt_lookup_min=8 underperforms MTP on prose

**Mechanism**

The strict ngram heuristic (`prompt_lookup_min=8`) requires an 8-token sequence to appear in the prompt before it will speculate. On code-completion or tool-use-heavy workloads (where the prompt is structured and repetitive) this works well — acceptance rate stays high. On free-form prose, an 8-token literal match almost never appears, so ngram falls back to single-token decode. You lose the speculation entirely.

**Impact**

27 TPS on 27B creative-writing workload with strict ngram, vs. 87-100 TPS with MTP K=3.

**Fix**

Configuration, not a patch.

- **General workloads:** use MTP K=3 if available, or ngram with `prompt_lookup_min=2, prompt_lookup_max=5` (the loose default).
- **Tool-use-heavy workloads:** strict ngram (`min=8, max=8`) was originally introduced to lift tool-call clean rate from 56% to 100% on a single-query benchmark. Use it only when tool-call quality matters more than prose throughput.

**Refs**

- `--speculative-config` in launch scripts
- See [docs/CONFIGS.md](CONFIGS.md) Step 6 for tuning guidance

---

## Cliff 6: MoE backend regression on v0.20+ for non-FP8 (vLLM #41306)

**Mechanism**

vLLM v0.20 refactored MoE dispatch into `PluggableLayer` / `DefaultMoERunner`. The new abstraction adds a per-step CPU dispatch overhead. FP8 paths take a fast path that bypasses most of it; non-FP8 (BF16, AWQ) hit the full dispatch cost.

**Impact**

−19% TPS on Mixtral-class BF16 MoE on v0.20+ vs. v0.19. Reported upstream at vLLM Issue #41306.

**Mitigation**

- `--moe-backend=triton` flag, or
- `VLLM_MOE_BACKEND=triton` env var.

This forces the older Triton MoE path that doesn't go through the new dispatcher.

**Refs**

- vLLM Issue #41306 (upstream)
- Affects only non-FP8 MoE on v0.20+; FP8 MoE (Qwen3.6-35B-A3B) unaffected

---

## Cliff 7: DFlash + 24 GB single-card OOM at >80K context

**Mechanism**

DFlash speculative decoding requires a small drafter model (typically a 2B BF16) co-resident with the main model. On a 24 GB card running 35B-A3B-FP8 + DFlash 2B BF16:

- Main model FP8: ~17 GB
- DFlash drafter BF16: ~4 GB
- Activation + KV cache: rest

TurboQuant KV is not currently supported with DFlash on Ampere (the draft path doesn't go through the TQ KV reader). So KV stays in `auto`/fp8_e5m2 — capacity-limited. At >80K context the KV cache pushes you past 24 GB.

**Impact**

35B-A3B-FP8 + DFlash OOMs at 200K context, max ~80K on 2× A5000.

**Mitigation**

- Backport vLLM PR #40898 (sliding-window attention for DFlash) — limits drafter context.
- Backport vLLM PR #40849 (FP8 draft inheritance) — drafter shares main model's FP8 cache.
- Set `num_speculative_tokens=4` (more aggressive verification reduces effective KV pressure).
- For now, accept the 80K ceiling on 24 GB. 48 GB cards (A6000, R6000 Pro) don't hit this.

**Refs**

- vLLM PR #40898 (SWA for DFlash) — pending merge
- vLLM PR #40849 (FP8 draft inheritance) — pending merge
- See [PATCHES.md](../PATCHES.md) for DFlash backport status

---

## Cliff 8: Anchor drift on vLLM pin bumps

**Mechanism**

Genesis text-patches anchor on verbatim upstream code. When upstream renames a variable, refactors a function, or even changes whitespace, the anchor no longer matches. The TextPatcher logs `INFO: anchor not found, sub-patch skipped` and moves on. If `required=True`, the whole patch is marked `failed`. If `required=False`, the patch reports `applied` despite a sub-patch missing.

**Impact**

Operator pulls a new vLLM pin, restarts, sees `[GENESIS] APPLY` for all expected patches in the boot log, and assumes everything works. In reality, a sub-patch silently skipped and the bug it was guarding against is back. Hard to catch without an integration test.

**Mitigation**

- Verify live container content matches expected post-patch state. Grep the patched file for the marker string (`# [Genesis wiring marker: Genesis PNN ...]`).
- Watch the `partial_apply_warnings` counter — TextPatcher hardening planned to surface these prominently in the boot summary.
- Run anchor-presence tests before bumping a pin.
- Pin vLLM commits in your launch script; don't float on `main`.

**Refs**

- `vllm/_genesis/wiring/text_patch.py` — TextPatcher implementation
- See [docs/COMPATIBILITY.md](COMPATIBILITY.md) for the currently-tested pin

---

## Cross-references

- [PATCHES.md](../PATCHES.md) — full patch catalog with attribution and metadata
- [docs/CONFIGS.md](CONFIGS.md) — adding your own model recipe
- [docs/COMPATIBILITY.md](COMPATIBILITY.md) — supported vLLM pins, models, GPUs
- [docs/GLOSSARY.md](GLOSSARY.md) — term definitions
- [CONTRIBUTING.md](../CONTRIBUTING.md) — reporting new cliffs you discover

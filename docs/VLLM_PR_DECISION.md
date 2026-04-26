# Should we send a PR to vLLM upstream?

**Status**: Decision document. Author: Genesis project. Date: 2026-04-26.

This document analyzes whether to upstream any of our 37 Genesis patches to vllm-project/vllm, and if so — which ones, in what form, and how.

---

## TL;DR — recommendation matrix

| Patch | Upstream candidate? | Why / why not | Effort | Action |
|---|---|---|---|---|
| **P67/P67b** (TQ multi-query kernel) | **YES — high value** | Proper fix for #40880 (noonghunna's bug class). Replaces our P65 workaround AND vLLM's existing PIECEWISE downgrade. +32% TPS measured. Genesis-original Triton kernel. | HIGH (kernel + 2-3 weeks PR review cycle, will need vllm CI tests for spec-decode K+1) | **PRIORITY: yes, but plan for ~1 month of review** |
| **P71** (block-verify w/ gemini fixes) | **NO — comment on PR #40819 instead** | Already has open PR upstream. Our value is the 2 bug-fixes. Better to comment on existing PR with the fixes rather than open competing PR. | LOW | Post comment on #40819 with our 2 fixes (no Sander approval needed for technical comment? — actually still needed per memory rules) |
| **P72** (profile_run cap) | NO — fix is workaround, not root | The real fix is making MoE moe_align_block_size symbolic-shape-friendly. Workaround works for us but isn't general. | LOW (small text-patch) | Document in README, don't upstream |
| **P73** (prealloc_budget central resolver) | NO — Genesis-internal infra | Pure Genesis package code, not generally useful unless others have similar prealloc patterns | n/a | Skip |
| **P74** (chunk-clamp via long_prefill_token_threshold) | NO — already works through built-in vLLM env | Just sets an existing config knob automatically. Better as documentation/recipe. | n/a | Document in README |
| **P75** (suffix-decoding auto-enable) | NO — operator convenience only | The real Suffix Decoding code is already upstream (PR #25784). We just auto-swap method. Not a vLLM concern. | n/a | Skip |
| **P77** (adaptive ngram K controller) | **MAYBE — interesting but EAGLE-only versions exist upstream** | DynamicProposer (#26504), Elastic Speculation (#28693), Dynamic SD (#32374) all open and EAGLE-only. We have it for ngram. **Check first if these get merged + extend to ngram.** | HIGH (would need to align with one of the existing PRs) | Watch upstream first; consider PR after one of #26504/#28693 merges |
| **P78** (tolist capture-guard) | NO — already exists in upstream PR space | noonghunna's logic could be upstreamed by him; we shouldn't compete | n/a | Encourage noonghunna to upstream his patch in our discussion |
| Other 30 patches | Mostly NO | Most are workarounds for bugs being actively fixed upstream. Few are production-grade code (Genesis-original `_genesis/` package abstractions don't fit upstream style). | n/a | Track upstream merges; auto-no-op via drift markers |

## Specifics: P67 PR plan (the one real candidate)

### Why P67 is the strongest candidate

1. **Solves real upstream bug** — #40880 (noonghunna) is a documented multi-rig issue
2. **Proper fix vs workaround** — vLLM's only current workaround is PIECEWISE downgrade (~30% TPS cost)
3. **Empirically validated** — +32% TPS on 35B-A3B + 2× A5000, cross-rig confirmation pending
4. **Generally useful** — affects ANY TurboQuant + spec-decode user, not Genesis-specific
5. **Architectural fit** — Triton kernel slots into existing `vllm/v1/attention/ops/` directory pattern

### Why this PR will be hard

- **Kernel review takes time** — vLLM's spec-decode area is heavily scrutinized (Sun, Beirami, Leviathan, kotori-yan all watch it)
- **Need cross-arch validation** — we tested only Ampere SM 8.6; reviewer will ask about Hopper / Blackwell
- **Need integration tests** — spec-decode K+1 paths are CI-gated; we'd need to add tests under `tests/v1/spec_decode/`
- **Naming/abstraction may not match upstream conventions** — our `p67_multi_query_kernel.py` calls into our own dispatcher; needs refactor for direct integration
- **No Genesis attribution** — per `feedback_no_ai_credit_in_public.md`, only human author credit

### Plan if we proceed

**Phase 1 (1 week prep)**:
- Sander explicitly approves PR
- Strip Genesis-specific abstractions (env gates, dispatcher entries) — make it a drop-in replacement for vLLM's existing K+1 path
- Add `tests/v1/spec_decode/test_turboquant_multi_query.py` with K+1 verify cases at multiple shapes
- Document cross-arch behavior (we know Ampere; document gracefully degrade on others)
- Cite #40880 explicitly + noonghunna's reproducer

**Phase 2 (open PR)**:
- Title: "[Kernel] TurboQuant multi-query attention for spec-decode K+1 verify (fixes #40880)"
- Body: empirical perf table (75.6 vs 57.2 tok/s on our rig), cross-rig validation note, link to noonghunna's repos as confirmation
- Request reviewers: @WoosukKwon, @robertgshaw2-redhat (regular spec-decode reviewers), @tdoublep (P60 author who knows GDN+spec interaction)
- Sander-voice: Ukraine/AI-translation disclaimer at end of PR description

**Phase 3 (review)**:
- Expect 2-3 weeks for first round of reviews
- Likely changes: kernel naming, test coverage, cross-arch fallback
- Maintain Genesis P67 as private/internal until merged — operators can use either

### Estimated total effort: 4-6 weeks calendar time (1-2 weeks active work)

## What about block-verify (PR #40819)?

Better path: **comment on #40819 with our 2 bug-fixes** instead of opening competing PR.

Comment draft (not yet approved for posting):

> Thanks for the work on this PR. We backported it as Genesis P71 with two bug-fixes that gemini-code-assist's review flagged as critical (both unresolved at time of writing):
>
> **Fix 1 (CRITICAL)**: `uniform_prob = tl.load(uniform_probs_ptr + token_idx)` loads per-position. Per Sun 2024 §3.2, block verification requires ONE shared Bernoulli decision per BLOCK (one shared `u` per request, not per-position). Our patched version uses `uniform_probs_ptr + start_idx` once per request.
>
> **Fix 2 (HIGH)**: when `denom == 0` (perfect draft match: `prefix=1, residual=0`), the current code returns `h_block = 0.0` which REJECTS perfect drafts. Should be `1.0` (always accept on perfect match). Both PyTorch and Triton paths affected.
>
> With both fixes applied + opt-in env gate + silent fall-through to upstream per-token rule on any kernel error, P71 ships in our v7.42. Empirical measurement on Qwen3.6-35B-A3B-FP8 + MTP=3 + 2× A5000 matches PR's own Qwen3-32B parity result (~+0-3%); not a structural breakthrough on our config but the math is now correct.
>
> Reference Sun et al. paper proof that the rule is unbiased + ≥ per-token rule: arXiv 2403.10444 §4 Theorem 1.

This is technical, no advocacy. Per memory rules, requires Sander's "ok" before posting.

## Summary recommendation

**Open PR for P67 only**, after Sander explicit approval and 1 week of prep work. Skip everything else. Comment on #40819 with our 2 fixes (also requires approval). Track everything else via drift markers.

**Reasoning**: Low signal-to-noise ratio of pushing 30+ workarounds upstream. P67 is the one piece that's genuinely a kernel contribution.

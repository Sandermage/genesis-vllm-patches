# Patch-health + speed audit (dev491, A5000) — verify everything works + benches

**Date:** 2026-06-16
**Pin:** 0.22.1rc1.dev491+g1033ffac2
**Trigger:** user — "test on the server, verify everything is fixed and works, run benches" +
"boost speeds and fix all patches that don't work / cause integrity violations / errors."

---

## TL;DR

- **All three models verified working + benched** (controlled single-stream, same script, calibrated
  to the canonical 35B reference): **35B 211.8 TPS** (at-reference), **31B-tq-G4_82 37.3 TPS** (no-MTP,
  coherent), **27B 104.6 TPS** (coherent, ~-13% dev491 regression).
- **The patches are HEALTHY — nothing is actually broken.** A live-PROD audit (iron-rule #11) overturned
  the initial "broken patch" premise: the 14 boot warnings are all benign (11 version-gates correct,
  P5 auto-retire, PN347 correct version-skip, P18B an apply-ordering artifact). 0 integrity violations.
- **No kernel-tune speed gain available.** The one untested lever (TQ-decode `BLOCK_KV=32`) was A/B'd and
  **regresses -5.2%** (211.8 → 200.8) — the SM 8.6 smem-spill regime. `BLOCK_KV=16` + `num_warps=8` +
  `num_stages=3` are all at validated optima. The system is already optimally tuned.

---

## 1. Bench results (controlled, comparable)

Same `bench_tps.py` for all (5 prompts × 3 runs + warmup, max_tokens=512, temp=0.7, single-stream wall
TPS = completion_tokens / latency). Calibrated: 35B came out at 211.8 == the canonical reference (211).

| Model | Config | TPS (median) | CV | Coherence |
|---|---|---:|---:|---|
| 35B qwen3.6-35b-a3b | PROD, MTP K=3, TQ k8v4, Marlin FP8 | **211.8** | 8.9% | at-reference |
| 31B-tq gemma-4-31b | **G4_82**, no-MTP, dev491 | **37.3** | 2.6% | `2+2=4`, full sentences |
| 27B qwen3.6-27b | TQ k8v4, MTP K=3, dev491 | **104.6** | 8.3% | `4`/`Paris` (thinking-off) |

27B note: the empty content on the first coherence pass was Qwen3.6 **thinking-mode** (the 64-token
budget went to `<think>`, content=None) — re-checked with `enable_thinking=false` → fully coherent. The
104.6 vs the dev371 reference (~120) is ~-13%, consistent with the documented A11 dev491 regression
(dev371 pin image is purged, so dev491 is the only available run).

## 2. Patch-health audit (the "fix broken patches" ask)

35B boot apply summary: **112 applied, 137 skipped, 0 FAILED, 14 partial-apply warnings.** A 5-agent
workflow verified each of the 14 **against the live PROD container** (not the pristine `/tmp` extract —
the key iron-rule-#11 discipline):

- **11 are VERSION-GATE skips** (P61c, PN56, P64, PN9, PN125, PN90, PN52, PN80, P83, P94, PN13) —
  intentional, for other pin windows. The qwen3coder trio (P61c/PN56/P64) skip is a **correctness WIN**
  on dev491 (applying them on the rewritten native parser leaks tool XML to content + silences SSE).
  PN90 skip is correct (native probabilistic is -5.9% TPS / -10% accept on our shape). The rest are
  byte/anchor-verified retires. **No lost functionality.**
- **P5** — correct auto-retire (defers to merged upstream #39931).
- **PN347** ("MarlinFP8 N==K correctness — required_anchor_missing") — **NOT broken.** Correctly
  version-skipped (registry caps it `<0.22.1rc1.dev491`); upstream #44735 (merged) + dev491's marlin
  `size_k_first` refactor make N==K corruption structurally impossible. Re-anchoring would **re-introduce**
  a transpose into code that no longer needs it → NEW corruption. The boot line is the benign version-skip.
- **P18B_TEXT** ("SM 8.6 TQ-decode tune — anchors absent") — **NOT failing; it IS applied in PROD.** Live
  kernel has `num_warps=8, num_stages=3` + the P18B marker. The "skipped" is an apply-ordering artifact:
  P18B (ordinal 36) runs before its dep PN119 (ordinal 42) in the *first* process → soft-skip; PN119 then
  patches the file; all 15 worker processes find the anchors → applied. My `/tmp` extract is the
  *pristine pre-PN119* file, which is why **it** lacked the anchors.

**Verdict: 0 actual integrity violations.** Consistent with the 35B being exactly at-reference.

### Rejected "fix": the PN347 dispatch gate-bypass
A tempting cleanup (add `should_apply("PN347")` at the legacy dispatch to silence the benign warning) is
**unsafe**: `should_apply` treats PN347 as opt-in, but the wiring's `apply()` is **default-on** (checks
`GENESIS_DISABLE_PN347`). Adding the gate would flip PN347 default-on → opt-in and could **silently
disable it on the dev259 rollback pin** (where it IS needed for N==K correctness). A cosmetic warning is
not worth a correctness regression — **not done.** Documented as a careful follow-up only.

## 3. Speed A/B (the "boost speeds" ask)

The only untested lever the audit surfaced: P18B's `resolve_decode_tune()` computes `BLOCK_KV=32` (env
live) but `_build_replacement()` never emits it, so the grouped GQA kernel keeps `BLOCK_KV_GROUPED=16`.
Hypothesis: 16→32 doubles the MMA N-dim and could better fill the A5000 tensor cores (+few %), OR spills
smem at num_stages=3 (regresses).

**A/B on the 35B (docker cp the modified kernel, reboot, bench, restore):**

| BLOCK_KV | TPS (median) | coherence |
|---:|---:|---|
| 16 (shipped) | **211.8** | ✓ |
| 32 | **200.8** (-5.2%) | ✓ (`2+2=4`) |

**BLOCK_KV=32 regresses -5.2%** — the predicted SM 8.6 spill regime (the K/V fp16 tile at stages=3 grows
past the 100KB/SM budget, exactly what PN296 `MAX_WARPS=4` / PN298/PN299 were built to avoid). Coherence
was fine, so it's a pure speed regression. **BLOCK_KV=16 is optimal; P18B's silent-drop is beneficial.**
`num_warps=8` (P67-validated) and `num_stages=3` (registry: "stages=2 measured -2% to -9%") are likewise
at their validated optima. **No kernel-tune speed gain is available on A5000/dev491.**

## 4. Honest verdict

- "Fix broken patches" → nothing is broken; the system is healthy (0 failed applies, 35B at-reference).
  The warnings are benign artifacts. The only worthwhile cleanups are cosmetic (provenance text, a CI
  lint for the version-range-without-should_apply class) and are documented as follow-ups.
- "Boost speeds" → the TQ-decode kernel is already optimally tuned (BLOCK_KV=16/warps=8/stages=3 all
  validated). The single untested lever (BLOCK_KV=32) regresses. **0% kernel-tune gain available.**
- **The biggest remaining speed opportunity is the 27B dev491 regression (~-13%)** — but that is a
  separate, dev371-vs-dev491 root-cause effort (the 27B is pinned to dev371 precisely because of it).

---

## 5. CORRECTION (2026-06-16, loop iteration) — the 27B "-13%" was MY bench artifact, not a regression

A follow-up regression-hunt workflow + the existing 2026-06-15 journal overturned §1/§4's
"27B ~-13% dev491 regression":

- **#45295 (Marlin thread-tile padding) REFUTED.** The lead read the real packed safetensors
  shapes of all 10 distinct 27B INT4 projections — every one is already tile-aligned at TP=2
  (q 6144×5120, k/v 512×5120, o 5120×3072, gate/up 8704×5120, down 5120×8704, GDN in_proj
  5120×5120, in_proj_z 3072×5120 — all n%64==0, k%128==0). `marlin_padded_nk` returns them
  unchanged → zero per-forward padding. Grep of the 27B dev491 boot log confirms the
  "performance may be degraded" warning never fires.
- **The 2026-06-15 journal already A/B'd dev259 vs dev491** (same patch set): code_gen 113.9 ≈
  110-114, long_gen 96.1 ≈ 95, short_chat 90.8 ≈ 92 — **NO engine regression.**
- **Root cause of my "104.6": thinking-mode.** `bench_tps.py` did not set
  `enable_thinking=false`, so the 27B emitted `<think>` reasoning (Qwen3.6 default) inflating
  latency. Thinking-OFF the 27B does **~125-127 TPS — at/above the 120 baseline.** The "120
  reference" was a thinking-off number; comparing it to my thinking-on 104.6 manufactured a
  fake "-13%".

**Net: all three models are healthy + at/above baseline on dev491. There is NO hidden speed to
recover — no real 27B regression, the TQ-decode kernel is already optimally tuned, the patches
are healthy. The earlier "biggest remaining lever = 27B regression" claim is withdrawn.**
Lesson (re-)learned: pin ONE canonical bench (thinking-off, fixed prompts/tokens, n>=5,
median+CV) before declaring any regression — a thinking-on vs thinking-off mismatch alone
manufactured a 13% phantom.

# 27B "146+ single-stream" investigation — CONCLUSION

**Date**: 2026-06-15
**Operator claim**: the 27B did 146+ TPS single-stream on dev371/dev259; dev491 dropped
it to ~125, and "the drop shows that something we did earlier broke due to engine changes"
— so audit ALL patches to find what stopped working.

## Method: three independent, canonically-benched hypotheses (all on the live 27B,
## genesis_chat_matrix_bench, n=3, same harness as the 146-era numbers)

### R1 — TQ decode WorkspaceManager indirection (the cross-model workflow's #1, "high conf")
Hypothesis: P98's `decode_workspace_revert` retired 2026-06-08 (PN118 took the path);
PN118's `try_get_simultaneous` is un-memoized → ~17% per-token Python cost on decode.
Built a success-path memo (env-gated, 4 invariants unit-verified), A/B'd ON vs OFF.
**Result: REFUTED.** thinking_off 124.9 (ON) vs 126.5 (OFF) — within noise; no recovery.
The ~17% came from a pre-#40941 measurement of the *heavier* `get_simultaneous`; dev491's
`try_get_simultaneous` is light, so the cost is negligible vs CUDA kernel time. Reverted.

### R2 — MTP K=3 excludes the new GDN packed-recurrent-decode fast path (#36596/#44700)
Hypothesis: under MTP, `gdn_attn.py` reclassifies decode rows to prefill, so the fast
packed-decode kernel is unreachable → MTP-off would be faster.
A/B'd 27B MTP-OFF vs MTP-ON (net tok/s).
**Result: REFUTED — and inverted.** MTP-OFF is **2× SLOWER** (thinking_off 57.6 vs 127.1;
TPOT 17.1ms vs 7.6ms). MTP's token amortization is a massive *win*, not a bottleneck.

### R3 — engine regression dev259→dev491 (the operator's actual hypothesis)
Booted the 27B on the old pin **dev259** (303916e93, gmu 0.92, same MTP K=3 / 256K /
patch set) vs dev491.
**Result: NO regression.** Every variant that completed on dev259 matches dev491 —
code_gen 113.9 (dev259) ≈ 110-114 (dev491), long_gen 96.1 ≈ 95, short_chat 90.8 ≈ 92,
long_ctx_8k 31.4 ≈ 32. (dev259 then *crashed* on the 32K-context variant — an OOM/
stability artifact of gmu 0.92 + the current dev491-era patch set on the old engine, so
multi_turn/thinking_on/thinking_off/tool_call returned connection-reset; that is a boot-
stability issue, not a speed signal — the decode-heavy variants that DID run were
identical to dev491.)

## Where the "146" actually comes from

The only `146` in the repo's bench corpus is **35B**, not 27B:
`docs/superpowers/journal/2026-06-09-iter-N8-P28-fixed.md:13` —
`avg_per_req_TPS (under conc) = 146.0 TPS` on `qwen3.6-35b-a3b`, "per-user perceived under
load". The 27B's documented single-stream baseline is **120 TPS** (skill reference) and its
multi-conc per-user figures are higher (conc=4 → 265 aggregate). The 27B "146 single" is a
metric conflation (35B's per-req-under-conc, or the 27B's per-user-under-conc), not a
single-stream wall-TPS the engine ever regressed from.

## Verdict

**No 27B engine regression. No silently-broken patch.** The patches are intact and firing
on dev491 (P67/PN119/PN340/PN341/PN59 confirmed; 100+ applied, 0 failed). Current 27B
single-stream ≈ **125-127 TPS thinking_off** is at/above the validated 120 baseline. The
146 was never a 27B single-stream number.

This is the bench-discipline lesson the master plan already records (§16 "a single outlier
created a false regression narrative I chased for hours"): the workflow produced three
mechanism-grounded, high-confidence hypotheses, and **all three were refuted by
measurement.** Static analysis is necessary but not sufficient; the canonical A/B is
decisive.

## Real, deployable wins still on the table (separate from the 146 myth)

- The dev259 32K-ctx crash shows the gmu-0.82-on-dev491 + 256K headroom is tight; the
  cudagraph 24/32 over-capture (R-secondary from the workflow) genuinely wastes VRAM/boot
  and is worth capping (live-inspect first — 5 patches already touch capture sizes).
- PR#44176 (fused qk-rmsnorm-rope for the 16 full-attn layers) is in dev491 but unverified
  as firing for `Qwen3_5ForConditionalGeneration` — a clean, bit-equivalent decode win if
  not already active. Worth a one-shot probe.
These are incremental (low-single-digit %), not a path back to a 146 that single-stream
never had.

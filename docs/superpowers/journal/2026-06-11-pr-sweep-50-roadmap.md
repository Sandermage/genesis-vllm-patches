# 2026-06-11 — Upstream PR sweep (50 PRs) — adaptation roadmap

50 upstream vLLM PRs deep-studied by the user's mandate (code, diffs,
linked issues; every PR gets a concrete use/adapt plan — no
rejections). Per-PR verification: state, presence in pin
0.22.1rc1.dev259+g303916e93 (pristine-tree grep), existing Genesis
vendor (registry grep). 50 study agents + 5 theme syntheses; the final
roll-up below. Implementation wave 1 is scheduled (cron c7865148).

Wave distribution: W1=12 PRs (implement now), W2=23 (next cycle),
W3=15 (next pin bump / research). Effort: 38 S, 12 M; risk all LOW.

Wave 1 queue (priority order from the syntheses):
1. #45100 -> PN370: racy accepted-counts under async+MTP — live
   silent-corruption class on 35B PROD config + ~2-5% TPOT (deletes a
   per-step synchronize); PN341 sub-d anchor ordering declared.
2. #45199 -> PN371: encoder-cache eviction engine-fatal (Gemma-4
   vision + MTP + async exact triple), flag-gated ON for gemma4.
3. #45005: eagle_step zero-seqlen Triton guard (kills #40756-class
   CUDA IMA on 262-280K MTP sessions); candidate to retire P108 sync
   workaround after A/B (2-6% TPOT recovery).
4. #45040 + #45038 pair: fp8_e5m2 KV for weight-only checkpoints +
   sub-SM90 auto-override guard (G4_31 extension, same site) —
   Gemma-31B KV halves -> full 256K ctx path without TQ.
5. #44955: parallel_tool_calls null!=false 1-liner — stops silent
   multi-tool truncation for LiteLLM/n8n clients.
6. #44877: G4_T1 quoted-key strip transplant (live dict-arg pollution
   on both Gemma PRODs).
7. #44741: multi-boundary streaming deltas under MTP (silent first
   tool-call argument loss; must strip G4_14 pad set first).
8. #44644 = PN348 (already vendored): ENABLE + measure (~1 GiB/rank
   peak VRAM + 1-3s boot).
9. #45076 = PN367 v2 (done this sweep: drift markers fixed + 1 MiB
   capture floor).
10. #44717 (done in-sweep: G4_T1 key-strip vendored, 10 tests) and
    #44752 = duplicate -> watchlist only.
11. #44784 doc half: fix stale GDN-offload caveat in PATCHES.md
    (vendor half is W2; unlocks prefix-reuse 4-8x second-pass TTFT).
12. W2-IMMEDIATE surfaced by #45146 study: P79d is STALE (backports a
    dead boolean; if enabled would assert-crash) — rewrite with
    discard-credit semantics.

Full theme syntheses (5 chunks) follow verbatim.
# Chunk 3/5 Synthesis — 10 upstream PRs for Genesis

## Theme A: MTP/Spec-decode stability (Qwen3.6 PROD hot path)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45005** eagle_step zero-seqlen guard | **1** | Vendor as new spec_decode patch: 8-line Triton guard (`seq_len <= 0`, stricter than upstream's `== 0`) in eagle_step_slot_mapping kernel; rebind in utils + llm_base_proposer | Kills #40756-class CUDA IMA on long MTP sessions (our exact 262-280K agent profile); if A/B then retires P108's draft-loop sync → 2-6% TPOT recovery |
| **#45060** OOV recovered token on all-NaN logits | **2** | Vendor kernel half only (tl.where -inf mask on vocab padding tile); do NOT take scheduler assert — extend PN133 with log.error instead | Closes persistent-NaN livelock hole PN133 leaves open (Qwen vocab 151936 % 8192 ≠ 0 → bug is live); ~0 latency cost |
| **#44993** grammar advance across reasoning boundary | **2** | Do NOT vendor code — port 50-trial E2E json_object reproducer + 4 regression tests against P62; add drift markers for #44297/#44993 | First-ever proof json_object/regex/choice works on PROD (upstream baseline 96% fail); protects P62 from wedged apply at pin bump |

**Duplicates:** #45060 scheduler hunk = same site as **PN133** (vendored #42722, enabled all 4 prod composes) — PN133 is the safer half, keep it. #44993 functionally **covered by P62** (vendored #36138; P62's prefix scan is stronger for multi-token markers) — validation-only adaptation. #45005 siblings: **P58** (#40768 root-cause) + **P108** (#42603 sync workaround, retirement candidate).
**Synergy:** #45005 + #45060 together harden the full MTP K=3 failure chain (IMA crash + NaN stall); land in same bench cycle.

## Theme B: Gemma-4-31B KV-cache dtype gates (vendor as a pair)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45040** allow fp8_e5m2 KV for weight-only checkpoints | **1** | Monkeypatch `_checkpoint_has_fp8_kv_scales` predicate into `_init_kv_cache_quant`; then new gemma4-31b-fp8e5m2-fallback profile (Triton-attn/FlashInfer, NOT FA2) | 31B KV halves (~9.4→4.7 GiB @200K) → restores full 256K ctx or frees CG/concurrency headroom; low-single-digit % TPOT at 32K+ |
| **#45038** guard fp8 KV auto-override on sub-SM90 | **1** | Extend G4_31's Attention.__init__ wrap with second suppress arm (cache_dtype=="auto" + kv_cache_scheme present); add late-mutation invariant log | Removes IMA-crash-on-burst landmine on live 31B kv-auto degraded profile (SM8.6, MTP K=3, max_num_seqs=8); 0 ms delta |

**Duplicates:** #45038 wraps the **same site as G4_31** — extend its predicate, don't double-wrap; audit G4_31 for redundancy on this pin (its turboquant_* arm can no longer trigger).
**Synergy:** both touch attention.py `_init_kv_cache_quant`/`__init__` on the same model; #45038 protects the kv-auto interim state, #45040 provides the escape to fp8_e5m2 — sequence: guard first, then fallback profile. Check drafter KV-spec patches (G4_71/72/76) before enabling on structured role.

## Theme C: Serving correctness (agent tool-call path)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#44955** parallel_tool_calls null ≠ false | **1** | 1-line text patch (`is not False`) in tool_calls_utils.py, modeled on PN288; add streaming-delta unit test upstream lacks; optional null-strip in Genesis_proxy_ai | Stops silent truncation of multi-tool-calls for explicit-null clients (LiteLLM/n8n); each recovered call saves a full agent round-trip (hundreds ms–s) |

**Duplicates:** none; adjacent to **PN288** (serving.py:821-828) — established territory.

## Theme D: Bench/validation infrastructure (GDN hybrid long-ctx)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45080** DecodeBenchConnector list/tuple KV fix | **2** | Backport tensor-vs-list split + EXTEND: per-block fill (pin's MambaSpec IS block-indexed — upstream's whole-pool fill clobbers concurrent state) + real group_idx map; wire 8K/32K/128K/280K TPOT sweep profile (MTP off) | Decode-TPOT-vs-depth benching on GDN hybrids: impossible today (crashes) → minutes instead of hours; unlocks A/B at depth for GDN/TQ patches |
| **#45022** Voxtral realtime re-anchor (RFC) | **2** | Don't vendor feature (all 3 gates fail on our stack); port vram_probe.py → endurance_probe.py in pin-validation playbook; document in-place-KV-mutation handshake as design template; R(-D) fold = research track | Catches multi-hour VRAM/RSS/KV creep before pin validation (day-later OOM class on 24GB rig); de-risks future live-KV-rewrite patches |

**Synergy:** #45080's connector + #45022's vram_probe + #45001's GB/s harness form one bench-suite upgrade; #45080 is the validation tool for any #45001-derived TQ kernel rework.

## Theme E: Technique adaptations (no direct path on SM8.6)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45096** mamba PD async-sched NIXL race | **3** | Zero code now: add stream-ordering invariant to pin-bump checklist (off-stream GDN state-pool writes MUST be event-gated); reuse event+FIFO-deque+query() idiom when building GDN state prefix-cache restore | 0 ms today (NIXL dormant); correctness insurance for planned GDN state-caching (4.4s@32K TTFT attack) — avoids multi-day load-dependent accuracy-jitter debug |
| **#45001** CUDA qkv_padded_fp8_quant for ViT | **3** | Don't vendor .cu (dead on SM8.6 + text-only + PN62 skips ViT); adapt bandwidth playbook to TQ k8v4 Triton kernels (BLOCK_M rows/program, grid-stride, vectorized loads) gated on <50%-of-768GB/s microbench; copy bit-exact test matrix | Kernel 1.2-1.5x plausible → ~1-3% TTFT@32K, small TPOT, larger multiconc TPS; test/bench methodology is certain zero-risk value |

**Synergy:** #45096's audit confirmed P7/PN204 side-streams already event-gated (no present hazard) — that audit is the template check for the checklist item. #45096 + #45022 both feed the same future GDN-state-caching design (stream-ordering + barrier-placement patterns).

## Cross-cutting flags

- **PN id collision:** #45060, #45005, and #44955 each claim "next free id PN370" in their plans — resequence at vendoring time (PN370/371/372).
- **Pin discrepancy in source data:** #45096 cites pin `0.20.2rc1.dev338+gbf0d2dc6d` (CLAUDE.md canonical), the other 9 verify against `0.22.1rc1.dev259+g303916e93` in /private/tmp/candidate_pin_current — confirm which pin these anchor-verifications bind to before vendoring.
- **All 10 PRs OPEN upstream** (as of 2026-06-11); none in pin; none already vendored by PR number. All need drift markers / `gh pr view` re-check at next pin bump.
- **Effort/risk:** 9× S/LOW, 1× M/LOW (#45001; #44993 M is tests-only). Wave 1 = 4 PRs (45005, 45040, 45038, 44955), Wave 2 = 4 (45080, 45060, 45022, 44993), Wave 3 = 2 (45096, 45001).
- **Theme-level gain shape:** Waves 1-2 are almost entirely stability/correctness insurance (crash classes, silent truncation, livelocks) + bench capability; the only direct perf plays are #45040 (31B ctx/KV) and the conditional #45005→P108-retire (2-6% TPOT) and #45001 TQ rework (1-3% TTFT).


---

# Chunk 5/5 Synthesis — 10 Upstream PR Studies (all S-effort, LOW-risk)

## Theme A: Gemma-4 Tool-Call Parser Correctness (3 PRs, all Wave 1)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#44717** dict-key sentinel strip | 1 | **DONE this session** — 8-line key-strip vendored into both G4_T1 overlay copies (v2 PR42237 + v1 rollback), 10 new tests, rides G4_T1 registry entry | Kills deterministic `<\|"\|>`-polluted dict keys on every dict-typed tool arg, both Gemma-4 PROD models; zero perf cost |
| **#44752** dict-key STRING_DELIM strip | 1 | **DUPLICATE of #44717** (same bug, same issue #44715) — collapse plan to: add #44752 to upstream_watchlist.yaml + retire-trigger note; do NOT vendor as new PN370 | Same gain already captured by #44717; avoids redundant patch |
| **#44741** multi-boundary streaming deltas under MTP | 1 | New monkey-patch module adding `_extract_streaming_delta_segments`; MUST strip G4_14 pad-token set from `current_text` before consistency check or fix silently degrades | Fixes silent first-tool-call argument loss in multi-tool streaming turns under MTP K=3 — live bug on both Gemma-4 models |

**Duplicate flag:** #44752 ≡ #44717 — identical fix surface; #44717 already landed, #44752 needs tracking only.
**Synergy:** #44741's vendor touches the same overlay files just edited for #44717 — apply on top of key-strip; add combined regression test (multi-boundary + `<pad>` + G4_14 active). Port MTP-sized-chunk test pattern to qwen3xml suite as free insurance.

## Theme B: Spec-Decode / MTP Boot & Sampling (3 PRs)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45076** (→#44745) negative cudagraph mem estimate | 1 | **DONE this session** — PN367 v2: fixed never-firing drift markers, added 1 MiB first-capture floor, registry re-pointed to consolidated #44745 | Correct KV-cache sizing under MTP K=3 at util=0.9 on 24GB cards; prevents capture/load OOM class; zero steady-state cost |
| **#44644** Qwen3.5 MTP backbone dedup | 1 | **Already vendored as PN348** (default OFF) — remaining: enable on next 35B restart, A/B measure peak VRAM + boot time, correct registry gain claim (pin proposer already reclaims steady-state) | ~1 GiB/rank lower PEAK load VRAM + 1-3s faster boot; steady-state TPS unchanged (reframed from original claim) |
| **#44742** allowed_token_ids metadata hardening | 2 | Vendor as new PN (anchored 1-line condition in `_make_sampling_metadata`, PN67 playbook); port CUDA regression test against OUR patched sampler | Defense-in-depth only (#35654 consumer fix already in pin); protects PN369/P71 rewritten rejection sampler from row-parity landmine |

**Duplicate flags:** PN367 (#45076) and PN348 (#44644) already exist — no new vendoring.
**PN-id collision:** plans for #44752 and #44742 both claim "PN370" — assign sequentially at vendor time; #44752 no longer needs one.
**Synergy:** #44742's regression test exercises allowed_token_ids + draft tokens through pn369/p71 — coverage upstream never had. PN348 + PN367 both shrink boot-time VRAM uncertainty on the same 24GB MTP boot path.

## Theme C: KV-Cache Lifecycle / Sleep-Wake (1 PR)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#44778** FP8 KV wake-up nested containers | 2 | **Functionally DUPLICATE of existing PN55v2** (drift marker matches helper) — no re-vendor; do: add 44778 to PN55 related_upstream_prs, replace hand-mirrored test with PR's exec-patched-text technique, add log.warning for skipped non-tensor leaves, review companion #44779 before enabling | Unblocks crash-free sleep/wake model hot-swap on shared 24GB rig (~18-19 GiB freed/GPU at L1, same Ampere class); closes real CI gap in PN55 test fidelity |

**Synergy:** with PN348 enabled, sleep/wake hot-swap between 35B/27B/Gemma becomes the fast model-switch mechanism (seconds vs minutes); #44779 (resume_scheduler gate) is the prerequisite review item.

## Theme D: Quantization / MoE Config Robustness (3 PRs)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#44628** FP8 `modules_to_not_convert` substring match | 2 | Vendor (text-patch fp8.py both call sites + quant_utils experts branch + parity one-liners); validate via per-layer quant-scheme log diff on 35B before default_on | Eliminates silent-gibberish class for HF-style ignore lists: Qwen3.6-VL FP8 broken TODAY on pin; unblocks selective-FP8 GDN re-quant experiments |
| **#44563** moe_wna16 BLOCK_SIZE_K clamp gs=32 | 2 | Vendor 4-line clamp (anchor unique, beside P24 surface) + PR's CPU-only test + boot-time legality assert for actual model grid | Converts deterministic warmup abort into serving for gs=32 int4 MoE on the live Marlin→wna16 fallback (awq_marlin:307, auto_gptq:242); unblocks gs=32 quant benchmarking of our exact model family |
| **#44754** revert of #44613 snapshot | 3 | **DO NOT vendor** (misattributed auto-revert, abandoned, conflicts; #44613 correctly in pin) — do: watch-list note, adopt snapshot-discipline lint, harden bare `get_current_vllm_config()` at turboquant_attn.py:496 | Stability only: removes boot/reload AssertionError class from PROD TQ path (same class dev134 already triggered once); de-mines future pin bump |

**Synergy:** #44754's snapshot-at-construction discipline is the lint rule that the #44628 vendor and any future FusedMoEConfig-adjacent Genesis patch (P24/PN352/PN96B) should follow; PN96B workspace sizing can reuse `moe_config.max_capture_size` already in pin. #44563's #40547 heuristic-swap finding feeds the P24 MoE-tiling research track.

## Rollup

- **Wave 1 (5):** 44717 done, 45076 done (PN367 v2), 44644 enable-and-measure (PN348), 44741 vendor, 44752 watchlist-only
- **Wave 2 (4):** 44778 (PN55 hygiene+tests), 44742, 44628, 44563 — all new small vendors except 44778
- **Wave 3 (1):** 44754 — anti-vendor: watch-list + TQ hardening only
- **Duplicates with existing Genesis patches:** 44778→PN55v2, 44644→PN348, 45076→PN367, 44752→44717 (this session's G4_T1 edit)
- **Aggregate gain profile:** zero hot-path latency/TPS changes anywhere; all 10 are correctness/stability/operability — tool-call corruption fixes (3), boot/VRAM robustness (3), sampler/quant landmine removal (3), sleep-wake hot-swap enablement (1)


---

# Chunk 4/5 Synthesis — 10 Upstream PR Studies (all OPEN, none in pin, none vendored)

## Theme 1: Gemma-4 Tool-Call Parser (PROD-live, G4_T1 overlay family)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **44877** quoted-key STRING_DELIM strip | **1** | Transplant ~25-line quoted-key branch into all 3 G4_T1 overlay copies (live bind-mount = our file carries the bug); audit qwen3_xml for same key/value asymmetry | Fixes dict-typed tool args on both Gemma-4 PRODs today (keys polluted with `<\|"\|>` → tool schema errors); zero perf |
| **44844** streaming parser rewrite (span-based) | **2** | Vendor as G4_T1 **v3** overlay (verbatim PR head + reset-guard hardening), A/B vs v2 on 7×5 tool-call harness; cross-pollinate stripped-marker recovery to qwen3_xml | Closes MTP-K=3 streaming corruption v2 can't handle (stripped `call:name{...}` bodies, coalesced calls); v2 baseline 31/35 keep-alive |

**DUPLICATE flag**: both edit the file already overlaid by G4_T1 v2 (vendor of rival open PR #42237; #42006 also open) — 3 competing upstream fixes, whichever merges first decides retirement.
**SYNERGY**: if v3 (44844) adopted, fold 44877's quoted-key hunk into it — one overlay, one watchlist row each.

## Theme 2: MTP / Spec-Decode Load Robustness

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **44943** Qwen3.5/3.6 MTP pre-fused expert loader | **2** | Vendor +21/-3 as insurance PN (prereq for INT4 35B-A3B trial) + Genesis-original draft-weight load-coverage guard (P29-style) + accept-rate floor (~0.55) in bench suite | 0 on today's PROD (verified unaffected); prevents silent -23pp accept (≈ -15-20% decode TPS) + TypeError on future pre-fused MoE ckpts |
| **44837** DSV4 MTP missing `prefix=` | **3** | Adopt lesson, not code: AST lint test flagging quantized Linears without `prefix=` across all engaged model files; dormant 6-line stub | 0 perf; kills one pin-bump crash class + silent AWQ/AutoRound mis-quantization variant; <1s CI |
| **44880** Bailing MTP + LINEAR_ATTN spec-decode | **2** | Don't vendor feature (pin's gdn_attn already has the design); port its buffer-stability tests (data_ptr identity + PAD_SLOT_ID) against PN340-patched GDN builder; flag as PN340/PN341 anchor-breaker in watchlist; cherry-pick llm_base_proposer getattr-hardening at next bump | 0 direct perf; regression protection on PROD full-CG MTP K=3 path (silent state corruption class); converts guaranteed PN341 anchor break into planned 5-min re-anchor |

**DUPLICATE flags**: 44943 touches qwen3_5_mtp.py (already patched by PN348); 44880 edits PN341's anchor neighborhood in gpu_model_runner.
**SYNERGY**: 44943's coverage guard + 44837's prefix lint = one "loud startup" validation family — both convert silent partial-load/mis-quant into explicit failures; build together.

## Theme 3: CUDA-Graph & Triton Kernel Silent-Corruption Guards

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **44868** FULL CG forward-context refresh | **2** | Vendor as **PN358** (journal pre-assigned) with 2 improvements: data_ptr-pruned copy (kills upstream's 1-3% TPOT cost) + `detect` mode logging stale-metadata mismatches; run detect through smoke+bench on 35B/27B | 0 TPS; prevents silent wrong-output on FULL CG replay; detect mode = definitive audit of whether our 280-patch overlay leaks fresh tensors into captured graphs |
| **44850** tile_mask in USE_TD KV load | **3** | Vendor 4-line tl.where fix (constexpr-dead when USE_TD=False → provably free); adopt NaN-canary harness, run against tq_grouped_decode / sparse_v / p67 kernels | 0 on PROD (USE_TD off); makes VLLM_TRITON_ATTN_USE_TD A/B safe; canary proves/breaks masking invariant in our own TQ decode kernels |

**DUPLICATE flags**: 44868 composes with PN353B/PN118 (no file conflict; only retired PN13 touched cuda_graph.py); 44850 shares file with PN351 — anchors verified non-colliding.
**SYNERGY**: 44868-detect + 44880's ported buffer tests attack the same stale-FULL-CG-metadata class from runtime and unit-test sides respectively — run as one validation campaign on the PN340/341/353 surface.

## Theme 4: Quantization Scale Invariants

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **44912** 0-D FP8 scale rank vs `torch._scaled_mm` | **3** | Don't vendor upstream (kernel dormant on SM8.6); fix the SAME bug in OUR PN77 lm_head (`amax()` → 0-D scales, add `.view(1)`, TDD unit test) + grep-audit CI for all `_scaled_mm` call sites | 0 today (Marlin tier selected); eliminates guaranteed future startup InductorError on sm89+ HW; inoculates pn347/p81/compressor paths |

**DUPLICATE flag**: bug class lives in our `lm_head_fp8_method.py:416-434` — this is a fix to Genesis code, not a vendor.

## Theme 5: KV Memory — Offload Unlock & Prefill Workspace

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **44784** KV-offload + MTP segfault | **2** (doc fix **1**) | Wave 1: fix stale docs/PATCHES.md:146 GDN-offload caveat. Wave 2: vendor 4 scheduler.py hunks (tests-first) as gate for CPU KV-offload experiment; EXTEND: Qwen3.6-specific is_eagle_group flagging (avoid all-groups fallback hit-rate loss) + re-add dropped pre-DMA bounds check | Dormant until offload enabled; unlocks prefix-reuse → second-pass TTFT ~4-8x cut on agent loops (4.4s@32K → plausibly 0.2-0.6s restore) |
| **44932** FP8 KV + DCP FlashInfer | **3** (part 1 ~S anytime) | Part 1: cherry-pick selector.py `info_once` hunk (log dedup). Part 2: build TQ-k8v4 fused gather+dequant Triton kernel (PR's CPU token_to_seq pattern) + narrow PN22/23 dequant buffer dtype (fp32→fp16 V). Part 3 research: quantize-before-collective for PCIe TP=2 | Part 2: 5-20ms off non-GDN TTFT share + few hundred MiB returned to KV blocks at 280K ctx (fewer preemptions); Part 3 speculative ~50% all-reduce bytes |

**DUPLICATE flag**: 44932's gather+dequant solves the same problem as our PN22/23 dequant_buffer — partial overlap, borrow pattern, keep TQ specifics.
**SYNERGY**: 44784 + 44932-part-2 are one "long-ctx KV memory" track — offload reuse and workspace shrink both relieve the 280K-ctx KV-block pressure; bench together on the 35B profile.

## Wave Summary & Bookkeeping

- **Wave 1 (now, S/LOW)**: 44877 parser fix; 44784 doc correction.
- **Wave 2 (next cycle)**: 44943, 44880, 44868 (PN358), 44844 (v3 A/B), 44784 vendor.
- **Wave 3 (next pin bump / opportunistic)**: 44932, 44912, 44850, 44837.
- **All 10 need `tools/upstream_watchlist.yaml` retire-on-merge rows** (currently zero entries for any). 6 of 10 deliver **zero perf on today's PROD** — value is insurance/correctness on silent-failure classes (the P91/Bug-Class-13 pattern); only 44877/44844 fix live PROD defects, only 44784/44932 carry latency upside.


---

# Chunk 2/5 Synthesis — 10 Upstream PR Studies for Genesis

**Wave distribution:** W1: 1 PR (vendor now) | W2: 4 PRs + 1 surfaced hotfix | W3: 5 PRs (dormant insurance / watch)

---

## Theme A — Async-scheduling × Spec-decode correctness (hottest: exact PROD config)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45100** racy accepted counts, async spec decode | **1** | Vendor NOW as PN370 (2 sub-patches: skip racy CPU accepted-counts read under async+non-align; size GDN FULL-graph metadata by `num_reqs` not `num_actual_tokens`); anchors verified byte-clean on pin | Eliminates live silent-corruption class on Qwen3.6-35B PROD (hybrid GDN + MTP K=3 + async = exact repro config); bonus ~2-5% TPOT (deletes per-step synchronize + gather) |
| **#45146** reset placeholders on KV-load-failure rewind | 3 | Vendor as P79e (copy P79d TextPatcher skeleton) — dormant until KV offloading adopted; EXTEND with discard-credit for in-flight async frames (gap worth posting upstream) | Zero today (no KV connector); pre-mitigates permanent request-hang (1 stranded slot = 50% capacity at max_num_seqs=2) |
| **#45144** MTP + fp8 KV + AITER SKV (ROCm) | 2 | Don't vendor ROCm code; use as blueprint for Variant B: multi-query direct routing in pr42637 TQ overlay to unblock MTP K=3 × TQ on Gemma-4-31B; add build()/build_for_drafting() coverage to patch checklist | +20-40% decode TPS on Gemma-4-31B dense TQ profile if Variant B lands; acceptance data (K=2: 0.75, K=3: 0.50) → free K=2-vs-K=3 bench experiment |

**Duplicates/conflicts:**
- #45100 ⚠ **PN341 sub-patch d anchors the IDENTICAL line** (`if self.num_accepted_tokens_event is not None:`) — declare ordering, soft-skip acceptable (PN341 path already device-authoritative). PN290 composes cleanly (producer vs consumer side). PN111 untouched.
- #45146 study **surfaced P79d staleness — promote to W2 IMMEDIATE**: P79d backports dead boolean `discard_latest_async_tokens` (0 hits in pin; upstream migrated to integer `async_tokens_to_discard`); if enabled it would trip `assert >= 0` at async_scheduler.py:60. Rewrite to grant discard credit before zeroing.
- #45144 = **second independent upstream validation of P67/P67b design** (PROD-active, +32% TPS) — add `watch_for_drift_via vllm#45144` to credit strings; converge P67b toward its shape on next bump.

**Synergy:** #45100's min-len/short-output distribution scoring → adopt as standard spec-decode corruption detector in bench methodology; reuse it to validate the P79d rewrite and Variant B.

---

## Theme B — Gemma-4 tool-call streaming (direct PROD hit)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45068** Gemma4 parallel tool calls + MTP | 2 | Do NOT replace G4_T1 v2; borrow (1) MTP-split regression test corpus (we have zero streaming parser tests), (2) token-id start-gate into v2 overlay, (3) xfail-then-fix the keep-alive state leak; track all 5 racing PRs in retire trigger | Closes remaining 4/35 (11.4%) keep-alive streaming failures on Gemma-4 + MTP K=3; regression shield for next pin bump |

**Duplicate:** same root issue #41967 as **G4_T1** (#42006 overlay) and live **G4_T1 v2** (#42237 overlay, 35/35 with Connection:close). Update retire trigger to enumerate #42006/#42237/#42300/#44741/#45068.

---

## Theme C — Triton kernel tuning & fusion

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45126** NVIDIA-tuned tiles + PID swizzle for triton_scaled_mm | 2 | Probe whether W8A8 fallback ever fires (likely dormant); main value = transfer recipe (offline sweep → frozen per-arch dict, bit-identical assert) to g4_kpad_moe_gemm + re-tune P81 on sm_86; static tables via PN362 for GDN determinism | 3-8% Gemma-4 MoE prefill TTFT (realistic e2e); kills 199-vs-228 TPS autotune jitter class |
| **#45151** FP8 group-quant fused into attn epilogue | 3 | Don't vendor (no matchable FP8-activation graphs on Ampere); (1) make PN351 anchor dual-form NOW — **guaranteed anchor-breaker**, (2) adopt 3D-decode guardrail (forcing 2D path = up-to-7x decode regression at our exact operating point), (3) reuse architecture as GDN-epilogue-fusion template | Direct: 0. Avoids silent loss of PN351's -3-7% decode TPOT on pin bump; speculative 3-8% GDN prefill if epilogue fusion pursued |
| **#45120** fused softmax in grouped_topk CUDA | 3 | WATCH only (compiled .cu, can't overlay; pin rejects scoring_func=2); port A/B method into routing microbench at batch 1-8; use fp32-fused finding as retire evidence for kernels_legacy/router_softmax.py | 0 direct (both our live routing paths verified already single-kernel fused); ~12us/MoE-layer free via pin if DeepSeek/GLM-family ever lands; one legacy patch retired |

**Duplicates/conflicts:**
- #45151 **inserts 7 kwargs precisely inside PN351's `PN351_LAUNCH_OLD` anchor** — fix before any pin containing it.
- #45126 = same optimization class as **P81** (#40925); P81's table was tuned on GB10, not sm_86 — re-sweep.
- #45120 enables **retirement of router_softmax.py** (legacy overlay, dormant on FusedTopKRouter path).

**Synergy:** #45126's sweep harness + #45151's 3D-decode-path guardrail → one shared "unified-attention/Triton-GEMM patch playbook" checklist; #45126 static tables consumed via PN362 force-first mechanism.

---

## Theme D — KV offload architecture (future-facing)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45053** OffloadingHandler → OffloadingWorker refactor | 3 | Borrow explicit-direction API into PN95 stream pool (`submit_demote`/`submit_promote` wrappers replacing string `direction=` kwarg, default "d2h" even for promotes — same anti-pattern upstream kills); registry note for renamed symbols on post-#45053 pins; research: reuse upstream SingleDirection handlers + Triton swap as PN203 transfer engine | 0 direct (we don't run the subsystem); makes PN95 d2h/h2d PCIe metrics trustworthy (prereq for PN203 tuning); plausible low-single-digit ms p99 trim on PN203 paging if Triton swap adopted |

**Synergy:** #45053 + #45146 are the two prerequisites for any future CPU-KV-offloading experiment (clean transfer API + hang-proof failure recovery); track #33689's hybrid-model item vs PN95 MambaSpec exclusion.

---

## Theme E — Guards, validation & pin-bump hygiene

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **#45109** AWQ outputs under Transformers v5 tokenizer | 2 | Test-only PR, vendor nothing; build tokenizer_class audit tool + **tokenizer-fingerprint gate** (sha256 of token-ids on canonical prompt set, diffed pre-bench every pin bump); use pin's `_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS` hook if mismatches found | Kills a whole false-FAIL class in bump gates (v5 semantics already live in our pin; AWQ/AutoRound checkpoints exactly the affected class); <1 min check vs hours of misdirected patch bisection |
| **#45130** fail-fast FP8 MoE + LoRA guard | 3 | Vendor on next bump (38-line additive, byte-verified clean; lands on our actual MarlinExperts class); transfer fail-fast pattern to TQ-k8v4+FA3 and Gemma-4 TQ-blocked drift guards; 5-min phantom-env audit (`VLLM_MOE_FORCE_MARLIN` is fake) | 0 perf; saves hours-long debug on future LoRA-on-35B-FP8 experiments; converts two silent drift hazards into loud startup errors |

**Synergy:** #45109's fingerprint gate must run BEFORE output benches that validate #45068 parser changes and #45151/PN351 anchor migrations — otherwise tokenizer drift gets misattributed to patches (iron-rule-#11 class). #45130's guard pattern is the enforcement mechanism for the TQ/FA2 convention that #45144's Variant B work depends on.

---

## Action queue (priority order)
1. **W1:** Vendor #45100 as PN370 (PROD corruption, anchors verified clean, PN341 conflict declared)
2. **W2-immediate:** Rewrite stale P79d (assert-crash latent, surfaced by #45146 study)
3. **W2:** #45068 test corpus + token-id gate; #45144 Variant B (Gemma-31B MTP×TQ); #45126 sweep transfer; #45109 fingerprint gate
4. **W3:** #45146→P79e, #45151 PN351 dual anchor (do before any pin bump), #45053 PN95 API, #45130 guard, #45120 watch+microbench
5. **Registry updates:** retire-triggers (5 racing PRs on G4_T1), drift-watch (#45144→P67b), anchor-breaker note (#45151→PN351), retire candidate (router_softmax.py)


---

# Upstream PR Synthesis — Chunk 1/5 (10 PRs)

All 10 PRs: OPEN upstream, NOT in pin `0.22.1rc1.dev259+g303916e93`, none vendored yet. All risk LOW.

## Theme A: KV-cache page-size unification (hybrid GDN + TQ choke point)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **45207** Mamba page padding | 2 (S) | Fold MambaSpec `page_size_padded` elif into G4_60E (not standalone — would double-monkey-patch); add Genesis-unique 3-way test (TQ+Mamba+bf16-drafter) | Stability: kills deterministic boot-crash on first bf16-drafter / asymmetric-KV experiment on 35B/27B hybrids; actionable errors aid Gemma-4-31B TQ diagnosis |
| **45181** Mixed KV page sizes (DFlash) | 2 (M) | Vendor generic AttentionSpec padding fallback + `_reshape_attention_kv_cache` stride hardening; reconcile with G4_60E (generic supersedes its Patch 2; keep TQ-native Patches 1/3/4) | Unblocks Gemma-4-31B TQ k8v4 profile on this pin (~2.7x KV memory vs fp16); eliminates NotImplementedError boot-fail class across kv-auto matrix |

**DUPLICATE FLAG:** Both PRs patch the SAME function (`unify_kv_cache_spec_page_size`) that Genesis **G4_60E** (PR #42637 cherry-pick) already owns, with the same `page_size_padded` technique. G4_60E covers only TQ specs — inherits the #43626 MambaSpec crash itself.
**SYNERGY:** 45207 (MambaSpec branch) + 45181 (AttentionSpec branch) together cover all spec types — implement as ONE reconciliation pass over G4_60E + pr42637 overlay; register both upstream_pr refs for retirement tracking. 45181's reshape hardening also pairs with G4_70B FAIL_FAST=1 for the Gemma-4-31B TQ retry.

## Theme B: Engine-fatal crash fix (multimodal + spec-decode)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **45199** Deferred ref-pinned encoder cache eviction | **1** (S) | Vendor as PN370 (multimodal family, flag-gated, ON for gemma4); wire 5 legacy-runner points + replace EncoderCache class; self-skip probe on `eager_eviction` signature; EXTEND: demote fatal assert to warn+skip in drafter path only | Stability: eliminates whole-engine-fatal "Encoder cache miss" on Gemma-4 vision + MTP K=3 + async-scheduling (our exact #38551 triple); zero PROD impact on text-only Qwen |

No Genesis duplicate. Only wave-1 item in this chunk — highest urgency.

## Theme C: Observability & metrics correctness

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **45202** Prefix-cache stats double-count on retries | 2 (S) | Vendor as p86: suppress recording in `get_computed_blocks`, record on `allocate_slots` success (avoids patching 2000-line Scheduler; p79d-style safer variant); fallback-disable if connector configured | Metrics-only: de-biases proxy `prefix_hit_rate` (can be tens of % inflated under our long-ctx burst retries); protects p85/TQ-KV A/B conclusions from false positives |
| **45182** TRTLLM BF16 MoE modular kernel | 2 (S) | Kernel is SM100-only (dead on Ampere) — borrow ONLY the `logger.info_once("Using %s experts")` one-liner into fp8/int_wna16 oracles + selection test asserting MarlinExperts | 0 perf; one-grep proof MarlinExperts is live → guards PN96b/PN368 (silent no-ops on selection drift) across pin bumps |

**SYNERGY:** 45202 requires re-baselining p85/TQ hit-rate numbers collected under max_num_seqs=8. 45182's log feeds existing drift-detection workflow.

## Theme D: Startup memory & weight loading

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **45197** Disable cudagraph memory estimate default | 2 (M) | Do NOT flip configs blindly (upstream CHANGES_REQUESTED); MEASURE =1 vs =0 on 35B/27B first; if >~200MiB overestimate confirmed, write root-cause patch: capture-all-and-measure for descs ≤16; reconcile presets.py(=0) vs a5000 YAML(=1) inconsistency | Capacity: est. 100-500 MiB/A5000 recoverable → ~5-30K extra TQ k8v4 KV tokens on 280K-ctx 35B, while keeping post-warmup OOM protection |
| **45196** Loader/LoadConfig fail-fast validation | 2 (S) | Vendor near-clean 4-hunk patch + tests; then use as safety prerequisite for `enable_multithread_load: true, num_threads: 8` experiment on 35B/27B; mirror checks into audit_config_keys.py | 0 hot-path; multithread-load experiment ~30-60s saved per 35B restart × dozens/session; converts 3 silent-misconfig classes (PN96-style drift) into loud ValueErrors |

**SYNERGY:** 45197 interacts with P66 (capture-size filter bounds the overestimate — our short descs lists make both the bug small and the exact-measure fix cheap).

## Theme E: Research-track enablers (wave 3, no immediate PROD effect)

| PR | Wave | Adaptation essence | Expected gain |
|---|---|---|---|
| **45184** Intermediate tensors for KV connectors | 3 (M) | Step 1: vendor 2-file hook, FIX upstream kwargs-shadowing bug; Step 2 (real win): connector-free "salience tap" at same decorator site → SnapKV-style scores as PN95 demote-ordering key | Direct: 0 (no connector in PROD; connectors crash on hybrid GDN anyway). Salience-aware PN95 demote: est. 5-15% fewer preemptions at high KV pressure; ~2x attention-KV headroom ceiling; NOT a TTFT fix |
| **45176** Marlin → torch stable ABI | 3 (S) | (1) Pin-bump guard: verify `_C_stable_libtorch`/`_moe_C` op resolution before patch probes + rerun Marlin A/B (all sm80 kernels recompiled); (2) adopt pattern for `genesis_kernels.abi3.so` — native p87, fused TQ dequant, native pn365 GDN GEMM | Direct: 0 (byte-identical kernels). Unlocks compiled Genesis kernels persisting across pins (overlay is currently Python/Triton-only); guards 100%-of-GEMMs Marlin hot path at bump |
| **45173** /v1/embeddings chat messages | 3 (S) | Track A now: copy ~20-line duck-typing into proxy to replace zero-vector stub + proxy-side instruction rendering; Track B: vendor when embedding sidecar lands (likely merged by next bump) | 0 generation impact; functional unblock of semantic memory (zero-vectors → MTEB ~64-70 via Qwen3-Embedding); 10-30x batch-embed throughput vs CPU ChromaDB |

**SYNERGY:** 45184 salience scores can later steer TQ per-block precision; 45176's stable-ABI sideload is the delivery vehicle for any native kernel 45184-derived work needs.

## Duplicate summary
- **G4_60E (#42637)** ← overlapped by **45207** (same function+technique, MambaSpec gap) and **45181** (generic superset of its Patch 2). Single reconciliation effort recommended.
- All other 8 PRs: zero existing Genesis patch overlap (grep-verified per study).

## Wave roll-up
- **Wave 1 (do first):** 45199 (engine-fatal crash, Gemma-4 vision+MTP)
- **Wave 2:** 45207+45181 (combined G4_60E pass), 45202, 45196, 45182 (log only), 45197 (measure-first)
- **Wave 3:** 45184, 45176, 45173
- Effort: 7×S, 3×M. All LOW risk. None deliver direct TPOT/TPS gains — chunk is stability (3 crash classes), metrics correctness (2), capacity/iteration speed (2), and research enablers (3).


---


# 2026-06-11 — Upstream PR sweep BATCH 2 (62 PRs 45530...44986) — adaptation roadmap

60 newer upstream vLLM PRs deep-studied (65-agent workflow wt2lj8sh3;
the user's second batch, all anchored at/after the 50-PR batch-1).
Per-PR: state, presence in pin 0.22.1rc1.dev259+g303916e93 (pristine
grep), existing Genesis vendor (registry grep). Full 5-chunk theme
syntheses are appended verbatim below.

Wave distribution: W1=5 (vendor now), W2=20 (next cycle), W3=35
(next pin bump / research). Effort: 47 S, 11 M, 2 L; risk 53 LOW, 7 MED.

## WAVE 1 — vendor now (LIVE PROD correctness/stability)

These five fix bugs that are LIVE on our exact PROD shape, all S-effort:

1. **#45477 → mamba-block-aligned prefill split** (HIGHEST): keeps
   intermediate prefill chunks mamba-block-aligned with spec decode.
   Kills a LIVE prefix-cache-POISONING bug (garbled output / malformed
   tool calls) on Qwen3.6 27B/35B GDN+Mamba + MTP K=3 + APC under
   concurrent multiconc. **We are CURRENTLY EXPOSED** — PN346 (#43650,
   hit-side) is vendored but the PR author states they are
   "complementary"; PN346 alone does NOT cover non-final-block
   poisoning under unequal concurrent prefixes. Compose with PN346+P85,
   default-ON. Genesis extra: re-A/B boundary timing with async overlap
   ON (PR validated --no-async-scheduling).
2. **#45346 → reject degenerate structured_outputs** (DoS): json_object
   with empty/degenerate grammar → EngineDeadError, confirmed
   instance-wide DoS on single-instance PROD. Source-overlay 2 guards
   after the pin's empty-grammar guard + gateway-edge guard.
3. **#45290 → forced-named empty-params → JSON object**: a no-arg tool
   (end_turn/noop/handoff) currently yields a bare string/number not
   `{}` → agent-loop parse-500s on 3/4 PROD families (qwen3_xml 35B/27B,
   gemma4 26/31B). Disjoint from PN70/P68.
4. **#45389 → tool-call streaming brace string-awareness**: required-
   streaming `_bracket_level`/`filter_delta_text` must be JSON-string-
   aware (don't break on `,`/braces inside string values). Kills
   corrupted streamed function.arguments for payloads with `{}"\` in
   string args (file paths, regex, shell). **Prerequisite for safely
   enabling P68** (long-ctx auto-force-required funnels traffic into
   this exact helper). Pair with #45310 (Hermes boundary, same class).
5. **#44986 → Eagle prefix-cache prefill fix**: thread
   skip_eagle_pop=is_prefill_phase through find_longest_cache_hit so the
   lookahead block isn't dropped during prefill. Recovers 1 prefix-cache
   block/prefill on every MTP request (entire hit on short prompts) —
   direct TTFT win on cache-warm agent/multi-turn + long-ctx GDN
   prefill, zero decode cost. Supersedes retired P83/P84.

PN-slot collision: #44986/#45290/#45389/#45265/#45299 each drafted as
"PN384" — assign sequential slots at vendor time (highest live = PN383).

## CRITICAL pin-bump landmine (must fix BEFORE next bump)

**#45171 (MERGED 2026-06-11)** deleted 74 lines from
chat_completion/serving.py and moved harmony streaming to
parser/harmony.py. **PN288 + P107 anchor on exactly those
use_harmony/harmony_tools_streamed[i] lines** → both break on any pin
bump past 2026-06-11. Re-anchor both before/at the bump (flagged by the
#45464 study). Also: #45415 (libtorch_stable _C migration) renames
csrc/activation_kernels.cu — migration radar for any Genesis C++ patch.

## WAVE 2 highlights (20 — next cycle, vendor + extension/gate)

- #45390 DoS grammar-compilation timeouts (7 GHSA) — set
  VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS=2 (PR's 10s violates our SLO).
- #45369 → PN384-class: streaming-LSE rejection sampler (no full-vocab
  fp8 softmax materialize) — LIVE on 35B MTP-K3 + PN90/P71, ~0.5-2% TPOT
  + 3.6-14.6MB transient reclaim. M/MED, A/B BLOCK_SIZE.
- #45343 DFlash spec-decode KV-layout fix — generic upstream supersedes
  our g4_72/g4_74/g4_76 band-aids (retire-after-A/B). L/MED.
- #45339 per-iteration max_num_seqs under async — likely INVALIDATES the
  "async neutral" verdict → re-A/B (confounder behind P86 re-baseline).
- #45306 modelopt_mixed on Ampere SM80/86 (+ mandatory dropped
  orig_dtype fix) — unblocks NVFP4 siblings of all PROD models; pair
  with #45320 (enable+harden).
- #45453 /health/decode forward-progress watchdog — our EXACT failure
  mode (NCCL P2P deadlock surviving restart, TP=2, /health stays 200,
  requests hang). Wire into tools/safe_container_recreate.py.
- #45471 completion_tokens_details.reasoning_tokens — all 4 models run
  qwen3 reasoning parser; extend with MTP accepted/rejected counters.
- #45299 Qwen3 no-</think> short-answer → content (MED — default
  behavior change, gate on truncation probe).
- #45517 pre-NCCL mem snapshot (~1-2 GiB/rank reclaim on 27B GMU 0.92).
- #45361 INT8 per-token-head KV rounding (+4.5pp int8 quality).
- #45265 FP8 MoE+LoRA→Marlin reroute (tested on our exact 35B-A3B-FP8).
- #45310 Hermes boundary string-awareness; #45351 ctx-length log
  hardening; #45417 generation_config max_new_tokens audit; #45383
  prompt_embeds mm crash; #45379 35B MoE FP8 tune seed (re-sweep on
  A5000); #45349 Mamba prefill cap (harvest invariant for PN95/PN203);
  #45290-pair #45389 tool-call; #45479 Gemma4 name-recovery.

## WAVE 3 (35 — pin-bump / research / audit artifacts)

Mostly Ampere-unreachable (CUTLASS SM90+/Blackwell/MLA/XPU/RISC-V) or
dormant (no KV connector / no sleep-mode / no LoRA). Harvested as:
review heuristics (int64-before-multiply, sentinel-guard symmetry,
CustomOp-not-at-forward, index-not-id delta merge), audit tools
(audit_arch_gates, mypy gate over patches, golden W4A8 CPU ref),
pin-bump watchlist rows, and future-feature prerequisites (sleep-mode
drain #45363, P/D role-aware #45280/#45283, FP8-KV evidence #45434:
gemma-4-31B fp8 KV accuracy-neutral-to-+1.9pt → trial kv_cache_dtype=fp8
on the reachable FA2/Triton path for ~2x attn-KV memory).

## Reusable review heuristics harvested (apply to OUR kernels)
1. every indexed global write in a borrowed MoE permute/scatter kernel
   must share the same sentinel/bounds guard;
2. token_idx*num_heads*stride → int64 BEFORE the multiply (TQ pack/
   unpack, GDN prefill, MTP draft gather);
3. cudagraph-captured WorkspaceManager consumers must not be reachable
   by an eager batch > max_cudagraph_capture_size before lock_workspace;
4. never construct a CustomOp at forward time;
5. streaming tool-call deltas: exactly one DeltaToolCall per tool/chunk,
   merge by INDEX not id (audit our qwen3xml — keys on id).

---

## Full 5-chunk theme syntheses (verbatim)

I'll synthesize the 12 PR studies into a compact, themed markdown. Let me note this is batch-2 chunk 1/5, with 12 PRs covering scheduler/KV-cache correctness, kernel bounds, memory budgeting, tool/reasoning parsing, and observability.

# Genesis Batch-2 Chunk 1/5 — Mini-Synthesis (12 upstream PRs)

## Theme A — Hybrid Mamba/GDN + Spec-Decode Correctness (our hottest area)

- **PR 45477** [W1, S/LOW] — Keep intermediate prefill chunks mamba-block-aligned with spec decode (`scheduler._mamba_block_aligned_split`). **Adapt:** vendor verbatim (round_down on chunk-END, not length); anchor on `else: pass` fall-through at scheduler.py:293-338. **Gain:** kills a LIVE prefix-cache-poisoning bug (garbled output/malformed tool calls, 5-10/10→clean) on our exact PROD shape (Qwen3.6 27B/35B GDN+Mamba + MTP K=3 + APC, concurrent multiconc). Zero hot-path cost.
  - **DUP/complement:** PN346 (#43650, hit-side) already vendored — PR author says "complementary"; PN346 alone does NOT cover non-final-block poisoning under unequal concurrent prefixes → **we are currently exposed.** Compose with PN346 + P85, default-ON.
  - **Genesis extra to verify:** PR validated `--no-async-scheduling`; our PROD runs async overlap ON → A/B re-confirm boundary timing vs GDN state-write on 30-GDN-layer 35B.

- **PR 45497** [W2, M/LOW] — Avoid hybrid KV-load-failure crash (`_update_requests_with_invalid_blocks` single-target unpack → ValueError → EngineDeadError). **Adapt:** vendor scheduler patch anchored on `# TODO (davidb): add support for hybrid memory allocator`; add all 3 `num_output_placeholders=0` resets. **Gain:** prevents guaranteed full-engine-death on hybrid 35B/Gemma-MoE the moment any KV connector is active. **Reachability TODAY = LOW** (Genesis configures NO upstream KV connector — kv_transfer_config is None), so it's a dormant landmine pre-positioned for future P/D disagg / pn95 disk-tier.
  - **SYNERGY:** the missing `num_output_placeholders=0` resets ALSO corrupt single-group `num_new_tokens`/async early-exit math under MTP K=3 — cross-check with **PN348** (async discard credit) & **P58** (async placeholder); order PN348 after this patch.
  - **Genesis extra:** PR flattens all groups → brutal full-recompute; build the deferred **precise per-group recovery** (invalidate attention group only, recompute Mamba from truncation) to avoid re-paying 4.4s@32K GDN prefill.

## Theme B — Memory Budgeting / Startup Stability

- **PR 45517** [W2, M/LOW] — Pre-NCCL memory snapshot (`VLLM_INIT_SNAPSHOT_BEFORE_NCCL`, default-off) + actionable OOM ValueError + per-rank `vllm:startup_free_bytes` gauge. **Adapt:** vendor env-gated branch verbatim (cannot regress, default-off); backport the actionable error + before_profile assert-rebase UNCONDITIONALLY; plumb env into launchers starting with 27B-int4 @ GMU 0.92 (least slack). **Gain:** reclaims ~1-2 GiB/rank NCCL workspace from the GMU budget (we force `--disable-custom-all-reduce` + `NCCL_P2P_DISABLE=1` → NCCL IS the collective path) → removes init-OOM class on highest-GMU profile; itemized error replaces misleading "decrease GMU". PP=1 headline case doesn't hit us but the mechanism does.
  - **DUP/adjacent:** **PN367** (#45076/#44745 cudagraph mem clamp) edits the SAME `determine_available_memory`/`init_device` path → co-locate. **Genesis extra:** A/B that pre-NCCL budgeting doesn't shrink usable KV on clean TP=2; verify shortfall math under TQ k8v4 + GDN/Mamba state pool + MTP K=3 buffers.

## Theme C — CUDA Kernel Bounds/Overflow (compiled wheel — NOT vendorable as .cu)

- **PR 45530** [W3, S/LOW] — Bounds-check moe_permute reverse-map write (OOB global write on sentinel/padding rows). **Reachability = ZERO on Ampere** (consumers = CUTLASS FP8/FP4 SM90+/Blackwell + Humming only; our Marlin/AWQ Ampere MoE never calls moe_permute). **Adapt:** registry knowledge-entry `status=not-applicable-arch` + guard-asymmetry review heuristic + pin-bump gate (free once merged into a future wheel). Hard prereq if we ever enable FP8-native CUTLASS MoE (MTP K=3 + seq-padding sentinel rows are the exact trigger).

- **PR 45527** [W3, S/LOW] — int32/uint→int64 promotion in `merge_attn_states` + `permute_cols`. **Mixed:** permute_cols DORMANT (Machete/Hopper-only; our act-order reorder is inside `gptq_marlin_gemm`). `merge_attn_states` .cu IS on our live FA2 cascade path (fires at common_prefix≥256 & num_reqs≥8, reachable at max_num_seqs=8) but overflow needs ~6.7M batched tokens — **unreachable for us.** **Adapt:** defer-vendor into next pin-bump past libtorch_stable migration (clean 2-file widening); register `status=monitor`. **Lesson:** codify int64-promote-before-multiply checklist for our OWN kernels (TurboQuant KV pack/unpack, GDN prefill, MTP draft gather) where we control batch×seq.

- **PR 45466** [W3, S/LOW] — Output-alignment check in `vectorize_with_alignment` (misaligned-address crash, non-mult-of-8 head sizes via FlexAttention). **Crash structurally impossible** (we force FA2 → head%8≠0 raises; our head_dim=128; TQ uses its own Triton `triton_turboquant_store`, not the C++ reshape kernel). **Adapt:** port the unaligned-rows test philosophy to assert `triton_turboquant_store` is safe on TQ's non-16B slot pitches (612/k8v4); add alignment-guard comment IF we ever vectorize the TQ store; upstream-watch.

## Theme D — Tool-Call / Reasoning Parsing (CPU-side, our agent hot path)

- **PR 45479** [W2, S/LOW] — Gemma4 tool parser: emit missing `name`/`id`/`type` on combined end-delta (#45449). **DIRECTLY relevant** — Gemma-4 31B/26B-MoE + G4_79/80/81 under MTP K=3 produce exactly the "name chunk before `{` then `{}<end>`" sequence (common for zero-arg tools). **DUP nuance:** live PROD path (G4_T1 v2 #42237 accumulated-text rescan + v3 prep) is **already immune**; **PN375** (#44741) does NOT fix #45449 (only re-splits multi-boundary deltas). **Adapt:** (1) add v2/v3 regression test locking the immunity; (2) port name-recovery into PN375 (pristine/rollback profile) — use index-based recovery, do NOT lift the newer-base guards verbatim, keep G4_14 pad-strip.

- **PR 45464** [W3, S/LOW] — Harmony (GPT-OSS) streaming cleanup: gate REASONING/TOOL on parser presence; "exactly one DeltaToolCall per tool per chunk" (append args, not duplicate index). **We serve NO Harmony model** → runtime fix never fires. **Two real hooks:** (A) **COLLISION RISK** — parent #45171 [MERGED 2026-06-11] deleted 74 lines from `chat_completion/serving.py` and moved harmony streaming to `parser/harmony.py`; **PN288 + P107 anchor on exactly those `use_harmony`/`harmony_tools_streamed[i]` lines → next pin-bump past 2026-06-11 breaks both anchors** (load-bearing takeaway). (B) audit our qwen3xml coalescing — it keys on `id` (line 446) not `index`; continuation deltas carry no id → switch to index-based merge.

## Theme E — Observability / Health (frontend, zero hot-path cost)

- **PR 45453** [W2, M/LOW] — `/health/decode` forward-progress endpoint (ok/idle/prefilling/stalled). **HIGH on-the-nose:** PR's failure mode = "NCCL P2P deadlock surviving restart in TP>1, /health stays 200, requests hang forever" = our exact topology (2× A5000, PCIe, `NCCL_P2P_DISABLE=1`, TP=2, `restart:unless-stopped`). **Adapt:** vendor 6-file additive overlay (all anchors verified clean), default-OFF until wired; then the Genesis value-add: wire `/health/decode` watchdog → `tools/safe_container_recreate.py` (currently polls weak `/health` at :296), swap its readiness gate, add `verify_stress.py` probe. Tune `VLLM_DECODE_LIVENESS_STALL_SECONDS≈20-30s`, prefill threshold ≥30s (the "prefilling" status protects our 4.4s@32K GDN prefill from false 503). DP path inert/harmless (we're single-engine TP=2).

- **PR 45471** [W2, S/LOW] — `completion_tokens_details.reasoning_tokens` in `/v1/chat/completions` usage. **MEDIUM-HIGH observability:** all 4 PROD models run qwen3 reasoning parser; `/v1/responses` already surfaces it but our actual chat path doesn't. **Adapt:** vendor 3 hunks verbatim (clean), then EXTEND with Genesis-unique fields the PR defers — `accepted/rejected_prediction_tokens` from MTP K=3 counters → specdec efficiency per request; plumb into bench/dashboard + Genesis_proxy_ai/agregator cost parsing. Gain = TPOT attribution (reasoning vs answer) + MTP/TQ tuning lever; zero GPU cost (one O(n) token-id walk).

## Theme F — Not-Applicable-Arch (record + move on)

- **PR 45480** [W3, S/LOW] — CPU MoE fallback "Current vLLM config is not set" (SILU constructs CustomOp at forward time). Buggy line PRESENT in pin (cpu_fused_moe.py:53) but CUDA-only PROD never reaches it. **Adapt:** defensive-lint rule (never construct `*AndMul` CustomOp inside forward()/activation tables — `get_current_vllm_config()` raises outside model-init); one-line cherry-pick only if we add a host-side CPU checkpoint-validation lane.

---

## Cross-Cutting Summary

**Wave roll-up:** W1 = {45477}. W2 = {45497, 45517, 45479, 45471, 45453}. W3 = {45530, 45527, 45487, 45480, 45466, 45464}.

**Vendor-now (live correctness/stability on PROD):** 45477 (exposed mamba-poison bug), 45517 (init-OOM headroom), 45453 (silent-brick watchdog), 45471 (reasoning-token observability), 45479-PN375-track.

**Pre-position (inert until feature enabled):** 45497 (KV connector), 45530/45527 (Hopper/Blackwell migration), 45466 (TQ store vectorization).

**Pin-bump landmines (must re-anchor before bumping past dates):**
- **2026-06-11 (#45171 merge)** → re-anchor **PN288 + P107** (harmony serving block moved/rewritten) — flagged by 45464.
- libtorch_stable migration → fold in 45527 (int64) and 45530 (moe_permute) for free from the wheel.

**Synergies / duplicate flags vs vendored list:**
- 45477 ↔ **PN346/#43650** + **P85**: complementary (split-side vs hit-side); neither sufficient alone.
- 45497 ↔ **PN348/P58**: `num_output_placeholders=0` resets must not double-count async discard credit; order PN348 after.
- 45517 ↔ **PN367**: same `determine_available_memory` touch-point, co-locate.
- 45487 ↔ **PN118/#42551 + PN353/#44053**: upstream confirms our pre-grow-before-lock + torch.empty discipline (TQ WorkspaceManager).
- 45479 ↔ **G4_T1 v2/#42237 + v3 + PN375/#44741**: live path immune; PN375 leaves the #45449 gap.
- 45530/45466 ↔ **TQ overlay (pr42637)**: guard-asymmetry + alignment review heuristics for TQ kernel borrows.

**Reusable review heuristics harvested:** (1) every indexed global write in a borrowed MoE permute/scatter kernel must share the same sentinel/bounds guard; (2) `token_idx*num_heads*stride` / `row*row_stride` → int64 before the multiply; (3) cudagraph-captured WorkspaceManager consumers must not be reachable by an eager batch > `max_cudagraph_capture_size` before `lock_workspace` (use torch.empty/graph-private pool); (4) never construct a CustomOp at forward time; (5) streaming tool-call deltas: exactly one DeltaToolCall per tool/chunk, merge by **index** not id.

---

This is a synthesis task — I have all 12 PR studies in the DATA payload. No file inspection needed; I'll group by theme, preserve every PR with wave/essence/gain/effort+risk, flag dup-vs-Genesis status, and note cross-PR synergies. Output is the verbatim return value.

# Mini-Synthesis — Batch-2 Chunk 2/5 (12 upstream PRs)

All 12 verified NOT already vendored in Genesis (zero registry hits for every PR#). None duplicate a waves-1-2 vendored patch; closest neighbors flagged per-PR below.

## THEME A — Tool-calling / structured-output correctness (HOT PATH, vendor-now)

**#45389** `[ToolParser] braces in required-tool streaming` — **Wave 1** · effort S · risk LOW
- Essence: make required-streaming `_bracket_level`/`filter_delta_text` JSON-string-aware (track in_string/escaped; don't break on `,` or count braces inside string values; fix param extraction to use prefix not previous_text). Vendor as TextPatcher on `tool_parsers/streaming.py` (pin lines 27-35/38-59/144-148 byte-identical to pre-fix).
- Gain: kills corrupted streamed `function.arguments` (client JSONDecodeError) for every coder/agent payload with `{}"\` in string args (file paths, regex, shell, nested JSON). No latency cost.
- Dup/neighbor: NO Genesis patch on this file. Genesis tool overlays touch qwen3coder/gemma4/qwen3xml EXTRACT logic only.
- **Synergy ⚑ (critical, P68):** Genesis **P68** auto-upgrades tool_choice auto→required at long-ctx (>~12.5K) for qwen3.x/gemma — funnels long-ctx agent traffic INTO this exact buggy helper. This fix is a **prerequisite for safely flipping P68 on** (`GENESIS_ENABLE_P68_AUTO_FORCE_TOOL`, currently default_on=False). Also pairs with sibling **#45310** (Hermes `</tool_call>` boundary string-awareness, same bug class one layer up) — vendor together.

**#45390** `fix(security): input validation + grammar compilation timeouts (7 GHSA DoS)` — **Wave 2** · effort M · risk LOW
- Essence: protocol-layer bounds (MAX_LOGIT_BIAS_SIZE=1024, stop_token_ids=128, allowed_token_ids=1024, bad_words 1000×1000) + generic `run_with_timeout` (daemon-thread+Queue+Semaphore(4)) + `_check_regex_complexity` wrapping ALL XGrammar types (JSON/GRAMMAR/REGEX/STRUCTURAL_TAG), not just regex. Full gap in our pin (no even first-gen timeout). Vendor additive across utils.py / backend_xgrammar.py / envs.py / sampling_params.py / split protocol files (layout matches).
- Gain: converts an indefinite engine-thread wedge (a pathological tool schema hangs ALL decode on our single-user low-latency profile, async overlap does NOT save us — compile is on CPU engine loop) into a bounded reject. Input bounds turn whole-engine crash into clean 400.
- Genesis-specific: set `VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS=2` (PR default 10s violates 70-160ms TTFT SLO); the naive bracket-depth counter can false-positive on legit JSON-schema-derived regex → **add unit test feeding our real gemma4/qwen3_coder tool schemas through `validate_xgrammar_grammar` before enabling**.
- Dup/neighbor: NO collision with P62/PN58 (those touch serving.py + spec-decode grammar-mask timing, not backend_xgrammar/utils/bounds). Purely additive.
- **Synergy:** complements #45389 — both harden the XGrammar tool-calling hot path every Genesis model uses; vendor in same wave.

## THEME B — Scheduler / engine stability (defense-in-depth)

**#45406** `[BugFix] don't strand parked async-KV-load reqs behind unschedulable queue head` — **Wave 3** · effort S · risk LOW
- Essence: at the `new_blocks is None`→`break` site (pin scheduler.py:774-781), gate the bare break: `if self.running: break` else pop head into existing `step_skipped_waiting` + `continue`. Vendor as P34-style TextPatcher (all reused symbols present at scope), with self-retire drift markers + hybrid-active gate.
- Gain: stability only. Eliminates a permanent single-node freeze class (Running:0/Waiting:N/GPU-KV 0%, needs restart). Byte-identical in steady state.
- Relevance: literal trigger (WAITING_FOR_REMOTE_KVS / KV connectors / priority policy) is UNREACHABLE — we're connector-free FCFS. Value is the **recovery pattern** for our GDN-prefill-unfittable + self.running-empty window.
- Dup/neighbor: **direct twin of Genesis P34** (mamba zero-collapse deadlock guard, same TextPatcher idiom, same break site family). Composes with already-present #44560 admission gate (gate=prevent over-admit; #45406=recover post-admit). Optional extension: bound the skip loop so a perpetually-unfittable GDN head under MTP-K3 lookahead inflation is errored not skipped forever.

## THEME C — Generation-config / output-cap correctness

**#45417** `[Bugfix] unset HF default max_new_tokens for DiffusionGemma` — **Wave 2** · effort S · risk LOW
- Essence: model-specific PR (`override_generation_config.setdefault("max_new_tokens", None)`) — we never run DiffusionGemma, but the **silent server-wide output-cap class** is live: all our models launch with `--generation-config auto`, so any checkpoint shipping `max_new_tokens` in generation_config.json becomes an invisible hard ceiling. Mechanism (get_diff_sampling_param `is not None` filter, max_new_tokens→max_tokens rename) IS in our pin.
- Gain: correctness — prevents silent truncation of long completions (agent loops, gemma-4-31b code) invisible in our YAMLs.
- Genesis plan: (A) **AUDIT** each prod model dir's generation_config.json for max_new_tokens/max_length; (B) one-line schema fix — `schema_v2.py:366` currently REJECTS None, widen to `or v is None` (compose.py already json.dumps→null); (C) apply `max_new_tokens: null` where audit finds a cap. Strictly better than the PR: declarative + model-agnostic vs hardcoded-per-model.
- Dup/neighbor: NO. Genesis already owns the override_generation_config surface (schema_v2.py:288, compose.py:482-491, used by qwen3.6 YAMLs for temp/top_p/top_k) but can only SET, not UNSET — this closes that gap.

## THEME D — Multimodal / prompt-embeds correctness

**#45383** `[Bugfix] prompt_embeds for multimodal models` — **Wave 2** · effort S · risk LOW
- Essence: 2-file fix — `gpu_model_runner._preprocess` mm branch clobbers user prompt_embeds with placeholder-id embeddings (stale -1 ids → OOB gather → EngineDeadError); fix masks the write (`torch.where(is_token_ids…)`, zero placeholder ids). Plus config guard auto-disabling async-scheduling for `enable_prompt_embeds + is_multimodal`. Buggy block live at pin gpu_model_runner.py:3447-3454; all needed infra present.
- Gain: kills a deterministic EngineCore-crash / DoS vector on our Gemma-4 mm endpoints when prompt-embeds enabled. DORMANT today (our launchers don't pass `--enable-prompt-embeds`) — vendor env-gated default-OFF.
- Relevance: we DO serve the affected class (Gemma-4-31B-AWQ, Gemma-4-26B-A4B AWQ MoE, is_multimodal=True). Sibling **#45252** (M-RoPE prompt_embeds DoS) MERGED 2026-06-13, after our 06-08 pin — reframes whole family as security.
- Dup/neighbor: adjacent to **PN35** (gates inputs_embeds buffer ALLOCATION) — #45383 fixes CONSUMPTION of that same buffer; no anchor conflict (PN35 at __init__ ~755, this at _preprocess ~3447). Keep consumption-side fix in lockstep with PN35's allocation-side to avoid a half-patched buffer. **Synergy with PN35/PN371** (same OOB-gather-from-stale-placeholder class).

## THEME E — MoE perf / weight-loader (Ampere-shape relevant)

**#45379** `perf(moe): tuned fused_moe FP8 config for RTX 5090` — **Wave 2** · effort S · risk LOW
- Essence: adds one Triton autotune JSON for E=256,N=256,fp8_w8a8,block[128,128] — the **EXACT MoE shape as our PROD 35B** (Qwen3.6-35B-A3B-FP8, TP=2), showing -1.5%→-5.4% kernel time at our decode/MTP band. Selection keys on device string (NVIDIA_RTX_A5000) so the 5090 file never auto-loads — value is the GRID, not the file.
- Gain: Track-1 (drop-in via `VLLM_TUNED_CONFIG_FOLDER`, rename JSON to A5000 token): plausibly +1.5-5% MoE GEMM at hot batch band, but Ampere-uncertain (5090=Blackwell stages=2/warps=8 may regress at batch 16) — treat as SEED. Track-2 (Wave 3, real win): run `benchmark_moe.py` ON A5000 over batch 1/2/4/8/16/32, emit native NVIDIA_RTX_A5000 JSON with Ampere-correct num_stages.
- Dup/neighbor: complements (not dups) **P81** (block-scaled dense GEMM, "no A5000 JSON" — different kernel). Distinct from PN96 Marlin-MoE workspace. 35B-only (27B/Gemma run int4/AWQ, different get_default_config branch).
- Genesis-specific: verify exact `get_device_name()` string on live container first; A/B-gate on genesis_bench_suite before default_on.

**#45404** `fix(moe_wna16): tp_size via moe_config for RoutedExperts` — **Wave 3** · effort S · risk LOW
- Essence: reroutes `layer.tp_size`→`layer.moe_config.tp_size` in moe_wna16 qzeros reshape. **Behaviorally INERT on our pin** — RoutedExperts=FusedMoE TypeAlias, `layer.tp_size` is a working @property resolving to the same value; no AttributeError can fire. Do NOT vendor now (no-op + anchor cost).
- Relevance: touches OUR only AWQ-MoE weight-loader path (Gemma-4-26B-A4B AWQ MoE, gs=32 int4 at TP=2 — PN377's documented fallback route).
- Dup/neighbor: file heavily in-scope (**PN377** BLOCK_SIZE_K clamp, **PN368** marlin_gemm) but this accessor change unvendored. Plan: fold a boot-time guard into PN377's `run_boot_legality_check` asserting BOTH accessors resolve; if a future pin drops the `tp_size` property shim, our un-fixed lines 483/493 crash AWQ-MoE load → vendor then. Adopt convention: reference MoE parallel via `layer.moe_config.*`.

## THEME F — Attention / KV-cache (Ampere-UNREACHABLE, knowledge-extract)

**#45434** `vLLM plumbing for cuteDSL hd512 FP8-KV (FA4)` — **Wave 3** · effort S · risk LOW
- Essence: routes FP16-Q + in-kernel-FP8-KV-dequant via FA4 cute kernel (`supports_quant_query_input` gated `!= 4`; SM90-only `device_capability[0]==9`). **Literal path UNREACHABLE on Ampere SM8.6** (FA4 gated cc 9/10/11; we force FA2). Do NOT vendor as runtime patch. Depends on OPEN flash-attention #147.
- Extractable value (research-only): (A) **citable GSM8K data** — on gemma-4-31B fp8 KV is accuracy-neutral-to-positive (triton-fp8 **+1.9pt** vs bf16, FA4 only -0.23pt) → justifies trialling `kv_cache_dtype=fp8` for our Gemma-4-31B AWQ dense via the **reachable** FA2/Triton `is_quantized_kv_cache` path (potential ~2x attn-KV memory cut, longer ctx / larger max_num_seqs at no quality cost). (B) descale stride-0 footgun: PR's `.clone(memory_format=contiguous_format)` WAR — our pin builds `k_descale=layer._k_scale.expand()` the same way (benign on FA2 C++ today, but the canonical fix to cite if TQ k8v4 / future cute/Triton descale-consumer ever hits it). (C) `supports_quant_query_input` gating as blueprint for a hypothetical Ampere FP16-Q+FP8-KV FA2 path.
- Dup/neighbor: NO. Adjacent G4_79/80/81 (Gemma-4-31B AWQ+TQ) don't touch FA4.

**#45393** `[Bugfix] bound-check block-table indexing in CP gather cache kernels` — **Wave 3** · effort S · risk LOW
- Essence: two in-kernel bound checks in `cache_kernels.cu` (gather `block_table_id >= num_block_indices` continue; cp_gather `token_offset < 0` return) + resurrects rotted `test_gather_cache_oob`. Reachable ONLY on DeepSeek-MLA (use_mla / flashmla_sparse gated cc[9,10]) — NONE of our Qwen3.6/Gemma-4 GQA models are MLA, and Ampere excludes the sparse path. .cu source not in our wheel overlay → "land on next pin bump," not a wave-1 .py overlay.
- Extractable value: (1) future-proof guard IF we ever serve DeepSeek-V3.2/Kimi-Linear MLA (already in pin); (2) **rotted-test lesson** (TDD rule) — mirror in a Genesis binding-shape verifier asserting every `torch.ops._C_cache_ops.*` wrapper is call-exercised so a future signature drift fails loud; (3) audit our G4_79/80/81 + TQ k8v4 gather paths for the same unbounded `block_table` index pattern. Sibling #45384 (concat_mla_q int32 overflow) → same MLA-future bucket.

## THEME G — Bench-tooling / measurement fidelity

**#45423** `[Bugfix] correct prompt lengths for timed_traces benchmark` — **Wave 3** · effort S · risk LOW
- Essence: 4-line fix — TimedTrace.sample() does `tokenizer.decode(prompt_ids)` then ships the STRING; server re-encodes non-idempotently (PR: 6758→7253 tok, ~7% inflation). Fix: pass raw `prompt_ids` list, widen prompt type unions to add `list[int]` (Completions endpoint already forwards token-id lists). Buggy code live in pin (datasets.py:1556/1572/82, endpoint_request_func.py:69).
- Gain: measurement fidelity only. ~7% silent prompt-length inflation would bias TTFT/TPOT/prefix-cache-hit on length-sensitive GDN/Mamba + TQ-KV A/Bs (our long-ctx TTFT is GDN-prefill-dominated, 1.05s@8K / 4.4s@32K).
- Relevance: our harness (`g4_tq_ab_bench.sh` → `genesis_bench_suite.py`) uses fixed-N random prompts not `timed_trace`, so NOT hit today. Adopt the PRINCIPLE: add a client-intended-vs-server-reported length assertion (fail run on drift >0 tok), critical for TQ k8v4 KV-budget sweeps (max_num_seqs=2/8, exact token count decides if a 2nd seq fits) and MTP-K3 acceptance-rate denominators.
- Dup/neighbor: NO Genesis patch touches vllm/benchmarks/. Optionally vendor the 4-line fix IF we adopt trace-replay.

## THEME H — Build-system / pin-bump migration radar (no runtime value)

**#45415** `[12/n] final _C library kernel migration` — **Wave 3** · effort S · risk LOW
- Essence: libtorch-stable-ABI migration — RENAMES `csrc/quantization/activation_kernels.cu`→`csrc/libtorch_stable/...`, dissolves CUDA `_C` extension into `_C_stable_libtorch.abi3.so`. Op namespace UNCHANGED (`torch.ops._C.silu_and_mul_quant` stays). Migrated kernels DEAD on Ampere (persistent_masked_m_silu_mul_quant Hopper/Blackwell-gated; silu_and_mul_quant FP8-hw-gated — our 35B is Marlin W8A16 not native FP8). Zero runtime/latency value.
- Value: **pin-bump migration-radar** — any future Genesis C++/CUDA patch referencing the OLD csrc path or hard-coding the `_C` .so will fail to apply once this 12-PR series lands (iron-rule-#11 "patch path moved" class). Actions: registry note flagging the path move; post-bump smoke asserting `torch.ops._C.silu_and_mul_quant` + `persistent_masked_m_silu_mul_quant` still resolve; write any future Genesis kernel against the stable ABI under `csrc/libtorch_stable/`.

---

## Wave roll-up

| Wave | PRs | Theme |
|---|---|---|
| **1** | #45389 | tool-call streaming brace fix (vendor now; P68 prerequisite) |
| **2** | #45390, #45417, #45383, #45379 | DoS/grammar-timeout, output-cap audit, mm-prompt-embeds crash, 35B MoE tune |
| **3** | #45406, #45404, #45434, #45393, #45423, #45415 | scheduler-freeze recovery, MoE-accessor guard, FP8-KV knowledge, MLA bound-checks, bench fidelity, ABI migration radar |

## Cross-PR synergies (flagged)
- **#45389 + #45390** — both harden the XGrammar tool-calling hot path every Genesis model uses; same wave window. #45389 also pairs with sibling **#45310** (Hermes boundary, same string-awareness bug class).
- **#45389 ⚑ P68** — fix is the prerequisite to safely enable Genesis P68 long-ctx auto-force-required.
- **#45406 ↔ P34** — same scheduler break-site hardening, different root cause; composes with in-pin #44560 admission gate.
- **#45383 ↔ PN35 / PN371** — consumption-side fix for the buffer PN35 allocation-gates; same OOB-stale-placeholder class.
- **#45404 ↔ PN377 / PN368** — fold guard into PN377's boot legality check; adopt `layer.moe_config.*` convention.
- **#45379 ⟂ P81 / PN96** — complementary MoE tuning (fused-MoE grouped-GEMM vs block-scaled dense vs Marlin-MoE), not duplicate.
- **#45434 ↔ G4_79/80/81 + TQ k8v4** — fp8-KV evidence for Gemma-4-31B; descale stride-0 WAR relevant to TQ descale forwarding.
- **#45415 ⚑ pin-bump** — migration radar for any Genesis C++/CUDA patch on activation_kernels.cu / `_C` .so.

## Dup-vs-Genesis verdict
Zero of the 12 are already vendored. No verbatim-duplicate of any waves-1-2 patch. Five touch files Genesis already patches without colliding (#45389 vs qwen3/gemma extract overlays; #45390 vs P62/PN58; #45406 vs P34; #45383 vs PN35/PN371; #45404 vs PN377/PN368) — all additive or guard-only.

---

I'll synthesize these 12 PR studies into a compact themed report.

# Genesis Batch-2 Chunk 3/5 — Mini-Synthesis (12 upstream PRs)

**Scope:** PRs #45352–#45378 against pin `0.22.1rc1.dev259+g303916e93`. NONE of the 12 are vendored. Waves: 2 PRs in Wave-2 (act now), 10 in Wave-3 (research/next-pin).

---

## THEME A — Spec-Decode / Rejection Sampler (LIVE PROD hot path)

| PR | Title | Wave | Adaptation essence | Gain | E/R |
|----|-------|------|--------------------|------|-----|
| **45369** | Avoid materializing target_probs in rejection sampling (streaming LSE) | **2** | Vendor as **PN384** (opt-in, default OFF): replace full-vocab fp8 softmax with `compute_target_lse` + on-the-fly `exp(logit-lse)`. LIVE on 35B MTP-K=3 + PN90/P71 (draft_probs present → heavy `else` arm). A/B BLOCK_SIZE {8192/16384/38912}, num_warps (8 too high at 6–24 rows). | -8..-11% sampler-mechanism latency (with-draft-probs rows) → ~0.5–2% TPOT, larger on A5000 (byte-bound); +3.6–14.6MB transient reclaim, cudagraph headroom | M / MED |
| **45378** | Fix drafter slot-mapping type mismatch under DBO | 3 | DON'T vendor behaviorally. (a) Register as **merge-surface watch** on dflash/extract_hidden_states/llm_base_proposer dummy_run; (b) optionally backport the 1-line `first_slot_mapping_if_ubatched` normalizer as pure type-hardening. DBO is hard-`ValueError` with V2 runner on our pin → crash unreachable today | None (dormant); removes latent landmine in our MOST-patched spec files; reduces pin-bump friction | S / LOW |
| **45352** | Forward callable hf_overrides to draft config | 3 | Vendor as tiny **PN8b** (default OFF): add `compose_draft_hf_overrides`. Prod uses dict/empty overrides → byte-identical, zero perf. Enables fast low-GPU MTP-draft-shrink regression test | Zero prod delta; test velocity + forecloses target/draft config-divergence | S / LOW |

---

## THEME B — KV-Cache Quant Correctness (TurboQuant-adjacent)

| PR | Title | Wave | Adaptation essence | Gain | E/R |
|----|-------|------|--------------------|------|-----|
| **45361** | Fix INT8 per-token-head KV rounding (truncate→round) | **2** | Vendor as **PN299E-sibling** (same TextPatcher, same file, line-orthogonal to PN299E's num_warps block). Round-then-clamp gated on int8. Default OFF (our 4 profiles use k8v4). Add regression test asserting our TQ V-store `+0.5` ALSO matches round-ref | +~4.5pp relative quality on int8_per_token_head; ~0 latency (1 ALU/elem, mem-bound); unlocks trustworthy A/B baseline + locks-in TQ correctness | S / LOW |
| **45370** | Fused K-RoPE + static FP8 per-tensor KV write (.cu) | 3 | DON'T vendor the .cu (ROCm-family, wrong cache format, needs rebuild). EXTEND technique into `triton_turboquant_store`: apply NeoX RoPE inside `_tq_fused_store_fp8` before quant → removes one K HBM round-trip/layer/token. Parity test @1-ULP. Watch siblings #43355/#43572 (FA2 bf16 analog) | Low single-% TPOT (single-user), 1.2–2.7x cache-write step in burst; marginal TTFT (GDN-dominated prefill) | L / MED |

---

## THEME C — KV-Connector / PD / Offload Races (ALL DORMANT — no connector, TP=2, DCP=1)

| PR | Title | Wave | Adaptation essence | Gain | E/R |
|----|-------|------|--------------------|------|-----|
| **45357** | Defer block-free under async-sched + PD KV-consumer | 3 | Literal patch dormant (gate=`is_kv_consumer`). **GENERALIZE**: vendor `pop_blocks_for_free()` refactor + re-gate fence on `has_mamba_layers` (not is_kv_consumer) → close single-node stale-SSM-write window on 35B GDN under async+MTP-K=3 preempt. Coexist w/ P58 + PN-async-v2. New **PN384'** (name collision w/ B-PN384 — renumber). PR explicitly calls hybrid/GDN MOST dangerous | Eliminates latent GDN state-corruption on preempt-storm (burst/long-ctx); refactor cuts pin-bump conflict surface | M / MED |
| **45371** | Disable Mooncake TP put-striding when DCP>1 | 3 | No Mooncake/DCP on rig. (a) LEARN: adopt "every GET key was PUT" invariant test for PN95/PN203 host-RAM pools; (b) cheap defensive 1-line guard; (c) config-audit rule | Correctness insurance (prevents future ~75% cache-miss); reusable sharding test | S / LOW |
| **45363** | Drain in-flight KV transfers before sleep() unmaps VA | 3 | No sleep-mode/offload today. Park as **prerequisite-for-sleep-mode** (Genesis_Doc wants /sleep+/wake_up to replace teardown model-swap). HARVEST stream-drain audit template for TQ/async-overlap CUDA-event code | Makes future sleep-mode usable (else first-wake EngineDeadError); audit may catch latent illegal-address in our streams | S / LOW |

---

## THEME D — Quant Capability Unlocks & Fusion (Ampere-gated)

| PR | Title | Wave | Adaptation essence | Gain | E/R |
|----|-------|------|--------------------|------|-----|
| **45375** | Enable modelopt_mixed on Turing SM75 | 3 | Vendor ONLY the Ampere half (inherited #45306: cap **89→80**, NOT 75). Tiny TextPatcher, default OFF + model-gated on `MIXED_PRECISION` via model_detect. OMIT FlashInfer 7.5→8.0 (no-op SM86) | Unlocks NVFP4-mixed checkpoints on A5000 via Marlin W4A16 (else hard load-reject); our Marlin patches PN347/PN362 transfer | S / LOW |
| **45364** | Port RMSNormQuantFusionPass to manual eager fusion | 3 | As-is ~0 (no FP8-act linear on Ampere; we're Marlin W8A16, act stays bf16; compiler pass already covers W8A8). **EXTEND**: add `kBf16Passthrough` branch → fuse residual-add+RMSNorm (no quant) for ALL our models; survives enforce_eager A/B | ~0 as-is; bf16-widen → 1 fewer launch/RMSNorm-site/token, low-% TPOT in burst; debuggability under enforce_eager | M / LOW |

---

## THEME E — Tool-Call Streaming (PROD agent stability)

| PR | Title | Wave | Adaptation essence | Gain | E/R |
|----|-------|------|--------------------|------|-----|
| **45365** | Close qwen3_coder streaming tool args (missing `}`) | 3 | **DUPLICATE** of bug P64 already fixes (via #39598) — MUTUALLY EXCLUSIVE rewrite, do NOT stack (anchor-skip / `}}` risk). Add to upstream_watchlist; harden PN287 observer sub-bucket; on merge → retire-trigger candidate for P64 | Stability (else ~17% tool-call HTTP-400 poisoning); future -1 Genesis patch | S / MED |

---

## THEME F — Hardware-Irrelevant (reference only)

| PR | Title | Wave | Adaptation essence | Gain | E/R |
|----|-------|------|--------------------|------|-----|
| **45368** | Fix PunicaWrapperXPU LoRA TP/MoE bugs | 3 | XPU-only; our `punica_gpu.py` ALREADY has every fix. No LoRA in PROD. Keep as readiness reference + pre-flight checklist for future LoRA-on-A3B-MoE (TP=2 all-gather + EP global-index are where bugs bite) | Zero direct; de-risks future LoRA-on-MoE first-attempt crash | S / LOW |

---

## DUPLICATE / CONFLICT FLAGS

- **#45365 ⨉ P64 (active PROD)** — same bug, two upstream PRs (#45365 vs #39598/P64), incompatible rewrites. P64 wins; #45365 → watchlist/retire-trigger only.
- **#45364 vs pin compiler pass** — RMSNorm+quant fusion ALREADY achieved via torch.compile (`rms_quant_fusion.py`); PR is eager-mode re-delivery, not new capability.
- **#45368 already covered** — `punica_gpu.py` carries all 5 fixes; only dead `punica_xpu.py` is buggy.
- **#45375 ⊃ #45306** — stacked; vendor base's Ampere value (89→80), skip Turing (75) + FlashInfer floor.

## CROSS-PR SYNERGIES

- **#45369 + PN378/PN369/P71/P82/PN90** — all consume rejection-sampler probs; #45369 reconstructs them identically (ULP-stable). **CRITICAL**: #45369 flips load `other=0.0→-inf`, same form as PN378's #45060 drift-marker → re-audit lint_drift_markers self-collision; PN369's torch-side prob read is NOT in the kernel rewrite (needs LSE-reconstruct or local softmax if promoted).
- **#45357 + pin's `new_block_ids_to_zero`** — bracket the same stale-GDN-write failure: pin zeroes the READER, #45357 orders the WRITER. Both gated on mamba/GDN.
- **#45370 + #45361 + #45364(B)** — three independent attn-store/RMSNorm kernel-fusion wins, all reducing per-token launch/HBM on the 11-attn-layer path.
- **#45352 + PN8** — together close callable-hf_overrides fragility on BOTH build-side (#45352) and quant-read-side (PN8).
- **#45378 + DFlash family (PN21/23/24/38/40/275, G4_71-78)** — #45378 shifts dummy_run top-of-body → anchor-drift risk for our DFlash overlay on next pin.
- **PN384 NAME COLLISION**: #45369 (Theme A) and #45357-generalize (Theme C) both proposed as "PN384" — renumber one before vendoring.

## ACTION SUMMARY

- **Wave-2 (act now):** #45369 (PN384, M/MED), #45361 (PN299E-sibling, S/LOW).
- **Wave-3 vendor-worthy:** #45375 (Ampere gate, S/LOW), #45352 (PN8b, S/LOW), #45357-generalize (M/MED), #45370-extend (L/MED), #45364(B) (M/LOW).
- **Wave-3 track/learn only:** #45378 (merge-watch), #45371 (test pattern), #45363 (sleep-mode prereq), #45365 (watchlist/retire P64), #45368 (LoRA readiness).

---

This is a synthesis task on the provided data — no codebase exploration needed since all 12 PR studies are fully self-contained. Producing the grouped mini-synthesis directly.

# Genesis Upstream-PR Mini-Synthesis — batch-2 chunk 4/5 (12 PRs)

## Theme A — Log/Request-Validation Hardening (control plane, model-agnostic)

| PR | Title (short) | Wave | Adaptation essence | Gain | Eff/Risk |
|----|---------------|------|--------------------|------|----------|
| 45351 | Suppress re-raise on OpenAI ctx-length ValueErrors | 2 | Post-app-build hook registering ValueError/TypeError/OverflowError (+TemplateError/NotImplementedError) as concrete handlers; emit 1-line PN65-structured WARNING instead of zero | Log signal/noise; kills ASGI tracebacks on long-ctx ctx-overflow; body already 400 in pin | S / LOW |
| 45346 | Reject degenerate `structured_outputs` (DoS) | **1** | Source-overlay 2 guards after pin's empty-grammar guard (line 888); add gateway-edge guard in lazy_reasoner/request_router; file Wave-3 item for EngineCore step-loop isolation | **Fixes confirmed instance-wide DoS** on single-instance PROD (json_object=False / json="" → EngineDeadError) | S / LOW |
| 45310 | Hermes streaming tool-call JSON-string-aware boundary | 2 | **Lift `_find_end_token_outside_string` into shared helper**; harden qwen3coder (:350, :471 delimiter fallback) + qwen3xml (our actual parsers); feed PN287 observer; vendor hermes-as-is only if Open WebUI re-exposed | Eliminates silent tool-call truncation when arg value contains literal `</tool_call>` (coding workloads) | S / LOW |

**Synergy A:** 45351 ↔ PN65 v3 (both de-noise uvicorn logs — 45351 complements, keep 1-line WARNING). 45346 ↔ 45310 are both control-plane string/validation guards, kernel-agnostic, compose cleanly with TQ/MTP/async.

## Theme B — Scheduler / Async-Admission (PROD operating mode)

| PR | Title (short) | Wave | Adaptation essence | Gain | Eff/Risk |
|----|---------------|------|--------------------|------|----------|
| 45339 | Per-iteration max_num_seqs count under async sched | 2 | Apply 2.5/3 mechanisms (use_async_scheduling flag, RUNNING+WAITING cap on `len(num_scheduled_tokens)`, invariant); **SKIP has_scheduled_reqs hunk** — our allocate_slots lacks the param | Restores realized batch to cap on burst (8/4); **likely invalidates "async neutral" verdict** → re-A/B | S / LOW |
| 45349 | Mamba align single-block prefill cap (+ext KV) | 2 | Flag **inert in PROD** (align off, no LMCache, TQ-incompatible). **Harvest the snapshot-boundary invariant into PN95/PN203** to unblock GDN-state host-RAM offload; vendor dormant | Direct=0; strategic=safe GDN recurrent-state L2 offload (the PN95/PN203 gap) | M / LOW |

**Synergy B:** 45339 ↔ PN371(#45199)/P86(#45202) — same async-sched + MTP-K=3 surface; **45339 is the likely confounder** behind P86's "max_num_seqs=8 re-baseline" and the a5000 YAML "async neutral" note. 45349 → PN95/PN203/P85 (its invariant is the precondition PN95's filtered-out Mamba-demote path needs).

## Theme C — Spec-Decode / KV-Layout Correctness (Ampere TQ + GDN/MTP)

| PR | Title (short) | Wave | Adaptation essence | Gain | Eff/Risk |
|----|---------------|------|--------------------|------|----------|
| 45343 | MiMo DFlash spec-decode + AL-collapse / KV-layout fix | 2 | Port **generic** `_reshape_attention_kv_cache` + page_size_padded AttentionSpec fallback + `share_across_groups=False` + block_size guard + DFlash-SWA Triton masking. **Skip MiMo/mxfp4 (B200/FP4)**. A/B then **retire g4_74/g4_76, re-scope g4_72** | Fixes 0%/collapsed AL, cudaErrorIllegalAddress, gibberish on TQ+spec; restores AL→TPOT; collapses 4-patch firefight into 1 | **L / MED** |

**Synergy C (high-value):** 45343 ↔ **g4_72/g4_74/g4_76 + pn21** — SAME root cause (wrong drafter KV layout), upstream/generic vs our brittle per-config band-aids. DFlash-SWA kernel masking layers ON TOP of pn21 (config-level) — complementary, not duplicate. Hard dependency on TQFullAttentionSpec page_size_padded path. **This is the maintenance/correctness anchor of the chunk.**

## Theme D — Quant Loader Guards / Enablement (modelopt family)

| PR | Title (short) | Wave | Adaptation essence | Gain | Eff/Risk |
|----|---------------|------|--------------------|------|----------|
| 45306 | modelopt_mixed on Ampere SM80/86 | 2 | get_min_capability 89→80; **MANDATORY supplement the dropped `layer.orig_dtype` fix** in ModelOptFp8LinearMethod.create_weights (merged diff alone → AttributeError on FP8-dense Marlin); compose w/ P87 TP-pad + PN347 | **Unblocks NVFP4 siblings of all PROD models** (Qwen3.6-35B-A3B / Gemma-4-26B-A4B / 31B NVFP4); denser experts, more KV headroom. Tradeoff: drops TQ k8v4 KV | S / LOW |
| 45320 | Reject NVFP4 MoE missing per-expert scales | 3 | Literal path dormant (no modelopt-NVFP4 MoE today). **Generalize `_validate_loaded_expert_scales` to FP8 (35B-A3B) + AWQ MoE**; optional env-gated graceful-repair (#45212 opt-2); vendor literal as thin backport | Boot-time ValueError vs hours of prompt-dependent NaN; closes inf-gscale surface on PROD 35B-A3B FP8 MoE | S / LOW |
| 45312 | Reject unsupported compressed-tensors KV schemes | 3 | `and`→`or` one-liner; scope `applies_to compressed_tensors` (no-op on PROD); **add Ampere-fp8-KV-emulation warning** steering to turboquant_k8v4; wire into pre-launch config audit | Silent wrong-KV-dtype mis-load → explicit NotImplementedError; quality landmine (TPS-invisible) | S / LOW |
| 45331 | Mixed in/wt dtype for rms-norm quant fusion | 3 | **Fusion unreachable on Ampere** (fp8-activation-quant ops absent, SM8.6 no FP8 hw). Learning template for GemmaRMSNorm fusions; vendor dormant; OUR extension = register int8 per-token (PR leaves gap) | Direct ~0; indirect correctness template + future int8-fusion low-single-digit TPOT | M / MED |

**Synergy D:** 45306 ↔ **P87 (#40361 Marlin TP-pad, A4B/35B expert n-shard MIN_THREAD_N=64 @ TP=2)** + **PN347 (#44113 MarlinFP8 N==K)** + PN376 — all FP8/Marlin-on-Ampere; 45306's own PR notes its A100 runs needed P87-class padding. 45320 ↔ **PN61** (same defective kaitchup-NVFP4 checkpoint family, different surface) + PN376 (both patch modelopt.py — extend in place). 45320's generalized guard protects the 45306-enabled NVFP4 MoE loads → **45306 + 45320 are a pair** (enable + harden).

## Theme E — Test/Method-Only (no runtime delta)

| PR | Title (short) | Wave | Adaptation essence | Gain | Eff/Risk |
|----|---------------|------|--------------------|------|----------|
| 45334 | Noise-aware block-fp8 MoE test tolerance | 3 | **Adopt idiom, not file**: 2-leg `calc_diff(<1e-3)`+`assert_close(4×atol)` for our MoE A/B gates (calc_diff already in pin `utils/deep_gemm.py`); **recalibrate constants for Ampere SM8.6, not SM120**; extend params to our expert dims; also bound TQ/MTP drift (PR misses) | Zero runtime; CI robustness + dependency-free MoE accuracy-gate idiom + documented ~1.5%-RMS Ampere noise floor | S / LOW |

## Theme F — CUDA-Graph Capture Principle (architectural, MLA-only today)

| PR | Title (short) | Wave | Adaptation essence | Gain | Eff/Risk |
|----|---------------|------|--------------------|------|----------|
| 45309 | DSV4 breakable-cudagraph fewer eager breaks (27% TTFT) | 3 | **No DeepSeek/MLA model; flag off; breakable-cudagraph is MLA-only in pin** (FA2/GDN ops undecorated). Document principle next to PN204/PN365; Wave-3 probe if breakable-cg extended to GDN; **audit-guard VLLM_USE_BREAKABLE_CUDAGRAPH stays off** | Direct=0; reinforces PN204/PN365 design rationale (why P7b retired); future GDN-prefill capture template | S / LOW |

**Synergy F:** 45309 ↔ **PN204 (#42301) / PN365 (#42746)** — same TRT-LLM `maybe_execute_in_parallel` overlap principle; crystallizes the "keep metadata-independent GEMMs inside captured graph, break only at metadata-dependent op" rule that explains **why P7b (raw Stream) was retired**.

---

## Cross-cutting notes

**Duplicate/overlap flags vs vendored waves 1-2:**
- 45343 **substantially duplicates** g4_72/g4_74/g4_76 + part of pn21 (same root cause; upstream generic supersedes our band-aids — retire candidate after A/B).
- 45309 technique already vendored via **PN204/PN365** (different code path; principle-only here).
- 45349, 45320, 45331 are **dormant/learn-and-port** — no live-path duplication; harvested into PN95/PN203 (45349) and PN61/PN376 (45320).
- 45310, 45346 extend the **PN374/PN56/PN287** tool-parser correctness domain Genesis already owns.

**Wave roll-up:** Wave 1 = 45346 (DoS, ship first). Wave 2 = 45351, 45339, 45349, 45343, 45310, 45306. Wave 3 = 45334, 45331, 45320, 45312, 45309.

**Highest-value pairs:** (45306+45320) NVFP4 enable+harden for PROD-model siblings; (45343→retire g4_74/76) correctness + maintenance; (45339→re-baseline async) potential strategic flip of an existing "neutral" verdict.

**Effort/risk profile:** 11/12 are S or M / LOW. Sole L/MEDIUM is 45343 (hot KV-reshape path, mandatory A/B before retiring the g4 chain). Only one carries a "merged-diff-insufficient" trap: 45306 (must add the dropped `orig_dtype` line or AttributeError on FP8-dense Marlin).

---

This is a synthesis task over the 12 PR studies already provided in the prompt data — no codebase exploration or server access needed. The skill context confirms terminology (PN naming, waves, pin-gate, iron rules). I'll produce the compact themed markdown directly.

# Genesis Upstream-PR Mini-Synthesis — Batch-2 Chunk 5/5 (12 PRs)

Pin baseline: `0.22.1rc1.dev259+g303916e93`. Stack: 2× A5000 SM8.6, TP=2 single-node, Qwen3.6-35B-FP8/27B-int4, Gemma-4 26B/31B, MTP K=3, TQ k8v4, XGrammar-2. Next free slot referenced variously as PN384 (collision — see note).

## Theme A — Reasoning / tool-call correctness (HIGH relevance, our exact PROD path)

| PR | Wave | Essence | Gain | E/R |
|---|---|---|---|---|
| **45290** Forced-named empty-params → JSON object | **1** | Normalize the two forced-named branches of public `get_json_schema_from_tools` so a no-arg tool yields `{}` not a bare string/number; mirror PN70 wrapper style; optionally tighten `additionalProperties:false`. | Kills agent-loop parse-500s on parameterless tools (end_turn/noop/handoff) for 3/4 PROD families (qwen3_xml 35B/27B, gemma4 26/31B); qwen3coder shielded. Tiny TPOT save with the tighter grammar. | S / LOW |
| **45299** Qwen3 no-`</think>` short-answer → content | **2** | Flip `return model_output,None` → `return None,model_output` at qwen3_reasoning_parser.py:156-158. **Genesis must extend beyond the PR**: also fix the streaming else-branch (224-231) for parity. | Eliminates empty-content failures on the default thinking-enabled path — short direct answers currently trapped in `delta.reasoning`. Correctness, no latency change. | S / **MEDIUM** (changes default behavior; validate truncation path) |

## Theme B — KV / prefix-cache / spec-decode under MTP (HIGH — TTFT levers)

| PR | Wave | Essence | Gain | E/R |
|---|---|---|---|---|
| **44986** Eagle prefix-cache prefill fix | **1** | Thread `skip_eagle_pop=is_prefill_phase` through `find_longest_cache_hit` (Unitary+Hybrid) so the lookahead block isn't dropped during prefill. Filed on Qwen3.6-27B, block_size=1536. | Recovers 1 prefix-cache block/prefill on every MTP request; on short prompts (block_size>prompt) recovers the entire hit (0%→full). Direct TTFT win on cache-warm agent/multi-turn + long-ctx GDN prefill. **Zero decode cost.** | S / LOW |
| **45280** Role-aware spec-decode P/D (KV-corruption root fix) | 3 | Runtime **inert** (no disaggregation). Harvest the block-boundary invariant as a regression test for TQ `single_type_kv_cache_manager` num_lookahead alignment; merge-conflict watch on p79c/p79d/TQ. | 0 perf. Correctness insurance: locks MTP-K3 + prefix-cache + TQ boundary-alignment (today untested at prompt==block_size multiple). | S / LOW |
| **45283** Skip speculator sampling on P (depends on 45280) | 3 | Mechanism inert (no P/D). **Real lesson**: gate redundant spec work from config not per-step heuristic → build a prefill-only MTP-skip patch; adopt the D-side `0 if load_kv_async` cleanup. Drift-watch: rewrites `propose()` early-exit region PN90 anchors. | 0 from PR; potential low/mid-single-digit-ms TTFT from the prefill-only MTP-skip adaptation (needs server A/B). | M / LOW |

**Synergy:** 44986 + 45280 + 45283 all touch `v1/core/sched/scheduler.py` + KV coordinator/manager — vendor 44986 now, treat 45280/45283 as a paired drift-watch bundle for the next pin bump; the 45280 regression test guards the same alignment math 44986 perturbs. 44986 also lets P83 flip from "unsafe-to-reanchor" to "fully superseded" (it skips only in prefill, num_output_tokens==0, preserving the convergence invariant P83 flagged).

## Theme C — FP8/MoE/Marlin backend selection (HIGH conceptual, LOW current-config)

| PR | Wave | Essence | Gain | E/R |
|---|---|---|---|---|
| **45265** FP8 MoE + LoRA → Marlin reroute | **2** | 19-line LoRA→MARLIN short-circuit in `oracle/fp8.py` (applies byte-clean before pin L391). Tested by author on **our exact 35B-A3B-FP8**. Action (C): wire the phantom `VLLM_MOE_FORCE_MARLIN` to actually reroute (watchlist:737 flags it fake). | 0 perf now (no LoRA in PROD; selection already resolves Marlin). Removes a hard init-crash the day we try `--enable-lora` on 35B/27B FP8; closes phantom-env gap; vendored regression test gives oracle-drift detection on our 35B backend-selection path. | S / LOW |

**Tracking gaps to close:** (1) add 45265 to `upstream_watchlist.yaml` as the make-it-WORK twin of already-tracked sibling **#45130** (make-it-fail-cleanly) — vendor the pair together. (2) The 45265 selection-assertion test parametrized on k8v4/TQ block-fp8 keys belongs in the patch-verification harness.

## Theme D — Loader / frontend / config fail-fast (PATTERN reuse, LOW direct)

| PR | Wave | Essence | Gain | E/R |
|---|---|---|---|---|
| **45291** Validate runai_streamer extra_config | 3 | Vendor as **single-file** sibling of PN379 (#45196, DefaultModelLoader fail-fast) — reuse PN379 scaffolding/env-flag/static-mirror. We run no `runai_streamer` (default safetensors). Also bolt runai keys onto `scripts/audit_config_keys.py`. | 0 perf (construction-only, unused path). Completes the loader-validation family; confirms PN379 design matches upstream. | S / LOW |

## Theme E — Sampling capability (MEDIUM, opt-in only)

| PR | Wave | Essence | Gain | E/R |
|---|---|---|---|---|
| **45288** p-less (Rényi-2) hyperparameter-free sampler | 3 | Do NOT vendor verbatim (PR is dead code — never added to BUILTIN list). Adopt the idea: per-request opt-in `p_less` SamplingParam, **sparse** processor modeled on MinP (`if not count: return`), **and fix the MTP gap** (argmax-invariant ⇒ silent no-op under our K=3 rejection_sampler — needs `apply_with_spec_decode` or fold into `apply_sampling_constraints`). | Quality/UX: entropy-adaptive decoding option, argmax-invariant (zero impact on greedy agent runs). Zero latency when unused; one extra full-vocab pass when enabled. | M / LOW |

## Theme F — Build / typing / portability (LOW direct, METHOD/audit artifacts)

| PR | Wave | Essence | Gain | E/R |
|---|---|---|---|---|
| **45277** CUDA arch build-coverage gaps | 3 | All runtime paths dead on SM86 (`cutlass_group_gemm_supported(86)==False`, Mxfp4/Fp4 need SM100+). Use: (1) `tools/audit_arch_gates.py` asserting cap=86 helpers return our assumed values; (2) authoritative map for a future Ampere-trimmed custom wheel (`TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9"`, no `+PTX`); (3) pin-bump watchlist. | 0 runtime now. Future: smaller image + faster cold-start on Ampere-only wheel; gate-audit prevents silent SM-family regressions on bumps. | S / LOW |
| **45284** Transformers-backend q/k/v + gate/up fusion | 3 | `model_impl=transformers` set nowhere → never loads. Backport of a win **we already have natively** (gemma4/qwen3_moe packed QKV/gate_up). Harvest: structural auto-detector as a read-only audit (flag any un-fused dense block in 35B's 11 attn layers), the `_stream_with_fused_shards` pattern to harden PN365, and the split-K/aten::mm profiling method for our Ampere hot path. | 0 from PR. Indirect: detector could surface a PN365-class fusion (PN365 ≈ +1-3% wall_TPS, PN350 +1-1.5%); profiler gives A5000-specific split-K waste numbers. | S / LOW |
| **45296** MyPy fixes for kimi family (3/5) | 3 | Kimi models never loaded — do NOT vendor. Harvest the 3 idioms (`getattr(...)  # noqa: B009`, typed wrapper method vs attr-aliasing, `cast(Cfg, hf_config)`) as house style for our MTP/composite wrapper patches; stand up a mypy gate over `sndr/engines/vllm/patches` (currently none). | 0 perf. Overlay maintainability: catches wrapper-attr typos at lint instead of model-load crash. | S / LOW |
| **45269** RVV W4A8 INT4 GEMM (RISC-V CPU) | 3 | CUDA-only rig — `int4_scaled_mm_cpu` fenced behind `is_cpu()`, never fires. Use the already-present op as a **golden W4A8 CPU reference** to validate AutoRound INT4 dequant vs GPU Marlin (protects G4_79/80/81, P87); document the activation-zero-point compensation formula. | 0 GPU perf. CPU golden-reference test cuts AutoRound INT4 dequant-regression debug time. | S / LOW |

## Duplicates / collisions vs existing Genesis (waves 1-2 vendored)

- **No byte-duplicates.** Several are **complements**, not dupes:
  - 45299 complements **PN71** (hallucinated-`</thinking>` subcase) + **PN51** (thinking-disabled short-circuit; generalize its `not self.thinking_enabled` precondition for streaming parity); chain after **P27**; pair with **P107** (truncation→retryable).
  - 45290 is **disjoint from PN70** (PN70 = internal `_get_json_schema_from_tools` required path; 45290 = public forced-named branches) and **P68** (auto→required gate).
  - 45291 is the runai **twin of PN379** (#45196).
  - 45265 is the make-it-WORK twin of tracked **#45130**; adjacent to PN368/PN96/PN96b.
  - 44986 complements **PN346** (#43650, Mamba-side variant); supersedes retired **P83/P84**.
  - 45284 validates **PN350/PN365** fusion direction as upstream-canonical.
- **Slot collision to resolve:** 44986, 45290, 45265, 45299, 45291 each propose "PN384". Assign distinct sequential slots at vendoring time (skill: highest existing was PN383; verify against live registry before assigning).

## Cross-PR synergies

1. **Scheduler/KV bundle (44986 + 45280 + 45283):** same files; vendor 44986, drift-watch the other two together; 45280's regression test guards 44986's alignment math.
2. **Reasoning chain (45299 + PN51 + PN71 + P27 + P107):** one combined-apply test covering short-no-`</think>` / full-`</think>` / mid-thought-truncated, streaming + non-streaming, on 27B+35B.
3. **Tool-call correctness (45290 + 45265 test):** both harden the structured-output/XGrammar path our agent loops hit; co-locate their harness assertions.
4. **Loader fail-fast family (45291 + PN379 + audit_config_keys.py):** static-mirror extension shared.
5. **Ampere build/fusion/typing audits (45277 + 45284 + 45296):** three read-only `tools/audit_*` guards reinforcing the iron-rule-#11 "verify the gate fires" discipline on pin bumps.

## Vendor-priority recommendation

- **Wave 1 (vendor now, byte-clean, direct PROD value):** 45290, 44986.
- **Wave 2 (vendor with extension + validation gate):** 45299 (needs streaming-parity extension; MEDIUM risk — gate on truncation probe), 45265 (forward-compat + watchlist fix).
- **Wave 3 (pattern/audit/test artifacts, no runtime vendor):** 45280, 45283, 45288, 45291, 45277, 45284, 45296, 45269.

All twelve are S effort except 45283/45288 (M); all LOW risk except 45299 (MEDIUM — sole default-behavior change).
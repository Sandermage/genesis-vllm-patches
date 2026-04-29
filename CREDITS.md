# Credits

Genesis vLLM Patches stands on the shoulders of the upstream vLLM project + the open-source community that reports bugs, ships fixes, and shares code. This file lists every external contribution we depend on or have directly backported.

If you are listed here and would prefer your name to be removed or changed, please open an issue or contact us.

---

## Upstream PRs we backport

We backport not-yet-merged upstream PRs as opt-in patches when they fix bugs that affect our production workload. These are temporary — when the upstream PR merges, our backport auto-no-ops via drift markers, and operators can drop the env flag.

| Genesis patch | Upstream PR | Author | What we backport |
|---|---|---|---|
| **P58** | [vllm#40768](https://github.com/vllm-project/vllm/pull/40768) | **z1ying** | Fix CUDA crash from stale async placeholder tokens in speculative decoding. Touches scheduler.py + async_scheduler.py + request.py. |
| **P59** | [vllm#39055](https://github.com/vllm-project/vllm/pull/39055) | **ZenoAFfectionate (Zeno Pang)** | Promotes `<tool_call>` XML out of `<think>` reasoning into content for the qwen3_coder parser. Fixes the case where Qwen3.5/3.6 emit tool calls inside the thinking block. |
| **P60** | [vllm#40738](https://github.com/vllm-project/vllm/pull/40738) (Phase 1) | **Thomas Parnell (@tdoublep)** | SSM state pre-copy from accepted block in GDN+ngram speculative decoding. Bug originally reported by **bhaktatejas922** in [#39273](https://github.com/vllm-project/vllm/issues/39273). |
| **P60b** | [vllm#40738](https://github.com/vllm-project/vllm/pull/40738) (Phase 2) | **Thomas Parnell (@tdoublep)** | Triton kernel offset for conv state read/write under spec-decode. Companion to P60. |
| **P61** | [vllm#40783](https://github.com/vllm-project/vllm/pull/40783) (slice) | **ExtReMLapin** | Multi-tool first-occurrence in `extract_content_ids` — preserves all tool calls instead of dropping all but the last. |
| **P61b** | [vllm#40783](https://github.com/vllm-project/vllm/pull/40783) (streaming slice) | **ExtReMLapin** | Defensive overlap guard preventing partial `<tool_call>` tag fragments leaking as reasoning during streaming. |
| **P62** | [vllm#36138](https://github.com/vllm-project/vllm/pull/36138) | **sfbemerk** | Reasoning-aware grammar acceptance + spec-token validation. Fixes grammar bypass when `</think>` arrives within speculative-decode token batch. Original bug report by **cicirori** in [#34650](https://github.com/vllm-project/vllm/issues/34650). |
| **P64** | [vllm#39598](https://github.com/vllm-project/vllm/pull/39598) + Genesis call-site guard | **kotori-yan** (upstream) + **[Quentin Machu (@Quentin-M)](https://github.com/Quentin-M)** (call-site guard sub-patch E) | Streaming tool-call early-return removal — fixes empty `tool_calls` when MTP/spec-decode bundles last parameter + `</function>` in single delta. Plus widens safety-net trigger condition. **Sub-patch E added 2026-04-28** by [@Quentin-M](https://github.com/Quentin-M) ([fork commit 09688b1d](https://github.com/Quentin-M/genesis-vllm-patches/commit/09688b1d)): guards `delta_message.tool_calls[0]` access in `chat_completion_stream_generator` — fixes `IndexError: list index out of range` on the final streaming delta when P64's widened `_should_check` fires on `finish_reason` alone but `tool_calls` is empty. Excellent root-cause writeup + minimal correct fix — thank you Quentin. (v7.14, opt-in) |
| **P71** | [vllm#40819](https://github.com/vllm-project/vllm/pull/40819) (DRAFT) + 2 fixes from [@gemini-code-assist](https://github.com/gemini-code-assist) review | **Z. Golpayegani** (PR author) + **gemini-code-assist** (bug reviewer) | Block-verify rejection sampler (Sun et al. arXiv 2403.10444). Backported with TWO critical fixes from PR review: (1) shared `u` per request (PR used per-position), (2) `denom==0 → 1.0` ACCEPT (PR returned 0.0 — rejected perfect drafts). Algorithm paper: Sun, Mendlovic, Leviathan et al. ICLR 2025. (v7.42, opt-in, MTP-only) |
| **P75 enabler** | [vllm#25784](https://github.com/vllm-project/vllm/pull/25784) (already MERGED in pin) | Snowflake Arctic team | Suffix Decoding (per-prompt suffix tree, dynamic K). Algorithm: arXiv 2411.04975. We add operator-convenience auto-swap from method=ngram to method=suffix when env enabled. (v7.43, opt-in) |
| **P77** | [SGLang `adaptive_spec_params.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/adaptive_spec_params.py) (Apache-2.0 port) + [Nightjar arXiv 2512.22420](https://arxiv.org/abs/2512.22420) extension | **SGLang team** + **Nightjar authors** | Adaptive ngram K controller — EMA + hysteresis + auto-disable to K=0 on `accept_rate < 30%`. Fixes free-form ngram pathology where K=3 wastes 4 forward passes per accepted token. (v7.43, opt-in) |
| **P79b** | [vllm#40610](https://github.com/vllm-project/vllm/pull/40610) (OPEN draft) | **vLLM contributor** | Async × spec-decode proposer-sync race fix backport. (v7.46, opt-in) |
| **P79c** | [vllm#37629](https://github.com/vllm-project/vllm/pull/37629) (OPEN, in review) | **vLLM contributor** | Stale `spec_token_ids` cleanup for unscheduled requests. Improved in v7.49 with `-1` placeholder discrimination + `prev_step_scheduled_req_ids` gate. (v7.46, opt-in) |
| **P81** | [vllm#40925](https://github.com/vllm-project/vllm/pull/40925) (OPEN) | **tonyliu312** | `w8a8_triton_block_scaled_mm` low-M (M≤8) decode tuning: `BLOCK_SIZE_M` 64→16, `num_stages` 2→3 (non-ROCm). Direct hit for Qwen3.6-A3B FP8 + max_num_seqs=2 (M=1 typical, M=4 for MTP K=3 verify). Empirical +23% median decode on GB10 per upstream PR. (v7.48, opt-in) |
| **P82** | [SGLang `speculative_sampling.cuh` line ~107](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/speculative/speculative_sampling.cu) | **SGLang team** ([sgl-project/sglang](https://github.com/sgl-project/sglang)) | `threshold_single` OR-clause acceptance: `accept = vanilla_rejection OR (target_prob >= threshold)`. Targets the structural ceiling `clean_rate ≈ accept_rate^num_spec` identified in our v7.13 strict-ngram analysis. BIASED rule (loses unbiased-sampling guarantee); validated empirically on Qwen3.6-35B-A3B-FP8 — at threshold=0.3: +12% TPS on 128-2048 tok generation, quality 32/33 preserved. **Production-deployed v7.53.** |
| **P83** | [vllm#38182 root-cause analysis](https://github.com/vllm-project/vllm/issues/38182) by **uOnePiece** + comment by **@Angazenn** | **uOnePiece** + **Angazenn** | MTP keep-last-cached-block — guards against premature pop in `single_type_kv_cache_manager.py:457` when `use_eagle=True`. **Empirically DISPROVEN** as actual cause for our workload (Genesis debug instrumentation showed `find_longest_cache_hit` was never called because num_hashes=0). Kept opt-in as research artifact. (v7.53, opt-in) |
| **P84** | Genesis-original discovery 2026-04-27 (via P83 DEBUG instrumentation) | **Sandermage** | hash_block_size override (vllm#38182 actual root cause for hybrid). `scheduler.py:234` hard-codes `hash_block_size=self.block_size`; on hybrid Qwen3.6-MoE with P5 LCM-pad becomes ≥2048, so `request_block_hasher` computes 0 hashes for prompts <2048 tokens. P84 text-patches both scheduler.py + engine/core.py to read from `GENESIS_P84_HASH_BLOCK_SIZE` env (recommended: 16). Dual-site override required. (v7.53.3, opt-in) |
| **P85** | Genesis-original 2026-04-27 (synthesis of 6-round empirical investigation) | **Sandermage** | Hybrid fine-shadow prefix cache (MambaManager fix). `MambaManager.cache_blocks` early-returns for prompts <self.block_size; align-mode pads with null_blocks. P85 patches MambaManager to register shadow fine-grained hash entries (scale_factor=block_size/hash_block_size duplicates) + walk fine hashes on lookup with eviction-safety re-derive verify. Requires P84. **NOTE:** v756 sustained-load crash bisect 2026-04-27 confirmed P85 NOT the cause (reproduced without it). (v7.53.4, opt-in) |
| **P86** | [vllm#40876](https://github.com/vllm-project/vllm/pull/40876) (OPEN) | **aaronagent** | ngram batch_propose O(N\*K) → O(N+K) direct-fill. Replaces O(N\*K) `i in valid_ngram_requests` membership scan with O(N+K) direct-fill loop iterating only the valid ngram requests. Algorithmic improvement, no behavioral change. (v7.53, opt-in) |
| **P87** | [vllm#40361](https://github.com/vllm-project/vllm/pull/40361) (OPEN) | **vLLM contributor** | Marlin W4A16/W8A16 sub-tile output dim pad-on-load. MarlinLinearKernel requires per-rank `out_features % GPTQ_MARLIN_MIN_THREAD_N (=64) == 0`; sub-tile shards (Qwen3.5 GatedDeltaNet at TP>=2, Intel/Qwen3.6-35B-A3B-int4-AutoRound n=32 shard) fail and force a slow non-Marlin fallback. P87 wraps three MarlinLinearKernel methods to zero-pad qweight/scales/qzeros/bias on load + slice extra columns at apply. Runtime cost zero. PR bench: +24% on 2× RTX 3090 SM 8.6 (137→170 t/s). **NOTE 2026-04-28:** v764c boot triggers torch.dynamo "marked as skipped" during cudagraph capture; class-rebind wrapper indirection incompatible with dynamo trace. Needs rewrite as text-patch. (v7.62, opt-in, blocked) |
| **P91** | [vllm#39460](https://github.com/vllm-project/vllm/pull/39460) (CLOSED but valid) | **vLLM contributor** | AutoRound row-parallel group cdiv + start-idx fix (non-MoE portion). `gptq_marlin.py` computes `scales_and_zp_size = input_size_per_partition // group_size` — when not divisible, drops trailing partial group of scales. Combined with `parameter.py` `tp_rank * shard_size` start_idx (in scale-rows units, but source tensor is indexed in input-element-groups), rank-1 scales load from wrong offset → silent dequant corruption. P91 (a) replaces both floor-divs with `cdiv()`, (b) tags scales/qzeros with row metadata, (c) makes load_row_parallel_weight compute group-aware start_idx. Hypothesized cause of Lorbus INT4 < INT8 perf gap. Sister bug vllm#38064 had 2.72× latency improvement when fixed. (v7.62, opt-in) |
| **P93** | Genesis-original 2026-04-28 (cross-engine research synthesis) | **Sandermage** | AllSpark bypass for INT8 W8A16 group_size=-1. vLLM CUDA kernel selector picks AllSparkLinearKernel first when `group_size=-1`; this disables P87/P91. P93 prepends "AllSparkLinearKernel" to `VLLM_DISABLED_KERNELS` at plugin register() time when `GENESIS_FORCE_MARLIN_W8A16=1`, forcing fallback to Marlin (which also supports `group_size=-1` + `uint8b128` per `marlin_utils.py:30,75` — not a workaround, just a priority change). AllSpark on consumer Ampere is hardcoded for A100 (108 SMs, 40MB L2) — under-tuned on A5000 (64 SMs, 6MB L2). Per Phase 1D research (cross-engine deep mining 2026-04-28). (v7.62, opt-in) |
| **P94** | [vllm#41043](https://github.com/vllm-project/vllm/pull/41043) (OPEN) | **wangluochao902** | Spec-decode `prepare_next_token_ids_padded` zero-alloc. Replaces 4-step Python+numpy chain (`.tolist()` + list-comp + `np.array()` + copy) in `LLMBaseProposer.prepare_next_token_ids_padded` hot path with an in-place loop. Removes GPU→CPU sync, list-comp Python objects, np.array allocation. PR author measured **P99 TPOT -9.3%** on Llama-3.1-8B + Eagle3 TP=4. For our MTP K=3 single-stream: expected +2-4% wall TPS + tighter CV. Applies to ALL spec methods (Eagle, MTP, ngram, draft model). (v7.62.7, opt-in) |

## Upstream issues that informed our investigation

Our investigation of the spec-decode + tool-call bug class (Genesis v7.11/12/13) was informed by these reports and discussions:

| Issue | Reporter | Insight we used |
|---|---|---|
| [vllm#40807](https://github.com/vllm-project/vllm/issues/40807) | **noonghunna** | TurboQuant + spec-decode + chunked-prefill `tolist()` crash — shared isolation matrix that narrowed our hypothesis space across six probes. |
| [vllm#40831](https://github.com/vllm-project/vllm/issues/40831) | **noonghunna** | Degenerate token loop (tool-call output corruption) — the original bug we're fixing. Six-probe ladder + `cudagraph_mode=NONE` workaround data. |
| [vllm#40756](https://github.com/vllm-project/vllm/issues/40756) | **SongXiaoMao** | MTP IMA on long sequences (Qwen3.6-27B-FP8) — sibling-bug data point. |
| [vllm#39273](https://github.com/vllm-project/vllm/issues/39273) | **bhaktatejas922** | Original report of GDN+ngram corruption. Their root-cause analysis comment narrowed the search to SSM state block indexing. |
| [vllm#34650](https://github.com/vllm-project/vllm/issues/34650) | **cicirori (yinghui)** | MTP + reasoning + structured output `</think>` detection failure. Identified the timing mismatch between `num_computed_tokens` and `all_token_ids`. |
| [vllm#36138](https://github.com/vllm-project/vllm/pull/36138) | **sfbemerk** | Grammar bypass when reasoning ends in spec tokens. **Backported as P62 in v7.13.** |
| [vllm#37150](https://github.com/vllm-project/vllm/issues/37150) | **HF-001 (kx)** | ngram + async-scheduling 1.22% acceptance rate report — informed our ngram_gpu Path B test. |
| [vllm#39056](https://github.com/vllm-project/vllm/issues/39056) | **ZenoAFfectionate** | Companion issue for #39055; identified the qwen3_reasoning + qwen3_coder parser interaction class. |
| [vllm#40880](https://github.com/vllm-project/vllm/issues/40880) | **noonghunna** | MTP × TurboQuant × FULL cudagraph degenerate output bug — exact reproducer + cross-rig confirmation that motivated v7.14 P65 root-cause investigation. |
| [vllm#28015](https://github.com/vllm-project/vllm/issues/28015) | (vLLM contributor) | Identified the broader bug class: capture-size divisibility in uniform decode CUDA graphs producing unexpected prefill branches. Inspired our P66 backport approach. |
| [vllm#23679](https://github.com/vllm-project/vllm/pull/23679) (closed/stale) | **fhl2000** | Proposed `cudagraph_capture_sizes` divisibility filter — never merged but provided the algorithmic blueprint we backported as P66. |
| [vllm-ascend#7148](https://github.com/vllm-project/vllm-ascend/pull/7148) (merged) | (vLLM-ascend contributor) | Concrete fix for `num_tokens_padded % uniform_decode_query_len == 0` assertion failure; confirmed the Ascend backend hits the same class of bug. |
| [vllm#38556](https://github.com/vllm-project/vllm/pull/38556) (merged) | **MatthewBonanni**, co-authored by **SandishKumarHN** | Stale `num_accepted_tokens_cpu` in hybrid models under async spec decode. Already in our pin via `757068dc6` — confirmed in our investigation. |
| [Wasif Basharat — "An Overnight Stack for Qwen3.6-27B" (Medium, Apr 2026)](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) | **Wasif Basharat** | Documented the `query_start_loc.tolist()` cudagraph capture crash in TurboQuant's continuation-prefill branch. Genesis loads `external_probe/patch_tolist_cudagraph.py` (which mirrors his `is_current_stream_capturing()` guard pattern) at boot. Source of inspiration for understanding the bypass design space. |
| [Liu et al. 2023 — "Lost in the Middle"](https://arxiv.org/abs/2307.03172) | **Liu, Lin, Hewitt et al.** | Empirical evidence that LLMs lose attention to instructions buried in middle of long context — motivated our P69 reminder-injection approach. |
| [Sun et al. 2024 — "Block Verification Accelerates Speculative Decoding"](https://arxiv.org/abs/2403.10444) | **Ziteng Sun, Uri Mendlovic, Yaniv Leviathan, Asaf Aharoni, Jae Hun Ro, Ahmad Beirami, Ananda Theertha Suresh** | Algorithm paper for our P71 backport. Theorem 4 proves block-verify is unbiased and ≥ per-token rule in expected accepted tokens. |
| [Snowflake / Arctic Inference — "Suffix Decoding"](https://arxiv.org/abs/2411.04975) | Snowflake Arctic team | Algorithm paper for our P75 enabler. Per-prompt suffix tree + dynamic K speculation; vLLM integration via PR #25784 (already in our pin). |
| [SGLang `adaptive_spec_params.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/adaptive_spec_params.py) | SGLang team | EMA + hysteresis algorithm we ported as P77 (Apache-2.0). Adaptive K controller for ngram speculation. |
| [Nightjar — adaptive speculation arXiv 2512.22420](https://arxiv.org/abs/2512.22420) | (Nightjar authors) | MAB-style auto-disable on low acceptance — extension we added on top of SGLang's logic in P77. |

## Cross-rig collaboration

| Person | Repo | Contribution |
|---|---|---|
| **[@noonghunna](https://github.com/noonghunna)** | [`qwen36-27b-single-3090`](https://github.com/noonghunna/qwen36-27b-single-3090), [`qwen36-dual-3090`](https://github.com/noonghunna/qwen36-dual-3090) | (1) Apache-2.0 `patch_tolist_cudagraph.py` adapted as our **P78** with attribution. (2) Cross-validation of our P65 root-cause for #40880 on RTX 3090 (cited in their dual-3090 README turbo section). (3) Multiple bug isolation matrices (#40807/#40831/#40880) that informed our v7.13/v7.14 investigation. |

## Earlier upstream PRs we depend on (already merged)

These are vLLM core features we build on top of. Genesis patches assume they are present.

| Reference | What we use |
|---|---|
| **TurboQuant KV cache compression** ([vllm PR #38280](https://github.com/vllm-project/vllm/pull/38280)) | The k8v4 KV-cache format that all our TQ-related patches (P22, P23, P36, P38, P40, P44, P51) build on. Original implementation by the vLLM team. |
| **GDN / hybrid linear-attention support** | Qwen3-Next layer pattern handling, mamba_cache_mode, GatedDeltaNetAttention class. Our P28, P34, P39a/b, P46 patches depend on this. |
| **PR #35687** (Qwen3 `<tool_call>` implicit reasoning end) | Pattern reference for our P12 mirror implementation. |
| **PR #40092** ([TurboQuant] enable FA3/FA4 for prefill paths) | The `fe9c3d6c5` commit we pinned for Genesis v7.0-v7.12. |
| **PR #40129** (A5000 MoE tuning JSON) | Sander's own community contribution — closed without merge upstream (consumer-card config policy), bind-mounted as overlay. |
| **PR #40384** (KV-cache token capacity for hybrid groups) | Upstream-merged version of our earlier Patch 9; P9 auto-skips when this is detected. |
| **PR #40074** (TurboQuant decode kernel IOOB safe_page_idx clamp) | Upstream-merged; Patch 13 auto-skips when present. |
| **PR #40792** (TurboQuant grouped decode stage1) | Upstream-merged; P40 auto-skips when present. |
| **PR #40798** (Workspace manager unified TQ buffer) | Upstream-merged; P36 + P51 graceful-handle. |

## Independent reference implementations we studied

These projects either implement the same algorithms or share design ideas with our work. We didn't directly copy code from them, but they helped us understand the problem space:

| Project | Author / org | What we learned |
|---|---|---|
| [0xSero/turboquant](https://github.com/0xSero/turboquant) | 0xSero | Canonical TurboQuant reference (1196★). Studied algorithm internals (Lloyd-Max, polar quant, QJL). |
| [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) | mitkox | Workstation-GPU support fork. Reference for A5000/A6000 deployment patterns. |
| [Alberto-Codes/turboquant-vllm](https://github.com/Alberto-Codes/turboquant-vllm) | Alberto-Codes | Plugin-style TurboQuant integration (45★, 8 models validated). Architectural inspiration for our `_genesis/` package layout. |
| [SGLang TurboQuant PR #21617](https://github.com/sgl-project/sglang/pull/21617) | scottgl9 | Parallel TurboQuant port to SGLang — wires through FlashInfer/Triton hooks. Confirmed the ecosystem-wide bug class. |
| [JartX/vllm](https://github.com/JartX/vllm/pull/11) | JartX | FP16 rotation approach for TurboQuant continuation prefill. Foundation for our Patch 20. |
| [noonghunna/qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090) | noonghunna | Six-probe isolation methodology + `patch_tolist_cudagraph.py` (`is_current_stream_capturing()` runtime detection pattern). |
| [tdoublep/vllm](https://github.com/tdoublep/vllm) (Tom Parnell's branches) | Thomas Parnell | Reference impl for #40738. We backport Phase 1 + Phase 2 directly. |


## Maintainers / authors

**Genesis vLLM Patches** is authored and maintained by:

- **Sandermage (Sander) Barzov Aleksandr** — Odessa, Ukraine — author, maintainer, primary investigator. Operates the homelab production stack the patches are tuned for.

All design decisions and patch acceptance are gated on empirical reproducer testing — see commit messages and `Genesis_Doc/` reports for full audit trails.

---

## Acknowledgements

**Independent multi-rig confirmation of v7.13 + #40875** (post-deploy):
@noonghunna ran our v7.13 patch tree + strict ngram config on a different rig (1× RTX 3090) with a different model family member (Qwen3.6-27B dense hybrid, int4-AutoRound, `turboquant_3bit_nc` KV) and confirmed the `prompt_lookup_min=8` config-only fix from [vllm#40875](https://github.com/vllm-project/vllm/issues/40875) works cross-rig cross-model. The same Probe 9 also identified that **MTP × TurboQuant × cudagraph** is a separate bug class that v7.13 does not cover — that finding will inform a future fix attempt and a follow-up upstream issue (to be opened from the rig where the bug actively reproduces). Without this independent re-test, the report would have remained single-rig observation rather than confirmed bug class.

Special thanks to:

- The **vLLM core team** ([@WoosukKwon](https://github.com/WoosukKwon), [@zhuohan123](https://github.com/zhuohan123), [@bnellnm](https://github.com/bnellnm), [@youkaichao](https://github.com/youkaichao), [@LucasWilkinson](https://github.com/LucasWilkinson), [@njhill](https://github.com/njhill), [@mgoin](https://github.com/mgoin), [@simon-mo](https://github.com/simon-mo), and many others) for building and maintaining vLLM.
- **noonghunna** for the rigorous bug-isolation methodology that made cross-rig data sharing possible (#40807 / #40831).
- **Thomas Parnell (@tdoublep)** for PR #40738 which addresses our most impactful production bug class.
- **bhaktatejas922** for the original GDN+ngram bug report (#39273) and the deeper root-cause analysis comments.
- **z1ying**, **ZenoAFfectionate**, **ExtReMLapin**, **sfbemerk**, **kotori-yan**, **tonyliu312** for the additional fix PRs we backport / will backport.
- **JartX** for the FP16 rotation approach.
- **0xSero**, **mitkox**, **Alberto-Codes** for parallel TurboQuant implementations that helped triangulate the algorithm space.
- **SGLang team** ([sgl-project/sglang](https://github.com/sgl-project/sglang)) for the `threshold_single` OR-clause acceptance rule (P82) and the adaptive K controller (P77 basis) — both ported as Apache-2.0 derivatives with attribution.
- **Nightjar authors** for the MAB-style auto-disable on low acceptance (extension we built on top of SGLang's logic in P77).
- **gemini-code-assist (bot)** for the review on PR #40914 catching the missing buffer-reuse parameters in our P67b routing — bot review made the code both more correct AND faster (v7.45 outcome).

---

## How to add yourself

If you contributed to vLLM upstream and we missed crediting you here, or if your code influenced one of our patches, please open an issue or PR with the addition. We aim to credit conservatively and accurately.

If you contributed a bug report whose data we used in our investigation, please ping us — we'd like to add you to the "Upstream issues that informed our investigation" table above.

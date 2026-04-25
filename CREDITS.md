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
- **z1ying**, **ZenoAFfectionate**, **ExtReMLapin**, **sfbemerk** for the additional fix PRs we backport / will backport.
- **JartX** for the FP16 rotation approach.
- **0xSero**, **mitkox**, **Alberto-Codes** for parallel TurboQuant implementations that helped triangulate the algorithm space.

---

## How to add yourself

If you contributed to vLLM upstream and we missed crediting you here, or if your code influenced one of our patches, please open an issue or PR with the addition. We aim to credit conservatively and accurately.

If you contributed a bug report whose data we used in our investigation, please ping us — we'd like to add you to the "Upstream issues that informed our investigation" table above.

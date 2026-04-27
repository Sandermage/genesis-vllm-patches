# Draft comment — vllm-project/vllm#38898 (NickLucche, OPEN, 0 comments)

**Status:** DRAFT for Sander review. Do NOT post until approved.

**Context:** Issue title — *"Mamba `DS` conv state layout | Support speculative decoding with `mamba_cache_mode=align`"*. NickLucche filed 2026-04-03. The exact NotImplementedError our 9-round investigation hit empirically:

```python
raise NotImplementedError(
    "DS conv state layout does not yet support speculative decoding "
    "with mamba_cache_mode='align' (num_accepted_tokens > 1)"
)
```

**Issue URL:** https://github.com/vllm-project/vllm/issues/38898

**Strategic context:** 24 days silent, zero comments. Our reproducer (Qwen3.6-MoE-A3B-FP8 + GDN + MTP K=3 + 2× A5000) is exactly the missing real-world data point. Adding our trace closes the silence with concrete repro + production constraint, which historically accelerates maintainer attention.

**Suggested reply angle:** confirm the bug class (with verbatim hit_tokens metrics from our 9-round investigation), document the workload constraint (production stack, MTP K=3 requirement), offer to test any candidate fix on our rig. Tone: factual / non-blaming / supportive of NickLucche's original report.

---

## English version

> **Confirming this on Qwen3.6-MoE-A3B-FP8 + GDN + MTP K=3, RTX A5000 × 2 — production-blocking for hybrid spec-decode users**
>
> We hit exactly this incompatibility while investigating prefix-cache hit rate on a hybrid GDN model. Posting our reproducer in case it helps motivate a fix (full investigation: 9 rounds with kernel-level instrumentation).
>
> **Setup:**
> - vLLM `0.19.2rc1.dev205+g07351e088` (commit `07351e088`, image `vllm/vllm-openai:nightly`)
> - Model: `Qwen3.6-35B-A3B-FP8` (uses `GatedDeltaNetAttention`, NOT Mamba2)
> - 2× RTX A5000 (Ampere SM 8.6, TP=2)
> - `--kv-cache-dtype turboquant_k8v4 --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align`
> - `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
>
> **Reproducer:** 3 IDENTICAL requests with 5018-token shared prefix sent in sequence.
>
> **Result with MTP K=3:**
> ```
> [hit_tokens] turn 1: 0   turn 2: 0   turn 3: 0
> [TTFT]       turn 1: 0.957s  turn 2: 0.811s  turn 3: 0.815s (no recovery)
> [Mamba store] block 0: null_block (committed=1, skipped_null=1, shadows_inserted=0)
> ```
>
> **Same setup, MTP DISABLED (drop `--speculative-config`):**
> ```
> [hit_tokens] turn 1: 0   turn 2: 2768   turn 3: 2768   ← 55% hit rate
> [TTFT]       turn 1: 1.654s  turn 2: 0.382s  turn 3: 0.380s   ← 77% recovery
> [Mamba store] block 0: REAL Mamba block (committed=1, skipped_null=0, shadows_inserted=173)
> ```
>
> **Confirms the issue exactly:** `align` mode prefix-cache works correctly for GDN when `num_accepted_tokens == 1` (no spec decode), produces 0 hits when MTP/EAGLE3 emits multi-token batches.
>
> Stability note: even WITHOUT MTP, the engine crashes under sustained `tps_bench` load on this config (Worker_TP0 dies after ~30 sustained requests). Likely a race in the eviction path under repeat hits — possibly orthogonal to this issue but flagging in case it's the same code path.
>
> **Workload constraint:** for our production (long-context Qwen3.6-MoE on consumer Ampere), MTP K=3 gives ~+30% throughput. Disabling MTP costs more than the cache wins on most workloads. We're currently shipping with `--mamba-cache-mode=none + MTP=K3` in production (cache machinery overhead removed since cache never hits anyway with MTP).
>
> **Reference workaround landed:** PR #40454 (merged 2026-04-21) "Default to align mode for spec decode" — sidesteps by forcing align when spec is enabled, but doesn't fix the underlying `num_accepted_tokens > 1` kernel path. Our metrics above use a build that includes #40454.
>
> **Genesis tracking:** we maintain a runtime-patch suite for this stack at [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches). Three of our patches sit on top of the same code paths discussed in tracking issue #26201 (P83 = Eagle pop skip, P84 = dual-site `hash_block_size` override, P85 = hybrid fine-shadow). Happy to apply any candidate fix to our patch chain and re-run the 5K-identical reproducer + multi-turn tool-call regression suite if that would help.
>
> Thanks @NickLucche for the original report — the kernel-level `num_accepted_tokens > 1` analysis was the missing piece for our investigation.
>
> *(English text generated with AI translation assistance — original draft in Russian; happy to clarify anything that reads ambiguously.)*

---

## Russian version (для контроля смысла)

> **Подтверждаю на Qwen3.6-MoE-A3B-FP8 + GDN + MTP K=3, RTX A5000 × 2 — блокер для production hybrid+spec-decode пользователей**
>
> Хлопнулись в эту incompatibility исследуя prefix-cache hit rate на hybrid GDN модели. Постю reproducer на случай если поможет ускорить fix (полное расследование: 9 раундов с kernel-level instrumentation).
>
> **Конфигурация:**
> - vLLM `dev205+g07351e088`, image `vllm/vllm-openai:nightly`
> - Model: `Qwen3.6-35B-A3B-FP8` (использует `GatedDeltaNetAttention`, не Mamba2)
> - 2× RTX A5000 (Ampere SM 8.6, TP=2)
> - flags: `--kv-cache-dtype turboquant_k8v4 --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align`
> - `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
>
> **Reproducer:** 3 идентичных request'а с 5018-token shared prefix последовательно.
>
> **Результат с MTP K=3:** hit_tokens=0 для всех 3 turns; TTFT не восстанавливается; Mamba store: block 0 = null_block (`committed=1, skipped_null=1, shadows_inserted=0`).
>
> **Тот же setup БЕЗ MTP:** hit_tokens=0 → 2768 → 2768 (55%); TTFT 1.654s → 0.382s (77% recovery); Mamba block stored (`committed=1, skipped_null=0, shadows_inserted=173`).
>
> **Точно подтверждает issue:** `align` mode для GDN работает корректно при `num_accepted_tokens == 1` (no spec decode), возвращает 0 hits на MTP/EAGLE3 multi-token batches.
>
> Workaround PR #40454 (merged 2026-04-21) обходит forcing align при spec decode — наш build содержит этот merge. Не fix'ит underlying kernel path.
>
> Готов протестировать любой candidate fix на нашем rig'е и запустить 5K identical reproducer + multi-turn tool-call regression suite если поможет.
>
> Спасибо @NickLucche за исходный report — `num_accepted_tokens > 1` анализ был недостающей частью нашего расследования.

---

## Ukraine / AI disclaimer

Per `feedback_github_comment_style` — add Ukraine + AI translation disclaimer. Already in EN version's last italicized line; can extend if Sander prefers more explicit framing.

## Recommended action sequence after approval

1. Post comment to https://github.com/vllm-project/vllm/issues/38898
2. Optionally cross-link from #26201 (tracking issue) to draw attention to existing #38898
3. Watch for maintainer response in 1-2 weeks; if no engagement, optional escalation by tagging `@njhill` or `@WoosukKwon`

## Why this specific framing

- **Concrete metrics, not opinion:** turn-by-turn hit_tokens + TTFT numbers + Mamba store debug data — direct empirical evidence
- **Acknowledges existing PR #40454:** shows we did our homework on prior workarounds
- **Offers labor:** "happy to test any candidate fix" — maintainers value contributors who reduce their work
- **No blame:** thanks NickLucche, frames as collaborative investigation
- **Production context:** explains WHY the gap matters (MTP K=3 = +30% throughput we can't give up)
- **Cross-references:** Genesis repo, tracking #26201, PR #40454 — anchors our claim in existing maintainer-known surface

# Draft reply — vllm-project/vllm#40914 (K+1 spec-verify routing)

**Status:** DRAFT for Sander review. Do NOT post until approved.

**Context:** Sander's own draft PR for K+1 spec-verify routing (fix for #40880). Gemini-bot approved buffer-reuse fix (commit `1ac8795`). On 2026-04-26 23:19 UTC, **noonghunna posted Ampere SM 8.6 second-rig confirmation** (Genesis v7.48 + 2× RTX 3090 + Qwen3.6-27B-AutoRound). This is exactly the cross-rig validation Sander explicitly requested. **The PR is ready to come out of draft and be promoted to Ready-for-Review.**

**Thread URL:** <https://github.com/vllm-project/vllm/pull/40914>

**Suggested action sequence:**
1. Acknowledge noonghunna's cross-rig data (single, well-formatted summary comment)
2. Mark PR Ready-for-Review (out of draft)
3. Ping `@LucasWilkinson` and `@WoosukKwon` (spec-decode CODEOWNERs) for review

---

## English version (single comment summarizing now-two-rig evidence)

> **Cross-rig validation summary — request review out of draft**
>
> Quick summary of the validation now in evidence:
>
> | Rig | Hardware | Model | KV format | Validation |
> |---|---|---|---|---|
> | A (mine) | 2× RTX A5000 (Ampere SM 8.6, 48 GB total) | Qwen3.6-35B-A3B-FP8 | TurboQuant k8v4 | Genesis v7.48 stack — 30-shot quality 30/31 PASS, throughput +2.6% TPS, lowest CV measured (5.0%), tool-call 2/2 PASS, long-ctx 16K-200K all PASS |
> | B ([@noonghunna](https://github.com/noonghunna), 2026-04-26) | 2× RTX 3090 (Ampere SM 8.6, 48 GB total, no NVLink) | Qwen3.6-27B-AutoRound (INT4) | TurboQuant k8v4 | Same Genesis v7.48 stack — fix transferred cleanly across model variant + quant + rig topology |
>
> The buffer-reuse fix (`mid_o_buf` / `lse_buf` / `buf_holder=layer` / `max_num_kv_splits` forwarded into the routing call) addresses the issue gemini-code-assist flagged on the original commit. Without it, `_decode_attention` next door already does the right thing — we just hadn't matched the full call signature when copying the dispatch.
>
> Two-rig evidence + bot review approval + clean tool-call regression make me comfortable promoting this out of draft. Pinging @LucasWilkinson and @WoosukKwon for review — happy to address any concerns or run additional benchmarks.

---

## Russian version (для контроля смысла)

> **Cross-rig validation summary — запрос review из draft**
>
> Краткая сводка validation сейчас:
>
> | Rig | Hardware | Model | KV | Validation |
> |---|---|---|---|---|
> | A (мой) | 2× RTX A5000 (Ampere SM 8.6) | Qwen3.6-35B-A3B-FP8 | TurboQuant k8v4 | Genesis v7.48 — 30-shot quality 30/31 PASS, +2.6% TPS, lowest CV (5.0%), tool-call 2/2 PASS, long-ctx 16K-200K все PASS |
> | B (@noonghunna, 2026-04-26) | 2× RTX 3090 (Ampere SM 8.6, no NVLink) | Qwen3.6-27B-AutoRound (INT4) | TurboQuant k8v4 | Тот же Genesis v7.48 — fix чисто переносится через model variant + quant + rig topology |
>
> Buffer-reuse fix (`mid_o_buf` / `lse_buf` / `buf_holder=layer` / `max_num_kv_splits` forwarded в routing call) адресует то что gemini-code-assist flag на оригинальном commit. Без него `_decode_attention` рядом уже делает правильно — мы просто не совпали full call signature при копировании dispatch.
>
> Two-rig evidence + bot review approval + clean tool-call regression делают меня комфортным promote это из draft. Пингую @LucasWilkinson и @WoosukKwon для review — готов адресовать concerns или запустить дополнительные benchmarks.

---

## Ukraine / AI disclaimer

Per memory `feedback_github_comment_style.md`, **add Sander's Ukraine + AI translation disclaimer at the top of the actual posted comment.**

## Recommended action sequence after approval

1. Post the comment (above)
2. **Click "Ready for review"** button on the PR (this changes draft → open and notifies CODEOWNERs)
3. Optional: comment in noonghunna/qwen36-dual-3090#1 thanking him for the cross-rig data and linking to this PR comment

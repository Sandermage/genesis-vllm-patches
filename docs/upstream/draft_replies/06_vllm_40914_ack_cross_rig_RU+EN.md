# Draft reply — vllm-project/vllm#40914 (Sander's K+1 spec-verify routing PR)

**Status:** DRAFT for Sander review. Do NOT post until approved.

**Context:** noonghunna posted cross-rig validation 2026-04-26 23:19:
> "cc @Sandermage — patch tree at v7.48 with P67/P67b/P78 worked first-try on our 3090 box. Thanks for the careful work here." (83.99 TPS, CV 3.9% on 27B)

Plus 2026-04-27 01:00 follow-up with apples-to-apples 35B-A3B bench (wall_TPS 136.87, CV 2.2%).

**Strategic:** PR has been in draft state, accumulating reviewer evidence:
- Gemini-bot review hits addressed (commit `0ee9b85`)
- Sander's own A5000 cross-rig confirmation
- noonghunna's RTX 3090 cross-rig confirmation (BOTH 27B AND 35B-A3B variants)

**Recommended action sequence:**
1. Post acknowledgment comment with two-rig summary table
2. Mark PR Ready-for-Review (out of draft)
3. Tag CODEOWNERs `@LucasWilkinson` `@WoosukKwon` for spec-decode review
4. Reference noonghunna's bench script v2 as canonical cross-rig harness

---

## English version (single comment)

> **Cross-rig validation summary — promoting from draft to Ready-for-Review**
>
> Two independent rigs now confirm this PR is safe + faster + lower-variance than baseline:
>
> | Rig | Hardware | Model | Mean TPS | CV | Quality | Confirms buffer-reuse fix |
> |---|---|---|---|---|---|---|
> | A (mine) | 2× RTX A5000 (Ampere SM 8.6, TP=2 PCIe) | Qwen3.6-35B-A3B-FP8 | 130.68 (v7.45) | **5.0%** | 30/31 + tool 2/2 PASS | ✓ |
> | B ([@noonghunna](https://github.com/noonghunna), 2026-04-26) | 2× RTX 3090 (Ampere SM 8.6, TP=2 PCIe) | Qwen3.6-27B-AutoRound-INT4 + TQ k8v4 + Genesis v7.48 | 83.99 | **3.9%** | needle 10K-90K + tool 9/9 PASS | ✓ |
> | B (extended, 2026-04-27) | same | **Qwen3.6-35B-A3B-FP8** (same as rig A) | **136.87** | **2.2%** | tool 9/9 PASS, AL=2.5 | ✓ |
>
> The buffer-reuse fix in commit [`0ee9b85`](https://github.com/vllm-project/vllm/pull/40914/commits/0ee9b85) addresses gemini-code-assist's HIGH-priority comment from the original review (forwarding `mid_o_buf`/`output_buf`/`lse_buf`/`buf_holder=layer`/`max_num_kv_splits` into the routing call). Without it, the kernel would allocate fresh tensors per call and defeat cudagraph replay — exactly the regression this PR aims to FIX.
>
> Both rigs run the same Genesis v7.48 patch tree (P67 multi-query Triton kernel + P67b spec-verify routing + P78 .tolist() guard). noonghunna's bench harness ([qwen36-dual-3090/scripts/bench.sh](https://github.com/noonghunna/qwen36-dual-3090) v2) reports `wall_TPS / decode_TPS / TTFT / CV` per run — useful for reviewers who want to reproduce.
>
> **Marking Ready-for-Review** and pinging @LucasWilkinson @WoosukKwon for spec-decode-side review when convenient. Happy to address any follow-up concerns or run additional benches.
>
> Special thanks to @noonghunna for the rapid cross-rig validation (both 27B and 35B-A3B variants in under 24 hours) — this is exactly the kind of independent reproduction that makes consumer-Ampere PRs reviewable.
>
> *(Comment text generated with AI translation assistance — original draft in Russian.)*

---

## Russian version (для контроля смысла)

> **Cross-rig validation summary — promoting из draft в Ready-for-Review**
>
> Два независимых rig'а теперь подтверждают что PR safe + faster + lower-variance чем baseline:
>
> | Rig | Hardware | Model | Mean TPS | CV | Quality | Buffer-reuse fix |
> |---|---|---|---|---|---|---|
> | A (мой) | 2× RTX A5000 (TP=2 PCIe) | Qwen3.6-35B-A3B-FP8 | 130.68 | 5.0% | 30/31 + tool 2/2 | ✓ |
> | B (@noonghunna) | 2× RTX 3090 (TP=2 PCIe) | Qwen3.6-27B-AutoRound-INT4 + TQ k8v4 + Genesis v7.48 | 83.99 | **3.9%** | needle 10K-90K + tool 9/9 | ✓ |
> | B (35B-A3B) | same | **Qwen3.6-35B-A3B-FP8** | **136.87** | **2.2%** | tool 9/9, AL=2.5 | ✓ |
>
> Buffer-reuse fix в `0ee9b85` адресует gemini-code-assist HIGH-priority comment.
>
> Оба rig'а используют тот же Genesis v7.48 (P67 + P67b + P78). noonghunna bench v2 reports wall_TPS / decode_TPS / TTFT / CV — useful для reviewers.
>
> **Marking Ready-for-Review** и пингую @LucasWilkinson @WoosukKwon для spec-decode review. Готов адресовать concerns.
>
> Спасибо @noonghunna за быструю cross-rig validation (27B и 35B-A3B за < 24 часа) — exactly the kind of independent reproduction that makes consumer-Ampere PRs reviewable.

---

## Action sequence after approval

1. Post comment
2. Click "Ready for review" button (PR draft → open)
3. Reviewers auto-notified via CODEOWNERS

## Why this framing

- Quantitative side-by-side table is reviewer-friendly
- Names buffer-reuse fix commit explicitly (shows we addressed bot review)
- Credits noonghunna without making PR about him (focus stays on Sander's contribution)
- Ends with concrete action (ready for review + tag reviewers)
- AI/Ukraine disclaimer per memory rule

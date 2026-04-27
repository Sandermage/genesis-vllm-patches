# Draft reply v2 — noonghunna/qwen36-dual-3090#1 (35B-A3B cross-rig data)

**Status:** DRAFT for Sander review. Do NOT post until approved.

**Context:** noonghunna posted 2026-04-27 01:00 with apples-to-apples 35B-A3B bench data on 2× 3090 with our P67/P67b/P78:
> wall_TPS = 136.87, CV = 2.2%, decode_TPS = 139.15, TTFT = 119ms, AL = 2.5, ~30% draft accept

**Thread URL:** https://github.com/noonghunna/qwen36-dual-3090/issues/1

**Suggested angle:** acknowledge cross-rig data, compare against our A5000 numbers, note CV importance, suggest next coordination point (P82 cross-rig or upcoming work).

---

## English version

> @noonghunna huge thanks for running the apples-to-apples 35B-A3B bench — that's exactly the second data point I needed for v7.48 P67 validation.
>
> **Side-by-side (your 2× 3090 vs my 2× A5000, same Genesis v7.48 + P67/P67b/P78, same 35B-A3B-FP8):**
>
> | Metric | Your 2× 3090 PCIe | My 2× A5000 (24GB each) |
> |---|---|---|
> | wall_TPS | 136.87 | ~167 |
> | CV | 2.2% | ~5.0% (v7.45) |
> | decode_TPS | 139.15 | ~165 |
> | TTFT | 119ms | ~140ms |
> | AL | 2.5 | ~3.0 |
>
> The gap is consistent with the per-card bandwidth difference (3090 PCIe vs A5000 server-grade memory). What's striking is your **CV is BETTER** — 2.2% vs my 5.0%. Suggests the 3090 stack has less interference noise (no NVLink + simpler PCIe topology may help here). Worth flagging to anyone who thinks "more cards always means more variance".
>
> **One follow-up ask, if interesting:** I just shipped **P82** (port of SGLang's `threshold_single` OR-clause acceptance from `speculative_sampling.cuh`, opt-in via `GENESIS_ENABLE_P82=1` + `GENESIS_P82_THRESHOLD_SINGLE=0.3`). On my prod sweep it gave +12% mean TPS on 128-2048 tok generation with quality 32/33 and 0 artifact flags. **Would you have time to add `-e GENESIS_ENABLE_P82=1 -e GENESIS_P82_THRESHOLD_SINGLE=0.3` to your `p67-bench-35b-a3b.yml`** and re-run? If P82 transfers cleanly to 3090, that's the strongest argument for promoting it to default-on in Genesis.
>
> Also — your bench.sh v2 with CV is exactly the metric harness vllm#40914 reviewers are going to want. Mind if I cite your bench script in my PR comment as the canonical cross-rig measurement tool?
>
> Thanks again for the sustained collaboration; it's making this whole stack more credible than either of us could do solo.
>
> *(English text generated with AI translation assistance — original draft in Russian.)*

---

## Russian version (для контроля)

> @noonghunna огромное спасибо за apples-to-apples 35B-A3B bench — это второй data point которого мне не хватало для v7.48 P67 validation.
>
> **Side-by-side (твой 2× 3090 vs мой 2× A5000, тот же Genesis v7.48 + P67/P67b/P78, тот же 35B-A3B-FP8):**
>
> | Метрика | 2× 3090 PCIe | 2× A5000 |
> |---|---|---|
> | wall_TPS | 136.87 | ~167 |
> | CV | **2.2%** | ~5.0% |
> | decode_TPS | 139.15 | ~165 |
> | TTFT | 119ms | ~140ms |
> | AL | 2.5 | ~3.0 |
>
> Gap consistent с per-card bandwidth difference. **Twой CV ЛУЧШЕ** — 2.2% vs мои 5.0%. Suggests 3090 stack менее шумный (no NVLink + simpler PCIe). Worth flagging для тех кто думает "больше карт = больше variance".
>
> **One follow-up ask:** только что зашипил **P82** (port SGLang `threshold_single` OR-clause). На моём prod sweep дал +12% mean TPS на 128-2048 tok с quality 32/33. **Можешь добавить `-e GENESIS_ENABLE_P82=1 -e GENESIS_P82_THRESHOLD_SINGLE=0.3` в `p67-bench-35b-a3b.yml`** и прогнать? Если P82 cleanly переносится на 3090 — это сильнейший аргумент чтобы продвинуть default-on в Genesis.
>
> Также — твой bench.sh v2 с CV это именно та метрика которая нужна reviewer'ам vllm#40914. Не возражаешь если процитирую твой bench script в моём PR comment как canonical cross-rig measurement tool?
>
> Спасибо за sustained collaboration — это делает весь stack более credible чем каждый по одиночке.

---

## Ukraine / AI disclaimer

Add per `feedback_github_comment_style`. Already prefixed in EN version.

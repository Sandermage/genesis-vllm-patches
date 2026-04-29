# Draft reply — noonghunna/qwen36-dual-3090#1 (cross-rig collab)

**Status:** DRAFT for Sander review. Do NOT post until approved.

**Context:** Sander opened this issue 2026-04-26 asking: (1) attribution language, (2) P67 kernel data exchange, (3) future collaboration cadence. noonghunna replied to all three, then posted a follow-up bench update on Qwen3.6-35B-A3B-AutoRound (his INT4 quant) vs Sander's FP8 — "same model class as yours, only quant differs."

**Thread URL:** <https://github.com/noonghunna/qwen36-dual-3090/issues/1>

**Suggested reply angle:** acknowledge apples-to-apples bench, share parity/divergence read, confirm attribution wording is good as proposed, propose the next concrete data exchange (P67 kernel + P82 cross-rig validation).

---

## English version

> Hi @noonghunna, thanks for that follow-up bench — having Qwen3.6-35B-A3B-AutoRound on your dual-3090 against my FP8 on dual-A5000 gives us a real apples-to-apples cross-rig point. The gap I see vs my numbers (need to do a proper side-by-side, but preliminary read: your AutoRound INT4 + DFlash N=5 = ~136 wall_TPS; my FP8 + MTP K=3 + P82 t=0.3 = ~187 @128 / ~154 @2048) looks consistent with what I'd expect — INT4 has slightly more overhead in the dequant path, MTP K=3 + P82 OR-clause is a different acceleration model than DFlash. **Net: this confirms the Genesis stack scales cleanly to a different quant + different draft method on a 3090 SM 8.6 rig.**
>
> Attribution language as you proposed is great — please go ahead and lift it as is.
>
> **One ask, if you have time** — I just shipped **P82** (port of SGLang's `threshold_single` OR-clause from `speculative_sampling.cuh`, opt-in via `GENESIS_ENABLE_P82=1` + `GENESIS_P82_THRESHOLD_SINGLE=0.3`). On my prod sweep it gave +12% mean TPS on 128-2048 tok generation with quality 32/33 and 0 artifact flags. Would you have time to run a single bench on your `turbo` variant (`turboquant_k8v4` + Genesis v7.48) with `GENESIS_ENABLE_P82=1 GENESIS_P82_THRESHOLD_SINGLE=0.3` added to the env? **A second-rig confirmation is the strongest argument for promoting this to default-on in Genesis.**
>
> Also — I noticed `3dluvr` posting `--mamba-cache-mode all --mamba-block-size 8` + `VLLM_ENABLE_CUDAGRAPH_GC=1` in your single-3090 thread. Did you A/B those on your dual rig? Curious if they stack with our patches or contradict.

---

## Russian version (для контроля смысла)

> Привет @noonghunna, спасибо за follow-up bench — Qwen3.6-35B-A3B-AutoRound на твоём dual-3090 vs мой FP8 на dual-A5000 даёт реальный apples-to-apples cross-rig point. Разрыв с моими цифрами (надо сделать нормальное side-by-side, но preliminary: твой AutoRound INT4 + DFlash N=5 = ~136 wall_TPS; мой FP8 + MTP K=3 + P82 t=0.3 = ~187 @128 / ~154 @2048) выглядит логично — INT4 имеет чуть больше overhead в dequant, MTP K=3 + P82 OR-clause это другая модель акселерации vs DFlash. **Нет: это подтверждает что Genesis stack нормально работает на другом quant + другом draft method на 3090 SM 8.6.**
>
> Attribution language как ты предложил — отлично, бери как есть.
>
> **Один запрос если будет время** — я только зарелизил **P82** (порт SGLang `threshold_single` OR-clause из `speculative_sampling.cuh`, opt-in через `GENESIS_ENABLE_P82=1` + `GENESIS_P82_THRESHOLD_SINGLE=0.3`). На prod sweep дал +12% mean TPS на 128-2048 tok с quality 32/33 и 0 artifact flags. Можешь прогнать один bench на своём `turbo` variant (`turboquant_k8v4` + Genesis v7.48) с добавлением `GENESIS_ENABLE_P82=1 GENESIS_P82_THRESHOLD_SINGLE=0.3` в env? **Second-rig confirmation — самый сильный аргумент чтобы продвинуть это в default-on в Genesis.**
>
> Также — заметил что `3dluvr` пишет про `--mamba-cache-mode all --mamba-block-size 8` + `VLLM_ENABLE_CUDAGRAPH_GC=1` у тебя в single-3090 thread. Ты A/B тестировал на dual-rig? Интересно стакаются ли с нашими патчами или контраст.

---

## Ukraine / AI disclaimer

Per memory `feedback_github_comment_style.md`, **add Sander's Ukraine + AI translation disclaimer at the top of the actual posted comment.** I have NOT prepended it here so you can edit the body first; add the standard prefix block before posting.

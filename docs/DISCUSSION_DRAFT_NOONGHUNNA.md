# Discussion Draft — Reaching out to @noonghunna

**Status**: DRAFT. Do NOT post until Sander reviews + approves both EN and RU versions per memory rules.

**Where to post**: GitHub Discussion (NOT Issue, NOT PR) on https://github.com/noonghunna/qwen36-27b-single-3090 OR https://github.com/noonghunna/qwen36-dual-3090

**Purpose**: friendly cross-rig collaboration. Acknowledge their work, request permission for the attribution we already wrote into our P78 docstring, offer back our P67 kernel design data.

---

## English version (primary)

> Hi @noonghunna,
>
> First — thank you for the careful documentation in `qwen36-27b-single-3090` and `qwen36-dual-3090`. The decision-tree READMEs and the 9-probe ladder you posted on #40880/#40807/#40831 saved us hours of investigation in our parallel debugging on 2× RTX A5000.
>
> A few words about us: we run a Genesis-themed runtime patcher for vLLM (`Sandermage/genesis-vllm-patches`) — same model class (Qwen3.6-35B-A3B-FP8) on Ampere SM 8.6, just with A5000s instead of 3090s. We've been independently developing patches for the same bug class you're hunting; you'll see your name in our `CREDITS.md` for cross-rig confirmation work, and you've graciously cited our P65 root-cause in your dual README turbo section — thank you for that.
>
> Two specific things I wanted to raise:
>
> **1) Asking permission for attribution.** We've adapted your `patch_tolist_cudagraph.py` (Apache-2.0) as our patch P78 — surgical capture-guard for `TurboQuant._prefill_attention`. It complements our existing P22/P26/P44 prealloc patches (which sidestep the `.tolist()` path on steady-state, but warmup capture can transit it before prealloc kicks in — your guard is the safety-net). Our patch file carries a docstring CREDIT block pointing to your repo, your authorship, and the Apache-2.0 license. We'd like to confirm you're OK with this before we publish. If you'd prefer different attribution wording (or any specific co-author line, separate file naming, etc.) — please tell us, we'll match exactly.
>
> **2) Offering back data on our P67 multi-query kernel.** When you reported #40880, we initially shipped a workaround (P65: cudagraph downgrade for spec-decode). Then we built P67 — a Triton kernel that handles the K+1 spec-verify path properly under FULL_AND_PIECEWISE cudagraph capture, eliminating the P65 cost. On our 2× A5000 + Qwen3.6-35B-A3B-FP8 + MTP=3, P67 measures +32% TPS over the v7.13 baseline (75.6 vs 57.2 tok/s). The Triton kernel design and benchmark methodology are documented; if it's useful as a comparison data point against your DFlash N=5 implementation in `qwen36-dual-3090`, I'd be happy to share more. Especially curious if DFlash N=5 vs ngram strict + MTP is a meaningful comparison axis on your rig.
>
> If a cross-link in our respective READMEs makes sense — happy to do that too.
>
> A few notes for context: I'm based in Odessa, Ukraine, and English is not my first language. Some of this message went through machine translation polishing, please excuse any awkward phrasing.
>
> Thank you again for the careful documentation work.
>
> — Sander (@Sandermage)

---

## Russian version (review side)

> Привет @noonghunna,
>
> Прежде всего — спасибо за подробную документацию в `qwen36-27b-single-3090` и `qwen36-dual-3090`. Decision-tree README и 9-probe ladder который ты опубликовал в #40880/#40807/#40831 сэкономили нам часы исследования при параллельном дебаге на 2× RTX A5000.
>
> Пара слов о нас: мы поддерживаем тематический runtime patcher для vLLM (`Sandermage/genesis-vllm-patches`) — тот же класс модели (Qwen3.6-35B-A3B-FP8) на Ampere SM 8.6, только A5000 вместо 3090. Мы независимо разрабатываем патчи для того же класса багов что и ты; твоё имя есть в нашем `CREDITS.md` за cross-rig confirmation, и ты любезно процитировал наш P65 root-cause в turbo секции твоего dual README — спасибо за это.
>
> Два конкретных момента которые я хотел поднять:
>
> **1) Просьба разрешить атрибуцию.** Мы адаптировали твой `patch_tolist_cudagraph.py` (Apache-2.0) как наш патч P78 — surgical capture-guard для `TurboQuant._prefill_attention`. Он дополняет наши существующие P22/P26/P44 prealloc патчи (которые обходят `.tolist()` на steady-state, но warmup capture может пройти через него до того как prealloc сработает — твой guard это safety-net). Наш файл патча содержит CREDIT блок указывающий на твой репо, авторство, и Apache-2.0 лицензию. Хотели бы подтвердить что ты не против перед публикацией. Если предпочитаешь другую формулировку атрибуции (или конкретный co-author line, отдельное название файла, и т.д.) — скажи, сделаем точно как ты хочешь.
>
> **2) Готовы поделиться данными по нашему P67 multi-query kernel.** Когда ты репортил #40880, мы сначала выпустили workaround (P65: cudagraph downgrade для spec-decode). Потом сделали P67 — Triton kernel который обрабатывает K+1 spec-verify путь правильно под FULL_AND_PIECEWISE cudagraph capture, устраняя стоимость P65. На нашем 2× A5000 + Qwen3.6-35B-A3B-FP8 + MTP=3, P67 даёт +32% TPS относительно v7.13 baseline (75.6 vs 57.2 tok/s). Дизайн Triton kernel и методология бенчмарка задокументированы; если это полезно как точка сравнения с твоей DFlash N=5 имплементацией в `qwen36-dual-3090` — с удовольствием поделюсь подробностями. Особенно интересно сравнение DFlash N=5 vs ngram strict + MTP на твоём железе.
>
> Если cross-link в наших соответствующих README имеет смысл — тоже с удовольствием.
>
> Несколько пометок для контекста: я из Одессы, Украина, английский не родной. Часть текста прошла через машинный перевод для полировки, заранее извини за любые неловкие формулировки.
>
> Ещё раз спасибо за тщательную документационную работу.
>
> — Sander (@Sandermage)

---

## Pre-post checklist (per memory rules)

- [ ] Sander reviews EN version
- [ ] Sander reviews RU version (cross-check against EN)
- [ ] Verify all GitHub links resolve (per `feedback_verify_links_before_posting.md`)
  - [ ] github.com/Sandermage/genesis-vllm-patches — public OR private with note
  - [ ] All vllm#XXXXX issue/PR links exist
  - [ ] `CREDITS.md` link works once repo is public
- [ ] Verify P67 perf data is publishable (75.6 vs 57.2 tok/s — already in our memory)
- [ ] Confirm P78 file already has CREDIT block (it does — see `wiring/patch_78_tolist_capture_guard.py`)
- [ ] No AI co-author / Claude mentions per `feedback_no_ai_credit_in_public.md`
- [ ] Sander says explicit "ok post" before publishing per `feedback_no_push_without_explicit_approval.md`

## Tone/style notes (per `feedback_github_comment_style.md`)

- Ukraine/AI-translation disclaimer at end ✓ (both versions)
- Warm greeting ✓
- Conversational, not corporate ✓
- Specific numbers + cross-references ✓
- No demands, only requests + offers ✓
- Acknowledges their work first, ours second ✓

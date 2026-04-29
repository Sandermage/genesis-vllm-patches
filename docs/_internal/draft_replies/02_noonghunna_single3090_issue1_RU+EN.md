# Draft reply — noonghunna/qwen36-27b-single-3090#1 (4 fresh @Sandermage mentions)

**Status:** DRAFT for Sander review. Do NOT post until approved.

**Context:** Sander posted ONE message 2026-04-26 23:39 explaining P26 vs Site B distinction. After that:

1. `3dluvr` @Sandermage 2026-04-27 00:08 — shared dual 2× 3090 startup args (incl. `--mamba-cache-mode all --mamba-block-size 8` + `VLLM_ENABLE_CUDAGRAPH_GC=1`)
2. `3dluvr` @Sandermage 00:17 — **CRITICAL:** "Site B anchor not detected on `dev209+g60cd878a3` — had to manually patch"
3. `noonghunna` @Sandermage 00:39 — lifting Sander's explanation into docs verbatim, noting `3dluvr`'s anchor regression
4. `3dluvr` later — unexpectedly *better* numbers (~85+ tok/s) on plain nightly with newer SHA `dev215+g32e45636e`, asking community to interpret

**Thread URL:** <https://github.com/noonghunna/qwen36-27b-single-3090/issues/1>

**Suggested reply angle:** thank for cross-rig data, confirm doc-lift attribution OK, **acknowledge the Site B anchor drift is exactly the documented "container R/W layer trap" + commit to bumping the patcher's anchor regex**, address the surprising plain-nightly performance.

---

## English version

> Hi @noonghunna @3dluvr, thanks for both data points and sorry for the slow reply.
>
> **On the Site B anchor drift** — @3dluvr that's exactly the failure mode I have documented internally as the "container R/W-layer trap" (anchor regex matched a specific upstream phrasing; when upstream rewrote the surrounding line on `dev209+g60cd878a3`, our exact-match patcher silently bailed instead of soft-warning). **This is on me to fix in Genesis.** I'll post a follow-up here when I push a more resilient anchor matcher (probably regex-based with multiple fallback patterns, plus a verbose mode that prints which anchors were tried and which matched). If you can paste the exact apply_all log output where Site B was skipped (`[Genesis] skipped: P56 ... reason=...`), it'd help me confirm whether the anchor moved or whether something else regressed.
>
> **On the doc-lift** — please go ahead and lift the explanation verbatim, attribution as Sandermage / Sander Barzov is fine.
>
> **On the surprising 85+ tok/s on plain nightly with `dev215+g32e45636e`** — that's interesting. Two things happened in vLLM main between `dev205` (my pin) and `dev215`:
>   - **#40941 (TurboQuant dequant buffer share + float16_copy elimination) MERGED 2026-04-27 as `2cc008e7`** — saves 57GB at 1M ctx, removes a redundant copy in decode. That's a big chunk of the gain you're seeing without Genesis patches.
>   - **#40915 (jow- state-machine Qwen3 XML parser)** is in flight — could be reducing parser-layer overhead on tool calls.
>
> So the right interpretation may be "vLLM main caught up on some of what Genesis patches independently" — which is the **best possible outcome** for this collaboration, since it means we can drop those Genesis patches once the upstream merge propagates. I'll do an A/B on my rig comparing Genesis v7.53 + `dev205` vs plain `dev215+g32e45636e` to confirm — will report back.
>
> **On `--mamba-cache-mode all --mamba-block-size 8` + `VLLM_ENABLE_CUDAGRAPH_GC=1`** — @3dluvr can you confirm what acceptance rate you're seeing with those flags? You mentioned 98.6% MTP acceptance somewhere; if that's stable across a 30-request burst, it's a strong signal this combo is doing real work for the GDN+MTP path. I want to A/B that against my Genesis stack and see if the gains are additive or substitutive.

---

## Russian version (для контроля смысла)

> Привет @noonghunna @3dluvr, спасибо за оба data point и сорри за медленный ответ.
>
> **По Site B anchor drift** — @3dluvr, это именно тот failure mode который я документировал внутренне как "container R/W-layer trap" (anchor regex совпадал с конкретной upstream формулировкой; когда upstream переписал окружающую строку на `dev209+g60cd878a3`, наш exact-match patcher молча отвалился вместо soft-warning). **Это на мне исправить в Genesis.** Я напишу follow-up здесь когда запушу более устойчивый anchor matcher (вероятно regex с multiple fallback patterns + verbose mode который печатает какие anchors пробовал и какие совпали). Если можешь скинуть exact apply_all log где Site B был skipped (`[Genesis] skipped: P56 ... reason=...`), это поможет подтвердить anchor двинулся или другое регрессировало.
>
> **По doc-lift** — давай, бери объяснение verbatim, attribution Sandermage / Sander Barzov ок.
>
> **По неожиданному 85+ tok/s на plain nightly с `dev215+g32e45636e`** — это интересно. Между `dev205` (мой pin) и `dev215` в vLLM main произошли две вещи:
>   - **#40941 (TurboQuant dequant buffer share + float16_copy elimination) ЗАМЁРЖЕНА 2026-04-27 как `2cc008e7`** — экономит 57GB на 1M ctx, убирает redundant copy в decode. Это большой кусок прироста который ты видишь без Genesis патчей.
>   - **#40915 (jow- state-machine Qwen3 XML parser)** в полёте — может снижать parser-layer overhead на tool calls.
>
> Так что правильная интерпретация может быть "vLLM main догнал часть того что Genesis патчи делали независимо" — что **лучший возможный исход** для этой коллаборации, потому что значит мы можем дропнуть эти Genesis патчи когда upstream merge propagated. Я сделаю A/B на своём rig сравнивая Genesis v7.53 + `dev205` vs plain `dev215+g32e45636e` чтобы подтвердить — отпишу.
>
> **По `--mamba-cache-mode all --mamba-block-size 8` + `VLLM_ENABLE_CUDAGRAPH_GC=1`** — @3dluvr можешь подтвердить какая acceptance rate с этими flags? Ты упомянул 98.6% MTP acceptance где-то; если это стабильно на 30-request burst, это сильный сигнал что combo делает real work для GDN+MTP path. Хочу A/B vs мой Genesis stack и посмотреть стакаются ли gains.

---

## Ukraine / AI disclaimer

Per memory `feedback_github_comment_style.md`, **add Sander's Ukraine + AI translation disclaimer at the top of the actual posted comment.**

## Action items if Sander approves

If we promise to fix the anchor drift, that creates a follow-up implementation task:
- Bump TextPatcher to support regex fallbacks (currently exact-string only)
- Add `--verbose-anchor-debug` mode to apply_all
- Add a "drift detector" that emits a warning rather than silent skip when anchor moved >N chars

Let me know if you want me to do that as the next sprint after the post.

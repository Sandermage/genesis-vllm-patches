# Отчёт по двум PR-свипам (50 + 62): что взято и как сделано

> Source-verified аудит (workflow `pr-sweep-implementation-audit`, 15 агентов,
> 2026-06-14). Каждая запись о vendored-патче подтверждена чтением исходника
> патча (apply_module → файл), каждая non-vendored — вердиктом из роадмапов
> `2026-06-11-pr-sweep-50-roadmap.md` / `2026-06-11-pr-sweep-62-batch2-roadmap.md`.

## 1. Сводка

**Объём изучения.** Два свипа — batch-1 («50 PR») и batch-2 («62 PR»),
суммарно ~112 PR изучено (часть пересекается → 115 уникальных номеров в наборе).

**Итог по диспозиции:**

| Категория | Кол-во PR |
|---|---|
| **Vendored** (стали Genesis-патчами) | **42** |
| Не взято — duplicate-of-existing | 11 |
| Не взято — watchlist / retire-trigger | 17 |
| Не взято — research-or-method-only | 18 |
| Не взято — not-applicable-arch (мертво на Ampere) | 7 |
| Не взято — fix-to-genesis-code | 8 |
| Не взято — pending-wave-2/3 | 18 |

**КРИТИЧЕСКИЙ КАВЕАТ — spec-only-boot.** Все вновь созданные в этих свипах
патчи (`PN370–PN392`, `G4_79/80/81`, `P88`, `P89`) — **SPEC-ONLY**: нет
legacy-хука `@register_patch` в `_per_patch_dispatch.py`, присутствуют в
`KNOWN_SPEC_ONLY_PATCHES` (`shadow.py`), применяются только через
registry-driven диспетчер. Из-за этого они были **инертны при загрузке** —
закоммичены, но не активны — **вплоть до spec-only-supplement fix от
14 июня 2026 (commit `1a84f632`)**. Честный статус каждого:
**«vendored & committed, boot-active только после фикса apply-loop от 14 июня»**.

Исключения (boot-active штатно, реальный хук): `PN367` (#45076→#44745),
`PN370` (#45100), `PN372` (#45005), `PN377` (#44563).

---

## 2. Взятые и реализованные

Легенда «план»: `yes` — реализация совпадает с roadmap-строкой; `partial` —
реализация верна, но направление плана частично расходится; `unknown` —
выделенной planned-adaptation строки нет (завендорен в более раннюю сессию).

### 2a. Предсуществующие патчи (НЕ созданы в этих свипах; roadmap — контекст/синергия)

| PR# | Патч | Как сделано | default | план |
|---|---|---|---|---|
| #40361 | **P87** | TextPatcher (5 sub) на `marlin.py`: round_up output до 64, `_maybe_pad_n()` zero-pad qweight/scales/qzeros/bias, slice в apply_weights. Dual-anchor. | off | yes |
| #40768 | **P58** | TextPatcher (3 файла): `num_pending_async_spec_placeholders` счётчик вместо shared `[-1]` списка. | off | yes |
| #40925 | **P81** | TextPatch на `fp8_utils.py`: `M<=8` → `BLOCK_SIZE_M=16`, `num_stages=3`. | off | yes |
| #42006 | **G4_T1** | Marker-only заглушка (`apply()`→skipped); реальный фикс — overlay-файл через bind-mount, не диспетчер. spec_only_boot_risk=FALSE. | off | partial |
| #42301 | **PN204** | TextPatch `gdn_linear_attn.py`: серийный in_proj → `maybe_execute_in_parallel` (aux stream). conflicts_with P7/PN365. | off | yes |
| #42551 | **PN118** | TextPatcher (4 anchor): `try_get_simultaneous()` (None→torch.empty fallback) + `reserve()` pre-alloc до lock. **default_on=True**. | **on** | yes |
| #42603 | **P108** | TextPatch `llm_base_proposer.py`: backend-gated sync, авто-вкл только FlashInfer. | **on** | partial |
| #42637 | **G4_60E** | Runtime monkey-patch `kv_cache_utils`: reconciled-ladder в `unify_kv_cache_spec_page_size`. **spec-only**. | off | yes |
| #42722 | **PN133** | TextPatch `scheduler.py`: `if scheduled_spec_token_ids:` + `max(len-1,0)` (Prometheus-краш). | off | yes |
| #42746 | **PN365** | TextPatch (2 файла): один MergedColumn GEMM вместо двух. conflicts_with PN204. | off | unknown |
| #43650 | **PN346** | TextPatch `single_type_kv_cache_manager.py`: 6-LOC guard — поиск не трогает финальный state-блок. **default_on (opt-out only)**. | **on** | yes |
| #44053 | **PN353A** | TextPatch `turboquant_attn.py`: `_reserve_workspace()` pre-alloc до CG lock. | off | yes |
| #44113 | **PN347** | TextPatch `scaled_mm/marlin.py`: shape-guard → `is_contiguous()` перед `.t()`. | off | yes |
| #44644 | **PN348** | TextPatch `qwen3_5_mtp.py` (3 sub): MTP-backbone reuse target embed+lm_head. ~1 GiB/rank peak + 1-3s boot. | off | yes |
| #44778 | **PN55** | PN55**v2** рекурсивный walker. #44778 — функц. ДУБЛИКАТ, взяли только hygiene (related_prs, warn-skip, test-технику), НЕ re-vendor. | off | yes |

### 2b. Wave-1 свипа-50 — commit `f84bf7b4` (PN370–375 / G4_80)

| PR# | Патч | Как сделано | default | план |
|---|---|---|---|---|
| #45100 | **PN370** | TextPatch: skip racy CPU accepted-counts под async + удаление per-step `event.synchronize` (~2-5% TPOT) + GDN `build()` size по `num_reqs`. Dual-anchor. **NOT spec-only** (хук есть). | off | yes |
| #45199 | **PN371** | Multi-file (11 anchor): ref-counted `EncoderCache` + 5 tracker-точек; fatal assert → warn+skip только в drafter. spec-only. | off | yes |
| #45005 | **PN372** | TextPatch `spec_decode/utils.py`: Triton-guard `seq_len<=0` (строже upstream `==0`). **NOT spec-only**. | off | yes |
| #44955 | **PN373** | TextPatch `tool_calls_utils.py`: `parallel_tool_calls` → `is not False` (explicit-null clients). spec-only. | off | yes |
| #44877 | **PN374** | TextPatch `qwen3xml_tool_parser.py`: strip whitespace+quote из param-name. **Genesis-ORIGINAL** (нет upstream на qwen3xml). spec-only. | off | partial |
| #44741 | **PN375** | Runtime monkey-patch: `_extract_streaming_delta_segments` + strip G4_14 pad перед endswith. spec-only. | off | yes |
| #45040 | **G4_80** | Runtime monkey-patch: маска `fp8_e5m2`→`fp8` для weight-only + обнуление query_quant. spec-only. | off | yes |

### 2c. Wave-2 свипа-50 — commit `ff3c0c56` (PN358 / 376–383 / G4_81 / P88)

| PR# | Патч | Как сделано | default | план |
|---|---|---|---|---|
| #44628 | **PN376** | MultiFile: `skip_with_substr=True` на ОБОИХ is_layer_skipped сайтах (на сайт больше PR). spec-only. | off | yes |
| #44563 | **PN377** | TextPatch `fused_moe.py`: clamp `block_size_k > group_size*8` + boot legality-check. **NOT spec-only** (хук line 5342). **default_on**. | **on** | yes |
| #44742 | **PN381** | TextPatch `gpu_input_batch.py`: needs_output_token_ids += allowed_token_ids условие. spec-only, NULL на PROD. | off | yes |
| #44784 | **PN383** | Multi-file: `is_eagle_group` + eagle-handling в scheduler + pre-DMA bounds-check. spec-only, dormant до KV-offload. | off | yes |
| #45060 | **PN378** | TextPatch `rejection_sampler.py` (kernel-half): `tl.where(vocab_mask,...,-inf)` (NaN-livelock). Dual-anchor. spec-only. | off | yes |
| #45080 | **PN382** | TextPatch `decode_bench_connector.py`: per-block-row fill для Mamba/GDN (Genesis extra). bench-инфра. spec-only. | off | yes |
| #45144 | **G4_81** | Runtime monkey-patch `TurboQuantAttentionImpl.forward`: route uniform K+1 spec-verify через single-token TQ decode. spec-only. | off | yes |
| #45202 | **P88** | TextPatch `kv_cache_manager.py`: stash pending record, commit раз после последнего fail-return (anti-double-count). spec-only, metrics-only. | off | yes |

### 2d. batch2 Wave-1 — commit `3ab6b57c` (PN384–388)

| PR# | Патч | Как сделано | default | план |
|---|---|---|---|---|
| #44986 | **PN384** | Multi-file: `skip_eagle_pop` в 4 сигнатуры find_longest_cache_hit; drop последнего prefix-блока подавлен только в prefill. Supersedes P83/P84. spec-only. | off | yes |
| #45477 | **PN388** | TextPatch `scheduler.py` `_mamba_block_aligned_split`: округление END-позиции; budget-fragmented chunk откладывается вместо mid-block mamba-state. Чинит LIVE prefix-cache-poisoning. requires P34. spec-only. | off | yes |

### 2e. batch3 — commit `7ca22764` (PN389–391 / P89)

| PR# | Патч | Как сделано | default | план |
|---|---|---|---|---|
| #45389 | **PN389** | 3-file: `run_with_timeout` + `_check_regex_complexity` + `VLLM_GRAMMAR_COMPILATION_TIMEOUT_SECONDS` (Genesis **2s** vs PR 10s). PARTIAL: только grammar-timeout half (input-bounds half отложена). spec-only. | off | partial |
| #45369 | **PN390** | TextPatch `rejection_sampler.py` (13 sub): `compute_target_lse` + on-the-fly `exp(logit-lse)` вместо full-vocab softmax. ULP-identical. -3.6-14.6MB. spec-only. (renamed из PN384-collision) | off | yes |
| #45453 | **PN391** | 6-file: additive `GET /health/decode` (ok/prefilling/idle/stalled). NCCL-deadlock TP=2. spec-only, off до wiring. | off | yes |
| #45471 | **P89** | 2-file: `completion_tokens_details.reasoning_tokens` + Genesis-extension schema-поля accepted/rejected (None — недостижимы per-request). spec-only. | off | yes |

### 2f. Одиночные коммиты

| PR# | Патч | Как сделано | default | план |
|---|---|---|---|---|
| #45290 | **PN385** | TextPatch `get_json_schema_from_tools`: no-arg forced tool → `{"type":"object","properties":{}}`. spec-only. | off | yes |
| #45346 | **PN387** | 2-layer: backport empty-grammar guards + Genesis edge-guard (чистый 400 до engine). CONFIRMED DoS-fix. spec-only. | off | yes |
| #45389→**PN386** | **PN386** | TextPatch `tool_parsers/streaming.py`: JSON-string-aware `_bracket_level`. P68-prerequisite. spec-only. | off | yes |
| #45196 | **PN379** | MultiFile (6 sub): loader/LoadConfig fail-fast validation. Constructor-only. spec-only. | off | yes |
| #45076→#44745 | **PN367** | v2: `max(delta,0)` + WARNING + 1 MiB floor. **NOT spec-only** (хук line 5274). **default_on**. | **on** | yes |

**PN392 / G4_79** (commits `a3b84468`, `86615ae6`) — spec-only-партия,
boot-active после фикса 14 июня. PN392 — dev491 streaming blocker (см. §4.3).

---

## 3. Не взято — по причинам

### 3a. Дубликаты существующих патчей
`#42237` (G4_T1 v2), `#44717`/`#44752` (G4_T1 key-strip), `#44993` (P62),
`#45068`/`#45449` (G4_T1 v2 immune), `#44560` (P34 в пине), `#45364`
(torch.compile rms_quant_fusion), `#45365` (P64 mutually-exclusive).

### 3b. Watchlist / retire-trigger
`#40756`, `#41967`, `#42300`, `#43626`, `#44715`, `#44779`, `#45120`,
`#43355`+`#43572`, `#45151` (PN351 anchor-breaker), `#45171` (MERGED —
PN288+P107 drift), `#45252` (MERGED security), `#45378`, `#45415`
(csrc миграция), `#45464` (Harmony pin-landmine), `#45527`.

### 3c. Research / метод-only
`#40547`, `#44297` (json_object reproducer→P62 тесты), `#45022`
(endurance_probe), `#45053`, `#45096`, `#45109` (tokenizer fingerprint-gate),
`#45126`, `#45280`, `#45283`, `#45349`, `#45363`, `#45370` (NeoX-RoPE
extension), `#45371`, `#45379` (A5000 re-sweep seed), `#45423`, `#45434`
(GSM8K fp8-KV data), `#45466`, `#45480`.

### 3d. Неприменимо к Ampere SM8.6
`#45001` (ViT .cu), `#45368` (XPU), `#45384` (MLA int32), `#45393`
(MLA bound-check), `#45530` (CUTLASS SM90+).

### 3e. Фиксы в наш код вместо вендоринга
`#44613`/`#44754` (snapshot-hardening turboquant_attn.py:496), `#45038`
(extend G4_31 wrap), `#45404` (PN377 boot-guard), `#45417` (schema_v2.py
`or v is None`), `#45479` (PN375 name-recovery + v2/v3 тест).

### 3f. Отложено на волны 2/3
**Wave 2:** `#45265`, `#45299`, `#45306`, `#45310`, `#45339`, `#45343`,
`#45351`, `#45361`, `#45379`, `#45383`, `#45497`, `#45517`.
**Wave 3:** `#45130`, `#45146`, `#45320`, `#45352`, `#45357`, `#45375`, `#45406`.

---

## 4. Ключевые выводы и риски

**4.1. Spec-only-boot gap (устранён 14 июня `1a84f632`).** Главный системный
риск: вся новая партия была инертна при буте до supplement-фикса. Закрыто.
Исключения boot-active: PN367/PN370/PN372/PN377. Особый случай: G4_60E
(предсуществующий, но spec-only — тоже был инертен). Ложная тревога: G4_T1
(нет хука, но runtime-мутации нет → терять нечего).

**4.2. Расхождения с планом (`matches_plan != yes`):**
- **P108 (#42603)** — реализация верна (refined defensive_overlay), но roadmap
  метит retirement-кандидатом против #45005; патч остаётся default-ON.
- **PN374 (#44877)** — Genesis-ORIGINAL qwen3xml-аналог (clause b), не primary
  Gemma4-deliverable PR; `related_not_superseding`.
- **PN389 (#45390/#45389)** — partial scope: только grammar-timeout half;
  input-bounds half отложена (P109/PN387 anchor-overlap).
- **G4_T1 (#42006)** — registry цитирует stale #42006, live на v2 (#42237).
- **PN365 (#42746)** — unknown: завендорен раньше, нет roadmap-строки.

**4.3. dev491 streaming blocker (PN392).** Spec-only, boot-active после
14 июня. Единственная незакрытая верификационная зона — live-проверка
streaming tool-calls на dev491-пине (требует авторизации, см. отдельный
журнал pin-bump-dev491-status §Update 2026-06-14).

**4.4. Конфликты/порядок для включения:**
- HARD CONFLICT `PN365`↔`PN204`; PN365 SKIP на FP8.
- Порядок: `PN370` после `PN341`; `PN388` requires `P34`; `G4_60E` requires `G4_60A`.
- default-ON по дизайну: `PN118`, `PN346`, `P108`, `PN367`, `PN377`.
- Dormant на PROD: `PN383`, `PN382`, `G4_80`/`PN376`, `PN381`.

**4.5. Итог.** 42 PR корректно завендорены с source-verified реализацией
(anchor/sub-patch/drift-marker подтверждены пофайлово). Главный риск
(spec-only-boot) устранён `1a84f632`. Расхождения с планом задокументированы
и не являются дефектами. Незакрыто: dev491 streaming (PN392).

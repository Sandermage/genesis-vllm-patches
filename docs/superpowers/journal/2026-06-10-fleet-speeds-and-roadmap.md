# 2026-06-10 — карта скоростей флота + roadmap интеграций

## Скорости флота (из валидированных YAML-метрик + свежие замеры)

| Модель/профиль | single | conc=4 | conc=8 | Источник |
|---|---|---|---|---|
| **35B balanced (PROD)** | проза 201 / код 220 / math 249 | 271-286 agg | — | свежие замеры 06-10 |
| 35B multiconc | 215 | 478 | **689** ⭐ | YAML 05-15 |
| 35B dflash / -multiconc | TTFT 74ms | — | 562 agg | YAML |
| **27B tq-k8v4** | **97** (цель 150+!) | 265 | — | YAML dev371 |
| 27B multiconc | 97 | 265 | **379** | YAML 05-15 |
| 27B dflash | ~98, TTFT 102ms | — | 385 agg | YAML |
| Gemma4 26B chat-k3 | 138-207 | — | — | club-3090 ref |
| Gemma4 31B tq-k3 | **~44** | — | — | YAML 06-01 |

## ГЛАВНАЯ НАХОДКА: 27B на dev259 СЛОМАН (P0-блокер)

Свап-кампания: 27B-tq-k8v4 на свежем pin'е падает на ПЕРВОМ decode:
`AssertionError: Workspace is locked but allocation from
'turboquant_attn.py:907:_decode_attention' requires 0.38 MB, current
size 0.00 MB. Growth not allowed after locking.`

- Это класс #44053 (наш PN353A), который скипается по drift-маркеру
  "upstream absorbed" — но upstream-версия НЕ покрывает 27B decode-путь
  (GQA=6 shape → свой kv_splits → свой workspace bucket).
- Warmup-флаги (PN130/126/128/129/364) добавлены в launcher — wrapper
  "installed", но worker-исполнения "warmup ✓" в логах НЕТ (iron-rule-12
  evidence отсутствует) — вторая загадка для дебага.
- Значит и in-pin wins для 27B (#36329/#43731/#44025) недоступны, пока
  блокер не снят. **27B = куда копать №1** (97 → 150+ требует сначала
  просто завестись на новом pin'е).
- 35B PROD восстановлен.

## Roadmap интеграций из смежных движков (research 06-10)

1. **vLLM Sleep Mode** (M): weights → pinned RAM, wake <1s. Заменяет
   ручные свапы контейнеров (18-200× быстрее); 35B+27B резидентно,
   один спит. Главная архитектурная цель.
2. **club-3090 сигнал** (S): они ЗААРХИВИРОВАЛИ Genesis-путь (#330,
   55/64 патчей deprecated, "Genesis on hold upstream") и ушли на
   сток v0.22.0 (172 TPS vs наши 211). Действие: держать валидированный
   сток-fallback compose + перенять их atomic vllm-pin-bump.sh.
3. **XGrammar-2** (S): 6× быстрее компиляция грамматик; constrained
   tool-args структурно убивают класс #178 (порча аргументов).
   Проверить наличие в pin + A/B на agentic.
4. **MambaRadixCache** (L, blueprint): SGLang'овский префикс-кеш для
   hybrid GDN (SSM-snapshot-on-match, dual-LRU). Чертёж для PN95
   Phase-3 и hybrid APC — главный TTFT-рычаг для RAG.
5. **Async-scheduling + spec overlap** (S): проверить что наш pin
   включает overlap с MTP — может быть спящий бесплатный выигрыш.
6. **Adaptive suffix↔MTP routing** (M-L): per-batch выбор драфтера
   (suffix для повторяющихся code-edit, MTP для прочего).
7. **ik-llama IQ4_XS lane** (S): 35B на ОДНОЙ карте, 262K ctx,
   112-137 TPS, NIAH-clean до 240K — escape hatch для аналитики,
   которого vLLM на 24GB дать не может.
8. **Целевая архитектура**: catalog-driven gateway → 2 резидентных
   vLLM (35B+27B, sleep-swap) + ik-llama long-ctx lane + workload-
   роутинг (код→35B+XGrammar, аналитика>120K→ik-lane, RAG→APC).

## Очередь次

1. 27B locked-workspace дебаг (P0 для цели 150+)
2. Sleep Mode пилот (архитектурный выигрыш)
3. XGrammar-2 verify + async-overlap verify (S, могут быть спящие)
4. 35B-multiconc свежий бенч (689 TPS ref — проверить наш стек)

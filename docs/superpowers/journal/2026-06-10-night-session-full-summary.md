# 2026-06-10 ночная сессия — полный итог: bisect, sweep, матрица чатов, 4 патча

## Закрытые вопросы

### 1. «Регрессия 217→197» — РАЗРЕШЕНА: артефакт методологии
`--max-tokens` default 200 vs историческая 1024. Подтверждено бенчем:
тот же конфиг @1024 = 217.44 wall_TPS. TPOT за сессию улучшился.
Правило: всегда явный `--max-tokens` (закреплено в журнале от 2026-06-10).

### 2. Leave-one-out sweep (9 циклов) — регрессоров НЕТ
PN350/FLA-warps/P81 = wins (+0.8-0.9% TPOT каждый), P24/PN345/PN340-341
small wins, P18B нейтрален, P67 ОБЯЗАТЕЛЕН (0/5 quality без него).
Вывод: путь к цели — новые оптимизации, не чистка старых.

### 3. Матрица скоростей по типам чатов (35B, @usage-токены)
| Тип | TTFT | TPOT | wall TPS |
|---|---|---|---|
| math/thinking | 155 | 3.65-3.71 | **250-253** |
| multi-turn (prefix hit) | 158 | 4.01 | 217 |
| code | 150 | 4.36 | 215 |
| short chat | 156 | 4.24 | 200 |
| creative long-gen | 145 | 5.10 | 191 |
| 8K-ctx QA | 1372 | 4.44 | 102 |
| 32K-ctx QA | 5595 | 4.82 | 37.5 |

**Скорость определяется контентом сильнее чем патчами**: MTP accept
на предсказуемом тексте (math/код-шаблоны) даёт 250+, проза — 191.
Prefill ~170ms/1K tok (GDN O(N) 30 слоёв) — аналитика/поиск prefill-bound.

### 4. A/B-эксперименты конфига
- PN125 FULL_AND_PIECEWISE @280K: TPOT +1.4% хуже → откат force, 280K оставлен.
- K=4: TPOT +5.5% хуже + коллапс конкурентности → K=3 оптимум (K=1 был −29%).
- Probabilistic draft: не включать (документированный −5.9% от 2026-05-15).

### 5. Патчи сессии
- **PN365 v2** (GDN single-GEMM): PN50-aware якоря написаны и применились,
  но **CRASH на FP8 Marlin** (size_n 6176 % 64). Author bench был NVFP4.
  FP8-guard добавлен, ливер ЗАКРЫТ для 35B-FP8 до FP8-aware layout.
- **PN352** (Triton moe_sum topk=8): OOB-баг исправлен (num_tokens из
  output), standalone ALL_PASS, но в engine — **stream race** (с
  CUDA_LAUNCH_BLOCKING=1 работает). ЗАПАРКОВАН по правилу 3-х
  диагностических рестартов. Resume-путь в docstring.
- **PN367** (clamp негативной оценки cudagraph памяти, vendor #45076 от
  пользователя): применён на PROD, 2/2 sub-patches. Защита KV-бюджета
  24GB карт + WARNING-видимость.
- **PN204 residue** снят с live-файла (новый bug class: спящий текст
  env-disabled патча блокирует якоря других патчей).

### 6. Новые правила/уроки
- Bug class: stale text-patch residue после env-disable → ревертить текст.
- TDD ядра standalone ДО wiring в engine (PN352 crash 1 был предотвратим).
- CUDA_LAUNCH_BLOCKING=1 — быстрый классификатор stream-race.
- docker start + exec-revert + restart гонится с boot apply.
- Правило 3-х диагностических рестартов для < 3% lever'ов.

## Цель 250+ (35B) / открытые рычаги

Текущее: mixed-контент 209-217 @1024, math-контент УЖЕ 250-253.
Для mixed 250+ нужно TPOT ≤ 3.86ms (−14% от 4.5):

1. **PN353-#43887** (TQ MTP K+1 verify через decode path): −5-9% TPOT —
   главный курс, ~1798 LOC порт, следующая фокус-сессия.
2. **PN355-#43642** (hybrid GDN warmup, lesj0610): TTFT −200-600ms.
3. **PN354 USE_EXP2** GDN chunk kernel: +3-5% TTFT prefill (аналитика!).
4. Research-агент свипует свежие PRs (память/кеш/TQ/контекст/код) — результаты следующим шагом.
5. tool_call: в launcher 35B НЕТ --tool-call-parser — агентские
   workloads не получают parsed tool_calls. Решение оператора:
   добавить `--tool-call-parser qwen3_xml` (рекомендация club-3090).

## Состояние PROD на конец сессии
35B-balanced: UP, стабилен, K=3, 280K ctx, PN367 applied, PN352/PN365
текст чист (реверчен), autotune cache реконвергирует (TPOT вернётся
к ~4.45-4.5 по мере прогрева). Все коммиты на sndr-dev, public origin
не тронут.

## Research sweep (агент, 3 недели vLLM PRs, наш профиль) — ranked

| PR | Статус | Что | Релевантность | Усилие | Win |
|---|---|---|---|---|---|
| **#44986** | open | MTP роняет последний matched prefix-block в prefill | MTP+prefix caching — может значить ПОЛНЫЙ промах кеша на RAG; автор тестил на Qwen3.6-27B | S | большой TTFT-win на warm RAG/search |
| **#44700** | merged 06-06 | GDN: split mixed prefill+decode батчей, decode → recurrent kernel | chunked prefill @280K = постоянные mixed-батчи; decode платит ~2x | M | TPOT-стабильность при длинном prefill (аналитика) |
| **#41824** | open | In-place SSM state в GDN chunk-prefill Triton | прямое попадание в 170ms/1K prefill bottleneck; +10-13% на больших Qwen | M | prefill speed (аналитика) |
| **#40172** | merged 05-21 | Fused Mamba state postprocess — убирает CPU sync при prefix caching | наш ТОЧНЫЙ конфиг (MTP+prefix+hybrid) | M | TPOT shave (bubble) |
| **#43878** | open | TQ streaming fallback long continuation | TQ @280K alloc-spike | S-M | OOM-защита |
| **#26807** | open | GDN Automatic Prefix Caching all-mode (SSM states на границах блоков) | Mamba-state caching для RAG | L | warm-TTFT трансформация |
| **#40914** | open | TQ K+1 spec-verify routing bugfix | TQ+MTP verify | S | корректность |
| **#44927** | open | structured output + MTP off-by-one | guided JSON код/агенты | S | корректность |
| **#39988** | open | TQ FP8 cast crash BF16 на Ampere | разблокирует TQ на 27B | S | 27B unblock |

Рекомендация агента: #44986 первым (S-усилие, прямое попадание в
поиск/RAG), затем кампания #44700+#41824 (prefill-bound аналитика),
затем #40172. TQ-гигиена (#43878/#40914/#39988) — в PN353-ревью.

## Очередь следующей фокус-сессии (к 250+)
1. #44986 prefix-cache fix (S) → warm-TTFT для поиска
2. PN353-#43887 TQ MTP verify routing (−5-9% TPOT) — главный decode-рычаг
3. #44700 + #41824 GDN prefill пара → аналитика (170ms/1K → цель ~120-140)
4. #40172 CPU bubble removal
5. PN354 USE_EXP2 (+3-5% TTFT prefill)
6. tool-call parser в launcher (оператор: qwen3_xml)
7. 27B вариант: #39988 + PN356 + бенч-матрица

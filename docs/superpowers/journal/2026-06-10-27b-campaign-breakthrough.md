# 2026-06-10 — 27B кампания: 97 → 138/143 TPS (+43%) на новом pin'е

## Результат

| Метрика | dev371 (YAML ref) | dev259 свежий бенч @1024 |
|---|---|---|
| wall_TPS single | 97 | **138.4 (median 143.4)** |
| TPOT | 29.5 ms | **7.2 ms (4×!)** |
| TTFT | 105 ms | 113 ms |
| Quality | — | 5/5 |

**+43% от связки новый pin (в нём #36329 GDN Marlin TP=2 fix, #43731,
#44025) + наши свежие патчи.** До цели 150+ осталось +5% (median уже 143).

## Урок кампании (важный): конфиги были правы

Крэш `Workspace is locked... 0.38 MB` на первых двух попытках был
вызван МОИМ ручным контейнером: я собрал docker run по образцу 35B,
но без env-набора из YAML. А YAML 27B
(qwen3.6-27b-int4-autoround-tq-k8v4.yaml:116-117,217) УЖЕ прописывает
G4_61 (TQ shared workspace, max-shape pre-reserve BEFORE lock — ровно
противоядие assertion'у), G4_62 и PN130.

Атрибуция подтверждена boot-логом: "G4_61 installed: ... max-shape
pre-reserved before lock" → decode заработал.

**Правило**: контейнеры собираются ИЗ конфигов (genesis model-config
launch), не руками. Ручная сборка теряет genesis_env-набор YAML.

## Verify спящих выигрышей (агент)

- **XGrammar-2: УЖЕ ACTIVE** (v0.2.1 в pin'е, default backend,
  request-level `response_format: json_schema` — ноль работ на сервере).
  Caveat: #44927 MTP off-by-one не зафиксен в pin'е — A/B перед
  продакшен-зависимостью.
- **Async-scheduling overlap: УЖЕ ACTIVE с MTP** (live-лог
  "Asynchronous scheduling is enabled" + step_with_batch_queue,
  max_concurrent_batches=2). Zero-bubble уже в наших цифрах.
  Замечание: guided-decoding частично сериализует overlap (core.py:524).

## Состояние

PROD 35B восстановлен и работает. 27B-bench контейнер остановлен
(остаётся для следующей сессии: 143 → 150+ = доводка + закрепление
в YAML reference_metrics).

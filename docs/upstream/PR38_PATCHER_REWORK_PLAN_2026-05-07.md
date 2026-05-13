# Genesis / SNDR Core: план по 38 vLLM PR и переделке патчера

Дата: 2026-05-07  
Статус: рабочий план для реализации  
База анализа:

- maintainer-local audit notes (`project_pr_batch_audit_DETAILED_2026_05_07.md`)
- maintainer-local audit notes (`project_pr_batch_audit_2026_05_07.md`)
- `genesis_deep_audit_2026-05-07.md`
- `sndr_structure_deep_audit_2026-05-07.md`
- текущий код `vllm/sndr_core`, `vllm/_genesis`, model configs, tests
- ручная сверка текущего состояния PR в `vllm-project/vllm`

## 1. Главный вывод

Переписывать Genesis/SNDR patcher с нуля не нужно. В проекте уже есть сильные
примитивы:

- `vllm/sndr_core/core/text_patch.py` — idempotent text patcher с drift markers,
  per-sub upstream markers, manifest fast-path и verification layer.
- `vllm/sndr_core/core/multi_file.py` — двухфазная multi-file транзакция с dry-run
  и rollback.
- `vllm/sndr_core/dispatcher/registry.py` — metadata registry по patch id.
- `vllm/sndr_core/env.py` — централизованный реестр флагов.
- `vllm/sndr_core/bundles/*` — атомарные feature bundles.
- `vllm/sndr_core/wiring/anchor_manifest.py` и `patcher_registry.py` — задел для
  site-map/manifest-based patching.

Проблема не в отсутствии ядра, а в раздвоении управления:

- metadata registry живет в `sndr_core/dispatcher/registry.py`;
- apply loop все еще живет через `sndr_core/apply/_per_patch_dispatch.py`;
- часть canonical кода все еще импортирует `vllm.sndr_core.*`;
- bundles используют новые транзакции, но вызывают legacy `_make_patcher`
  factories;
- model configs включают наборы флагов напрямую, без единого профиля риска.

Правильное направление: сохранить существующие хорошие части и постепенно
перевести систему на единый registry-driven patch engine.

## 2. Рекомендуемый вариант

Есть три варианта.

### Вариант A: минимальный

Сделать только новые backports:

1. Добавить PN82 по vLLM PR #41873.
2. Переписать PN55 под nested KV cache из PR #41896.
3. Стандартизировать P61c/tool-parser bundle в model configs.

Срок: 1-2 дня.  
Риск: низкий.  
Минус: архитектурный долг остается.

### Вариант B: рекомендуемый

Сделать fixes из варианта A и одновременно начать registry-driven миграцию:

1. PN82 как новый patch.
2. PN55v2 вместо отдельного PN83.
3. Tool parser bundle как единая Qwen3Coder policy.
4. Ввести `PatchSpec` как thin layer над текущим registry.
5. Перевести apply loop с `_per_patch_dispatch.py` на `apply_module`.
6. Оставить legacy `_genesis` только как compatibility shim.

Срок: 5-8 рабочих дней.  
Риск: средний, но контролируемый.  
Плюс: после этого новые PR/backports будут добавляться быстрее и безопаснее.

### Вариант C: полный rewrite

Переписать patcher как полностью декларативный движок: patch specs, validators,
AST/text hybrid anchors, CI manifest builder, model-profile policy engine.

Срок: 2-4 недели.  
Риск: высокий.  
Минус: можно сломать рабочий production path ради архитектуры.

Рекомендация: вариант B. Он улучшает качество без большой остановки разработки.

## 3. Исправления, которые надо сделать первыми

### 3.1. P61c / Qwen3Coder tool-call parser policy

Текущее состояние:

- `P61c` уже есть в `vllm/sndr_core/integrations/tool_parsing/p61c_qwen3coder_deferred_commit.py`.
- `BUNDLE_TOOL_PARSING_QWEN3CODER` уже есть в `vllm/sndr_core/bundles/tool_parsing_qwen3coder.py`.
- В `a5000-2x-35b-prod.yaml` `GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT=1`
  уже помечен как promoted/live-verified.
- В `a5000-2x-27b-int4-tq-k8v4.yaml` и DFlash-вариантах P61c еще не включен.
- Upstream issue #22975 закрыт как `not_planned`, значит ждать vLLM fix нельзя.

Что делать:

1. Не писать новый P61c.
2. Сделать P61c частью стандартной Qwen3Coder policy.
3. После live smoke на 27B включить:
   - либо `GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT=1`;
   - либо лучше `SNDR_ENABLE_BUNDLE_TOOL_PARSING_QWEN3CODER=1`, если bundle
     после проверки стабилен на целевом pin.

Куда интегрировать:

- `vllm/sndr_core/model_configs/builtin/a5000-2x-27b-int4-tq-k8v4.yaml`
- `vllm/sndr_core/model_configs/builtin/a5000-2x-27b-int4-tq-k8v4-dflash.yaml`
- `vllm/sndr_core/model_configs/builtin/a5000-2x-27b-dflash-true.yaml`
- возможно `a5000-1x-27b-int4-tested.yaml`, если tool parser используется там.

Сноска:

- Что делает: не дает qwen3coder parser навсегда перейти в режим tool-call,
  когда модель в обычном тексте упомянула `<tool_call>`.
- Как делает: откладывает commit `is_tool_call_started=True` до появления
  подтверждающего `<function=` в short slack window.
- Что меняет: поведение streaming parser, но только в Qwen3Coder tool parser path.
- Что дает: убирает 30-120 секунд SSE silence и потерю content deltas.
- Что получим: стабильный tool-call streaming для Qwen3Coder в PROD-конфигах.
- Как получим: через существующий P61c или atomic bundle.
- Риски: false negative, если реальный tool-call header приходит сильно позже
  текущего slack window; нужен live regression на tool-calls.
- Как лучше интегрировать: сначала single flag на 27B, затем umbrella bundle.
- Авторство: логика mitigation и интеграция Genesis/SNDR — примерно 70-80%
  Sander/Genesis; upstream issue/reproducer credit должен оставаться отдельно.

### 3.2. PN82: Mamba CUDA graph stale prefill rows, vLLM PR #41873

Текущее состояние:

- PR #41873 открыт.
- Patch маленький и точечный.
- В текущем Genesis/SNDR нет PN82 для этой проблемы.
- В проекте уже есть `P82`, но это другой patch: SGLang acceptance threshold.
  Поэтому новый id нельзя назвать просто `P82`. Нужен `PN82` или более явный
  `PN82_MAMBA_PREFILL_ZERO`.

Что делать:

Добавить новый patch:

- файл: `vllm/sndr_core/integrations/worker/pn82_mamba_cudagraph_prefill_zero.py`
- patch id: `PN82`
- env flag: `GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO`
- family: `worker`
- tier: `community`
- upstream_pr: `41873`
- default_on: `False`
- applies_to: `{"is_hybrid": [True]}`

Суть изменения в upstream:

```python
is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu
is_prefilling[num_reqs:] = False
```

Куда интегрировать:

- `vllm/sndr_core/env.py` — добавить `PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO`.
- `vllm/sndr_core/dispatcher/registry.py` — добавить entry `PN82`.
- `vllm/sndr_core/integrations/worker/pn82_mamba_cudagraph_prefill_zero.py` —
  сам TextPatcher.
- `vllm/sndr_core/apply/_per_patch_dispatch.py` — только если до registry-driven
  apply loop еще не дошли.
- `tests/unit/env/test_registry_flag_coverage.py` уже поймает отсутствие flag.
- Новый тест: `vllm/sndr_core/tests/test_pn82_mamba_cudagraph_prefill_zero.py`.

Сноска:

- Что делает: не дает padded CUDA graph rows наследовать старые значения
  `is_prefilling` после `condense()`.
- Как делает: зануляет `is_prefilling[num_reqs:]` для padded rows.
- Что меняет: один участок в `v1/worker/gpu_model_runner.py`.
- Что дает: меньше ложных prefill decisions в Mamba/hybrid CUDA graph path.
- Что получим: более надежный hybrid path при batch padding и CUDA graphs.
- Как получим: маленький TextPatch с exact anchor и upstream drift marker.
- Риски: anchor drift из-за PR #41728/#41703; риск runtime regression низкий.
- Как лучше интегрировать: default OFF, включить только на hybrid configs после
  unit + smoke.
- Авторство: upstream algorithm 80-90% авторства PR #41873; Genesis/SNDR
  интеграция, gating, tests и registry — 10-20% Sander/Genesis.

### 3.3. PN55v2 вместо отдельного PN83: nested KV cache wake_up, vLLM PR #41896

Текущее состояние:

- У нас уже есть `PN55` в `vllm/sndr_core/integrations/worker/pn55_wake_up_hybrid_kv.py`.
- Он backport-ит #41602 и чинит `list[Tensor]`.
- Новый PR #41896 расширяет тот же bug class: KV cache может быть nested
  `list`, `tuple`, `dict/Mapping`, а не только `list`.
- Если сделать отдельный PN83 поверх PN55, патчи будут конкурировать за один
  и тот же anchor в `gpu_model_runner.py`.

Что делать:

Не создавать отдельный runtime patch `PN83`.

Вместо этого:

1. Переписать `PN55` как `PN55v2`.
2. В registry оставить id `PN55`, но обновить title/credit:
   - upstream_pr: `41602`
   - related_upstream_prs: `[41896]` или указать в `credit`, если schema пока
     не поддерживает list.
3. Заменить текущий list-only replacement на recursive iterator:
   - tensor -> yield tensor
   - mapping -> recurse values
   - list/tuple -> recurse elements
   - None/non-tensor -> skip
4. Добавить upstream drift markers:
   - `_iter_kv_cache_tensors`
   - `Mapping`
   - `init_fp8_kv_scales`

Куда интегрировать:

- `vllm/sndr_core/integrations/worker/pn55_wake_up_hybrid_kv.py`
- `vllm/sndr_core/dispatcher/registry.py`
- `vllm/sndr_core/tests/test_pn55_wake_up_hybrid_kv.py`
- возможно новый semantic helper test с nested `list/tuple/dict`.

Сноска:

- Что делает: wake_up больше не падает, если KV cache вложен в container.
- Как делает: вместо прямого `.zero_()` по top-level элементам рекурсивно
  обходит nested containers и zero-ит только реальные tensors.
- Что меняет: один patch replacement в `gpu_model_runner.py`.
- Что дает: защита не только от Mamba `list[Tensor]`, но и от будущих/nested
  KV structures.
- Что получим: один coherent wake_up patch вместо двух конфликтующих.
- Как получим: upgrade PN55, а не новый PN83.
- Риски: надо аккуратно добавить `Mapping` import, если upstream file его еще
  не импортирует; надо не сломать existing marker idempotency.
- Как лучше интегрировать: сохранить old env flag
  `GENESIS_ENABLE_PN55_WAKE_UP_HYBRID_KV`, потому что operator intent тот же.
- Авторство: идея nested recursion из upstream PR #41896; Genesis/SNDR авторство
  в объединении #41602/#41896, idempotent patch design и tests — примерно 25-35%.

## 4. Что из 38 PR делать, мониторить или пропустить

| PR | Текущий смысл | Решение | Причина |
|---|---|---|---|
| #41703 | DFlash/Gemma4 batched verification | Watch | Большой open PR, высокий риск drift для PN21/PN24/PN38/PN40, но не backport сейчас. |
| #41763 | Qwen3 MoE shared expert precision under sequence-parallel | Watch | Genesis TP=2 без SP/EP; полезно только при другой topology. |
| #41896 | Nested KV cache FP8 wake_up | Do via PN55v2 | Тот же bug class, что PN55; отдельный PN83 не нужен. |
| #41873 | Mamba stale `is_prefilling` padded rows | Do as PN82 | Маленький, безопасный, relevant для hybrid CUDA graph. |
| #41747 | Optional router logits MoE refactor | Watch | Может сломать P31 anchors после pin bump. |
| #41728 | Scheduler prefill chunk alignment | Watch | Может задеть PN52/scheduler anchors. |
| #41883 | W16A16 linear kernel abstraction | Watch | Большой refactor, не backport. |
| #41890 | Circular import lazy loading | Watch | Хорошая infra, но лучше дождаться merge. |
| #41939 | Strip thinking tokens from prefix cache | Watch | Prefix caching у нас обычно disabled из-за hybrid crash. |
| #41748 | Persistent topk perf | Watch | Может быть смежно с PN26/P37, но не срочно. |
| #41931 | TRTLLM MXFP4 fake output shape | Skip | Не текущий stack. |
| #41947 | Marlin MoE TP padding NVFP4/H100 | Watch | Будущее H100/NVFP4, не A5000. |
| #41915 | RTX 5090 Triton MoE optimizations | Watch | Будущее Blackwell. |
| #41889 | FusedMoEWithLoRA attr | Skip | Merged, LoRA не PROD path. |
| #41892 | Quark W8A8 INT8 Step-3.5 | Skip | Model-specific. |
| #41882 | NVFP4 AsyncTP fusion | Skip/Watch | NVFP4 не текущий A5000 stack. |
| #41868 | CUTLASS scaled mm non-compatible sizes | Watch | Может быть полезно Ada/Blackwell/PN77, но не now. |
| #41785 | LoRA H2D overlap | Skip | LoRA не primary. |
| #41796 | KV transfer docs | Skip | Docs only. |
| #41928 | KV offload HND | Skip | KV offload не используется. |
| #41929 | KV load error in OpenAI responses | Skip | Только KV offload. |
| #41945 | KV offload store deferral | Skip | Только KV offload. |
| #41777 | SimpleCPUOffload final block flush | Skip | Не current deployment. |
| #41727 | KV offload manager refactor | Skip | Не current deployment. |
| #41847 | KV transfer HMA default | Skip/Watch | Только если включать KV transfer/disaggregated serving. |
| #41923 | GDN NIXL P/D disagg | Skip | Disaggregated serving не current path. |
| #41887 | Auto-revert test infra | Skip | CI infra. |
| #41943 | CI output surfacing | Skip | CI infra. |
| #41910 | collect_env mac/Windows | Optional | Может помочь dev на Mac, но не engine fix. |
| #41776 | MIG UUID support | Skip | Не текущий hardware. |
| #41723 | Qwen multimodal budget | Skip | VL/multimodal not primary. |
| #41875 | BitsAndBytes Mamba2 | Skip | Не current quantization path. |
| #41936 | Qwen3-ASR cap | Skip | Model-specific. |
| #41944 | Gemma4 K=V projection optimization | Watch | Будущее Gemma4. |
| #41905 | MiMo 2.5 MTP | Skip | Model-specific. |
| #41755 | GLM4-MoE NVFP4 loading | Skip | Merged, not current stack. |
| #41769 | ModelOpt NVFP4 W4A16 | Watch | Future NVFP4/Blackwell. |
| #41683 | Gemma4 NVFP4 per expert | Watch/Ignore | Huge open PR, high churn, not current stack. |

## 5. Архитектурный план переделки патчера

### 5.1. Цель

Получить patch engine, где каждый patch описан один раз и применяется одним
путем:

```text
PatchSpec -> decision -> patcher factory -> transaction -> result -> report
```

Сейчас путь раздвоен:

```text
dispatcher.PATCH_REGISTRY     -> decision/metadata
apply._per_patch_dispatch     -> фактический порядок и вызовы
legacy vllm.sndr_core imports  -> compatibility and historical paths
```

Цель — убрать второй источник истины.

### 5.2. Ввести `PatchSpec`

Новый файл:

- `vllm/sndr_core/dispatcher/spec.py`

Минимальная структура:

```python
@dataclass(frozen=True)
class PatchSpec:
    patch_id: str
    title: str
    tier: str
    family: str
    env_flag: str | None
    default_on: bool
    lifecycle: str
    upstream_pr: int | None
    apply_module: str
    applies_to: dict[str, list[Any]]
    requires_patches: tuple[str, ...] = ()
    conflicts_with: tuple[str, ...] = ()
```

Что делает:

- превращает dict metadata в typed contract;
- позволяет validation раньше, чем patch начнет менять файлы;
- делает apply loop независимым от `_per_patch_dispatch.py`.

Как интегрировать:

1. На первом этапе не переписывать `PATCH_REGISTRY`, а добавить converter
   `iter_patch_specs()`.
2. Проверять, что каждый active patch имеет `apply_module`.
3. Если `apply_module` отсутствует, временно использовать legacy dispatch.
4. Когда coverage станет 100%, удалить `_per_patch_dispatch.py`.

Авторство: 85-90% Sander/Genesis, потому что это внутренняя архитектура SNDR.

### 5.3. Registry-driven apply loop

Изменить:

- `vllm/sndr_core/apply/orchestrator.py`
- `vllm/sndr_core/apply/_state.py`

Новый порядок:

1. Load registry.
2. Validate schema.
3. Apply enabled bundles.
4. Iterate active `PatchSpec`.
5. Run `should_apply(patch_id)`.
6. Import `spec.apply_module`.
7. Call `apply()`.
8. Normalize result into `PatchResult`.

Что дает:

- исчезает ручная регистрация 95 функций;
- меньше риска, что patch есть в registry, но не вызывается;
- проще добавлять новый PR backport;
- проще строить docs/CLI/reporting.

Риск:

- можно изменить порядок patch application.

Как снизить:

- добавить `order` или `phase` field в registry;
- сначала запускать новый loop в shadow mode и сравнивать с текущим
  `_state.PATCH_REGISTRY`.

### 5.4. Legacy `_genesis` оставить только как shim

Сейчас часть canonical кода импортирует `vllm.sndr_core.*`. Это мешает понять,
где настоящий код, а где совместимость.

Что делать:

1. Все `vllm/sndr_core/*` модули переводить на `vllm.sndr_core.*` imports.
2. В `vllm/_genesis` оставить thin re-export files.
3. Тесты постепенно перевести с `vllm.sndr_core.tests` на `tests/unit` или
   `tests/patches`.

Что получим:

- понятный ownership;
- меньше circular import риска;
- проще сделать plugin/community packaging.

Риск:

- старые тесты и external scripts могут импортировать `_genesis`.

Как снизить:

- compatibility shim не удалять до следующего major release;
- добавить deprecation warnings только в CLI/debug mode, не в hot path.

### 5.5. Anchor manifest сделать обязательным для стабильных патчей

Сейчас manifest-aware path есть, но opt-in не везде.

Что делать:

1. Для каждого stable patch у `TextPatcher` должен быть `patch_id`.
2. Все `_make_patcher()` должны регистрировать patcher в
   `vllm/sndr_core/wiring/patcher_registry.py`.
3. CI/build step генерирует manifest для known-good vLLM pin.
4. Runtime:
   - если pristine md5 совпал, применяем O(1) offset splice;
   - если нет, fallback на обычный anchor scan;
   - если required anchor missing, warning не прячется.

Что дает:

- faster boot на большом наборе patch;
- точнее drift diagnosis;
- меньше silent skips.

Риск:

- stale manifest может дать false confidence.

Как снизить:

- manifest использовать только при exact md5 match;
- любое mismatch = legacy path.

## 6. Тестовая стратегия

Для каждого нового/измененного patch:

1. Anchor test:
   - old anchor содержит реальный buggy code;
   - replacement содержит marker;
   - patch idempotent.

2. Registry/env test:
   - flag есть в `Flags`;
   - registry entry есть;
   - `should_apply()` false by default и true при env flag.

3. Semantic unit test:
   - PN82: padded rows становятся `False`;
   - PN55v2: nested list/tuple/dict zero-ится без AttributeError.

4. Drift test:
   - если upstream marker уже есть, patch skip-ится как upstream absorbed;
   - если anchor отсутствует без upstream marker, это warning/failure class.

5. Model config smoke:
   - 27B INT4 TQ k8v4;
   - 27B DFlash true;
   - 35B FP8 DFlash;
   - 35B PROD if touched by tool parser policy.

## 7. Интеграция в model configs

### 7.1. P61c/tool parser

Для 35B PROD уже включено. Для 27B:

1. Сначала добавить в один 27B config:

```yaml
GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT: '1'
```

2. После smoke перевести на bundle:

```yaml
SNDR_ENABLE_BUNDLE_TOOL_PARSING_QWEN3CODER: '1'
```

3. Когда bundle стабилен, убрать отдельные P15/P61c/P64/PN56 flags из configs
   или оставить их только как comments/reference.

### 7.2. PN82

Не включать глобально сразу.

Добавить в configs как disabled documented flag:

```yaml
GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO: '0'
```

После smoke на hybrid CUDA graph включить там, где есть Mamba/GDN и CUDA graph:

```yaml
GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO: '1'
```

### 7.3. PN55v2

Не включать во всех configs автоматически. Это `/sleep` -> `/wake_up` management
path. Если production scripts не вызывают sleep, patch нужен как defensive guard.

Рекомендуемая политика:

- default OFF в registry;
- enable в configs, где management API sleep/wake реально используется;
- либо включать только в deployment profile, а не в model config.

## 8. Авторство и provenance

Это техническая оценка авторства, не юридическая.

| Область | Примерный вклад Sander/Genesis | Комментарий |
|---|---:|---|
| P61c mitigation | 70-80% | Local design/integration; issue/reproducer credit отдельно. |
| PN82 backport | 10-20% | Основной алгоритм из upstream #41873; Genesis вклад в packaging/tests/gating. |
| PN55v2 | 25-35% | Идея nested iterator из #41896; ценность Genesis в объединении с PN55 и безопасной интеграции. |
| PatchSpec/registry-driven engine | 85-90% | Внутренняя архитектура SNDR. |
| TextPatcher/MultiFileTransaction hardening | 80-90% | Уже существующий Genesis/SNDR infrastructure. |
| Bundles policy | 80-90% | Genesis-specific operational layer. |
| Model config policy | 90-100% | Полностью deployment/IP слой Genesis. |

Если отправлять что-то upstream:

- backport logic из чужого PR нельзя представлять как полностью свое;
- PR body должен явно ссылаться на upstream issue/PR/provenance;
- Sander авторствует integration, tests, hardening, production validation;
- Genesis-original kernels/engine pieces имеют максимальный процент авторства.

## 9. Конкретный порядок работ

### День 1

1. Создать PN82 file.
2. Добавить flag в `env.py`.
3. Добавить registry entry.
4. Добавить тесты PN82.
5. Прогнать targeted tests.

Definition of Done:

- PN82 default OFF.
- `should_apply("PN82")` работает.
- idempotency test проходит.
- semantic test проходит.

### День 2

1. Переписать PN55 в PN55v2.
2. Обновить tests.
3. Обновить registry credit.
4. Убедиться, что old flag preserved.

Definition of Done:

- list-only test заменен nested container test.
- old env flag still works.
- no separate PN83 runtime patch.

### День 3

1. Проверить P61c на 27B.
2. Добавить P61c или bundle в 27B configs.
3. Записать empirical note в config comment.

Definition of Done:

- 27B tool-call smoke passes.
- No SSE silence.
- No regression на normal text streaming.

### День 4-5

1. Ввести `PatchSpec`.
2. Добавить converter from existing registry.
3. Добавить shadow apply ordering report.
4. Проверить расхождения старого и нового apply lists.

Definition of Done:

- Новый код ничего не применяет по-новому, только сравнивает.
- Есть report: registry patches without apply_module, dispatch patches without
  registry entry.

### День 6-8

1. Перевести low-risk patches на registry-driven apply.
2. Убрать прямую зависимость от `_per_patch_dispatch.py` для новых patches.
3. Сделать `_per_patch_dispatch.py` legacy fallback.

Definition of Done:

- PN82 и PN55v2 идут через `apply_module`.
- Старые patches все еще работают.
- Boot summary показывает единый registry count.

## 10. Риски проекта

### Риск 1: drift от upstream DFlash refactor

PR #41703 большой и может изменить `dflash.py`, `qwen3_dflash.py`,
`gpu_model_runner.py`, scheduler integration.

Митигация:

- не backport сейчас;
- добавить watch note к PN21/PN24/PN38/PN40;
- после vLLM pin bump запускать drift audit до performance testing.

### Риск 2: patch order

Registry-driven loop может изменить порядок применения.

Митигация:

- добавить `phase/order`;
- сначала shadow compare;
- запрещать migration, если order diff не объяснен.

### Риск 3: слишком много флагов в model configs

Сейчас configs превращаются в длинные списки отдельных patches.

Митигация:

- stable feature groups переводить на bundles;
- individual flags оставить для debug/rollback.

### Риск 4: authorship confusion

Часть кода — upstream backports, часть — Genesis original.

Митигация:

- registry `credit` держать точным;
- для upstream PR/issue всегда сохранять provenance;
- для Genesis-original engine patches явно писать `tier=engine`.

## 11. Что будет результатом

После реализации рекомендуемого варианта B:

1. Qwen3Coder tool parsing станет стандартным и воспроизводимым across configs.
2. Hybrid/Mamba CUDA graph path получит два актуальных safety fixes.
3. PN55 не будет раздвоен с PN83.
4. Новый PR backport будет добавляться через один понятный flow:
   `env.py` -> `registry.py` -> `patch module` -> `tests` -> `model config`.
5. `_per_patch_dispatch.py` перестанет быть вторым источником истины.
6. Авторство и provenance будут зафиксированы на уровне registry и docs.

## 12. Итоговое решение

Делать:

1. PN82 как новый patch.
2. PN55v2 как upgrade existing PN55.
3. P61c/tool parser bundle policy для 27B configs.
4. PatchSpec + registry-driven apply migration.

Не делать:

1. Не делать отдельный PN83 runtime patch.
2. Не backport-ить #41703 сейчас.
3. Не тянуть KV offload/NVFP4/LoRA/model-specific PR в текущую A5000 линию.
4. Не переписывать TextPatcher/MultiFileTransaction с нуля.

Главный принцип: меньше новых параллельных механизмов, больше единого ownership
и точного provenance.

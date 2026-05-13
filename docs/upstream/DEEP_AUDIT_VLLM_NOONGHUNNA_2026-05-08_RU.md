# Глубокий аудит Genesis / SNDR Core + внешние интеграции vLLM / club-3090

Дата локального запуска: 2026-05-08  
Таймзона окружения: Europe/Kiev  
Статус: аудит и дорожная карта, исходный код проекта не изменялся  
Рабочее дерево: `$HOME/Documents/Visual Studio Code/genesis-vllm-patches`

> Важно по датам: в проекте есть локальные документы с датой `2026-05-09`
> (`docs/_internal/BACKLOG_2026-05-09.md`, `docs/upstream/PRODUCTION_ROADMAP_2026-05-09.md`).
> Я анализировал их как уже существующие локальные файлы, хотя текущая дата
> локального audit environment — 2026-05-08.

## 1. Короткий вывод

Текущая структура стала существенно лучше предыдущих вариантов: `sndr_engine`
сейчас фактически пустой reserved namespace, все существующие патчи и утилиты
находятся в `sndr_core`, `sndr` CLI уже объединяет installer, launcher,
memory/report/patches и legacy compat-команды. Это правильное направление для
public core + future private engine.

Но проект пока не production-ready. Главные реальные блокеры сейчас не
синтаксические, а связаны с рассинхронизацией тестов, конфигов, документации,
runtime-dependency gating и качеством release/bench pipeline:

| Приоритет | Проблема | Где видно |
|---|---|---|
| P0 | `pytest -q` падает на 4 тестах | `tests/legacy/test_model_config_cli.py:20`, `:30`, `:47`; `tests/legacy/test_model_config_audit_rules.py:194-201` |
| P0 | PROD config содержит `genesis_env` ключи, неизвестные R-011 audit rule | `vllm/sndr_core/model_configs/builtin/a5000-2x-35b-prod.yaml:110-113`; `vllm/sndr_core/model_configs/audit_rules.py:605-650` |
| P0 | Тесты ожидают старый TPS `231.4/231.41`, а config уже обновлен до `237.95` | YAML `:172`, тесты `:20`, `:30`, `:47` |
| P1 | `sndr install --dry-run` успешно завершается, но печатает реальные `ERROR` по torch-less imports | `_per_patch_dispatch.py:421-425`, `:464-466`, `:4324-4329`, `:4368-4373`, `:4407-4408`, `:4600` |
| P1 | registry/spec metadata неполная: `implementation_status` и `category` отсутствуют у 132/132 specs | `iter_patch_specs()` runtime inventory |
| P1 | README и часть docs устарели относительно текущего split | `README.md:23-31`, `README.md:39-41`, `README.md:90-92` |
| P1 | `PRODUCTION_ROADMAP_2026-05-09.md` уже не соответствует текущему коду | `docs/upstream/PRODUCTION_ROADMAP_2026-05-09.md:20-31`, `:57-68` |
| P1 | community sample configs все еще монтируют старый `_genesis` path | `vllm/sndr_core/model_configs/community/*.yaml.sample` |
| P2 | hardcoded адреса и локальные пути остались в bench/probe/docs | `benchmarks/harness/run_all.py:14-47`, `tests/probes/streaming_thinking_probe.py:45-49`, `assets/chat_templates/README.md:35-39` |
| P2 | bare-metal renderer все еще делает editable install plugin path | `vllm/sndr_core/compat/model_config_cli.py:582-615` |

Оценка готовности:

| Слой | Оценка |
|---|---|
| Python syntax / import-level parse | хорошо: AST clean по 557 файлам |
| `sndr_core` как публичный package | архитектурно правильный, но metadata/docs/tests не добиты |
| `sndr_engine` как future paid overlay | правильно пустой сейчас; нужно не тащить туда public код |
| Installer/launcher | usable beta; production path требует жестче отделить dev mounts от wheel mode |
| Model config system | полезный, но сейчас падает на audit rule и stale tests |
| Тесты | не зеленые: 4 failures |
| Документация | богатая, но содержит stale claims |
| Security/release | каркас есть, но нужен real trust anchor, SBOM in CI, strict license tests |
| Bench/soak/quality gates | пока недостаточно встроены в `sndr` как repeatable pipeline |

## 2. Что проверено локально

Команды и фактические результаты:

| Проверка | Результат |
|---|---|
| `git status --short` | рабочее дерево грязное; много новых `sndr_core` файлов, root `pyproject.toml` untracked, старые `_genesis` файлы удалены |
| `rg --files ... | wc -l` | 808 файлов в дереве с исключением `.git`/`__pycache__` |
| AST parse `vllm/sndr_core`, `vllm/sndr_engine`, `tests`, `scripts` | `PY_AST_CHECK count=557 errors=0` |
| `python3 -m vllm.sndr_core.apply.shadow --strict` | clean: 129 legacy registrations, 132 specs, 119 specs with `apply_module`, 13 without |
| `python3 -m vllm.sndr_core.compat.schema_validator --quiet` | без errors; warnings по `research_note`: P82, P83, PN26b |
| `python3 -m vllm.sndr_core.compat.lifecycle_audit_cli --quiet` | warnings only: 90 experimental, 5 retired |
| `python3 -m vllm.sndr_core.cli --help` | top-level CLI уже содержит `install`, `launch`, `memory`, `patches`, `report` и bridged compat-команды |
| `python3 -m vllm.sndr_core.cli launch --dry-run --non-interactive a5000-2x-35b-prod` | exit 0; dry-run показывает unresolved mounts, live launch должен блокировать |
| `python3 -m vllm.sndr_core.cli install --dry-run --non-interactive` | exit 0, но smoke summary содержит `failed=6`, в логах видны import errors без torch |
| `pytest -q` | `4 failed, 2440 passed, 71 skipped in 22.30s` |

Вывод: синтаксически проект чистый, но behavioral gate не пройден, потому что
pytest красный и dry-run installer все еще шумит ошибками, которые должны быть
классифицированы как expected skip в no-torch среде.

## 3. Архитектура проекта: что сейчас делает код

### 3.1. `vllm/sndr_core`

Фактическая роль `sndr_core`:

- registry и spec model для всех public patches;
- apply orchestrator и legacy `_per_patch_dispatch.py`;
- CLI: install, launch, memory, patches, report плюс bridged legacy compat;
- model-config schema, renderer, preflight/audit rules;
- license gate и future engine overlay awareness;
- runtime kernels/helpers;
- structured patch taxonomy по папкам `patches/attention`, `patches/spec_decode`, `patches/quantization`, `patches/middleware`, `patches/moe`, `patches/kernels`;
- docs/tools integration points.

Это правильное место для бесплатной public части. На текущем этапе проекта
платная версия еще рано: public core должен стать надежным, удобным и полезным,
чтобы закрепить доверие. Private engine стоит держать пустым до появления
действительно новых закрытых разработок, которых нет в public GitHub и нет в PR
upstream.

### 3.2. `vllm/sndr_engine`

Текущее состояние правильное:

- `vllm/sndr_engine/__init__.py:4-16` прямо говорит, что public package пустой;
- `engine_available()` возвращает `False`, если нет будущего private overlay:
  `vllm/sndr_engine/__init__.py:45-59`;
- `pyproject-engine.toml:10-16` фиксирует, что PN72 и helper перенесены в core;
- `pyproject-engine.toml:80-85` оставляет future entry point group
  `sndr.engine.overlay`.

Рекомендация: не переносить текущие public разработки в engine. Engine должен
оставаться пустым до появления реального приватного overlay. Core может знать о
наличии engine через optional discovery, но не должен от него зависеть.

### 3.3. CLI

Состояние лучше, чем в старом roadmap:

- `vllm/sndr_core/cli/__init__.py:49-53` подключает native `install`, `launch`,
  `memory`, `patches`, `report`;
- `vllm/sndr_core/cli/__init__.py:61-81` содержит bridge для `doctor`,
  `verify`, `self-test`, `model-config`, `lifecycle-audit`, `validate-schema`,
  `explain`, `list-models`, `categories`, `plugins`, `telemetry`,
  `update-channel`, `preflight`, `bench`, `migrate`, `recipe`, `preset`,
  `init`, `pull`;
- `vllm/sndr_core/cli/__init__.py:125-131` fast-path делегирует bridged
  команды в compat CLI.

Это закрывает старую претензию “top-level `sndr` показывает только install/launch”.
Но нужно проверить UX команд не только `--help`, а реальные сценарии:
`sndr report bundle`, `sndr memory explain`, `sndr patches plan`,
`sndr bench compare`, `sndr verify --stress`.

## 4. Текущие реальные ошибки и несостыковки

### P0-1. `pytest` падает из-за неизвестных env keys в PROD config

Файлы:

- `vllm/sndr_core/model_configs/builtin/a5000-2x-35b-prod.yaml:110-113`
- `vllm/sndr_core/model_configs/audit_rules.py:397-408`
- `vllm/sndr_core/model_configs/audit_rules.py:605-650`
- `tests/legacy/test_model_config_audit_rules.py:193-201`

Проблема:

`a5000-2x-35b-prod.yaml` включает новые ключи:

- `GENESIS_PN16_TOOL_THINK_BUDGET`
- `GENESIS_PN16_CLASSIFIER_MAX_TOKENS`
- `GENESIS_OBSERVABILITY`

R-011 строит allowlist из `PATCH_REGISTRY.env_flag` и `tunable_prefixes`.
Префиксы `GENESIS_PN16_` и `GENESIS_OBSERVABILITY` сейчас не разрешены:
`audit_rules.py:620-631`. В результате builtin PROD config получает error,
а тест ожидает `errors == []`.

Что сделать:

1. Добавить `GENESIS_PN16_` и `GENESIS_OBSERVABILITY` в разрешенный список
   tunable/system Genesis keys, если это действительно config knobs.
2. Либо зарегистрировать их в metadata конкретного PN16/observability patch.
3. Добавить unit-тест на новые ключи, чтобы следующий Wave не ломал R-011.

Рекомендую вариант 2 как более правильный: registry должен знать не только
`env_flag`, но и `env_knobs`, чтобы config audit не разрастался hardcoded
prefix-list.

### P0-2. `pytest` падает из-за stale TPS expectations

Файлы:

- `vllm/sndr_core/model_configs/builtin/a5000-2x-35b-prod.yaml:164-183`
- `tests/legacy/test_model_config_cli.py:20`
- `tests/legacy/test_model_config_cli.py:30`
- `tests/legacy/test_model_config_cli.py:47`

Факт:

Config уже обновлен:

- `long_gen_sustained_tps: 237.95` на `a5000-2x-35b-prod.yaml:172`
- `genesis_pin: v11.0.0+wave7` на `:182`
- `vllm_pin: 0.20.2rc1.dev93+g51f22dcfd` на `:183`

Но тесты все еще ждут `231.4` / `231.41`:

- `test_model_config_cli.py:20`
- `test_model_config_cli.py:30`
- `test_model_config_cli.py:47`

Что сделать:

1. Обновить ожидаемые значения на `237.95` или лучше не хардкодить точное TPS
   в CLI snapshot-тестах.
2. Правильнее проверять, что CLI выводит значение из YAML, а не фиксированную
   старую цифру. Тогда bench update не ломает unit tests.
3. Для regression thresholds держать отдельный bench-golden файл с tolerances,
   а не смешивать UI CLI tests и perf assertions.

### P1-1. Installer dry-run маскирует no-torch import failures как нормальный skip

Файл:

- `vllm/sndr_core/apply/_per_patch_dispatch.py`

Проблемные точки:

- P31 импортирует `router_softmax` внутри try и возвращает failed при
  `No module named 'torch'`: `:421-425`.
- P22 импортирует `dequant_buffer` до platform skip: `:461-466`.
- P32/P33 ловит import failure и возвращает failed: `:4324-4329`.
- P28 ловит import failure и возвращает failed: `:4368-4373`.
- P7 импортирует `gdn_dual_stream` до no-torch-safe skip: `:4407-4408`.
- P20 импортирует `tq_continuation_prefill` без try до guard: `:4600`.

Симптом:

`python3 -m vllm.sndr_core.cli install --dry-run --non-interactive` завершается
успешно, но в smoke summary есть `failed=6`, а лог содержит `ERROR`/exceptions
из-за отсутствия torch/triton/vLLM runtime на Mac/no-GPU host.

Почему это важно:

Dry-run installer должен быть чистым operator UX. Если отсутствие torch на
control host ожидаемо, такие патчи должны возвращать `skipped` до импорта
torch-heavy модулей. Сейчас пользователь видит красные ошибки и не может
отличить реальные поломки от expected environment gap.

Что сделать:

1. Ввести helper `optional_runtime_import()` или decorator
   `requires_runtime("torch")`.
2. Для torch-heavy patches сначала выполнять lightweight platform/runtime probe,
   затем импортировать kernel module.
3. В dry-run/no-torch среде возвращать `skipped: torch runtime unavailable`,
   не `failed`.
4. CI-gate: `sndr install --dry-run --non-interactive` должен иметь `failed=0`
   на Mac/no-GPU host.

### P1-2. Registry specs не имеют обязательных полей качества

Runtime inventory:

```text
spec_count 132
tier: community=132
lifecycle: experimental=90, legacy=33, retired=5, research=3, coordinator=1
implementation_status: <missing>=132
category: <missing>=132
engine_ids: []
missing_apply_module: PN26b, P1, P17, P18b, P20, P23, P29, P32, P51, P102, PN60, PN63, PN64
```

Проблема:

`tier` и `lifecycle` уже есть, но `implementation_status` и `category`
отсутствуют у всех specs. Это делает `sndr patches list/plan/report` менее
полезным: нельзя надежно разделить bugfix/perf/research/stub/retired/metadata
уровнем данных.

Что сделать:

1. Добавить обязательные поля:
   - `category`: `memory`, `spec_decode`, `structured_output`, `quantization`,
     `gdn`, `moe`, `launcher`, `security`, `observability`, `research`;
   - `implementation_status`: `live`, `text_patch`, `runtime_hook`,
     `metadata_only`, `retired`, `research`, `blocked`, `upstream_merged`;
   - `source`: `genesis_original`, `vllm_pr_backport`, `club_3090_adapted`,
     `cross_engine_research`.
2. Сделать schema validator warning сейчас, error для новых патчей после cutoff.
3. Использовать эти поля в `sndr patches plan` и release notes.

### P1-3. README противоречит текущему состоянию engine

Файл:

- `README.md:19-35`
- `README.md:83-92`
- `README.md:39-41`

Проблема:

README все еще говорит:

- `sndr_core` = 130 community patches;
- `sndr_engine` = commercial tier, 1 PN72 patch + private kernel helper;
- pytest baseline `2425 → 2621`;
- patch coverage 131 entries.

Текущее состояние по проверкам:

- `iter_patch_specs()` показывает 132 specs;
- все specs `tier=community`;
- `sndr_engine` skeleton и `engine_available() == False`;
- PN72 moved to core;
- фактический pytest: `2440 passed`, `4 failed`, `71 skipped`.

Что сделать:

1. Обновить README под реальное состояние: public repo сейчас содержит только
   community tier, engine namespace reserved/empty.
2. Убрать claims про “commercial tier, 1 PN72 patch”.
3. Автоматизировать badges/patch count/test count из CI artifacts, а не руками.

### P1-4. `PRODUCTION_ROADMAP_2026-05-09.md` уже stale

Файл:

- `docs/upstream/PRODUCTION_ROADMAP_2026-05-09.md:20-31`
- `docs/upstream/PRODUCTION_ROADMAP_2026-05-09.md:57-68`

Проблема:

Roadmap говорит, что:

- installer dry-run падает из-за P4/P5;
- pytest падает на 2 тестах;
- root package deps пустой;
- top-level `sndr` содержит только `install`/`launch`;
- Docker launcher зависит от `/plugin` install.

Текущее состояние другое:

- P4/P5 уже не являются текущими failures;
- pytest падает на 4 других тестах;
- `sndr` CLI уже расширен;
- Docker bootstrap больше не делает plugin install без `SNDR_DEV_INSTALL_PLUGIN=1`
  (`schema.py:964-1004`);
- но builtin YAML все еще содержит `${plugin_src}:/plugin:ro`,
  например `a5000-2x-35b-prod.yaml:154-160`.

Что сделать:

1. Считать этот файл историческим snapshot, а не source of truth.
2. Текущий source of truth перенести в этот отчет или обновить отдельным PR.
3. Добавить в roadmap “last verified commands” с датой и exact output.

### P1-5. Community sample configs все еще монтируют `_genesis`

Файлы:

- `vllm/sndr_core/model_configs/community/gemma-4-26b-a4b-awq.yaml.sample:163`
- `vllm/sndr_core/model_configs/community/EXAMPLE_symbolic_mounts.yaml.sample:130`

Проблема:

Samples используют:

```text
${genesis_src}:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro
```

После перехода на `sndr_core` это вводит в заблуждение. Builtin configs уже
монтируют `vllm/sndr_core`, а samples отстали.

Что сделать:

1. Обновить samples на `vllm/sndr_core`.
2. Если legacy shim нужен, явно назвать его legacy-only и не ставить в примеры.
3. Добавить `rg`-gate в CI: public docs/samples не должны рекомендовать
   `_genesis` как основной путь.

### P2-1. Остались hardcoded адреса и локальные пути

Файлы:

- `benchmarks/harness/run_all.py:14-16`, `:43-47`
- `tests/probes/streaming_thinking_probe.py:45-49`
- `assets/chat_templates/README.md:35-39`
- `tools/genesis_bench_suite.py:1078-1099`

Проблемы:

1. `benchmarks/harness/run_all.py` по умолчанию использует
   `http://<host>:8000/v1`.
2. `tests/probes/streaming_thinking_probe.py` по умолчанию использует
   `http://<host>:8000/v1/chat/completions`.
3. `assets/chat_templates/README.md` показывает `$HOME/...` как путь.
4. `tools/genesis_bench_suite.py:1089` вызывает старый
   `vllm._genesis.compat.cli`, хотя canonical CLI теперь `vllm.sndr_core`.

Что сделать:

1. Defaults должны быть `http://127.0.0.1:8000/v1` или отсутствовать с явной
   ошибкой “set GENESIS_BENCH_ENDPOINT”.
2. Локальные пути заменить на `${SNDR_REPO}` / `$(pwd)` / `sndr paths`.
3. `tools/genesis_bench_suite.py` перевести на
   `python3 -m vllm.sndr_core.compat.cli self-test --json`.

### P2-2. Bare-metal renderer все еще тянет plugin editable install

Файл:

- `vllm/sndr_core/compat/model_config_cli.py:568-615`

Проблема:

Bare-metal script:

- берет `genesis_src` и `plugin_src` из host config;
- добавляет `PYTHONPATH`;
- делает `pip install --quiet -e {plugin_src} 2>/dev/null || true`.

Это dev-friendly, но production-нестрого: ошибка install скрывается `|| true`,
а plugin path может быть нерешенным placeholder.

Что сделать:

1. Ввести режимы:
   - `--runtime bare_metal --mode wheel`: требует установленный wheel, не делает
     editable install;
   - `--mode dev`: разрешает editable install и source path.
2. Убрать `|| true` из production path.
3. Preflight должен явно проверять, что выбранный mode может загрузить
   `vllm.sndr_core.plugin:register`.

## 5. Backlog 2026-05-09: что добавить и что исправить

Файл:

- `docs/_internal/BACKLOG_2026-05-09.md`

Что в нем полезно и нужно сохранить:

- Wave 7 PN16 V8 + observability как важный текущий direction:
  `BACKLOG_2026-05-09.md:24-29`;
- Priority 1 boost sweeps P67/P82/matmul/max_num_batched_tokens:
  `:36-44`;
- patcher infrastructure: CI regression bench, spec-driven dispatch,
  decode_TPOT reporting: `:45-52`;
- conflict/dependency resolver, apply contract tests, bench compare,
  CUDA graph hit-rate, PROD YAML audit: `:55-66`;
- security hardening: real Ed25519, SBOM, plugin sig gate, GPU CI:
  `:147-156`;
- CLI roadmap: models, bench, memory explain phases: `:159-168`;
- bench/soak matrix: `:171-178`;
- club-3090 issues section: `:182-189`.

Что устарело в backlog:

- `Test totals: 1919 unit pass` на `:30` уже не соответствует текущему
  локальному `pytest` result (`2440 passed`, `4 failed`, `71 skipped`);
- “No-stubs violations closed” верно по смыслу, но нужно добавить новый gate:
  no-torch dry-run must not emit failed patches;
- club-3090 section слишком узкая: нужно добавить #72, #82, #85, #91, #93,
  #95, #97, #103 и Gemma 4 INT8/DFlash данные.

Что нужно добавить в backlog:

1. P0 stabilization sprint:
   - fix R-011 env knobs;
   - update stale TPS tests;
   - no-torch dry-run failed=0;
   - update README and production roadmap stale claims.
2. Gemma 4 sprint:
   - port Gemma4 parser fixes from PR #42006/#41991;
   - add Gemma4 INT8 PTH config;
   - add Gemma4 DFlash config gated on #42102/#42069.
3. Memory/KV sprint:
   - backport/evaluate #42102;
   - port `club-3090/tools/kv-calc.py` into `sndr memory explain`;
   - port residency instrumentation from old `_genesis` imports to
     `sndr_core`.
4. Report/verify sprint:
   - port `club-3090/scripts/report.sh` behavior into `sndr report bundle`;
   - port `verify-stress.sh` into `sndr verify --stress`.

## 6. vLLM PR за 2026-05-08: что важно для Genesis

GitHub search по `repo:vllm-project/vllm type:pr created:2026-05-08`
показал 73 PR. Ниже не полный список, а технически релевантные для проекта.

| PR | Что делает | Значение для Genesis |
|---|---|---|
| [#42102](https://github.com/vllm-project/vllm/pull/42102) | DFlash drafter + quantized target KV через независимые KV groups + dtype override | Высокий приоритет. Прямо бьет в DFlash/Gemma4/INT8 PTH/Ampere |
| [#42069](https://github.com/vllm-project/vllm/pull/42069) | DFlash drafter backend autoselect на Gemma 4 | Высокий. Нужен для Gemma4 DFlash стабильности |
| [#42105](https://github.com/vllm-project/vllm/pull/42105) | Gemma4 reasoning batch chat completions fix | Высокий для Gemma4 agent traffic |
| [#42006](https://github.com/vllm-project/vllm/pull/42006) | Gemma4 MTP streaming multi-tool calls | Высокий, перекликается с club Gemma4 parser overlay |
| [#42070](https://github.com/vllm-project/vllm/pull/42070) | Remove nested torch.compile in GDN rearrange_mixed_qkv | Высокий для GDN/cudagraph stability |
| [#42076](https://github.com/vllm-project/vllm/pull/42076) | GDN KKT precision loss на Hopper WGMMA alignment | Medium/High, особенно если добавлять Hopper profiles |
| [#42074](https://github.com/vllm-project/vllm/pull/42074) | Reset offloader singleton in shutdown to prevent GPU memory leak | High для долгоживущих серверов |
| [#42086](https://github.com/vllm-project/vllm/pull/42086) | bounded early prefetch for waiting requests | Medium/High для KV connector/perf |
| [#42050](https://github.com/vllm-project/vllm/pull/42050) | kv_offload request_finished + store policy decouple | Medium/High для будущего KV offload |
| [#42044](https://github.com/vllm-project/vllm/pull/42044) | token-budget-bounded early KV prefetch | Medium/High, memory-aware scheduling |
| [#42095](https://github.com/vllm-project/vllm/pull/42095) | FlexAttention/FlashAttention num-blocks-first layouts | Medium, attention backend performance |
| [#42097](https://github.com/vllm-project/vllm/pull/42097) | NIXL HMA transfer kernel/logical block mismatch | Medium для disaggregated/KV transfer |
| [#42080](https://github.com/vllm-project/vllm/pull/42080) | FP8 per-tensor Q scale in Triton attention | Medium для quantized attention |
| [#42089](https://github.com/vllm-project/vllm/pull/42089) | FlashInfer CUTLASS MXFP4-MXFP8 MoE swizzled scale | Medium для FP8/MoE stack |
| [#42030](https://github.com/vllm-project/vllm/pull/42030) | AutoRound GPTQ routing for group-misaligned TP shards | High для AutoRound/Gemma/Qwen configs |
| [#42029](https://github.com/vllm-project/vllm/pull/42029) | Gemma4 AutoRound/GPTQ quantized router and packed MoE weights | High для Gemma4 INT4 |
| [#42028](https://github.com/vllm-project/vllm/pull/42028) | Gemma4 per-layer embedding dtype/device on quantized checkpoints | High для Gemma4 reliability |
| [#42022](https://github.com/vllm-project/vllm/pull/42022) | INT8 GPTQ MoE to WNA16 fallback | Medium для quant fallback |
| [#42015](https://github.com/vllm-project/vllm/pull/42015) | scheduler priority queue lazy deletion | Medium, scheduler perf |
| [#42101](https://github.com/vllm-project/vllm/pull/42101) | OpenAI API pre-serve warmup | Useful для launcher/warmup UX |
| [#42064](https://github.com/vllm-project/vllm/pull/42064) | auth bypass on `/inference/v1/generate` | Security watch; проверить, есть ли аналогичные routes у Genesis proxy |

### 6.1. Deep dive: vLLM PR #42102

Ссылка: [vllm-project/vllm#42102](https://github.com/vllm-project/vllm/pull/42102)

Название PR:

```text
[Spec Decode] Allow DFlash drafter to coexist with quantized target KV via independent KV groups + dtype override
```

Ключевая идея:

Gemma/Qwen DFlash drafter может требовать BF16/auto KV, а target model хочет
quantized KV (`int8_per_token_head`, `fp8_per_token_head`). Старый vLLM пытается
унифицировать page size для всех layers, из-за чего DFlash + quant target KV
ломается. PR разделяет DFlash draft layers в отдельные KV groups и дает drafter
локальный dtype override.

Измененные файлы PR:

- `vllm/v1/core/kv_cache_utils.py`
- `vllm/model_executor/models/qwen3_dflash.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `tests/v1/core/test_kv_cache_utils.py`

Что PR дает по данным автора:

- 2× RTX 3090 Gemma 4 31B + DFlash + INT8 PTH KV boot healthy;
- narrative TPS около 95.89;
- code TPS около 168.09;
- 32K NIAH pass;
- KV pool около 149k tokens против около 38k baseline;
- max context около 65K против 32K;
- VRAM около 23.85 GB/card.

Критичный review point:

В review PR был замечен риск широкого `try/except Exception` вокруг
`get_num_layers`: если DFlash включен, ошибка конфигурации не должна тихо
возвращать старый unify path. Для Genesis backport это обязательно:

- при включенном DFlash strict fail лучше silent fallback;
- fallback можно оставить только если DFlash точно выключен.

Решение для Genesis:

1. Создать отдельный patch/backport item, например `PN104_DFLASH_QUANT_KV_GROUPS`.
2. Не смешивать его с существующими PN21-PN24/PN38/PN40 без явного migration plan.
3. Добавить tests:
   - DFlash layers partitioned;
   - local page size bytes для isolated group;
   - target quant KV + BF16 drafter;
   - strict failure если layer introspection сломан при enabled DFlash.
4. Добавить model configs:
   - Gemma4 MTP INT8 PTH;
   - Gemma4 DFlash INT8 PTH;
   - A5000/3090 profiles отдельно.

## 7. noonghunna / club-3090: что взять в проект

Источники:

- [noonghunna/club-3090](https://github.com/noonghunna/club-3090)
- [BENCHMARKS.md](https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md)
- [scripts/report.sh](https://github.com/noonghunna/club-3090/blob/master/scripts/report.sh)
- [scripts/verify-stress.sh](https://github.com/noonghunna/club-3090/blob/master/scripts/verify-stress.sh)
- [tools/kv-calc.py](https://github.com/noonghunna/club-3090/blob/master/tools/kv-calc.py)

### 7.1. Bench database и contribution process

`club-3090/BENCHMARKS.md:1-15` задает хороший стандарт: все цифры measured,
append-friendly, каждая строка привязана к rig/date. `:19-29` фиксирует
canonical prompts/sampling, что делает результаты между пользователями
сравнимыми.

Что взять:

1. Создать в Genesis `docs/BENCHMARKS_COMMUNITY.md` или расширить текущий
   `docs/BENCHMARKS.md` по формату:
   - rig;
   - GPU topology;
   - power cap;
   - image SHA;
   - Genesis pin;
   - vLLM pin;
   - config key;
   - verify-full/stress/soak status.
2. Добавить issue template “Numbers from your rig”.
3. `sndr report bundle --bench` должен генерировать paste-ready markdown,
   который можно вставить в issue без ручной сборки.

### 7.2. `scripts/report.sh` как образец для `sndr report bundle`

`club-3090/scripts/report.sh:1-27` делает именно то, что нужно operator tool:
hardware, OS, GPU, runtime, stack version, active container state, optional
verify/stress/soak/bench, redaction by default.

Особенно полезные элементы:

- `--full` как canonical full pass: `report.sh:9-18`;
- отдельный `--soak`, потому что verify/bench могут пройти, а continuous
  multi-turn workload ловит Cliff 2b: `report.sh:20-24`;
- redaction paths/user/tokens: `report.sh:26-27`, `:66-80`;
- GPU hardware section с power cap / PCIe lane width:
  `report.sh:199-220`.

Что сделать в Genesis:

1. `sndr report bundle --full`:
   - markdown report;
   - redacted tarball;
   - `nvidia-smi topo -m`, `nvidia-smi topo -p2p r`;
   - Docker/Podman info;
   - active vLLM image digest;
   - `sndr patches list --json`;
   - model config rendered script;
   - verify/stress/soak/bench results.
2. `--no-redact` только явно.
3. Включить report artifact в bug template.

### 7.3. `verify-stress.sh` как обязательный quality gate

`club-3090/scripts/verify-stress.sh:1-39` содержит правильную философию:
не только короткий smoke, а boundary tests для KV-cache и prefill activation
memory.

Проверки:

- long-context needle small rungs: `:16-20`;
- tool response prefill OOM: `:21-23`;
- IDE-agent prompt shape: `:24-26`;
- multi-turn agent: `:27-28`;
- LCB coding shape: `:29-30`;
- reasoning-heavy: `:31-33`;
- long-context needle large rungs last, чтобы не убить engine раньше:
  `:34-38`;
- engine detection vLLM/llama.cpp/SGLang: `:77-114`.

Что сделать в Genesis:

1. Добавить `sndr verify --stress`.
2. Разделить:
   - `sndr verify --smoke` до 2 минут;
   - `sndr verify --stress` 5-10 минут;
   - `sndr verify --soak` 30+ минут;
   - `sndr verify --full` все вместе.
3. Сделать engine-aware diagnostics, не vLLM-only.
4. Считать stress fail blocker для stable configs.

### 7.4. `kv-calc.py` как основа `sndr memory explain`

`club-3090/tools/kv-calc.py:1-31` уже делает то, что нужно Genesis memory UX:
predict weights, KV pool, GDN activation peak, cudagraph/workspace overhead,
verdict PASS/TIGHT/FAIL.

Полезные детали:

- Qwen3.6-27B spec: `kv-calc.py:42-60`;
- KV format bytes: `:62-73`;
- GDN activation coefficients: `:75-89`;
- compose presets: `:91-106`;
- measured calibration rows: `:108-122`;
- formula per-card KV pool: `:139-161`;
- GDN activation estimate: `:164-180`.

Что сделать:

1. Перенести формулы в `vllm/sndr_core/cli/memory.py` или отдельный
   `vllm/sndr_core/memory/estimator.py`.
2. Сделать profile registry:
   - Qwen3.6-27B;
   - Qwen3.6-35B-A3B;
   - Gemma4-31B;
   - Qwen3-Coder;
   - Nemotron 30B A3B позже.
3. Поддержать KV formats:
   - fp16/bf16;
   - fp8_e5m2/e4m3;
   - int8_per_token_head;
   - turboquant_k8v4 / TQ3;
   - q4_0/k8v4 для llama.cpp reference.
4. Добавить compare actual:
   - predicted vs `nvidia-smi`;
   - predicted vs vLLM KV blocks;
   - predicted vs Genesis pools.

### 7.5. Residency instrumentation

`club-3090/tools/residency-instrument/instrument.py:1-8` — observational
sitecustomize-style instrumentation. Оно не меняет scheduling/memory policy,
только пишет CSV snapshots.

Что полезно:

- поля для KV blocks, Genesis pools, MTP resident, FlashInfer workspace,
  cudagraph private, fragmentation: `instrument.py:40-84`;
- request/turn summary fields: `:86-113`;
- safe CSV writing with locks: `:156-177`.

Проблема для прямого переноса:

Инструмент все еще импортирует старые `_genesis` модули:

- `instrument.py:260`;
- `instrument.py:273`;
- `instrument.py:295-319`.

Что сделать:

1. Переписать imports на `vllm.sndr_core`.
2. Подружить с текущим `vllm/sndr_core/runtime/memory_metrics.py`.
3. Добавить `sndr report bundle --residency-log`.
4. Использовать в long soak для обнаружения:
   - KV growth;
   - cudagraph private growth;
   - fragmentation reserved-unallocated;
   - Genesis pools drift.

### 7.6. Gemma 4: что конкретно взять

`club-3090` содержит ценные Gemma4 данные:

- Gemma4 MTP BF16 KV: `BENCHMARKS.md:210-217`;
- Gemma4 MTP INT8 PTH: `BENCHMARKS.md:218-219`;
- Gemma4 DFlash BF16 KV: `BENCHMARKS.md:220`;
- Gemma4 INT8 PTH compose: `models/gemma-4-31b/vllm/compose/docker-compose.gemma-mtp-int8.yml:1-80`;
- Gemma4 DFlash compose: `models/gemma-4-31b/vllm/compose/docker-compose.gemma-dflash.yml:1-56`;
- DFlash n-sweep: `docker-compose.gemma-dflash.yml:169-180`;
- Gemma4 parser overlay:
  `models/gemma-4-31b/vllm/patches/vllm-gemma4-tool-parser-fixes/tool_parsers/gemma4_tool_parser.py`.

Рекомендации:

1. Добавить `sndr_core/model_configs/builtin/gemma4-31b-2x3090-mtp-int8.yaml.sample`
   и A5000 variant.
2. Добавить Gemma4 parser patch family:
   - adjust_request для ChatCompletionRequest и ResponsesRequest;
   - special token buffering;
   - streaming multi-tool;
   - parser bounds tests.
3. DFlash для Gemma4 не включать в stable до backport/evaluate #42102 и #42069.
4. DFlash `num_speculative_tokens` сделать workload-dependent:
   - n=5 chat/narrative;
   - n=7 code;
   - n=8 не использовать, dominated.

### 7.7. club-3090 issues: уроки для Genesis

| Issue | Что важно | Что делать в Genesis |
|---|---|---|
| [#103](https://github.com/noonghunna/club-3090/issues/103) | Gemma4 AWQ/INT4 + assistant MTP, full 262K, power-cap sweep | добавить Gemma4 AWQ/AutoRound configs + power cap capture |
| [#95](https://github.com/noonghunna/club-3090/issues/95) | patched P2P дает DFlash-noviz +22%/+19% | `sndr doctor topology` должен объяснять P2P/NVLink impact |
| [#58](https://github.com/noonghunna/club-3090/issues/58) | single 3090 Cliff 2b multi-turn OOM | `sndr verify --soak` обязателен для single-card GDN configs |
| [#57](https://github.com/noonghunna/club-3090/issues/57) | xgrammar schema unsupported keys, P68/PN70 relevance | keep PN70 + schema subset filter tests |
| [#72](https://github.com/noonghunna/club-3090/issues/72) | qwen3coder SSE silence on literal `<tool_call>` prose | structured output parser regression test |
| [#82](https://github.com/noonghunna/club-3090/issues/82) | missing `GENESIS_ENABLE_PN34_WORKSPACE_LOCK_RELAX=1` breaks runtime decode | PROD YAML audit: enabled env must cause patch to fire |
| [#85](https://github.com/noonghunna/club-3090/issues/85) | llama.cpp MTP 131K passes stress | cross-engine harness should not be vLLM-only |
| [#94](https://github.com/noonghunna/club-3090/issues/94) | llama.cpp MTP 164K ctx, code lower than havenoammo | benchmark matrix must include quality/TPS/context tradeoff |
| [#97](https://github.com/noonghunna/club-3090/issues/97) | clients hang when only `reasoning_content`, no `content` | streaming/reasoning compatibility probe |
| [#50](https://github.com/noonghunna/club-3090/issues/50) | WSL2 CUDA device-not-ready/TDR, not normal OOM | WSL/TDR detector in `sndr doctor` |
| [#47](https://github.com/noonghunna/club-3090/issues/47) | 20GB 3080: TurboQuant worse, fp8 works | model-config recommender must be VRAM-class-aware |
| [#37](https://github.com/noonghunna/club-3090/issues/37) | host bind mounts made setup non-turnkey | Genesis production launch must prefer wheel/image mode |

## 8. Cross-engine идеи, которые стоит изучить

### 8.1. llama.cpp MTP

Из club data:

- 2× V100 llama.cpp MTP: 100K ctx, verify-stress pass;
- 1× 3090 llama.cpp MTP: 131K и 164K stress-passing варианты.

Что взять:

- не переносить код напрямую, но взять architecture lesson:
  static allocator + less JIT/Triton shape churn avoids vLLM Cliff 2b;
- добавить `sndr compare-engine` docs: когда пользователю лучше vLLM, а когда
  llama.cpp;
- harness должен работать с vLLM/llama.cpp/SGLang одинаково.

### 8.2. SGLang / HiCache / radix cache

Из backlog:

- SGLang HiCache;
- fused GDN gating;
- `<think>` strip from radix cache.

Рекомендация:

Не делать большой cache integration сейчас, пока prefix-cache в PROD выключен
из-за TQ k8v4 + spec-decode crash. Но нужно:

1. Держать research branch.
2. Сначала закрыть compatibility с prefix-cache.
3. Сделать microbench: cache hit rate, memory saved, quality impact.

### 8.3. LMCache / KV offload

vLLM PR #42050/#42086/#42044 показывают, что upstream двигается в сторону
более аккуратного KV connector/offload lifecycle.

Что делать:

1. Не писать свой тяжелый KV offload раньше времени.
2. Следить за native connector API.
3. Добавить Genesis wrapper только вокруг policy/diagnostics:
   - `request_finished`;
   - store policy;
   - eviction reason;
   - memory budget.

### 8.4. PFlash

club-3090 `BENCHMARKS.md:139-193` показывает честный статус:

- standalone compression до 131K source на 1×3090 — сильный signal;
- integrated OpenAI server path `verify-stress` 0/7;
- arbitrary short needle может выпадать;
- 3-й compress cycle может падать CUDA illegal memory access.

Рекомендация:

Не добавлять PFlash в stable. Держать как research:

1. long-context QA harness: RULER/LongBench/multi-needle;
2. daemon stability 100 cycles;
3. retrieval preservation under arbitrary needles;
4. only after that `sndr pflash` experimental.

## 9. Рекомендуемая структура проекта

Цель: верхний уровень репозитория не должен быть runtime dependency для
установленного package, кроме разрешенных docs/scripts/tests/assets. Runtime
должен жить в `vllm/sndr_core` и optional private overlay.

Рекомендуемая структура:

```text
vllm/
  sndr_core/
    apply/
    cli/
    compat/
    dispatcher/
    model_configs/
    patches/
    kernels/
    runtime/
    security/
    memory/
    report/
  sndr_engine/
    __init__.py
    version.py
    LICENSE-NOTICE

scripts/
  dev/
  release/
  bench/
  ops/

tools/
  license/
  sbom/
  external_probe/
  migration/

benchmarks/
  harness/
  results/        # gitignored or artifact-only

docs/
  upstream/
  _internal/
  operations/
  model_configs/

assets/
tests/
```

Что перенести/нормализовать:

1. `scripts` разделить по назначению:
   - `scripts/dev`: локальная разработка;
   - `scripts/bench`: benchmark runners;
   - `scripts/ops`: server/admin;
   - `scripts/release`: build/SBOM/signing.
2. Runtime helpers из top-level `tools` не должны импортироваться production
   path. Если нужно runtime — переносить в `vllm/sndr_core`.
3. Старые `_genesis` references:
   - оставить только явно legacy/backcompat shims;
   - docs/samples/bench scripts перевести на `sndr_core`.
4. Public core:
   - все существующие патчи, включая PN72;
   - launcher/config/report/memory/doctor;
   - signed license checker as dormant infrastructure.
5. Private engine:
   - пока пустой;
   - future overlay через entry point;
   - никаких public dependencies on engine.

## 10. Production readiness: что требуется до релиза

### P0 gates

1. `pytest -q` green.
2. `python3 -m vllm.sndr_core.cli install --dry-run --non-interactive`:
   - exit 0;
   - `failed=0`;
   - no scary `ERROR` для expected no-torch environment.
3. `python3 -m vllm.sndr_core.apply.shadow --strict` clean.
4. `schema_validator --quiet` no errors; research warnings either accepted or fixed.
5. `lifecycle_audit_cli --quiet` no missing lifecycle.
6. README matches current code.
7. `sndr launch --dry-run` renders clean script in wheel mode and dev mode separately.

### P1 release gates

1. Build wheel for `vllm-sndr-core`.
2. Install wheel into clean venv, import without repo checkout.
3. Container image smoke:
   - no source mount;
   - no `/plugin` editable install;
   - `python3 -m vllm.sndr_core.apply` works from installed wheel.
4. SBOM generated and attached to release.
5. `KNOWN_GOOD_IMAGES` enforced for production presets.
6. Real Ed25519 public key inserted after offline keygen ceremony.
7. Tests run once with `SNDR_ALLOW_LEGACY_LICENSE_KEYS` disabled.

### P2 quality gates

1. `sndr verify --stress`.
2. `sndr verify --soak`.
3. Per-patch quality tests for structured output/tool-call patches.
4. PROD YAML audit: every enabled patch must either apply or have an explicit
   expected skip reason.
5. Regression bench:
   - decode_TPOT;
   - wall TPS;
   - TTFT;
   - accept rate;
   - tool-call pass rate;
   - VRAM delta.

## 11. Дорожная карта работ

### Sprint 0: стабилизация текущего дерева

1. Исправить R-011 env knobs:
   - `GENESIS_PN16_*`;
   - `GENESIS_OBSERVABILITY`.
2. Обновить stale TPS tests или заменить на YAML-driven assertions.
3. Сделать no-torch dry-run clean:
   - P31/P22/P32/P28/P7/P20 skip вместо failed.
4. Обновить README:
   - engine пустой;
   - patch count;
   - test status;
   - убрать PN72 commercial claim.
5. Обновить community sample mounts `_genesis` → `sndr_core`.
6. Перевести `tools/genesis_bench_suite.py` на `vllm.sndr_core`.

### Sprint 1: production launcher/package

1. Разделить launch modes:
   - `wheel`;
   - `dev-source`;
   - `bare-metal-dev`.
2. Убрать source/plugin mounts из production builtin configs или сделать их
   dev-only overlay.
3. Добавить `sndr install --mode wheel` smoke.
4. Добавить release checklist с wheel/container/SBOM.

### Sprint 2: report/verify/bench

1. Реализовать `sndr report bundle --full` по образцу club-3090.
2. Реализовать `sndr verify --stress` по образцу `verify-stress.sh`.
3. Реализовать `sndr verify --soak`.
4. Реализовать `sndr bench compare A.json B.json`.
5. Перевести primary perf metric на decode_TPOT + wall TPS как secondary.

### Sprint 3: memory/KV/DFlash

1. Backport/evaluate vLLM #42102.
2. Evaluate #42069 для Gemma4 DFlash backend selection.
3. Перенести `kv-calc.py` идеи в `sndr memory explain`.
4. Перенести residency instrumentation на `sndr_core`.
5. Добавить `sndr doctor topology`:
   - PCIe lane width;
   - P2P matrix;
   - NVLink;
   - power cap;
   - WSL/TDR warning.

### Sprint 4: Gemma 4 support

1. Добавить Gemma4 INT8 PTH config.
2. Добавить Gemma4 parser fixes tests.
3. Добавить Gemma4 MTP/DFlash profiles.
4. Считать Ampere FP8 PTH blocked, INT8 PTH default.
5. Для DFlash использовать workload-dependent `num_speculative_tokens`.

### Sprint 5: long-term integrations

1. LMCache/KV offload только после prefix-cache compatibility.
2. SGLang HiCache/radix ideas как research.
3. llama.cpp MTP как comparison path, не как dependency.
4. PFlash только research до прохождения stress/soak/quality gates.

## 12. Внешние источники

vLLM:

- [vLLM PR #42102](https://github.com/vllm-project/vllm/pull/42102)
- [vLLM PR #42069](https://github.com/vllm-project/vllm/pull/42069)
- [vLLM PR #42105](https://github.com/vllm-project/vllm/pull/42105)
- [vLLM PR #42006](https://github.com/vllm-project/vllm/pull/42006)
- [vLLM PR #42070](https://github.com/vllm-project/vllm/pull/42070)
- [vLLM PR #42074](https://github.com/vllm-project/vllm/pull/42074)
- [vLLM PR #42086](https://github.com/vllm-project/vllm/pull/42086)
- [vLLM PR #42095](https://github.com/vllm-project/vllm/pull/42095)
- [vLLM PR #42029](https://github.com/vllm-project/vllm/pull/42029)
- [vLLM PR #42028](https://github.com/vllm-project/vllm/pull/42028)
- [vLLM PR #42080](https://github.com/vllm-project/vllm/pull/42080)
- [vLLM PR #42030](https://github.com/vllm-project/vllm/pull/42030)
- [vLLM PR #42022](https://github.com/vllm-project/vllm/pull/42022)
- [vLLM PR #42064](https://github.com/vllm-project/vllm/pull/42064)

club-3090:

- [noonghunna/club-3090](https://github.com/noonghunna/club-3090)
- [club-3090 BENCHMARKS.md](https://github.com/noonghunna/club-3090/blob/master/BENCHMARKS.md)
- [club-3090 scripts/report.sh](https://github.com/noonghunna/club-3090/blob/master/scripts/report.sh)
- [club-3090 scripts/verify-stress.sh](https://github.com/noonghunna/club-3090/blob/master/scripts/verify-stress.sh)
- [club-3090 tools/kv-calc.py](https://github.com/noonghunna/club-3090/blob/master/tools/kv-calc.py)
- [club-3090 issue #103](https://github.com/noonghunna/club-3090/issues/103)
- [club-3090 issue #95](https://github.com/noonghunna/club-3090/issues/95)
- [club-3090 issue #58](https://github.com/noonghunna/club-3090/issues/58)
- [club-3090 issue #57](https://github.com/noonghunna/club-3090/issues/57)
- [club-3090 issue #72](https://github.com/noonghunna/club-3090/issues/72)
- [club-3090 issue #82](https://github.com/noonghunna/club-3090/issues/82)
- [club-3090 issue #85](https://github.com/noonghunna/club-3090/issues/85)
- [club-3090 issue #94](https://github.com/noonghunna/club-3090/issues/94)
- [club-3090 issue #97](https://github.com/noonghunna/club-3090/issues/97)
- [club-3090 issue #50](https://github.com/noonghunna/club-3090/issues/50)
- [club-3090 issue #47](https://github.com/noonghunna/club-3090/issues/47)
- [club-3090 issue #37](https://github.com/noonghunna/club-3090/issues/37)

Примечание: при повторной попытке GitHub API уже вернул rate limit для IP,
поэтому часть сведений по PR #42102 и списку PR была взята из ранее успешных
API-запросов в этом же аудите и из локального клона `club-3090`.

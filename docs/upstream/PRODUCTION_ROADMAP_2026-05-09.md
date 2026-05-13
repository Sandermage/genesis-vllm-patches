# Genesis / SNDR Production Roadmap - актуализация аудита 2026-05-08

> **⚠ HISTORICAL SNAPSHOT taken 2026-05-08.** Many findings here have
> already been addressed in Wave 6/7 (closed 2026-05-09). The current
> source of truth is `docs/_internal/INTEGRATED_PLAN_2026-05-09.md`,
> which merges this document's findings with the noonghunna external
> audit (`DEEP_AUDIT_VLLM_NOONGHUNNA_2026-05-08_RU.md`) and the Wave 7
> closure data.
>
> **Quick delta vs this snapshot (as of 2026-05-09):**
> - pytest: was failing 2 tests → now `2994 passed / 0 failed / 94 skipped`
> - installer smoke/dry-run: was failing P4/P5 → now `failed=0` (114 applied / 20 skipped)
> - Top-level `sndr` CLI: was install+launch only → now includes `doctor`,
>   `verify`, `model-config`, `plugins`, `patches`, `report` (and bridged compat)
> - Base wheel `pyproject.toml`: still has `dependencies = []` — release-gate item
> - Container Docker launcher: no longer `pip install -e /plugin` by default
>   (gated behind `SNDR_DEV_INSTALL_PLUGIN=1`)
>
> Use this file as a historical reference for what the audit found at the
> time, NOT as a checklist of currently-open issues.

Статус документа: исторический snapshot 2026-05-08. Live source of truth — `INTEGRATED_PLAN_2026-05-09.md`.

Цель документа: зафиксировать фактическое состояние проекта после последних исправлений, отделить закрытые проблемы от открытых, описать реальные баги с указанием файлов и строк, расширить варианты интеграций и дать рабочую дорожную карту для доведения проекта до production-ready состояния.

Важно: этот файл был аудитом и планом на момент создания. Исходный код проекта в рамках того прохода не изменялся.

## 1. Короткий вывод

Проект стал заметно ближе к правильной архитектуре:

- `vllm/sndr_engine/` сейчас является пустым skeleton namespace. В нем нет патчей, kernel-кода или алгоритмов.
- PN72 frequency ngram drafter перенесен в `sndr_core` и помечен как `community`.
- `sndr_core` уже содержит основной registry, apply-loop, schema validator, lifecycle audit, CLI installer, launcher, model-config system, bundles, plugin discovery, license gate и тестовый слой.
- Python-синтаксис чистый по текущему дереву.
- Registry/shadow/schema проверки проходят.
- Launcher dry-run теперь явно показывает нерешенные mount placeholders и не пытается запускать production script с `${...}`.

Но production-ready состояние еще не достигнуто. Главные оставшиеся блокеры:

- Installer smoke/dry-run все еще падает из-за реальных wiring ошибок P4/P5.
- Полный pytest падает на 2 тестах в no-torch среде.
- Base wheel пока не самодостаточен: `pyproject.toml` оставляет `dependencies = []`, хотя runtime code фактически использует PyYAML и `requests`.
- Top-level `sndr` CLI пока содержит только `install` и `launch`, тогда как документация и внутренние compat-модули уже подразумевают `doctor`, `verify`, `model-config`, `plugins`, `patches`, `report`.
- Документация и docstring'и частично устарели: местами все еще заявляют, что PN72/P67-like features относятся к engine, упоминают `_genesis`, `~/.genesis`, private IP и локальные пути.
- License layer технически встроен, но Ed25519 trust anchor пока placeholder zero-key.
- Тестовый `conftest.py` глобально включает legacy unsigned license mode, что может маскировать реальные production сценарии.
- Bundle tier gate проверяет только импорт `vllm.sndr_engine`, а skeleton импортируется успешно. Это слабый gate для будущих engine-only функций.
- Docker launcher все еще монтирует и editable-install'ит `/plugin`, то есть public core пока не полностью самодостаточен в container boot path.

Итоговая оценка:

| Слой | Текущее состояние | Production оценка |
|---|---|---|
| Python syntax | чисто | готово |
| Registry/schema/shadow | чисто | почти готово |
| `core/engine` граница | стратегически исправлена, но docs/bundles stale | частично готово |
| Installer | есть, но dry-run красный по P4/P5 | не готов |
| Launcher | сильно улучшен, но зависит от `/plugin` mount | beta |
| CLI | базовый, неполный | не готов |
| Tests | 2056 pass, 2 fail, 69 skip | не готов |
| Docs | большие и полезные, но устаревшие места | не готово |
| License/security | каркас есть, production key отсутствует | не готово |
| Packaging | root package неполный по deps/runtime surface | не готово |

## 2. Проверки, выполненные в этом проходе

Рабочая папка:

```text
$HOME/Documents/Visual Studio Code/genesis-vllm-patches
```

Команды и фактический результат:

| Проверка | Результат |
|---|---|
| AST parse `vllm/sndr_core`, `vllm/sndr_engine`, `tests` | `PY_AST_CHECK count=504 errors=0` |
| `python3 -m vllm.sndr_core.apply.shadow --strict` | clean, unexpected divergence нет |
| `python3 -m vllm.sndr_core.compat.schema_validator --quiet` | `PATCH_REGISTRY schema clean` |
| `python3 -m vllm.sndr_core.compat.lifecycle_audit_cli --quiet` | только warnings: experimental PN80/PN79/PN82, retired P61/P63/PN78/PN13/P8 |
| `python3 -m vllm.sndr_core.compat.cli self-test --quiet` | exit 0 |
| `python3 -m vllm.sndr_core.compat.model_config_cli list` | 8 configs total, 6 working, 2 tested |
| `python3 -m vllm.sndr_core.cli --help` | exit 0, но показывает только `install`, `launch` |
| `python3 -m vllm.sndr_core.cli launch --dry-run --non-interactive a5000-2x-35b-prod` | exit 0, правильно показывает unresolved mounts |
| `python3 -m vllm.sndr_core.cli install --dry-run --non-interactive` | exit 2, P4/P5 wiring failures |
| `pytest -q` | `2 failed, 2056 passed, 69 skipped in 7.05s` |

Model config inventory:

| Key | Статус | TPS | Tool |
|---|---:|---:|---:|
| `a5000-2x-27b-dflash-true` | working | 97.6 | 10/10 |
| `a5000-2x-27b-int4-long-ctx` | working | 38.6 | 10/10 |
| `a5000-2x-27b-int4-tq-k8v4` | working | 109.8 | 10/10 |
| `a5000-2x-27b-int4-tq-k8v4-dflash` | working | 83.9 | 10/10 |
| `a5000-2x-35b-fp8-dflash` | working | 127.2 | 9/10 |
| `a5000-2x-35b-prod` | working | 196.7 | 10/10 |
| `a5000-1x-27b-int4-tested` | tested/QA-only | 66.8 | 10/10 |
| `a5000-2x-27b-int4-tested` | tested/QA-only | 57.4 | 10/10 |

## 3. Что исправлено по сравнению с прошлым аудитом

### 3.1. `sndr_engine` теперь действительно skeleton

Файлы:

- `vllm/sndr_engine/__init__.py:1-59`
- `vllm/sndr_engine/LICENSE-NOTICE:1-37`
- `vllm/sndr_engine/version.py`

Состояние:

- В `sndr_engine` остались только `__init__.py`, `version.py`, `LICENSE-NOTICE`.
- `engine_available()` возвращает `False`, если нет будущего private overlay.
- PN72 перенесен в core: `vllm/sndr_core/integrations/spec_decode/pn72_frequency_ngram_drafter.py`.
- Helper перенесен в core: `vllm/sndr_core/kernels/ngram_frequency_filter.py`.

Оценка: архитектурно правильно. Это соответствует текущей стратегии: public repo содержит все существующие наработки, private engine пока пустой, core знает о возможности engine, но не зависит от него.

### 3.2. PN72 теперь community

Файл:

- `vllm/sndr_core/dispatcher/registry.py:235-258`

Факты:

- PN72 имеет `"tier": "community"`.
- PN72 больше не должен считаться commercial/engine фичей.

Оценка: правильно. Старые документы, которые все еще называют PN72 engine patch, нужно обновить.

### 3.3. Shadow/schema/lifecycle checks приведены в рабочее состояние

Факты:

- `apply.shadow --strict` clean.
- `schema_validator --quiet` clean.
- `lifecycle_audit_cli --quiet` дает только ожидаемые предупреждения.

Оценка: хороший фундамент. Следующий шаг - сделать lifecycle metadata обязательной не только через soft warning, но и через CI gate для новых патчей.

### 3.4. Launcher dry-run стал безопаснее

Команда:

```bash
python3 -m vllm.sndr_core.cli launch --dry-run --non-interactive a5000-2x-35b-prod
```

Факт:

Dry-run показывает:

```text
UNRESOLVED MOUNTS - host.yaml is missing entries for:
    ${genesis_src}
    ${models_dir}
    ${plugin_src}
```

Оценка: это правильное поведение. Live launch должен отказываться запускаться при нерешенных mount placeholders.

### 3.5. Runtime deps в generated Docker bootstrap закреплены версиями

Файл:

- `vllm/sndr_core/model_configs/schema.py:724-750`

Факт:

Внутри generated docker command runtime deps теперь задаются pinned:

```text
pandas==2.2.3 scipy==1.14.1 xxhash==3.5.0
```

Оценка: лучше, чем unpinned install. Но текущая логика противоречит комментарию: установка происходит только при `SNDR_DEV_INSTALL_RUNTIME_DEPS=1`, а не "pinned by default". Комментарий и реальное условие нужно синхронизировать.

## 4. Блокирующие ошибки текущего состояния

### P0-1. Installer dry-run падает из-за P4/P5 variable typo

Файл:

- `vllm/sndr_core/apply/_per_patch_dispatch.py:263-344`

Проблема P4:

```python
from vllm.sndr_core.patches.scheduler import p4_tq_hybrid
assert callable(patch_4_tq_hybrid.apply)
```

Строки:

- импорт: `vllm/sndr_core/apply/_per_patch_dispatch.py:289`
- неверное имя: `vllm/sndr_core/apply/_per_patch_dispatch.py:290`
- real apply path тоже вызывает старое имя: `vllm/sndr_core/apply/_per_patch_dispatch.py:301`

Проблема P5:

```python
from vllm.sndr_core.patches.kv_cache import p5_page_size
assert callable(patch_5_page_size.apply)
```

Строки:

- импорт: `vllm/sndr_core/apply/_per_patch_dispatch.py:328`
- неверное имя: `vllm/sndr_core/apply/_per_patch_dispatch.py:329`
- real apply path тоже вызывает старое имя: `vllm/sndr_core/apply/_per_patch_dispatch.py:339`

Фактический вывод installer dry-run:

```text
applied=104, skipped=17, failed=8
6 patches reported missing runtime ... expected
P4 TurboQuant hybrid model support: wiring import failed: name 'patch_4_tq_hybrid' is not defined
P5 KV cache page size unification: wiring import failed: name 'patch_5_page_size' is not defined
```

Почему это критично:

- Installer smoke является первым пользовательским health check.
- Сейчас fresh install на Mac/CPU/dev-host завершается exit 2 даже после правильного учета missing torch/vllm.
- Это не external dependency, а локальная ошибка имени.

Решение:

- P4 должен использовать `p4_tq_hybrid.apply`.
- P5 должен использовать `p5_page_size.apply`.
- Добавить unit test на dry-run каждого registered patch wrapper, чтобы такая ошибка не возвращалась.

Критерий закрытия:

```bash
python3 -m vllm.sndr_core.cli install --dry-run --non-interactive
```

должен завершаться exit 0 на no-GPU/no-torch dev host, если только отсутствия runtime корректно классифицированы как skip.

### P0-2. Полный pytest падает на no-torch assumptions

Факт:

```text
2 failed, 2056 passed, 69 skipped
```

Падение 1:

- `tests/legacy/test_guards.py:146-152`
- test: `TestDependencyVersions.test_get_torch_version_returns_tuple`

Проблема:

```python
v = get_torch_version()
assert v is not None  # torch is available (we imported it)
```

В текущей среде `torch` не установлен, поэтому `get_torch_version()` корректно возвращает `None`. Тест должен либо:

- skip при отсутствии torch;
- либо проверять контракт `None allowed when torch absent`.

Падение 2:

- `tests/unit/integrations/attention/gdn/test_p103_fla_cliff2.py:186-207`
- test: `test_p103_self_install_succeeds_with_mock_chunk_globals`
- implementation: `vllm/sndr_core/integrations/attention/gdn/p103_fla_cliff2_chunked.py:142-159`

Проблема:

`_make_chunked_wrapper()` импортирует `torch` при создании wrapper:

```python
def _make_chunked_wrapper(...):
    ...
    import torch
```

Тест создает synthetic globals, но не обеспечивает torch. Поэтому `_genesis_p103_install_at_import(fake_globals)` возвращает `False`, хотя test ожидает `True`.

Варианты решения:

1. Если P103 не обязан работать без torch, тест должен skip/xfail при отсутствии torch.
2. Если installation helper должен работать в no-torch import path, импорт `torch` нужно отложить внутрь actual hot path wrapper.

Рекомендация:

- Для production лучше отложить `torch` import внутрь ветки, где реально используются torch операции.
- Для теста добавить отдельный no-torch regression case, чтобы helper не ломал import-time install.

### P0-3. Base wheel не самодостаточен по зависимостям

Файл:

- `pyproject.toml:65-80`

Текущее:

```toml
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "pyyaml>=6.0",
]
telemetry = [
    "requests>=2.28",
]
```

Проблема:

Комментарий говорит, что PyYAML уже transitive dependency vLLM. Но `vllm-sndr-core` не должен полагаться на случайную transitive dependency, если CLI/model-config работает как самостоятельный package.

Фактические места:

- YAML parsing/model configs: `vllm/sndr_core/model_configs/schema.py`
- host config: `vllm/sndr_core/model_configs/host.py`
- boot probe HTTP: `vllm/sndr_core/utils/boot_probe.py`

Решение:

- Минимальный вариант: добавить `pyyaml>=6.0` в base dependencies.
- Для `requests`: либо добавить в base, либо сделать optional lazy import с понятным сообщением `pip install vllm-sndr-core[telemetry]`/`[http]`.
- Зафиксировать policy: CLI install/launch/model-config не должны падать из-за отсутствия optional deps.

### P0-4. Top-level `sndr` CLI неполный

Файл:

- `vllm/sndr_core/cli/__init__.py:1-61`

Факт:

Top-level CLI регистрирует только:

- `install`
- `launch`

При этом в `compat` уже существуют:

- `vllm/sndr_core/compat/doctor.py`
- `vllm/sndr_core/compat/verify.py`
- `vllm/sndr_core/compat/model_config_cli.py`
- `vllm/sndr_core/compat/plugins.py`
- `vllm/sndr_core/compat/schema_validator.py`
- `vllm/sndr_core/compat/lifecycle_audit_cli.py`
- `vllm/sndr_core/compat/explain.py`
- `vllm/sndr_core/compat/bench.py`

Проблема:

Документация и пользовательская логика уже ожидают единый `sndr` интерфейс, но он пока не является единым launcher/control plane.

Рекомендация:

Сделать top-level surface:

```text
sndr install
sndr launch
sndr doctor
sndr verify
sndr model-config list|validate|preflight|launch|diagnose|verify
sndr patches list|explain|audit|shadow|lifecycle
sndr plugins list|audit|apply
sndr report bundle
sndr config init|show|validate|doctor
```

Критерий:

- README и CLI help должны совпадать.
- Все старые `python3 -m vllm.sndr_core.compat.*` остаются как internal/backcompat, но пользователь видит только `sndr ...`.

## 5. Архитектурные несостыковки и stale места

### P1-1. Документация все еще говорит, что PN72 лежит в engine

Файлы и строки:

- `README.md:28-31`
- `README.md:91`
- `docs/INSTALL.md:124-138`
- `docs/PATCHES.md:41-42`
- `vllm/sndr_core/__init__.py:12-26`

Проблема:

Фактическое состояние уже другое:

- PN72 находится в `vllm/sndr_core/integrations/spec_decode/pn72_frequency_ngram_drafter.py`.
- Helper находится в `vllm/sndr_core/kernels/ngram_frequency_filter.py`.
- `sndr_engine` skeleton-only.

Риск:

- Пользователь и будущий maintainer будут думать, что public core неполный без private wheel.
- Это противоречит текущей стратегии доверия: все существующее публично, private engine будет только для будущих новых разработок.

Решение:

- Обновить README/docs так, чтобы `sndr_engine` описывался как reserved namespace.
- PN72/P67/P67b и все текущие патчи оставить как core/community.
- В docs четко написать: private engine пока пустой и не нужен для текущего public repo.

### P1-2. `vllm/sndr_core/__init__.py` содержит устаревший migration status

Файл:

- `vllm/sndr_core/__init__.py:12-26`

Проблема:

Docstring говорит:

- advanced features live in `vllm.sndr_engine`;
- current stage skeleton only;
- all code still lives in `vllm/_genesis`;
- final will make `_genesis` forward alias.

Фактически:

- `_genesis` удален из текущего дерева.
- основной код уже живет в `vllm/sndr_core`.
- `sndr_engine` skeleton-only и не содержит текущих advanced features.

Решение:

- Обновить docstring под v11 architecture.
- Записать текущую стратегию: `sndr_core` содержит все public functionality, `sndr_engine` только optional future overlay.

### P1-3. Bundle tier gate слабый

Файлы:

- `vllm/sndr_core/bundles/_common.py:64-77`
- `vllm/sndr_core/bundles/attention_tq_multi_query.py:1-41`
- `vllm/sndr_core/bundles/__init__.py`

Проблема:

`run_bundle()` для `tier == "engine"` проверяет только:

```python
import vllm.sndr_engine
```

Но public skeleton `vllm.sndr_engine` теперь импортируется всегда. Значит такой gate не доказывает наличие private overlay или license.

Дополнительная несостыковка:

- `attention_tq_multi_query.py` все еще помечает P67/P67b как engine tier.
- Registry уже помечает P67/P67b как `community`: `vllm/sndr_core/dispatcher/registry.py:384-414`.

Решение:

1. Bundle `attention_tq_multi_query` привести к `community`, если P67/P67b остаются public.
2. Для будущих engine bundles использовать не import-check, а:
   - `vllm.sndr_core.license.check_engine_tier_eligible()`;
   - `vllm.sndr_engine.engine_available()`;
   - signed token validation.
3. Добавить тест: skeleton `vllm.sndr_engine` не должен unlock'ать engine tier.

### P1-4. License gate имеет placeholder trust anchor

Файл:

- `vllm/sndr_core/license.py:58-60`

Текущее:

```python
_TRUST_ANCHOR_PUBKEY_B64URL = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"  # 32 zero bytes — placeholder
)
```

Риск:

- Signed license tokens фактически не production-ready.
- Сейчас это нормально для skeleton stage, но нельзя выпускать commercial activation как готовую фичу.

Дополнительная проблема:

- `tests/conftest.py:27-34` глобально устанавливает:
  - `SNDR_ENGINE_LICENSE_KEY=test-license-key-pytest`
  - `SNDR_ALLOW_LEGACY_LICENSE_KEYS=1`

Риск:

- Тесты могут маскировать поведение production strict mode.

Решение:

- Сгенерировать реальную Ed25519 keypair offline.
- Публичный ключ положить в `license.py`.
- Приватный ключ не хранить в repo.
- Legacy unsigned mode оставить только в точечных тестах через fixture/monkeypatch.
- Добавить strict tests без `SNDR_ALLOW_LEGACY_LICENSE_KEYS`.

### P1-5. Docker launch still depends on `/plugin`

Файл:

- `vllm/sndr_core/model_configs/schema.py:708-750`

Проблема:

Generated Docker command все еще делает:

```bash
cp -r /plugin /tmp/genesis_vllm_plugin
pip install --no-deps -e /tmp/genesis_vllm_plugin
```

То есть даже при наличии `vllm.sndr_core.apply` Docker path остается завязан на `tools/genesis_vllm_plugin`.

Риск:

- Public wheel не самодостаточен.
- Container boot depends on repo layout.
- У пользователя без полного repo/`tools` mount launch сломается.

Решение:

Вариант A, предпочтительный:

- Убрать `/plugin` как обязательный runtime mount.
- Встраивать plugin entry point в `vllm-sndr-core` package.
- Docker image должен получать `vllm-sndr-core` через wheel или bind mount самого package.

Вариант B, временный:

- Явно документировать `/plugin` как dev-only mode.
- Production launcher должен иметь `--mode wheel|editable`.

### P1-6. Hardcoded/private paths and IP still present

Найденные места:

- `README.md:594`, `README.md:616` - `<host>`
- `README.md:600`, `README.md:639` - `/nfs/genesis/models`
- `compose/docker-compose.unit.yml:8-13` - `192.168.1.15`, `$HOME`
- `.github/ISSUE_TEMPLATE/community_config.md:23` - `~/.genesis/model_configs/`
- `tests/soak/pn40_soak_1000.py:6`, `:22` - default `<host>`
- `tests/soak/cliff2_multiturn_soak.py:18`, `:54`, `:57` - default endpoint/SSH host
- `tests/bench/comprehensive_bench.py:18`, `:40`, `:43` - default endpoint/SSH host
- `scripts/fetch_models.sh:8-29` - `/nfs/genesis/models`
- `scripts/launch/preflight_check.sh:85-121` - `/nfs/genesis/models`, `_genesis` path check
- `scripts/launch.sh:12` - `~/.genesis/model_configs`

Оценка:

- В docs/examples допустимы примерные IP, если явно написано "example".
- В tests/probes/soak default private IP лучше заменить на env-required или localhost-safe default.
- В active scripts нельзя оставлять `_genesis` как required path.

### P1-7. `_genesis` references still exist in active comments/tests/scripts

Контекст:

Многие `_genesis` строки являются легитимными marker names или историческими комментариями. Но часть активных скриптов все еще проверяет old path.

Критичные места:

- `scripts/launch/preflight_check.sh:112-121` проверяет `vllm/_genesis`.
- `vllm/sndr_core/utils/boot_probe.py:22-31` в examples запускает `vllm._genesis.utils.boot_probe`.
- `vllm/sndr_core/__init__.py:23-26`, `:40-45`, `:61` содержит old migration narrative.

Решение:

- Разделить `rg "_genesis"` на категории:
  - allowed markers in injected code;
  - historical archived scripts;
  - active docs/comments;
  - active runtime checks.
- В CI запретить новые runtime/doc references к `vllm._genesis`, кроме allowlist.

## 6. Полнота текущих скриптов и утилит

### 6.1. Installer

Файл:

- `vllm/sndr_core/cli/install.py`

Сильные стороны:

- есть preflight;
- есть dry-run;
- есть hardware detection;
- есть workload selection;
- есть smoke test;
- missing runtime теперь частично классифицируется как skip.

Недостатки:

- P4/P5 dry-run bug делает installer красным;
- GitHub tags fallback в dry-run пишет warning;
- installer все еще устанавливает `tools/genesis_vllm_plugin`;
- нет user-facing report bundle после fail;
- нет строгого режима "offline install from local checkout/wheel";
- нет clear separation dev editable vs production wheel.

Что добавить:

- `sndr install --mode editable|wheel|docker-only`;
- `sndr install --offline --wheel path.whl`;
- `sndr install --report-on-fail`;
- post-install `sndr doctor`;
- machine-readable JSON summary.

### 6.2. Launcher/model configs

Файлы:

- `vllm/sndr_core/cli/launch.py`
- `vllm/sndr_core/model_configs/schema.py`
- `vllm/sndr_core/model_configs/registry.py`
- `vllm/sndr_core/model_configs/builtin/*.yaml`

Сильные стороны:

- уже есть schema;
- есть working/tested tier;
- есть references TPS/tool/CV;
- есть dry-run script rendering;
- есть unresolved mount diagnostics;
- есть host.yaml placeholder model.

Недостатки:

- generated docker script still tied to `/plugin`;
- comments around pinned deps inaccurate;
- host config docs partly still mention `~/.genesis` as primary;
- отсутствует `sndr config doctor`;
- нет diff между выбранным preset и фактическим host.yaml/env;
- нет `render --format compose|systemd|bash|docker-run`.

Что добавить:

- `sndr launch render <key> --format bash|compose|systemd|k8s`;
- `sndr config init`;
- `sndr config validate`;
- `sndr config doctor`;
- `sndr model-config compare <key> --against live`;
- `sndr launch --explain` с выводом почему включен каждый patch flag.

### 6.3. Apply loop / registry / dispatcher

Сильные стороны:

- registry большой и структурированный;
- shadow check clean;
- schema validator clean;
- lifecycle audit есть;
- `applies_to`, `requires_patches`, `conflicts_with` используются как metadata foundation.

Недостатки:

- много lifecycle fields unset, installer spam'ит предупреждения;
- dry-run wrappers еще не покрыты тестом на переменные/импорты;
- bundles conflict awareness пока в комментариях, не в enforced plan;
- apply-loop и model-config flags пока не дают единого "activation graph".

Что добавить:

- `sndr patches plan --model-config <key>`;
- DAG dependencies/conflicts check before apply;
- strict lifecycle CI gate для новых патчей;
- `patch_id -> file -> env_flag -> default_on -> model_config` cross-report;
- `sndr patches explain P67 --why-enabled`.

### 6.4. Bundles

Текущие bundles:

- `attention_gdn_spec`
- `attention_tq_multi_query`
- `reasoning_qwen3`
- `spec_decode_async_cleanup`
- `tool_parsing_qwen3coder`

Сильные стороны:

- правильная идея: атомарные группы патчей;
- `MultiFilePatchTransaction` already exists;
- bundles подходят для user-facing presets.

Недостатки:

- engine gate слабый;
- P67/P67b bundle stale tier;
- нет CLI для bundles;
- нет dry-run plan/explain;
- нет conflict enforcement at bundle layer.

Что добавить:

```text
sndr bundles list
sndr bundles explain attention_tq_multi_query
sndr bundles plan --model-config a5000-2x-35b-prod
sndr bundles apply <bundle> --dry-run
```

### 6.5. Plugin system

Файл:

- `vllm/sndr_core/compat/plugins.py`

Сильные стороны:

- discovery через `vllm_genesis_patches`;
- opt-in через `GENESIS_ALLOW_PLUGINS`;
- schema validation есть.

Недостатки:

- не интегрирован в top-level `sndr`;
- нет provenance/signature policy;
- нет registry-level conflict/dependency integration;
- public launcher все еще использует old repo-local plugin mount.

Рекомендации:

- `sndr plugins list|audit|apply`;
- plugin metadata: author, source, license, required vllm pins, required core version;
- optional signature;
- plugin quarantine: disabled by default, explicit enable per plugin.

## 7. Рекомендуемая целевая структура проекта

Текущая стратегия правильная: пока не делать платную версию, но подготовить архитектуру.

Целевая модель:

```text
vllm/
  sndr_core/
    apply/
    bundles/
    cli/
    compat/
    dispatcher/
    detection/
    kernels/
    license.py
    model_configs/
    patches/
    paths/
    runtime/
    schemas/
    utils/
  sndr_engine/
    __init__.py
    version.py
    LICENSE-NOTICE
```

Правило разделения:

| Слой | Что хранить |
|---|---|
| `sndr_core` | все, что уже есть в public repo; backports; community fixes; launcher; config manager; plugin loader; public kernels; docs; tests |
| `sndr_engine` public tree | только skeleton namespace и notice |
| private `sndr_engine` repo | будущие новые private-only kernels/patches, которых нет в GitHub и которые не являются переносом чужих PR |
| `tools/` | dev tooling only, не runtime dependency |
| `scripts/launch/_archive` | historical only, не участвует в active docs |
| root scripts | только thin wrappers на `python -m vllm.sndr_core...` |

Что нужно привести:

- убрать engine labels с текущих public патчей;
- переместить user-facing commands в `sndr`;
- сделать package self-contained;
- отделить archived examples от active scripts;
- в docs явно указать, что private engine появится позже и сейчас не нужен.

## 8. Варианты интеграций

### Вариант A. Public Core + private Engine overlay через entry points

Идея:

- `sndr_core` не импортирует private code напрямую.
- Private wheel регистрирует entry point, например `sndr_engine_plugins`.
- Core discovery ищет installed engine providers.
- License gate проверяет signed token.

Плюсы:

- core не зависит от engine;
- public repo не содержит private code;
- можно добавлять future paid features без ломки public API.

Минусы:

- нужна строгая версия контракта между core и engine;
- нужна crypto/signature инфраструктура.

Что внедрить:

- `EngineProvider` protocol;
- `sndr engine status`;
- `sndr engine doctor`;
- `sndr engine features`;
- signed feature manifest.

### Вариант B. Bundle layer как главный user-facing слой

Идея:

Пользователь включает не отдельные Pxx, а понятные feature bundles:

- `qwen3-tool-stability`
- `turboquant-mtp-prod`
- `gdn-spec-decode`
- `long-context-memory`
- `ampere-a5000-prod`

Плюсы:

- меньше ручных flags;
- проще поддержка;
- проще объяснять, что включено и почему.

Что внедрить:

- `sndr bundles list`;
- `sndr bundles plan`;
- `sndr bundles apply`;
- связать bundles с model configs;
- добавлять conflict/dependency report.

### Вариант C. Community plugin ecosystem

Идея:

Сторонние пользователи могут добавлять патчи через entry points, но только opt-in.

Текущее основание уже есть:

- `vllm/sndr_core/compat/plugins.py`
- `GENESIS_ALLOW_PLUGINS`
- schema validation

Что добавить:

- `sndr plugins audit`;
- source/provenance;
- plugin lockfile;
- trusted plugin allowlist;
- signature verification later.

### Вариант D. Unified launcher integrations

Цель:

Из одного model-config рендерить разные deployment targets:

- docker run;
- docker compose;
- systemd service;
- Kubernetes manifest;
- bare metal command;
- remote SSH runbook.

Команды:

```text
sndr launch render a5000-2x-35b-prod --format bash
sndr launch render a5000-2x-35b-prod --format compose
sndr launch render a5000-2x-35b-prod --format systemd
sndr launch render a5000-2x-35b-prod --format k8s
```

Важно:

- никакие private paths не должны быть default;
- все host-specific values только через `~/.sndr/host.yaml` или env;
- dry-run должен быть default-friendly.

### Вариант E. Remote GPU server integration

Контекст:

У пользователя есть remote server `<host>`, но в коде нельзя hardcode'ить этот адрес.

Правильная модель:

```text
sndr doctor --remote genesis-a2
sndr report bundle --remote genesis-a2
sndr launch --remote genesis-a2 a5000-2x-35b-prod
```

Host profile:

```yaml
hosts:
  genesis-a2:
    ssh: <user>@<host>
    models_dir: /nfs/genesis/models
    genesis_src: $HOME/genesis-vllm-patches/vllm/sndr_core
```

Что собирать remote:

- `hostname`;
- `nvidia-smi`;
- driver/CUDA;
- docker ps;
- vllm image tag;
- active env flags;
- `/v1/models`;
- health endpoint;
- last boot logs.

### Вариант F. Model registry / Hugging Face integration

Идея:

Добавить слой управления моделями:

```text
sndr models search qwen3
sndr models pull <hf-repo>
sndr models register <path> --family qwen3 --quant int4
sndr models doctor <key>
```

Что проверять:

- HF token;
- disk space;
- snapshot path;
- safetensors presence;
- tokenizer/chat template;
- model family auto-detect;
- compatibility with presets;
- required patches.

### Вариант G. Upstream vLLM drift watcher

Текущее основание:

- `tools/check_upstream_drift.py`
- `.github/workflows/upstream_drift_watcher.yml`
- `vllm/sndr_core/integrations/upstream_compat.py`

Что добавить:

- auto-map PR/issue -> affected patch;
- status: `active`, `superseded`, `merged_upstream`, `needs_rebase`;
- weekly report;
- retirement candidate list;
- GitHub issue auto-update.

### Вариант H. Memory/cache integrations

Цель:

Системно закрывать OOM/long-context проблемы.

Направления:

- KV cache compression tuning: TurboQuant, fp8 KV, block size, skip layers;
- prefix cache/hybrid cache correctness: P84/P85 family;
- CPU/offload-aware планирование;
- LMCache integration для external KV reuse/offload;
- SGLang HiCache идеи для hierarchical cache;
- TensorRT-LLM KV cache reuse идеи для stable cache reuse keys;
- per-request memory budget и admission control;
- post-warmup release and cache pressure reporting.

Что добавить в project:

```text
sndr memory doctor
sndr memory plan --model-config <key>
sndr memory simulate --ctx 128k --sequences 2
sndr memory report --live
```

### Вариант I. Observability/support integration

Текущее:

- PN65 structured access log;
- memory metrics modules;
- boot probe;
- tests/probes.

Что добавить:

- unified `sndr report bundle`;
- Prometheus metrics exporter;
- structured JSON logs;
- patch activation summary at boot;
- request-level fields: model, prompt tokens, completion tokens, tool use, spec-decode mode, KV dtype, GPU memory pressure;
- redaction policy for prompts/API keys.

### Вариант J. club-3090 support integration

Источник:

- [club-3090 issues](https://github.com/noonghunna/club-3090/issues)

Актуальные открытые темы на странице issues:

- #77/#76/#75/#74/#73 - bench series for Qwen3.6 dual 3090, DFlash, Turbo, NVLink/PCIe.
- #72 - qwen3coder tool parser causes indefinite SSE silence on prose containing literal `<tool_call>`.
- #60 - EngineCore failed to start.
- #58 - CUDA OOM with 50 MiB allocation and almost full 24 GB GPU.
- #50 - CUDA driver error device not ready around 157K tokens.
- #47 - cliff 2 90K dual-turbo with fp8_e5m2.

Что стоит перенести в Genesis/SNDR:

- standardized benchmark profile import from club-3090 issues;
- `sndr report bundle` format compatible with club issue templates;
- qwen3coder parser regression suite based on #72;
- long-context OOM recipes based on #58/#50/#47;
- PCIe/NVLink profile detection for dual GPU presets;
- comparison matrix: A5000 x2 vs 3090 x2 vs 4090 x1/x2.

## 9. Внешние источники и что из них брать

### vLLM TurboQuant PR #38479

Источник:

- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479)

Факты из PR:

- PR merged 2026-04-15.
- Добавляет TurboQuant KV cache compression.
- Основной usage: `--kv-cache-dtype turboquant_k8v4`.
- Заявлены presets `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`.
- Scope PR: full-attention and uniform sliding-window transformer models; hybrid architectures planned follow-up.
- В summary указаны WHT rotation, fused Triton store kernels, compact slot sizes, CUDAGraph memory fix, stream overlap.

Что применить:

- Genesis должен держать TurboQuant compatibility layer, но не дублировать upstream без необходимости.
- P4/P5 остаются важны для hybrid/mamba cases, потому что upstream PR scope прямо исключал hybrid.
- Launcher должен валидировать `kv_cache_dtype` и `skip_layers`.
- Memory doctor должен рассчитывать slot bytes and expected compression.

### vLLM PR #40914

Источник:

- [vLLM PR #40914](https://github.com/vllm-project/vllm/pull/40914)

Факты:

- Статус страницы: Open.
- PR фиксит TurboQuant K+1 spec-verify routing for #40880.
- Summary описывает routing branch in `TurboQuantAttentionImpl.forward()` for uniform K+1 spec-verify batches through `triton_turboquant_decode_attention` via synthetic sequence lengths.

Что применить:

- P67/P67b должны оставаться в core/community до merge upstream.
- После merge нужен A/B:
  - upstream #40914 path;
  - Genesis P67 custom kernel path;
  - performance/correctness on Qwen3.6 35B/27B, A5000, 3090, 4090.
- Registry должен уметь marking: `watch_for_upstream: vllm#40914`.

### vLLM issue #40069

Источник:

- [vLLM issue #40069](https://github.com/vllm-project/vllm/issues/40069)

Связь:

- Tracking issue for TurboQuant/HIGGS attention follow-ups.
- На странице #40914 видно упоминание #40069 как tracking issue.

Что применить:

- Добавить TurboQuant follow-up watcher в upstream drift report.
- Связать P4/P5/P67/P67b/P78/P98/P99/P101/PN57 с этим watcher.

### Sandermage genesis-vllm-patches issues

Источник:

- [Sandermage/genesis-vllm-patches issues](https://github.com/Sandermage/genesis-vllm-patches/issues)

Факт:

- На странице issues виден открытый issue #1: `OOM with long context`.

Что применить:

- `sndr memory doctor` должен стать priority feature.
- OOM recipes должны быть связаны с model configs.
- Launcher должен уметь предлагать safe fallback config:
  - lower max-model-len;
  - lower max-num-batched-tokens;
  - lower max-num-seqs;
  - fp8/turboquant KV;
  - disable risky spec decode;
  - reduce cudagraph capture sizes.

## 10. Что сделать по qwen/gemma/TurboQuant/MTP/DFlash/PFlash

### Qwen3/Qwen3.6

Уже есть:

- tool parsing fixes;
- reasoning parser fixes;
- qwen3coder streaming/tool patches;
- long-context tool adherence;
- qwen3coder subset schema filter;
- P67/P67b TurboQuant spec-decode routing;
- MTP-related truncation and streaming handling.

Не хватает:

- unified parser regression corpus;
- SSE silence detector;
- literal `<tool_call>` prose test from club-3090 #72;
- live smoke test against OpenAI-compatible endpoint;
- model-specific compatibility matrix for 27B/35B/Nemotron-like MoE cases.

### Gemma

Нужно добавить:

- model detection for Gemma 3/4 variants;
- separate parser/reasoning profile;
- memory presets for Gemma long context;
- sanity tests for tokenizer/chat template behavior;
- compatibility notes for vLLM upstream issues affecting Gemma.

### TurboQuant

Уже есть:

- P4 hybrid support;
- P5 page size unification;
- P67/P67b spec-decode routing;
- P67c sparse-V config;
- P78 tolist capture guard;
- P98/P99 workspace manager performance strategy;
- P101 continuation slicing;
- PN57 centroids disk cache.

Блокеры:

- P4/P5 apply wrapper typo.
- Need upstream drift watcher around #38479/#39931/#40069/#40914.
- Need memory doctor for TQ slot math.

### MTP/spec decode

Уже есть:

- P58/P60/P60b/P61/P61b/P61c/P64/P70/P71/P75/P77/P79b/P79c/P79d/P82/P83/P86/P94/PN8/PN9/PN33/PN38/PN40.

Не хватает:

- unified activation graph;
- bundle-level plan;
- automatic conflict resolution;
- model-config level "spec decode safety grade";
- per-model acceptance/rollback metrics.

### DFlash/PFlash

Уже есть DFlash-related patches:

- PN21 DFlash SWA support;
- PN22 local argmax TP;
- PN23 combine hidden dtype;
- PN24 aux layer indexing;
- PN38 DFlash quant drafter;
- PN40 DFlash omnibus/classifier hooks.

Не хватает:

- clear docs explaining DFlash vs PFlash vs upstream MTP;
- DFlash benchmark matrix;
- DFlash fallback story;
- detection when DFlash patch should not apply;
- config presets for PCIe vs NVLink dual GPU.

## 11. Рекомендуемый порядок работ

### Sprint 0. Красный build/smoke

Цель: убрать фактические падения.

Задачи:

1. Исправить P4/P5 variable typos in `_per_patch_dispatch.py`.
2. Исправить/промаркировать 2 failing pytest cases.
3. Добавить dry-run wrapper test для всех `@register_patch`.
4. Повторить:
   - `install --dry-run`;
   - `pytest -q`;
   - `shadow --strict`;
   - `schema_validator`.

Критерий:

```text
install --dry-run exit 0
pytest exit 0
```

### Sprint 1. Self-contained package

Задачи:

1. Добавить runtime deps policy в `pyproject.toml`.
2. Перенести/встроить plugin entry point в core package или сделать `/plugin` dev-only.
3. Проверить package data для schemas.
4. Проверить install from wheel in clean venv.
5. Проверить no-torch CLI surface.

Критерий:

```text
pip install dist/vllm_sndr_core-*.whl
sndr --help
sndr doctor
sndr model-config list
```

работают без checkout repo.

### Sprint 2. CLI unification

Задачи:

1. Подключить `doctor`, `verify`, `model-config`, `patches`, `plugins`, `report`, `config` в top-level `sndr`.
2. Обновить README под реальный CLI.
3. Добавить shell completion позже.

Критерий:

```text
sndr --help
```

показывает полный и честный surface.

### Sprint 3. Docs cleanup

Задачи:

1. Убрать stale engine claims for PN72/P67.
2. Обновить `vllm/sndr_core/__init__.py` docstring.
3. Обновить `docs/INSTALL.md`, `docs/PATCHES.md`, `README.md`.
4. Перевести active examples на `~/.sndr`.
5. Private IP/path оставить только как explicitly marked examples или убрать.

Критерий:

```bash
rg "vllm\\._genesis|~/.genesis|192\\.168\\.|$HOME|/nfs/genesis" README.md docs scripts vllm/sndr_core tests
```

должен выдавать только allowlisted examples/tests/markers.

### Sprint 4. Engine boundary hardening

Задачи:

1. Bundle tier gate перевести на license/engine_available.
2. Убрать engine tier с текущих public bundles.
3. Оставить `sndr_engine` skeleton.
4. Подготовить private overlay protocol, но не публиковать paid engine.

Критерий:

- public core fully works without private engine;
- skeleton import does not unlock engine features;
- future private engine can be added without changing core imports.

### Sprint 5. Memory/cache roadmap

Задачи:

1. `sndr memory doctor`.
2. `sndr memory plan`.
3. OOM recipes linked to model configs.
4. Add live memory probe.
5. Integrate ideas from LMCache/SGLang HiCache/TensorRT-LLM KV reuse as optional research notes, not immediate dependencies.

### Sprint 6. Qwen/tool/spec-decode stabilization

Задачи:

1. Regression suite for club-3090 #72.
2. SSE silence detector.
3. Qwen3coder literal `<tool_call>` tests.
4. MTP truncation and stream parser live tests.
5. Tool-call quality harness for 27B/35B configs.

### Sprint 7. Benchmark and remote support

Задачи:

1. `sndr report bundle`.
2. Remote host profile.
3. A5000/3090/4090 matrix.
4. PCIe/NVLink detection.
5. Config compare across rigs.

### Sprint 8. Release/security

Задачи:

1. Real Ed25519 public key.
2. Signed release artifacts.
3. SBOM.
4. Dependency policy.
5. Remove global legacy license env from tests.
6. Security docs for plugin execution.

### Sprint 9. Upstream integration and retirement

Задачи:

1. Watch #38479/#40069/#40914 and related TurboQuant PRs.
2. Mark patches as superseded/active.
3. A/B upstream vs Genesis.
4. Auto-open retirement recommendations.
5. Keep local patches only where they beat upstream or cover unsupported cases.

## 12. Production readiness checklist

Перед public beta:

- [ ] `pytest -q` green locally.
- [ ] installer dry-run green.
- [ ] no-torch CLI green.
- [ ] README matches actual CLI.
- [ ] `sndr_engine` docs fixed to skeleton-only.
- [ ] P4/P5 wiring bug fixed.
- [ ] base deps policy fixed.
- [ ] `/plugin` not mandatory for production launch.
- [ ] `sndr doctor` available from top-level CLI.
- [ ] `sndr report bundle` available or at least planned with schema.

Перед production:

- [ ] wheel install tested in clean venv.
- [ ] Docker launch tested on GPU server.
- [ ] A5000 x2 presets validated live.
- [ ] 3090/4090 community profile support documented.
- [ ] signed license trust anchor real.
- [ ] legacy license mode not global in tests.
- [ ] plugin system has security/provenance policy.
- [ ] docs contain no accidental private paths.
- [ ] upstream drift watcher produces actionable report.
- [ ] release artifacts signed/versioned.

Перед private engine:

- [ ] public core stable and trusted.
- [ ] private overlay protocol exists.
- [ ] no private code in public repo.
- [ ] signed feature manifest.
- [ ] paid feature is materially new and not just moved public work.
- [ ] core works fully without engine.

## 13. Конкретный список задач

P0:

1. `vllm/sndr_core/apply/_per_patch_dispatch.py:289-301` - заменить `patch_4_tq_hybrid` на `p4_tq_hybrid`.
2. `vllm/sndr_core/apply/_per_patch_dispatch.py:328-339` - заменить `patch_5_page_size` на `p5_page_size`.
3. `tests/legacy/test_guards.py:146-152` - сделать no-torch aware.
4. `tests/unit/integrations/attention/gdn/test_p103_fla_cliff2.py:186-207` и `vllm/sndr_core/integrations/attention/gdn/p103_fla_cliff2_chunked.py:142-159` - решить torch import contract.

P1:

5. `pyproject.toml:65-80` - определить runtime deps policy.
6. `vllm/sndr_core/cli/__init__.py:1-61` - подключить top-level subcommands.
7. `vllm/sndr_core/model_configs/schema.py:708-750` - убрать production dependency on `/plugin`.
8. `vllm/sndr_core/bundles/_common.py:64-77` - заменить import gate на license/engine_available gate.
9. `vllm/sndr_core/bundles/attention_tq_multi_query.py:1-41` - привести tier к registry.
10. `vllm/sndr_core/license.py:58-60` - заменить placeholder trust anchor перед production license.
11. `tests/conftest.py:27-34` - убрать global legacy license env.

P2:

12. `README.md`, `docs/INSTALL.md`, `docs/PATCHES.md` - обновить PN72/sndr_engine narrative.
13. `vllm/sndr_core/__init__.py:12-26` - обновить migration docstring.
14. `scripts/launch/preflight_check.sh:112-121` - убрать active `_genesis` path requirement.
15. Active tests/probes - убрать default private IP или сделать env-required.
16. Добавить `sndr memory doctor`.
17. Добавить `sndr report bundle`.

P3:

18. Entry point protocol for future private engine.
19. Plugin signatures/provenance.
20. Cross-engine research: LMCache, SGLang HiCache, TensorRT-LLM KV cache reuse.
21. Community benchmark import from club-3090 issues.

## 14. Финальная рекомендация по стратегии

Не делать платную версию сейчас. Текущая правильная стратегия:

1. Все, что уже есть в public repo и на GitHub, оставить в `sndr_core`.
2. `sndr_engine` оставить пустым skeleton с готовым license/overlay контрактом.
3. Сделать public core максимально удобным: installer, launcher, model configs, doctor, report bundle, memory doctor.
4. Наращивать доверие через стабильность, документацию, benchmark reproducibility и быстрые fixes.
5. Private engine использовать позже только для новых разработок, которых нет в public repo и которые реально дают измеримый выигрыш.

Главная ближайшая цель: не новые патчи, а закрытие красных проверок и превращение проекта в самодостаточный, понятный, проверяемый toolchain.

---

## 15. Расширенные планы интеграций (deep-dive по вариантам A-J)

Каждая интеграция здесь раскрывается до уровня: целевой контракт, точные точки соединения в коде, тестовая поверхность, миграционный путь для существующих операторов, rollback-сценарий и трудоёмкость.

### 15.1. Вариант A — Public Core + private Engine overlay через entry points

**Целевой контракт**: `sndr_core` импортирует optional overlay только через `importlib.metadata.entry_points(group="sndr.engine.overlay")`. Никаких прямых import'ов `vllm.sndr_engine.<...>` из core. Overlay package устанавливается отдельно (`pip install vllm-sndr-engine`), регистрирует `register(core_api)` callback, и core зовёт его при boot.

**Точки соединения**:

- `vllm/sndr_core/license.py:engine_available()` — переписать через `importlib.metadata.entry_points`.
- Новый файл `vllm/sndr_core/overlay.py` — публичный API контракт overlay'а:

  ```python
  class OverlayAPI:
      """Контракт для private engine overlay.

      Engine overlay вызывает register_patch / register_kernel /
      register_bundle через этот API; core никогда не импортирует
      engine модули напрямую. Это позволяет публиковать публичный
      core без зависимости от engine, но при наличии overlay'а
      core корректно подключает дополнительный функционал."""
      def register_patch(self, spec: PatchSpec, apply: Callable) -> None: ...
      def register_kernel(self, name: str, ctor: Callable) -> None: ...
      def register_bundle(self, bundle_id: str, plan: dict) -> None: ...
      def require_license(self, feature_id: str) -> bool: ...
  ```

- `vllm/sndr_core/apply/orchestrator.py:run` — после загрузки registry зовёт `_load_engine_overlays()`.
- `vllm/sndr_core/dispatcher/decision.py:_check_engine_tier_eligible` — вместо проверки `import vllm.sndr_engine` зовёт `engine_available()`, который проверяет наличие overlay entry point'а.
- `vllm/sndr_core/bundles/_common.py` — заменить `import vllm.sndr_engine` gate на `engine_available()` функцию.

**Тестовая поверхность**:

- `tests/unit/overlay/test_overlay_contract.py` — все 4 метода `OverlayAPI` имеют негативные тесты при отсутствии overlay'а.
- `tests/unit/overlay/test_engine_available_via_entry_points.py` — мокировать entry point group, проверять обнаружение.
- `tests/integration/test_overlay_smoke.py` — фейковый overlay package в tmpdir, проверка регистрации и activation.

**Миграционный путь для операторов**:

- v11.x: overlay контракт добавлен, но не обязателен. Существующее поведение `engine_available() == False` без overlay package сохраняется.
- v11.y: deprecation warning при прямом import `vllm.sndr_engine.<X>` — пользователи переносятся на overlay API.
- v12.0: прямой import engine забанен через `__init__.py` raise; единственный путь — overlay.

**Rollback**: если overlay контракт не работает у operator'а, fallback в core читает `SNDR_DISABLE_ENGINE_OVERLAY=1` и игнорирует все entry points группы `sndr.engine.overlay`.

**Трудоёмкость**: 5-7 дней, включая тесты и docs.

**Приоритет**: P1, blocker для будущего commercial release.

### 15.2. Вариант B — Bundle layer как главный user-facing слой

**Целевой контракт**: вместо 130 env флагов (`GENESIS_ENABLE_*`) операторы включают/выключают 8-12 крупных bundles (`SNDR_ENABLE_BUNDLE_TURBOQUANT_K8V4`, `SNDR_ENABLE_BUNDLE_TOOL_PARSING_QWEN3CODER`, и т.д.). Каждый bundle декларирует список patches + проверяет совместимость + показывает explain output.

**Точки соединения**:

- `vllm/sndr_core/bundles/_common.py` — расширить контракт `Bundle` с полем `description`, `prerequisites`, `mutually_exclusive_with`, `risk_level`.
- `vllm/sndr_core/bundles/<family>.py` — каждый bundle получает `def explain() -> dict` для CLI.
- `vllm/sndr_core/cli/patches.py` (новый) — `sndr patches bundles list` показывает все bundles + их статус.
- `vllm/sndr_core/model_configs/builtin/*.yaml` — `genesis_env` секция начинает использовать bundle flags вместо individual флагов.

**Тестовая поверхность**:

- `tests/bundles/test_bundle_contract.py` — каждый bundle имеет `description`, `risk_level`, `apply()` testing.
- `tests/bundles/test_mutually_exclusive.py` — взаимоисключающие bundles не активируются вместе без явного override.
- `tests/integration/test_bundle_yaml_migration.py` — старые YAML с individual флагами всё ещё работают.

**Миграционный путь**:

- Step 1 (v11.x): bundles co-existуют с individual флагами. Documentation советует bundles.
- Step 2 (v11.y): YAML configs migrated на bundle flags. Individual флаги deprecated через INFO.
- Step 3 (v12.0): individual флаги работают только для debug/rollback, default exposure через bundles only.

**Rollback**: bundles полностью opt-in через env. Любой bundle падающий по compatibility check'у автоматически отключается с сообщением.

**Трудоёмкость**: 8-10 дней (8 bundles × ~1 день + tests + migration docs).

**Приоритет**: P1, существенно улучшает operator UX.

### 15.3. Вариант C — Community plugin ecosystem

**Целевой контракт**: third-party разработчики могут публиковать `genesis-plugin-<name>` package в PyPI. Plugin регистрируется через entry point group `genesis.community.plugin`, добавляет patches/kernels/configs в registry. Core имеет sandbox layer + signature verification + provenance metadata.

**Точки соединения**:

- `vllm/sndr_core/compat/plugins.py` (расширить существующий) — security layer + signature verification.
- `vllm/sndr_core/dispatcher/registry.py:_KNOWN_FIELDS` — поле `_plugin_origin` уже существует. Добавить `_plugin_signature`, `_plugin_publisher`.
- Новый `vllm/sndr_core/security/plugin_verifier.py` — проверяет signature plugin'а через bundled trust roots.
- Tools `tools/examples/genesis-plugin-hello-world/` — расширить как полный шаблон.

**Тестовая поверхность**:

- `tests/unit/plugins/test_plugin_signature_required_in_prod.py` — plugin без signature не загружается в production mode.
- `tests/unit/plugins/test_plugin_sandbox.py` — plugin не может монкипатчить core registry.
- `tests/integration/test_hello_world_plugin.py` — install + activate + verify.

**Миграционный путь**:

- v11.x: plugin discovery работает без обязательной signature (dev mode).
- v11.y: production mode требует signature, dev mode читает `GENESIS_ALLOW_UNSIGNED_PLUGINS=1`.
- v12.0: signature mandatory кроме dev flag.

**Rollback**: `GENESIS_DISABLE_PLUGINS=1` отключает discovery полностью.

**Трудоёмкость**: 10-14 дней (security infrastructure нетривиально).

**Приоритет**: P3, не блокер; ценность зависит от community demand.

### 15.4. Вариант D — Unified launcher integrations

**Целевой контракт**: `sndr launch` принимает не только builtin presets, но и live конфигурации из Hugging Face Hub, custom YAML по URL, K8s/systemd/docker-compose targets. Один CLI surface для всех runtime backends.

**Точки соединения**:

- `vllm/sndr_core/cli/launch.py:_resolve_config` — расширить поиск:
  - `sndr launch a5000-2x-35b-prod` (текущее, builtin)
  - `sndr launch ~/.sndr/configs/my.yaml`
  - `sndr launch https://hf.co/<user>mage/genesis-config-hub/resolve/main/35b-prod.yaml`
  - `sndr launch hub:<user>mage/community-configs/35b-prod`
- Новый `vllm/sndr_core/model_configs/loader.py:load_from_uri(uri)` — единый загрузчик с проверкой checksum.
- Backend abstraction `vllm/sndr_core/runtime_backend/` — Docker / podman / k8s / bare-metal / systemd рендереры. Каждый backend имеет `render()` метод.

**Тестовая поверхность**:

- `tests/integration/test_loader_local_yaml.py`
- `tests/integration/test_loader_hub_url.py` (с моком HTTP)
- `tests/integration/test_backend_render_each.py` — каждый backend рендерит валидный launch script.

**Миграционный путь**: chunked roll-out — local YAML, потом URL, потом hub: scheme.

**Rollback**: `--strict-builtin` флаг ограничивает loader только builtin presets.

**Трудоёмкость**: 7-10 дней.

**Приоритет**: P2.

### 15.5. Вариант E — Remote GPU server integration

**Целевой контракт**: `sndr` CLI может работать с удалённым GPU сервером через SSH. Operator на Mac dev rig запускает `sndr --remote <user>@<host> launch a5000-2x-35b-prod`, и команда выполняется в нужном container'е на сервере.

**Точки соединения**:

- Новый `vllm/sndr_core/cli/_remote.py` — SSH wrapper. Использует `paramiko` или `subprocess.run(["ssh", host, ...])`.
- Каждая команда в `vllm/sndr_core/cli/__init__.py` принимает `--remote <user@host>` флаг.
- `sndr report bundle --remote` собирает bundle с удалённой машины.

**Тестовая поверхность**:

- `tests/integration/test_remote_dry_run.py` — мокированный SSH endpoint.
- `tests/manual/test_remote_live.py` — пометить `slow`, требует фактический SSH сервер.

**Миграционный путь**: чисто аддитивная фича, не ломает существующее.

**Rollback**: фича опциональна, отсутствие SSH host в args = local mode.

**Трудоёмкость**: 5-7 дней.

**Приоритет**: P2.

### 15.6. Вариант F — Model registry / Hugging Face integration

**Целевой контракт**: `sndr models pull qwen3.6-35b-a3b-fp8` загружает модель с проверкой SHA, регистрирует в local model registry, привязывает к подходящим model_configs. `sndr models doctor <name>` верифицирует целостность.

**Точки соединения**:

- `vllm/sndr_core/compat/models/pull.py` (существует) — расширить с per-shard SHA verification.
- `vllm/sndr_core/compat/models/registry.py` (существует) — добавить association с model_configs.
- Новый `vllm/sndr_core/cli/models.py` — top-level subcommand wrapper.

**Тестовая поверхность**:

- `tests/unit/models/test_pull_sha_verification.py`
- `tests/unit/models/test_registry_association.py`
- `tests/integration/test_models_doctor.py`

**Миграционный путь**: opt-in для существующих операторов; их modesl на disk остаются нетронутыми.

**Rollback**: registry — это JSON файл, удаление = clean state.

**Трудоёмкость**: 4-5 дней.

**Приоритет**: P1.

### 15.7. Вариант G — Upstream vLLM drift watcher (расширение `tools/check_upstream_drift.py`)

**Целевой контракт**: nightly CI запускает drift watcher, сравнивает Genesis patch anchors с upstream vllm HEAD. Если upstream merged equivalent fix, помечает Genesis patch как `merged_upstream` candidate. Если upstream refactored region, помечает как `anchor_drift`.

**Точки соединения**:

- `tools/check_upstream_drift.py` (существует, частично переписан) — расширить:
  - Сравнение по anchor md5 vs текущий upstream commit.
  - Детектирование `merge` keywords в upstream commit message → автоматическая retire-кандидатура.
- Новый `.github/workflows/drift_nightly.yml` — раз в сутки.
- `docs/upstream/UPSTREAM_WATCHLIST.md` (новый) — генерируется автоматически.

**Тестовая поверхность**: `tests/integration/test_drift_watcher_synthetic.py` — синтетический mini-vllm дерево.

**Миграционный путь**: чисто observability, без runtime impact.

**Rollback**: в CI workflow можно отключить job без последствий.

**Трудоёмкость**: 4-6 дней.

**Приоритет**: P1, экономит много времени на pin upgrade.

### 15.8. Вариант H — Memory/cache integrations

Уже частично описаны в §10 текущего файла. Дополнительные точки:

- `vllm/sndr_core/runtime/memory_metrics.py:genesis_memory_summary()` (существует) — расширить с per-component breakdown (weights / KV / scratch / CUDA graph reserve / fragmentation).
- Новый `vllm/sndr_core/runtime/memory_estimator.py` — статический estimator до launch'а (что выводит `sndr memory explain`).
- `vllm/sndr_core/cache/response_cache.py` (существует, prefix cache layer) — интегрировать pluggable eviction policy из vllm#40270.

**Трудоёмкость**: 6-8 дней.

**Приоритет**: P0 (через `sndr memory explain` пользовательский запрос #1).

### 15.9. Вариант I — Observability/support integration

**Целевой контракт**: `sndr report bundle` собирает structured artifacts. Опционально автоматически анонимизирует и предлагает upload в support endpoint (если operator opt'нул в `SNDR_TELEMETRY_ENABLE=1`).

**Точки соединения**:

- Новый `vllm/sndr_core/cli/report.py`.
- `vllm/sndr_core/runtime/redact.py` (новый) — masks IPs / hostnames / API keys / license tokens.
- Опционально (P3) `vllm/sndr_core/compat/telemetry.py` (существует) — usage signals.

**Трудоёмкость**: 4-5 дней (без telemetry uploader, который — P3).

**Приоритет**: P0.

### 15.10. Вариант J — club-3090 support integration

**Целевой контракт**: каждая закрытая club-3090 issue превращается в:
1. Регрессионный тест (если связан с patch).
2. Doctor rule (если связан с environment detection).
3. Recipe в `docs/COOKBOOK.md` (новый файл) — конкретные команды + конфигурации для типового сценария.

См. §17 ниже для детального плана решений по каждому открытому issue.

**Трудоёмкость**: 8-12 дней (для всех 4-х открытых issues).

**Приоритет**: P1.

---

## 16. Полный список upstream vLLM PRs для backports (из аудита 2026-05-08 §10)

В §9 текущего файла раскрыты #38479, #40914, #40069. Расширяю до полного списка из аудита:

### 16.1. vLLM PR #40269 — Probabilistic draft rejection

- **Источник**: [vllm#40269](https://github.com/vllm-project/vllm/pull/40269).
- **Что важно**: передача `draft_probs` улучшает acceptance rate в spec decoding.
- **Где интегрируется**:
  - `vllm/sndr_core/integrations/spec_decode/p77_adaptive_ngram_k.py`
  - `vllm/sndr_core/integrations/spec_decode/pn40_*` (DFlash family).
  - `vllm/sndr_core/integrations/serving/p70_auto_strict_ngram.py`.
- **Что сделать**:
  - Новый `PN90` патч: wrap `Proposer.propose()` для прокидывания probabilities. Fallback на текущее поведение если probs отсутствуют.
  - Метрики: `acceptance_rate`, `rollback_count` в boot summary.
  - A/B harness: strict-ngram vs probabilistic на tool_call clean rate Qwen3.6.
- **Risk**: MEDIUM. Tool-call clean rate является operator-visible.
- **Effort**: 5-7 дней.
- **Priority**: P1.

### 16.2. vLLM PR #40270 — Pluggable GPU KV cache eviction (LRU/2Q/ARC)

- **Источник**: [vllm#40270](https://github.com/vllm-project/vllm/pull/40270).
- **Что важно**: защищает hot prefixes от scan pollution. Ключевая фича для agent / RAG / multi-turn workloads.
- **Где интегрируется**:
  - `vllm/sndr_core/cache/response_cache.py` — существующий prefix cache layer.
  - `vllm/sndr_core/model_configs/schema.py` — новое поле `CacheConfig`.
  - `vllm/sndr_core/integrations/kv_cache/*`.
- **Что сделать**:
  - Новый `PN91` патч: подключает eviction policy hook. Конфиг `--kv-eviction-policy {lru, 2q, arc}` с default `2q`.
  - Метрики: `kv_eviction_rate`, `kv_hit_rate`, `kv_hot_prefix_age`.
  - Boot summary: prefix cache health section.
  - Doctor rule: hit rate < 30% + agent-style workload → suggest 2Q/ARC.
- **Risk**: LOW (pluggable hook, fallback на LRU).
- **Effort**: 3-4 дня.
- **Priority**: P1.

### 16.3. vLLM PR #37160 — CPU KV offload connector

- **Источник**: [vllm#37160](https://github.com/vllm-project/vllm/pull/37160).
- **Что важно**: spillover cold KV в system RAM. Single-card (24 GB) операторы получают 96K+ context.
- **Что сделать**:
  - Новый `PN92` патч: config validator + scheduler hook.
  - Single-card preset (`a5000-1x-27b-int4-tested`, новый `3090-1x-27b-int4`) получает поле `kv_offload_size: 24GB`.
  - Preflight: проверка PCIe Gen3/Gen4 link width; warn если bandwidth < 8 GB/s.
- **Compatibility**: НЕ совместим с hybrid Mamba (Qwen3.5/3.6 GDN). Жёсткий guard `applies_to.is_hybrid: [False]`.
- **Risk**: MEDIUM. Hybrid Mamba incompatibility silently degrades в crashes без guard.
- **Effort**: 5-7 дней (compatibility matrix).
- **Priority**: P1 (community demand, club-3090#58).

### 16.4. vLLM PR #37190 — MoE expert CPU offloading

- **Источник**: [vllm#37190](https://github.com/vllm-project/vllm/pull/37190).
- **Что важно**: загрузка Qwen3.6-35B-A3B (256 experts × 8 active) на single 24 GB карте через cold-expert offload.
- **Что сделать**:
  - Research-only branch первое время. НЕ default-on.
  - Новый experimental `PN93` (`lifecycle="experimental"`, `implementation_status="placeholder"`).
  - Estimator: `sndr memory explain --moe-cpu-offload <size>` показывает throughput trade-off.
- **Risk**: HIGH. Throughput regression жёсткий (30-60% TPS hit).
- **Effort**: 7-10 дней.
- **Priority**: P2.

### 16.5. vLLM PR #38330 — Multimodal encoder cache

- **Источник**: [vllm#38330](https://github.com/vllm-project/vllm/pull/38330).
- **Что важно**: cache vision encoder embeddings для повторных images (agent screen-grab loops, RAG с images).
- **Что сделать**:
  - Новый `PN94`: LRU image-embedding cache, bounded `GENESIS_PN94_CACHE_MAX_MIB` (default 512).
  - Hash image bytes; lookup before encoder forward pass.
  - Cache invalidation on preprocessor version change.
- **Risk**: MEDIUM. Non-deterministic preprocessors отравляют cache; нужен preprocessor-version key.
- **Effort**: 4-6 дней.
- **Priority**: P2 (vision не primary workload Genesis, но Qwen3.6-VL делает её актуальной).

### 16.6. vLLM PR #40281 — Gemma4 support (watchlist)

- **Источник**: [vllm#40281](https://github.com/vllm-project/vllm/pull/40281).
- **Статус**: PR closed unmerged. Tracking only.
- **Что сделать**:
  - Stub Gemma4 model_config preset с `lifecycle="research"`, `implementation_status="placeholder"`.
  - Запись в `docs/upstream/UPSTREAM_WATCHLIST.md`.
  - Когда upstream merge'ит: создать patch family + preset + A/B vs Qwen3.6.
- **Effort**: 3-5 дней (когда landed).
- **Priority**: P3.

### 16.7. vLLM PR #40265 — FLA TP device index overflow guard

- **Источник**: [vllm#40265](https://github.com/vllm-project/vllm/pull/40265).
- **Статус**: closed unmerged. Pattern важен.
- **Что сделать**:
  - Новый regression test: `tests/unit/integrations/attention/gdn/test_fla_tp_device_index_no_overflow.py`. Синтетический `tp_size=8`, проверка `int32` overflow boundary.
  - Guard в `vllm/sndr_core/integrations/attention/gdn/p7_gdn_dual_stream.py` — tightening device-index dtype до `int64`.
- **Risk**: LOW.
- **Effort**: 1-2 дня.
- **Priority**: P2.

### 16.8. Сводная таблица backports

| PR | Patch ID | Lifecycle | Effort | Risk | Priority |
|---|---|---|---:|---|---|
| #38479 | parity tracking (P3..P67 audit) | существующий | 7-10 d | HIGH | P1 |
| #40269 | PN90 | experimental | 5-7 d | MED | P1 |
| #40270 | PN91 | experimental → stable | 3-4 d | LOW | P1 |
| #37160 | PN92 | experimental | 5-7 d | MED | P1 |
| #37190 | PN93 | experimental | 7-10 d | HIGH | P2 |
| #38330 | PN94 | experimental | 4-6 d | MED | P2 |
| #40281 | (watchlist) | research | 3-5 d (later) | LOW | P3 |
| #40265 | guard в P7 | full | 1-2 d | LOW | P2 |

Совокупная трудоёмкость: ~35-50 дней по всему списку.

---

## 17. Решения по открытым club-3090 issues

### 17.1. club-3090 #60 — Image regression / GPTQ Marlin repack OOM

- **Симптом**: после bump nightly image EngineCore не стартует, OOM в Marlin repack.
- **Корень**: nightly image меняет vllm/torch/quant backend без явного version pin.
- **Решение**:
  - В `vllm/sndr_core/model_configs/schema.py:DockerConfig` добавить поле `image_digest: Optional[str]` (sha256:...).
  - `sndr launch --strict-image` отказывается launch'ить если container digest не совпадает с config'ом.
  - Новый `sndr doctor image-regression`:
    - читает container's vllm version + nvidia-smi driver + pip show torch;
    - сравнивает с `KNOWN_GOOD_IMAGES` allowlist в `vllm/sndr_core/compat/image_allowlist.py` (новый);
    - показывает recommendation: какой digest pinned работает.
  - Marlin repack scratch estimator: `weights × 1.5x` vs free VRAM. Warn если cutting it close.
- **Trial validation tests**:
  - `tests/integration/test_marlin_repack_estimator.py` — известные модели + ожидаемые scratch sizes.
- **Effort**: 2 дня.

### 17.2. club-3090 #58 — Long-context + vision OOM на 3090 (24 GB)

- **Симптом**: 50 MiB-per-call `torch.empty_like(v)` allocation фрагментирует за длинную сессию. Single-card 24 GB hits OOM.
- **Корень**: в GDN/FFN/chunk path Genesis не пре-аллоцирует scratch; каждый forward = новый allocation.
- **Решение**:
  - Memory scratch profiler в `vllm/sndr_core/runtime/memory_metrics.py`:
    - Названные tracking points: prefill_scratch, decode_scratch, gdn_chunk_scratch, vision_encoder_scratch, activation_spikes.
    - Каждая allocation > 10 MiB логируется.
  - Новый preset `3090-1x-low-vram-24gb.yaml` с `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` только при detected fragmentation risk.
  - Doctor rule: total expected scratch + KV + weights > 22 GiB → refuse launch с explanatory error.
- **Trial validation**:
  - Simulated 60-minute long-context soak с PN59 ON.
- **Effort**: 3 дня.

### 17.3. club-3090 #50 — WSL2 device-not-ready на 157K context

- **Симптом**: long context, WSL2, 2x3090, FP8 KV, chunked prefill, MTP. Ошибка `device not ready`.
- **Корень**: WSL2 имеет уникальные pin-memory + GPU runtime quirks при высокой concurrency.
- **Решение**:
  - WSL detection в `vllm/sndr_core/detection/runtime_caveat.py` (расширить existing):
    - `/proc/version` содержит `Microsoft` / `WSL2`.
    - `nvidia-smi` returns CSV format quirk (отличается от native Linux).
  - Doctor checks:
    - pin-memory warning;
    - driver/CUDA compat для WSL2 (в общем требуется свежее);
    - PCIe topology;
    - Docker GPU runtime presence (`docker info | grep nvidia`).
  - `probe_max_ctx.sh` интегрировать в doctor:
    - binary search safe max-model-len;
    - запись recommended config в `~/.sndr/probe_max_ctx_<gpu>.yaml`.
- **Trial validation**:
  - Manual: live WSL2 box (cross-rig partner).
  - Unit: WSL detection mock test.
- **Effort**: 2-3 дня.

### 17.4. club-3090 #47 — Read-only mount blocks text-patches

- **Симптом**: operator mounted Genesis tree read-only в container; text-patcher молча failed.
- **Решение**:
  - В `vllm/sndr_core/core/text_patch.py:apply` preflight: `os.access(target, os.W_OK)` check; если False, raise structured error.
  - В `vllm/sndr_core/cli/launch.py` preflight: scan model_config docker.mounts для `:ro` flags на paths внутрь vllm tree; warn.
  - Documentation overlay strategy в `docs/INSTALL.md` (mount overlay so patches survive container restart):

    ```bash
    # Overlay mount: patches live in writable upper layer, vllm
    # site-packages stays read-only at lower layer.
    mount -t overlay overlay -o lowerdir=/usr/local/lib/python3.12/dist-packages/vllm,\
           upperdir=/var/lib/sndr/overlay-upper,workdir=/var/lib/sndr/overlay-work \
           /usr/local/lib/python3.12/dist-packages/vllm
    ```
- **Effort**: 1-2 дня.

### 17.5. Новый файл `docs/COOKBOOK.md`

Каждая закрытая club-3090 issue → recipe раздел с:

- симптом (как operator его описывает);
- detection (как doctor ловит);
- workaround (что сделать чтобы запустить);
- fix (если patch есть);
- prevention (как избежать в будущем).

---

## 18. Скелеты новых CLI команд (детально)

### 18.1. `sndr doctor --full` — расширение существующего

Существующий `vllm/sndr_core/compat/doctor.py` (573 строки) имеет 12 секций. Не хватает 6 секций по аудиту §12.1.

**Новые секции (добавить функции в `compat/doctor.py`)**:

```python
def _section_wsl() -> dict[str, Any]:
    """WSL2 detection + pin memory warnings + GPU runtime presence.

    Возвращает:
      {is_wsl: bool, kernel: str, distro: str, pin_memory_ok: bool,
       docker_gpu_runtime: bool, recommendations: list[str]}
    """

def _section_image() -> dict[str, Any]:
    """Container image digest verification.

    Возвращает:
      {expected_digest: str|None, actual_digest: str|None,
       drift: bool, allowlist_status: 'known_good'|'unknown'|'known_bad'}
    """

def _section_mounts() -> dict[str, Any]:
    """Mount writability scan.

    Возвращает:
      {mounts: [{src, dst, ro: bool, writable: bool}],
       writability_violations: list[str]}
    """

def _section_license() -> dict[str, Any]:
    """License gate status.

    Возвращает:
      {trust_anchor: str (zero|real),
       license_present: bool, license_status: LicenseStatus,
       legacy_mode_active: bool}
    """

def _section_engine() -> dict[str, Any]:
    """Optional engine overlay status.

    Возвращает:
      {engine_available: bool, overlay_packages: list[str],
       version: str|None}
    """

def _section_remote_capability() -> dict[str, Any]:
    """Remote/SSH support sanity.

    Возвращает:
      {ssh_keys_present: bool, can_resolve_remote_targets: bool}
    """
```

**Новые CLI флаги**:

- `--full` — все 18 секций (текущие 12 + 6 новых).
- `--json` — machine-readable output.
- `--remote <user>@host` — выполняет doctor в SSH-сессии.
- `--container vllm-server` — выполняет doctor в `docker exec`.
- `--redact` — маскирует IPs, hostnames, API keys.

**Effort**: 4 дня.

### 18.2. `sndr patches` — новый top-level subcommand

Файл: `vllm/sndr_core/cli/patches.py` (новый).

```python
"""sndr patches — patch transparency CLI.

Subcommands:
  list [--tier engine|community] [--lifecycle stable|...]
       [--default-on] [--changed-since v10.0]
  explain <PATCH_ID>      — full metadata + credit + tests.
  doctor                  — every-patch dispatcher + anchor health.
  plan --preset <KEY>     — what WOULD apply for a given preset.
  diff-upstream           — patches whose upstream merged.
  bundles list/explain    — bundle-level commands.
"""
```

**Реализация**:

- `list` queries `dispatcher.PATCH_REGISTRY` + filters via flags + prints rich table (формат `tabulate` или ASCII art).
- `explain` deep-prints одного патча + читает его docstring + `grep tests/` для related tests.
- `plan` запускает dispatcher decision per patch с env пресела, печатает would-apply / would-skip breakdown.
- `diff-upstream` зовёт `tools/check_upstream_drift.py`.

**Test surface**: `tests/unit/cli/test_patches_*` — golden output tests для `explain P67`, `plan a5000-2x-35b-prod`.

**Effort**: 4-5 дней.

### 18.3. `sndr memory explain` — VRAM budget breakdown

Файл: `vllm/sndr_core/cli/memory_explain.py` (новый).

**Output sections**:

```text
Memory budget for preset: a5000-2x-35b-prod
GPU 0 (RTX A5000, 24 GiB)
─────────────────────────────────────────
Model weights (after TP shard)          14.2 GiB
KV cache (320K context, fp8_e5m2)        6.8 GiB
Activations / scratch                    2.1 GiB
CUDA graph reserve                       0.4 GiB
Marlin repack scratch                    1.5 GiB (peak, transient)
─────────────────────────────────────────
Subtotal (committed)                    23.5 GiB / 24 GiB (98%)
Headroom for fragmentation               0.5 GiB
─────────────────────────────────────────
⚠ WARNING: very tight budget. Recommendations:
  - Drop max-model-len to 256K → −1.8 GiB KV
  - Enable PN59 streaming-GDN → −0.4 GiB scratch
```

**Реализация**:

- Per-component estimators:
  - `_estimate_weights(model_path, tp_size, dtype)` — читает `config.json` + `safetensors` headers.
  - `_estimate_kv(max_len, dtype, n_layers, n_heads, head_dim)` — формула `2 × n_layers × n_heads × head_dim × max_len × bytes_per_element / tp_size`.
  - `_estimate_marlin_scratch(model_path)` — peak repack memory.
  - `_estimate_cuda_graph_reserve(max_num_seqs, batch_size)`.
- CLI presents waterfall: cumulative vs cap. Warn at >85%.
- Hooks в `sndr launch --dry-run` так что preview emits breakdown automatically.

**Test surface**:

- `tests/unit/cli/test_memory_estimators.py` — известные shapes (Qwen3.6-35B-A3B-FP8, Qwen3.6-27B-int4-AutoRound, Gemma3-27B-int4).
- `tests/integration/test_memory_estimate_vs_actual.py` — live VRAM measurement vs estimate (acceptable +/- 1 GiB).

**Effort**: 5-7 дней.

### 18.4. `sndr models` — model registry browser

Файл: `vllm/sndr_core/cli/models.py` (новый, glue layer над существующим `compat/models/*`).

**Subcommands**:

```text
sndr models list                    — Genesis-recognized models + status
sndr models pull <slug>             — wraps existing pull.py
sndr models register <path>         — adds operator-local model
sndr models doctor [<name>]         — verify SHAs + config.json + tokenizer
sndr models compatibility <name>    — what model_configs work with this model
```

**Effort**: 2-3 дня.

### 18.5. `sndr report bundle` — diagnostic bundle exporter

Файл: `vllm/sndr_core/cli/report.py` (новый).

**Bundle contents** (9 артефактов в `tar.gz`):

1. `sndr doctor --full --json` output.
2. `sndr patches list --json`.
3. `sndr launch --dry-run <preset>` rendered script.
4. Last 200 lines of vllm boot log (`docker logs <container> --tail 200`).
5. `~/.sndr/host.yaml` (redacted).
6. `nvidia-smi -q` output.
7. `pip freeze` of operator's env.
8. `git log -10` of Genesis checkout.
9. vLLM image digest + `docker inspect` summary.

**Redaction utility** в новом `vllm/sndr_core/runtime/redact.py`:

```python
"""Маскирует чувствительные данные перед share с support.

Default rules:
  - IPv4 / IPv6 addresses → <IP>
  - hostnames в SSH targets → <HOSTNAME>
  - API keys (Bearer ... / GENESIS_*_KEY=) → <REDACTED>
  - License tokens (anything matching base64.base64 pattern) → <LICENSE>
  - Filesystem paths под $HOME → /home/<USER>

Override rules: пользователь может расширить через
~/.sndr/redact_rules.yaml.
"""
```

**Public issue template**: `.github/ISSUE_TEMPLATE/bug_report.yml` требует attachment of bundle ИЛИ explicit "I cannot run sndr report bundle because ..." reason.

**Test surface**:

- `tests/unit/cli/test_redact.py` — синтетические inputs.
- `tests/integration/test_report_bundle.py` — bundle создаёт parseable tarball со всеми 9 артефактами.

**Effort**: 4-5 дней.

---

## 19. Карта зависимостей между sprints + критический путь

```text
┌─────────────────────────────────────────────────────────────┐
│ Sprint 0: Красный build (P0-1..P0-4)                        │
│ Закрывает: blocking failures; устраняет красные CI signals  │
└─────────┬───────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────┐  ┌───────────────────────────────┐
│ Sprint 1: Self-contained │  │ Sprint 2: CLI unification     │
│ package (deps, schemas,  │  │ (top-level subcommands)       │
│ no /plugin зависимость)  │  │ → разблокирует §18.x команды  │
└─────────┬────────────────┘  └─────────┬─────────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────────────┐  ┌───────────────────────────────┐
│ Sprint 3: Docs cleanup   │  │ Sprint 4: Engine boundary     │
│ (sndr_engine narrative,  │  │ hardening (overlay protocol)  │
│ private path purge)      │  │                                │
└─────────┬────────────────┘  └─────────┬─────────────────────┘
          │                              │
          └──────────┬───────────────────┘
                     ▼
          ┌──────────────────────────────────────┐
          │ Sprint 5-7: feature work             │
          │  — Sprint 5: Memory/cache (§16, §18.3)│
          │  — Sprint 6: Qwen/spec stab (§16.1-2) │
          │  — Sprint 7: Bench/remote (§15.5)     │
          └──────────┬───────────────────────────┘
                     ▼
          ┌──────────────────────────────────────┐
          │ Sprint 8: Release/security           │
          │  Trust anchor; signed releases;      │
          │  SBOM; constraints                   │
          └──────────┬───────────────────────────┘
                     ▼
          ┌──────────────────────────────────────┐
          │ Sprint 9: Upstream integration       │
          │  Drift watcher continuous; PR        │
          │  retirement automation               │
          └──────────────────────────────────────┘
```

**Критический путь**: Sprint 0 → 1 → 2 → 5 → 8 (release blocker).

**Параллелизуемые ветки** (можно вести двум разработчикам):
- A: Sprint 1 + 5 (packaging + memory)
- B: Sprint 2 + 6 (CLI + Qwen stab)
- C: Sprint 3 + 7 (docs + bench)

**Sprint 4 (engine boundary)** должен идти ПОСЛЕ Sprint 0-2 чтобы не блокировать installer fixes.

**Sprint 9 (upstream integration)** — параллельный observability track, можно запускать с самого начала.

---

## 20. Rollback планы для P0/P1 fixes

Каждый fix должен иметь известный rollback путь чтобы regression в production был обратимым.

### 20.1. P0-1 (P4/P5 typo fix)

- **Изменение**: `patch_4_tq_hybrid` → `p4_tq_hybrid`, `patch_5_page_size` → `p5_page_size` в `_per_patch_dispatch.py`.
- **Risk**: typo fix не должен ничего поломать; опасность только в случае если CI не покрывает реальный wiring (см. P1-9 в предыдущих audit'ах).
- **Rollback**: revert commit; `_per_patch_dispatch.py` достаточно автономный.
- **Detection**: после fix `sndr install --dry-run --non-interactive` должен exit 0; если exit ≠ 0, rollback.

### 20.2. P0-2 (no-torch test contract)

- **Изменение**: тесты получают `requires_torch` marker; module-level torch imports перенесены в TYPE_CHECKING / function-local.
- **Risk**: production кода которое legitimately нуждается в torch на module top-level может быть случайно сломано.
- **Rollback**: marker fix чисто in tests/, не в production коде. Production код changes — per-file revert.
- **Detection**: `pytest --tb=line` должен пройти оба mode'а: с torch и без torch (через CI no-torch job).

### 20.3. P0-3 (base wheel deps policy)

- **Изменение**: `pyproject.toml` `dependencies = ["pyyaml>=6.0", "requests>=2.28"]` (или сделать optional через extras).
- **Risk**: новые transitive deps могут конфликтовать с vllm install в существующих environments.
- **Rollback**: вернуть `dependencies = []`. Документировать что operator must install pyyaml/requests manually.
- **Detection**: `pip install -e . && python -c "from vllm.sndr_core.compat import schema_validator"` в чистом venv. Если ImportError, rollback.

### 20.4. P0-4 (CLI subcommand registration)

- **Изменение**: `vllm/sndr_core/cli/__init__.py` регистрирует `doctor`, `verify`, `model-config`, `plugins`, `patches`, `report`.
- **Risk**: scope creep — каждая subcommand регистрация подгружает новый модуль и может потянуть зависимости.
- **Rollback**: subcommand list — это data structure; удаление одной строки удаляет команду. Полная rollback тривиальна.
- **Detection**: `sndr <subcommand> --help` для каждой должно exit 0.

### 20.5. P1-x rollback общая стратегия

Все P1 изменения:

- помечены в commit message паттерном `[P1-N]`;
- содержат `Reverts: <revert-anchor>` в bottom для облегчения автоматического revert;
- каждое изменение в `_VALID_*` enum'ах в `dispatcher/audit.py` помечено как `# Added in P1-X` для tracking.

---

## 21. Память — что уже пробовали (mining auto-memory)

Анализ feedback memory (~50 entries) показывает закономерности; интегрирую их в roadmap чтобы не повторять циклы.

### 21.1. Что НЕ работает (avoid в planning)

- **Synthetic acceptance mode** ([feedback_synthetic_mode_breaks_tools_api.md](feedback_synthetic_mode_breaks_tools_api.md)): vllm#40662 +12% TPS на 35B, но breaks OpenAI tools API. NOT deploy ready. Это коррелирует с Sprint 4.2 (#40269 prob draft rejection) — нужна less-aggressive sweep с tool-call as primary metric.
- **P67 Genesis kernel `USE_UPSTREAM=0`** ([feedback_p67_genesis_kernel_quality_mirage.md](feedback_p67_genesis_kernel_quality_mirage.md)): +42% TPS оказались quality mirage — drift в acceptance вызвал tool-call repetition spam. PROD locked `USE_UPSTREAM=1`. Архитектурный rewrite needed для real gain.
- **P67 fp16 dot precision** ([feedback_p67_fp16_dot_regression.md](feedback_p67_fp16_dot_regression.md)): -13% regression на TQ k8v4 stack. Phase 2 fp16 dot kept opt-in only.
- **Triton 3.6 flags** ([feedback_triton36_flags_regress_split_m.md](feedback_triton36_flags_regress_split_m.md)): `disallow_acc_multi_buffer` + `loop_unroll_factor=2` regress split-M kernel −6.5%. Generic compiler hints НЕ переносятся; всегда per-kernel A/B.
- **P104 L2 persistence** ([feedback_p104_l2_persistence_thrashing.md](feedback_p104_l2_persistence_thrashing.md)): −16.2% regression через cache thrashing. Architectural mismatch с 32+ layer transformer + KV >> L2. Removed.
- **PN29 GDN scale fold** ([feedback_pn29_scale_fold_rejected.md](feedback_pn29_scale_fold_rejected.md)): rejected на validation.
- **Prefix cache + spec-decode** ([project_genesis_27b_prefix_cache_fix.md](project_genesis_27b_prefix_cache_fix.md)): `--enable-prefix-caching` + MTP `accept>1` = DS conv state crash. Sprint 4.3 (KV eviction) revisits с pluggable policy.

### 21.2. Что работает (валидируется в roadmap)

- **PN59 streaming-GDN** ([feedback_pn59_validated_prod.md](feedback_pn59_validated_prod.md)): −142 MiB/GPU + 95% drift reduction на 60-min soak. PROD-validated.
- **PN25 Silu inductor pool** ([feedback_pn25_real_win_27b_tq_k8v4.md](feedback_pn25_real_win_27b_tq_k8v4.md)): −871 MiB на 27B+TQ k8v4. CV 0.25 → 0.21. Promoted.
- **MTP K=3** ([feedback_mtp_k3_optimal_35b_prod.md](feedback_mtp_k3_optimal_35b_prod.md)): K=2 (167/190 TPS, 4/5 tool ❌), K=3 (228/204, 10/10 ✅), K=4 (198/190, 5/5). EAGLE3/P-EAGLE/ngram все dead-end для Qwen3.6-35B-A3B+TQ k8v4.
- **TurboQuant speed bonus** ([feedback_turboquant_speed_bonus.md](feedback_turboquant_speed_bonus.md)): +11.3% TPS bonus от TQ vs auto KV. TQ — НЕ overhead; packed-slot layout помогает cache locality.
- **Strict ngram** ([project_genesis_v7_13_strict_ngram_breakthrough.md](project_genesis_v7_13_strict_ngram_breakthrough.md)): `prompt_lookup_min=8` → 100% clean (single-query) / 96% (multi-query diverse).

### 21.3. Architectural learnings (feed в Sprint 4)

- **Genesis patches are source-level edits** ([feedback_genesis_patches_are_source_level.md](feedback_genesis_patches_are_source_level.md)): apply_all = parent-process diagnostic only; runtime monkey-patches НЕ survive vllm spawn workers. Worker-visible changes требуют source-level edits в vllm core. Это объясняет почему PN78 был tombstoned.
- **Container R/W layer trap** ([feedback_container_rw_layer_trap.md](feedback_container_rw_layer_trap.md)): compose stop/start preserves patched fs → monolith re-run fails on anchor drift. Use `compose down && up -d` чтобы reset.
- **Replace parameter required** ([feedback_replace_parameter_required.md](feedback_replace_parameter_required.md)): never raw `layer.weight = nn.Parameter(...)` — orphans weight_loader callback. Always use `vllm.model_executor.utils.replace_parameter()`.
- **Ampere ceiling** ([feedback_ampere_a5000_kernel_ceiling.md](feedback_ampere_a5000_kernel_ceiling.md)): realistic max +15% над PROD 162 TPS на Ampere. 200+ TPS требует Blackwell или 4-bit KV. Sprint 4.1 (TQ) + 4.2 (prob draft) — кандидаты.

### 21.4. Process learnings (feed в release discipline)

- **Verify by reproducer, not anchor passing** ([feedback_verify_by_reproducer.md](feedback_verify_by_reproducer.md)): anchor verification + tests passing ≠ patch fixes the bug. Always run actual workload reproducer in blue/green container.
- **Comprehensive research first** ([feedback_comprehensive_research_first.md](feedback_comprehensive_research_first.md)): research subagent prompts должны быть rigorous (≥80% confidence threshold). Generic kernel optimization advice не переносится.
- **No push without explicit approval** ([feedback_no_push_without_explicit_approval.md](feedback_no_push_without_explicit_approval.md)): local commits OK, push to GitHub только по явному "ok push" Sander'а.
- **No internal docs in public** ([feedback_no_internal_docs_in_public.md](feedback_no_internal_docs_in_public.md)): community surveys / sprint reports на русском / Tier action plans — local only. `docs/_internal/` уже gitignored.
- **Verify links before posting** ([feedback_verify_links_before_posting.md](feedback_verify_links_before_posting.md)): перед posting upstream comments с `github.com/Sandermage/...` URLs, verify files pushed to origin/main. Local-only files = immediate 404.

### 21.5. Тестовые learnings

- **Close bug class requires matrix** ([feedback_close_bug_class_requires_matrix.md](feedback_close_bug_class_requires_matrix.md)): закрытие bug class требует matrix testing (model × hardware × config).
- **Verify all models** ([feedback_verify_all_models.md](feedback_verify_all_models.md)): когда patch claims "fixes Qwen3 stuff", тест нужен на всех Qwen3 variants.
- **Needle 10K LITM reproducible** ([feedback_needle_litm_10k_reproducible.md](feedback_needle_litm_10k_reproducible.md)): 3× re-run all 1K/51K/92K FOUND, 10K MISS. Classic Lost-In-The-Middle. README headline = 3/4 honest.

---

## 22. Метрики успеха + Definition of Done

### 22.1. Per-sprint success metrics

| Sprint | Success metric | Measurement |
|---|---|---|
| 0 (red build) | `pytest -q` exit 0; `sndr install --dry-run -y` exit 0 | CI |
| 1 (self-contained) | `pip install vllm-sndr-core` в чистом venv → `sndr --help` works | CI new job |
| 2 (CLI unification) | `sndr {doctor,patches,memory,models,report} --help` все exit 0 | unit tests |
| 3 (docs cleanup) | `rg "_genesis|192\.168\.1\.10|$HOME|~/\.genesis"` в `docs/`, `README.md`, `*.yml`, активный `vllm/sndr_core/` → 0 hits | CI grep gate |
| 4 (engine boundary) | `import vllm.sndr_engine` exit 0 в empty package; `engine_available()` False; core works fully | smoke test |
| 5 (memory/cache) | `sndr memory explain a5000-2x-35b-prod` показывает <1 GiB ошибки vs actual; KV eviction LRU↔2Q switch noop в functional smoke | live test |
| 6 (Qwen/spec) | tool_call clean rate Qwen3.6 27B/35B remains 10/10; SSE silence detector ловит regression | regression suite |
| 7 (bench/remote) | `sndr launch --remote <user>@host` works; bench reproducible on 2 rigs | manual + CI |
| 8 (release) | wheel signed; SBOM generated; constraints.txt pinned; trust anchor real | release artifacts |
| 9 (upstream integration) | drift watcher отчитывается nightly; ≥3 patches retired automatically per quarter | dashboard |

### 22.2. Production-readiness DoD (полный checklist)

Дополняет §12 текущего файла.

**Code quality**:

- [ ] AST parse all `.py` exit 0.
- [ ] `pytest -q` 100% pass с torch.
- [ ] `pytest -q` 100% pass БЕЗ torch (no-torch job).
- [ ] `mypy --strict vllm/sndr_core/cli/` exit 0.
- [ ] Coverage report ≥75% on `vllm/sndr_core/`.

**Operational**:

- [ ] `sndr doctor --full --json` valid JSON output.
- [ ] `sndr launch --dry-run` любой builtin preset exit 0 + правильный warning when host.yaml missing.
- [ ] `sndr install --dry-run -y` exit 0.
- [ ] `sndr report bundle` создаёт valid tar.gz всеми 9 артефактами.

**Security**:

- [ ] Trust anchor pubkey не placeholder (real Ed25519).
- [ ] `git secrets --scan` clean.
- [ ] No private key в repo (`tools/license_keygen.py generate-keypair` documented but private key gitignored).
- [ ] SBOM published per release.

**Documentation**:

- [ ] README показывает реальное current state.
- [ ] CHANGELOG имеет entry для каждой опубликованной версии.
- [ ] `docs/COOKBOOK.md` имеет recipe для каждой closed club-3090 issue.
- [ ] `docs/INSTALL.md` тестировался cross-rig (не только Sander's box).

**Community**:

- [ ] Public issue template требует `sndr report bundle` или explicit reason.
- [ ] CHANGELOG entries link к community issues когда applicable.
- [ ] `.github/CODEOWNERS` — minimum дает Sander review request.

### 22.3. Anti-goals (НЕ оптимизировать на эти метрики)

- **Patch count** — больше не лучше. Целевой количество patches должно сокращаться по мере того как upstream merges.
- **TPS на benchmark** — без acceptance/quality regressions. Synthetic +12% не stoit OpenAI tools API breakage.
- **Star count GitHub** — vanity metric. Useful как sanity check (не отрицательный sign), не как priority driver.
- **Lines of code** — should DECREASE при правильном refactoring (см. install.sh 783 → 106 lines).

---

## 23. Заключение и план первой недели

Расширенный roadmap (§15-22) даёт всё необходимое для перехода от "audit closed" к "production beta release". Ниже — конкретный план первой недели чтобы запустить движение.

### 23.1. Неделя 1 (рекомендуемая последовательность)

| День | Задача | Sprint | Effort |
|---|---|---|---|
| 1 | P4/P5 typo fix (§13 P0 #1, #2) | 0 | 0.5 d |
| 1-2 | No-torch test markers (§13 P0 #3, #4) | 0 | 1 d |
| 2-3 | Base wheel deps policy (§13 P1 #5) + smoke test | 1 | 1.5 d |
| 3-4 | Top-level CLI registration (§13 P1 #6) + skeletons §18.1-2 | 2 | 2 d |
| 4-5 | `/plugin` dependency removal (§13 P1 #7) | 1 | 1 d |
| 5 | Trust anchor real key generation + commit (§7.1, §13 P1 #10) | 8 | 0.5 d |
| 5 | conftest legacy license unhook (§13 P1 #11) | 8 | 0.5 d |

После недели 1: `pytest -q` green, CLI имеет полный subcommand surface, base wheel self-contained, license layer real.

### 23.2. Неделя 2

| Задача | Sprint | Effort |
|---|---|---|
| Implementation `sndr memory explain` (§18.3) | 5 | 5-7 d |
| Documentation cleanup (§13 P2 #12-13) | 3 | 2 d |
| `_genesis` purge в active scripts (§13 P2 #14-15) | 3 | 1 d |

### 23.3. После 2 недель → переход к Sprint 4-9

Используя § 16 (vLLM PR backports), §17 (club-3090 решения), §15 (integration variants) как working backlog.

---

**Конец расширения**. Этот документ — living. Обновлять по мере перехода items planned → in flight → shipped.


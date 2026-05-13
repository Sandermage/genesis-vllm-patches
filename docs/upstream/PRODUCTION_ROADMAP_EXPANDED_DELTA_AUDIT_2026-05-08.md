# Genesis / SNDR - расширенный delta-аудит roadmap и кода

Дата: 2026-05-08  
База анализа: `docs/upstream/PRODUCTION_ROADMAP_2026-05-09.md` после пользовательского обновления до 2298 строк.  
Цель: не заменить roadmap, а дополнить его отдельным техническим файлом: что в нем верно, что расходится с кодом, какие новые пункты добавить, какие изменения делать первыми, и где нужны правки, улучшения, стабилизация, оптимизация и безопасность.

В рамках этого аудита исходный код не изменялся. Создан только этот Markdown-файл.

## 1. Executive Summary

Обновленный `PRODUCTION_ROADMAP_2026-05-09.md` стал гораздо полнее. В нем уже есть:

- базовый production-аудит;
- integration deep-dive по вариантам A-J;
- список upstream vLLM PR для backport/watchlist;
- решения по club-3090 issues;
- CLI skeleton planning;
- rollback plans;
- memory learnings;
- Definition of Done;
- план первой и второй недели.

Основная оценка: roadmap теперь правильно задает направление, но требует еще одного слоя конкретизации. В текущем коде есть несколько расхождений с roadmap, которые нужно явно добавить в backlog:

1. `sndr_engine` skeleton существует локально, но полностью игнорируется `.gitignore`. Если это должен быть public skeleton, его сейчас нельзя нормально опубликовать без force-add или изменения `.gitignore`.
2. Все 131 registry entries сейчас `community`; engine-tier patches отсутствуют. Это хорошо для текущей стратегии, но roadmap должен прямо зафиксировать: `sndr_engine` пока не функциональный слой, а будущий packaging/overlay контракт.
3. У 91 registry entry отсутствует `lifecycle`. Это не ломает schema validator, но засоряет installer output и мешает production governance.
4. Top-level `sndr` CLI все еще не совпадает с документацией и roadmap: фактически есть только `install` и `launch`, тогда как `compat.cli` уже знает 18 subcommands.
5. `pyproject.toml` говорит, что после install доступны `sndr doctor` и `sndr verify`, но canonical `sndr` их не регистрирует.
6. `pyproject-engine.toml` устарел относительно skeleton-only стратегии: комментарии все еще говорят, что engine wheel ships private kernels for PN72.
7. Launcher dry-run безопасен, но production container path все еще завязан на `/plugin` mount и editable install.
8. Документ §21 ссылается на `feedback_*.md` и `project_genesis_*.md` файлы, которых в текущем repo не видно. Если это internal memory, их нельзя оставлять как публичные ссылки без переноса/санитизации.
9. `docs/COOKBOOK.md` упоминается как DoD, но файла в проекте нет.
10. В дереве присутствуют `.DS_Store` и 41 `__pycache__` directory. Они игнорируются, но физически лежат в рабочем дереве и могут мешать упаковке/аудиту при ручных копиях.

Текущий production статус:

| Слой | Факт | Оценка |
|---|---|---|
| Python syntax | `PY_AST_CHECK count=515 errors=0` | чисто |
| Registry count | `131` entries | большой, но управляемый |
| Registry tier | `131 community`, `0 engine` | соответствует public-core стратегии |
| Lifecycle metadata | `91 missing`, `31 legacy`, `5 retired`, `3 experimental`, `1 coordinator` | требует нормализации |
| Shadow audit | clean | хорошо |
| Schema validator | clean | хорошо |
| Lifecycle audit | warnings only | хорошо, но неполное metadata |
| Installer dry-run | exit 2 | блокер |
| Full pytest | `2 failed, 2056 passed, 69 skipped` | блокер |
| Launcher dry-run | exit 0 | частично готов |
| Top-level CLI | только `install`, `launch` | неполно |
| Packaging | base deps empty, engine ignored | неполно |

## 2. Проверки, выполненные для этого delta-аудита

Команды:

```bash
python3 -c "AST parse vllm/sndr_core vllm/sndr_engine tests scripts"
python3 -m vllm.sndr_core.apply.shadow --strict
python3 -m vllm.sndr_core.compat.schema_validator --quiet
python3 -m vllm.sndr_core.compat.lifecycle_audit_cli --quiet
python3 -m vllm.sndr_core.cli --help
python3 -m vllm.sndr_core.compat.cli self-test --quiet
python3 -m vllm.sndr_core.compat.model_config_cli list
python3 -m vllm.sndr_core.cli install --dry-run --non-interactive
python3 -m vllm.sndr_core.cli launch --dry-run --non-interactive a5000-2x-35b-prod
pytest -q
```

Результаты:

```text
PY_AST_CHECK count=515 errors=0
shadow --strict: CLEAN
schema_validator: PATCH_REGISTRY schema clean
lifecycle_audit: warnings only
compat self-test: exit 0
model configs: 8 total, 6 working, 2 tested
install dry-run: exit 2
launch dry-run: exit 0
pytest: 2 failed, 2056 passed, 69 skipped
```

Registry summary:

```text
patch_count: 131
tiers: {'community': 131}
lifecycle: {'<missing>': 91, 'retired': 5, 'experimental': 3, 'legacy': 31, 'coordinator': 1}
implementation_status: {'<missing>': 128, 'retired': 1, 'marker_only': 1, 'placeholder': 1}
engine_ids: []
```

Вывод: архитектурно проект уже ушел в public-core модель. Поэтому все engine-related пункты должны быть сформулированы как будущий overlay/packaging контракт, а не как текущий функциональный слой.

## 3. Delta к обновленному roadmap

### 3.1. Что в roadmap уже верно

Roadmap корректно фиксирует:

- P4/P5 wiring bugs как P0.
- No-torch pytest failures как P0.
- Base wheel dependency policy как P0/P1.
- Top-level CLI unification как P0/P1.
- Stale docs around `_genesis`, `~/.genesis`, private IP/path.
- License trust anchor placeholder.
- `/plugin` mount как production smell.
- Need for `sndr doctor`, `sndr report bundle`, `sndr memory explain`.
- Public core now, private engine later.
- Need for backport/watchlist around TurboQuant, KV eviction, CPU offload, Gemma, FLA.
- Need for club-3090 issue absorption.

### 3.2. Что нужно уточнить

Roadmap говорит, что `sndr_engine` skeleton exists in public repo. Локально skeleton есть, но:

- `.gitignore:86` игнорирует `vllm/sndr_engine/`;
- `git ls-files vllm/sndr_engine` пустой;
- `git check-ignore -v vllm/sndr_engine/__init__.py` показывает `.gitignore:86`.

Значит есть два допустимых решения:

| Решение | Что сделать | Плюсы | Минусы |
|---|---|---|---|
| A. Public skeleton | убрать ignore или force-add `vllm/sndr_engine/{__init__.py,version.py,LICENSE-NOTICE}` | tests/docs честно видят skeleton | риск случайно добавить private code позже |
| B. No public skeleton | оставить ignore, убрать утверждения о public skeleton из docs/tests | сильнее защищает private boundary | core/tests должны уметь работать без `vllm.sndr_engine` вообще |

Рекомендация: выбрать B для public repo. Public core не должен поставлять `vllm.sndr_engine` вообще. Private engine позже должен быть отдельным wheel с entry point overlay. Если нужен reserved namespace, держать его только в private repo/template, а не в public tree.

### 3.3. Что нужно добавить

Добавить в roadmap отдельный Sprint 0.5: "Repository publication boundary".

Задачи:

1. Решить policy по `vllm/sndr_engine/`: force-add skeleton или полностью убрать из public assumptions.
2. Привести `.gitignore`, `pyproject.toml`, `pyproject-engine.toml`, README и tests к одному решению.
3. Если engine не публикуется, тесты должны проверять core без engine package, а не локальный skeleton.
4. Если skeleton публикуется, `.gitignore` должен защищать только `vllm/sndr_engine/private/`, `vllm/sndr_engine/patches/`, `vllm/sndr_engine/kernels/`, а не всю папку.

## 4. P0 блокеры, которые нужно исправить первыми

### P0-1. P4/P5 dry-run and real-apply variable typo

Файл:

- `vllm/sndr_core/apply/_per_patch_dispatch.py:263-344`

P4:

- импорт правильный: `vllm/sndr_core/apply/_per_patch_dispatch.py:289`
- использование неправильное: `vllm/sndr_core/apply/_per_patch_dispatch.py:290`
- real apply тоже неправильный: `vllm/sndr_core/apply/_per_patch_dispatch.py:301`

Сейчас:

```python
from vllm.sndr_core.patches.scheduler import p4_tq_hybrid
assert callable(patch_4_tq_hybrid.apply)
...
status, reason = patch_4_tq_hybrid.apply()
```

Должно быть:

```python
from vllm.sndr_core.patches.scheduler import p4_tq_hybrid
assert callable(p4_tq_hybrid.apply)
...
status, reason = p4_tq_hybrid.apply()
```

P5:

- импорт правильный: `vllm/sndr_core/apply/_per_patch_dispatch.py:328`
- использование неправильное: `vllm/sndr_core/apply/_per_patch_dispatch.py:329`
- real apply тоже неправильный: `vllm/sndr_core/apply/_per_patch_dispatch.py:339`

Сейчас:

```python
from vllm.sndr_core.patches.kv_cache import p5_page_size
assert callable(patch_5_page_size.apply)
...
status, reason = patch_5_page_size.apply()
```

Должно быть:

```python
from vllm.sndr_core.patches.kv_cache import p5_page_size
assert callable(p5_page_size.apply)
...
status, reason = p5_page_size.apply()
```

Дополнение к roadmap:

- добавить тест `tests/unit/apply/test_per_patch_dispatch_imports.py`;
- пройти все `@register_patch` wrappers в dry-run и проверить, что они не падают на `NameError`;
- отдельно проверить, что `failed` допускается только для настоящих missing target/vllm files, но не для local symbol errors.

### P0-2. No-torch test contract

Файлы:

- `tests/legacy/test_guards.py:146-152`
- `tests/unit/integrations/attention/gdn/test_p103_fla_cliff2.py:186-207`
- `vllm/sndr_core/integrations/attention/gdn/p103_fla_cliff2_chunked.py:142-159`

Проблема:

`test_get_torch_version_returns_tuple` предполагает наличие torch, но в no-torch среде `get_torch_version()` корректно возвращает `None`.

`test_p103_self_install_succeeds_with_mock_chunk_globals` ожидает `True`, но `_make_chunked_wrapper()` импортирует torch в момент построения wrapper:

```python
def _make_chunked_wrapper(...):
    ...
    import torch
```

Решение:

- В `test_guards.py` добавить marker/skip when torch absent.
- Для P103 выбрать контракт:
  - если install helper должен работать без torch, перенести `import torch` внутрь фактической execution ветки wrapper;
  - если P103 требует torch, тест должен быть `pytest.mark.requires_torch`.

Рекомендация: P103 лучше сделать import-safe без torch, потому что installer/patcher должен быть максимально cold-import friendly.

### P0-3. Installer output polluted by missing lifecycle

Факт:

- `PATCH_REGISTRY`: 91 entries without lifecycle.
- `install --dry-run` выводит десятки строк `lifecycle field unset`.

Проблема:

Даже после P4/P5 fix installer будет выглядеть шумно. Для production installer это плохо: пользователь не отличит важную ошибку от metadata warning.

Решение:

1. Добавить lifecycle для всех 91 entries.
2. Разделить audit modes:
   - installer mode: показывать только P0/P1 blockers;
   - registry audit mode: показывать весь lifecycle noise.
3. Ввести CI ratchet:
   - existing missing lifecycle allowed temporarily;
   - new entries without lifecycle fail.

Предлагаемая taxonomy:

| Lifecycle | Для чего |
|---|---|
| `stable` | validated and default-ready |
| `validated` | live-tested but default-off |
| `experimental` | implemented but still under A/B |
| `research` | not production, may be partial |
| `legacy` | pre-dispatcher/compat |
| `retired` | not engaged unless override |
| `merged_upstream` | patch should self-disable |
| `coordinator` | meta-patch/no direct wiring |

Если schema сейчас не знает `validated`, либо добавить его, либо использовать `stable`/`experimental` строго.

### P0-4. Public engine boundary ambiguity

Файлы:

- `.gitignore:78-87`
- `vllm/sndr_engine/__init__.py`
- `vllm/sndr_engine/LICENSE-NOTICE`
- `pyproject.toml:114-145`
- `pyproject-engine.toml:67-88`
- `vllm/sndr_core/license.py:131-138`

Факты:

- `pyproject.toml` excludes `vllm.sndr_engine*`.
- `.gitignore` ignores all `vllm/sndr_engine/`.
- `pyproject-engine.toml` claims engine wheel ships private kernels for PN72, but PN72 is now core/community.
- `license.py` still detects engine by importing `vllm.sndr_engine`.

Решение:

В roadmap нужно добавить четкое решение:

```text
Public repo:
  no vllm.sndr_engine package in wheel
  no dependency on local skeleton
  core tests must simulate no engine

Private repo:
  vllm-sndr-engine wheel
  entry point group: sndr.engine.overlay
  signed license required
```

Если оставить skeleton в public, нужно менять `.gitignore`:

```gitignore
vllm/sndr_engine/private/
vllm/sndr_engine/patches/
vllm/sndr_engine/kernels/
!vllm/sndr_engine/__init__.py
!vllm/sndr_engine/version.py
!vllm/sndr_engine/LICENSE-NOTICE
```

Но предпочтительнее не публиковать `vllm.sndr_engine` вообще до появления реального private wheel.

## 5. P1 архитектурные исправления

### P1-1. Top-level CLI должен стать настоящим control plane

Файлы:

- `vllm/sndr_core/cli/__init__.py:1-61`
- `vllm/sndr_core/compat/cli.py:39-68`
- `pyproject.toml:26-30`

Проблема:

`pyproject.toml` обещает:

```text
sndr doctor
sndr verify
```

Но canonical `sndr` показывает только:

```text
{install,launch,...}
install
launch
```

При этом `compat.cli` уже имеет:

```text
doctor, explain, init, list-models, pull, lifecycle-audit,
validate-schema, categories, migrate, recipe, preset, plugins,
telemetry, update-channel, self-test, verify, preflight, bench,
model-config
```

Решение:

- Не писать все заново.
- В top-level `sndr` добавить bridge subcommands, которые lazy-import соответствующие compat modules.
- Для user-facing нового API лучше сгруппировать команды:

```text
sndr doctor
sndr verify
sndr model-config ...
sndr patches ...
sndr plugins ...
sndr report ...
sndr memory ...
sndr models ...
sndr bench ...
```

Совместимость:

- `genesis` console script может оставаться alias to `compat.cli`.
- `sndr` должен стать canonical CLI.

### P1-2. Base wheel dependencies

Файл:

- `pyproject.toml:65-80`

Проблема:

`dependencies = []`, но core utilities используют YAML/HTTP behavior.

Решение:

Минимальный production set:

```toml
dependencies = [
  "pyyaml>=6.0",
  "packaging>=23.0",
]

[project.optional-dependencies]
http = ["requests>=2.28"]
dev = [...]
```

Если `requests` нужен только `boot_probe`, сделать lazy import с понятной ошибкой.

Дополнение:

- В roadmap добавить clean-venv wheel test:

```bash
python -m venv /tmp/sndr-clean
/tmp/sndr-clean/bin/pip install dist/vllm_sndr_core-*.whl
/tmp/sndr-clean/bin/sndr --help
/tmp/sndr-clean/bin/sndr model-config list
/tmp/sndr-clean/bin/sndr validate-schema
```

### P1-3. `/plugin` dependency убрать из production launch path

Файл:

- `vllm/sndr_core/model_configs/schema.py:708-760`

Проблема:

Docker bootstrap still:

```bash
cp -r /plugin /tmp/genesis_vllm_plugin
pip install --no-deps -e /tmp/genesis_vllm_plugin
python3 -m vllm.sndr_core.apply
```

Риск:

- production launch зависит от repo-local `tools/genesis_vllm_plugin`;
- wheel install path не является самодостаточным;
- `/plugin` mount является operational footgun.

Решение:

1. Production mode:

```bash
pip install vllm-sndr-core==<pin>
python3 -m vllm.sndr_core.apply
```

2. Dev mode:

```bash
-v ${plugin_src}:/plugin:ro
pip install -e /plugin
```

3. Model config schema:

```yaml
runtime:
  install_mode: wheel | editable | bind_mount
  core_wheel: /path/to/vllm_sndr_core.whl
  plugin_src: optional
```

### P1-4. Bundle tier mismatch

Файлы:

- `vllm/sndr_core/bundles/_common.py:64-77`
- `vllm/sndr_core/bundles/attention_tq_multi_query.py:1-41`
- `vllm/sndr_core/dispatcher/registry.py` for P67/P67b

Проблема:

Registry says P67/P67b are community. Bundle says engine.

Решение:

- Change `attention_tq_multi_query` tier to `community`.
- Keep future engine gate generic, but not used by current bundles.
- Add test: all bundle tier values match registry tier of included patches.

### P1-5. License gate must not rely on importable skeleton

Файл:

- `vllm/sndr_core/license.py:131-138`

Проблема:

License gate detects engine via:

```python
import vllm.sndr_engine as _engine
```

This is weak if a skeleton package exists.

Решение:

- Detect private engine via entry point or explicit `engine_available()` exported by private package.
- If skeleton exists, `engine_available()` must return False and license gate must respect that.
- Add test: importable skeleton + valid legacy key still does not unlock engine features.

### P1-6. `pyproject-engine.toml` is stale

Файл:

- `pyproject-engine.toml:67-88`

Проблема:

Comment says:

```text
engine wheel now ships private kernels (ngram_frequency_filter for PN72)
```

Но PN72 now core/community.

Решение:

- Mark `pyproject-engine.toml` as private-repo template only.
- Remove PN72/kernels claims.
- If kept in public tree, make it a non-buildable template with explicit warning.

### P1-7. Updated roadmap has public broken links to internal memory docs

Файл:

- `docs/upstream/PRODUCTION_ROADMAP_2026-05-09.md:2155-2196`

Проблема:

Roadmap links to files like:

- `feedback_synthetic_mode_breaks_tools_api.md`
- `feedback_p67_genesis_kernel_quality_mirage.md`
- `feedback_pn59_validated_prod.md`
- `project_genesis_v7_13_strict_ngram_breakthrough.md`

`rg --files` does not find these files in repo.

Решение:

- Если это internal memory: заменить ссылки на plain text references or move sanitized summaries to `docs/reference/`.
- Если это public docs: создать `docs/reference/learnings/*.md` with sanitized content.
- В public README/roadmap не оставлять relative links to absent local memory files.

## 6. P2 качество, документация, чистота репозитория

### P2-1. `.DS_Store` and `__pycache__` cleanup

Факты:

- `.DS_Store` present:
  - `vllm/sndr_core/compat/.DS_Store`
  - `vllm/sndr_core/model_configs/.DS_Store`
- `__pycache__` dirs under `vllm/sndr_core` and `vllm/sndr_engine`: 41.

Они игнорируются `.gitignore`, но лучше добавить housekeeping task:

```bash
find vllm/sndr_core vllm/sndr_engine -name .DS_Store -delete
find vllm/sndr_core vllm/sndr_engine -type d -name __pycache__ -prune -exec rm -rf {} +
```

В рамках этого аудита команды удаления не выполнялись.

### P2-2. Active docs still stale around engine

Файлы:

- `README.md:28-31`
- `README.md:91`
- `docs/INSTALL.md`
- `docs/PATCHES.md`
- `vllm/sndr_core/__init__.py:12-26`

Дополнение:

`vllm/sndr_core/__init__.py` особенно важно исправить, потому что это не просто docs, а package docstring. Сейчас он говорит:

```text
All code still lives in vllm/_genesis/
```

Фактически код уже в `vllm/sndr_core`.

### P2-3. Active scripts still contain old `_genesis` assumptions

Критичные места:

- `scripts/launch/preflight_check.sh` проверяет `vllm/_genesis`.
- `vllm/sndr_core/utils/boot_probe.py` examples use `vllm._genesis`.
- Several active docs and tests mention old path.

Решение:

- Create allowlist file: `tools/allowed_legacy_refs.txt`.
- Add CI grep gate:

```bash
rg "vllm\\._genesis|vllm/_genesis" README.md docs scripts vllm/sndr_core tests
```

but ignore allowed marker/test-history references.

### P2-4. `docs/COOKBOOK.md` is planned but absent

Roadmap DoD mentions:

- `docs/COOKBOOK.md` recipes for closed club-3090 issues.

Факт:

- `find docs -maxdepth 2 -name COOKBOOK.md` returns nothing.

Решение:

Create `docs/COOKBOOK.md` with sections:

- OOM long context.
- Qwen3Coder tool parser silence.
- TurboQuant/MTP safe launch.
- DFlash/PFlash configs.
- 3090 24GB memory recipes.
- A5000 dual GPU recipes.
- Docker/container R/W layer reset.
- Prefix cache + spec-decode warning.

## 7. Оптимизация и стабилизация реализации

### 7.1. Apply loop optimization

Проблема:

Installer dry-run triggers registry warnings for many patches. Apply-loop should be less noisy.

Изменения:

- Add structured severity to registry validator:
  - `error`: fail.
  - `warn`: show in audit.
  - `info`: hide from installer unless `--verbose`.
- Installer smoke should summarize:

```text
dispatcher dry-run:
  applied: 104
  skipped expected: 17
  failed real: 2
  metadata warnings: 91 hidden, run sndr patches audit
```

### 7.2. Model-config optimization

Улучшения:

- Add `sndr model-config score <key>`:
  - memory safety score;
  - tool-call stability score;
  - spec-decode risk score;
  - upstream drift risk;
  - deployment readiness.
- Add `sndr model-config diff <key1> <key2>`.
- Add `sndr model-config explain-flags <key>`.
- Add `constraints` section to YAML:

```yaml
constraints:
  min_gpu_memory_gib: 24
  min_gpu_count: 2
  pcie_ok: true
  nvlink_recommended: false
  forbidden_flags:
    - --enable-prefix-caching
```

### 7.3. Memory optimization

Add `sndr memory explain` as roadmap says, but implement in phases:

Phase 1:

- static estimate from model config:
  - weights;
  - KV cache;
  - cudagraph reserve;
  - Triton/workspace;
  - overhead.

Phase 2:

- live compare against `nvidia-smi`.

Phase 3:

- recommendations:
  - lower `max_num_batched_tokens`;
  - lower `max_num_seqs`;
  - choose fp8/turboquant KV;
  - disable risky spec decode;
  - choose smaller context preset.

Phase 4:

- per-patch memory attribution:
  - PN25;
  - PN59;
  - P98;
  - P99;
  - P101;
  - PN78/PN79 family.

### 7.4. Stability and quality gates

Add mandatory gates before marking a patch stable:

- unit test;
- dry-run import test;
- text-patch anchor test;
- idempotency test;
- no-torch import test if module is imported by CLI/registry;
- live reproducer if patch claims production bug fix;
- rollback instruction;
- upstream drift marker.

## 8. Security and release hardening

### 8.1. License

Current:

- `license.py:58-60` uses zero placeholder trust anchor.
- `tests/conftest.py:27-34` globally sets legacy unsigned license mode.

Required:

1. Real Ed25519 public key before any commercial release.
2. Private key generated offline and never committed.
3. Test strict mode without `SNDR_ALLOW_LEGACY_LICENSE_KEYS`.
4. Legacy key tests only inside scoped fixtures.
5. `sndr license status --json`.

### 8.2. Plugin security

Current:

- Plugin discovery exists.
- Opt-in env exists.

Missing:

- plugin provenance;
- plugin signature;
- plugin lockfile;
- plugin permission model;
- production mode refusing unsigned plugins.

Recommended stages:

1. Dev plugins: opt-in via `GENESIS_ALLOW_PLUGINS=1`.
2. Trusted plugins: signed manifest.
3. Production: unsigned plugins refused unless `SNDR_ALLOW_UNSIGNED_PLUGINS=1`.

### 8.3. Supply chain

Add:

- constraints file for release builds;
- SBOM;
- `pip-audit` job;
- reproducible wheel build;
- signed git tag;
- release checklist.

## 9. Integration roadmap additions

### 9.1. Engine overlay

Roadmap §15.1 is good, but add one correction:

Do not use `vllm.sndr_engine` import as availability signal. Use entry points:

```text
group = "sndr.engine.overlay"
```

Core should expose a minimal stable API:

```text
register_patch
register_bundle
register_kernel
register_model_config
require_license
```

### 9.2. Bundles

Add `BundleSpec` dataclass:

```text
id
title
tier
patches
requires
conflicts
risk
default_for_profiles
explain
```

Add tests:

- bundle tier matches all included patches;
- conflicts enforced;
- dry-run explain works without vLLM;
- bundle cannot silently apply partial patch set without transaction summary.

### 9.3. Remote server

Roadmap mentions `sndr --remote <user>@<host>`. For public docs, avoid hardcoded address. Use named host profile:

```yaml
hosts:
  genesis-a2:
    ssh: ${SNDR_REMOTE_SSH}
```

CLI:

```text
sndr remote add genesis-a2 --ssh <user>@host
sndr remote doctor genesis-a2
sndr remote report genesis-a2
sndr remote launch genesis-a2 a5000-2x-35b-prod
```

### 9.4. club-3090 integration

Add `sndr community import-issue` later:

```text
sndr community import-issue noonghunna/club-3090#72
```

It should create:

- local reproducer fixture;
- cookbook entry;
- patch tracking note;
- test TODO;
- model/hardware profile tag.

### 9.5. Upstream vLLM tracking

Roadmap lists PRs, but code should encode them.

Add `upstream_watch.yaml`:

```yaml
watch:
  - upstream: vllm#38479
    local_patches: [P4, P5, P67, P67b, P78, P98, P99, P101, PN57]
    action: drift-check
  - upstream: vllm#40914
    local_patches: [P67, P67b]
    action: a-b-test
```

Then `tools/check_upstream_drift.py` reads structured watchlist instead of hardcoded notes.

## 10. Suggested new sprint split

Existing roadmap has Sprint 0-9. Add finer split:

### Sprint 0A - Red checks

- P4/P5 typo fix.
- No-torch test fixes.
- Full pytest green.
- Installer dry-run green.

### Sprint 0B - Publication boundary

- Decide public skeleton vs no public engine.
- Align `.gitignore`, pyproject, tests, docs.
- Fix `pyproject-engine.toml` stale PN72 claims.

### Sprint 0C - Metadata hygiene

- Add lifecycle to 91 entries.
- Hide non-critical metadata warnings from installer.
- Add ratchet tests.

### Sprint 1 - Self-contained wheel

- Base deps.
- Remove `/plugin` from production launch.
- Clean venv wheel test.
- Package data verification.

### Sprint 2 - Canonical CLI

- Register existing compat commands under `sndr`.
- Add `sndr patches`.
- Add `sndr report bundle` skeleton.
- Add `sndr memory explain` skeleton.

### Sprint 3 - Docs and repo hygiene

- README engine correction.
- `sndr_core/__init__.py` docstring correction.
- `docs/COOKBOOK.md`.
- Remove public broken feedback links or create sanitized docs.
- Clean `.DS_Store` and pycache from working tree.

### Sprint 4 - Memory and launch quality

- Memory estimator.
- Host profiles.
- Remote doctor/report.
- Model-config scoring.

### Sprint 5 - Qwen/Gemma/TurboQuant/DFlash stabilization

- Qwen3Coder SSE silence regression.
- Gemma profile.
- DFlash docs and benchmark matrix.
- TurboQuant upstream drift A/B.

## 11. Concrete backlog additions

Add these to roadmap as explicit tasks:

| ID | Priority | File/Area | Task |
|---|---|---|---|
| DA-001 | P0 | `_per_patch_dispatch.py` | Fix P4/P5 symbol typos in dry-run and apply path |
| DA-002 | P0 | tests | Add dry-run import test for every `@register_patch` wrapper |
| DA-003 | P0 | tests | Make no-torch pytest fully green |
| DA-004 | P0 | repo boundary | Decide and implement engine publication policy |
| DA-005 | P0 | registry | Add lifecycle to 91 missing entries or add CI ratchet |
| DA-006 | P1 | `cli/__init__.py` | Bridge compat commands into canonical `sndr` |
| DA-007 | P1 | `pyproject.toml` | Add/clarify runtime deps and clean-venv test |
| DA-008 | P1 | launcher | Remove `/plugin` dependency from production launch |
| DA-009 | P1 | bundles | Fix P67/P67b bundle tier mismatch |
| DA-010 | P1 | license | Replace import-based engine gate with overlay/availability gate |
| DA-011 | P1 | `pyproject-engine.toml` | Remove stale PN72/private-kernel comments |
| DA-012 | P2 | docs | Fix README/docs engine/PN72 story |
| DA-013 | P2 | docs | Create `docs/COOKBOOK.md` or remove DoD reference |
| DA-014 | P2 | docs/reference | Replace missing `feedback_*.md` links with real sanitized docs |
| DA-015 | P2 | repo hygiene | Remove local `.DS_Store` and `__pycache__` directories |
| DA-016 | P2 | scripts | Update active launch/preflight scripts from `_genesis` to `sndr_core` |
| DA-017 | P2 | model configs | Add `constraints` and `risk_score` fields |
| DA-018 | P2 | memory | Implement phase-1 `sndr memory explain` |
| DA-019 | P3 | upstream | Move upstream watchlist into structured YAML |
| DA-020 | P3 | community | Create club-3090 issue import/cookbook workflow |

## 12. Validation matrix after fixes

Minimum local:

```bash
python3 -m vllm.sndr_core.apply.shadow --strict
python3 -m vllm.sndr_core.compat.schema_validator --quiet
python3 -m vllm.sndr_core.compat.lifecycle_audit_cli --quiet
python3 -m vllm.sndr_core.cli --help
python3 -m vllm.sndr_core.cli install --dry-run --non-interactive
python3 -m vllm.sndr_core.cli launch --dry-run --non-interactive a5000-2x-35b-prod
pytest -q
```

Clean wheel:

```bash
python -m build -w
python -m venv /tmp/sndr-wheel-test
/tmp/sndr-wheel-test/bin/pip install dist/vllm_sndr_core-*.whl
/tmp/sndr-wheel-test/bin/sndr --help
/tmp/sndr-wheel-test/bin/sndr model-config list
```

No-engine:

```bash
python -c "import importlib.util; assert importlib.util.find_spec('vllm.sndr_engine') is None"
python -m vllm.sndr_core.compat.schema_validator --quiet
```

If choosing public skeleton instead:

```bash
python -c "import vllm.sndr_engine as e; assert e.engine_available() is False"
python -c "from vllm.sndr_core.license import check_engine_tier_eligible; print(check_engine_tier_eligible())"
```

GPU/live:

```bash
sndr remote doctor <host>
sndr report bundle --remote <host>
sndr launch --dry-run --remote <host> a5000-2x-35b-prod
```

## 13. Final technical recommendation

Держать стратегию:

```text
Everything public today -> sndr_core
Private engine later -> separate wheel/repo via overlay entry point
Core never depends on engine
Operator UX -> sndr CLI + model configs + doctor/report/memory
```

Но перед любыми новыми backports и интеграциями нужно закрыть четыре слоя:

1. Красные проверки: P4/P5, no-torch pytest.
2. Publication boundary: engine skeleton vs no engine in public.
3. Self-contained package: deps, `/plugin`, clean wheel.
4. CLI truth: `sndr` help must match docs and roadmap.

Только после этого имеет смысл идти в memory/cache integrations, club-3090 import, upstream PR mining, Gemma/Qwen expansion и private engine overlay.


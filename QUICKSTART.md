# Genesis vLLM Patches — Quickstart

Step-by-step: from cloning the repo to a healthy `vllm serve` with all 28 Genesis patches applied. About 15 minutes if your model is already downloaded.

> 🇬🇧 English first · 🇷🇺 Русский ниже

---

## 🇬🇧 What you need before starting

| Requirement | Notes |
|---|---|
| Linux host | Tested on Ubuntu 22.04 / 24.04 with kernel 6.x |
| Docker + Docker Compose v2 | `docker compose version` should report v2.x |
| NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` must work |
| NVIDIA driver 570+ | `nvidia-smi` reports it; older drivers may work but untested |
| 2× GPU with ≥ 24 GiB each | Validated on 2× RTX A5000. Single-GPU works for smaller models with `--tensor-parallel-size 1` |
| ~80 GiB free disk | Model + HuggingFace cache |
| Internet (first run) | To pull the Docker image and (optionally) the model |

If you don't have the model locally yet, the container will pull it from HuggingFace on first run.

## Step 1 — Clone the repository

```bash
cd ~
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches
git checkout v7.10.0   # pin to the validated release
```

## Step 2 — Pull the exact pinned vLLM image

This is the **same image** we used for all v7.10 validation runs. Pinned by SHA, immutable.

```bash
docker pull vllm/vllm-openai:nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52
```

About 36 GiB. Once downloaded, it's cached locally — re-runs are instant.

For convenience, tag it with a friendly name:

```bash
docker tag \
  vllm/vllm-openai:nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52 \
  vllm/vllm-openai:genesis-v7.10-baseline
```

## Step 3 — Pick your compose file

The repo ships several compose files for different scenarios:

| File | Model | When to use |
|---|---|---|
| `docker-compose.example.yml` | template only | Read it, copy, adapt |
| `docker-compose.integration.yml` | Qwen3-Next-35B-A3B-FP8 + TQ k8v4 | Production-mirror — what we test against |
| `docker-compose.integration-awq.yml` | Qwen3-Next-35B-A3B-AWQ + TQ k8v4 | AWQ 4-bit weights, 2.5× more KV memory |
| `docker-compose.integration-fp16kv.yml` | Qwen3-Next FP8 weights + fp16 KV | If you want non-TurboQuant baseline |
| `docker-compose.qwen3-5-dense.yml` | RYS-Qwen3.5-27B-FP8-XL dense | Dense model, no MoE/hybrid |
| `docker-compose.gemma4-26b-moe.yml` | Gemma 4 26B MoE AWQ | ⚠️ currently blocked by vLLM × model incompatibility |

**For first-time users, start with `docker-compose.integration.yml`** — that's the canonical config.

## Step 4 — Adapt paths to your machine

Open the compose file and update **two** sections:

### 4a. Model path

If your model files are in a different location than `/nfs/genesis/models/`, edit:

```yaml
volumes:
  - /nfs/genesis/models:/models:ro    # ← change /nfs/genesis/models to your path
```

The model directory should contain the `Qwen3.6-35B-A3B-FP8/` (or whichever model) subfolder with `config.json`, `tokenizer.json`, safetensor shards, etc.

Alternative: pull from HuggingFace directly. Replace `--model /models/Qwen3.6-35B-A3B-FP8` with `--model Qwen/Qwen3-Next-35B-A3B-FP8` (the HF repo id) and let the container download on first start.

### 4b. Image tag

If you tagged the image (Step 2), replace:

```yaml
image: vllm/vllm-openai:genesis-v7.0-baseline
```

with whatever name you used (e.g. `vllm/vllm-openai:genesis-v7.10-baseline`). Or just keep the `nightly-fe9c3d6c5...` long form — both work.

## Step 5 — Start the container

```bash
docker compose -f docker-compose.integration.yml up -d
```

Watch the logs:

```bash
docker logs -f vllm-integration-v7
```

Boot takes about **3–5 minutes** the first time (vLLM downloads/installs deps, applies all Genesis patches, loads model weights, captures CUDA graphs).

You'll see this sequence:

```
=== Install prod-equivalent runtime deps + Genesis plugin ===
=== Apply Genesis wiring (text-patches + rebinds, BEFORE vllm serve) ===
[INFO genesis.apply_all] Genesis Results: 28 applied, 4 skipped, 0 failed
=== Start vLLM server ===
(APIServer pid=1) INFO ... Application startup complete
(APIServer pid=1) INFO ... Uvicorn running on http://0.0.0.0:8000
```

When you see `Uvicorn running on http://0.0.0.0:8000` — server is ready.

## Step 6 — Verify it works

### 6a. Health check

```bash
curl http://localhost:8000/health
# → 200 OK
```

### 6b. Smoke chat

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer genesis-local" \
  -d '{
    "model": "qwen3.6-35b-a3b-integration",
    "messages": [{"role":"user","content":"Say hello in one word."}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

You should get a normal JSON response with `choices[0].message.content`.

### 6c. Verify Genesis applied

```bash
docker logs vllm-integration-v7 2>&1 | grep "Genesis Results:"
# → [INFO:genesis.apply_all] Genesis Results: 28 applied, 4 skipped, 0 failed
```

If you see **`0 failed`** — you're good. The 4 skipped ones are opt-in patches that need explicit env flags (P5b / P7b / P40 / P41), see [README#opt-in-patches](README.md#4-opt-in-patches).

### 6d. Verify dispatch profile (v7.9 model_detect)

```bash
docker exec vllm-integration-v7 python3 -c "
from vllm._genesis.model_detect import get_model_profile
import json
print(json.dumps(get_model_profile(), indent=2, default=str))
"
```

Expected for Qwen3-Next: `"moe": true, "hybrid": true, "turboquant": true`.

## Step 7 — Stopping cleanly

**Always use `docker compose down`, NEVER plain `docker stop`.**

```bash
docker compose -f docker-compose.integration.yml down
```

This removes the container so the next `up -d` starts with a clean filesystem. If you only `docker stop` then `docker start`, the patches will fail on the second boot due to anchor-already-applied (the "R/W layer trap" — see Troubleshooting below).

## Troubleshooting

### "Genesis Results: N applied, M skipped, 1 failed" on second boot

**Cause**: you used `docker stop` + `docker start` instead of `docker compose down` + `up -d`. Genesis text-patches are applied to files inside the container's writable layer; restarting the same container shows already-patched files, so anchors don't match.

**Fix**:

```bash
docker compose -f <your-compose>.yml down
docker compose -f <your-compose>.yml up -d
```

### Container restarts in a loop

Same root cause as above. Check `docker logs <container>` for `[FAILED]` patches. The fix is `down + up -d`.

### Model fails to load with `KeyError: 'layers.0.moe.experts.0.down_proj_packed'`

Some AWQ-quantized MoE models (notably `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) use a per-expert tensor naming scheme that the current vLLM dev134 loader doesn't recognise. This is a vLLM × model compatibility issue, **not a Genesis bug**. Workaround: use a different quantization of the same model, or wait for upstream vLLM to support it.

### `docker run --rm --gpus all` fails

NVIDIA Container Toolkit isn't installed or not configured. See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### "Cannot connect to the Docker daemon"

`docker compose` needs your user to be in the `docker` group:

```bash
sudo usermod -aG docker $USER
# log out and back in
```

### Out-of-memory at long context

Genesis patches reduce memory footprint significantly, but you can still hit OOM if `max_model_len` is too aggressive for your VRAM. For 2× A5000 (24 GiB each):
- `Qwen3-Next-35B-A3B-FP8` at `max_model_len=262144` works (proven to 258k actual tokens)
- For smaller cards, drop to `max_model_len=131072` or `65536`

Edit the value in your compose file's `command:` section.

## Optional: spec-decode mode (v7.11, opt-in workaround for #40831)

If you want to try **speculative decoding with TurboQuant**, there is a known upstream interaction bug ([#40831](https://github.com/vllm-project/vllm/issues/40831)) that causes degenerate token loops. We ship `P56` as a partial workaround.

- Test compose: `docker-compose.spec-decode-test.yml` (ngram n=3, port 8000)
- Enable P56 via env: `GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=1`
- After boot, look for `[INFO genesis.apply_all] [Genesis] applied: P56 ... — spec-decode fast-path guard wired`

**What P56 closes**: catastrophic XML/JSON loops, restored `tool_calls[]` population.
**What P56 does NOT close**: token-level duplication (`for for`, `age age`, `parameter parameter`) — that's deeper architectural and needs upstream fix. Full analysis: [#40831 comment](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4317214311).

For investigating spec-decode regressions yourself, two diagnostic scripts:

```bash
# Run a 9-prompt probe against the live backend, write JSONL
python3 scripts/sequential_backend_probe.py run \
    --host http://localhost:8000 --api-key genesis-local \
    --model qwen3.6-35b-a3b --label baseline --out /tmp/baseline.jsonl

# Switch backend, run same probes against the second one
python3 scripts/sequential_backend_probe.py run \
    --host http://localhost:8000 --api-key genesis-local \
    --model qwen3.6-35b-a3b-specdec --label specdec --out /tmp/specdec.jsonl

# Diff side-by-side — surfaces token duplication, missing tool_calls, etc.
python3 scripts/sequential_backend_probe.py diff /tmp/baseline.jsonl /tmp/specdec.jsonl
```

If you have GPU headroom for two concurrent backends on different ports, see also `scripts/dual_backend_diagnostic_proxy.py` for fan-out + diff in one call.

---

## Where to go next

| Topic | Where |
|---|---|
| Patch list and what each does | [README.md#patch-roster-v710](README.md#patch-roster-v710) |
| Validation methodology and raw bench data | [benchmarks/v7_10_validation_20260424/](benchmarks/v7_10_validation_20260424/) |
| Upstream PR tracking and backport plan | [README.md#upstream-status-tracking](README.md#upstream-status-tracking) |
| In-depth analysis of upstream PRs | [benchmarks/v7_10_validation_20260424/upstream_compare/PR_DEEP_DIVE.md](benchmarks/v7_10_validation_20260424/upstream_compare/PR_DEEP_DIVE.md) |
| Architecture overview | [README.md#architecture](README.md#architecture) |
| Running unit tests (CPU only) | `./validate_unit.sh` |
| Running integration tests (GPU required) | `./validate_integration.sh` |
| How to support / sponsor | [SPONSORS.md](SPONSORS.md) |

---

## 🇷🇺 Что нужно перед стартом

| Требование | Заметки |
|---|---|
| Linux хост | Тестировано на Ubuntu 22.04 / 24.04, ядро 6.x |
| Docker + Docker Compose v2 | `docker compose version` должна показать v2.x |
| NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` должен работать |
| NVIDIA драйвер 570+ | Старые версии могут работать, но не тестировались |
| 2× GPU ≥ 24 GiB | Валидировано на 2× RTX A5000. Один GPU подойдёт для меньших моделей с `--tensor-parallel-size 1` |
| ~80 GiB свободного диска | Модель + HuggingFace кеш |
| Интернет (первый запуск) | Для загрузки Docker image и (опционально) модели |

Если модели локально нет — контейнер скачает её с HuggingFace при первом запуске.

## Шаг 1 — Клонировать репозиторий

```bash
cd ~
git clone https://github.com/Sandermage/genesis-vllm-patches.git
cd genesis-vllm-patches
git checkout v7.10.0   # фиксируемся на validated релизе
```

## Шаг 2 — Скачать pinned vLLM образ

Тот же образ что использовался во всех v7.10 валидационных прогонах. Зафиксирован по SHA, не меняется.

```bash
docker pull vllm/vllm-openai:nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52
```

Около 36 GiB. После скачивания кешируется локально — повторные запуски мгновенные.

Для удобства можно перетегнуть:

```bash
docker tag \
  vllm/vllm-openai:nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52 \
  vllm/vllm-openai:genesis-v7.10-baseline
```

## Шаг 3 — Выбрать compose файл

Репо содержит несколько compose-файлов под разные сценарии:

| Файл | Модель | Когда использовать |
|---|---|---|
| `docker-compose.example.yml` | только шаблон | Прочитать, скопировать, адаптировать |
| `docker-compose.integration.yml` | Qwen3-Next-35B-A3B-FP8 + TQ k8v4 | Production-mirror — на чём мы тестируем |
| `docker-compose.integration-awq.yml` | Qwen3-Next-35B-A3B-AWQ + TQ k8v4 | AWQ 4-bit веса, 2.5× больше KV памяти |
| `docker-compose.integration-fp16kv.yml` | Qwen3-Next FP8 веса + fp16 KV | Если нужен non-TurboQuant baseline |
| `docker-compose.qwen3-5-dense.yml` | RYS-Qwen3.5-27B-FP8-XL dense | Dense модель, без MoE/hybrid |
| `docker-compose.gemma4-26b-moe.yml` | Gemma 4 26B MoE AWQ | ⚠️ заблокирован vLLM × model несовместимостью |

**Для первого запуска возьми `docker-compose.integration.yml`** — это канонический конфиг.

## Шаг 4 — Адаптировать пути под свою машину

Открой compose файл и поправь **две** секции:

### 4a. Путь к модели

Если файлы модели лежат не в `/nfs/genesis/models/`, отредактируй:

```yaml
volumes:
  - /nfs/genesis/models:/models:ro    # ← замени на свой путь
```

Директория модели должна содержать подпапку `Qwen3.6-35B-A3B-FP8/` (или другую) с `config.json`, `tokenizer.json`, safetensor shards и т.д.

Альтернатива: тянуть с HuggingFace напрямую. Замени `--model /models/Qwen3.6-35B-A3B-FP8` на `--model Qwen/Qwen3-Next-35B-A3B-FP8` (HF repo id), контейнер сам скачает при первом старте.

### 4b. Тег образа

Если перетегнул образ (Шаг 2), замени:

```yaml
image: vllm/vllm-openai:genesis-v7.0-baseline
```

на твоё имя (например `vllm/vllm-openai:genesis-v7.10-baseline`). Или оставь длинную форму `nightly-fe9c3d6c5...` — обе работают.

## Шаг 5 — Запустить контейнер

```bash
docker compose -f docker-compose.integration.yml up -d
```

Смотри логи:

```bash
docker logs -f vllm-integration-v7
```

Boot занимает **3–5 минут** на первый раз (vLLM скачивает/ставит зависимости, применяет все Genesis патчи, грузит веса модели, компилирует CUDA графы).

Увидишь такую последовательность:

```
=== Install prod-equivalent runtime deps + Genesis plugin ===
=== Apply Genesis wiring (text-patches + rebinds, BEFORE vllm serve) ===
[INFO genesis.apply_all] Genesis Results: 28 applied, 4 skipped, 0 failed
=== Start vLLM server ===
(APIServer pid=1) INFO ... Application startup complete
(APIServer pid=1) INFO ... Uvicorn running on http://0.0.0.0:8000
```

Как только видишь `Uvicorn running on http://0.0.0.0:8000` — сервер готов.

## Шаг 6 — Проверка

### 6a. Health check

```bash
curl http://localhost:8000/health
# → 200 OK
```

### 6b. Тестовый чат

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer genesis-local" \
  -d '{
    "model": "qwen3.6-35b-a3b-integration",
    "messages": [{"role":"user","content":"Привет в одно слово."}],
    "max_tokens": 16,
    "temperature": 0
  }'
```

Должен прийти нормальный JSON ответ с `choices[0].message.content`.

### 6c. Проверка что Genesis применился

```bash
docker logs vllm-integration-v7 2>&1 | grep "Genesis Results:"
# → [INFO:genesis.apply_all] Genesis Results: 28 applied, 4 skipped, 0 failed
```

Если **`0 failed`** — всё хорошо. 4 skipped — это opt-in патчи которые нужно явно включать env-флагами (P5b / P7b / P40 / P41), см. [README#opt-in-patches](README.md#4-opt-in-patches).

### 6d. Проверка dispatch profile (v7.9 model_detect)

```bash
docker exec vllm-integration-v7 python3 -c "
from vllm._genesis.model_detect import get_model_profile
import json
print(json.dumps(get_model_profile(), indent=2, default=str))
"
```

Для Qwen3-Next ожидаемо: `"moe": true, "hybrid": true, "turboquant": true`.

## Шаг 7 — Корректная остановка

**Всегда используй `docker compose down`, НИКОГДА не делай простой `docker stop`.**

```bash
docker compose -f docker-compose.integration.yml down
```

Это удаляет контейнер чтобы следующий `up -d` стартанул со свежей файловой системой. Если сделать только `docker stop` потом `docker start`, патчи провалятся на втором boot из-за того что anchors уже применены ("R/W layer trap" — см. Troubleshooting).

## Решение проблем

### "Genesis Results: N applied, M skipped, 1 failed" на втором запуске

**Причина**: использовал `docker stop` + `docker start` вместо `docker compose down` + `up -d`. Genesis text-патчи применяются к файлам внутри writable layer контейнера; при перезапуске того же контейнера файлы уже патченые, и anchors не находятся.

**Решение**:

```bash
docker compose -f <твой-compose>.yml down
docker compose -f <твой-compose>.yml up -d
```

### Контейнер в цикле перезапуска

Та же причина что выше. Проверь `docker logs <container>` на `[FAILED]` патчи. Решение — `down + up -d`.

### Модель не загружается с `KeyError: 'layers.0.moe.experts.0.down_proj_packed'`

Некоторые AWQ-квантизованные MoE модели (например `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) используют per-expert tensor naming которое текущий vLLM dev134 loader не понимает. Это **vLLM × model совместимость, НЕ баг Genesis**. Workaround: возьми другую квантизацию той же модели, или подожди когда upstream vLLM это поддержит.

### `docker run --rm --gpus all` не работает

NVIDIA Container Toolkit не установлен или не настроен. См. https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### "Cannot connect to the Docker daemon"

`docker compose` требует чтобы пользователь был в группе `docker`:

```bash
sudo usermod -aG docker $USER
# выйди и войди заново в систему
```

### OOM на длинном контексте

Genesis патчи существенно уменьшают расход памяти, но всё равно можно поймать OOM если `max_model_len` слишком агрессивный для VRAM. Для 2× A5000 (24 GiB каждая):
- `Qwen3-Next-35B-A3B-FP8` на `max_model_len=262144` работает (доказано до 258k фактических токенов)
- Для меньших карт ставь `max_model_len=131072` или `65536`

Меняй значение в `command:` секции своего compose-файла.

## Куда дальше

| Тема | Где |
|---|---|
| Список патчей и что каждый делает | [README.md#patch-roster-v710](README.md#patch-roster-v710) |
| Методология валидации и сырые бенчи | [benchmarks/v7_10_validation_20260424/](benchmarks/v7_10_validation_20260424/) |
| Tracking upstream PR-ов и план backport | [README.md#upstream-status-tracking](README.md#upstream-status-tracking) |
| Глубокий разбор upstream PR | [benchmarks/v7_10_validation_20260424/upstream_compare/PR_DEEP_DIVE.md](benchmarks/v7_10_validation_20260424/upstream_compare/PR_DEEP_DIVE.md) |
| Обзор архитектуры | [README.md#architecture](README.md#architecture) |
| Запуск unit тестов (только CPU) | `./validate_unit.sh` |
| Запуск integration тестов (нужен GPU) | `./validate_integration.sh` |
| Как поддержать проект | [SPONSORS.md](SPONSORS.md) |

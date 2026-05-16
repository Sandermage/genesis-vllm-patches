# Genesis / SNDR Cookbook

Recipe-style guide for production scenarios that have surfaced
through community feedback (noonghunna/club-3090) and Genesis's own
operational history. Each recipe has:

- **Симптом** — как операторы описывают проблему
- **Корень** — техническая причина
- **Workaround** — что сделать чтобы запустить сейчас
- **Fix** — какой Genesis patch / config помогает (если есть)
- **Prevention** — как избежать в будущем

DA-013 (audit 2026-05-08): создан как живой документ; пополнять по
мере закрытия community issues.

---

## 1. OOM на длинном контексте — single 24 GB карта

**Симптом**:

```text
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate
50.00 MiB. GPU 0 has a total capacity of 23.99 GiB of which 12.34 MiB is free.
```

Возникает после 30-60 минут sustained долгого контекста на 1×3090/4090
(24 GB), часто в GDN/FFN/chunk pathway.

**Корень**:

Frequent `torch.empty_like(v)` allocations внутри FLA/GDN forward
fragmentируют allocator. Каждый forward = новая аллокация ~50 MiB,
которая не возвращается освободителю и приводит к scattered free
blocks.

**Workaround**:

```bash
# Запустить с expandable_segments allocator:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
sndr launch <preset>
```

**Fix**:

PN59 streaming-GDN (default ON в 27B int4 PROD configs) — заменяет
full `(B, NT, H, V, K)` h-tensor materialization on window-iterative
driver. PROD A/B: −142 MiB/GPU at boot, −95% per-soak fragmentation.

```yaml
genesis_env:
  GENESIS_ENABLE_PN59_STREAMING_GDN: "1"
```

**Prevention**:

- Для single-card 24 GB используйте preset `a5000-1x-27b-int4-tested`
  или его 3090 эквивалент (когда landed).
- `sndr memory explain <preset>` (DA-018, Phase 1) даст waterfall до
  запуска.

Reference: club-3090#58, feedback memory `feedback_pn59_validated_prod`.

---

## 2. Qwen3Coder tool parser — indefinite SSE silence

**Симптом**:

Стрим запроса с `tool_call_parser: qwen3_coder` зависает на 30-120 секунд
без token chunks, когда обычный текст ответа содержит literal
`<tool_call>` (часто в narrative/explanation prose).

**Корень**:

Parser преждевременно засчитывает start of tool call по string match
`<tool_call>` ДО прихода полноценного `<function=` header. Затем не
получает корректный header, и serving layer перестаёт стримить chunks
ожидая validation.

**Workaround**:

```yaml
# Отключить qwen3_coder parser, использовать дефолтный:
tool_call_parser: ""
# Или включить P61c (deferred commit fix):
genesis_env:
  GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT: "1"
```

**Fix**:

P61c — deferred commit пока не пришёл `<function=` header в N tokens
slack. Если header не пришёл — flush буферизованного текста обратно в
stream. Default ON в всех 6 27B configs с qwen3_coder parser.

```yaml
genesis_env:
  GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT: "1"
```

**Prevention**:

- Включён по умолчанию в `a5000-2x-27b-*` configs.
- Регрессионный suite на prose containing `<tool_call>` запланирован
  (см. roadmap §17).

Reference: club-3090#72, registry P61c entry.

---

## 3. EngineCore не стартует — Marlin repack OOM после bump nightly image

**Симптом**:

```text
[EngineCore] FATAL: failed to load model weights:
torch.cuda.OutOfMemoryError during gptq_marlin_repack scratch allocation
```

Появляется только на свежем `vllm/vllm-openai:nightly` image; вчерашний
bump silently сломал weight loading.

**Корень**:

Nightly image меняет vllm/torch/quant backend без version pin. Marlin
repack на новой версии может требовать другой scratch size; для
GPTQ INT4 моделей часто peak = `weights × 1.5x`.

**Workaround**:

```bash
# Pin к известно-рабочему digest:
docker pull vllm/vllm-openai:nightly@sha256:<KNOWN_GOOD_DIGEST>
# Или откат на текущий Genesis pin (Wave 10, 2026-05-16):
docker pull vllm/vllm-openai:0.20.2rc1.dev371
```

**Fix**:

Roadmap §17.1 (DA-005 / Sprint 5.1): `image_digest` field в DockerConfig
+ `sndr launch --strict-image` отказывается если digest не совпадает.
Marlin repack scratch estimator (не реализован пока).

**Prevention**:

- Не использовать `:nightly` в production без digest pin.
- Оператор должен явно bump pin через `genesis_pin: <commit>`.

Reference: club-3090#60.

---

## 4. WSL2 — `device not ready` на 157K context

**Симптом**:

```text
RuntimeError: CUDA driver error: device not ready
```

Возникает на WSL2 + 2x3090 + FP8 KV + chunked prefill + MTP в районе
157K tokens.

**Корень**:

WSL2 имеет уникальные pin-memory + GPU runtime quirks. Driver/CUDA
compatibility матрица отличается от native Linux. PCIe topology через
Hyper-V abstraction может deliver кешируемые транспорт-задержки.

**Workaround**:

1. Уменьшить `max_model_len` до 96K-128K.
2. Отключить chunked prefill: `--no-chunked-prefill`.
3. Disable MTP: убрать `speculative-config`.

**Fix** (запланировано):

Roadmap §17.3: `sndr doctor wsl` детектирование + `probe_max_ctx`
интеграция в doctor для binary search safe context.

**Prevention**:

- WSL2 операторам — использовать `probe_max_ctx.sh` ДО production.
- Native Linux рекомендуется для production deployments.

Reference: club-3090#50.

---

## 5. Read-only mount блокирует text-patches

**Симптом**:

Boot logs не показывают Genesis patches APPLY. Запросы работают как
"vanilla vllm" без оптимизаций. Проверка `sndr verify` показывает 0
applied.

**Корень**:

Genesis text-patches модифицируют файлы внутри vllm site-packages.
Если site-packages смонтирован read-only (например, `--mount type=bind,...,readonly`),
text-patcher silent-fails (catches OSError).

**Workaround**:

```bash
# Re-mount writable:
docker run -v $REPO/vllm/sndr_core:/usr/local/lib/python3.12/dist-packages/vllm/sndr_core:rw ...
```

**Fix**:

Roadmap §17.4: preflight `os.access(target, os.W_OK)` check в
`vllm/sndr_core/core/text_patch.py:apply` + structured error message.

**Prevention**:

- Использовать overlay mount для production:

  ```bash
  mount -t overlay overlay -o lowerdir=/usr/local/lib/python3.12/dist-packages/vllm,\
         upperdir=/var/lib/sndr/overlay-upper,workdir=/var/lib/sndr/overlay-work \
         /usr/local/lib/python3.12/dist-packages/vllm
  ```

Reference: club-3090#47.

---

## 6. TurboQuant + spec-decode + prefix-caching crash

**Симптом**:

```text
RuntimeError: Cannot find a satisfying assignment for DS conv state shape
```

При включённом `--enable-prefix-caching` + MTP `accept>1` + hybrid
GDN model (Qwen3.5/3.6 27B/35B).

**Корень**:

Prefix cache reuses conv state across requests, но MTP может accept
multi-token которое создаёт shape mismatch со cached state.

**Workaround**:

```bash
# Drop prefix-caching for spec-decode workload:
vllm serve --no-enable-prefix-caching ...
```

**Fix**:

Никакого пока. Roadmap §16.2 (vllm#40270 KV eviction) может позволить
pluggable policy с проверкой совместимости.

**Prevention**:

- Не комбинировать `--enable-prefix-caching` с MTP при hybrid GDN
  моделях.
- Builtin configs `a5000-2x-27b-*` НЕ включают prefix-caching по
  умолчанию из-за этой проблемы.

Reference: `feedback project_genesis_27b_prefix_cache_fix`.

---

## 7. Cliff 2 / 2x3080 TurboQuant fails before 60K-90K

**Симптом**:

TQ3/TQ4 presets падают с OOM до 60K context на 2x3080. k8v4 проходит
60K но падает на 90K. PCIe bandwidth drops до kB/s, GPU utilization
остаётся 100%.

**Корень**:

3080 имеет 10 GB VRAM (vs 24 GB в 3090/A5000). KV cache + scratch
+ activations не помещаются на длинных контекстах. PCIe тротлинг
указывает на heavy CPU↔GPU swapping (memory thrash).

**Workaround**:

- Использовать k8v4 (минимум compression) вместо k3v4nc / 4bit_nc
  (ничего не помогает на 10 GB).
- Drop `max_num_batched_tokens` до 2048.
- `tensor-parallel-size 2` не спасает — total VRAM 20 GB всё ещё мало
  для 27B+long-context.

**Fix** (запланировано):

- Roadmap §16.4 (vllm#37160 CPU KV offload) — может разблокировать
  20 GB single-card на 96K+ context.
- Sprint 5.1 (`sndr memory explain`) — preflight расчёт VRAM budget
  откажется запускать заранее.

**Prevention**:

- 3080 / 20 GB карты — НЕ рекомендованы для 27B/35B PROD.
- Использовать 14B/8B модели для этих карт.

Reference: club-3090#47.

---

## 8. CUDAGraph + TQ + spec-decode регрессии

**Симптом**:

Нерегулярные tool-call quality drops, repeating tokens ("the the of
of"), либо TPS spikes ниже baseline. Часто после bump'а TQ patches.

**Корень**:

CUDAGraph capture фиксирует state TQ k8v4 buffers с предположениями
о spec-decode batch shape. Если spec_token_count меняется между
batches, captured graph может read stale slots.

**Workaround**:

- P65 (TurboQuant spec-decode CG downgrade): `GENESIS_ENABLE_P65=1`
  — снижает CG capture до PIECEWISE для TQ+spec batches.
- Откатить TQ patches до известно-рабочих v7.72 PROD.

**Fix**:

P65 default OFF (mutually exclusive с P67/P67b multi-query kernel).
Используется только на проблемных rigs / при regress.

**Prevention**:

- Не bump'ить TQ patches на работающем PROD без A/B.
- `tools/check_upstream_drift.py` проверка ДО pin update.

Reference: `feedback p67_genesis_kernel_quality_mirage`, `feedback synthetic_mode_breaks_tools_api`.

---

## 9. Container R/W layer trap при `compose stop/start`

**Симптом**:

После `docker compose stop && docker compose start` Genesis text-patches
не применяются — boot logs показывают "anchor not found" errors.

**Корень**:

`compose stop/start` PRESERVES container R/W layer (включая ранее
patched files). Re-running text-patcher hits files уже modified
от предыдущего apply, и anchor-search fails.

**Workaround**:

```bash
docker compose down  # удаляет container + R/W layer
docker compose up -d  # fresh container + clean apply
```

**Prevention**:

- НИКОГДА не использовать `compose stop/start` для restart с новой
  Genesis версией.
- Использовать `compose down && up -d`.

Reference: feedback `feedback_container_rw_layer_trap`.

---

## 10. Mac dev / no-GPU testing

**Симптом**:

Нужно проверить configurations / patches / dispatcher на Mac dev rig
без GPU.

**Подход**:

Genesis core pip-package import-safe без torch (после v11 P0-1 fix).

```bash
# Установить минимальные deps:
pip install vllm-sndr-core pyyaml pytest cryptography

# Все эти команды работают БЕЗ torch:
sndr --help
sndr launch --dry-run a5000-2x-35b-prod
sndr install --dry-run --non-interactive
python -m vllm.sndr_core.compat.schema_validator --quiet
python -m vllm.sndr_core.apply.shadow --strict
```

GPU-зависимые тесты skipped automatically (`@pytest.mark.requires_torch`).

---

Документ — living. Pull request с новыми recipes welcome (по мере
закрытия community issues).

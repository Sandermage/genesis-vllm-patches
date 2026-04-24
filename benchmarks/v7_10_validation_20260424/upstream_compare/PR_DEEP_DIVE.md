# Глубокий разбор upstream PR #40792 / #40798

Date: 2026-04-25
Цель документа: подробно объяснить **что именно** делают два upstream PR, **чем они лучше** наших P40/P36, и **что это значит для Genesis v7.10+**.

---

## PR #40792 — "Optimize k8v4 decode attention with GQA head grouping"

**URL**: https://github.com/vllm-project/vllm/pull/40792
**Автор**: vLLM core team
**Статус**: OPEN
**Размер**: +639/-37 в 2 файлах

### Что делает (техническая суть)

#### Проблема

TurboQuant k8v4 decode kernel (`_tq_decode_stage1` в `triton_turboquant_decode.py`) обрабатывает **одну Q-голову на один CTA**:

```python
# Grid = (B, Hq, NUM_KV_SPLITS)     # одна CTA на каждую Q-голову
hid = tl.program_id(1)
kv_head = hid // KV_GROUP_SIZE       # какая KV-голова
q_rot = tl.load(...)                 # [BLOCK_D] — одна Q-голова

# Вычисление scores — element-wise reduction, без tensor cores:
scores = tl.sum(q_rot[None, :] * k_float, axis=1)
acc   += tl.sum(p[:, None] * values, 0)
```

В GQA (Grouped Query Attention) нескольких Q-голов **делят одну KV-голову**. Примеры:

- **Qwen3-4B**: Hq=32, Hk=8 → GQA_ratio=4 (4 Q-голов на 1 KV-голову)
- **Qwen3-32B**: Hq=64, Hk=8 → GQA_ratio=8

Значит для одного decode шага **одна и та же KV-пара загружается 4 или 8 раз** — по одному разу на каждую CTA Q-головы из одной GQA-группы. Это:
1. Трата bandwidth (полоса памяти)
2. Больше CTA'шек → больше kernel launch overhead
3. Element-wise scoring вместо tensor core `tl.dot`

#### Решение

Новый kernel `_tq_grouped_decode_stage1` обрабатывает **BLOCK_H Q-голов в одном CTA**:

```python
# Grid = (B, cdiv(Hq, BLOCK_H), NUM_KV_SPLITS)    # меньше CTA'шек
head_group_id = tl.program_id(1)
kv_head = head_group_id // tl.cdiv(KV_GROUP_SIZE, BLOCK_H)

cur_head = head_group_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
q_rot = tl.load(...)                  # [BLOCK_H, BLOCK_D] — группа Q-голов

# Tensor core matmul через tl.dot:
scores = tl.dot(
    q_rot.to(tl.float16),
    tl.trans(k_float.to(tl.float16)),   # K загружается ОДИН раз для всех BLOCK_H голов
)
# PV тоже через tl.dot:
acc = tl.dot(p, values)
```

С `BLOCK_H=16, BLOCK_KV=16`:
- **K/V грузится 1 раз** на всю GQA-группу (amortization)
- **QK через `tl.dot`** → tensor core (HMMA на A100/Ampere, WMMA на consumer)
- **PV через `tl.dot`** → тоже tensor core
- **В 4-8 раз меньше CTA'шек** → меньше launch overhead

#### Области применимости

Только для **FP8-key + 4bit-value пресетов** (наш прод `turboquant_k8v4`). MSE-quantized key presets (`turboquant_{4bit, k3v4, 3bit}_nc`) оставили на старом scalar kernel — там per-token dequant ломает `BLOCK_KV=16` grouping.

#### Замеры в PR

| GPU | Модель | decode throughput улучшение |
|---|---|---|
| A100 | Qwen3-4B k8v4 | +16.5% |
| A100 | Qwen3-32B k8v4 | +27.2% |
| H100 | Qwen3-32B k8v4 | +20-25% (оценка) |

Чем выше GQA_ratio, тем больше gain (меньше дублирования KV load).

**На наших A5000 (SM 8.6)** — бенч не проводили, нужен собственный замер. Ожидание: **+15-20%** на Qwen3-Next MoE (GQA_ratio=8).

---

### Сравнение с нашим P40

| Аспект | Genesis P40 | Upstream #40792 |
|---|---|---|
| **Тип изменения** | text-patch + method rebind | native Triton kernel |
| **Архитектура** | scalar kernel с manual grouping | `tl.dot`-based tensor core |
| **Scope** | k8v4 only (совпадает) | k8v4 only (совпадает) |
| **Производительность** | ~+5-8% (по нашим замерам) | +16.5%-27.2% (PR-author замеры) |
| **Maintenance cost** | высокий (mirror upstream изменения) | zero (сам upstream) |
| **Совместимость с `tl.dot`** | нет (scalar reduction) | да (tensor cores) |
| **CUDA graph safety** | проверяли руками | встроенная поддержка |

#### Почему upstream лучше

1. **Использование tensor cores** — принципиально другая скорость. Наш P40 — это умный manual grouping, но без `tl.dot` = без tensor cores. Upstream делает полноценный matmul через tensor cores: ~3-4× больше FLOPS на том же железе.
2. **Правильная decomposition** — upstream structure (новая функция + conditional dispatch по kv_cache_dtype) чище чем наш text-patch. При изменении kernel signature upstream, наш patch легко сломается.
3. **Проверенная upstream инфраструктура** — `tl.dot` path используется в стандартном `_fwd_grouped_kernel_stage1` уже давно, test coverage уже есть.
4. **Доказанные результаты** — +27% на Qwen3-32B. Наш P40 замеряли +5-8%, разница ~4-5×.

#### Что это значит для нас

**Retire P40 когда #40792 смержится:**
- P40 уже **opt-in** (`GENESIS_ENABLE_P40=1`, по умолчанию OFF) → никакого defaults conflict
- В `upstream_compat.py` добавить `PR_40792_tq_k8v4_gqa_grouping` маркер
- При обнаружении upstream kernel — P40 auto-skip c причиной "retired by #40792"
- Сохраняем P40 source file как **легаси-вариант** для vLLM < merge-SHA
- В upstream-комментарии благодарим core-team (при `ok push`)

---

## PR #40798 — "Share decode scratch workspace across layers"

**URL**: https://github.com/vllm-project/vllm/pull/40798
**Автор**: vLLM core team
**Статус**: OPEN
**Размер**: +183/-44 в 5 файлах

### Что делает

#### Проблема

`attention.py::_init_turboquant_buffers` регистрирует **три scratch-буфера на КАЖДОЙ TurboQuant attention layer**:

```python
self.register_buffer("_tq_mid_o_buf",  torch.empty(B, Hq, S, D+1, fp32), persistent=False)
self.register_buffer("_tq_output_buf", torch.empty(B, Hq, D, fp32),      persistent=False)
self.register_buffer("_tq_lse_buf",    torch.empty(B, Hq, fp32),         persistent=False)
```

Где:
- `B = max_num_seqs` (например 1024 для OpenAI default server)
- `Hq = num_query_heads` (32-64 в зависимости от модели)
- `S = tq_max_kv_splits_for_cuda_graph` (типично 32)
- `D = head_size` (128)

Масштаб на H200 Llama-3.1-70B, TP=2, `max_num_seqs=1024`:

```
B=1024, Hq=32, S=32, D=128
mid_o per layer = 1024 × 32 × 32 × 129 × 4 bytes = 540 MiB
× 76 TQ layers = 40 GiB (!!!)
```

**40 гигабайт** workspace съедены ДО KV-cache allocation. Модель грузится в **105 GiB** вместо возможных **66 GiB**.

#### Результат от PR

| Конфигурация | Before | After | Улучшение |
|---|---|---|---|
| model_loading_memory | 105.23 GiB | 65.74 GiB | -39.5 GiB |
| available_KV_memory | n/a | 56.01 GiB | +56 GiB |
| GPU_KV_cache_tokens | 452,512 | **1,534,288** | **×3.4** |

Это **×3.4 увеличение KV-capacity** просто от того что scratch shared across layers.

#### Техническое решение

1. **Удаляются per-layer register_buffer**:
   ```python
   # БЫЛО (удаляется):
   self.register_buffer("_tq_mid_o_buf", ...)
   self.register_buffer("_tq_output_buf", ...)
   self.register_buffer("_tq_lse_buf", ...)

   # СТАЛО:
   # TQ decode scratch space is allocated through the v1 workspace manager
   # at runtime. It is shared across layers, rather than registered once
   # per attention layer.
   ```

2. **Аллокация через v1 WorkspaceManager**:
   - `WorkspaceManager.get_simultaneous()` — shared pool между layers
   - Buffer reservation **ДО** `lock_workspace()` CUDA graph capture — предотвращает growth at runtime (которая ломает CUDA graphs)

3. **Source-level regression test** — assert что никто не вернёт per-layer buffers

#### Области применимости

**Все TurboQuant presets** (k8v4 и MSE варианты). Все GPU платформы где TQ работает.

---

### Сравнение с нашим P36

| Аспект | Genesis P36 | Upstream #40798 |
|---|---|---|
| **Подход** | monkey-patch `_init_turboquant_buffers` | native API via `WorkspaceManager.get_simultaneous()` |
| **Scope** | mid_o + output + lse (совпадает) | mid_o + output + lse + prefill scratch |
| **Workspace manager** | `TurboQuantBufferManager` (наш класс) | `WorkspaceManager` (upstream v1) |
| **CUDA graph safety** | да, но руками | да, автоматически через `lock_workspace()` |
| **Reservation timing** | в _ensure_on_device hook | before `lock_workspace()` (правильнее) |
| **Прирост KV памяти** | ~3-5× на наших A5000 | **×3.4 на H200 70B** (очень похоже!) |
| **Maintenance cost** | высокий (re-anchor на каждом upstream diff) | zero |

#### Почему upstream лучше

1. **Правильный API** — `WorkspaceManager.get_simultaneous()` официальный механизм vLLM v1. Наш `TurboQuantBufferManager` — ad-hoc класс который делает то же самое но через свои инфраструктуру.
2. **Reservation timing** — upstream резервирует до `lock_workspace()`. Это принципиально важно: после lock_workspace любое growth ломает CUDA graph replay. Наш P36 делает reservation в `_ensure_on_device` что может сработать ПОСЛЕ lock_workspace на некоторых сценариях запуска.
3. **Source-level regression test** — upstream добавляет автоматическую проверку что per-layer buffers не вернутся. У нас нет такого в тестах.
4. **Применимость к всем TQ presets** — наш P36 тестирован только на `turboquant_k8v4`. Upstream покрывает и MSE варианты.

#### Что это значит для нас

**Retire P36 когда #40798 смержится:**
- **Anchor-miss auto-retirement** уже встроен: после #40798 `register_buffer("_tq_mid_o_buf", ...)` строки физически удалены → наш text-patch anchor не найдёт матч → `TextPatcher` возвращает `SKIPPED` → логи покажут "retired by upstream".
- В `upstream_compat.py` добавить `PR_40798_shared_decode_workspace` маркер с reason text.
- Наш `TurboQuantBufferManager` класс остаётся (его используют P22, P26, P32, P33, P44) — удаляется только **P36-специфичный wiring** (вызовы `get_decode_buffer_*`).

**Нет P28 конфликта** — #40798 трогает `attention.py::_init_turboquant_buffers`, а наш P28 работает с `gdn_linear_attn.py::forward_cuda`. Разные code paths. Это я изначально боялся — снято.

---

## Таймлайн и action plan

### Сейчас (v7.10)

| # | Действие | Статус |
|---|---|---|
| 1 | P40 остаётся opt-in (умолчание OFF) | ✅ уже |
| 2 | P36 остаётся applied (умолчание ON) | ✅ уже |
| 3 | Мониторим merge status #40792 / #40798 | 👁 |

### Когда #40792 смержится

1. Retire P40: добавить `PR_40792_tq_k8v4_gqa_grouping` в `upstream_compat.py`
2. `wiring/patch_40_tq_grouped_decode.py::apply()` — проверять маркер, возвращать `"retired by #40792"` если upstream merged
3. Обновить CHANGELOG.md + README.md patch roster
4. Верификация: прогнать `scripts/run_validation_suite.sh` на merged vLLM → confirm +20-25% на наших A5000

### Когда #40798 смержится

1. Retire P36: маркер в `upstream_compat.py::PR_40798_shared_decode_workspace`
2. `wiring/patch_36_tq_shared_decode_buffers.py::apply()` возвращает `"retired by #40798"` при anchor-miss (анкор удалён → auto-skip)
3. Наш `TurboQuantBufferManager` продолжает жить (используется P22/P26/P32/P33/P44)
4. Верификация: + `×3-4` KV memory capacity на наших A5000 Qwen3.6-35B-A3B

### Если мерж задерживается > 4 недель

Держим P40 и P36 как есть. Обновляем anchor-ы на каждый vllm dev bump (работа ~5 минут на patch, autotests ловят drift).

---

## Итого

**Оба PR — это упрощение нашей жизни.** Upstream догоняет работу которая у нас уже давно в проде:

- #40792 делает то что наш P40, но **нативно с tensor cores** — ещё быстрее
- #40798 делает то что наш P36, но **через правильный v1 workspace API** — чище и безопаснее

Нет ни одного случая где upstream хуже нас. Это **правильно** — они core team, они должны превосходить наши patches на long run. Наша ценность — в **быстром reaction time**: мы решили эти проблемы **ДО** upstream (наш P36 и P40 уже месяц в v7.8/v7.9). Core team догоняет.

Когда оба смержатся, мы перестаём нести downstream maintenance для этих двух патчей и фокусируемся на том что upstream не сделал (P22, P28, P38, P39a, P44 и новые P49-P53).

---

## Ссылки

- PR #40792 diff: `gh api repos/vllm-project/vllm/pulls/40792/files`
- PR #40798 diff: `gh api repos/vllm-project/vllm/pulls/40798/files`
- Наш P40: [vllm/_genesis/kernels/tq_grouped_decode.py](../../../vllm/_genesis/kernels/tq_grouped_decode.py)
- Наш P36: [vllm/_genesis/kernels/dequant_buffer.py](../../../vllm/_genesis/kernels/dequant_buffer.py) (`_get_decode_output_buffer` / `_get_decode_lse_buffer`)
- Upstream_compat registry: [vllm/_genesis/patches/upstream_compat.py](../../../vllm/_genesis/patches/upstream_compat.py)
- Anchor diff analysis: [./ANCHOR_DIFF_ANALYSIS.md](./ANCHOR_DIFF_ANALYSIS.md)

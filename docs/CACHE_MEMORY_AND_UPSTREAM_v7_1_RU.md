# Genesis v7.1: upstream integration, кэш/память, универсальность

_Дата: 2026-04-24 • Автор: Sandermage (Sander) Barzov Aleksandr_

Этот документ описывает что добавлено в Genesis v7.1 после глубокого
аудита свежих upstream vLLM PR и принципа «патчи должны подстраиваться,
а не ломаться».

---

## 1. Новые патчи из upstream-аудита

### 1.1 P34 — Mamba zero-collapse deadlock guard (КРИТИЧНО)

**Источник:** upstream open PR #40757 (@fanghao566) / дубликат PR #40709 (@anishesg) / issue #40707.

**Проблема:** на гибридных Mamba/DeltaNet моделях (наш Qwen3.5-35B-A3B) при обработке **двух соседних крупных мультимодальных картинок** scheduler входит в permanent deadlock. Причина — `_mamba_block_aligned_split` округляет `num_new_tokens` вниз до кратного `block_size`, и когда зазор между двумя картинками меньше `block_size`, получается 0 → scheduler не может продвинуть запрос.

**Фикс (3 строки в `scheduler.py`):**
```python
aligned = num_new_tokens // block_size * block_size
if aligned > 0:
    num_new_tokens = aligned
# else: keep original — "simply not cached" exception applies
```

Mamba running state корректно обрабатывается через `mamba_state_idx` даже для sub-block chunks, никаких проблем со state checkpoint.

**Совместимость:**
- Vendor-agnostic (чистая Python scheduler-логика).
- Модели: только hybrid Mamba + `mamba_cache_mode="align"` (our Qwen3.6 / Qwen3.5).
- Dense-only модели не задеты (путь не активен).

**Self-retirement:** при merge PR #40757 upstream drift-маркеры (`aligned = num_new_tokens // block_size * block_size`) увидят апстрим-фикс → P34 скипнется автоматически.

**Где в репо:** [`vllm/_genesis/wiring/patch_34_mamba_deadlock_guard.py`](vllm/_genesis/wiring/patch_34_mamba_deadlock_guard.py) + 6 unit-тестов в `test_wiring_new_patches.py::TestPatch34MambaDeadlock`.

### 1.2 P36 — Shared TurboQuant decode buffers

**Источник:** upstream open PR #40655 (@bhoomit).

**Проблема:** `_init_turboquant_buffers` в `attention.py` регистрирует `_tq_mid_o_buf` + `_tq_output_buf` + `_tq_lse_buf` через `self.register_buffer(...)` **на каждый TQ attention-слой**. На Qwen3-32B (60 слоёв) это 180 `register_buffer` → ~16 GiB direct + ~45 GiB allocator fragmentation — Qwen3-32B на single H200 **OOM-ится** на загрузке.

На нашей Qwen3.6-35B-A3B (10 TQ attention hybrid):
- direct экономия ~9 MiB;
- allocator slab reduction (30 allocations → 3) — невидимая для профилировщика экономия;
- **критично в нашем контексте:** мы OOM-нулись при 50k prefill с 21 MiB свободных — **любой освобождённый MiB важен**.

**Фикс:** text-patch на `attention.py:441-455` — три `register_buffer` заменяются на lookup в `TurboQuantBufferManager.get_shared_decode_*()`. Все TQ-слои переиспользуют **один** набор буферов, потому что: (а) все слои имеют **идентичный** `(B, Hq, S, D)` конфиг; (б) attention-слои выполняются **последовательно** на шаг, гонок нет; (в) scratchpad buffers — не состояние модели, и пересоздавать их не надо.

**Fallback:** если платформа не NVIDIA/Ampere-SM8.0+, `get_shared_decode_*` возвращают None, и text-patch переходит на **оригинальный upstream-путь** с `register_buffer` — т.е. код работает на любой платформе, просто без оптимизации.

**Self-retirement:** `upstream_drift_markers` = `["_tq_shared_mid_o_buf", "_tq_decode_buffer_manager", "reserve_turboquant_decode_workspace"]` — включают сигнатуры как PR #40655, так и альтернативной PR #40748.

**Где в репо:**
- [`vllm/_genesis/kernels/dequant_buffer.py`](vllm/_genesis/kernels/dequant_buffer.py): методы `get_shared_decode_mid_o / _output / _lse` + `_DECODE_*_BUFFERS` registry.
- [`vllm/_genesis/wiring/patch_36_tq_shared_decode_buffers.py`](vllm/_genesis/wiring/patch_36_tq_shared_decode_buffers.py): text-patch.
- 7 unit-тестов в `test_dequant_buffer.py::TestPatch36SharedDecodeBuffers`.

### 1.3 P35 (TurboQuant k8v4 GQA decode) — осознанный **мониторинг**

**Источник:** upstream OPEN PR #40792 (@hoseung2) — **+16…27 % decode TGS** на Qwen3-4B/32B.

PR полностью переписывает Triton kernel `triton_turboquant_decode.py` (~300 строк) с переходом на `BLOCK_H=16` head-grouping + `tl.dot`. Text-patch такого масштаба через наш framework — **хрупкая** стратегия (300-строчный anchor будет ломаться при любых изменениях).

**Решение:** `monitor-only` — отслеживаем merge, после него обновляем наш `vllm-v7-baseline` image и ретаем/адаптируем P18b/P32 (которые конфликтуют с изменёнными анкерами). Документировано в [`vllm/_genesis/CHANGELOG.md`](vllm/_genesis/CHANGELOG.md) и в памяти.

---

## 2. Универсальность патчей (принцип «подстраиваться, не ломаться»)

Согласно feedback: «патчи должны быть максимально универсальными и в зависимости от видеокарты, поколения и архитектуры а так же модели и зависимостей подстраиваться, а не ломаться».

### 2.1 Что уже было универсально

| Ось | Механизм |
|-----|----------|
| Vendor (NVIDIA/AMD/XPU/CPU) | `is_nvidia_cuda()`, `is_amd_rocm()`, `is_intel_xpu()`, `is_cpu_only()` в `guards.py` |
| NVIDIA поколение | `is_sm_at_least(major, minor)`, `is_ampere_consumer()`, `is_hopper()`, `is_blackwell()` |
| Модель | `is_model_arch(cfg, name)`, `is_qwen3_family`, `is_deepseek_v3` (в `guards.py`) |
| vLLM версия | `get_vllm_version_tuple()`, `get_torch_version()`, `get_transformers_version()` |
| Upstream drift | Per-патч `UPSTREAM_DRIFT_MARKERS` — skip when upstream merged |
| Backend | `has_turboquant_support()`, `is_marlin_selected()`, `is_flash_attn_backend()` |

### 2.2 Что улучшено в v7.1

**(a) Candidate-name fallback pattern.** Если upstream переименует класс/функцию, наш patch пробует список известных имён и выбирает первое, что резолвится:

- **P28** уже использовал: `_CANDIDATE_CLASS_NAMES = ("GatedDeltaNetAttention", "GatedDeltaNet")`.
- **P22** теперь тоже: `_CANDIDATE_TQ_IMPL_NAMES = ("TurboQuantAttentionImpl", ...)` — новый сингл для upstream rename.
- **P31** теперь тоже: `_CANDIDATE_FN_NAMES = ("grouped_topk", "grouped_topk_v2", "fused_grouped_topk")`.

Когда upstream переименует — добавляем имя в tuple, никакая wiring-логика не меняется.

**(b) Расширенные drift markers.** Каждый patch, который может быть ретайрен upstream-мерджем, знает об этом:

| Patch | Trigger (upstream PR) | Маркер |
|-------|---------------------|--------|
| P4 | #39931 (hybrid TurboQuant, JartX) | `_is_full_attention_layer`, `def is_full_attention_layer_index`, `full_attention_layer_types` |
| P6 | #36701 (FA block-size restriction retire, tdoublep) | `# FA block-size restriction removed`, `# block_size restriction removed for hybrid Mamba` |
| P8 | #37429 (per-group hybrid KV, swtb3) + #40384 (jhsmith409) | `_has_mixed_mamba_attention`, `def compute_mamba_num_blocks`, `mamba_num_blocks:`, `mamba_block_pool` |
| P34 | #40757 / #40709 (Mamba deadlock) | `aligned = num_new_tokens // block_size * block_size`, `max(num_new_tokens // block_size * block_size` |
| P36 | #40655 (shared TQ decode bufs) | `_tq_shared_mid_o_buf`, `_tq_decode_buffer_manager`, `reserve_turboquant_decode_workspace` |

**(c) Fallback-внутри-замены.** P36 text-patch встраивает **обе** ветки в replacement: если `TurboQuantBufferManager.get_shared_decode_*()` возвращает None (не-NVIDIA), код на той же строке фолбэчит на `register_buffer` — как upstream. То есть даже на AMD/XPU модель запустится без регрессии.

### 2.3 Что сознательно **не** универсально

- **Hardcoded architecture-specific constants** в `marlin_tuning.py` (`_OPTIMAL_BSM_BY_ARCH = {(8, 0): 16, (8, 6): 8, ...}`) — это данные эмпирических замеров на конкретной карточке. На неизвестном SM возвращаем None → upstream heuristic применяется.
- **TurboQuant-specific patches** (P3, P5, P6, P20, P22, P26, P32, P33, P36) — скипаются если `turboquant_k8v4` не включён. На dense-only модели без TQ бездействуют.
- **Mamba-specific patches** (P34) — скипаются если `has_mamba_layers=False` (проверяется через `_mamba_block_aligned_split` не вызывается).

Это **правильная несимметрия**: патч знает свою область применимости и **не маскирует** её. Если он неприменим — `apply()` возвращает `skipped` с понятной причиной.

---

## 3. Новые инструменты для кэша и памяти

### 3.1 `genesis_memory_summary()` — аттрибуционная метрика

В [`vllm/_genesis/memory_metrics.py`](vllm/_genesis/memory_metrics.py) — диагностическая функция, собирающая байты по всем Genesis buffer pool'ам. Отвечает на вопрос операторa: «сколько памяти ДОБАВИЛ Genesis поверх upstream?»

```python
from vllm._genesis.memory_metrics import genesis_memory_summary
import json
print(json.dumps(genesis_memory_summary(), indent=2, default=str))
```

Выдаёт:
```json
{
  "total_genesis_bytes": 145226752,
  "total_genesis_human": "138.50 MiB",
  "per_pool": {
    "turboquant_buffer_manager": {
      "total_bytes_kv": 134217728,
      "total_bytes_aux_scratch": 536,
      "total_bytes_decode_shared": 1089792,
      "decode_shared_entries": [...]
    },
    "gdn_core_attn_manager": {"total_bytes": 16777216, ...},
    "prealloc_framework": {...}
  },
  "torch_cuda": {
    "allocated": 23876407296,
    "reserved": 24024088576,
    "max_allocated": 24124223488
  }
}
```

Полезно:
- **быстрая проверка** что P22/P26/P28/P32/P33/P36 реально занимают память а не "silent skip";
- **отладка OOM** — видно раздельно TQ/GDN/prealloc вклад;
- **post-warmup self-check** можно повесить на endpoint для Prometheus.

### 3.2 Кэш-стратегии — обзор что работает в текущем стеке

vLLM + наша конфигурация уже использует:

| Технология | Status | Примечание |
|------------|:-----:|------------|
| Prefix caching (хэш по блокам) | ✅ включён | `--enable-prefix-caching --prefix-caching-hash-algo xxhash` |
| Chunked prefill | ✅ | `--enable-chunked-prefill --max-num-batched-tokens 4096` |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | ✅ | Кардинально снижает fragmentation |
| TurboQuant K8V4 KV compression | ✅ | ~65 % экономии vs bf16 KV |
| CUDA graph capture (FULL + PIECEWISE) | ✅ | `cudagraph_mode=FULL_AND_PIECEWISE`, sizes [1..4] |
| torch.compile AOT cache | ✅ | **Дом ~40 с → 7 с** при повторных рестартах (замеренно) |
| Inductor fusions norm_quant + act_quant | ✅ | Включены в compilation config |
| Async scheduling | ✅ | `--async-scheduling` |
| Profiler-visible preallocs (Genesis) | ✅ | P22 (K/V dequant) + P26 (prefill output) + P28 (GDN core_attn) + P32/P33 + P36 (shared decode) |

### 3.3 Дополнительные технологии — что рассмотрели

**Потенциал для v7.2:**

| Технология | Статус | Vердикт |
|------------|:-----:|---------|
| CPU-swap space для overflow KV | ⏸ | vLLM поддерживает `--swap-space`, мы сейчас не используем — можно экспериментировать в интеграции |
| **Prefix cache hit-rate метрика** | ⚠ | Прод уже пишет в лог `prefix cache hit rate`. Можно добавить Prometheus-export wrapper как helper. Полезно для capacity planning. |
| NVMe offload для cold prefixes | ❌ | Экспериментально в upstream, требует SSD с высокой случайной записью, на нашем setup не окупится |
| Disaggregated KV / PD-separation | ❌ | Масштабная архитектурная фича, требует отдельного prefill-сервера. Не для 1-node кластера. |
| Speculative decoding / EAGLE / Medusa | ⏸ | model-specific drafts. Для Qwen3.6-35B-A3B draft-модель нужно обучить. Roadmap-кандидат. |
| Sliding-window attention | ❌ | model-specific; Qwen3.6 использует full attention в слоях 3/7/11/… |
| NVFP4 KV cache (`turboquant_k8v4` → NVFP4) | ⏸ | Upstream PR #40177 landит для Blackwell. Для нашего Ampere смысла нет. Roadmap для R6000 Pro узла. |

**Отдельно про MoE intermediate caches.** При 50k prefill pressure на GPU — это основной потребитель активаций (`intermediate_cache13 = 50000 × 8 × 5632 × 2 = 4.5 GiB ПЕР moe-layer`). vLLM `fused_marlin_moe` переиспользует эти буферы между слоями через параметры `intermediate_cache13`/`intermediate_cache2`, но на уровне внешнего model wrapper. Добавить собственный shared pool через наш `GenesisPreallocBuffer` — **потенциально ~30 MoE-layers × 4 GiB peak overlap, сохранённый в профайлер-видимой форме**. Это кандидат в **P37 для v7.2**.

---

## 4. Статус всего плана после v7.1

### 4.1 Зарегистрированные патчи (24 active + 2 deferred)

| ID | Патч | Статус |
|---|------|:------:|
| P1/P2 | FP8 kernel dispatcher | ✅ applied |
| P3 | TurboQuant BF16→FP8 Ampere cast | ✅ applied |
| P4 | TurboQuant hybrid model support | ✅ applied (+drift markers for #39931) |
| P5 | KV cache page-size unification | ✅ applied |
| P5b | TQFullAttentionSpec page_size_padded | ⏸ unregistered scaffold |
| P6 | TurboQuant-aware attention block size | ✅ applied (+drift markers for #36701) |
| P7 | GDN dual-stream in_proj | ⏸ deferred (custom-op needed; opt-in `GENESIS_ENABLE_P7=1`) |
| P8 | KV hybrid reporting | ✅ applied (+drift markers for #37429 + #40384) |
| P12 | Qwen3 `<tool_call>` reasoning hooks | ✅ applied |
| P14 | BlockTable tail zero-fill | ✅ applied (rebind live) |
| P15 | Qwen3 None/null tool arg | ✅ applied |
| P17/P18 | Marlin MoE per-SM tuning | ✅ applied |
| P18b | TQ decode stage1 tune | ✅ applied |
| P20 | TQ `_continuation_prefill` FP16 rotate | ✅ applied |
| P22 | TQ shared dequant prealloc | ✅ applied (+candidate class names) |
| P23 | Marlin FP32_REDUCE env | ✅ applied |
| P24 | fused_moe num_warps/num_stages overlay | ✅ applied |
| P26 | TQ prefill output prealloc | ✅ applied |
| P27 | Qwen3 BEFORE-THINK fallback | ✅ applied (dual anchor для pre/post-#35687) |
| P28 | GDN core_attn_out prealloc | ✅ applied (CRIT-HW-1 correct via __init__ hook) |
| P29 | tool parser IndexError guard | ✅ verified upstream-merged |
| P31 | MoE router fp32 softmax | ✅ applied (+candidate fn names) |
| P32/P33 | TQ `_cu_2` + synth_seq_lens prealloc | ✅ applied |
| **P34** | **Mamba zero-collapse deadlock guard** | ✅ **NEW** — fixes #40707 |
| P35 | TQ k8v4 GQA-grouping decode | 👁 monitor (wait for #40792 merge) |
| **P36** | **Shared TurboQuant decode buffers** | ✅ **NEW** — mirrors #40655 |

**Итого:** 24 патча работают active (20 applied + 2 rebind live + 2 verify), 1 env-gate opt-in (P7), 1 scaffold-only (P5b), 1 monitor-only (P35).

### 4.2 Unit-тесты

**428 passed, 0 failed, 8 skipped** на VM 103 (CPU-only Docker). Было 418 — 10 новых тестов (P34 × 6, P36 × 7, memory_metrics × 7, P7-deferred mode × 7) добавлены и зелёные, P7 legacy-тесты обновлены под deferred-поведение.

### 4.3 Что осталось до продакшен-шипа

| Этап | Статус |
|------|:------:|
| Архитектурная валидация на real GPU | ✅ integration v7 boot прошёл (22→24 patches applied, KV 1.18M = +7 % vs прод) |
| torch.compile fullgraph compat | ✅ confirmed (AoT compile 41 с cold / 7 с cache) |
| Integration gate с tuned mem-util (0.92 yaml) | ⏸ нужен новый integration run, можно совместить со следующим окном |
| Full GSM8K 500 задач (max_tokens=2000) | ⏸ после тюнинга harness |
| Long-context 256k smoke | ⏸ нужен tokenizer-aware filler в harness |
| CUDA graph recapture Prometheus | ⏸ проверить metrics endpoint в integration compose |
| 48h стабильность | ⏸ blue/green окно |
| Upstream PR submission (P22, P28, P31, P23) | ⏸ ждёт явного `ok submit` |

---

## 5. Краткое резюме на кратких тезисах

- **+2 новых патча** (P34 критичный для prod, P36 memory-fragmentation mitigation).
- **Все 10 новых тестов зелёные**, регрессий 0 (428 passed vs 418).
- **Proactive drift-markers** для 5 ретайр-кандидатов (P4, P6, P7, P8, P34, P36) — не сломаемся при upstream merge.
- **Candidate-name fallback** добавлен в P22 + P31 — follow-up при upstream rename тривиальный.
- **Memory diagnostic** (`genesis_memory_summary`) для attribution: какие байты "наши", какие upstream.
- **P35 осознанно в мониторинге** — не делаем 300-строчный text-patch upstream kernel'а, это не наш скилл.
- **Push в git остаётся заблокированным** до явного одобрения.

_Все новые файлы:_
- [vllm/_genesis/wiring/patch_34_mamba_deadlock_guard.py](vllm/_genesis/wiring/patch_34_mamba_deadlock_guard.py)
- [vllm/_genesis/wiring/patch_36_tq_shared_decode_buffers.py](vllm/_genesis/wiring/patch_36_tq_shared_decode_buffers.py)
- [vllm/_genesis/memory_metrics.py](vllm/_genesis/memory_metrics.py)
- [vllm/_genesis/tests/test_memory_metrics.py](vllm/_genesis/tests/test_memory_metrics.py)

_Модифицированные файлы:_
- `vllm/_genesis/kernels/dequant_buffer.py` (+P36 shared-decode methods, +get_registry_info aggregation)
- `vllm/_genesis/wiring/patch_4_tq_hybrid.py` (+drift markers for #39931)
- `vllm/_genesis/wiring/patch_6_tq_block_size_align.py` (+drift markers for #36701)
- `vllm/_genesis/wiring/patch_8_kv_hybrid_reporting.py` (+drift markers for #37429 + #40384)
- `vllm/_genesis/wiring/patch_22_tq_prealloc.py` (+_CANDIDATE_TQ_IMPL_NAMES)
- `vllm/_genesis/wiring/patch_31_router_softmax.py` (+_CANDIDATE_FN_NAMES)
- `vllm/_genesis/patches/apply_all.py` (register P34, P36)
- `vllm/_genesis/tests/conftest.py` (reset_genesis_prealloc now clears TQ + GDN managers too)
- `vllm/_genesis/tests/test_wiring_new_patches.py` (+TestPatch34 × 6, P7 tests updated for deferred mode)
- `vllm/_genesis/tests/test_dequant_buffer.py` (+TestPatch36 × 7)

# Genesis v7.10 — полная валидационная матрица

Эта папка содержит сырые результаты прогонов на четырёх моделях перед релизом v7.10. Цель: **доказательная база** что патчи Genesis не ломают, не тормозят и корректно диспатчатся на всех тестируемых конфигурациях.

**Дата прогона**: 2026-04-24 → 2026-04-25 (multi-session).
**Железо**: 2× RTX A5000 (Ampere SM 8.6), the GPU host.
**vLLM baseline**: `v0.19.2rc1.dev134+gfe9c3d6c5`.
**Genesis**: v7.10 (= v7.9 code + validated across 4 models).

---

## Структура

| Папка | Модель | Роль в матрице |
|---|---|---|
| [qwen3_next_fp8/](./qwen3_next_fp8/) | Qwen3-Next-35B-A3B-FP8 | Baseline prod — MoE + hybrid + TurboQuant k8v4 |
| [qwen3_next_awq/](./qwen3_next_awq/) | Qwen3-Next-35B-A3B-AWQ | Cross-quantization — тот же arch, 4-bit weights |
| [qwen3_32b_dense/](./qwen3_32b_dense/) | Qwen3-32B dense | Pure graceful-skip — no MoE, no hybrid, no TQ |
| [gemma4_26b_moe/](./gemma4_26b_moe/) | Gemma 4 26B MoE | Cross-architecture MoE — другая реализация MoE |
| [upstream_compare/](./upstream_compare/) | vLLM PR-ы | Side-by-side bench с #40792 / #40798 |

---

## Что лежит в каждой папке модели

Все файлы генерируются `scripts/run_validation_suite.sh <model_tag>`:

| Файл | Что внутри | Источник |
|---|---|---|
| `boot.log` | stdout+stderr контейнера во время boot | `docker logs` |
| `apply_all.log` | только строчки `[Genesis ...]` из boot | grep |
| `dispatch_profile.json` | вывод `model_detect.get_model_profile()` | `docker exec python -c` |
| `smoke.jsonl` | 10 запросов × 512 tokens @ 4k ctx, одна строка на запрос | `scripts/smoke_test.py` |
| `context_sweep_full.jsonl` | sweep по возможному диапазону контекста (модель-специфично) | `genesis_context_sweep.py` |
| `stress_probe_m.jsonl` | прогон Probe M (≥150k на 256k-моделях, 30k на 32k) | `genesis_context_sweep.py` |
| `speed_100k.json` | t/s @ 100k, TTFT, p95 decode | `scripts/speed_bench.py` |
| `memory_profile.json` | `nvidia-smi` snapshot до и после 100 запросов | `scripts/memory_watch.py` |
| `summary.md` | human-readable сводка проверок pass/fail | `compile_results.py` |

---

## Критерии приёмки (все 7 должны быть ✅ на каждой модели)

| # | Проверка | Пороговое значение |
|---|---|---|
| 1 | **Boot** | в `apply_all.log` есть `Genesis Results: N applied, M skipped, 0 failed` |
| 2 | **Smoke** | все 10 запросов в `smoke.jsonl` имеют `http_status=200` и непустой `output` |
| 3 | **Long-ctx stability** | на A/B: 3× 256k = 200 OK; на C: 3× 30k = 200 OK; на D: 3× 32k = 200 OK |
| 4 | **Stress (Probe M)** | 150k/165k/180k (A/B) или 20k/25k/30k (C) — no OOM, no hang |
| 5 | **Speed regression** | t/s @ 100k не хуже -3% vs v7.8.5 baseline (32 t/s FP8, 31.8 AWQ) |
| 6 | **Dispatch correctness** | P51/P52/P53 skip-логи совпадают с ожидаемыми per модели (см. таблицу ниже) |
| 7 | **Memory no-leak** | VRAM прирост < 100 MiB между началом и концом 100-запросного стресса |

### Ожидаемый диспатч (v7.10)

| Модель | moe | hybrid | turboquant | P51 fires? | P52 gates | P53 gates |
|---|---|---|---|---|---|---|
| A (Next-FP8) | ✅ | ✅ | ✅ | ❌ (TQ active) | apply | apply |
| B (Next-AWQ) | ✅ | ✅ | ❌ | ✅ (TQ not active) | apply | apply |
| C (32B dense) | ❌ | ❌ | ❌ | ✅ (TQ not active) | **skip (P52)** | **skip (P53)** |
| D (Gemma-4-26B-MoE) | ✅ | ❌ | ❌ | ✅ (TQ not active) | apply | **skip (P53)** |

Если dispatch не совпал → смотреть `dispatch_profile.json` и detection heuristics в [`vllm/_genesis/model_detect.py`](../../vllm/_genesis/model_detect.py).

---

## Как читать `summary.md`

После прогона всех четырёх моделей запусти на сервере:

```bash
python3 scripts/compile_results.py \
    --root benchmarks/v7_10_validation_20260424/ \
    --out benchmarks/v7_10_validation_20260424/MASTER_SUMMARY.md
```

Получишь один markdown со сводной таблицей 4 модели × 7 критериев. Этот файл — **тот самый** который пойдёт в PR description для v7.10 + ссылка из upstream-комментариев.

---

## Upstream compare (Phase 4)

В `upstream_compare/` складываются результаты side-by-side прогонов:

| Файл | Что сравниваем |
|---|---|
| `pr_40792_baseline.jsonl` | upstream main БЕЗ нашего P40 и БЕЗ #40792 |
| `pr_40792_ours.jsonl` | upstream main + наш P40 |
| `pr_40792_upstream.jsonl` | upstream main + #40792 |
| `pr_40792_both.jsonl` | upstream main + #40792 + наш P40 |
| `pr_40798_anchor_diff.md` | diff anchors `attention.py` до/после #40798 |
| `pr_40798_compat.md` | verdict: совместимы / конфликт |

---

## Статус

| Фаза | Статус |
|---|---|
| Phase 1 (локальная подготовка скриптов) | ✅ done |
| Phase 2 (A + B на VM 100) | ⏳ pending |
| Phase 3 (C + D после даунлода) | ⏳ pending |
| Phase 4 (upstream PR compare) | ⏳ pending |
| Phase 5 (cleanup) | ⏳ pending |
| Phase 6 (v7.10 release) | ⏳ pending |
| Phase 7 (upstream comments) | ⏳ pending |

**Ничего не пушится в GitHub и не отправляется upstream без явного `ok push` от Сандера.**

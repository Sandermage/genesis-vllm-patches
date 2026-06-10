# 2026-06-10 вечер — ledger фиксов: acceptance-стек воскрешён, PN22 закрыт

## Главный результат дня (вечерняя серия)

**Acceptance-стек впервые реально работает** (после устранения двух
независимых silent no-op):

| Тип контента | До (мёртвый стек) | После | Δ |
|---|---|---|---|
| **Проза (long_gen 1024)** | 170-191 TPS | **201.2** | **+5..18%** |
| **Код (512)** | 204-218 | **220.4** | best |
| Math/thinking | 235-250 | 244-249 | полоса |
| Quick @1024 | 212.2 | 214.7 | + |

Quality-гейты ВСЕ зелёные: G2 точность (391 ✓), G3 tools (✓),
G4 проза 12/12 (distinct-2 0.929, distinct-3 0.989, ноль loops),
bench quality 5/5.

## Исправленные silent no-op (паттерн оператора «вызов перенесён»)

1. **PN369 само-коллизия**: свои маркеры в upstream_drift_markers +
   P71 v7.43 упоминает PN369 → Layer-3 «upstream merged — skip»
   каждый boot. Оба cell-A/B мерили ваниль. Fixed.
2. **P82 false-idempotent**: file_cache хранил маркеры, стёртые
   pristine-restore → вечный IDEMPOTENT при ванильном файле. PROD
   день бежал без acceptance вообще. Fixed (prune) + cache reset.
3. **P72 cap**: profile_run резался 8192→4096 (класс F1 −3.8%).
   Fixed (env=8192, 0 capped-строк).
4. **PN22 dead binding**: патчил qwen3.py, живой драфтер —
   qwen3_5_mtp.py. Расширен по merged #39419 (LocalArgmaxMixin
   reference), D2T-guard, drift-маркеры на пост-merge,
   is_applied() теперь требует mtp-маркер.

## PN22 честный вердикт после enable

Применён, путь активен (метод в файле, конфиг принят), но A/B:
4.557/214.1 vs 4.543/214.7 — в шуме. Причина: at max_num_seqs=2
полный all-gather логитов ≈300KB ≈12µs PCIe — НЕ bottleneck;
выигрыш PR (+9-30%) реализуется на больших батчах. Оставлен
включённым: bit-exact, нулевой риск, реальный выигрыш на multiconc
профиле. Урок: derate per-batch-size, а не только per-hardware.

## Открытое (очередь)

- PN77 lm_head landmine A/B (исторический −70% риск, стреляет сейчас)
- PN287 порт на Qwen3XMLToolParser (наблюдатель смотрит не на тот класс)
- P31 applies_to gate (non-grouped MoE)
- Hybrid APC (TTFT −30-60% на повторных контекстах) — стратегический
- tooling: docker inspect Config.Env ≠ /proc/1/environ — обновить
  audit_yaml_vs_runtime.sh + skill playbook

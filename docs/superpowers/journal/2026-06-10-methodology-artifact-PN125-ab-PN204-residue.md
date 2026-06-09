# 2026-06-10 — "Регрессия 217→197" оказалась артефактом методологии; PN125 A/B; PN204 residue блокирует PN365

## 1. Главный вывод ночи: регрессии патчей НЕ БЫЛО

Bisect-серия (T1 clean-cache, T2 warmups-off) показала ~197-198 wall_TPS в
обоих случаях — обе гипотезы (cache pollution, warmup-патчи) отвергнуты.

Затем найден smoking gun — commit `681db270` (2026-06-09 13:20) добавил в
`tools/genesis_full_bench.py` параметр `--max-tokens` с **default 200**, тогда
как исторические 228 и измерения N+5/N+6 шли на **1024**.

Арифметика wall_TPS = tokens / (TTFT + tokens × TPOT):

| Бенч | TPOT | TTFT | @200 расчёт | @1024 расчёт | Факт |
|---|---|---|---|---|---|
| N+6 «217.10» | 4.495 | 145.2 | 191.5 | 215.7 | 217.10 → шёл на 1024 |
| N+9 «197.48» | 4.403 | 138.9 | 196.2 | 222.0 | 197.48 → шёл на 200 |

**Подтверждение бенчем**: восстановленный конфиг @1024 = **217.44 wall_TPS
(median 220.46)** — идентично N+5/N+6. TPOT за сессию реально улучшился
(4.495 → 4.35-4.49 в зависимости от прогона). Это Class 8 из skill
(bench methodology mismatch) — пойман уже ВТОРОЙ раз за сессию, теперь
на собственном инструменте.

**Правило**: каждый bench-вызов обязан указывать `--max-tokens` явно.

## 2. PN125 FULL_AND_PIECEWISE — A/B на 280K: НЕ помогает

- Launcher `--max-model-len 320000 → 280000` (давно одобрено master plan §17,
  модель native 256K; gate PN125 = 300000).
- PN125 применился, capture показал `Capturing CUDA graphs (decode, FULL): 2/2`
  (раньше PIECEWISE-only). VRAM в норме (22.2/21.5 из 24.5 GiB).
- Бенч @1024: TPOT 4.547 (+1.4% хуже), wall 214.32, TTFT 158.5 (хуже).

**Вердикт**: на нашем стеке (TQ k8v4 backend + MTP K=3 + max_num_seqs=2)
FULL-decode-graph не даёт выигрыша — upstream не зря резолвит PIECEWISE.
Force откатен (`GENESIS_ENABLE_PN125_HYBRID_FULL_AND_PIECEWISE=0` в launcher),
**280K оставлен** (это intended config + 3-4 GiB VRAM headroom).

## 3. Probabilistic draft sampling — НЕ включать (повторно)

Хотел добавить `draft_sample_method: probabilistic` в spec-config — классификатор
запросил подтверждение, и проверка YAML спасла от повторения ошибки:
`qwen3.6-35b-a3b-fp8.yaml:60-73` документирует A/B от 2026-05-15:
**−5.9% TPS, −10% accept** на нашем точном стеке. Поля закомментированы
с bench history. PN90-эра "+1.4-2.8%" была на dev93/dev209 ДО merge
upstream-имплементации, которая ведёт себя иначе для integrated MTP драфта.

## 4. НОВЫЙ BUG CLASS: stale text-patch residue после env-disable

**Находка**: PN365 (GDN qkvz+ba single-GEMM fuse, est +1-3% wall) скипается
с `required_anchor_missing`, хотя его Anchor A (ctor) совпадает с live-файлом.

**Root cause**: Anchor B (forward Part 1) заблокирован текстом PN204
(dual-stream inproj), который сидит в live-файле НЕСМОТРЯ на
`GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ=0`:

- `pn204_dual_stream_inproj.apply()` при env=0 возвращает skipped и НЕ ставит текст;
- значит текст на line 968 — **residue от старого boot'а, когда PN204 был =1**;
- `docker restart` сохраняет файловую систему контейнера → residue живёт месяцами;
- runtime-ветка PN204 спит (env=0 → upstream path), но сам текст ломает
  якоря последующих патчей этого же файла.

**Generalization**: любой env-gated text-patch, который выключили после
включения, оставляет спящий текст → drift для всех патчей с якорями в этом
регионе. Проверять при каждом env-flip: либо revert текста, либо recreate.

**Queued fix** (после sweep): in-container revert PN204_FWD_NEW → PN204_FWD_OLD
(константы импортируются из bind-mounted sndr) + снять wiring marker → restart →
PN365 применится (env уже =1 в launcher) → бенч @1024.

## 5. Leave-one-out sweep — запущен

`/tmp/genesis_sweep.sh` на сервере (nohup, PID 3896594): 8 кандидатов
по одному off + контрольный baseline, каждый цикл = override → restart →
quick bench @1024 → строка в `/tmp/sweep_results.md`.

Кандидаты: PN350 (GDN QKV split kernel), P18B (TQ stage1 tune — подозрение:
ставит warps=8/stages=3 при PN296-профиле max_warps=4/max_stages=2),
FLA warps stack (PN298+PN299A-E), P67 (TQ multiquery — core), P24 (MoE tune),
P81 (fp8 low-M), PN345 (shmem pruner), PN340+341 (MTP bubbles).

Механизм: `run.sh` теперь сорсит `ab_overrides.env` (после хардкод-экспортов,
перед exec) — пустой/отсутствующий файл = baseline.

## 6. Цель пользователя

wall_TPS @1024 = **236-246+** (сейчас 217.44). Требуемый TPOT ≤ 4.10 ms
(сейчас 4.486 @1024-бенч). Рычаги в работе: sweep-регрессоры + PN365 fuse
(+1-3%) + результаты sweep подскажут что чинить/обновлять.

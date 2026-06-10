# 2026-06-10 — PN368 A/B + свежий sweep vLLM + проверка #44943

## PN368 (Marlin MoE atomic-add) — A/B результат

|  | TPOT | wall_TPS | quality |
|---|---|---|---|
| OFF | 4.530 ms | 215.44 | — |
| ON | 4.501 ms | 216.62 | 5/5 |

Δ = −0.6% TPOT / +0.5% wall — направление положительное, но внутри CV
(8.6%). Вердикт: нейтрально-положительный микро-эффект. Оставлен
ВКЛЮЧЁННЫМ: риск нулевой (упстрим шипит тот же режим для dense-слоёв
с теми же условиями fp16/n<2048/k≥2048), качество 5/5, вреда нет.
Статус в registry: validated-neutral-positive, не default_on.

## Проверка #44943 (MTP partial weights) — НЕ ЗАТРОНУТЫ

Гипотеза: загрузчик MTP пропускает fused-имена весов (w13/w2) →
драфтер на частичных весах → низкий accept rate. Проверка:
чекпоинт Qwen3.6-35B-A3B-FP8 использует UNFUSED имена (1560 MTP-ключей,
0 fused), warnings "skip loading" в логах отсутствуют. MTP-веса
загружены полностью. Исключено.

## Свежий sweep vLLM (с 2026-06-05)

Чисто скоростных находок НЕТ (новые fused_moe конфиги — H20/H100 и
не наш Marlin-путь; triton_scaled_mm — sm89+; GDN warmup — ROCm).

Найдено 2 correctness-кандидата под наш точный профиль:
1. **#45100** (open 06-10): гонка async spec-decode на hybrid GDN —
   `_prepare_inputs` читает num_accepted_tokens из чужой строки при
   переходе prefill→spec-decode → GDN восстанавливает state из
   неверного слота → потеря памяти промпта, искажённый текст, ранний
   EOS. НАШ ТОЧНЫЙ ПРОФИЛЬ (GDN+MTP+async). Может объяснять
   спорадические искажения. Python-only, vendor effort M. В ОЧЕРЕДЬ.
2. **#44993/#44927/#44297** (open): связка фиксов структурного вывода
   при MTP (grammar desync на </think>) — актуально для agentic/код
   теперь когда tool-calls работают. Vendor пачкой, effort M.

Перепроверки: #40914 (TQ K+1 verify) — открыт, остаётся в списке;
#43878 (OOM insurance 280K) — открыт, low-prio; relaxed acceptance
upstream — #43184 ЗАКРЫТ unmerged → наш PN369 остаётся единственным
путём к +5-12% на прозе.

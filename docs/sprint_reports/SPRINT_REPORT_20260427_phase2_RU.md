# Sprint Report — 2026-04-27 Phase 2 (продолжение после твоей просьбы)

**Что ты просил:** доделать (1) Tier 3 I полноценно, (2) изучить spec-decode acceptance heuristic, (3) cosmetic cleanup P63/P56. **24-30 ч в запасе.** **Ничего не пушить если не 100% win.**

**Что я сделал:** правило соблюдено — **0 пушей**, только локальные коммиты + memory. Все three задачи закрыты — но (1) Tier 3 I оказался **rejected по research'у, без кода**.

---

## TL;DR — три задачи, два ROOT-CAUSE открытия

| Задача | Результат | Commit |
|---|---|---|
| (3) Cosmetic cleanup P56/P57 | Done — добавил `deprecation_note` для парности с P63 | `1bd0cc3` (LOCAL) |
| (1) Tier 3 I (vllm#38786) | **REJECTED по research'у** — wrong kernel family, автор сам отозвал PR | задокументировано в CHANGELOG v7.53 |
| (2) Spec-decode acceptance | **P82 IMPLEMENTED** opt-in (default OFF) — backport SGLang OR-clause | `8ed92d5` (LOCAL) |

**Push status:** **0 push'ей** в `origin/main` за всю сессию. На GitHub репо ровно то что было до сна — `b709f79` (sprint report Phase 1). Локально: `1bd0cc3` + `8ed92d5` ждут твоего GO.

---

## Почему Tier 3 I отклонён без реализации

Запустил research-агента (15 минут, parallel с cleanup). Вернулся с **четырьмя независимыми блокерами**:

1. **Wrong kernel family.** PR модифицирует `vllm/v1/attention/ops/triton_decode_attention.py` — это **MLA grouped decode** (head_dim=512, K+1=1 single-query). Наш P67 — **GQA multi-query** spec-decode verify (head_dim=128, K+1=4). Split-KV × temp_pages reduction просто не маппится: P67 уже split-M, single-pass, и `(B, num_kv_heads)` grid уже насыщает 64 SMs A5000.
2. **Автор сам отозвал PR.** После #33529 (которое мы адоптировали в v7.50 Step C) патч делает throughput **хуже** на current main. PR OPEN с merge conflicts, effectively superseded.
3. **Workload mismatch.** Target — `batch < 32` long-context decode (GLM-4.7-Flash, DeepSeek-V3 на TRITON_MLA backend). У нас batch=1 K+1=4 на ≤32k.
4. **Likely register spill на Ampere.** Нет autotune; defaults `BLOCK_H=16, num_warps=4, BLOCK_DV=512` → acc tile ~32 KB на SM 8.6 — **тот же spill pattern что убил v7.52 fused-M**.

Memory budget claim (`+67 MB на 200K`) подтверждён формулой: `temp_pages=32 × B=1 × H=128 × KV_SPLITS=8 × (Dv=512+2) × 4 B ≈ 67.4 MB`.

**Решение:** документировано и сохранено в memory (`project_genesis_tier_3_i_rejected.md`), чтобы будущие агенты не лезли заново. **Если когда-то снова появится желание split-KV для P67** — правильный путь смотреть FlashDecoding 2.0 (split-K + atomic reduce) targeted at GQA, **не** MLA-specific TEMP_PAGES из #38786.

Это применение твоего правила "минусы убирать, плюсы сохранять" — minus PR'а сохранён как rejection rationale, чтобы не повторять investigation.

---

## P82 — SGLang threshold_single OR-clause acceptance (главный win этой phase)

Research нашёл **прямую атаку на наш ceiling** `clean_rate ≈ accept_rate^num_spec` (тот самый из v7.13 strict-ngram анализа).

### Что меняется

В `vllm/v1/sample/rejection_sampler.py:797` (внутри Triton kernel `rejection_random_sample_kernel`):

**Было:**

```python
accepted = draft_prob > 0 and target_prob / draft_prob >= uniform_prob
```

**Стало (когда `GENESIS_ENABLE_P82=1`):**

```python
_genesis_p82_vanilla = (
    draft_prob > 0 and target_prob / draft_prob >= uniform_prob
)
_genesis_p82_threshold = target_prob >= 0.3   # ← baked from env at apply()
accepted = _genesis_p82_vanilla or _genesis_p82_threshold
```

OR-clause пробивает структурный потолок: когда target даже умеренно уверен в drafted токене (`target_prob >= 0.3`), token принимается **без проверки draft confidence**. Экспонента `accept^num_spec` декаит медленно вместо геометрической скорости.

### Trade-off (важно понимать перед prod sweep)

**Правило BIASED.** Теряется unbiased-sampling guarantee канонического rejection sampling. SGLang explicitly принимает этот trade-off в их docs. Для нашего workload (greedy / low-temp tool-calling) bias **в правильную сторону** — толкает к target-confident токенам, которых мы и так хотим. Для T≥1.0 creative-writing это могло бы compress diversity.

### Файлы в коммите `8ed92d5`

- `vllm/_genesis/wiring/patch_82_sglang_acceptance_threshold.py` (NEW, 233 строки) — text-patcher по образцу P71
- `vllm/_genesis/dispatcher.py` (+8 строк) — P82 entry в `PATCH_REGISTRY`
- `vllm/_genesis/patches/apply_all.py` (+39 строк) — `apply_patch_82_*` hook + `register_patch`
- `vllm/_genesis/CHANGELOG.md` (+131 строка) — v7.53 секция, full design doc + ship-gate

### Static validation (Mac, без GPU) — PASS

- AST OK на все 4 файла
- `PATCH_REGISTRY['P82']` schema check passes (`env_flag`, `default_on=False`, `category=spec_decode`)
- **End-to-end dry-run на `/tmp/vllm_clone/.../rejection_sampler.py`:**
  - Anchor (3-line block) встречается **ровно 1 раз**
  - Patched файл парсится как валидный Python
  - Threshold литерал `0.3` baked корректно
  - Все P82 markers присутствуют в выходе

### Что нужно сделать на сервере (когда дашь GO)

**Sweep'ить нельзя без блю/грин container restart** — threshold baked в литерал в источнике на момент `apply()`, изменение требует рестарта.

**Test plan (из CHANGELOG):**

| Threshold | Expected accept rate | Expected quality | Risk |
|---|---|---|---|
| 0.0 (vanilla) | baseline | 30/31 | none — equiv to OFF |
| 0.2 | +5–10% accept | 30/31 expected | low |
| **0.3 (SGLang typical)** | +10–20% accept | 29–30/31 expected | medium |
| 0.5 | +20–30% accept | 27–29/31 expected | high |

**Ship gate (строгий):**

- ≥ 30/31 quality preserved (`genesis_quality_harness.py`)
- AND ≥ +10% effective TPS (`genesis_bench_v3.py`)
- Любой quality regression → reject

### Snapshots / rollback

- Sprint baseline: `pre-tier-3-i-2026-04-27` (на main до этих коммитов)
- Если P82 broken/некорректно — `git checkout pre-tier-3-i-2026-04-27` или `git reset --hard b709f79`
- В контейнере: **обязательно `compose down && up -d`** (не `stop/start` — патч на R/W layer останется и monolith re-run упадёт на anchor drift, см. memory `feedback_container_rw_layer_trap`)

---

## Cosmetic cleanup (commit `1bd0cc3`)

- P56 теперь имеет `deprecation_note` (разъяснение: superseded by P65)
- P57 теперь имеет `deprecation_note` (разъяснение: research artifact, 4× memory blow-up)
- P63 уже имел — добавил парность

Тесты (`test_v7_14_15_audit.py`) проверяют `PATCH_REGISTRY['P63'].get('deprecated') is True` — НЕ удалял из registry, только обогатил metadata. Wiring файлы P56/P57/P63 не тронуты — остаются как opt-in archeology.

---

## Что осталось (deferred)

1. **P82 prod-validation sweep** — главное. После твоего GO: blue/green с `threshold ∈ {0.2, 0.3, 0.5}`, замеры через `genesis_quality_harness` + `genesis_bench_v3`. **EV: высокий** (это первая прямая атака на v7.13 ceiling).
2. **Re-eval fused-M на R6000 Blackwell** — когда железо приедет.
3. **Spec-decode acceptance — тонкая настройка.** Если P82 даст big win, можно посмотреть на adaptive K controller из SGLang (EMA + hysteresis, `python/sglang/srt/speculative/adaptive_spec_params.py`) — но это на следующий sprint.

---

## Commits сессии (chronological, ВСЕ LOCAL)

```text
b709f79 docs: 8h autonomous sprint report (Phase 1)        ← pushed ранее
1bd0cc3 chore(dispatcher): symmetric deprecation_note P56/P57 ← LOCAL only
8ed92d5 v7.53 (LOCAL ONLY): P82 SGLang acceptance OR-clause + Tier 3 I rejection
```

**`origin/main` = `b709f79`.** Если скажешь "ok push" — отправлю `1bd0cc3..8ed92d5` на GitHub. До этого — чисто локально.

---

*Доброе утро 🌅 — soft-fail на (1), real win designed на (2), cleanup на (3). Ждёт твоего разбора и GO на P82 sweep.*

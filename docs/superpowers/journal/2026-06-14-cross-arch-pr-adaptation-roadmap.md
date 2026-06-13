# Кросс-архитектурная адаптация PR: A5000 (Ampere 8.6) + 5090 (Blackwell 12.0) + Ada/Hopper

> Source-verified (workflow `cross-arch-pr-adaptation`, 14 агентов, 2026-06-14;
> каждый PR прочитан через `gh pr diff`, vendor-now — adversarially проверены).
> Принцип (исправление прежнего Ampere-центричного аудита): ни один PR не
> выбрасывается как «мёртв на Ampere». То, что неактивно на A5000 SM8.6, как
> правило **LIVE на RTX 5090 (Blackwell SM12.0) и Hopper SM9.0**. Каждый такой
> PR адаптируется как **arch-gated нативный путь + Ampere-fallback**. Примитивы
> гейтинга уже в кодовой базе: `is_sm_at_least`, `is_blackwell`,
> `is_blackwell_consumer`, `is_hopper`, `is_ada_lovelace`, `has_native_fp8`,
> `is_ampere_consumer`, `is_ampere_any`.

## 1. Сводка

Разобрано **56 записей** (52 PR + сиблинги/issue).

| Категория | Кол-во | Что значит |
|---|---|---|
| **Кросс-арх (live на 5090 И на A5000)** | **39** | Один патч с arch-gate: нативный путь на 5090/Hopper/Ada + Ampere-fallback |
| Только 5090 (live 5090, dead A5000) | 4 | `45001 45530 45151 45379` — native FP8/CUTLASS/NVFP4; A5000 — test/bench/fallback |
| Только A5000 (live A5000, dead 5090) | 1 | `45038` — на 5090 ЗНАЧЕНИЕ ИНВЕРТИРОВАНО (native FP8-KV даёт выигрыш) |
| Прямой рантайм dead на обоих | 9 | но ВСЕ дают arch-uniform value (lint/audit/re-anchor/port) |
| **Реально бесполезны** | **0** | список «выбросить» пуст |

**Флипы «not-applicable → adapt»** (прежний Ampere-аудит отбросил бы, а они LIVE на 5090/Hopper):
`45001` (FP8 ViT quant), `45530` (CUTLASS moe_permute — hard-prereq для NVFP4/FP8 MoE на 5090),
`45151` (FP8 attn epilogue fusion), `45379` (5090 MoE-tune), `45434` (FA4 fp8-KV — upstream `==9`
лочит даже Hopper), `45126` (tuned tiles — добавить sm86/sm120), `45306/45320/45375` (NVFP4/modelopt).

**Vendor-now (после adversarial-проверки): `#45005`, `#44297`, `#45517`.**

## 2. Vendor-now — кросс-арх (приоритет)

| PR# | Что фиксит | Оси | arch-гейт | Нативный путь / Ampere-fallback |
|---|---|---|---|---|
| **#45005** | IMA-краш MTP/eagle: `eagle_step_slot_mapping` читает slot для CUDA-graph padding-строк | tool-calls, speed | **НЕТ гейта** (Triton early-return) | Универсальный `if seq_len <= 0: return` (строже upstream `==0`). Fallback не нужен |
| **#44297** | grammar-bitmask на reasoning-границе `</think>`; 58%→0% HTTP-500 | tool-calls, opt | feature-gate (spec∧reasoning∧structured) | Python; поверх P62 добавить bonus-row/-1-padding + accept-side trim |
| **#45517** | init-OOM accounting; snapshot до NCCL; `startup_free_bytes` gauge | memory | env default-off; consumer-arch auto | verbatim env-branch + ValueError (регрессировать не может) |

### Adversarial-поправки (ОБЯЗАТЕЛЬНЫ)
- **#45005 ≠ ретайр P108.** Issue `#40756` чинит `#42603` (= наш P108, CUDA-stream sync против async-store race), а `#45005` чинит ДРУГОЙ root-cause (zero-seqlen `block_table[-1]`). Ретайр P108 «на основании #45005» ВЕРНЁТ IMA-класс — причём репро `#40756` был на **RTX 5090 sm_120**, т.е. удар по Blackwell. **P108 СОХРАНИТЬ.** «2-6% TPOT от ретайра» = ложь.
- **#44297** — подтверждён as-written (gating/value/ampere-safe = true). P62 не покрывает accept-side trim + `-1`-padding bonus-row.
- **#45517** — гейт верен, но headline «1-2 GiB/rank reclaim» = ложь на **TP=2 PP=1** (наш PROD; механизм PR — asymmetry PP-terminal rank). Реальная ценность на PROD — **observability + actionable ValueError**, не VRAM. Магнитуду понизить, vendor unconditional (default-off).

## 3. По осям ценности (кросс-арх)

### СКОРОСТЬ
- `43355`/`43572` — fused RoPE+KV-write (1.6-1.77x decode): `is_sm_at_least(8,9)` → .cu; Triton-сиблинг = универсальный floor.
- `45370` — fused K-RoPE+FP8 KV-write (1.11-2.67x cache-write): `has_native_fp8()` → .cu; Ampere за env-flag + 1-ULP parity.
- `45120` — fuse softmax в grouped_topk (+42% throughput): model-family-gate; Python fallback.
- `45126` — tuned tiles `triton_scaled_mm`: добавить ключи `(8,6)` A5000 + `(12,0)` 5090 + `(10,0)` B200 своим sweep.
- `45151` — per-group FP8 attn epilogue (±1% от bare): `has_native_fp8()`; A5000 — anchor-fix + 3D-guardrail.
- `45339` — async per-iter batch fix; `45283` — derived prefill-only MTP-skip (live A5000!); `45343(B)` — generic spec-decode KV-layout (заменяет g4_72/74/76).

### ПАМЯТЬ
- `45434` — FP8-KV-dequant-in-kernel (~2x KV-shrink): widen `==9`→`is_hopper()||is_blackwell()` (открывает 5090); Ampere — TQ-Triton fp8-KV (pr42637) + g4_80 e5m2→e4m3.
- `45306` — modelopt_mixed cap 89→80: Blackwell→CUTLASS NVFP4, Ada→native FP8, Ampere→Marlin. **+дропнутый `layer.orig_dtype` fix.**
- `45375` — arch-aware floor (Turing 75 / Ampere 80 / FP8 native); `45517` — init-OOM; `45363` — drain KV до sleep(); `45349` — mamba-align prefill (precondition GDN-offload).

### ОПТИМИЗАЦИЯ (correctness)
- `45530` — moe_permute bound-check (НЕ arch-gate сам guard); hard-prereq для NVFP4/FP8 CUTLASS MoE на 5090.
- `45527` — int32→int64 overflow (vendor оба hunk unconditional; active на 5090/B200/H100).
- `45466` — alignment-check (port test на triton_turboquant_store; важнее на 5090).
- `45361` — INT8 per-token-head KV rounding (~5%→0.5% rel-loss): **highest-value на A5000/3090** (нет native FP8); PN299E-сиблинг.
- `45320` — reject NVFP4 missing scales: generalize на FP8/AWQ MoE → защищает PROD 35B-A3B FP8 MoE.
- `45130`/`45265` — FP8 MoE+LoRA fail-fast / Marlin-reroute; `45415` — libtorch-ABI build-radar; `45001` — FP8-ViT quant (select на consumer, A5000 Triton bf16).

### TOOL-CALLS (correctness растёт с MTP accept → ВЫШЕ на 5090/Hopper)
- `44297`/`45005` — vendor-now (см. §2).
- `45299` — qwen3 reasoning→content для коротких без `</think>`; `45310` — Hermes JSON-string-aware boundary (**lift `_find_end_token_outside_string` → qwen3coder+qwen3xml**).
- `45479` — gemma4 missing-name (port index-based recovery в PN375); `45464` — **port index-based merge в qwen3xml** (сейчас ключ по `id`, continuation без id → merge ломается).
- `45252` *(MERGED, live PROD)* — prompt_embeds M-RoPE DoS: Gemma-4-31B-AWQ/26B-A4B; vendor as-is + regression-тест.
- `45383` — prompt_embeds OOB (lockstep PN35/PN371); `45417` — `max_new_tokens:null` (schema widen); `45351` — ctx-length 400 без traceback.

## 4. Волна 2 / Волна 3 / Research

**Волна 2 (приоритет, после vendor-now):** `45252` (MERGED, единственный live-PROD сегодня), `45530`, `45151` (**сделать PN351 anchor DUAL-FORM до пин-бампа**), `45434`, `45370`, `45306`(+orig_dtype), `45320`, `45265`, `45130`, `45299/45310/45479/45464`, `45383/45417/45351`, `45339`, `45343` (split-vendor generic half), `45038` (value-inverted), `45126`, `45361`, `40547` (**collision с PN377** — fold dedupe), `45466`, `45001`, `45379` (seed).

**Волна 3 (insurance/dormant/drift-watch):** `45368` (LoRA parity-checklist), `45384`/`45393` (int64 + binding-shape audit), `45415` (build-radar), `45527`, `45280/45283`, `45349`, `45363`, `45371`, `45378`, `45096`, `45357` (**GDN-generalization live A5000+5090**), `45406`, `45146/45497`, `45120` (retire router_softmax.py), `45109` (tokenizer-fingerprint), `45404` (PN377 boot-guard), `45352` (PN8b), `45375`, `45423`, `45171` (**MERGED — re-anchor PN288/P107 на vllm/parser/harmony.py**).

**Research:** `45022` (endurance_probe для 24GB A5000 / 32GB 5090), `45053` (explicit-direction API в PN95), `45480` (repo-wide lint: не конструировать CustomOp в forward()).

## 5. Реально неприменимо нигде — СТРОГО 0

Даже `live_5090=F ∧ live_a5000=F` несут arch-uniform value: `45368` (LoRA parity-checklist),
`45384` (int64-multiply audit — first-victim Blackwell 256-bit PTX), `45393` (binding-shape verifier),
`45480` (CustomOp-in-forward lint), `45417`/`45171` (model-agnostic generalization / re-anchor),
`45126`/`45434`/`45464` («as-shipped даёт ничего → must be extended/widened/ported»). Список «выбросить» пуст.

## 6. Ключевые архитектурные паттерны адаптации (переиспользуемые рецепты)

**A. SM8.9+ native FP8 fused-epilogue + Ampere unfused-fallback** — `45151/45370/45415/45001`.
```
if has_native_fp8() and is_sm_at_least(8,9) and backend in {TRITON_ATTN,FLASHINFER}:
    <native fused-quant epilogue>      # Ada/Hopper/Blackwell/5090
else:
    <standalone per_token_group_quant_fp8>   # Ampere: correct, unfused, no regression
```

**B. SM9.0+ CUTLASS grouped-MoE native + Ampere Marlin fallback** — `45530/45415/45265/45306/45375`.
```
if is_sm_at_least(9,0) and cutlass_moe: <CUTLASS + guarded moe_permute (45530 HARD PREREQ)>
elif has_native_fp8(): <native FP8 fused store>          # Ada
else: <MarlinExperts.apply>                              # Ampere PN96b (PROD-validated)
```

**C. NVFP4 Blackwell-gate + Ampere Marlin W4A16 emulation** — `45306/45375/45320/45343(A)`.
```
if is_blackwell_consumer() or is_blackwell_datacenter(): <CUTLASS/FlashInfer NVFP4>   # 5090/B200
elif has_native_fp8(): <native FP8 + NVFP4>              # Ada/Hopper
elif is_ampere_any(): <Marlin W4A16 + MarlinFP8 W8A16>  # cap-floor 80, +orig_dtype fix
```
Capability = **floor загрузки** (get_min_capability), не ceiling исполнения; route отдельно.

**D. FP8-KV: native-dequant (Hopper/Blackwell) + TQ-Triton masquerade (Ampere)** — `45434/45370/45038`.
```
if has_native_fp8() and (is_hopper() or is_blackwell()) and fa4_built: <FA4 fp8-KV>  # widen 45434 ==9
elif is_ada_lovelace(): <FlashInfer fp8-KV e5m2 / TQ-Triton>
else: <TQ-Triton fp8-KV (pr42637) + e5m2->e4m3 masquerade (g4_80)>   # Ampere — нет native FP8
```
`45038` value-inverted: suppress на Ampere, ALLOW на Blackwell/Hopper (gate по real backend-capability).

**E. Universal correctness guard — НИКОГДА не arch-gate** — `45005/44297/45527/45384/45530/45466/45393/45130/45252/45383`.
int64-widening, bound-check, early-return, type-hardening, fail-fast. Gate ТОЛЬКО на feature/model/pin-presence.
**Arch-gating correctness-фикса = landmine** (возвращает баг на исключённой арке).

**F. Tuned-table per-device (loader-ключ = arch-gate)** — `45126/45379`.
Loader-ключ `(E,N,dtype,block,device_name)`/`(cap.major,cap.minor)` УЖЕ партиционирует по арке.
Эмитить tuned config только где sweep BEAT default; freeze static + PN362 force-first для GDN-детерминизма.

**G. Topology/model-gate, НЕ SM-gate** — `45280/45283/45096/45357/45363/45371/45378/45146/45497/45406/45349`.
applies_to = (kv_transfer_config)/(has_mamba_layers)/(async)/(spec_decode); arch = `is_cuda_alike()`.
GDN-generalization (`45357/45096`) переводит dormant PD-фиксы в live single-instance защиту 35B GDN+Mamba.

**H. Parser/serving CPU-path — workload-gate** — `44297/45299/45310/45464/45479/45351/45417`.
Байт-идентичен V100→Blackwell, но MTP accept выше на быстрых GPU → tool-call correctness **ценнее на 5090/Hopper**.

---

**Якоря для drift-маркеров до пин-бампа:** PN351 dual-form (`45151`), PN288/P107 re-home в
`vllm/parser/harmony.py` (`45171/45464`), PN377 collision (`40547/45404`), PN28 reuse (`45527`),
G4_31 extend (`45038`), PN299E sibling (`45361`), G4_72/74/76+pn21 retire-after-A/B (`45343`),
**P108 СОХРАНИТЬ — не ретайрить на #45005**.

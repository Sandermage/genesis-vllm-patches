# GDN Kernel Fusion Design — путь к снижению TTFT и росту prefill TPS

**Дата**: 2026-05-15
**Контекст**: bench dev371 показал что **GDN prefill** структурно ограничивает
TTFT на conc=8 до 237 ms (target пользователя 100-120 ms) и не даёт
снизить per-token prefill cost ниже ~163 µs/tok на 35B-A3B-FP8.

---

## 1. Проблема — 6 sequential Triton kernels × 30 GDN layers

Каждый GDN layer (Mamba2-style) в prefill вызывает 6 Triton kernels подряд:

```
mixed_qkvz, _ = in_proj_qkvz(hidden_states)              # GEMM (Marlin/cuBLAS)
ba,         _ = in_proj_ba(hidden_states)                # GEMM
[split / reshape / contiguous chain]                     # — PN50 fused
torch.ops.vllm.gdn_attention_core(...) → calls FLA:
  1. chunk_local_cumsum_scalar              (Triton)
  2. chunk_scaled_dot_kkt_fwd               (Triton)
  3. solve_tril_chunk_inv                   (Triton)
  4. recompute_w_u_fwd                      (Triton)
  5. chunk_gated_delta_rule_fwd_h           (Triton, recurrence)
  6. chunk_fwd_o                            (Triton)
```

На **35B-A3B FP8 / 30 GDN layers**:
- 30 × 6 = **180 Triton kernel launches** на каждый prefill forward pass
- Каждый launch ~50-100 µs CPU overhead (driver + kernel dispatch + cmd queue)
- Итого: 9-18 ms чистого overhead **только на launch**, без compute

На SM 8.6 (A5000): occupancy на коротких kernels недостаточно для скрытия latency.
**Это и есть наш TTFT floor.**

---

## 2. Что уже сделано (low-hanging)

- **PN50 (SGLang #21019)** — fused split/reshape/cat/.contiguous (6 ops → 1
  Triton kernel) — *восстановлен сегодня после anchor-drift fix*
- **PN106** — GDN scratch pool 2/2 (eliminates per-call alloc)
- **P28** — GDN core_attn_out prealloc
- **P39a** — FLA chunk_scaled_dot_kkt persistent A pool

Все они дают **per-call savings**, но не уменьшают **число kernel launches**.

---

## 3. План фьюжна — 6 kernels → 3 (target)

### Phase 1: Fuse kernels 1 + 2 (`chunk_local_cumsum` + `chunk_scaled_dot_kkt_fwd`)
Зависимость: cumsum результат — input для KKT. Currently separate kernels из-за
разной grid topology (cumsum по chunks, KKT по chunk pairs).

**Решение**: один Triton kernel с двумя program_ids:
- pid_chunk для cumsum strip
- pid_pair для KKT (внутри одного chunk)

```python
@triton.jit
def fused_cumsum_kkt(
    g_ptr, q_ptr, k_ptr, A_ptr,
    T, H, D,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk_idx = tl.program_id(0)
    # Stage 1: scan cumsum within chunk
    g_block = tl.load(g_ptr + chunk_idx * BLOCK_T * H + ...)
    g_cumsum = tl.cumsum(g_block, axis=0)
    # Stage 2: KKT using cumsum'd g
    # ... compute A = (q · k^T) * exp(g_cumsum_diff)
    tl.store(A_ptr + chunk_idx * BLOCK_T * BLOCK_T * H + ..., A)
```

**Expected**: 2 launches → 1, saves ~50-100 µs × 30 layers = **1.5-3 ms TTFT**.

### Phase 2: Fuse kernels 3 + 4 (`solve_tril_inv` + `recompute_w_u_fwd`)
Tril solve — strictly sequential по rows. `recompute_w_u` использует solve
output. Fuse через cooperative warps: одна team warps делает back-substitution,
вторая team стартует w/u computation как только diagonal row finished.

**Expected**: ещё 1-2 ms saving × 30 layers = **30-60 ms TTFT**.

### Phase 3: Fuse kernels 5 + 6 (`chunk_gated_delta_rule_fwd_h` + `chunk_fwd_o`)
Это **горячая точка** — gated delta rule recurrence (kernel 5) занимает 60-70%
GDN time. chunk_fwd_o использует h state from kernel 5. Currently SEPARATE
because state is written to global memory between iterations.

**Решение**: keep state in shared memory / registers across the boundary:

```python
@triton.jit
def fused_recurrence_o(
    q, k, v, g, beta,
    o_ptr,
    T, H, D_k, D_v,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,  # chunk size = 64
):
    # State h is [D_v, D_k] — keep in registers if D_v*D_k*4 < 96 KB
    # SM 8.6: 100 KB shared mem / SM
    # For D_v=128, D_k=128: 64 KB → fits
    h = tl.zeros((BLOCK_D_v, BLOCK_D_k), dtype=tl.float32)
    for c in range(num_chunks):
        # ... gated delta rule update of h in registers
        # ... immediately compute o[c] = q[c] @ h  (no global write of h)
        tl.store(o_ptr + c * BLOCK_T * D_v + ..., o_chunk)
```

**Expected**: −1 launch + lower memory traffic = **3-5 ms TTFT × 30 layers = 90-150 ms**.

### Phase 4: Numerical validation
Each fused kernel must match unfused output bit-for-bit (FP16/BF16 tolerance
1e-3 abs, 1e-4 rel). Validation suite:
1. Random inputs, varying chunk count {1, 4, 8, 16}
2. Edge cases: T not multiple of BLOCK_C
3. Real model: compare logits last_hidden_states with PN50 / PN106 / fused
   stack vs upstream stack — KL divergence < 1e-4

---

## 4. Risk + mitigation matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| FP precision drift on fused recurrence | high | medium | accumulate state in FP32, downcast on store |
| Shared-mem overflow on SM 8.6 (100 KB) | medium | high | conditional BLOCK_D split, fall back to unfused for large head_dim |
| Triton autotune fails to find good config | medium | medium | hand-tune for our shapes (BS=1..8, T=1..32K) |
| Inductor compile interaction | low | high | mark fused kernel as opaque op via `torch.library.custom_op` |
| GDN layer count varies between models | low | low | configure via `model_config.num_gdn_layers` lookup |

---

## 5. Effort estimate

- Phase 1 (fuse 1+2): **2-3 дня**, low risk
- Phase 2 (fuse 3+4): **3-5 дней**, medium risk (tril solve fusion tricky)
- Phase 3 (fuse 5+6): **5-7 дней**, high risk (recurrence in registers)
- Phase 4 (validation): **2-3 дня**

**Total: 12-18 working days** (3-4 weeks calendar)

---

## 6. Expected gains (если все 3 phases успешны)

- **TTFT @ conc=8**: 237 ms → estimated 150-180 ms (-25 to -36%)
- **TTFT @ conc=1**: 59 ms → estimated 35-45 ms (-25%)
- **Aggregate TPS @ conc=8**: 689 → estimated 800-900 (+15-30%)
- **Per-req TPS @ conc=1**: 215 → estimated 240-260 (+12-21%) — **может достичь user target 240 TPS**

---

## 7. Альтернатива — отложить fusion, использовать `torch.compile` mode='reduce-overhead'

На dev371 Inductor mode='reduce-overhead' автоматически фьюзит мелкие kernels.
Можно попробовать `--compilation-config.optimization_level O3` и сравнить.
Если Inductor сам справится — кастомный fusion не нужен.

Эксперимент (быстрый): запустить с `optimization_level=3`, посмотреть на boot
log "Inductor cache miss" частоту и финальный TTFT/TPS.

---

## 8. Открытые вопросы (требуют research)

1. Lock-free per-chunk state update — возможен ли на SM 8.6 без deadlock?
2. Tensor cores на FP32 accumulator: SM 8.6 имеет ограниченную TF32 throughput
   (нет в FP8 path); влияет на recurrence kernel
3. Memory layout: A_ptr / o_ptr — какой stride pattern оптимален для L2?
   Нужен micro-benchmark
4. FlashInfer GDN kernel — landed для Hopper SM 9.0+. Можно ли портировать
   FlashInfer's design (без Hopper-specific instructions) на Ampere?

---

## 9. Decision: do-not-implement until pin v0.22.x

Текущий vllm pin dev371 (0.20.2rc1) ещё в active development; upstream может
landить equivalent fusion в ближайшие месяцы. Trade-off:
- **Implement сейчас**: 3-4 недели работы, риск что upstream merge'нет аналог
- **Подождать v0.22**: возможен upstream fusion (vllm-project следит за FLA
  prog'rams). Pin bump в v0.22.x = re-evaluation.

**Решение**: monitor upstream vllm-project/vllm GDN PRs до 2026-06-01. Если
nothing merged — implement Phase 1 (lowest risk, biggest win/effort).

---

## Источники research

- FLA repo: github.com/fla-org/flash-linear-attention
- Mamba2 paper: arXiv 2405.21060 (Dao 2024)
- SGLang PR #21019 (PN50 backport — split/reshape fusion)
- Triton tutorial 09 "persistent matmul" — pattern для in-register state
- PyTorch blog 2026-05 "FULL_AND_PIECEWISE on hybrid_gdn_moe"
- vLLM #41446 (chunk_o scale-fold pattern — PN29 backport, opt-in)

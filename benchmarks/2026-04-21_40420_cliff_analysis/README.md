# TurboQuant long-context cliff analysis — 2026-04-21

Independent hardware reproduction of the `_continuation_prefill` OOM reported
in [vllm-project/vllm#40420](https://github.com/vllm-project/vllm/issues/40420),
plus a 4-way config matrix that isolates which configuration lever actually
moves the cliff on 24 GiB-per-GPU setups.

## Setup

- **Hardware:** 2× RTX A5000 (24 GiB each = 48 GiB total), Ampere SM 8.6,
  TP=2, P2P disabled
- **Model:** `Qwen3.6-35B-A3B-FP8` (hybrid linear-attn + attention MoE)
- **KV cache dtype:** `turboquant_k8v4` (8-bit keys, 4-bit values — different
  from jhsmith409's `turboquant_4bit_nc` in JartX/vllm#11)
- **vLLM:** nightly `befc9151fea3` + 17-patch runtime overlay (see
  [../../patch_genesis_unified.py](../../patch_genesis_unified.py))
- **max_model_len:** 262144 (256k)
- **Harness:** genesis_bench_v3.py — 3 runs per sweep point, reports TTFT
  + tokens/sec + pass/fail + server metrics

## The 6 experiments (phase 1: 1–4 config-only; phase 2: 5–6 with Patch 22 code fix)

| # | Folder | `gpu_util` | `max_seqs` | Patch 20 (JartX#11)? | Patch 22 (shared dequant pre-alloc)? | Result |
|---|---|---|---|---|---|---|
| 1 | [`raw/01_baseline_sweep_128-256k/`](raw/01_baseline_sweep_128-256k/) | 0.905 | 2 | ❌ no | ❌ no | Cliff @ 234k — NCCL cascade |
| 2 | [`raw/02_patch20_only_234cliff/`](raw/02_patch20_only_234cliff/) | 0.905 | 2 | ✅ yes | ❌ no | Cliff @ 234k — direct CUDA OOM (88 MiB) |
| 3 | [`raw/03_patch20_util085_seqs2/`](raw/03_patch20_util085_seqs2/) | **0.85** | 2 | ✅ yes | ❌ no | All 228→260k PASS (KV 589k, -38%) |
| 4 | [`raw/04_patch20_util0905_seqs1/`](raw/04_patch20_util0905_seqs1/) | 0.905 | **1** | ✅ yes | ❌ no | Cliff @ 228k — *worse* than baseline |
| 5 | [`raw/05_patch22_util085/`](raw/05_patch22_util085/) | 0.85 | 2 | ✅ yes | ✅ yes | All 228→260k PASS (KV 517k) |
| **6** 🏆 | [`raw/06_patch22_util0905_TRUTH/`](raw/06_patch22_util0905_TRUTH/) | **0.905** | 2 | ✅ yes | ✅ yes | **All 228→260k PASS (KV 877k)** ✅ breakthrough |

**Interpretation (phase 1, exp 1–4 — config-only workaround):**

- **Patch 20 works** (exp #2): the crash signature changes from a
  10-minute NCCL watchdog cascade (silent worker death first, timeout in
  a downstream `_gather_logits` all-gather later) to a direct
  `CUDA OOM: tried 88 MiB, 42 MiB free`. That's the FP32-rotation spike
  being removed — but what's left is still hitting the VRAM wall on
  24 GiB cards.
- **`gpu_memory_utilization` is the lever** (exp #3): dropping from
  0.905 → 0.85 frees ~1.5 GiB from the KV-cache pool (946k → 589k tokens)
  for PyTorch's transient activation/prefill workspace. Cliff vanishes
  through 260k (full `max_model_len`).
- **`max_num_seqs` is not the lever** (exp #4): reducing from 2 → 1
  actually *grows* KV by 3% (vLLM repacks into a single slot) and
  *shrinks* transient headroom → crash at 228k, earlier than baseline.

**Interpretation (phase 2, exp 5–6 — Patch 22 code fix):**

- **Root cause of residual 88 MiB spike identified:** lazy
  `torch.empty(buf_shape)` inside `_continuation_prefill` for the K/V
  dequant buffers. Not visible to vLLM's memory profiler during warmup
  (which runs at `max_num_batched_tokens=4096` = small buffers), so KV
  cache is sized as if the buffer stays small. First real long-context
  request grows the buffer; allocator finds no room → `tried 88 MiB,
  42 MiB free` crash.
- **Patch 22 fix:** pre-allocate the buffer in `_ensure_on_device()`
  (runs during warmup, visible to profiler) and SHARE it across all
  attention layers via a class-level cache keyed by
  `(num_kv_heads, head_size, max_alloc_len, device)`. Single
  ~1 GiB allocation instead of N-layers × 1 GiB (naive per-layer
  approach failed: 10 layers × 1 GiB = 10 GiB exceeded any gpu_util
  budget).
- **Exp #5** confirms Patch 22 doesn't regress the working config:
  same 17-point 228→260k all PASS at `gpu_util=0.85`, KV 517k
  (-12% from phase 1 baseline due to visible buffer — expected).
- **Exp #6 is the breakthrough:** with Patch 22 we can restore
  `gpu_util=0.905` and still get all 17 points 228→260k PASS.
  KV recovers from 589k (phase 1 workaround) to **877k tokens** —
  a 49% KV capacity recovery while maintaining stable 256k context.
  Speed identical within 1% noise across all "works" configs.
- **Net bottom line:** the real fix is the pre-alloc, not the gpu_util
  knob. `gpu_util=0.905` + Patch 22 is strictly better than
  `gpu_util=0.85` alone.

See [RESULTS_TABLE.md](RESULTS_TABLE.md) for the full speed/TTFT/memory
numbers across all experiments.

## Baseline sweep (experiment #1 details)

The 9-point `max_model_len` sweep (128k → 256k, 9 values) in
[`raw/01_baseline_sweep_128-256k/`](raw/01_baseline_sweep_128-256k/)
isolates a different question: does raising `max_model_len` cost
short-context speed? Answer: **no**. Speed is flat within ±1% at 64
tokens across all 9 values. The tax of unlocking longer
`max_model_len` is paid only at long-context, in the form of a
different transient-memory ceiling (the cliff). See
[aggregated/baseline_aggregate.md](aggregated/baseline_aggregate.md).

## Production config that works

After Patch 22 (exp #6 breakthrough), we run:

```yaml
max_model_len: 262144      # 256k
gpu_util: 0.905            # baseline (not the 0.85 workaround)
max_num_seqs: 2
kv_cache_dtype: turboquant_k8v4
# + 20 Genesis runtime patches (includes Patches 20, 22, 23, 25)
```

[`configs/docker-compose-winning-vllm-only.yml`](configs/docker-compose-winning-vllm-only.yml)
was exp #3's config (config-only workaround, `gpu_util=0.85`).
The current production is essentially the same but with `gpu_util=0.905`
once Patch 22 is applied — strictly more KV (877k vs 517k vs phase 1
589k) at no stability cost.

## How to reproduce

Requirements:
- 2× Ampere+ GPU with ≥ 24 GiB each
- Docker with nvidia-runtime
- Model downloaded to `/models/Qwen3.6-35B-A3B-FP8`

Steps:
1. Clone this repo (`Sandermage/genesis-vllm-patches`)
2. Copy `docker-compose-winning.yml` + fix paths/ports for your env
3. Bind-mount `patch_genesis_unified.py` into `/patches/` in the container
   (see existing compose snippet)
4. `docker compose up -d vllm-server`
5. Inside `vllm_engine/` (or wherever your bench lives), run
   `python3 genesis_bench_v3.py --label repro --sweep-from 228 --sweep-to 260 --sweep-step 2 --sweep-runs 3 --skip-speed --skip-context --skip-stability --skip-stress --skip-long`
6. Expect all 17 points PASS 3/3 with speed 38.5 → 34.9 t/s

## Notes on redactions

All raw artifacts have had the following redacted before commit:
- Internal server IP → `SERVER_IP`
- Local API key → `API_KEY_REDACTED`
- No model weights or prompts contain PII (bench prompts are synthetic
  repetitive Lorem-Ipsum-style text designed to reach target token counts)

## Links
- JartX/vllm#11 — jhsmith409's `_continuation_prefill` FP16-rotation fix
- vllm-project/vllm#40420 — original cliff report
- vllm-project/vllm#39931 — parent TurboQuant hybrid-support PR
- `../../patch_genesis_unified.py` — our 18-patch runtime overlay
  (Patch 20 in the file is the port of JartX#11)

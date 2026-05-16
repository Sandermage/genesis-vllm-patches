# Genesis vLLM Patches — Benchmarks

Canonical PROD bench numbers for the Genesis reference rig.
Reproducible from any host that runs the same vLLM pin against
the listed Genesis preset. See [`BENCHMARK_GUIDE.md`](BENCHMARK_GUIDE.md)
for the full methodology and the [`HARDWARE.md`](HARDWARE.md) envelope.

> **Current canonical stack (Wave 10, 2026-05-16)**
>
> - Genesis `v11.0.0+wave10` — 169 PATCH_REGISTRY entries
>   (154 full + 11 marker-only + 2 retired + 1 partial + 1 placeholder).
> - vLLM `0.20.2rc1.dev371+gbf610c2f5`.
> - Reference rig: **2× RTX A5000 24 GB** (Ampere SM 8.6),
>   driver 580.142, CUDA 13.0.2.
> - Spec-decode: MTP K=3 (probabilistic draft rejection, vllm#40269).
> - Attention: TurboQuant k8v4 KV cache + FlashAttention 2, TP=2.

## Latest PROD numbers (Wave 10, 2026-05-15)

| Model | wall_TPS (sustained) | decode_TPOT | CV% | Tool-call | Method |
| --- | ---: | ---: | ---: | :---: | --- |
| **Qwen3.6-27B-int4-AutoRound** | **132.93** | 7.27 ms | 3.5% | 8/8 | `genesis_bench_suite.py --quick --ctx 8k` (5×5×1024) |
| **Qwen3.6-35B-A3B-FP8** (decode-only, max_num_seqs=2) | **216.02** | 4.38 ms | 5.4% | 7/7 | same harness |
| **Qwen3.6-35B-A3B-FP8** (multi-conc, max_num_seqs=8) | **~675** agg | — | within CV | — | `genesis_bench_suite.py --multi-conc` |

### Wave 10 Δ vs Wave 8 baseline (27B PROD, same harness)

| Metric | Wave 8 (2026-05-11) | Wave 10 (2026-05-15) | Δ |
| --- | ---: | ---: | ---: |
| wall_TPS | 130.76 | **132.93** | **+1.66%** |
| decode_TPOT_ms | 7.31 | 7.27 | -0.5% (faster) |
| CV% | 5.29 | 3.5 | tightened |

**Wave 10 components on top of Wave 9**: PN116 / PN118 / PN119
TurboQuant backports, PN125–PN130 warmup-orchestrator family,
PN132 / PN133 correctness backports, PN204 GDN dual-stream
consolidation (off in single-conc; on in `prod-35b-multiconc`),
PN96b Marlin MoE persistent workspace (renamed after the silent
dict-key collision with kv_cache/PN96 was fixed), PN95 tier-aware
cache wiring closure.

The 27B improvement vs Wave 8 is small but outside CV. Most of
it comes from PN122 (the renamed `SPRINT26_CG_DISPATCH_TRACE`)
no longer crashing on import: each failed `@register_patch` hook
added ~30–50 ms boot overhead and one log/exception event that
introduced jitter on the worker decode path.

## What is currently on for `prod-35b`

Per Genesis structured boot summary printed once at boot end:

```text
══════════════════════════════════════════════════════════════════════
Genesis vLLM Patcher — boot summary
══════════════════════════════════════════════════════════════════════
  Genesis:  v11.0.0+wave10
  vLLM:     0.20.2rc1.dev371+gbf610c2f5
  GPU:      2× NVIDIA RTX A5000 (sm_86)
──────────────────────────────────────────────────────────────────────
  Patches:  169 total → ~80 APPLY | ~89 SKIP
  By family (APPLY only):
    • attention.gdn          ~5
    • attention.turboquant   ~12 (incl. PN116/118/119)
    • compile_safety         ~4
    • kernels                ~3
    • kv_cache               ~6 (incl. PN95)
    • moe                    ~3 (incl. PN96b)
    • observability          ~4 (incl. PN122)
    • reasoning              ~5
    • scheduler              ~3
    • serving                ~3
    • spec_decode            ~9 (incl. PN90 probabilistic)
    • streaming              ~3
    • tool_parsing           ~6
    • worker                 ~10 (incl. PN35, warmup PN125–130)
══════════════════════════════════════════════════════════════════════
```

The complete machine-readable per-patch state lands in the proof
artefacts under `evidence/patch_proof/<id>__*.json` after a
`sndr patches release-check` run.

## Reproduction recipe

The canonical bench harness is `tools/genesis_bench_suite.py` (shim
under `tools/`, canonical source under `vllm/sndr_core/tools/`). It
reads a `ModelConfig` preset and runs five stages: short-gen TTFT,
sustained long-gen TPS, tool-call clean, multi-turn stability,
long-context probe (skippable).

```bash
# 1. Install + boot
sndr install                # or `bash install.sh --workload tool_agent -y`
sndr launch a5000-2x-35b-prod    # V1 key, or use V2 alias `prod-35b`

# 2. Wait for the structured boot summary in docker logs

# 3. Run the canonical bench
python3 tools/genesis_bench_suite.py \
    --quick --ctx 8k \
    --model qwen3.6-35b-a3b \
    --out ~/.sndr/bench-results/35b_wave10.json

# 4. Verify against the preset's reference_metrics
sndr model-config verify prod-35b
```

Multi-conc runs flip `max_num_seqs=8` and use the
`prod-35b-multiconc` V2 alias (35b-multiconc.yaml profile);
expect aggregate TPS ~675 at the cost of higher TTFT.

## Historical reference

Older points are kept for regression-detection. Wave 8 (dev93)
numbers remained the operator-facing baseline until Wave 10 confirmed
the small uplift above; Wave 7 / v7.72 (dev9) is pre-v11-rename and
is not directly comparable because the patch registry was much
smaller (134 entries vs 169 today).

### Wave 7 / v7.72 dev9 snapshot (2026-05-05, pre-v11 rename)

| Model | Sustained TPS | CV% | Cold-warm latency | Tool-call clean | Multi-turn 10/10 | VRAM steady-state |
| --- | --- | --- | --- | --- | --- | --- |
| **Qwen3.6-35B-A3B-FP8** (MoE) | 192.9 tok/s | 4.19% | 2.34s | 10/10 | 10/10 survived (avg 1.1s) | 22687 + 21998 = 44685 MiB |
| **Qwen3.6-27B-int4-AutoRound** | 95.6 tok/s | 4.04% | 4.76s | 10/10 | 10/10 survived (avg 2.3s) | 22753 + 22064 = 44817 MiB |

### Wave 8 dev93 snapshot (2026-05-11)

| Model | wall_TPS | decode_TPOT | CV% | Tool-call |
| --- | ---: | ---: | ---: | :---: |
| Qwen3.6-27B-int4-AutoRound | 132.28 | 7.31 ms | 5.29% | 8/8 |
| Qwen3.6-35B-A3B-FP8 (Sprint 1) | 241.35 | 3.85 ms | 3.02% | 7/7 |

The 35B Sprint-1 number (241 TPS) was a single-prompt cherry-pick
captured before the methodology shift to 5×5×1024 sustained — the
~216 TPS sustained figure in the Wave 10 table above is the
correct apples-to-apples comparison.

## Cross-rig validators (call for replication)

Genesis numbers above are 2× RTX A5000 single-rig. Cross-rig
validation requested from operators on:

- **noonghunna** (1× RTX 3090, 4× RTX 3090 club-3090) — long-time
  Cliff 2 + tool-call collaborator.
- **apnar** (1× RTX 5090, sm_120 consumer Blackwell) — first
  sm_120 production rig (club-3090#51 thread).
- **tfriedel** (4× RTX 3090) — vendors Genesis as submodule.
- **Quentin Machu** — P64 sub-patch E author + bug-class triage.
- **MidasMining**, **JartX**, **jhsmith409**, **webcodes-cz** —
  hardware variety (5090, H20, R6000 Blackwell, 8× A4000).

If you run Genesis on hardware not listed, drop a bench JSON into
`tests/integration/baselines/` (PR welcome).

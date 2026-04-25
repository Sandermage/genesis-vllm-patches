# Genesis vLLM Patches — v7.13

**Runtime patches for [vLLM](https://github.com/vllm-project/vllm) — long-context Qwen3-class inference on NVIDIA Ampere, with TurboQuant k8v4 KV-cache and 256k context.**

> 30+ active patches (P58 + P59 + P60 + P60b + P61 added in v7.13). Zero vLLM source modifications beyond container startup.
> Defense-in-depth interface guards (P49). ASGI response-cache middleware (P50).
> Runtime architecture-dispatch detection (P51 / P52 / P53 — v7.9).
> Cross-model validated on FP8 / AWQ 4-bit / FP16-KV configurations.
>
> ⚠️ **Test matrix is 2× RTX A5000 (Ampere SM 8.6) + NVIDIA drivers 570+.**
> Patches are written defensively for AMD ROCm / Intel XPU / Hopper / Blackwell / FP8-native paths — they graceful-skip on platform mismatch rather than crash.
> Bug reports from other hardware are very welcome.
> Optional support / hardware sponsorship — see [SPONSORS.md](SPONSORS.md).

---


## What's new in v7.13 (2026-04-25)

**Spec-decode + structured output bug class — root cause located + multi-layer fix backported.**

After deep investigation of the noonghunna #40831 + #40807 thread + parallel community reports (#39273 / #34650 / #36138), located the actual root cause class for tool-call output corruption (`<<`, `parameter=parameter`, `<argname>` patterns) on hybrid GDN models with ngram speculative decoding.

| New patch | Source | Author | Files | Effect |
|---|---|---|---|---|
| **P58** | vllm#40768 backport | @z1ying | scheduler.py + async_scheduler.py + request.py | Async-scheduler `-1` placeholder gating (opt-in research artifact; empirically NOT our bug for ngram method since vLLM auto-disables async_scheduling) |
| **P59** | vllm#39055 backport | @ZenoAFfectionate | qwen3_reasoning_parser.py | Promotes `<tool_call>` XML out of `<think>` reasoning into content (opt-in research) |
| **P60** | vllm#40738 backport (Phase 1) | @tdoublep, @bhaktatejas922 (#39273) | gdn_attn.py + gdn_linear_attn.py + gpu_model_runner.py | SSM state pre-copy from accepted block. **+23-40% clean tool-call rate** (43-60% vs 20% baseline) |
| **P60b** | vllm#40738 backport (Phase 2) | @tdoublep | causal_conv1d.py (Triton) + gdn_linear_attn.py | Conv state Triton kernel offset. **Combined with P60: 70% clean** (3.5× baseline) |
| **P61** | vllm#40783 backport (slice) | @ExtReMLapin | qwen3_reasoning_parser.py | Multi-tool first-occurrence fix — preserves all tool calls in agentic flows |
| **P61b** | vllm#40783 backport (streaming slice) | @ExtReMLapin | qwen3_reasoning_parser.py | Defensive overlap guard for partial `<tool_call>` tags during streaming |
| **P62** | vllm#36138 backport | @sfbemerk, @cicirori (#34650) | structured_output/__init__.py + scheduler.py | Reasoning-aware grammar acceptance — fixes grammar bypass when `</think>` arrives in spec-decode batch |

### Empirical results (2026-04-25)

| Config | Clean rate | Notes |
|---|---|---|
| baseline (no Genesis) | ~20% | bug pattern present |
| Genesis v7.11 (P56 only) | ~20% | P56 was workaround attempt — disproven |
| + P58 | ~20% | wrong code path (ngram → async_scheduling auto-disabled) |
| + P59 | 40% | helps when `<tool_call>` nested inside `<think>` |
| + cudagraph_mode=NONE | 40% | partial, tradeoff: ~25% TPS loss |
| **+ P60** | **43-60%** | SSM pre-copy main fix |
| **+ P60 + P60b** | **70%** (test) / **53%** (prod sample) | Triton kernel offset complete fix |
| + P60 + P60b + P61 | 50% (prod) | P61 = multi-tool fix, marginal on single-tool reproducer |
| **+ P60 + P60b + P61 + P61b + P62** | **53-56%** (prod) | Full v7.13 — best reachable with current patches |
| no spec-decode | 100% | reference |

### v7.13 BREAKTHROUGH (2026-04-25 evening) — config-only fix achieves 100% clean

After deep token-level investigation, **the structural ceiling formula was identified**:

```
clean_rate ≈ (per_token_accept_rate)^num_speculative_tokens
0.8^3 ≈ 51%   ← matches our 53% empirical measurement with default ngram
0.8^1 = 80%   ← matches 65% (n=20, statistical variance)
```

The mechanism: ngram with default `prompt_lookup_min=2` finds spurious 2-token suffix matches in the system prompt's tool definitions, drafting wrong tokens that get accepted by the rejection sampler.

**Config-level fix (no code changes):**

```bash
--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":10,"prompt_lookup_min":8}'
```

Setting `prompt_lookup_min=8` requires ngram to find an 8-token suffix match — this almost never matches spurious system-prompt template fragments, only true natural repetitions. Result: ngram drafts almost nothing on tool-call requests, close to no-spec correctness with whatever speedup is achievable on natural repetitions.

**Empirical results with strict ngram config:**

| Test | Clean rate | n |
|---|---|---|
| Single-query Tokyo | **100%** | 30/30 |
| Multi-query diverse | **96%** | 24/25 |

This is the working production configuration as of v7.13.


### Genesis Dispatcher v2

New `vllm/_genesis/dispatcher.py` provides:

- `should_apply(patch_id) -> (bool, reason)` — unified gate combining env-flag + `config_detect.recommend()` + `model_detect`
- `dump_apply_matrix() / log_apply_matrix()` — single condensed startup summary instead of scattered INFO lines
- CLI: `python3 -m vllm._genesis.dispatcher` — diagnostic table

Built on the existing `model_detect.py` (P52/P53 dispatch) + `config_detect.py` (v7.12 runtime probes) layers. Adds P58/P59/P60/P60b/P61 to the recommendation registry.

### Pin bump

| Component | v7.9 | v7.13 |
|---|---|---|
| Test base | `dev134+gfe9c3d6c5` (2026-04-23) | `dev205+g07351e088` (2026-04-25) |
| Prod | `dev8+g4b7f5ea1a` (2026-04-19, legacy v5.12 monolith) | `dev205+g07351e088` (2026-04-25, modular `_genesis/` plugin) |
| Patcher | v5.12 monolith | Modular plugin + Dispatcher v2 |

## What's new in v7.9 (2026-04-24)

**Runtime architecture-dispatch detection — three defense-in-depth guards** that answer "did this patch even need to fire?" before doing any work.

| Patch | Role | Gate |
|---|---|---|
| **P51 — TQ-active runtime detection** | `ensure_turboquant_buffers` | skip preallocs if `impl.kv_cache_dtype != turboquant_*` |
| **P52 — MoE-active detection** | P24 / P31 / P37 apply() | skip on dense models (no `num_experts` / no `*MoE*` arch) |
| **P53 — Hybrid-active detection** | P28 / P34 / P39a / P46 apply() | skip on pure-attention models (no Mamba / GDN / linear-attn) |

The central `vllm/_genesis/model_detect.py` probes `hf_config` once per process and returns a cached profile `(moe, hybrid, turboquant)`. Each patch's `apply()` consults it and either engages or logs a clean single-line skip. Conservative fallback: unknown or unavailable config → True for all flags (patch still attempts; its own call-site guards decide).

**Cross-model validation** (Config 1 FP8 prod / Config 2 AWQ 4-bit / Config 3 FP16-KV):
- 28 applied / 4 skipped / 0 failed (identical dispatch across all three)
- 3× 256k stable on FP8 + AWQ
- AWQ frees ~9 GiB/rank → 2.5× more KV capacity (KV tokens 1.099M → 2.787M)
- Speed: AWQ 1–4% slower than FP8 (4-bit dequant cost on SM 8.6)

See [`docs/REPORT_v7_1_INTEGRATION_20260424_RU.md`](docs/REPORT_v7_1_INTEGRATION_20260424_RU.md) §13 (v7.8) + §14 (v7.8.5) + §15 (v7.9) for full figures.

---

## Compatibility matrix

### vLLM version

| Component | Pin | Notes |
|---|---|---|
| Integration baseline | `v0.19.2rc1.dev134+gfe9c3d6c5` | Patches authored against this SHA |
| Upstream branch | `main` | Rebased periodically; all patches graceful-skip on anchor drift |
| Minimum supported | `v0.19.2rc1.dev8+` (P8a) | Older builds: anchors may not match |
| Known-broken | — | No incompatibilities at time of v7.9 release |

### Hardware

| GPU class | SM | FP8 | TurboQuant | Status |
|---|---|---|---|---|
| RTX A5000 / A6000 / 4090 | 8.6 | ✅ via Marlin | ✅ | **tested** (2× A5000 is the prod target) |
| RTX 3090 / 3090 Ti | 8.6 | ✅ via Marlin | ✅ | should work; untested — reports welcome |
| A100 / H100 | 8.0 / 9.0 | ✅ native | ✅ | graceful — patches platform-guard |
| Blackwell (B200) | 10.0+ | ✅ native | ✅ (FA4 path) | graceful — patches platform-guard |
| AMD ROCm CDNA3 | — | — | — | graceful-skip |
| Intel XPU | — | — | — | graceful-skip |
| CPU-only | — | — | — | graceful-skip |

### Models

| Model | Architecture | Status |
|---|---|---|
| Qwen3.6-35B-A3B-FP8 | Qwen3-Next MoE + hybrid linear-attn | **prod** — full 256k |
| Qwen3.6-35B-A3B-AWQ | Qwen3-Next MoE + hybrid linear-attn + 4-bit weights | **validated** — 3× 256k, 2.5× KV capacity vs FP8 |
| Qwen3-32B dense (planned) | Qwen3 dense attention | graceful-skip cross-model test (v7.9) |
| Gemma 4 26B MoE (planned) | Gemma 4 MoE | cross-arch test (v7.9 follow-up) |

---

## Installation / deployment

### Option 1 — docker compose (recommended)

```yaml
# docker-compose.yml (minimal example — see docker-compose.example.yml)
services:
  vllm-server:
    image: vllm/vllm-openai:nightly   # or pinned commit image
    volumes:
      - ./vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro
      - ./docker/genesis-entrypoint.sh:/opt/genesis-entrypoint.sh:ro
    entrypoint: ["/opt/genesis-entrypoint.sh"]
    command: [
      "--model", "Qwen/Qwen3-Next-35B-A3B-FP8",
      "--kv-cache-dtype", "turboquant_k8v4",
      "--max-model-len", "262144",
      "--tensor-parallel-size", "2",
      # ...
    ]
    environment:
      # Default 28-patch set (opt-in features off):
      - GENESIS_ENABLE=1
      # Opt-in features — set to 1 to enable:
      - GENESIS_ENABLE_P5B=0      # Page-size pad-smaller allocator
      - GENESIS_ENABLE_P7B=0      # GDN dual-stream custom-op
      - GENESIS_ENABLE_P37=0      # MoE intermediate-cache text-patch
      - GENESIS_ENABLE_P40=0      # TQ grouped-decode (watch upstream #40792)
      - GENESIS_ENABLE_P41=0      # ResponseCacheMiddleware (needs P50 wired)
      # ResponseCache backend (P50):
      - GENESIS_RESPONSE_CACHE=redis://cache:6379/0
```

The entrypoint runs `python -m vllm._genesis.patches.apply_all` before `vllm serve`. All patches are idempotent per container filesystem layer.

### Option 2 — manual

```bash
# Install vLLM at the integration baseline
pip install vllm==0.19.2rc1.dev134

# Drop the Genesis package into the vLLM install
cp -r vllm/_genesis /path/to/site-packages/vllm/

# Apply patches once
python -m vllm._genesis.patches.apply_all

# Start vLLM normally
vllm serve ...
```

### Verify apply-time dispatch

On a healthy boot you should see lines like:

```
[INFO genesis.apply_all] Genesis v7.9 — 28 applied, 4 skipped, 0 failed
[INFO genesis.model_detect] profile resolved: model_type=qwen3_next_moe moe=True hybrid=True turboquant=True (kv=turboquant_k8v4)
[INFO genesis.wiring.p28_gdn_core_attn] forward_cuda patched + __init__ wrapped
[INFO genesis.prealloc] [P51 TQ-active] ...   (absent on turboquant_k8v4 — P51 only fires when non-TQ)
```

On a dense-model boot:

```
[INFO genesis.model_detect] profile resolved: model_type=qwen3 moe=False hybrid=False turboquant=False (kv=fp16)
[INFO genesis.model_detect] [Genesis v7.9 dispatch] P24 MoE num_warps/num_stages overlay skipped — dense model (no fused_moe dispatch)
[INFO genesis.model_detect] [Genesis v7.9 dispatch] P28 GDN core-attn forward rewire skipped — pure-attention model (no GDN)
# ...etc
```

---

## Patch roster (v7.9)

### 28 patches applied by default

| # | Area | What it does |
|---|---|---|
| P3 | TurboQuant | bf16 cast for Ampere non-FP8 fallback |
| P4 | TurboQuant | Hybrid reporting guard |
| P5 | Block table | Page size reporting |
| P6 | TurboQuant | Block-size align |
| P8 | KV cache | Hybrid reporting |
| P12 | Chat | Tool-call reasoning extraction |
| P14 | Scheduler | Block table zero-fill |
| P15 | Chat | Qwen3 None/null reasoning guard |
| P17 | MoE | Marlin bsm=8 (A5000 SM 8.6) |
| P18 | MoE | Marlin num_warps overlay |
| P22 | TurboQuant | Shared dequant pre-allocation |
| P23 | TurboQuant | cu_seqlens scratch reuse |
| P26 | TurboQuant | Prefill output buffer reuse |
| P27 | Chat | Reasoning-before-think ordering |
| P28 | GDN | Core-attn forward-cuda rewire |
| P31 | MoE | Grouped-topk fp32 softmax upcast |
| P32 | TurboQuant | Second-hop cu_seqlens |
| P33 | TurboQuant | Synthetic seq_lens mirror |
| P34 | Hybrid | Mamba zero-collapse deadlock guard |
| P36 | TurboQuant | Shared decode buffer |
| P38 | TurboQuant | Continuation 4-D K/V memory |
| P39a | GDN | FLA `chunk_scaled_dot_kkt` persistent A pool |
| P42 | Chat | — (reserved) |
| P44 | TurboQuant | Mixed-attn-out capture-safe reuse |
| P46 | GDN | fused_gdn_gating buffer pool |
| **P49** | **Infra** | **Interface-contract validation (v7.8)** |
| **P50** | **Infra** | **ASGI ResponseCacheMiddleware for cliproxyapi (v7.8)** |
| **P51** | **Dispatch** | **TQ-active runtime detection (v7.9)** |
| **P52** | **Dispatch** | **MoE-active dispatch gate (v7.9)** |
| **P53** | **Dispatch** | **Hybrid-active dispatch gate (v7.9)** |

### 4 opt-in patches

| # | Env var | What it enables |
|---|---|---|
| P5b | `GENESIS_ENABLE_P5B=1` | page-size pad-smaller allocator |
| P7b | `GENESIS_ENABLE_P7B=1` | GDN dual-stream custom-op (+8% decode on hybrid) |
| P40 | `GENESIS_ENABLE_P40=1` | TurboQuant grouped-decode (watch upstream #40792) |
| P41 | `GENESIS_ENABLE_P41=1` | ResponseCache + P50 middleware (needs a backend) |

### Retired / shelved

| Status | Patch | Reason |
|---|---|---|
| ❌ retired | P43 → P28 | covered by P28 rewire |
| ❌ retired | P47 → P38 | covered by P38 4-D memory patch |
| ❌ dropped | swap-space | upstream removed the knob |
| ❌ dropped | LMCache | no measurable gain on our workload |
| 💤 shelved | P7 (eager-only) | Torch.compile conflict, deferred |
| 💤 shelved | P41b semantic cache | hallucination risk — **permanently excluded** |
| 💤 shelved | P45 qkv concat | byte-copy stride risk |
| 💤 shelved | P48 conv-view | corruption risk |
| 👁 monitor | P40 retire | upstream PR [#40792](https://github.com/vllm-project/vllm/pull/40792) |
| 👁 monitor | P36 retire | upstream PR [#40798](https://github.com/vllm-project/vllm/pull/40798) |

---

## Testing

Two gates run separately and independently.

### Unit gate (CPU-only, no GPU required)

```bash
cd <repo-root>
./validate_unit.sh
# or:
pytest vllm/_genesis/tests/ -v -m 'not cuda_required and not gpu_required'
```

Full unit suite passes: **605 + 43 (v7.8) + ~40 (v7.9 model_detect + P51) = ~688 tests**. No network, no CUDA.

### Integration gate (GPU required)

```bash
cd <repo-root>
./validate_integration.sh      # brings up docker-compose.integration.yml
# For cross-quantization verification:
docker compose -f docker-compose.integration-awq.yml up -d
docker compose -f docker-compose.integration-fp16kv.yml up -d
```

Integration checks: 3× 256k prefill+decode, speed (t/s @ 100k), KV tokens, all applied/skipped/failed counters.

### Manual smoke (production container)

```bash
# Verify model_detect profile:
docker exec vllm-prod python -c "from vllm._genesis.model_detect import get_model_profile; import json; print(json.dumps(get_model_profile(), indent=2))"

# Verify applied patches:
docker exec vllm-prod python -m vllm._genesis.patches.apply_all --verify
```

---

## Upstream status tracking (as of 2026-04-24)

Items we track / monitor:

| PR / Issue | Status | Impact |
|---|---|---|
| [#40807 — TurboQuant+spec-decode capture crash](https://github.com/vllm-project/vllm/issues/40807) | OPEN | Reporter namechecks Sander's Patch 23; our P44 aligns. Comment drafted, not yet posted. |
| [#40792 — TQ k8v4 GQA head grouping](https://github.com/vllm-project/vllm/pull/40792) | OPEN | May supersede our P40. Diff + bench pending. |
| [#40798 — TQ scratch workspace across layers](https://github.com/vllm-project/vllm/pull/40798) | OPEN | Superset of #40655+#40706. May conflict with P28 anchor. |
| [#40794 — MoE unpad routed output](https://github.com/vllm-project/vllm/pull/40794) | **MERGED 2026-04-24** | Smoke test pending on Qwen3.6-35B-A3B. |
| [#40420 — TurboQuant continuation-prefill OOM at 185K](https://github.com/vllm-project/vllm/issues/40420) | OPEN | Our P22 covers adjacent class. Adding ≥150k regression test. |
| [#40172 — Fused Mamba postprocess (+15-17% decode)](https://github.com/vllm-project/vllm/pull/40172) | OPEN | On merge: drop P25 guard. |
| [#40384 — Exclude O(1) Mamba groups](https://github.com/vllm-project/vllm/pull/40384) | OPEN | Sander co-author credit. On merge: drop Patch 9. |

Three-source truth: every patch is verified against the tagged release, `main` HEAD, and nightly image before each deploy.

---

## Architecture

```
vllm/_genesis/
├── __init__.py               Public API
├── guards.py                 Canonical vendor/chip/model detection
├── interface_guard.py        v7.8 — GenesisInterfaceMismatch + validate_impl
├── model_detect.py           v7.9 — MoE/hybrid/TQ dispatch probing
├── memory_metrics.py         Per-pool reporting (snapshot + humanize)
├── prealloc.py               GenesisPreallocBuffer framework
├── kernels/                  Pure-python drop-in replacements
│   ├── dequant_buffer.py     P22/P26/P32/P33/P36 + v7.9 P51 hook
│   ├── fla_kkt_buffer.py     P39a pool
│   ├── fp8_dispatcher.py     Ampere FP8 routing
│   ├── gdn_core_attn_manager.py  P28 buffer
│   ├── gdn_dual_stream.py    P7 opt-in
│   ├── gdn_gating_buffer.py  P46 pool
│   ├── marlin_tuning.py      P17/P18 bsm+num_warps
│   ├── moe_intermediate_cache.py  P37 pool
│   ├── page_size_padded.py   P5/P5b
│   ├── router_softmax.py     P31 fp32 upcast
│   ├── tq_continuation_prefill.py  P22/P23/P26
│   ├── tq_decode_tune.py     P36/P40
│   └── tq_grouped_decode.py  P40 opt-in
├── middleware/
│   ├── __init__.py
│   └── response_cache_middleware.py  P50 ASGI cache
├── patches/
│   ├── apply_all.py          Orchestration entrypoint
│   └── upstream_compat.py    PR marker registry
├── wiring/                   Attach-to-vLLM layer
│   ├── __init__.py
│   ├── rebind.py             Attribute rebind registry
│   ├── text_patch.py         Text anchor replacement (idempotent)
│   ├── patch_22_tq_prealloc.py
│   ├── patch_24_moe_tune.py         (P52 gated)
│   ├── patch_28_gdn_core_attn.py    (P53 gated)
│   ├── patch_31_router_softmax.py   (P52 gated)
│   ├── patch_34_mamba_deadlock_guard.py  (P53 gated)
│   ├── patch_37_moe_intermediate_cache.py  (P52 gated)
│   ├── patch_38_tq_continuation_memory.py
│   ├── patch_39_fla_kkt_buffer.py   (P53 gated)
│   ├── patch_46_gdn_gating_buffers.py  (P53 gated)
│   └── ...
└── tests/                    pytest TDD suite
```

---

## Contributing

PRs welcome. Bug reports from hardware we don't test (3090, 4090, A6000, A100, H100, ROCm, XPU) are especially valuable.

Note on MoE tuning: on Ampere with FP8 block quantization, vLLM selects **MARLIN**, not Triton, for MoE. Before tuning, check the startup log:

```
Using MARLIN Fp8 MoE backend out of potential backends: [...]
```

If MARLIN, the only runtime lever is `block_size_m` (Patch 17).

---

## Attribution

Genesis draws on and credits prior work:

- **DeepSeek-V3 team** — fp32 router upcast pattern (basis for P31)
- **@JartX** — TurboQuant author, `JartX/vllm#11` FP16 rotation (P20 prerequisite)
- **@jhsmith409** — endorsed Genesis Ampere investigation, pre-approved P22
- **@ZJY0516** — hybrid prefix cache design clarifications
- **@vibhavagarwal5** — collaborative PR scope guidance
- **@youkaichao** — memory profiler invariants documentation
- **vLLM core team** (@WoosukKwon, @zhuohan123, @robertgshaw2-redhat, @bnellnm) — responsive community, educational codebase

Per-kernel attribution lives in each module's docstring.

---

## Author

**Sandermage(Sander)-Barzov Aleksandr**
Ukraine, Odessa
GitHub: [@Sandermage](https://github.com/Sandermage)
Project: [genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches)

---

## License

Apache-2.0 — see `LICENSE`.

---

*Genesis vLLM Master Plan v7.0 / integration v7.1–v7.9.*
*Canonical reference: `Genesis_Doc/common/GENESIS_VLLM_MASTER_PLAN_v7.0_20260424.md`.*
*Integration report: `docs/REPORT_v7_1_INTEGRATION_20260424_RU.md`.*

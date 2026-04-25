# Genesis vLLM Patches — v7.10 (+ v7.11 P56 spec-decode workaround)

**Runtime patches for [vLLM](https://github.com/vllm-project/vllm) — long-context Qwen3-class inference on NVIDIA Ampere, with TurboQuant k8v4 KV-cache and 256k+ context.**

> **28 active patches. 5 configurations validated. 258,528 tokens proven end-to-end.**
> Defense-in-depth interface guards (P49). ASGI response-cache middleware (P50).
> Runtime architecture-dispatch detection (P51 / P52 / P53).
> Spec-decode safe-path guard (P56 — opt-in workaround for [vllm#40831](https://github.com/vllm-project/vllm/issues/40831)).
> Cross-model validated on FP8 / AWQ 4-bit / FP16-KV / dense configurations.
>
> ⚠️ **Test matrix is 2× RTX A5000 (Ampere SM 8.6) + NVIDIA drivers 570+.**
> Patches are written defensively for AMD ROCm / Intel XPU / Hopper / Blackwell / FP8-native paths — they graceful-skip on platform mismatch rather than crash.
> Bug reports from other hardware are very welcome.
> Optional support / hardware sponsorship — see [SPONSORS.md](SPONSORS.md).

---

## 🔒 Exact pinned versions (reproducibility)

This is the **exact vLLM build** against which every v7.10 patch is validated. If you install this SHA, you get byte-for-byte the same vLLM we tested. No "nightly", no "latest" — a real immutable pin.

| Component | Pinned value |
|---|---|
| **vLLM version string** | `v0.19.2rc1.dev134+gfe9c3d6c5` |
| **Git commit SHA** | `fe9c3d6c5f66c873d196800384ed6880687b9e52` |
| **Commit date** | **2026-04-23 04:35 UTC** |
| **Commit title** | `[TurboQuant] enable FA3/FA4 for prefill paths (#40092)` |
| **Docker image** | `vllm/vllm-openai:nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52` |
| **Genesis tag** | `v7.10.0` (this release) |

**Genesis v7.10 validation matrix (2026-04-24 → 2026-04-25 session on 2× A5000 VM 100):**

| Config | Quantization | Arch | Tokens proven | Decode @ context | Smoke | Leak |
|---|---|---|---|---|---|---|
| Qwen3-Next-35B-A3B-FP8 | FP8 | MoE + hybrid + TQ k8v4 | **258,528** | 66.6 t/s @ 100k | ✅ 10/10 | 0 MiB |
| Qwen3-Next-35B-A3B-AWQ | AWQ 4-bit | MoE + hybrid + TQ k8v4 | **258,528** | 64.8 t/s @ 100k | ✅ 10/10 | 0 MiB |
| RYS-Qwen3.5-27B-FP8-XL dense | FP8 | dense attn, no MoE, no hybrid | 28,528 (max_model_len) | 39.5 t/s @ 28k | ✅ 10/10 | — |
| Qwen3-Next fp16kv (bonus) | FP8 weights + fp16 KV | MoE + hybrid, non-TQ KV | 28,528 | 159.6 t/s @ 28k | ✅ 10/10 | — |

Raw results: [benchmarks/v7_10_validation_20260424/](benchmarks/v7_10_validation_20260424/)
Master summary: [benchmarks/v7_10_validation_20260424/MASTER_SUMMARY.md](benchmarks/v7_10_validation_20260424/MASTER_SUMMARY.md)

---

## What's new in v7.11 (2026-04-25) — spec-decode workaround + diagnostic tooling

**Investigation + workaround for [vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831)** — TurboQuant × any speculative decoding (MTP or ngram) produces degenerate token loops on structured outputs (tool calls, recall, streaming).

**Layer 1 root cause**: `turboquant_attn.py:192` declares `_init_reorder_batch_threshold(1, supports_spec_as_decode=False)`. Spec-decode batches (q_len > 1) are routed into `_prefill_attention`'s synthetic-decode fast path (lines 622-646) where the per-row online softmax breaks GQA causal coupling between draft positions → catastrophic XML/JSON loops on structured outputs.

**P56 — TQ spec-decode safe-path guard** (opt-in via `GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=1`):
- 5-line text-patch tightens fast-path entry from `q_len ≤ _CONTINUATION_DECODE_THRESHOLD` to `q_len == 1`
- Spec-decode batches now fall through to `_continuation_prefill`'s `flash_attn_varlen_func(causal=True)` — causal-correct
- **Closes the catastrophic Layer 1**: `tool_calls[]` populated again, narrative output coherent, no `<tool_call><tool_call>` infinite loops

**Layer 2 (still open)**: independent diff probing surfaced **token-level duplication** that P56 does not close — `for for`, `age age`, `parameter parameter` patterns. Appears to be acceptance-criterion × TQ quant noise interaction (not routing). Hypothesis + analysis posted upstream as [#40831 comment](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4317214311) for the right hands to pick up.

**New diagnostic scripts** (reusable for future regression hunts):
- [`scripts/sequential_backend_probe.py`](scripts/sequential_backend_probe.py) — 9-prompt probe set + `diff` subcommand for side-by-side comparison of two backends
- [`scripts/dual_backend_diagnostic_proxy.py`](scripts/dual_backend_diagnostic_proxy.py) — FastAPI proxy on :9000 that fans each request to two backends concurrently, captures diff + degenerate-pattern detection

**Upstream interactions**:
- Issue [#40807](https://github.com/vllm-project/vllm/issues/40807#issuecomment-4316663581) — pointed at our P44+P23 as fix direction for the CUDA graph crash
- Issue [#40124](https://github.com/vllm-project/vllm/issues/40124#issuecomment-4316828133) — replied to noonghunna's heads-up; promised the test we then ran
- Issue [#40831](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4317214311) — full Layer 1 root cause + P56 workaround + Layer 2 finding

---

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

### vLLM versions

| Component | Value |
|---|---|
| **Tested & pinned** | `v0.19.2rc1.dev134+gfe9c3d6c5` ([SHA `fe9c3d6c5`](https://github.com/vllm-project/vllm/commit/fe9c3d6c5f66c873d196800384ed6880687b9e52), 2026-04-23) |
| Docker image | `vllm/vllm-openai:nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52` |
| Rebase cadence | Only when upstream lands PRs we track (see "Upstream status") |
| Minimum supported | `v0.19.2rc1.dev8+` — older builds: some anchors miss and graceful-skip |
| Forward-compat | All patches graceful-skip on anchor drift — future builds still boot, just with fewer Genesis patches applied |

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

### Models validated in v7.10

| Model | Arch | Quant | Max tokens proven | Status |
|---|---|---|---|---|
| Qwen3-Next-35B-A3B-FP8 | MoE + hybrid | FP8 | **258,528** | ✅ prod baseline |
| Qwen3-Next-35B-A3B-AWQ | MoE + hybrid | AWQ 4-bit | **258,528** | ✅ cross-quantization |
| RYS-Qwen3.5-27B-FP8-XL | dense attn | FP8 | 28,528 | ✅ dense graceful-skip |
| Qwen3-Next-FP8 (fp16 KV) | MoE + hybrid | FP8 wt + fp16 KV | 28,528 | ✅ non-TQ dispatch |
| Gemma-4-26B MoE AWQ | MoE multimodal | AWQ 4-bit | — | ❌ blocked by vLLM × model (not ours, see [benchmarks/v7_10_validation_20260424/gemma4_26b_moe/FAILURE_NOTE.md](benchmarks/v7_10_validation_20260424/gemma4_26b_moe/FAILURE_NOTE.md)) |

---

## Installation / deployment

**👉 Full step-by-step guide:** [QUICKSTART.md](QUICKSTART.md) (EN + RU, ~15 min from clone to working server).

Short version (3 commands):

```bash
git clone https://github.com/Sandermage/genesis-vllm-patches.git && cd genesis-vllm-patches
git checkout v7.10.0
docker pull vllm/vllm-openai:nightly-fe9c3d6c5f66c873d196800384ed6880687b9e52
docker compose -f docker-compose.integration.yml up -d
```

After ~3-5 min boot you'll see in `docker logs vllm-integration-v7`:

```
[INFO genesis.apply_all] Genesis Results: 28 applied, 4 skipped, 0 failed
(APIServer pid=1) INFO ... Uvicorn running on http://0.0.0.0:8000
```

Then test:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer genesis-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b-integration","messages":[{"role":"user","content":"Hi"}],"max_tokens":16}'
```

### Compose files included

| File | Model | Purpose |
|---|---|---|
| `docker-compose.integration.yml` | Qwen3-Next-35B-A3B-FP8 + TQ k8v4 | **Default — production-mirror** |
| `docker-compose.integration-awq.yml` | Qwen3-Next-35B-A3B-AWQ + TQ k8v4 | AWQ 4-bit, 2.5× more KV memory |
| `docker-compose.integration-fp16kv.yml` | Qwen3-Next, kv=auto | Non-TurboQuant baseline |
| `docker-compose.qwen3-5-dense.yml` | RYS-Qwen3.5-27B-FP8-XL | Dense model test |
| `docker-compose.example.yml` | template | Reference for custom configs |

### Important: stopping cleanly

**Always use `docker compose down`, never plain `docker stop`.** Genesis text-patches modify container filesystem; restarting an already-patched container fails. See [QUICKSTART.md#troubleshooting](QUICKSTART.md#troubleshooting) for the full explanation.

### Opt-in patches (default OFF)

| Env var | Patch |
|---|---|
| `GENESIS_ENABLE_P5B=1` | Page-size pad-smaller allocator |
| `GENESIS_ENABLE_P7B=1` | GDN dual-stream custom-op (+8% decode on hybrid) |
| `GENESIS_ENABLE_P37=1` | MoE intermediate-cache pool text-patch |
| `GENESIS_ENABLE_P40=1` | TurboQuant grouped-decode (until [#40792](https://github.com/vllm-project/vllm/pull/40792) merges) |
| `GENESIS_ENABLE_P41=1` | ResponseCacheMiddleware (needs P50 wiring) |

### Healthy boot signature

```
[INFO genesis.apply_all] Genesis Results: 28 applied, 4 skipped, 0 failed
[INFO genesis.wiring.p28_gdn_core_attn] forward_cuda patched + __init__ wrapped
[INFO genesis.wiring.p22_tq_prealloc] rebound TurboQuantAttentionImpl._ensure_on_device
```

On a dense (non-MoE / non-hybrid) model boot you'll additionally see P52/P53 dispatch skips:

```
[INFO genesis.model_detect] [Genesis v7.9 dispatch] P24 MoE num_warps/num_stages overlay skipped — dense model
[INFO genesis.model_detect] [Genesis v7.9 dispatch] P28 GDN core-attn forward rewire skipped — pure-attention model
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
| **P56** | `GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=1` | **TQ spec-decode safe-path guard — partial workaround for [#40831](https://github.com/vllm-project/vllm/issues/40831). Closes catastrophic loops; subtle token duplication remains, see Layer 2 in the upstream comment** |

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

## Upstream status tracking

Compat-verified against our pinned SHA `fe9c3d6c5` (2026-04-25).

| PR / Issue | Status | Our verdict | Compat with our pin |
|---|---|---|---|
| [#40124 — TurboQuant + Hybrid MoE broken on Ampere (13 patches)](https://github.com/vllm-project/vllm/issues/40124) | OPEN (our own filing, Sander) | PR [#40384](https://github.com/vllm-project/vllm/pull/40384) already extracts our Patch 9 with `Co-authored-by: Sandermage` by @jhsmith409 | — |
| [#40807 — TQ+spec-decode capture crash](https://github.com/vllm-project/vllm/issues/40807) | OPEN issue | **We are the fix** (our P44 + P23). Reporter (@noonghunna) namechecks our Patch 23 and runs our patches in production. [Comment posted](https://github.com/vllm-project/vllm/issues/40807#issuecomment-4316663581) | — |
| [#40831 — TurboQuant × spec-decode degenerate loops](https://github.com/vllm-project/vllm/issues/40831) | OPEN issue | **Layer 1 root cause located + workaround P56**. **Layer 2 (token duplication) remains** — outside our scope (acceptance × quant noise). [Full analysis posted](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4317214311) | partial fix shipped as opt-in P56 |
| [#40792 — TQ k8v4 GQA head grouping](https://github.com/vllm-project/vllm/pull/40792) | OPEN PR | Supersedes our P40 functionally (+16.5-27.2% in their bench) | ✅ **backport-clean** — target file unchanged between our SHA and PR base |
| [#40798 — TQ scratch workspace sharing](https://github.com/vllm-project/vllm/pull/40798) | OPEN PR | Supersedes our P36 (×3.4 KV capacity in their bench) | ✅ **backport workable** — `WorkspaceManager.get_simultaneous()` already present in our pin |
| [#40794 — MoE unpad routed output](https://github.com/vllm-project/vllm/pull/40794) | **MERGED 2026-04-24** | Post-merge; MoE smoke shows no drift on Qwen3-Next-35B-A3B | — |
| [#40420 — TQ continuation-prefill OOM at 185k](https://github.com/vllm-project/vllm/issues/40420) | OPEN | Our P22+P38 cover the class. Probe M now defends in integration gate. | — |
| [#40172 — Fused Mamba postprocess (+15-17%)](https://github.com/vllm-project/vllm/pull/40172) | OPEN | On merge: drop our P25 guard. | — |
| [#40384 — Exclude O(1) Mamba groups](https://github.com/vllm-project/vllm/pull/40384) | OPEN | Sander co-author credit. On merge: drop Patch 9. | — |

Deep-dive (RU+EN): [benchmarks/v7_10_validation_20260424/upstream_compare/PR_DEEP_DIVE.md](benchmarks/v7_10_validation_20260424/upstream_compare/PR_DEEP_DIVE.md)

### v7.11 roadmap — backport #40792 / #40798 as opt-in

Neither PR has merged yet; #40807 may sit OPEN for weeks. Rather than wait, we plan:

| v7.11 patch | Source | Mode | Retire when |
|---|---|---|---|
| `P40b` (backport of #40792 kernel) | Port `_tq_grouped_decode_stage1` Triton kernel | opt-in: `GENESIS_ENABLE_PR40792_BACKPORT=1` | #40792 merges |
| `P36b` (backport of #40798 workspace) | Rewire via `WorkspaceManager.get_simultaneous()` + text-patch anchor adjust | opt-in: `GENESIS_ENABLE_PR40798_BACKPORT=1` | #40798 merges |

Both will include **full attribution** to original PR authors in docstring + our measured A5000 numbers. If our Ampere bench reveals tuning specifics not obvious from the original PR (e.g. `BLOCK_H=8` vs `16` tradeoff on SM 8.6), we will contribute that back upstream as a follow-up comment on the PR.

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
│   ├── patch_56_spec_decode_decode_path_guard.py  (v7.11 — opt-in)
│   └── ...
└── tests/                    pytest TDD suite

scripts/                      Operator + diagnostic tooling
├── run_validation_suite.sh   Universal per-model validation runner
├── compile_results.py        Aggregate per-model JSONL → MASTER_SUMMARY.md
├── sequential_backend_probe.py  9-prompt probe + diff (v7.11)
└── dual_backend_diagnostic_proxy.py  FastAPI :9000 dual-backend proxy (v7.11)
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

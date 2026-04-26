# Supported Models — Genesis vLLM Patches

This document explains which models the Genesis patcher targets, **why we chose Qwen3.6-35B-A3B-FP8 as the default**, and how to evaluate alternative models.

---

## Default model: `Qwen/Qwen3.6-35B-A3B-FP8`

**HuggingFace**: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8

| Property | Value |
|---|---|
| Total parameters | 35.95 B |
| Active per token | ~3 B (8 of 256 experts × 512 + shared expert) |
| Architecture | `qwen3_5_moe` (hybrid GDN+full-attention, 10 full + 30 GDN layers) |
| Hidden size | 2048 |
| Layers | 40 |
| Q/KV heads | 16 / 2 (GQA 8:1) |
| Experts | 256, top-8, shared expert |
| MTP layer | 1 (`mtp_num_hidden_layers: 1`) |
| Native context | 262,144 (256K) |
| Disk size | ~36 GB FP8 |
| License | Apache-2.0 |

### Why this specific model

#### 1. Hardware envelope match (48GB VRAM, 2× A5000 Ampere)
The `qwen3_5_moe` architecture in FP8 lands at ~36GB on disk and ~18GB per GPU under TP=2. Combined with TurboQuant `k8v4` KV cache (FP8 K + 4-bit V), a single 256K-context session fits in <22GB per GPU, leaving headroom for batch prefill.

Larger Apr-2026 releases — DeepSeek V4 Flash (158GB), GLM-5.1 (754B), Qwen3-Next-80B (80GB) — all exceed our VRAM budget and would require NVFP4 (Blackwell-only) or REAP-style expert pruning to fit.

#### 2. Active-parameter throughput on Ampere
A3B = 3B active parameters per forward pass on top of a 35B sparse backbone. On 2× A5000 (no FP8 native compute, Marlin weight-only path), this yields ~57 tok/s baseline and 127 tok/s with the Genesis MTP stack. A dense 35B model would saturate the PCIe Gen4 bus and run 3-4× slower. **A3B is the highest-throughput configuration the SM 8.6 generation can sustain at this parameter count.**

#### 3. Genesis patch lock-in (37 runtime patches)
Genesis maintains 37 vLLM runtime patches specifically targeting the qwen3_5_moe layer family:
- **P17/P31/P37** — 256-expert MoE routing fixes
- **P28/P46/P60/P60b** — GDN+full-attention hybrid (10/30 split) state recovery
- **P67/P67b/P75** — Triton TurboQuant kernels for spec-decode K+1 verify
- **P22/P26/P38/P40/P44** — TurboQuant continuous prefill / mixed-batch buffer pools
- **P58/P59/P61/P61b/P62/P64/P68/P69/P70/P71/P77** — Qwen3 tool-call/parser/spec-decode

Switching architecture (Gemma 4, DeepSeek V4, GLM 5) would require a new patch port (estimated 2-3 weeks per family). The patches deliver +32% TPS and 96-100% tool-call clean rate — measured in production.

#### 4. Long-context budget (262,144 tokens verified)
Our agentic workload (tool-call chains + free-form generation) routinely exceeds 100K tokens. Qwen3.6-35B-A3B's native 262K window has been **end-to-end verified at 252K under load** (96% of cap). The closest sub-50GB alternatives (Gemma 4 26B A4B at 128K, Qwen3.6 dense variants at 32K) fall short.

#### 5. Tool-calling fidelity
Qwen3 series has best-in-open-weight Hermes-style XML tool-call templates and a chat template that survives ngram speculative decoding when paired with Genesis's strict spec-decode config (`prompt_lookup_min=8`). Abliterated/distilled variants (Huihui, batsclamp) trade refusal removal for tool-call grammar fragility — a losing trade for an aggregator stack.

#### 6. Ecosystem and maintenance
1.2M downloads, 179 likes on `Qwen/Qwen3.6-35B-A3B-FP8` as of Apr 2026. Active upstream (Qwen team patches GDN/MoE bugs in vLLM main) means our patches keep tracking real upstream progress rather than diverging into a personal fork.

---

## Tested alternative models

### `Qwen/Qwen3.6-27B-FP8` (dense variant)

**HuggingFace**: https://huggingface.co/Qwen/Qwen3.6-27B-FP8

A friend of the project ([@noonghunna](https://github.com/noonghunna/qwen36-27b-single-3090)) runs this on a single RTX 3090.

| vs default | Detail |
|---|---|
| Architecture | `qwen3_5` dense — same hybrid 3:1 GDN+full pattern, no MoE |
| Total params | 27.78 B (vs 35.95B) |
| Active params/token | **27.78 B (full forward — no MoE sparsity)** |
| VRAM @ FP8 | ~28 GB (fits on 1× 24GB or 2× 24GB) |
| Speed estimate | ~50-65 tok/s decode (vs our 127 with MTP A3B) |
| Patch compat | **32 of 37 patches apply** (5 MoE-only auto-skip via `model_detect.py`) |

**Compose template**: `docker-compose.qwen3-5-dense.yml` (already in repo).

**Recommendation**: validated for friend's setup — works with our patcher. **Not recommended as primary** because dense 27B is 2-2.5× slower than our A3B baseline.

### `google/gemma-4-26B-A4B-it`

**HuggingFace**: https://huggingface.co/google/gemma-4-26B-A4B-it

| Property | Value |
|---|---|
| Architecture | `gemma4` (128 experts, A4B = 4B active) |
| VRAM @ FP8 | ~27 GB (fits) |
| Context | 128K (vs our 262K) |
| Hybrid attention | NO — different MoE design from `qwen3_5_moe` |

**Compose template**: `docker-compose.gemma4-26b-moe.yml` (experimental).

**Recommendation**: experimental support — may work but **needs new patches** for Gemma 4-specific MoE layout. Not first-class supported. Most Genesis patches will skip via dispatcher; performance will likely be lower than Qwen3.6-A3B.

---

## Models we evaluated but did NOT adopt

### `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
**Why not**: 80GB FP8 weights don't fit in our 48GB VRAM budget. Would require AWQ-INT4 quant (no official release) + multi-node setup.

### `deepseek-ai/DeepSeek-V4-Flash` (284B / 13B active)
**Why not**: 158GB on disk — needs 4× A100 80GB. Architecture `deepseek_v4` (CSA+HCA hybrid) is incompatible with our 37 GDN/MoE patches. Would require 2-3 weeks of new patch port work even if hardware was available.

### `deepseek-ai/DeepSeek-V4-Pro` (862B)
**Why not**: 860GB on disk. Out of scope for any single-node Ampere setup. Use cloud inference providers if you need V4-Pro quality.

### `zai-org/GLM-5.1` (754B)
**Why not**: same as DeepSeek-V4-Pro — requires Blackwell + REAP-pruned variant (~218B-class) which doesn't exist yet.

### `Infatoshi/Qwen3.6-35B-A3B-NVFP4-FP8`
**Why not**: NVFP4 requires Blackwell sm_120; Ampere SM 8.6 is unsupported. Bookmark for when we get a Blackwell card.

### `batsclamp/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated-FP8`
**Status**: WAIT-AND-SEE.

This is a four-stage derivative:
1. Base: `Qwen/Qwen3.6-35B-A3B`
2. Abliterated: `huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated` (refusals removed via orthogonalization)
3. Claude-tuned: `huihui-ai/Huihui-Qwen3.6-35B-A3B-Claude-4.7-Opus-abliterated` (claimed Claude 4.7 Opus distillation — undocumented training)
4. This repo: `batsclamp` FP8 quantization (blockwise [128,128] + dynamic per-token)

**Pros**: drop-in by architecture (all 37 patches applicable); abliteration removes refusals; Claude distillation *might* improve tool-calling.

**Cons**:
- Only 2 likes / 784 downloads on HF — community traction zero
- No published benchmarks
- "Claude-4.7-Opus" naming is marketing — no proof of distillation quality
- Abliteration **always degrades quality** (~3-5% MMLU drop in literature, more on reasoning)
- Tool-calling is the FIRST thing that breaks under abliteration (structured-output grammar fragility)
- Blockwise FP8 dynamic activation untested on our Ampere pipeline
- batsclamp is solo quantizer — no organizational backing

**Recommendation**: don't switch in production. If curious, run a 30-min blue/green test, but expect tool-call clean rate to drop noticeably below our current 96-100%.

---

## How to add a new model

### Step 1: Architecture compatibility check

```bash
# Get the model's config
huggingface-cli download <org>/<model> --local-dir /tmp/check --include="config.json"
cat /tmp/check/config.json | jq '.model_type, .architectures, .num_local_experts // "dense"'
```

| `model_type` | Genesis support | Patches applicable |
|---|---|---|
| `qwen3_5_moe` | **PRIMARY** | All 37 |
| `qwen3_5` | Good | 32/37 (5 MoE-only skip) |
| `qwen3_next` | Likely good | Most apply |
| `gemma4` | Experimental | Subset only |
| `deepseek_v4` | NO | Different arch family |
| `glm_moe_dsa` | NO | Different arch family |
| `mixtral` | Partial | Plain MoE patches only |

### Step 2: VRAM math

```python
# Rough check (rule of thumb)
total_disk_gb = <from HF model card>
bytes_per_param = 1 if fp8 else 2  # fp16
weights_vram = total_disk_gb  # FP8 keeps full size
kv_cache_vram_at_256K = (max_model_len * num_kv_heads * head_dim * 2 * num_layers / 1e9)  # bytes / GB
total_per_gpu = (weights_vram + kv_cache_vram_at_256K) / tp_size
assert total_per_gpu < 22.0, f"Won't fit on 24GB A5000 (need {total_per_gpu:.1f}GB)"
```

### Step 3: Test container

Copy `docker-compose.example.yml` → `docker-compose.<model>.yml`, swap `--model` path, leave Genesis env enable flags as-is. Boot, watch dispatcher log for SKIP messages — those are patches detecting your arch differs and gracefully not applying. No code changes needed.

### Step 4: Quality gate

```bash
# Free-form quality
/tmp/quality_check.sh 8000

# Tool-call quality (if applicable)
/tmp/tool_call_check.sh 8000

# Long-context recall
python3 /tmp/max_ctx_probe.py
```

If 5/5 quality + 3/3 tool-call + recall up to advertised max_model_len → safe for production. If degraded — note which patches skipped and which fired in startup log; some workloads benefit from disabling specific patches.

---

## Looking ahead

**When we get NVIDIA Blackwell (planned RTX 6000 Pro 96GB):**
- DeepSeek V4 Flash AWQ-INT4 (~80GB) becomes feasible
- Qwen3-Next-80B-FP8 fits in single-card setup
- NVFP4 quants unlock larger models at same VRAM
- Genesis patch port for `deepseek_v4` arch becomes worthwhile

Until then — Qwen3.6-35B-A3B-FP8 + Genesis patches is the empirical best for 2× A5000-class hardware.

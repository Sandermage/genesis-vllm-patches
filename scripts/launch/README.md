# Genesis vLLM — Launch Scripts

Production-tested launch scripts for the 4 spec-decode method variants. Each spawns a Docker container with the full Genesis patch stack (37+ patches) bind-mounted into a stock vLLM nightly image.

> **Tested on (v7.48 baseline, 2026-04-27):**
>
> - vLLM `0.19.2rc1.dev212+g8cd174fa3` (image `vllm/vllm-openai:nightly`)
> - PyTorch `2.11.0+cu130`, Triton `3.6.0`, CUDA `13.0`
> - **NVIDIA driver `≥580.126.09`** (REQUIRED — driver 570 puts PyTorch in compat fallback ≈ 3× slower decode)
> - 2× RTX A5000 (Ampere SM 8.6), Ubuntu 24.04 + kernel 6.8

## Quick start

```bash
# 1. Set env paths to match your system (or accept defaults)
export MODELS_DIR=/nfs/your/models           # default: /path/to/models — MUST set
export GENESIS_REPO=$HOME/genesis-vllm-patches  # default: $HOME/genesis-vllm-patches
export HF_CACHE=$HOME/.cache/huggingface     # default: $HOME/.cache/huggingface
export VLLM_CACHE_BASE=$HOME/.cache/genesis_vllm  # triton + torch compile caches
export CONTAINER_NAME=vllm-genesis            # default: vllm-genesis

# 2. Pick the script for your workload (see "Which script?" below)
./scripts/launch/start_mtp.sh

# 3. Wait ~5-8 min for cold compile cache, ~1-2 min warm
docker logs -f $CONTAINER_NAME

# 4. Health check
curl http://localhost:8000/health -H "Authorization: Bearer genesis-local"
```

## Which script?

| Script | When to use | Empirical TPS (v7.48, Qwen3.6-35B-A3B-FP8 / 2× A5000) | CV (stability) |
|---|---|---|---|
| **`start_no_spec_async.sh`** | Free-form chat WITHOUT tool calls. Fastest + most stable. | **134 tok/s** mean | **0.3%** (rock-solid) |
| **`start_mtp.sh`** ⭐ | Default for tool-call / agentic + general chat. Full correctness stack. | **160-190 tok/s** (v7.48: +15-30% vs v7.13) | 5-7% |
| `start_suffix.sh` | Tool-call workload — repetitive JSON/code. Peak speed on right input. | 71 tok/s mean (sweep best 84 @ prob=0.10/depth=32) | 17-30% |
| `start_ngram.sh` | Ngram-only fallback (no MTP available). | 46 tok/s | 4.4% |
| `start_ngram_p77adaptive.sh` | Ngram + P77 adaptive K controller (auto-disables on low accept). | 48-50 tok/s (+4-9%) | 6.1% |

**v7.48 long-context capability** (start_mtp.sh @ GMU 0.90, P38/P40 shared singleton): 16K-200K all PASS needle test, 240K processed without OOM. See [vllm/_genesis/CHANGELOG.md](../../vllm/_genesis/CHANGELOG.md) v7.48 entry for full details.

## Honest trade-off — `--async-scheduling` vs spec-decode

vLLM has a hard mutual exclusion: **`--async-scheduling` is automatically disabled when `--speculative-config` is set**. We measured both sides:

| Metric | no-spec + async (start_no_spec_async.sh) | MTP (start_mtp.sh) |
|---|---|---|
| Free-form throughput | **134 tok/s** | 130 tok/s |
| Stability (CV 12 runs) | **0.3%** (extreme) | 5.0% |
| Tool-call clean rate | ❌ Broken — cascades | ✅ 3/3 PASS |
| Long-context | 160K (per stable config) | **252K** |

**Pick by use-case**:
- Aggregator with tool calls / agents → `start_mtp.sh` (sacrifice 4 tok/s for tool correctness + long ctx)
- Pure chat / no tools → `start_no_spec_async.sh` (faster + more stable)
- Hybrid: run BOTH containers on different GPUs and route requests by `tools` field

### Why we DIDN'T merge MTP + `--async-scheduling`

Surprising empirical finding (2026-04-26): vLLM technically allows MTP/EAGLE/ngram_gpu + `--async-scheduling` (only ngram-CPU/suffix/medusa are auto-disabled). We tested:

| Config | Mean tok/s | CV |
|---|---|---|
| MTP without async (our standard) | 130 | 5.0% |
| **MTP WITH async** | **123** | 7.4% — **WORSE!** |
| no-spec + async | 134 | 0.3% |

**Why MTP+async is slower on single-user setup** (max_num_seqs=2):
- CPU scheduler overhead is tiny (very small batches)
- Async adds synchronization overhead (events, locks, thread dispatch)
- When CPU work < async overhead → async LOSES

This matches upstream evidence:
- vLLM PR #24799 own benchmark: 1.8% gain at 24 prompts, only 7.1% at 96 prompts (single-user is FAR below this band)
- vLLM PR #32951 (zero-bubble async+spec, merged 2026-03-23): ~3% TPOT improvement on H100/DeepSeek-V3.2 — high-concurrency only
- SGLang Spec V2 + EAGLE3: [issue #12411](https://github.com/sgl-project/sglang/issues/12411) reports overlap **slower than no-spec at concurrency=1**
- Theoretical Amdahl ceiling for single-user: +3-7% maximum (scheduler is only 5-10% of step time)

**Conclusion**: async-scheduling helps multi-user / high-throughput servers (24+ concurrent prompts). For single-user / aggregator-style workloads (max_num_seqs ≤ 4), it's neutral-to-negative. Stay with sync MTP unless you're running at concurrency 24+.

## Common configuration (all 4 scripts)

All scripts share these baseline parameters:

- **Hardware**: 2× GPU (TP=2), `--gpus all`
- **Memory**: 8GB shm, 64GB container memory
- **Port**: 8000
- **Model**: Qwen3.6-35B-A3B-FP8 (override via `--model` arg in script if needed)
- **Context**: 262144 tokens (256K max model length)
- **GPU memory utilization**: 0.91 (with P74 chunk-clamp)
- **API key**: `genesis-local` (CHANGE for production — see Security note below)

## Required env vars (override defaults if your paths differ)

| Env var | Default | What it points to |
|---|---|---|
| `MODELS_DIR` | `/path/to/models` ⚠️ MUST SET | Directory with Qwen3.6-35B-A3B-FP8 weights |
| `GENESIS_REPO` | `$HOME/genesis-vllm-patches` | Repo root (we bind-mount `_genesis/` from here) |
| `HF_CACHE` | `$HOME/.cache/huggingface` | HuggingFace download cache |
| `VLLM_CACHE_BASE` | `$HOME/.cache/genesis_vllm` | Container-internal: triton-cache + torch-compile-cache |
| `CONTAINER_NAME` | `vllm-genesis` | Docker container name |

## Genesis env vars (already set in scripts — see INSTALL.md for full reference)

The launch scripts hard-code production defaults for all `GENESIS_ENABLE_*` flags. To override, edit the script or set env BEFORE running. Key flags:

- `GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1` — TurboQuant multi-query kernel (ALL configs)
- `GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1` — unblocks `--max-num-batched-tokens=8192`
- `GENESIS_ENABLE_P74_CHUNK_CLAMP=1` — prealloc safety net
- `GENESIS_ENABLE_P75_SUFFIX_DECODING=1` — only in `start_suffix.sh`
- `GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K=1` — only in `start_ngram_p77adaptive.sh`

See [`../../INSTALL.md`](../../INSTALL.md) for the full env reference.

## Security note

Default `--api-key genesis-local` is for **local development only**. For any deployment beyond localhost:

1. Generate a strong key: `openssl rand -hex 32`
2. Edit the script — replace `--api-key genesis-local` with `--api-key $YOUR_KEY`
3. NEVER commit the real key to git
4. Pass via env: `export VLLM_API_KEY=...` and reference as `--api-key "$VLLM_API_KEY"`

## Verifying patches loaded correctly

After boot:

```bash
docker logs $CONTAINER_NAME 2>&1 | grep "Genesis Dispatcher"
# Expect 30+ APPLY lines, ~10 SKIP (opt-in patches not enabled)

docker logs $CONTAINER_NAME 2>&1 | grep -c "applied:"
# Expect ~37 active patches
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Boot fails with `cudaErrorStreamCaptureInvalidated` | P67 not enabled OR P65 also enabled | Use any of the provided scripts as-is — they configure correctly |
| `RuntimeError: tensor a (65536) must match` | `batched-tokens=8192` without P72 | Use `start_*.sh` scripts (P72 already enabled) |
| `setStorage out of bounds` on long-context | P72 enabled without P74 | Same — provided scripts include P74 |
| Container `Up 5 minutes` then crashes | OOM during cudagraph capture | Lower GMU from 0.91 → 0.88 (edit script) |

See [`../../INSTALL.md`](../../INSTALL.md) for full troubleshooting.

## Production discipline

These scripts are **derived from production-running configs** on 2× RTX A5000. Each variant has been:
- Bench-tested (12-run mean ± std, see README empirical tables)
- Long-context verified (252K tokens single-needle PASS)
- Tool-call quality regression checked (3/3 PASS minimum)
- Pushed to private repo BEFORE prod deploy (per project discipline)

Replicate the same discipline if you fork: bench → quality gate → push to your private branch → deploy to prod.

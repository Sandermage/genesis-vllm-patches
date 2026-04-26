# Genesis vLLM — Launch Scripts

Production-tested launch scripts for the 4 spec-decode method variants. Each spawns a Docker container with the full Genesis patch stack (37+ patches) bind-mounted into a stock vLLM nightly image.

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

| Script | When to use | Empirical TPS (Qwen3.6-35B-A3B-FP8 / 2× A5000) |
|---|---|---|
| **`start_mtp.sh`** | Default for free-form / general chat. Best overall. | **130 tok/s** mean (CV 5.0%) |
| `start_suffix.sh` | Tool-call / agentic-heavy workload. Requires `pip install arctic-inference` (added to entrypoint). | **99 tok/s** mean, peak **175** |
| `start_ngram.sh` | Ngram-only deployment (no MTP available). | 46 tok/s free-form |
| `start_ngram_p77adaptive.sh` | Ngram + P77 adaptive K controller. Auto-disables spec on low acceptance. | 50 tok/s + adaptive |

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

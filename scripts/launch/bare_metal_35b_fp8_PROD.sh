#!/bin/bash
# Genesis 35B-A3B-FP8 PROD — BARE-METAL launch (no Docker).
#
# Use this when vLLM is installed natively on your host (pip install vllm,
# or git clone + editable install). For Docker workflow see the same-named
# `start_35b_fp8_PROD.sh` in this directory.
#
# Prereqs:
#   - vLLM installed and `vllm` CLI on PATH (pip or editable)
#   - Python 3.10+ with the same env vLLM uses
#   - Model checkpoint at $MODEL_PATH (default: /models/Qwen3.6-35B-A3B-FP8)
#   - Genesis Patches checkout cloned somewhere; export GENESIS_REPO=<path>
#   - 2× GPU with at least 24 GB each (TP=2). Single-card users: see
#     ../docs/BENCHMARK_GUIDE.md for `--tensor-parallel-size 1` adjustments.
#
# Tested config: 2× RTX A5000 24 GB, vLLM nightly pin 8cd174fa3, 320 K context,
# MTP K=3 spec-decode, TurboQuant k8v4 KV cache. wall_TPS 183 over N=500 stress.

set -euo pipefail

# ─── Customize these to your environment ───────────────────────────────
: "${MODEL_PATH:=/models/Qwen3.6-35B-A3B-FP8}"
: "${API_KEY:=genesis-local}"
: "${GENESIS_REPO:=$HOME/genesis-vllm-patches}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
# ────────────────────────────────────────────────────────────────────────

if [ ! -d "${GENESIS_REPO}/vllm/_genesis" ]; then
    echo "ERROR: GENESIS_REPO=$GENESIS_REPO does not contain vllm/_genesis."
    echo "       Set GENESIS_REPO to your git clone of genesis-vllm-patches."
    exit 1
fi

# Find the actual vllm site-packages dir to mount Genesis _genesis package into.
VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
if [ -z "${VLLM_DIR}" ]; then
    echo "ERROR: cannot find installed vllm. pip install vllm first."
    exit 1
fi

# Symlink (idempotent) Genesis _genesis into the vllm package dir.
GENESIS_TARGET="${VLLM_DIR}/_genesis"
if [ ! -e "${GENESIS_TARGET}" ]; then
    ln -s "${GENESIS_REPO}/vllm/_genesis" "${GENESIS_TARGET}"
    echo "Linked Genesis _genesis -> ${GENESIS_TARGET}"
elif [ -L "${GENESIS_TARGET}" ]; then
    echo "Genesis _genesis already linked at ${GENESIS_TARGET}"
else
    echo "WARN: ${GENESIS_TARGET} exists and is NOT a symlink — leaving as-is."
    echo "      If you upgraded vllm and want Genesis re-linked, remove that path manually."
fi

# Apply text-patches before launching the server.
python3 -m vllm._genesis.patches.apply_all

# ─── Genesis env flags (matches start_35b_fp8_PROD.sh) ─────────────────
export VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL=WARNING
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6"
export VLLM_FLOAT32_MATMUL_PRECISION=high
export NCCL_P2P_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_FUSED_MOE_GROUPED_TOPK=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=8
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP8=0

# Genesis patches (matches PROD v780)
export GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1
export GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1
export GENESIS_ENABLE_P60B_TRITON_KERNEL=1
export GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1
export GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1
export GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1
export GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1
export GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1
export GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1
export GENESIS_P67_USE_UPSTREAM=1
export GENESIS_P67_NUM_KV_SPLITS=32
export GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1
export GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1
export GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1
export GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS=8000
export GENESIS_ENABLE_P37=1
export GENESIS_TQ_MAX_MODEL_LEN=320000
export GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1
export GENESIS_PROFILE_RUN_CAP_M=4096
export GENESIS_ENABLE_P74_CHUNK_CLAMP=1
export GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1
export GENESIS_ENABLE_P82=1
export GENESIS_P82_THRESHOLD_SINGLE=0.3
export GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1   # v780: ~1 GiB VRAM saved
export GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1       # defensive
export GENESIS_ENABLE_P99=1
export GENESIS_ENABLE_P101=1
export GENESIS_PREALLOC_TOKEN_BUDGET=4096
export GENESIS_BUFFER_MODE=shared
# Knobs we EMPIRICALLY DON'T enable on Ampere consumer (see README):
#   GENESIS_ENABLE_P40 — needs L2 ≥ 24 MB (4090/5090/H100); A5000 = no-op
#   GENESIS_ENABLE_P83/P84/P85 — prefix-cache regression on current pin

# ─── Launch ─────────────────────────────────────────────────────────────
exec vllm serve --model "${MODEL_PATH}" --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 --max-model-len 320000 \
    --kv-cache-dtype turboquant_k8v4 --max-num-seqs 2 --max-num-batched-tokens 4096 \
    --enable-chunked-prefill --dtype float16 \
    --disable-custom-all-reduce --language-model-only --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --api-key "${API_KEY}" --served-model-name qwen3.6-35b-a3b \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --performance-mode interactivity --attention-config.flash_attn_version 2 \
    --no-scheduler-reserve-full-isl --disable-log-stats \
    --host "${HOST}" --port "${PORT}"

#!/bin/bash
# Genesis 27B-int4-Lorbus PROD — short-ctx, BARE-METAL launch (no Docker).
#
# Hybrid GDN INT4 model, MTP K=3 spec-decode, no TurboQuant (uses fp8_e5m2 KV).
# This is the high-throughput short-context config (≤8 K prompts). For long
# context (256 K) use bare_metal_27b_int4_no_TQ_long_256K.sh.
# For TurboQuant variant see bare_metal_27b_int4_TQ_k8v4.sh.
#
# Tested: 2× RTX A5000, wall_TPS 89.23 over N=500 stress, tool-call 4/4.

set -euo pipefail

: "${MODEL_PATH:=/models/Qwen3.6-27B-int4-AutoRound}"
: "${API_KEY:=genesis-local}"
: "${GENESIS_REPO:=$HOME/genesis-vllm-patches}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

if [ ! -d "${GENESIS_REPO}/vllm/_genesis" ]; then
    echo "ERROR: GENESIS_REPO=$GENESIS_REPO does not contain vllm/_genesis."; exit 1
fi
VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
GENESIS_TARGET="${VLLM_DIR}/_genesis"
[ -e "${GENESIS_TARGET}" ] || ln -s "${GENESIS_REPO}/vllm/_genesis" "${GENESIS_TARGET}"

python3 -m vllm._genesis.patches.apply_all

# Genesis env (matches start_27b_int4_no_TQ_short.sh / v771b)
export VLLM_NO_USAGE_STATS=1 VLLM_LOGGING_LEVEL=WARNING
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export VLLM_FLOAT32_MATMUL_PRECISION=high VLLM_SSM_CONV_STATE_LAYOUT=DS
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0
export VLLM_USE_FLASHINFER_SAMPLER=1 VLLM_USE_FUSED_MOE_GROUPED_TOPK=1
export OMP_NUM_THREADS=1 CUDA_DEVICE_MAX_CONNECTIONS=8
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_MARLIN_USE_ATOMIC_ADD=1
export GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1
export GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1 GENESIS_ENABLE_P60B_TRITON_KERNEL=1
export GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1 GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1
export GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1
export GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1
export GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1
export GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1
export GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1
export GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 GENESIS_PROFILE_RUN_CAP_M=4096
export GENESIS_ENABLE_P74_CHUNK_CLAMP=1
export GENESIS_ENABLE_P82=0   # P82 disabled here (not yet swept on INT4)
export GENESIS_P82_THRESHOLD_SINGLE=0.3
export GENESIS_ENABLE_P83=1 GENESIS_ENABLE_P85=1   # OK on hybrid GDN (different from 35B)
export GENESIS_ENABLE_P87=1 GENESIS_ENABLE_P91=1
export GENESIS_ENABLE_P99=1 GENESIS_ENABLE_P100=1 GENESIS_ENABLE_P101=1
export GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1
export GENESIS_PREALLOC_TOKEN_BUDGET=4096 GENESIS_BUFFER_MODE=shared

# IMPORTANT: NO --enable-prefix-caching (DS conv state layout crash, see memory)
exec vllm serve --model "${MODEL_PATH}" --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 --max-model-len 131072 \
    --max-num-seqs 4 --max-num-batched-tokens 8192 \
    --enable-chunked-prefill --dtype float16 \
    --kv-cache-dtype fp8_e5m2 \
    --disable-custom-all-reduce --language-model-only --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --api-key "${API_KEY}" --served-model-name qwen3.6-27b \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --host "${HOST}" --port "${PORT}" --disable-log-stats

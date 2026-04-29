#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚠️  EXPERIMENTAL — NOT TESTED on the maintainer rig.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# This is the single-card (TP=1) derivative of the 2× A5000 PROD config.
# Sander runs 2× A5000; he has NOT validated this config end-to-end.
# Empirical numbers (TPS, CV, max ctx, tool-call) HAVE NOT been measured.
#
# Differences from the TP=2 version:
#   - --tensor-parallel-size 1
#   - May need lower --gpu-memory-utilization (single card carries the full
#     model + KV pool + workspace; 0.85-0.92 is a starting point)
#   - --disable-custom-all-reduce + NCCL_P2P_DISABLE are no-ops at TP=1
#     (kept for parity, harmless)
#
# Hardware suitability (rough — please share results via GitHub Discussions):
#   27B-int4-AutoRound weights ≈ 14 GB (INT4) → fits on 24 GB cards with
#   KV pool + workspace headroom. Suitable single-card targets:
#     RTX 3090 24 GB, RTX 4090 24 GB, RTX 5090 32 GB, RTX A5000 24 GB,
#     RTX A6000 48 GB, RTX PRO 4000 Blackwell 24 GB, RTX PRO 4500 32 GB,
#     RTX PRO 5000 48 GB, RTX PRO 6000 96 GB, A100 / H100 / H200.
#   On 24 GB cards, expect tight headroom — consider --gpu-memory-
#   utilization 0.85-0.90 instead of 0.95.
#
# WHEN YOU RUN THIS — please open a GitHub Discussion with:
#   - Your card model + driver version
#   - Final --gpu-memory-utilization that worked
#   - JSON from `python3 tools/genesis_bench_suite.py --quick`
# We will fold confirmed-working configs back into the main scripts and
# remove the EXPERIMENTAL warning for that card class.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Genesis 27B-int4-Lorbus PROD — TQ k8v4 KV (long-ctx capable), BARE-METAL launch (no Docker).
#
#
# Hybrid GDN INT4 + TurboQuant k8v4 KV (packed slot layout, hybrid-aware).
# Requires GENESIS_ENABLE_P98=1 (WorkspaceManager revert for vllm#40941 lock
# semantics — without P98 the first decode call hits AssertionError).
# Tool-call 4/4 verified (Paris/Tokyo/NewYork/London with thinking on/off).
#
# Empirical: only ~+2% TPS over fp8_e5m2 variant on 2x A5000 (Welch p=0.067 NS).
# The big +11.3% TQ bonus seen on 35B-A3B-FP8 does NOT reproduce here because
# Lorbus 27B-INT4 routes via AllSparkLinearKernel (not Marlin / compressed-tensors).
# May give measurable gain on Hopper / Blackwell or with multi-stream workloads.

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
export GENESIS_ENABLE_P82=0
export GENESIS_ENABLE_P98=1   # REQUIRED for TQ k8v4 on hybrid (WorkspaceManager fix vs vllm#40941)
export GENESIS_P82_THRESHOLD_SINGLE=0.3
export GENESIS_ENABLE_P83=1 GENESIS_ENABLE_P85=1   # OK on hybrid GDN (different from 35B)
export GENESIS_ENABLE_P87=1 GENESIS_ENABLE_P91=1
export GENESIS_ENABLE_P99=1 GENESIS_ENABLE_P100=1 GENESIS_ENABLE_P101=1
export GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1
export GENESIS_PREALLOC_TOKEN_BUDGET=4096 GENESIS_BUFFER_MODE=shared

# IMPORTANT: NO --enable-prefix-caching (DS conv state layout crash, see memory)
exec vllm serve --model "${MODEL_PATH}" --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 --max-model-len 280000 \
    --max-num-seqs 2 --max-num-batched-tokens 2048 \
    --enable-chunked-prefill --dtype float16 \
    --kv-cache-dtype turboquant_k8v4 \
    --disable-custom-all-reduce --language-model-only --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --api-key "${API_KEY}" --served-model-name qwen3.6-27b \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --host "${HOST}" --port "${PORT}" --disable-log-stats

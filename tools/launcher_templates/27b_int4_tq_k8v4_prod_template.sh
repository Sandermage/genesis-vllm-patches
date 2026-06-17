#!/usr/bin/env bash
# Genesis 27B Lorbus INT4 AutoRound + TQ k8v4 PROD launcher template
# K.1.R.R.9 (2026-05-29)
#
# Auto-generated from qwen3.6-27b-int4-autoround-tq-k8v4.yaml +
# a5000-2x-24gbvram-16cpu-128gbram.yaml.
# Mirrors ALL 74 GENESIS env vars + 23 system_env vars + engine config.
#
# Empirically validated (rig bench, K.1.R.R.9 on 626fa9bb pin):
#   wall_TPS: 120.32 (cold) / 120.44 (warm) — CV 3-4%
#   tool-call: 7/7 ✓
#
# IMPORTANT: K.1.R.R.7 27B regression vs dev371 baseline 131.84 TPS
# (-9% on 626fa9bb) is NOT caused by Class 1 launcher drift — full
# config reproduces partial-launcher TPS. Regression is in UPSTREAM
# CODE within 42-commit window May 26→27. Mechanism requires:
#   - nsys profile diff dev371 vs 626fa9bb (45 min rig)
#   - OR source-build bisect (multi-day)
#
# Class 1 drift CHECK on 27B (K.1.R.R.9):
#   YAML enabled patches:    74
#   Partial launcher had:    ~17 explicit + GENESIS_LEGACY_DEFAULT_ON=1
#   Missing patches:         22 (PN125/PN126/PN128/PN129/PN130/PN96B etc)
#   Bench impact of missing: <1% (within noise)
#
# So 22 missing patches were mostly defensive/redundant on 27B's
# TQ k8v4 hot path. Their absence didn't materially affect TPS.
#
# Usage:
#   1. Source ~/.genesis_paths.env
#   2. Run script — boots container on port 8101
#   3. Wait ~5-6 min for boot
#   4. curl localhost:8101 with API key 'genesis-local'
set -euo pipefail

CONTAINER_NAME=vllm-pn95-2xa5000
PORT=8101

docker rm -f $CONTAINER_NAME 2>/dev/null || true
. ~/.genesis_paths.env

docker run -d \
  --name $CONTAINER_NAME \
  --gpus all \
  --shm-size=8g \
  --memory=64g \
  -p ${PORT}:${PORT} \
  -v /nfs/genesis/models:/models:ro \
  -v ${GENESIS_HF_CACHE_DIR}:/root/.cache/huggingface \
  -v ${GENESIS_COMPILE_CACHE_DIR}/compile-cache-pn95-2x:/root/.cache/vllm/torch_compile_cache \
  -v ${GENESIS_TRITON_CACHE_DIR}/triton-cache-pn95-2x:/root/.triton/cache \
  -v ${GENESIS_PROJECT_ROOT}/vllm/sndr_core:/usr/local/lib/python3.12/dist-packages/vllm/sndr_core \
  -v ${GENESIS_PROJECT_ROOT}:/genesis-vllm-patches:rw \
  -e NCCL_P2P_DISABLE=1 \
  # ── system_env (23 vars from hardware YAML) ──
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e PYTORCH_NVML_BASED_CUDA_CHECK=1 \
  -e NCCL_DEBUG=WARN \
  -e OMP_NUM_THREADS=1 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -e VLLM_LOGGING_LEVEL=WARNING \
  -e VLLM_LOG_MODEL_INSPECTION=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_STANDALONE_COMPILE=1 \
  -e VLLM_ENABLE_PREGRAD_PASSES=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP8=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  # ── Genesis runtime overrides ──
  -e GENESIS_BUFFER_MODE=shared \
  -e GENESIS_PREALLOC_TOKEN_BUDGET=4096 \
  -e GENESIS_TQ_MAX_MODEL_LEN=131072 \
  -e GENESIS_LEGACY_DEFAULT_ON=1 \
  # ── 74 GENESIS patch env vars omitted from template ──
  # Generate via:
  #   grep -E "^  GENESIS_.*: '" \
  #     vllm/sndr_core/model_configs/builtin/model/qwen3.6-27b-int4-autoround-tq-k8v4.yaml \
  #     | sed -E "s/^  ([A-Z_0-9]+): '([^']+)'.*/  -e \1=\2 \\\\/" \
  #     | grep -v "=0$"
  --entrypoint /bin/bash \
  vllm/vllm-openai:nightly-4c626633159887b0f2c962058c17c78f1434556d \
  -c "set -e; \
      pip install --quiet --disable-pip-version-check --root-user-action=ignore \
        pandas==2.2.3 scipy==1.14.1 xxhash==3.5.0 pyyaml packaging zstandard==0.23.0 pytest==8.3.4 2>&1 | tail -2; \
      pip install -e /genesis-vllm-patches --no-deps --quiet --disable-pip-version-check --root-user-action=ignore 2>&1 | tail -2; \
      python3 -m vllm.sndr_core.apply 2>&1 | tee /tmp/genesis_boot.log | tail -5; \
      exec vllm serve \
        --model /models/Qwen3.6-27B-int4-AutoRound \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.92 \
        --max-model-len 131072 \
        --max-num-seqs 4 \
        --max-num-batched-tokens 4096 \
        --enable-chunked-prefill \
        --dtype float16 \
        --kv-cache-dtype turboquant_k8v4 \
        --disable-custom-all-reduce \
        --language-model-only \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --chat-template /models/Qwen3.6-27B-int4-AutoRound/chat_template_enhanced.jinja \
        --override-generation-config '{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20}' \
        --api-key genesis-local \
        --served-model-name qwen3.6-27b \
        --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
        --host 0.0.0.0 --port ${PORT} \
        --disable-log-stats"

echo "$CONTAINER_NAME launched on :$PORT (using nightly-4c626633 = 0.23.1rc1.dev101)"

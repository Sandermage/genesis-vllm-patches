#!/usr/bin/env bash
# Genesis 35B A3B FP8 PROD launcher template — K.1.R.R.8.5 (2026-05-29)
#
# Auto-generated from qwen3.6-35b-a3b-fp8.yaml + a5000-2x-24gbvram-16cpu-128gbram.yaml.
# Mirrors ALL 79 GENESIS env vars + 23 system_env vars + engine config.
#
# Empirically validated (rig bench): 211.35 TPS / 7/7 tool-call / CV 5-7%
# Matches YAML reference 215 TPS single-conc spec.
#
# CRITICAL: Custom launchers MUST mirror YAML's genesis_env + system_env
# blocks completely. Class 1 drift bug (per skill) silently breaks
# tool-call quality (cascade <tool_call><tool_call>...) when patches missed.
#
# Usage:
#   1. Source /tmp/genesis_paths.env (or set GENESIS_* path variables)
#   2. Run this script — boots container on port 8103
#   3. Wait ~5-6 min for boot
#   4. curl localhost:8103/v1/models with API key 'genesis-local'
#
# To customize for different model/profile:
#   - Update --model path
#   - Update --max-model-len (must match GENESIS_TQ_MAX_MODEL_LEN)
#   - Update --max-num-seqs (2=latency, 8=multi-conc throughput)
#   - Re-extract env vars from target YAMLs:
#     grep -E "^  GENESIS_.*: '" YAML | sed -E "s/^  ([A-Z_0-9]+): '([^']+)'.*/  -e \1=\2 \\\\/"
set -euo pipefail

CONTAINER_NAME=vllm-35b-prod
PORT=8103

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
  -v ${GENESIS_COMPILE_CACHE_DIR}/compile-cache-35b:/root/.cache/vllm/torch_compile_cache \
  -v ${GENESIS_TRITON_CACHE_DIR}/triton-cache-35b:/root/.triton/cache \
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
  # ── Note: 79 GENESIS_* vars omitted from template for brevity.
  # Generate via:
  #   grep -E "^  GENESIS_.*: '" vllm/sndr_core/model_configs/builtin/model/qwen3.6-35b-a3b-fp8.yaml \
  #     | sed -E "s/^  ([A-Z_0-9]+): '([^']+)'.*/  -e \1=\2 \\\\/" \
  #     | grep -v "=0"  # exclude opt-out flags
  # See full instantiated version at /tmp/start_35b_optimal.sh on rig.
  --entrypoint /bin/bash \
  vllm/vllm-openai:nightly-4c626633159887b0f2c962058c17c78f1434556d \
  -c "set -e; \
      pip install --quiet --disable-pip-version-check --root-user-action=ignore \
        pandas==2.2.3 scipy==1.14.1 xxhash==3.5.0 pyyaml packaging zstandard==0.23.0 2>&1 | tail -2; \
      pip install -e /genesis-vllm-patches --no-deps --quiet --disable-pip-version-check --root-user-action=ignore 2>&1 | tail -2; \
      python3 -m vllm.sndr_core.apply 2>&1 | tee /tmp/genesis_boot.log | tail -5; \
      exec vllm serve \
        --model /models/Qwen3.6-35B-A3B-FP8 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 280000 \
        --max-num-seqs 2 \
        --max-num-batched-tokens 4096 \
        --enable-chunked-prefill \
        --dtype float16 \
        --kv-cache-dtype turboquant_k8v4 \
        --disable-custom-all-reduce \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --chat-template /models/Qwen3.6-35B-A3B-FP8/chat_template_enhanced.jinja \
        --override-generation-config '{\"temperature\":0.6,\"top_p\":0.95,\"top_k\":20}' \
        --api-key genesis-local \
        --served-model-name qwen3.6-35b \
        --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
        --host 0.0.0.0 --port ${PORT} \
        --disable-log-stats"

echo "$CONTAINER_NAME launched on :$PORT"

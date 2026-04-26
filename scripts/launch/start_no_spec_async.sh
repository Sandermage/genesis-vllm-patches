#!/bin/bash
# Genesis NO-SPEC config: --async-scheduling enabled (incompatible with spec-decode)
#
# Empirical: 134 tok/s mean (CV 0.3% — extremely stable)
# Trade-offs:
#   - Free-form text: FASTEST option (134 tok/s vs MTP 130 vs Suffix 46)
#   - Tool-call: BROKEN — without spec-decode + correctness patches, tool-call cascades
#   - Long-context: max_model_len=163840 (160K) per stable config — extend at your risk
#
# Use this if:
#   - Your workload is mostly free-form chat WITHOUT tool calls
#   - You need rock-solid stability (CV 0.3% over 12 runs)
#   - You can sacrifice tool-call quality for raw speed
#
# Use start_mtp.sh instead if:
#   - You use tool calls / function calling
#   - You need long-context > 160K
#   - You can accept 4 tok/s slower for full Genesis correctness stack
set -euo pipefail
docker stop ${CONTAINER_NAME:-vllm-genesis} 2>/dev/null || true
docker rm ${CONTAINER_NAME:-vllm-genesis} 2>/dev/null || true

docker run -d \
  --name ${CONTAINER_NAME:-vllm-genesis} \
  --network genesis-vllm-patches_default \
  --shm-size=8g --memory=64g -p 8000:8000 --gpus all \
  --security-opt label=disable --entrypoint /bin/bash \
  -v ${MODELS_DIR:-/path/to/models}:/models:ro \
  -v ${HF_CACHE:-$HOME/.cache/huggingface}:/root/.cache/huggingface:ro \
  -v ${VLLM_CACHE_BASE:-$HOME/.cache/genesis_vllm}/triton-cache-21apr:/root/.triton/cache \
  -v ${VLLM_CACHE_BASE:-$HOME/.cache/genesis_vllm}/compile-cache-21apr:/root/.cache/vllm/torch_compile_cache \
  -v ${GENESIS_REPO:-$HOME/genesis-vllm-patches}/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -v ${GENESIS_REPO:-$HOME/genesis-vllm-patches}/genesis_vllm_plugin:/plugin:ro \
  -v ${GENESIS_REPO:-$HOME/genesis-vllm-patches}/external_probe:/external_probe:ro \
  -v "${GENESIS_REPO:-$HOME/genesis-vllm-patches}/vllm/_genesis/configs/moe_tuning/E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json:ro" \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 -e VLLM_MOE_USE_DEEP_GEMM=0 -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP8=0 -e VLLM_LOGGING_LEVEL=WARNING \
  -e GENESIS_TQ_MAX_MODEL_LEN=163840 \
  vllm/vllm-openai:nightly -c \
  "set -e; pip install --quiet --disable-pip-version-check pandas scipy xxhash; \
cp -r /plugin /tmp/genesis_vllm_plugin; \
pip install --quiet --disable-pip-version-check --no-deps -e /tmp/genesis_vllm_plugin 2>&1 | tail -3; \
python3 /external_probe/patch_tolist_cudagraph.py || echo tolist bypass failed; \
python3 -m vllm._genesis.patches.apply_all --verify-rebinds; \
exec vllm serve --model /models/Qwen3.6-35B-A3B-FP8 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.905 --max-model-len 163840 \
  --kv-cache-dtype turboquant_k8v4 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --enable-prefix-caching --dtype float16 \
  --disable-custom-all-reduce --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --api-key genesis-local --served-model-name qwen3.6-35b-a3b --host 0.0.0.0 \
  --async-scheduling --performance-mode interactivity \
  --attention-config.flash_attn_version 2 --port 8000 \
  --no-scheduler-reserve-full-isl --prefix-caching-hash-algo xxhash --disable-log-stats"
sleep 5
echo "Container started — waiting for boot"

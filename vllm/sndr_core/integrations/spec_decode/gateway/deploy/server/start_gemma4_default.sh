#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Launcher: gemma4-default — production-safe TQ-only + MTP OFF.
#
# Canonical SNDR_* envs. GENESIS_* aliases still work.
# Matches the docker-compose `default` service.
set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:nightly}"
CONTAINER_NAME="${CONTAINER_NAME:-sndr-gemma4-default}"
PORT="${PORT:-8101}"
TP="${TP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
GEMMA4_MODEL="${GEMMA4_MODEL:-/models/gemma-4-31B-it-AWQ-4bit}"
MODEL_ROOT="${MODEL_ROOT:-/nfs/genesis/models}"
GENESIS_REPO="${GENESIS_REPO:-/home/sander/genesis-vllm-patches}"
API_KEY="${API_KEY:-genesis-local}"

OVL="${GENESIS_REPO}/vllm/sndr_core/integrations/gemma4/upstream_overlay_pr42637"
TGT=/usr/local/lib/python3.12/dist-packages/vllm

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

mkdir -p /tmp/sndr_default_launcher
cat > /tmp/sndr_default_launcher/run.sh <<INNER_EOF
#!/bin/bash
set -e
pip install -e ${GENESIS_REPO} --no-deps --quiet 2>&1 | tail -2
exec vllm serve ${GEMMA4_MODEL} \\
  --served-model-name gemma-4-31b \\
  --tensor-parallel-size ${TP} \\
  --disable-custom-all-reduce \\
  --dtype bfloat16 \\
  --kv-cache-dtype turboquant_4bit_nc \\
  --attention-backend TURBOQUANT \\
  --max-model-len ${MAX_MODEL_LEN} \\
  --max-num-seqs ${MAX_NUM_SEQS} \\
  --max-num-batched-tokens 8192 \\
  --enable-chunked-prefill \\
  --trust-remote-code \\
  --gpu-memory-utilization ${GPU_MEM_UTIL} \\
  --api-key ${API_KEY} \\
  --host 0.0.0.0 --port ${PORT} \\
  --disable-log-stats
INNER_EOF
chmod +x /tmp/sndr_default_launcher/run.sh

docker run -d --name "${CONTAINER_NAME}" \
  --gpus all --ipc=host -p ${PORT}:${PORT} \
  --entrypoint /tmp/sndr_default_launcher/run.sh \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_LOGGING_LEVEL=WARNING \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e SNDR_ENABLE_G4_60A_TQ_SLIDING_SPEC=1 \
  -e SNDR_ENABLE_G4_60B_TQ_ATTN_OVERLAY=1 \
  -e SNDR_ENABLE_G4_60C_TQ_DECODE_OVERLAY=1 \
  -e SNDR_ENABLE_G4_60D_TQ_STORE_OVERLAY=1 \
  -e SNDR_ENABLE_G4_60E_KV_CACHE_UTILS=1 \
  -e SNDR_ENABLE_G4_60G_TQ_DISPATCH=1 \
  -e SNDR_ENABLE_G4_60H_TQ_CONFIG_AUGMENT=1 \
  -e SNDR_ENABLE_G4_60K_TQ_ENGINE_CONFIG=1 \
  -e SNDR_ENABLE_G4_61_TQ_SHARED_WORKSPACE=1 \
  -e SNDR_ENABLE_G4_62_TQ_KERNEL_WARMUP=1 \
  -e SNDR_ENABLE_G4_31_TQ_DTYPE_PRESERVE=1 \
  -e SNDR_ENABLE_G4_32_TQ_VALIDATION_BYPASS=1 \
  -v /tmp/sndr_default_launcher:/tmp/sndr_default_launcher:ro \
  -v "${GENESIS_REPO}":"${GENESIS_REPO}":rw \
  -v "${MODEL_ROOT}":/models:ro \
  -v "${GENESIS_REPO}/vllm/sndr_core":"${TGT}/sndr_core":ro \
  -v "${OVL}/turboquant_attn.py":"${TGT}/v1/attention/backends/turboquant_attn.py":ro \
  -v "${OVL}/triton_turboquant_decode.py":"${TGT}/v1/attention/ops/triton_turboquant_decode.py":ro \
  -v "${OVL}/triton_turboquant_store.py":"${TGT}/v1/attention/ops/triton_turboquant_store.py":ro \
  -v "${OVL}/turboquant_config.py":"${TGT}/model_executor/layers/quantization/turboquant/config.py":ro \
  -v "${OVL}/kv_cache_interface.py":"${TGT}/v1/kv_cache_interface.py":ro \
  -v "${OVL}/kv_cache_utils.py":"${TGT}/v1/core/kv_cache_utils.py":ro \
  -v "${OVL}/single_type_kv_cache_manager.py":"${TGT}/v1/core/single_type_kv_cache_manager.py":ro \
  -v "${OVL}/block_pool.py":"${TGT}/v1/core/block_pool.py":ro \
  "${VLLM_IMAGE}"

echo "${CONTAINER_NAME} on port ${PORT} (TQ-only, MTP OFF — production-safe)"

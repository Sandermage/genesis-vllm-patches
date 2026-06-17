#!/usr/bin/env bash
# Final restart — ALL JIT warmup patches enabled:
#   - PN126: V1 decode kernel warmup orchestrator
#   - PN127: Qwen 3.5/3.6 chat-template auto-install
#   - PN128: eagle helper kernel warmup (vllm#41481, 4 kernels)
#   - PN129: V1 slot mapping warmup (vllm#42165, 1 kernel + do_not_specialize)
#   - PN130: TurboQuant decode kernel warmup (vllm#42215, 1 kernel + workspace prealloc)
# All on the nightly bf610c2f (dev371) pin + 280K context.
#
# Configure paths via env vars before invoking (no hardcoded operator paths):
#   GENESIS_REPO_ROOT  — path to the genesis-vllm-patches checkout (default: $HOME/genesis-vllm-patches)
#   GENESIS_MODELS_DIR — model weight root (default: /nfs/genesis/models)
#   HF_HOME            — HuggingFace cache (default: $HOME/.cache/huggingface)
#   TRITON_CACHE_DIR   — Triton kernel cache (default: $HOME/.cache/triton)
#   VLLM_COMPILE_CACHE — vLLM torch.compile cache (default: $HOME/.cache/vllm/torch_compile_cache)
#
# Example override:
#   GENESIS_REPO_ROOT=/opt/genesis HF_HOME=/data/hf bash tools/restart_35b_dev371_multiconc.sh

set -euo pipefail

# ─── Operator-configurable paths (env-overridable, no hardcoded /home/user) ──
: "${GENESIS_REPO_ROOT:=${HOME}/genesis-vllm-patches}"
: "${GENESIS_MODELS_DIR:=/nfs/genesis/models}"
: "${HF_HOME:=${HOME}/.cache/huggingface}"
: "${TRITON_CACHE_DIR:=${HOME}/.cache/triton}"
: "${VLLM_COMPILE_CACHE:=${HOME}/.cache/vllm/torch_compile_cache}"

CONTAINER=vllm-qwen3.6-35b-a3b-fp8
IMAGE=vllm/vllm-openai:nightly-4c626633159887b0f2c962058c17c78f1434556d  # bumped dev371 -> 0.23.1 2026-06-17 (filename retained; retire-candidate — superseded by the YAML-rendered launcher)

echo "→ Pull/check image ${IMAGE}..."
docker pull "$IMAGE" 2>&1 | tail -3

echo "→ Stopping previous container (if alive)..."
docker rm -f $CONTAINER 2>/dev/null || true
sleep 2

echo "→ Starting with PN126+127+128+129+130 (Wave 10 stack)..."

docker run -d \
  --name "$CONTAINER" \
  --entrypoint /bin/bash \
  --gpus all \
  --shm-size=8g \
  --network genesis-vllm-patches_default \
  -p 8000:8000 \
  -v "${GENESIS_REPO_ROOT}/tools/genesis_vllm_plugin:/plugin:ro" \
  -v "${GENESIS_MODELS_DIR}:/models:ro" \
  -v "${HF_HOME}:/root/.cache/huggingface:ro" \
  -v "${TRITON_CACHE_DIR}:/root/.triton/cache" \
  -v "${VLLM_COMPILE_CACHE}:/root/.cache/vllm/torch_compile_cache" \
  -v "${GENESIS_REPO_ROOT}/vllm/sndr_core:/usr/local/lib/python3.12/dist-packages/vllm/sndr_core:ro" \
  -e SNDR_DEV_INSTALL_PLUGIN=1 \
  -e GENESIS_BUFFER_MODE=shared \
  -e GENESIS_ENABLE_P101=1 \
  -e GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1 \
  -e GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN=1 \
  -e GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR=1 \
  -e GENESIS_ENABLE_P37=1 \
  -e GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1 \
  -e GENESIS_ENABLE_P60B_TRITON_KERNEL=1 \
  -e GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1 \
  -e GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1 \
  -e GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT=1 \
  -e GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1 \
  -e GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1 \
  -e GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1 \
  -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1 \
  -e GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 \
  -e GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1 \
  -e GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1 \
  -e GENESIS_ENABLE_P71_BLOCK_VERIFY=1 \
  -e GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 \
  -e GENESIS_ENABLE_P74_CHUNK_CLAMP=1 \
  -e GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1 \
  -e GENESIS_ENABLE_P82=1 \
  -e GENESIS_ENABLE_P87=1 \
  -e GENESIS_ENABLE_P91=1 \
  -e GENESIS_ENABLE_P95=1 \
  -e GENESIS_ENABLE_P99=1 \
  -e GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT=0 \
  -e GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT=0 \
  -e GENESIS_ENABLE_PN106_GDN_H_POOL=1 \
  -e GENESIS_ENABLE_PN108_FUSED_RECURRENT_PREFILL=0 \
  -e GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1 \
  -e GENESIS_ENABLE_PN16_LAZY_REASONER=1 \
  -e GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1 \
  -e GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT=1 \
  -e GENESIS_ENABLE_PN201_SCHEDULER_EMPTY_CACHE=1 \
  -e GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ=1 \
  -e GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION=1 \
  -e GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK=1 \
  -e GENESIS_ENABLE_PN66=1 \
  -e GENESIS_ENABLE_PN67=1 \
  -e GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE=1 \
  -e GENESIS_ENABLE_PN73_TOOL_ARGS_SAFE_NORMALIZE=1 \
  -e GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED=1 \
  -e GENESIS_ENABLE_PN77_FP8_LM_HEAD=1 \
  -e GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1 \
  -e GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT=1 \
  -e GENESIS_ENABLE_PN91_DEVELOPER_ROLE=1 \
  -e GENESIS_ENABLE_PN92_NIXL_EP_TRIAL_IMPORT=1 \
  -e GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1 \
  -e GENESIS_ENABLE_PN96_EMERGENCY_DEMOTE=1 \
  -e GENESIS_ENABLE_PN96B=1 \
  -e GENESIS_ENABLE_SNDR_WORKSPACE_001=1 \
  -e GENESIS_OBSERVABILITY=1 \
  -e GENESIS_P67_NUM_KV_SPLITS=48 \
  -e GENESIS_P67_USE_UPSTREAM=1 \
  -e GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS=50000 \
  -e GENESIS_P82_THRESHOLD_SINGLE=0.3 \
  -e GENESIS_PN16_CLASSIFIER_MAX_TOKENS=0 \
  -e GENESIS_PN16_TOOL_THINK_BUDGET=200 \
  -e GENESIS_PN95_CONFIG_KEY=a5000-2x-tier-aware-example \
  -e GENESIS_PN95_DEMOTE_FREE_MIB_THRESHOLD=2048 \
  -e GENESIS_PN95_TICK_EVERY=100 \
  -e GENESIS_PREALLOC_TOKEN_BUDGET=4096 \
  -e GENESIS_PROFILE_RUN_CAP_M=4096 \
  -e GENESIS_TQ_MAX_MODEL_LEN=280000 \
  -e GENESIS_ENABLE_PN126_V1_DECODE_WARMUP=1 \
  -e GENESIS_ENABLE_PN127_AUTO_CHAT_TEMPLATE=1 \
  -e GENESIS_ENABLE_PN128_SPEC_DECODE_WARMUP=1 \
  -e GENESIS_ENABLE_PN129_SLOT_MAPPING_WARMUP=1 \
  -e GENESIS_ENABLE_PN130_TQ_DECODE_WARMUP=1 \
  -e GENESIS_ENABLE_PN132_TOPK_TOPP_CONTIGUOUS=1 \
  -e GENESIS_LEGACY_P7=0 \
  -e GENESIS_ENABLE_PN133_MTP_EMPTY_OUTPUT_FIX=1 \
  -e GENESIS_ENABLE_PN125_HYBRID_FULL_AND_PIECEWISE=1 \
  -e GENESIS_ENABLE_PN50_GDN_FUSED_PROJ=1 \
  -e CUDA_MODULE_LOADING=LAZY \
  -e VLLM_LOG_MODEL_INSPECTION=1 \
  -e NCCL_DEBUG=WARN \
  -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e NCCL_P2P_DISABLE=1 \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:256' \
  -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -e VLLM_LOGGING_LEVEL=WARNING \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_MOE_USE_DEEP_GEMM=0 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP8=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "$IMAGE" \
  -c 'set -euo pipefail
if [ "${SNDR_DEV_INSTALL_PLUGIN:-0}" = "1" ]; then
  cp -r /plugin /tmp/genesis_vllm_plugin && pip install --quiet --disable-pip-version-check --root-user-action=ignore --no-deps -e /tmp/genesis_vllm_plugin 2>&1 | tail -2
fi
python3 -m vllm.sndr_core.apply 2>&1 | tail -5
exec vllm serve --model /models/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.9 \
  --max-model-len 280000 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --dtype float16 \
  --kv-cache-dtype turboquant_k8v4 --served-model-name qwen3.6-35b-a3b \
  --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-chunked-prefill --no-scheduler-reserve-full-isl --disable-custom-all-reduce \
  --language-model-only --trust-remote-code --enable-auto-tool-choice \
  --api-key genesis-local --host 0.0.0.0 --port 8000 \
  --speculative-config '"'"'{"method": "mtp", "num_speculative_tokens": 3}'"'"' \
  --chat-template /tmp/genesis/chat_templates/qwen3.6_enhanced.jinja'

echo "-> Container started. Waiting for API readiness..."
for i in $(seq 1 90); do
  if curl -fsS -m 2 -H "Authorization: Bearer genesis-local" http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "-> API ready after ${i}x 5s = $((i*5))s"
    exit 0
  fi
  sleep 5
done
echo "→ failed to become ready in 450s"
docker logs --tail 30 "$CONTAINER"
exit 1

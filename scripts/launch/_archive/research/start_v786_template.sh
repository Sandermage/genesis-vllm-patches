#!/bin/bash
# v786 TEMPLATE — base for Phase 1/2/3 A/B testing
#
# Inherits v775 PROD baseline (162 TPS, CV 3.61%, 50/50 stability).
# All NEW Phase toggles are COMMENTED OUT — copy this file and uncomment
# only the line(s) for the arm you want to test.
#
# Phase 1 (prefix-cache + P83/P85 A/B on 35B PROD):
#   Arm A = this file as-is (refresh baseline measurement)
#   Arm B = remove `--enable-prefix-caching` from serve args
#   Arm C = uncomment GENESIS_ENABLE_P83 + GENESIS_ENABLE_P85 below
#
# Phase 2 (stack winner + P40 GQA grouping kernel):
#   Arm D = winner of Phase 1 + uncomment GENESIS_ENABLE_P40 below
#
# Phase 3 (INT8 27B sprint resume with PN8 + PN9):
#   Use v788 INT8 launch scripts; add PN8/PN9 env vars there.
#
# Snapshot before each arm:
#   bash scripts/launch/snapshot_pre_arm.sh <arm_name>
set -euo pipefail
docker stop vllm-server-mtp-test 2>/dev/null || true
docker rm vllm-server-mtp-test 2>/dev/null || true

docker run -d \
  --name vllm-server-mtp-test \
  --network genesis-vllm-patches_default \
  --shm-size=8g --memory=64g -p 8000:8000 --gpus all \
  --security-opt label=disable --entrypoint /bin/bash \
  -v /nfs/genesis/models:/models:ro \
  -v /home/sander/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /home/sander/Genesis_Project/vllm_engine/triton-cache-mtp-test:/root/.triton/cache \
  -v /home/sander/Genesis_Project/vllm_engine/compile-cache-prod-mirror-test:/root/.cache/vllm/torch_compile_cache \
  -v /home/sander/genesis-vllm-patches/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -v /home/sander/genesis-vllm-patches/genesis_vllm_plugin:/plugin:ro \
  -v /home/sander/genesis-vllm-patches/external_probe:/external_probe:ro \
  -v "/home/sander/genesis-vllm-patches/vllm/_genesis/configs/moe_tuning/E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_RTX_A5000,dtype=fp8_w8a8,block_shape=[128,128].json:ro" \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 -e VLLM_NO_USAGE_STATS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6" \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high -e NCCL_P2P_DISABLE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e OMP_NUM_THREADS=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_MOE_USE_DEEP_GEMM=0 -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP8=0 -e VLLM_LOGGING_LEVEL=WARNING \
  -e GENESIS_ENABLE_P56_SPEC_DECODE_GUARD=0 -e GENESIS_ENABLE_P57_SPEC_DECODE_CAPTURE_SAFE=0 \
  -e GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1 -e GENESIS_ENABLE_P59_QWEN3_TOOL_RECOVERY=0 \
  -e GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1 -e GENESIS_ENABLE_P60B_TRITON_KERNEL=1 \
  -e GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1 -e GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1 \
  -e GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1 -e GENESIS_ENABLE_P63_MTP_GDN_STATE_RECOVERY=0 \
  -e GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1 -e GENESIS_ENABLE_P65_TURBOQUANT_SPEC_CG_DOWNGRADE=0 \
  -e GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1 -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1 \
  -e GENESIS_P67_USE_UPSTREAM=1 -e GENESIS_P67_NUM_KV_SPLITS=32 \
  -e GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 -e GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1 \
  -e GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1 -e GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS=8000 \
  -e GENESIS_ENABLE_P37=1 -e GENESIS_TQ_MAX_MODEL_LEN=320000 \
  -e GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 -e GENESIS_PROFILE_RUN_CAP_M=4096 \
  -e GENESIS_ENABLE_P74_CHUNK_CLAMP=1 \
  -e GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC=0 -e GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP=0 \
  -e GENESIS_ENABLE_P79D_PREEMPT_ASYNC_DISCARD=0 \
  -e GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1 -e GENESIS_ENABLE_P82=1 \
  -e GENESIS_ENABLE_P99=1 -e GENESIS_ENABLE_P101=1 \
  -e GENESIS_P82_THRESHOLD_SINGLE=0.3 \
  -e GENESIS_PREALLOC_TOKEN_BUDGET=4096 -e GENESIS_BUFFER_MODE=shared \
  \
  ` # ============== PHASE 1 toggles (uncomment for Arm C only) ============== ` \
  ` # -e GENESIS_ENABLE_P83=1 ` \
  ` # -e GENESIS_ENABLE_P85=1 ` \
  \
  ` # ============== PHASE 2 toggle (uncomment for Arm D only) =============== ` \
  ` # -e GENESIS_ENABLE_P40=1 ` \
  \
  ` # ============== PHASE 3 toggles (INT8 sprint, do NOT use on 35B) ======== ` \
  ` # -e GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1 ` \
  ` # -e GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN=1 ` \
  ` # -e GENESIS_PN9_DRAFTER_BACKEND=auto ` \
  \
  vllm/vllm-openai:nightly -c \
  "set -e; echo \"=== v786 TEMPLATE (= v775 baseline; copy + uncomment per Phase) ===\"; \
pip install --quiet --disable-pip-version-check pandas scipy xxhash; \
cp -r /plugin /tmp/genesis_vllm_plugin; \
pip install --quiet --disable-pip-version-check --no-deps -e /tmp/genesis_vllm_plugin 2>&1 | tail -3; \
python3 /external_probe/patch_tolist_cudagraph.py || echo tolist bypass failed; \
python3 /external_probe/patch_40074_iooo.py || echo PR40074 failed; \
python3 -m vllm._genesis.patches.apply_all ; \
exec vllm serve --model /models/Qwen3.6-35B-A3B-FP8 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 320000 \
  --kv-cache-dtype turboquant_k8v4 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --dtype float16 \
  --disable-custom-all-reduce --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --api-key genesis-local --served-model-name qwen3.6-35b-a3b --host 0.0.0.0 \
  --enable-prefix-caching \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
  --performance-mode interactivity --attention-config.flash_attn_version 2 --port 8000 \
  --no-scheduler-reserve-full-isl --disable-log-stats"
sleep 5
docker logs --tail 5 vllm-server-mtp-test 2>&1 | sed "s/^/  /"
echo "v786 TEMPLATE container started."

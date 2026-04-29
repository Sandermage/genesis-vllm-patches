#!/bin/bash
# v788: INT8 27B GS128 + MTP K=3 + P87 + P91 + PN8 + PN9 (+ optional P40 if Phase 2 winning)
#
# Resume INT8 27B sprint per locked priority. Builds on v764d (untested original).
# Target model: Minachist Qwen3.6-27B-INT8-gs128 (group_size=128 → Marlin path
# where P87/P91 fire). Baseline target 86 → 121 t/s per recipe.
#
# Phase 3 additions:
#   + GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1   (saves ~600MB on draft worker)
#   + GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN=1 (drafter auto-selects backend)
#   + GENESIS_PN9_DRAFTER_BACKEND=auto              (or FLASH_ATTN if needed)
#
# Phase 2 winner toggle (uncomment if Phase 2 confirmed P40 +TPS win):
#   # -e GENESIS_ENABLE_P40=1                       (TQ k8v4 GQA grouping kernel)
#
# Run order:
#   1. bash scripts/launch/snapshot_pre_arm.sh phase3_v788_int8
#   2. ssh sander@192.168.1.10 'bash -s' < scripts/launch/start_v788_int8_27b_pn8_pn9.sh
#   3. wait ~3 min for boot, verify /v1/models reachable
#   4. bash tools/phase1_test_harness.sh phase3_v788_int8
#      (model name: qwen3.6-27b)
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
  -v /home/sander/Genesis_Project/vllm_engine/triton-cache-int8-gs128:/root/.triton/cache \
  -v /home/sander/Genesis_Project/vllm_engine/compile-cache-int8-gs128:/root/.cache/vllm/torch_compile_cache \
  -v /home/sander/genesis-vllm-patches/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -v /home/sander/genesis-vllm-patches/genesis_vllm_plugin:/plugin:ro \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_LOGGING_LEVEL=INFO \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256" \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high -e NCCL_P2P_DISABLE=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e OMP_NUM_THREADS=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e GENESIS_ENABLE_P7B=1 \
  -e GENESIS_ENABLE_P87=1 \
  -e GENESIS_ENABLE_P91=1 \
  -e GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1 \
  -e GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1 -e GENESIS_ENABLE_P60B_TRITON_KERNEL=1 \
  -e GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1 -e GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1 \
  -e GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1 \
  -e GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1 \
  -e GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1 \
  -e GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 -e GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1 \
  -e GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS=8000 \
  -e GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=0 \
  -e GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 -e GENESIS_PROFILE_RUN_CAP_M=4096 \
  -e GENESIS_ENABLE_P74_CHUNK_CLAMP=1 \
  -e GENESIS_ENABLE_P82=1 -e GENESIS_P82_THRESHOLD_SINGLE=0.3 \
  -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=0 \
  -e GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=0 \
  -e GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=0 \
  -e GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1 \
  -e GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN=1 \
  -e GENESIS_PN9_DRAFTER_BACKEND=auto \
  -e GENESIS_PREALLOC_TOKEN_BUDGET=4096 -e GENESIS_BUFFER_MODE=shared \
  vllm/vllm-openai:nightly -c \
  "set -e; echo === v788 INT8 27B GS128 + MTP + P87 + P91 + PN8 + PN9 ===; \
pip install --quiet --disable-pip-version-check pandas scipy xxhash; \
cp -r /plugin /tmp/genesis_vllm_plugin; \
pip install --quiet --disable-pip-version-check --no-deps -e /tmp/genesis_vllm_plugin 2>&1 | tail -3; \
python3 -m vllm._genesis.patches.apply_all ; \
exec vllm serve --model /models/Qwen3.6-27B-INT8-gs128 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 --max-model-len 131072 \
  --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --dtype float16 \
  --disable-custom-all-reduce --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --api-key genesis-local --served-model-name qwen3.6-27b \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
  --host 0.0.0.0 --port 8000 --disable-log-stats"
sleep 5
docker logs --tail 5 vllm-server-mtp-test 2>&1 | sed "s/^/  /"
echo "v788 INT8 GS128 + PN8 + PN9 container started."

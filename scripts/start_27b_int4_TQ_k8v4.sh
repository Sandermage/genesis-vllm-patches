#!/bin/bash
# v791: Lorbus INT4 + LONG-CONTEXT config + PN8 test
#
# Targets 128K context window (v771b OOMed at 16K with util=0.95).
# Changes from v771b:
#   - --gpu-memory-utilization 0.95 → 0.85 (frees ~2.4 GB headroom)
#   - --max-num-seqs 4 → 2 (halves KV pool footprint)
#   - --max-num-batched-tokens 8192 → 2048 (smaller chunked-prefill chunks)
#   + GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1 (verify if PN8 fires on INT4 AutoRound — it might no-op since AutoRound is offline-quant)
#
# Test plan:
#   1. Boot, verify GPU memory < 21 GB after load (vs 22.69 GB v771b)
#   2. /v1/models reachable
#   3. Sequential context-size scan: 16K → 32K → 64K → 96K → 128K (200 tokens each)
#   4. If 128K passes, escalate: 160K → 192K → 256K
#   5. Document max stable context for reference config
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
  -v /home/sander/Genesis_Project/vllm_engine/triton-cache-int4-mtp:/root/.triton/cache \
  -v /home/sander/Genesis_Project/vllm_engine/compile-cache-int4-mtp:/root/.cache/vllm/torch_compile_cache \
  -v /home/sander/genesis-vllm-patches/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -v /home/sander/genesis-vllm-patches/genesis_vllm_plugin:/plugin:ro \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_LOGGING_LEVEL=WARNING \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e NCCL_P2P_DISABLE=1 -e NCCL_CUMEM_ENABLE=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e OMP_NUM_THREADS=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1 \
  -e GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1 -e GENESIS_ENABLE_P60B_TRITON_KERNEL=1 \
  -e GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1 -e GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1 \
  -e GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1 \
  -e GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1 \
  -e GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1 \
  -e GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 -e GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1 \
  -e GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 -e GENESIS_PROFILE_RUN_CAP_M=4096 \
  -e GENESIS_ENABLE_P74_CHUNK_CLAMP=1 \
  -e GENESIS_ENABLE_P82=0 -e GENESIS_ENABLE_P98=1 -e GENESIS_ENABLE_P99=1 -e GENESIS_P82_THRESHOLD_SINGLE=0.3 \
  -e GENESIS_ENABLE_PN26_SPARSE_V=0 -e GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE=1 \
  -e GENESIS_ENABLE_P38B_COMPILE_SAFE=1 -e GENESIS_ENABLE_P15B_FA_VARLEN_CLAMP=1 -e GENESIS_ENABLE_PN28_MERGE_ATTN_NAN_GUARD=1 \
  -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1 -e GENESIS_ENABLE_P91=1 -e GENESIS_ENABLE_P87=1 -e GENESIS_ENABLE_P85=1 -e GENESIS_ENABLE_P83=1 -e GENESIS_ENABLE_P101=1 -e GENESIS_ENABLE_P100=1 \
  -e GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=0 \
  -e GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=0 \
  -e GENESIS_PREALLOC_TOKEN_BUDGET=4096 -e GENESIS_BUFFER_MODE=shared \
  -e GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1 \
  -e GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1 \
  -e GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1 \
  -e GENESIS_ENABLE_PN13_CUDA_GRAPH_LAMBDA_ARITY=1 \
  -e GENESIS_ENABLE_P94=1 \
  -e GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1 \
  -e GENESIS_ENABLE_PN9_INDEPENDENT_DRAFTER_ATTN=1 \
  -e GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1 \
  -e GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT=1 -e GENESIS_ENABLE_PN22_LOCAL_ARGMAX_TP=1 -e GENESIS_ENABLE_P103=1 \
  vllm/vllm-openai:nightly -c \
  "set -e; echo === v771b 27B Lorbus INT4 NO-prefix-cache MTP K=3 ===; \
pip install --quiet --disable-pip-version-check pandas scipy xxhash; \
cp -r /plugin /tmp/genesis_vllm_plugin; \
pip install --quiet --disable-pip-version-check --no-deps -e /tmp/genesis_vllm_plugin 2>&1 | tail -3; \
python3 -m vllm._genesis.patches.apply_all ; \
exec vllm serve --model /models/Qwen3.6-27B-int4-AutoRound --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 280000 \
  --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --enable-chunked-prefill --dtype float16 \
  --kv-cache-dtype turboquant_k8v4 \
  --disable-custom-all-reduce --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --api-key genesis-local --served-model-name qwen3.6-27b \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
  --host 0.0.0.0 --port 8000 --disable-log-stats"
sleep 5
docker logs --tail 5 vllm-server-mtp-test 2>&1 | sed "s/^/  /"
echo "v763 Lorbus INT4 + MTP container started."

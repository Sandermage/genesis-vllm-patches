#!/bin/bash
# 27B Lorbus INT4 + TurboQuant k8v4 + NGRAM strict spec-decode (variant 3)
# ════════════════════════════════════════════════════════════════════════
#
# Alternative spec-decode strategy vs MTP K=3:
#   - method=ngram with prompt_lookup_default closes the #40875 acceptance bug
#     class (Genesis v7.13 strict-ngram breakthrough — 100% clean rate single-
#     query, 96% multi-query diverse vs ~20% baseline).
#   - No GDN draft module → bypasses #40807 / #40880 entire bug class.
#   - Bonus: ngram is config-only, no model weights for draft, no hidden state
#     propagation — cleaner cudagraph capture path.
#
# Expected behaviour:
#   - TPS: comparable to MTP K=3 in code-completion / repetitive workloads
#     where ngram acceptance is high; lower in pure-creative free-form text.
#   - Tool-call: 7/7 PASS (ngram acceptance cliff fixes the cascade).
#   - Tradeoff: less universal than MTP — relies on prompt repetition.
#
# Reference: project_genesis_v7_13_strict_ngram_breakthrough.md
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
  -v /home/sander/Genesis_Project/vllm_engine/triton-cache-int4-ngram:/root/.triton/cache \
  -v /home/sander/Genesis_Project/vllm_engine/compile-cache-int4-ngram:/root/.cache/vllm/torch_compile_cache \
  -v /home/sander/genesis-vllm-patches/vllm/_genesis:/usr/local/lib/python3.12/dist-packages/vllm/_genesis:ro \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_LOGGING_LEVEL=WARNING \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high -e VLLM_SSM_CONV_STATE_LAYOUT=DS -e NCCL_P2P_DISABLE=1 -e NCCL_CUMEM_ENABLE=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e OMP_NUM_THREADS=1 -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1 \
  -e GENESIS_ENABLE_P61_QWEN3_MULTI_TOOL=1 -e GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1 \
  -e GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1 \
  -e GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1 \
  -e GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1 \
  -e GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 -e GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1 \
  -e GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 -e GENESIS_PROFILE_RUN_CAP_M=4096 \
  -e GENESIS_ENABLE_P74_CHUNK_CLAMP=1 \
  -e GENESIS_ENABLE_P98=1 -e GENESIS_ENABLE_P99=1 \
  -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1 -e GENESIS_ENABLE_P91=1 -e GENESIS_ENABLE_P87=1 -e GENESIS_ENABLE_P101=1 -e GENESIS_ENABLE_P100=1 \
  -e GENESIS_ENABLE_P78_TOLIST_CAPTURE_GUARD=0 \
  -e GENESIS_PREALLOC_TOKEN_BUDGET=4096 -e GENESIS_BUFFER_MODE=shared \
  -e GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1 \
  vllm/vllm-openai:nightly -c \
  "set -e; echo === 27B Lorbus INT4 + TQ k8v4 + NGRAM strict prompt_lookup_default ===; \
pip install --quiet --disable-pip-version-check pandas scipy xxhash; \
python3 -m vllm._genesis.patches.apply_all ; \
exec vllm serve --model /models/Qwen3.6-27B-int4-AutoRound --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 --max-model-len 280000 \
  --max-num-seqs 2 --max-num-batched-tokens 2048 \
  --enable-chunked-prefill --dtype float16 \
  --kv-cache-dtype turboquant_k8v4 \
  --disable-custom-all-reduce --language-model-only --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --api-key genesis-local --served-model-name qwen3.6-27b \
  --speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_min\":2,\"prompt_lookup_max\":5}' \
  --host 0.0.0.0 --port 8000 --disable-log-stats"
sleep 5
docker logs --tail 5 vllm-server-mtp-test 2>&1 | sed 's/^/  /'
echo '27B Lorbus INT4 + TQ k8v4 + NGRAM strict container started.'

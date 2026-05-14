#!/bin/bash
# ────────────────────────────────────────────────────────────────────────
# Genesis 2× A5000 (TP=2) launcher
#   model    : Qwen3.6-27B-int4-AutoRound (Lorbus)
#   KV       : TurboQuant k8v4
#   image    : vllm/vllm-openai:nightly-dcacdf9a8860a86401127d1c8f93ebf3cfbfd026
#   context  : 200K-256K (set --max-model-len below)
#   spec     : MTP K=3
#   patches  : full Genesis Wave 9 stack including PN59 (anchor-fixed)
#              streaming GDN, PN95 tier-aware KV, PN106/200/201 GDN
#              scratch pool, PN77 fp8 lm_head, PN204 dual-stream input
#              projection (port of vllm#42301), etc.
#
# Why this script exists:
#   The legacy `start_pn95_2xa5000_test.sh` was pinned to an older
#   nightly without the new chat/completion split refactor. This
#   variant targets the dcacdf9a pin with the env flags actually
#   verified on hardware during the 2026-05-14 TP=2 200K ladder.
#
# Verified on: 2× NVIDIA RTX A5000 24 GB.
# Bench: progressive 50K-200K all PASS (see
#   docs/_internal/runs/tp2_progressive_probe.json).
#
# Env overrides (set via the operator's host.env or shell rc):
#   SNDR_REPO_ROOT     repo root with vllm/sndr_core (default $HOME/genesis-vllm-patches-v11)
#   SNDR_CACHE_ROOT    triton + compile cache root  (default $HOME/Genesis_Project/vllm_engine)
#   SNDR_HF_CACHE      HuggingFace cache dir         (default $HOME/.cache/huggingface)
#   SNDR_RUNTIME_CACHE genesis runtime cache         (default $HOME/.sndr/cache)
#   SNDR_MODELS_DIR    model dir mounted as /models  (default /nfs/genesis/models)
SNDR_REPO_ROOT="${SNDR_REPO_ROOT:-${HOME}/genesis-vllm-patches-v11}"
SNDR_CACHE_ROOT="${SNDR_CACHE_ROOT:-${HOME}/Genesis_Project/vllm_engine}"
SNDR_HF_CACHE="${SNDR_HF_CACHE:-${HOME}/.cache/huggingface}"
SNDR_RUNTIME_CACHE="${SNDR_RUNTIME_CACHE:-${HOME}/.sndr/cache}"
SNDR_MODELS_DIR="${SNDR_MODELS_DIR:-/nfs/genesis/models}"
#
# Credit:
#   Genesis stack — Sandermage / Sander Barzov Aleksandr, Odessa.
#   Env-flag set tuned by the same author during the 2026-05-14 TP=2
#   commissioning ladder.
# ────────────────────────────────────────────────────────────────────────
set -e

NAME=vllm-pn95-2xa5000
docker stop $NAME 2>/dev/null || true
docker rm $NAME 2>/dev/null || true
docker stop vllm-pn95-1xa5000 2>/dev/null || true
docker rm vllm-pn95-1xa5000 2>/dev/null || true

mkdir -p "${SNDR_CACHE_ROOT}/compile-cache-pn95-2x" \
         "${SNDR_CACHE_ROOT}/triton-cache-pn95-2x"

docker run -d --name $NAME \
  --gpus all \
  --network host --shm-size=8g --ipc=host \
  --entrypoint /bin/bash \
  --memory=64g \
  --security-opt label=disable \
  --ulimit memlock=-1 \
  \
  -e CUDA_DEVICE_MAX_CONNECTIONS=8 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_P2P_DISABLE=1 \
  -e OMP_NUM_THREADS=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
  -e TRITON_CACHE_DIR=/root/.triton/cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -e VLLM_LOGGING_LEVEL=WARNING \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  \
  -e GENESIS_BUFFER_MODE=shared \
  -e GENESIS_OBSERVABILITY=1 \
  -e GENESIS_P67_USE_UPSTREAM=1 \
  -e GENESIS_P67_NUM_KV_SPLITS=32 \
  -e GENESIS_P68_P69_LONG_CTX_THRESHOLD_CHARS=50000 \
  -e GENESIS_PN16_CLASSIFIER_MAX_TOKENS=0 \
  -e GENESIS_PN16_TOOL_THINK_BUDGET=200 \
  -e GENESIS_PREALLOC_TOKEN_BUDGET=4096 \
  -e GENESIS_PROFILE_RUN_CAP_M=4096 \
  -e GENESIS_TQ_MAX_MODEL_LEN=200000 \
  \
  -e GENESIS_ENABLE_P58_ASYNC_PLACEHOLDER_FIX=1 \
  -e GENESIS_ENABLE_P60_GDN_NGRAM_FIX=1 \
  -e GENESIS_ENABLE_P60B_TRITON_KERNEL=1 \
  -e GENESIS_ENABLE_P61B_STREAMING_OVERLAP=1 \
  -e GENESIS_ENABLE_P61C_QWEN3CODER_DEFERRED_COMMIT=1 \
  -e GENESIS_ENABLE_P62_STRUCT_OUT_SPEC_TIMING=1 \
  -e GENESIS_ENABLE_P64_QWEN3CODER_MTP_STREAMING=1 \
  -e GENESIS_ENABLE_P66_CUDAGRAPH_SIZE_FILTER=1 \
  -e GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL=1 \
  -e GENESIS_ENABLE_P68_AUTO_FORCE_TOOL=1 \
  -e GENESIS_ENABLE_P69_LONG_CTX_TOOL_REMINDER=1 \
  -e GENESIS_ENABLE_P70_AUTO_STRICT_NGRAM=1 \
  -e GENESIS_ENABLE_P72_PROFILE_RUN_CAP=1 \
  -e GENESIS_ENABLE_P74_CHUNK_CLAMP=1 \
  -e GENESIS_ENABLE_P82=1 \
  -e GENESIS_P82_THRESHOLD_SINGLE=0.1 \
  -e GENESIS_ENABLE_P87=1 \
  -e GENESIS_ENABLE_P91=1 \
  -e GENESIS_ENABLE_P94=1 \
  -e GENESIS_ENABLE_P95=1 \
  -e GENESIS_ENABLE_P98=1 \
  -e GENESIS_ENABLE_P99=1 \
  -e GENESIS_ENABLE_P101=1 \
  -e GENESIS_ENABLE_P103=1 \
  -e GENESIS_ENABLE_P107_MTP_TRUNCATION_DETECTOR=1 \
  -e GENESIS_ENABLE_PN8_MTP_DRAFT_ONLINE_QUANT=1 \
  -e GENESIS_ENABLE_PN11_GDN_AB_CONTIGUOUS=1 \
  -e GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1 \
  -e GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1 \
  -e GENESIS_ENABLE_PN16_LAZY_REASONER=1 \
  -e GENESIS_ENABLE_PN17_FA2_LSE_CLAMP=1 \
  -e GENESIS_ENABLE_PN19_SCOPED_MAX_SPLIT=1 \
  -e GENESIS_ENABLE_PN25_SILU_INDUCTOR_SAFE=1 \
  -e GENESIS_ENABLE_PN30_DS_LAYOUT_SPEC_DECODE=1 \
  -e GENESIS_ENABLE_PN52_PROMPT_LOGPROBS_EVICTION=1 \
  -e GENESIS_ENABLE_PN56_QWEN3CODER_XML_FALLBACK=1 \
  -e GENESIS_ENABLE_PN59_STREAMING_GDN=1 \
  -e GENESIS_PN59_STREAMING_GDN_THRESHOLD=2048 \
  -e GENESIS_ENABLE_PN66=1 \
  -e GENESIS_ENABLE_PN67=1 \
  -e GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE=1 \
  -e GENESIS_ENABLE_PN73_TOOL_ARGS_SAFE_NORMALIZE=1 \
  -e GENESIS_ENABLE_PN77_FP8_LM_HEAD=1 \
  -e GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO=1 \
  -e GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT=1 \
  -e GENESIS_ENABLE_PN91_DEVELOPER_ROLE=1 \
  -e GENESIS_ENABLE_PN92_NIXL_EP_TRIAL_IMPORT=1 \
  -e GENESIS_ENABLE_PN95_TIER_AWARE_CACHE=1 \
  -e GENESIS_PN95_CONFIG_KEY=a5000-2x-tier-aware-pn95 \
  -e GENESIS_PN95_TICK_EVERY=10 \
  -e GENESIS_PN95_DEMOTE_FREE_MIB_THRESHOLD=2048 \
  -e GENESIS_PN95_PROACTIVE_FREE_BLOCKS_THRESHOLD=32 \
  -e GENESIS_PN95_VIRT_ENABLE=0 \
  -e GENESIS_ENABLE_PN96_EMERGENCY_DEMOTE=1 \
  -e GENESIS_ENABLE_PN104_OFFLOAD_PREFETCH_REDIRECT=0 \
  -e GENESIS_ENABLE_PN105_AUTOROUND_OFFLOAD_COMPAT=0 \
  -e GENESIS_ENABLE_PN106_GDN_H_POOL=1 \
  -e GENESIS_PN106_EMPTY_CACHE_EVERY_N_TICKS=100 \
  -e GENESIS_ENABLE_PN108_FUSED_RECURRENT_PREFILL=0 \
  -e GENESIS_ENABLE_PN200_GDN_SCRATCH_REUSE=1 \
  -e GENESIS_ENABLE_PN201_SCHEDULER_EMPTY_CACHE=1 \
  -e GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ=0 \
  -e GENESIS_PN201_EMPTY_CACHE_FREE_BLOCKS_THRESHOLD=8 \
  -e GENESIS_PN201_EMPTY_CACHE_COOLDOWN=50 \
  -e GENESIS_ENABLE_SNDR_WORKSPACE_001=1 \
  \
  -v "${SNDR_REPO_ROOT}/vllm/sndr_core:/usr/local/lib/python3.12/dist-packages/vllm/sndr_core" \
  -v "${SNDR_MODELS_DIR}:/models" \
  -v "${SNDR_HF_CACHE}:/root/.cache/huggingface" \
  -v "${SNDR_CACHE_ROOT}/compile-cache-pn95-2x:/root/.cache/vllm/torch_compile_cache" \
  -v "${SNDR_CACHE_ROOT}/triton-cache-pn95-2x:/root/.triton/cache" \
  -v "${SNDR_RUNTIME_CACHE}:/root/.sndr/cache" \
  vllm/vllm-openai:nightly-dcacdf9a8860a86401127d1c8f93ebf3cfbfd026 \
  -c "
set -e
pip install --quiet --disable-pip-version-check --root-user-action=ignore pandas==2.2.3 scipy==1.14.1 xxhash==3.5.0 pyyaml packaging zstandard==0.23.0 2>&1 | tail -2
python3 -m vllm.sndr_core.apply 2>&1 | tee /tmp/genesis_boot.log | tail -25
exec vllm serve \
  --model /models/Qwen3.6-27B-int4-AutoRound \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 262144 \
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
  --chat-template /usr/local/lib/python3.12/dist-packages/vllm/sndr_core/assets/chat_templates/qwen3.6-froggeric.jinja \
  --api-key genesis-local \
  --served-model-name qwen3.6-27b \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}' \
  --host 0.0.0.0 \
  --port 8101 \
  --disable-log-stats
"
echo "── 2×A5000 TP=2 with PN59 anchor-fixed + full Genesis stack ──"
docker ps --filter name=$NAME --format "{{.Names}}  {{.Status}}"

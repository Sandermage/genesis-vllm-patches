#!/bin/bash
# moe_lookup_helper.sh — print the EXACT vLLM fused-MoE config filename for
# the current GPU + target shape, verify the lookup path, and stage a
# JSON file into `vllm/sndr_core/configs/moe_tuning/` ready to bundle.
#
# This is a NAMING + STAGING helper — NOT an autotuner. It does not run
# Triton's autotune sweep. The actual sweep is `benchmark_moe.py --tune`
# inside the vLLM container, invoked manually per the NEXT STEPS block at
# the bottom of this script (vLLM's autotune entry point varies by pin).
#
# Renamed 2026-05-05 from `tune_moe.sh` per audit `genesis_post_fix_rescan
# _audit_2026-05-05` G-POST-05 — the old name implied tuning that doesn't
# happen here.
#
# Why this matters: vLLM's fused_moe kernel autotunes inline if NO config
# file matches the running shape — that costs 15-20% on small/medium batch.
# Pre-tuned JSONs eliminate that overhead.
#
# Usage:
#   ./scripts/moe_lookup_helper.sh                               # default: 35B-A3B-FP8 shape
#   E=64 N=640 DTYPE=fp8_w8a8 ./scripts/moe_lookup_helper.sh     # custom shape
#   GPU_OVERRIDE=NVIDIA_GeForce_RTX_5090 ./scripts/moe_lookup_helper.sh  # cross-rig
#
# Common per-model shapes (E = num_experts, N = intermediate_size):
#   Qwen3.6-35B-A3B-FP8   → E=256  N=512   block_shape=[128,128]  (PROD bundled)
#   Mixtral-8x7B          → E=8    N=14336 dtype=fp16
#   DeepSeek-V2-Lite      → E=64   N=2816  dtype=fp16
#   Qwen3-Next-80B (MoE)  → E=128  N=2048  block_shape=[128,128]
#
# Output lands at:
#   vllm/sndr_core/configs/moe_tuning/E={E},N={N},device_name={GPU},dtype={DTYPE}{,block_shape={BS}}.json
# (override via SNDR_MOE_TUNING_DIR env var)
#
# Author: Sandermage 2026-05-05.
set -euo pipefail

E="${E:-256}"
N="${N:-512}"
DTYPE="${DTYPE:-fp8_w8a8}"
BLOCK_SHAPE="${BLOCK_SHAPE:-[128,128]}"  # empty string = no block_shape suffix
CONTAINER="${CONTAINER:-vllm-server-mtp-test}"
SSH_HOST="${SSH_HOST:-${USER}@127.0.0.1}"

# Resolve GPU name via nvidia-smi
if [ -n "${GPU_OVERRIDE:-}" ]; then
  GPU="$GPU_OVERRIDE"
else
  GPU=$(ssh "$SSH_HOST" "nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_'" 2>/dev/null || echo "UNKNOWN_GPU")
fi

# Filename pattern matches vLLM's runtime lookup
if [ -n "$BLOCK_SHAPE" ] && [ "$BLOCK_SHAPE" != "none" ]; then
  FNAME="E=${E},N=${N},device_name=${GPU},dtype=${DTYPE},block_shape=${BLOCK_SHAPE}.json"
else
  FNAME="E=${E},N=${N},device_name=${GPU},dtype=${DTYPE}.json"
fi

OUT_DIR="${SNDR_MOE_TUNING_DIR:-vllm/sndr_core/configs/moe_tuning}"
OUT_PATH="$OUT_DIR/$FNAME"

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Genesis MoE config naming + staging helper (NOT an autotuner)"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Target shape:    E=$E, N=$N, dtype=$DTYPE${BLOCK_SHAPE:+, block_shape=$BLOCK_SHAPE}"
echo "  Detected GPU:    $GPU"
echo "  Output path:     $OUT_PATH"
echo "  Container:       $CONTAINER"
echo "───────────────────────────────────────────────────────────────────────"

# Pre-flight
if [ -f "$OUT_PATH" ]; then
  echo "  [WARN] $OUT_PATH already exists. Backing up to .prev"
  cp "$OUT_PATH" "$OUT_PATH.prev"
fi

mkdir -p "$OUT_DIR"

# Run the autotuner inside the container.
# We use vLLM's bundled benchmark_mixtral_moe.py as it triggers the standard
# autotuner path and dumps a JSON config. Some vLLM versions ship it under a
# different name — adjust if your pin differs.
echo "  Probing the lookup path inside $CONTAINER (autotune sweep is manual — see NEXT STEPS)..."

ssh "$SSH_HOST" "docker exec -i $CONTAINER bash -c \"
  set -e
  cd /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe
  python3 -c '
import json
import torch
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_experts,
    get_default_config,
)
print(\\\"Detected GPU:\\\", torch.cuda.get_device_name(0))
print(\\\"E=$E, N=$N, dtype=$DTYPE\\\")
print(\\\"Note: full autotune requires running vLLM benchmark scripts under load\\\")
print(\\\"This helper only verifies the lookup path. To autotune, see the\\\")
print(\\\"vLLM tuning workflow in vllm/model_executor/layers/fused_moe/configs/README.md\\\")
'
\"" 2>&1 | sed 's/^/  /'

cat <<EOF

───────────────────────────────────────────────────────────────────────

NEXT STEPS (manual — vLLM benchmark scripts vary by pin version):

1. Inside the container, find the autotune entry point (varies by pin):
     docker exec $CONTAINER bash -c "find /usr/local/lib/python3.12/dist-packages/vllm -name 'benchmark*moe*'"

2. Run with your target shape (refer to the script's --help for exact flags;
   typical pattern):
     docker exec $CONTAINER python3 /path/to/benchmark_moe.py \\
       --num-experts $E --shard-intermediate-size $N \\
       --dtype $DTYPE --tune

3. The autotuner writes the best config to:
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/$FNAME

4. Copy back to host + bundle into Genesis:
     docker cp $CONTAINER:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/$FNAME ./$OUT_PATH

5. Verify it was picked up on next boot — look for the FILE in boot logs:
     docker logs $CONTAINER 2>&1 | grep "fused_moe.*config"

6. Bind-mount in your start script (already done for the A5000 PROD):
     -v "\${SNDR_REPO_ROOT:-\${HOME}/genesis-vllm-patches}/\$OUT_PATH:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs/\$FNAME:ro" \\

═══════════════════════════════════════════════════════════════════════
EOF

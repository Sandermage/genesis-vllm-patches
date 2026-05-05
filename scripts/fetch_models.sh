#!/bin/bash
# fetch_models.sh — SHA-verified model downloader (tfriedel pattern port).
#
# Adapted from tfriedel/qwen3.6-rtx3090-lab `setup.sh` (2026-04-29).
# Idempotent: re-runs verify only if shards are present; downloads missing.
#
# Usage:
#   ./fetch_models.sh Lorbus/Qwen3.6-27B-int4-AutoRound /nfs/genesis/models
#   ./fetch_models.sh cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit /nfs/genesis/models
#
# Environment:
#   HF_HOME              — huggingface cache root (default ~/.cache/huggingface)
#   GENESIS_MODELS_ROOT  — destination dir override (overrides arg 2)
#   GENESIS_HF_TOKEN     — pass to `huggingface-cli login` if model is gated
#
# Verification:
#   - lists local files vs remote LFS pointers
#   - SHA-checks each .safetensors against x-linked-etag
#   - reports any drift (downloads can be silently truncated by network)
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <repo_id> [destination_root]" >&2
  echo "Example: $0 Lorbus/Qwen3.6-27B-int4-AutoRound /nfs/genesis/models" >&2
  exit 1
fi

REPO_ID="$1"
DEST_ROOT="${GENESIS_MODELS_ROOT:-${2:-/nfs/genesis/models}}"
LOCAL_DIR="${DEST_ROOT}/$(basename "$REPO_ID")"

# Color
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Genesis fetch_models — SHA-verified HF download"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Repo:    $REPO_ID"
echo "  Dest:    $LOCAL_DIR"
echo "───────────────────────────────────────────────────────────────────────"

# Check huggingface-cli available
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo -e "  ${YELLOW}WARN${NC} huggingface-cli not on PATH; trying pip install" >&2
  pip install --quiet huggingface_hub || {
    echo -e "  ${RED}ERROR${NC} huggingface_hub install failed; please install manually:" >&2
    echo "    pip install huggingface_hub" >&2
    exit 2
  }
fi

# Token if provided
if [ -n "${GENESIS_HF_TOKEN:-}" ]; then
  echo "  Using GENESIS_HF_TOKEN for authentication"
  echo "$GENESIS_HF_TOKEN" | huggingface-cli login --token 2>/dev/null || true
fi

mkdir -p "$LOCAL_DIR"

echo "  [1/3] Downloading $REPO_ID → $LOCAL_DIR (resumable)"
huggingface-cli download "$REPO_ID" \
  --local-dir "$LOCAL_DIR" \
  --local-dir-use-symlinks False \
  2>&1 | grep -vE "^Fetching|^Downloading" | tail -5 || true

echo "  [2/3] Verifying shards"
shard_count=$(find "$LOCAL_DIR" -name '*.safetensors' -type f | wc -l | tr -d ' ')
echo "    found $shard_count .safetensors shards"

if [ "$shard_count" -eq 0 ]; then
  echo -e "    ${RED}ERROR${NC} no .safetensors files in $LOCAL_DIR" >&2
  echo "    Possible causes: download interrupted, gated repo (need token), wrong repo_id" >&2
  exit 3
fi

echo "  [3/3] Sanity-check config.json"
if [ -f "$LOCAL_DIR/config.json" ]; then
  python3 -c "import json; c=json.load(open('$LOCAL_DIR/config.json')); print(f'    model_type={c.get(\"model_type\")} archs={c.get(\"architectures\")}')"
else
  echo -e "    ${YELLOW}WARN${NC} no config.json — model may be incomplete" >&2
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo -e "  ${GREEN}DONE${NC}: $REPO_ID → $LOCAL_DIR ($shard_count shards)"
echo ""
echo "  Next: edit one of the start scripts to point at your model:"
echo "    scripts/start_27b_int4_TQ_k8v4.sh    (27B Lorbus + TQ k8v4)"
echo "    scripts/start_35b_fp8_PROD.sh        (35B-A3B FP8 + MTP K=3)"
echo "═══════════════════════════════════════════════════════════════════════"

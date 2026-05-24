#!/bin/bash
# Capture full state snapshot BEFORE swapping a Phase 1/2 arm.
# Saves: launch script copy, env vars, vllm version, container ID, GPU state, time.
# Output: sndr_private/snapshots/<timestamp>_<arm_name>/
#
# Usage:  bash scripts/launch/snapshot_pre_arm.sh <arm_name>
# Example: bash scripts/launch/snapshot_pre_arm.sh phase1_arm_a_baseline_refresh

set -euo pipefail

ARM_NAME="${1:-unnamed}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/sndr_private/snapshots/${TS}_${ARM_NAME}"
mkdir -p "${OUT_DIR}"

echo "Snapshot dir: ${OUT_DIR}"

# F-017 fix (audit 2026-05-07): env fallback for SSH host / container /
# API key. Defaults preserve Sander's PROD setup; community/test rigs
# override via env vars without editing the script.
SSH_HOST="${GENESIS_SSH_HOST:-${SSH_HOST:-sander@192.168.1.10}}"
CONTAINER="${GENESIS_CONTAINER:-vllm-server-mtp-test}"
API_KEY="${GENESIS_API_KEY:-genesis-local}"

# Capture from server
ssh "${SSH_HOST}" "
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.ID}}' | head -10
echo '---ENV---'
docker inspect ${CONTAINER} --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -E 'GENESIS|VLLM|PYTORCH|NCCL|CUDA' | sort
echo '---CMD---'
docker inspect ${CONTAINER} --format '{{join .Config.Cmd \" \"}}' 2>/dev/null
echo '---GPU---'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
echo '---CONTAINER LOG TAIL---'
docker logs --tail 50 ${CONTAINER} 2>&1
echo '---HEALTH---'
curl -s -H 'Authorization: Bearer ${API_KEY}' http://localhost:8000/v1/models 2>&1 | head -200
" > "${OUT_DIR}/server_state.txt" 2>&1 || true

# Capture local state
git -C "${REPO_ROOT}" log -1 --format='%H %ai %s' > "${OUT_DIR}/repo_head.txt"
git -C "${REPO_ROOT}" status --short > "${OUT_DIR}/repo_status.txt"
git -C "${REPO_ROOT}" diff --stat > "${OUT_DIR}/repo_diff_stat.txt" || true

# Capture the launch script for THIS arm if it exists
LAUNCH_NAME="start_${ARM_NAME%%_*}_${ARM_NAME#*_}.sh"
for cand in \
    "${REPO_ROOT}/scripts/launch/start_v786_${ARM_NAME#phase1_}.sh" \
    "${REPO_ROOT}/scripts/launch/start_${ARM_NAME}.sh" \
    "${REPO_ROOT}/scripts/launch/${ARM_NAME}.sh"; do
    if [ -f "${cand}" ]; then
        cp "${cand}" "${OUT_DIR}/launch_script.sh"
        echo "Captured launch script: $(basename "${cand}")"
        break
    fi
done

echo ""
echo "Snapshot captured at ${OUT_DIR}"
echo "Files:"
ls -la "${OUT_DIR}"

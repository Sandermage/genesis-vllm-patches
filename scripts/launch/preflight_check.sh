#!/bin/bash
# Genesis vLLM Patches — pre-flight dependency check for a fresh / clean system.
#
# Run BEFORE any start_*.sh. Reports green/yellow/red on every prerequisite.
# Exits 0 if all green, 1 if any red. Yellow = warning, doesn't fail.
#
# Tested on: Ubuntu 22.04+ / Debian 12+ with NVIDIA driver >= 580.126.09.

set -uo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
EXIT_CODE=0
WARN_COUNT=0

ok()   { echo -e "  ${GREEN}OK${NC}    $1"; }
warn() { echo -e "  ${YELLOW}WARN${NC}  $1"; WARN_COUNT=$((WARN_COUNT+1)); }
fail() { echo -e "  ${RED}FAIL${NC}  $1"; EXIT_CODE=1; }

echo "=== Genesis vLLM Patches pre-flight check ==="
echo

# ─── 1. OS / kernel ───────────────────────────────────────────────────
echo "--- OS / kernel ---"
. /etc/os-release 2>/dev/null && ok "OS: $PRETTY_NAME" || warn "Cannot read /etc/os-release"
KERNEL=$(uname -r)
ok "Kernel: $KERNEL"

# ─── 2. NVIDIA driver ────────────────────────────────────────────────
echo
echo "--- NVIDIA driver ---"
if command -v nvidia-smi >/dev/null; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
  CUDA=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "n/a")
  if [ -z "$DRIVER" ]; then
    fail "nvidia-smi returns no driver version (driver not loaded?)"
  else
    # Genesis memory: driver >= 580.126.09 REQUIRED (570 → 3× slowdown)
    DRIVER_MAJOR=$(echo "$DRIVER" | cut -d. -f1)
    if [ "$DRIVER_MAJOR" -lt 580 ]; then
      fail "Driver $DRIVER < 580.126.09 — Genesis A5000 stack expects >= 580 (570 = 3× slowdown)"
    else
      ok "Driver: $DRIVER (CUDA $CUDA)"
    fi
  fi
  GPU_LIST=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
  echo "$GPU_LIST" | while IFS= read -r line; do ok "GPU: $line"; done
else
  fail "nvidia-smi not found — install NVIDIA driver first"
fi

# ─── 3. Docker + nvidia-container-toolkit ───────────────────────────
echo
echo "--- Docker + nvidia container toolkit ---"
if command -v docker >/dev/null; then
  DV=$(docker --version | awk '{print $3}' | tr -d ',')
  ok "docker: $DV"
  if docker info 2>/dev/null | grep -q nvidia; then
    ok "Docker has nvidia runtime configured"
  else
    fail "Docker runtime 'nvidia' not registered — install nvidia-container-toolkit + restart docker"
  fi
  # Quick GPU exposure sanity (lightweight image)
  if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L >/dev/null 2>&1; then
    ok "GPUs visible inside containers"
  else
    warn "GPU passthrough test failed — try 'docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi'"
  fi
else
  fail "docker not found — install Docker Engine"
fi

# ─── 4. Python (host) ───────────────────────────────────────────────
echo
echo "--- Python host tools ---"
if command -v python3 >/dev/null; then
  PV=$(python3 --version | awk '{print $2}')
  ok "python3: $PV"
else
  warn "python3 not found on host (only needed for repo-side scripts; container has its own)"
fi

# ─── 5. NFS / model store ──────────────────────────────────────────
echo
echo "--- Model store ---"
MODEL_DIR="${GENESIS_MODEL_DIR:-/nfs/genesis/models}"
if [ -d "$MODEL_DIR" ]; then
  COUNT=$(find "$MODEL_DIR" -maxdepth 1 -type d | wc -l)
  ok "Model dir: $MODEL_DIR ($((COUNT-1)) model dirs)"
else
  fail "Model dir $MODEL_DIR does not exist (set GENESIS_MODEL_DIR or mount /nfs/genesis/models)"
fi

# ─── 6. Compile / triton caches ────────────────────────────────────
echo
echo "--- Cache dirs ---"
for D in \
  "${HOME}/Genesis_Project/vllm_engine/triton-cache-int8-mtp" \
  "${HOME}/Genesis_Project/vllm_engine/compile-cache-int8-mtp" \
  "${HOME}/.cache/huggingface"; do
  if [ -d "$D" ]; then
    ok "$D ($(du -sh "$D" 2>/dev/null | awk '{print $1}'))"
  else
    warn "$D missing — will be created on first launch"
    mkdir -p "$D" 2>/dev/null && ok "  ↳ created"
  fi
done

# ─── 7. genesis-vllm-patches local checkout ────────────────────────
echo
echo "--- Genesis patcher checkout ---"
PATCHER_DIR="${GENESIS_PATCHER_DIR:-${HOME}/genesis-vllm-patches}"
if [ -d "$PATCHER_DIR/vllm/_genesis" ]; then
  PATCH_COUNT=$(find "$PATCHER_DIR/vllm/_genesis/wiring" -maxdepth 1 -name "patch_*.py" -not -name "__*" | wc -l)
  ok "Patcher: $PATCHER_DIR ($PATCH_COUNT wiring patches)"
  if [ -f "$PATCHER_DIR/.git/HEAD" ]; then
    HEAD=$(cd "$PATCHER_DIR" && git rev-parse --short HEAD 2>/dev/null)
    BRANCH=$(cd "$PATCHER_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null)
    ok "  ↳ branch=$BRANCH HEAD=$HEAD"
  fi
else
  fail "$PATCHER_DIR/vllm/_genesis/ not found (set GENESIS_PATCHER_DIR or clone genesis-vllm-patches)"
fi

# ─── 8. vLLM nightly image cached ───────────────────────────────────
echo
echo "--- vLLM container image ---"
if command -v docker >/dev/null; then
  if docker image inspect vllm/vllm-openai:nightly >/dev/null 2>&1; then
    SIZE=$(docker image inspect vllm/vllm-openai:nightly --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1024/1024/1024}')
    ID=$(docker image inspect vllm/vllm-openai:nightly --format '{{.Id}}' | head -c 19)
    ok "vllm/vllm-openai:nightly cached ($SIZE, $ID)"
  else
    warn "vllm/vllm-openai:nightly not cached locally — will pull on first launch (~12 GB)"
  fi
fi

# ─── 9. Network: docker network for genesis ─────────────────────────
echo
echo "--- Docker network ---"
if command -v docker >/dev/null; then
  if docker network inspect genesis-vllm-patches_default >/dev/null 2>&1; then
    ok "Docker network 'genesis-vllm-patches_default' exists"
  else
    warn "Network 'genesis-vllm-patches_default' missing — will be auto-created by compose, or run: docker network create genesis-vllm-patches_default"
  fi
fi

# ─── 10. Free GPU memory check ──────────────────────────────────────
echo
echo "--- GPU free memory ---"
if command -v nvidia-smi >/dev/null; then
  while IFS=, read -r idx free total; do
    free=$(echo "$free" | tr -d ' ')
    total=$(echo "$total" | tr -d ' ')
    PCT=$((free * 100 / total))
    if [ "$PCT" -lt 80 ]; then
      warn "GPU $idx: $free MiB free / $total MiB total (${PCT}%) — other processes may be holding memory"
    else
      ok "GPU $idx: $free / $total MiB free (${PCT}%)"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)
fi

echo
echo "=== Summary ==="
if [ "$EXIT_CODE" -eq 0 ]; then
  if [ "$WARN_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}PASS with $WARN_COUNT warning(s)${NC} — system can boot vLLM, review warnings"
  else
    echo -e "${GREEN}ALL GREEN${NC} — system fully ready"
  fi
else
  echo -e "${RED}FAIL${NC} — fix red items before launching vLLM"
fi
exit "$EXIT_CODE"

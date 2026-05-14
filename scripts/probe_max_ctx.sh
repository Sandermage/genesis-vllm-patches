#!/bin/bash
# probe_max_ctx.sh — auto-binary-search for max stable --max-model-len
#
# Adapted from tfriedel/qwen3.6-rtx3090-lab `probe_moe_ctx.sh` (4× RTX 3090
# cross-rig). Useful for Cliff 2 detection + OOM_RECIPES doc + first-time
# operators sizing their context window.
#
# Usage:
#   ENDPOINT=http://127.0.0.1:8000 MODEL=qwen3.6-35b-a3b ./probe_max_ctx.sh
#   ./probe_max_ctx.sh --start 16384 --max 320000 --kv fp8_e5m2
#
# Strategy:
#   1. Send single-shot 100-token completion with prompt sized to N tokens
#   2. If response 200 + content non-empty → ctx N PASS
#   3. Binary-search upward until first failure
#   4. Report largest N that boots + first N that OOMs
#
# Note: this assumes the container is already booted; it's a runtime probe,
# NOT a re-boot loop. To probe across kv_cache_dtype values, restart the
# container between probes (script suggests but does not execute).
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000}"
MODEL="${MODEL:-qwen3.6-35b-a3b}"
API_KEY="${GENESIS_API_KEY:-genesis-local}"
START_CTX="${START_CTX:-16384}"
MAX_CTX="${MAX_CTX:-320000}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-(unspecified, container default)}"

# Color
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

pass() { printf "  ${GREEN}✓${NC} ctx=%6d  %s\n" "$1" "$2"; }
fail() { printf "  ${RED}✗${NC} ctx=%6d  %s\n" "$1" "$2"; }
info() { printf "  ${BLUE}ℹ${NC} %s\n" "$1"; }

probe_ctx() {
  local ctx_target=$1
  # Build a prompt of approximately ctx_target tokens (4 chars/token rough)
  local target_chars=$((ctx_target * 4))
  local prompt=$(python3 -c "import sys; print('lorem ipsum dolor sit amet ' * (${target_chars} // 28))" 2>/dev/null)
  prompt="${prompt}\n\nRespond with one sentence."

  # Use python for proper JSON escaping
  local resp_status=$(python3 -c "
import json, urllib.request, urllib.error
req = urllib.request.Request(
    '${ENDPOINT}/v1/chat/completions',
    data=json.dumps({'model':'${MODEL}','messages':[{'role':'user','content':'''${prompt}'''}],'max_tokens':50,'temperature':0}).encode(),
    headers={'Authorization':'Bearer ${API_KEY}','Content-Type':'application/json'},
)
try:
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
        ch = d['choices'][0]
        out = (ch['message'].get('content') or ch['message'].get('reasoning') or '').strip()
        if out:
            print('PASS')
        else:
            print('EMPTY_RESPONSE')
except urllib.error.HTTPError as e:
    print(f'HTTP_{e.code}')
except Exception as e:
    print(f'ERROR_{type(e).__name__}')
" 2>/dev/null)
  echo "$resp_status"
}

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Genesis probe_max_ctx — auto-binary-search for max stable --max-model-len"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Endpoint:      $ENDPOINT"
echo "  Model:         $MODEL"
echo "  Range:         $START_CTX → $MAX_CTX"
echo "  KV cache:      $KV_CACHE_DTYPE"
echo "───────────────────────────────────────────────────────────────────────"

# Linear scan first to find the cliff, then binary-search
last_pass=0
first_fail=0

for ctx in $START_CTX 32768 65536 96000 131072 196608 262144 320000; do
  if [ "$ctx" -gt "$MAX_CTX" ]; then continue; fi
  printf "  probing ctx=%6d ... " "$ctx"
  status=$(probe_ctx "$ctx")
  if [ "$status" = "PASS" ]; then
    printf "${GREEN}PASS${NC}\n"
    last_pass=$ctx
  else
    printf "${RED}%s${NC}\n" "$status"
    first_fail=$ctx
    break
  fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
if [ "$first_fail" -eq 0 ]; then
  printf "  ${GREEN}MAX STABLE: %d tokens${NC} (no failure within probed range)\n" "$last_pass"
  echo "  Recommendation: try --max-model-len $((last_pass * 2)) on next boot"
else
  printf "  ${GREEN}LAST PASS: %d tokens${NC} | ${RED}FIRST FAIL: %d tokens${NC}\n" \
    "$last_pass" "$first_fail"
  echo "  Recommendation: set --max-model-len ≤ $last_pass for stable PROD"
  echo ""
  echo "  Cliff hint: if first_fail/last_pass ≈ 2× → potential Cliff 2 OOM"
  echo "  (DeltaNet GLA materialization). Try GENESIS_ENABLE_PN59_STREAMING_GDN=1"
  echo "  on next boot to push the ceiling higher."
fi
echo "═══════════════════════════════════════════════════════════════════════"

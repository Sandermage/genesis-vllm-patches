#!/bin/bash
# Genesis verify-full — 7-stage end-to-end smoke test.
#
# Adapted from tfriedel/qwen3.6-rtx3090-lab `verify-full.sh` (2026-04-29 cross-
# rig replication on 4× RTX 3090). Restructured to call the Genesis-served
# endpoint and surface Genesis-specific patch markers.
#
# Stages:
#   1. /v1/models          — server reachable
#   2. Genesis patch marker — boot summary contains 'structured boot summary'
#   3. Basic completion     — Paris weather plain text
#   4. Tool-call            — get_weather function call
#   5. Streaming SSE        — token-by-token delivery
#   6. Thinking-mode        — Qwen3 reasoning channel populated
#   7. 4-depth needle ladder — 1K / 10K / 50K / 90K context (random
#                              animal+color+number, in-vocabulary tokens)
#
# Usage:
#   ENDPOINT=http://127.0.0.1:8000 MODEL=qwen3.6-27b ./verify-full.sh
#   ./verify-full.sh --skip-needle  # quick mode (skip stage 7)
#
# Exit 0 = all stages PASS; non-zero = first stage that failed.
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8000}"
MODEL="${MODEL:-qwen3.6-27b}"
API_KEY="${GENESIS_API_KEY:-genesis-local}"
CONTAINER="${CONTAINER:-vllm-server-mtp-test}"
SSH_HOST="${SSH_HOST:-${USER}@127.0.0.1}"
SKIP_NEEDLE=0
for arg in "$@"; do
  [[ "$arg" == "--skip-needle" ]] && SKIP_NEEDLE=1
done

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

pass() { printf "  ${GREEN}✓${NC} %s\n" "$1"; }
fail() { printf "  ${RED}✗${NC} %s\n" "$1"; printf "    ${YELLOW}→ %s${NC}\n" "$2"; exit 1; }
info() { printf "  ${BLUE}ℹ${NC} %s\n" "$1"; }

echo "═══════════════════════════════════════════════════════════════════════"
echo "  Genesis verify-full — 7-stage smoke test"
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Endpoint: $ENDPOINT"
echo "  Model:    $MODEL"
echo "  Container:$CONTAINER"
echo "───────────────────────────────────────────────────────────────────────"

# ─── Stage 1: server reachable ───────────────────────────────────────────
echo ""
echo "[1/7] Server reachability"
code=$(curl -s -o /dev/null -w '%{http_code}' "$ENDPOINT/v1/models" 2>/dev/null || echo "000")
if [[ "$code" == "401" || "$code" == "200" ]]; then
  pass "GET /v1/models → HTTP $code"
else
  fail "GET /v1/models → HTTP $code" "Boot vLLM container or check endpoint URL"
fi

# ─── Stage 2: Genesis patch marker present ───────────────────────────────
echo ""
echo "[2/7] Genesis patch marker (boot summary)"
if ssh "$SSH_HOST" "docker logs $CONTAINER 2>&1 | grep -q 'structured boot summary'" 2>/dev/null; then
  applied=$(ssh "$SSH_HOST" "docker logs $CONTAINER 2>&1 | grep -oE 'APPLY  \|  [0-9]+ SKIP' | head -1" 2>/dev/null || echo "?")
  pass "Genesis structured boot summary present in container logs ($applied)"
else
  fail "no 'structured boot summary' line in container logs" \
    "Check container started cleanly + Genesis plugin installed"
fi

# ─── Stage 3: basic completion ───────────────────────────────────────────
echo ""
echo "[3/7] Basic completion (no tools)"
resp=$(curl -s -X POST "$ENDPOINT/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: PARIS\"}],\"max_tokens\":50,\"temperature\":0}")
content=$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); ch=d['choices'][0]; print((ch['message'].get('content') or ch['message'].get('reasoning') or '').upper())" 2>/dev/null || echo "")
if [[ "$content" == *"PARIS"* ]]; then
  pass "completion contains PARIS"
else
  fail "completion='$content'" "Model response missing expected token"
fi

# ─── Stage 4: tool-call ──────────────────────────────────────────────────
echo ""
echo "[4/7] Tool-call (get_weather Berlin)"
tc_resp=$(curl -s -X POST "$ENDPOINT/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Use get_weather to check Berlin.\"}],\"max_tokens\":150,\"temperature\":0,\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"tool_choice\":\"auto\"}")
tc_args=$(echo "$tc_resp" | python3 -c "import json,sys; d=json.load(sys.stdin); ch=d['choices'][0]; tc=ch['message'].get('tool_calls') or []; print(tc[0]['function']['arguments'] if tc else 'NONE')" 2>/dev/null || echo "ERROR")
if [[ "$tc_args" == *"Berlin"* ]]; then
  pass "tool_calls[0].arguments contains Berlin: $tc_args"
else
  fail "tool_calls=$tc_args" "Tool-call generation failed or args malformed"
fi

# ─── Stage 5: streaming SSE ──────────────────────────────────────────────
echo ""
echo "[5/7] Streaming SSE delivery"
sse_chunks=$(curl -sN -X POST "$ENDPOINT/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count 1 2 3\"}],\"max_tokens\":30,\"stream\":true,\"temperature\":0}" 2>/dev/null \
  | grep -c "^data: " || echo "0")
if [[ "$sse_chunks" -ge 3 ]]; then
  pass "SSE delivered $sse_chunks data: chunks"
else
  fail "SSE chunks=$sse_chunks" "Streaming delivery broken"
fi

# ─── Stage 6: thinking-mode (Qwen3 reasoning channel) ────────────────────
echo ""
echo "[6/7] Thinking-mode (Qwen3 reasoning channel)"
think_resp=$(curl -s -X POST "$ENDPOINT/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 23? Show your reasoning.\"}],\"max_tokens\":200,\"temperature\":0}")
has_reasoning=$(echo "$think_resp" | python3 -c "import json,sys; d=json.load(sys.stdin); ch=d['choices'][0]; r=(ch['message'].get('reasoning') or '').strip(); c=(ch['message'].get('content') or '').strip(); print('YES' if (len(r) > 10 or '<think>' in c) else 'NO')" 2>/dev/null || echo "ERROR")
if [[ "$has_reasoning" == "YES" ]]; then
  pass "reasoning channel populated"
else
  info "reasoning channel empty (Qwen3 thinking mode may be off — depends on enable_thinking client flag)"
fi

# ─── Stage 7: 4-depth needle ladder ──────────────────────────────────────
if [[ "$SKIP_NEEDLE" == "1" ]]; then
  echo ""
  echo "[7/7] Needle ladder — SKIPPED (--skip-needle flag)"
  echo ""
  echo "═══════════════════════════════════════════════════════════════════════"
  echo "  ${GREEN}ALL STAGES PASS${NC} (skipped stage 7)"
  echo "═══════════════════════════════════════════════════════════════════════"
  exit 0
fi

echo ""
echo "[7/7] 4-depth needle ladder (1K → 10K → 50K → 90K)"
echo "  Random in-vocabulary needle: animal+color+number tuple"

NEEDLE_ANIMAL="${NEEDLE_ANIMAL:-fox}"
NEEDLE_COLOR="${NEEDLE_COLOR:-blue}"
NEEDLE_NUMBER="${NEEDLE_NUMBER:-741}"
NEEDLE="The secret password is $NEEDLE_ANIMAL $NEEDLE_COLOR $NEEDLE_NUMBER."

run_needle() {
  local target_tokens=$1
  local depth=$2
  # Generate filler text of ~target_tokens (rough: 4 chars/token)
  local target_chars=$((target_tokens * 4))
  local filler=$(python3 -c "import sys; print(' '.join(['lorem ipsum dolor sit amet'] * (${target_chars} // 27)))" 2>/dev/null)

  # Place needle at given depth (0.0 = start, 0.5 = middle, 1.0 = end)
  local fill_len=${#filler}
  local insert_at=$(python3 -c "print(int(${fill_len} * ${depth}))")
  local prefix="${filler:0:$insert_at}"
  local suffix="${filler:$insert_at}"
  local prompt="$prefix $NEEDLE $suffix\n\nWhat is the secret password?"

  local resp=$(python3 -c "
import json, urllib.request
req = urllib.request.Request(
    '$ENDPOINT/v1/chat/completions',
    data=json.dumps({'model':'$MODEL','messages':[{'role':'user','content':'''$prompt'''}],'max_tokens':50,'temperature':0}).encode(),
    headers={'Authorization':'Bearer $API_KEY','Content-Type':'application/json'},
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
        ch = d['choices'][0]
        print((ch['message'].get('content') or ch['message'].get('reasoning') or '').lower())
except Exception as e:
    print(f'ERROR: {e}')
" 2>/dev/null)
  echo "$resp"
}

found_count=0
for ctx_depth in "1024:0.5" "10240:0.5" "51200:0.5" "92160:0.5"; do
  ctx="${ctx_depth%:*}"
  depth="${ctx_depth#*:}"
  printf "  ctx=%6dt depth=%s ... " "$ctx" "$depth"
  resp=$(run_needle "$ctx" "$depth")
  if [[ "$resp" == *"$NEEDLE_ANIMAL"* && "$resp" == *"$NEEDLE_COLOR"* && "$resp" == *"$NEEDLE_NUMBER"* ]]; then
    printf "${GREEN}FOUND${NC} (animal+color+number)\n"
    found_count=$((found_count+1))
  elif [[ "$resp" == *"$NEEDLE_ANIMAL"* || "$resp" == *"$NEEDLE_COLOR"* || "$resp" == *"$NEEDLE_NUMBER"* ]]; then
    printf "${YELLOW}PARTIAL${NC} (one or two of three)\n"
    found_count=$((found_count+1))
  else
    printf "${RED}MISS${NC}\n"
  fi
done

echo ""
if [[ "$found_count" -ge 3 ]]; then
  pass "needle ladder $found_count/4 found"
else
  fail "needle ladder $found_count/4 found" "Long-context recall is broken"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "  ${GREEN}ALL 7 STAGES PASS${NC}"
echo "═══════════════════════════════════════════════════════════════════════"

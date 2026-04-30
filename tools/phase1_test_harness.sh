#!/bin/bash
# Phase 1 test harness — runs the full test battery for ONE arm.
#
# Usage:  bash tools/phase1_test_harness.sh <arm_name>
# Example: bash tools/phase1_test_harness.sh arm_a_baseline_refresh
#
# Tests run (in order, ALL must pass before promoting):
#   1. /v1/models reachability (server up)
#   2. tool-call quality probe (4 cases must pass cleanly)
#   3. decode-only TPOT bench (N=25, 5 prompts × 1024 tokens)
#   4. multi-turn TTFT probe (5 sequential same-prefix requests)
#   5. 30-iteration stability stress (no NaN, no crash, no quality drift)
#   6. context window probe (256K, 280K, 300K, 317K — ensure no OOM)
#
# Output: docs/_internal/runs/<arm_name>/{tool_call.json, bench.json, ttft.json,
#         stress.json, ctx_probe.json, summary.md}
#
# Requires: server running on localhost:8000 with api-key genesis-local

set -euo pipefail

ARM_NAME="${1:-unnamed}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-genesis-local}"
MODEL="${MODEL:-qwen3.6-35b-a3b}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_DIR="${REPO_ROOT}/docs/_internal/runs/${ARM_NAME}"
mkdir -p "${RUNS_DIR}"
SUMMARY="${RUNS_DIR}/summary.md"

echo "Phase 1 test harness — arm: ${ARM_NAME}" | tee "${SUMMARY}"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${SUMMARY}"
echo "Host: ${HOST}:${PORT}  Model: ${MODEL}" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

curl_jh() { curl -sS -H "Authorization: Bearer ${API_KEY}" "$@"; }

# ============ TEST 1: server reachable ============
echo "[1/6] /v1/models reachability..." | tee -a "${SUMMARY}"
if curl_jh "http://${HOST}:${PORT}/v1/models" -o /dev/null -w "%{http_code}\n" | grep -q '^200'; then
    echo "  PASS" | tee -a "${SUMMARY}"
else
    echo "  FAIL — server not reachable. Aborting." | tee -a "${SUMMARY}"
    exit 1
fi

# ============ TEST 2: tool-call quality probe (4 cases) ============
echo "[2/6] Tool-call quality (4 cases)..." | tee -a "${SUMMARY}"
TOOLCALL_OUT="${RUNS_DIR}/tool_call.json"
PASSED=0
FAILED=0
for case_id in 1 2 3 4; do
    case "$case_id" in
        1) THINKING=false; PARSER_HINT="hermes-xml"; PROMPT="What's the weather in Paris? Use the get_weather tool." ;;
        2) THINKING=true;  PARSER_HINT="hermes-xml"; PROMPT="Think step by step then call get_weather for Tokyo." ;;
        3) THINKING=false; PARSER_HINT="oai-tools";  PROMPT="Call get_weather for New York." ;;
        4) THINKING=true;  PARSER_HINT="oai-tools";  PROMPT="Reason about which city, then call get_weather for London." ;;
    esac
    RESP=$(curl_jh -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"chat_template_kwargs\":{\"enable_thinking\":${THINKING}},\"max_tokens\":1500}" 2>/dev/null)
    HAS_TOOL=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); tc=d.get('choices',[{}])[0].get('message',{}).get('tool_calls'); print('yes' if tc else 'no')" 2>/dev/null || echo "err")
    echo "  case $case_id (think=$THINKING parser=$PARSER_HINT): $HAS_TOOL" | tee -a "${SUMMARY}"
    if [ "$HAS_TOOL" = "yes" ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi
    echo "{\"case\":$case_id,\"thinking\":$THINKING,\"parser_hint\":\"$PARSER_HINT\",\"has_tool_call\":\"$HAS_TOOL\",\"raw\":$RESP}" >> "$TOOLCALL_OUT"
done
echo "  Tool-call pass: $PASSED/4 (need 4/4 to promote)" | tee -a "${SUMMARY}"

# ============ TEST 3: decode-only TPOT bench (N=25) ============
echo "[3/6] Decode-only TPOT bench (N=25, 5 prompts × 1024 tokens)..." | tee -a "${SUMMARY}"
python3 "${REPO_ROOT}/tools/bench_decode_tpot_clean_ab.py" \
    --host "${HOST}" --port "${PORT}" --api-key "${API_KEY}" --model "${MODEL}" \
    --arm-name "${ARM_NAME}" --runs 25 --prompts standard \
    --out "${RUNS_DIR}/bench.json" \
    --quiet 2>&1 | tail -20 | tee -a "${SUMMARY}"

# ============ TEST 4: multi-turn TTFT probe (5 sequential same-prefix, single-process) ============
echo "[4/6] Multi-turn TTFT probe (5 same-prefix requests)..." | tee -a "${SUMMARY}"
TTFT_OUT="${RUNS_DIR}/ttft.json"
set +e
python3 -c "
import sys, json, time, urllib.request
host, port, api_key, model, out_path = sys.argv[1:6]
prefix = 'In the year 2030, scientists discovered a new method to '
results = []
for i in range(1, 6):
    payload = json.dumps({'model': model, 'messages': [{'role': 'user',
        'content': f'{prefix}turn {i}: explain quantum entanglement in 1 sentence.'}],
        'stream': True, 'max_tokens': 50}).encode()
    req = urllib.request.Request(f'http://{host}:{port}/v1/chat/completions',
        data=payload, headers={'Content-Type':'application/json','Authorization':f'Bearer {api_key}'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            ttft_ms = -1
            for line in r:
                if line.startswith(b'data:') and b'\"content\"' in line:
                    ttft_ms = round((time.perf_counter() - t0) * 1000, 1)
                    break
    except Exception as e:
        print(f'  turn {i} TTFT: ERROR ({e})')
        results.append({'turn': i, 'ttft_ms': None, 'error': str(e)})
        continue
    print(f'  turn {i} TTFT: {ttft_ms}ms')
    results.append({'turn': i, 'ttft_ms': ttft_ms})
json.dump(results, open(out_path, 'w'), indent=2)
" "$HOST" "$PORT" "$API_KEY" "$MODEL" "$TTFT_OUT" 2>&1 | tee -a "${SUMMARY}"
set -e

# ============ TEST 5: 30-iteration stability stress ============
echo "[5/6] 30-iteration stability stress..." | tee -a "${SUMMARY}"
STRESS_OUT="${RUNS_DIR}/stress.json"
set +e
python3 "${REPO_ROOT}/tools/bench_decode_tpot_clean_ab.py" \
    --host "${HOST}" --port "${PORT}" --api-key "${API_KEY}" --model "${MODEL}" \
    --arm-name "${ARM_NAME}_stress" --runs 30 --prompts standard \
    --out "${STRESS_OUT}" \
    --quiet 2>&1 | tail -5 | tee -a "${SUMMARY}"
STRESS_EC=$?
echo "  stress exit code: $STRESS_EC" | tee -a "${SUMMARY}"
set -e

# ============ TEST 6: context window probe ============
echo "[6/6] Context window probe (256K, 280K, 300K, 317K)..." | tee -a "${SUMMARY}"
CTX_OUT="${RUNS_DIR}/ctx_probe.json"
set +e
python3 -c "
import sys, json, urllib.request, urllib.error
host, port, api_key, model, out_path = sys.argv[1:6]
results = []
for ctx in [262144, 286720, 307200, 324352]:
    prompt_len = (ctx - 100) // 2
    p = 'hello ' * prompt_len
    payload = json.dumps({'model': model, 'messages': [{'role':'user','content':p}], 'max_tokens': 10}).encode()
    req = urllib.request.Request(f'http://{host}:{port}/v1/chat/completions',
        data=payload, headers={'Content-Type':'application/json','Authorization':f'Bearer {api_key}'})
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            code = r.getcode()
            verdict = 'PASS'
    except urllib.error.HTTPError as e:
        code = e.code
        verdict = f'FAIL({code})'
    except Exception as e:
        code = 0
        verdict = f'ERROR({e})'
    print(f'  ctx ~{ctx}: {verdict}')
    results.append({'context_size': ctx, 'http_status': code, 'verdict': verdict})
json.dump(results, open(out_path, 'w'), indent=2)
" "$HOST" "$PORT" "$API_KEY" "$MODEL" "$CTX_OUT" 2>&1 | tee -a "${SUMMARY}"
set -e

echo "" | tee -a "${SUMMARY}"
echo "Phase 1 harness complete for ${ARM_NAME}" | tee -a "${SUMMARY}"
echo "Outputs in: ${RUNS_DIR}/" | tee -a "${SUMMARY}"

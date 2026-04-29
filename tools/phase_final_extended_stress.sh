#!/bin/bash
# Phase final extended stress test for ONE model.
#
# Runs:
#   1. /v1/models check
#   2. Tool-call quality (4 cases)
#   3. Long bench: N=50 runs × 5 prompts × 1024 decode tokens
#   4. Stability stress: N=100 runs × 5 prompts × 1024 (200 mins of inference)
#   5. Throughput vs context-size scan: 1K, 4K, 16K, 64K, 128K, 256K
#   6. Context window probe: 256K, 280K, 300K, 317K (240s timeout)
#   7. CV measurement across 5 separate boots (TODO — requires container restarts)
#
# Output: docs/_internal/runs/<model>_final/{...}
# Reference config: docs/_internal/REFERENCE_CONFIG_<model>.md

set -euo pipefail

MODEL_TAG="${1:-35b}"  # 35b or 27b
HOST="${HOST:-192.168.1.10}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-genesis-local}"
case "$MODEL_TAG" in
    35b) MODEL="qwen3.6-35b-a3b" ;;
    27b) MODEL="qwen3.6-27b" ;;
    *) echo "Usage: $0 <35b|27b>"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS_DIR="${REPO_ROOT}/docs/_internal/runs/${MODEL_TAG}_final"
mkdir -p "${RUNS_DIR}"
SUMMARY="${RUNS_DIR}/summary.md"

echo "Phase final extended stress for ${MODEL_TAG} (${MODEL})" | tee "${SUMMARY}"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"

curl_jh() { curl -sS -H "Authorization: Bearer ${API_KEY}" "$@"; }

# ============ TEST 1: server reachable ============
echo "[1/6] Server up?" | tee -a "${SUMMARY}"
if curl_jh "http://${HOST}:${PORT}/v1/models" -o /dev/null -w "%{http_code}\n" | grep -q '^200'; then
    echo "  PASS" | tee -a "${SUMMARY}"
else
    echo "  FAIL — abort" | tee -a "${SUMMARY}"; exit 1
fi

# ============ TEST 2: tool-call quality ============
echo "[2/6] Tool-call quality (4 cases)..." | tee -a "${SUMMARY}"
TOOLCALL_OUT="${RUNS_DIR}/tool_call.json"
PASSED=0
echo "[" > "$TOOLCALL_OUT"
for case_id in 1 2 3 4; do
    case "$case_id" in
        1) THINKING=false; PROMPT="What's the weather in Paris? Use the get_weather tool." ;;
        2) THINKING=true;  PROMPT="Think step by step then call get_weather for Tokyo." ;;
        3) THINKING=false; PROMPT="Call get_weather for New York." ;;
        4) THINKING=true;  PROMPT="Reason about which city, then call get_weather for London." ;;
    esac
    RESP=$(curl_jh -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"chat_template_kwargs\":{\"enable_thinking\":${THINKING}},\"max_tokens\":1500}" 2>/dev/null)
    HAS_TOOL=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); tc=d.get('choices',[{}])[0].get('message',{}).get('tool_calls'); print('yes' if tc else 'no')" 2>/dev/null || echo "err")
    echo "  case $case_id (think=$THINKING): $HAS_TOOL" | tee -a "${SUMMARY}"
    [ "$HAS_TOOL" = "yes" ] && PASSED=$((PASSED+1))
    SEP=$([ $case_id -lt 4 ] && echo "," || echo "")
    echo "{\"case\":$case_id,\"thinking\":$THINKING,\"has_tool_call\":\"$HAS_TOOL\"}$SEP" >> "$TOOLCALL_OUT"
done
echo "]" >> "$TOOLCALL_OUT"
echo "  Tool-call: $PASSED/4" | tee -a "${SUMMARY}"

# ============ TEST 3: extended decode bench (N=50) ============
echo "[3/6] Long bench (N=50 runs × 5 prompts × 1024 tokens)..." | tee -a "${SUMMARY}"
python3 "${REPO_ROOT}/tools/bench_decode_tpot_clean_ab.py" \
    --host "${HOST}" --port "${PORT}" --api-key "${API_KEY}" --model "${MODEL}" \
    --arm-name "${MODEL_TAG}_final_long" --runs 50 --prompts standard \
    --out "${RUNS_DIR}/bench_long.json" \
    --quiet 2>&1 | tail -5 | tee -a "${SUMMARY}"

# ============ TEST 4: stability stress (N=100) ============
echo "[4/6] Stability stress (N=100 runs)..." | tee -a "${SUMMARY}"
set +e
python3 "${REPO_ROOT}/tools/bench_decode_tpot_clean_ab.py" \
    --host "${HOST}" --port "${PORT}" --api-key "${API_KEY}" --model "${MODEL}" \
    --arm-name "${MODEL_TAG}_stress_100" --runs 100 --prompts standard \
    --out "${RUNS_DIR}/stress_100.json" \
    --quiet 2>&1 | tail -5 | tee -a "${SUMMARY}"
echo "  stress exit code: $?" | tee -a "${SUMMARY}"
set -e

# ============ TEST 5: TPS vs context-size scan ============
echo "[5/6] TPS vs context-size scan (1K, 4K, 16K, 64K, 128K, 256K)..." | tee -a "${SUMMARY}"
SCAN_OUT="${RUNS_DIR}/ctx_scan.json"
set +e
python3 -c "
import sys, json, time, urllib.request
host, port, api_key, model, out_path = sys.argv[1:6]
results = []
for ctx_target in [1024, 4096, 16384, 65536, 131072, 262144]:
    prompt_len_words = max(1, (ctx_target - 200) // 2)
    p = 'hello ' * prompt_len_words
    payload = json.dumps({'model': model, 'messages': [{'role':'user','content':p}], 'max_tokens': 200, 'stream': False}).encode()
    req = urllib.request.Request(f'http://{host}:{port}/v1/chat/completions',
        data=payload, headers={'Content-Type':'application/json','Authorization':f'Bearer {api_key}'})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            data = json.loads(r.read())
            elapsed = time.perf_counter() - t0
            ct = data.get('usage', {}).get('completion_tokens', 0)
            pt = data.get('usage', {}).get('prompt_tokens', 0)
            tps = ct / elapsed if elapsed > 0 else 0
            print(f'  ctx_target={ctx_target}: prompt_tokens={pt} completion_tokens={ct} elapsed={elapsed:.2f}s tps={tps:.2f}')
            results.append({'ctx_target': ctx_target, 'prompt_tokens': pt, 'completion_tokens': ct, 'elapsed_s': round(elapsed,2), 'tps': round(tps,2)})
    except Exception as e:
        print(f'  ctx_target={ctx_target}: ERROR ({e})')
        results.append({'ctx_target': ctx_target, 'error': str(e)})
json.dump(results, open(out_path, 'w'), indent=2)
" "$HOST" "$PORT" "$API_KEY" "$MODEL" "$SCAN_OUT" 2>&1 | tee -a "${SUMMARY}"
set -e

# ============ TEST 6: context window probe ============
echo "[6/6] Context window probe (256K, 280K, 300K, 317K, 240s timeout)..." | tee -a "${SUMMARY}"
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
            verdict = 'PASS'
            code = r.getcode()
    except urllib.error.HTTPError as e:
        verdict = f'FAIL({e.code})'; code = e.code
    except Exception as e:
        verdict = f'ERROR({e})'; code = 0
    print(f'  ctx ~{ctx}: {verdict}')
    results.append({'context_size': ctx, 'http_status': code, 'verdict': verdict})
json.dump(results, open(out_path, 'w'), indent=2)
" "$HOST" "$PORT" "$API_KEY" "$MODEL" "$CTX_OUT" 2>&1 | tee -a "${SUMMARY}"
set -e

echo "" | tee -a "${SUMMARY}"
echo "Phase final stress complete for ${MODEL_TAG}" | tee -a "${SUMMARY}"
echo "Outputs: ${RUNS_DIR}/" | tee -a "${SUMMARY}"

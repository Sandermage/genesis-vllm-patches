#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# G4-TurboQuant A/B bench: compare {pack_mode × wht_mode} on Gemma 4 256K.
#
# Run on the server with PROD docker. Iterates 4 configs:
#
#   uint32_signs   pack=uint32  wht=signs_only   (baseline — validated 256K boot)
#   uint32_wht     pack=uint32  wht=full_wht     (real Hadamard rotation)
#   tight_signs    pack=tight   wht=signs_only   (5.33× compression)
#   tight_wht      pack=tight   wht=full_wht     (5.33× + Hadamard — quality+savings)
#
# For each config:
#   1. Restart container with the env-flag set.
#   2. Wait for boot (vllm serve reports "Application startup complete").
#   3. Run bench (5 runs × 5 prompts × 384 tokens, n=25) and log JSON.
#   4. Save results to ./results/g4_tq_ab/<config_name>.json.
#
# After all 4 configs, generate a pivot table and a punch list of which
# combo gives the best TPS / quality / memory trade-off.
#
# Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/sander/genesis-vllm-patches}"
CONTAINER="${CONTAINER:-vllm-g4-256k-packed}"
PORT="${PORT:-8027}"
SERVED_MODEL="${SERVED_MODEL:-gemma-4-31b-packed}"
RESULTS_DIR="${RESULTS_DIR:-${HOME}/g4_tq_ab_results}"
N_RUNS="${N_RUNS:-5}"
N_PROMPTS="${N_PROMPTS:-5}"
N_TOKENS="${N_TOKENS:-384}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-${HOME}/start_g4_256k_packed.sh}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-600}"

mkdir -p "${RESULTS_DIR}"

# Each row: name | pack_mode | wht_mode | expected_compression
CONFIGS=(
  "uint32_signs|uint32|signs_only|3.88x"
  "uint32_wht|uint32|full_wht|3.88x"
  "tight_signs|tight|signs_only|5.12x"
  "tight_wht|tight|full_wht|5.12x"
)

echo "=========================================================="
echo "G4-TurboQuant A/B bench"
echo "  container        : ${CONTAINER}"
echo "  port             : ${PORT}"
echo "  served model     : ${SERVED_MODEL}"
echo "  results dir      : ${RESULTS_DIR}"
echo "  runs × prompts × tokens : ${N_RUNS} × ${N_PROMPTS} × ${N_TOKENS}"
echo "=========================================================="
echo

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name pack wht expected_ratio <<< "${cfg}"
  RESULT_FILE="${RESULTS_DIR}/${name}.json"

  echo "──────────────────────────────────────────────────────────"
  echo "CONFIG ${name}: pack=${pack} wht=${wht} (compression ~${expected_ratio})"
  echo "──────────────────────────────────────────────────────────"

  # 1. Stop current container if running
  echo "[1/4] Stopping container ${CONTAINER} (if running)..."
  docker rm -f "${CONTAINER}" 2>/dev/null || true
  sleep 2

  # 2. Patch the launch script's env-flags via in-place sed before starting.
  # We assume the launch script has GENESIS_G4_TQ_PACK_MODE= and
  # GENESIS_G4_TQ_WHT_MODE= lines that we can rewrite. If not, we APPEND
  # them just before the closing image-name line.
  TMP_LAUNCH="$(mktemp /tmp/g4_ab_launch.XXXXXX.sh)"
  cp "${LAUNCH_SCRIPT}" "${TMP_LAUNCH}"

  # Rewrite or append PACK_MODE
  if grep -q "GENESIS_G4_TQ_PACK_MODE" "${TMP_LAUNCH}"; then
    sed -i "s|GENESIS_G4_TQ_PACK_MODE=[a-z0-9_]*|GENESIS_G4_TQ_PACK_MODE=${pack}|g" "${TMP_LAUNCH}"
  else
    sed -i "/-v .* :ro \\\\/i\\  -e GENESIS_G4_TQ_PACK_MODE=${pack} \\\\" "${TMP_LAUNCH}"
  fi

  # Rewrite or append WHT_MODE
  if grep -q "GENESIS_G4_TQ_WHT_MODE" "${TMP_LAUNCH}"; then
    sed -i "s|GENESIS_G4_TQ_WHT_MODE=[a-z_]*|GENESIS_G4_TQ_WHT_MODE=${wht}|g" "${TMP_LAUNCH}"
  else
    sed -i "/-v .* :ro \\\\/i\\  -e GENESIS_G4_TQ_WHT_MODE=${wht} \\\\" "${TMP_LAUNCH}"
  fi

  echo "[2/4] Launching container with pack=${pack} wht=${wht}..."
  chmod +x "${TMP_LAUNCH}"
  bash "${TMP_LAUNCH}"

  # 3. Wait for boot
  echo "[3/4] Waiting up to ${BOOT_TIMEOUT_S}s for vllm serve to come up..."
  T_START=$(date +%s)
  while true; do
    if docker logs "${CONTAINER}" 2>&1 | grep -q "Application startup complete"; then
      ELAPSED=$(($(date +%s) - T_START))
      echo "    booted in ${ELAPSED}s"
      break
    fi
    if [ $(($(date +%s) - T_START)) -gt "${BOOT_TIMEOUT_S}" ]; then
      echo "    ERROR: boot timeout — saving last 80 log lines and skipping"
      docker logs --tail 80 "${CONTAINER}" > "${RESULTS_DIR}/${name}-boot-fail.log" 2>&1
      printf '{"config":"%s","status":"boot_timeout"}\n' "${name}" > "${RESULT_FILE}"
      continue 2
    fi
    sleep 5
  done

  # Sanity: confirm apply summary shows the right modes
  docker logs "${CONTAINER}" 2>&1 | grep -E "G4_19.*pack=|G4_19.*wht=|G4_19b.*pack=" \
    > "${RESULTS_DIR}/${name}-apply.log" 2>&1 || true

  # 4. Bench (canonical genesis_bench_suite.py path; uses --json output)
  echo "[4/4] Running bench (${N_RUNS} runs × ${N_PROMPTS} prompts × ${N_TOKENS} tokens)..."
  python3 "${REPO_ROOT}/vllm/sndr_core/tools/genesis_bench_suite.py" \
    --host 127.0.0.1 --port "${PORT}" \
    --model "${SERVED_MODEL}" \
    --runs "${N_RUNS}" \
    --num-prompts "${N_PROMPTS}" \
    --max-tokens "${N_TOKENS}" \
    --json > "${RESULT_FILE}" \
    || { echo "    bench FAILED — see ${RESULT_FILE}"; }

  echo "    -> result saved to ${RESULT_FILE}"
  echo
done

echo "=========================================================="
echo "All 4 configs done — generating summary pivot"
echo "=========================================================="
python3 - <<'PYEND'
import glob, json, os, statistics
RESULTS_DIR = os.environ.get('RESULTS_DIR', os.path.expanduser('~/g4_tq_ab_results'))
print(f"\n{'config':<14} {'wall_TPS':>9} {'TPOT_ms':>9} {'TTFT_ms':>9} {'accept':>7} {'CV%':>6}")
print("-" * 65)
rows = []
for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
    name = os.path.splitext(os.path.basename(f))[0]
    try:
        data = json.load(open(f))
    except Exception:
        print(f"{name:<14}  <unparseable>")
        continue
    if data.get('status') == 'boot_timeout':
        print(f"{name:<14}  BOOT TIMEOUT")
        continue
    tps = data.get('wall_tps_mean') or data.get('wall_tps')
    tpot = data.get('tpot_ms_mean') or data.get('tpot_ms')
    ttft = data.get('ttft_ms_mean') or data.get('ttft_ms')
    accept = data.get('accept_rate_mean') or data.get('accept_rate', 0.0)
    cv = data.get('wall_tps_cv') or 0.0
    print(f"{name:<14} {tps:>9.2f} {tpot:>9.2f} {ttft:>9.1f} {accept:>7.3f} {cv*100 if cv<1 else cv:>5.1f}%")
    rows.append((name, tps, tpot, ttft, accept, cv))

# Find baseline (uint32_signs) for delta
baseline = next((r for r in rows if r[0] == 'uint32_signs'), None)
if baseline:
    print(f"\nΔ vs baseline (uint32_signs):")
    print(f"{'config':<14} {'ΔTPS':>10} {'ΔTPOT':>10}")
    for r in rows:
        if r[0] == 'uint32_signs': continue
        dtps = (r[1] - baseline[1]) / baseline[1] * 100
        dtpot = (r[2] - baseline[2]) / baseline[2] * 100
        print(f"{r[0]:<14} {dtps:>+9.2f}% {dtpot:>+9.2f}%")

print(f"\nResults dir: {RESULTS_DIR}")
print("Recommended next step: compare quality on a long-context prompt")
print("(needle-in-haystack 128K+, document QA) between uint32_signs and")
print("the configurations that gave acceptable TPS regression.")
PYEND

echo
echo "Done. See ${RESULTS_DIR}/ for raw JSON + apply logs."

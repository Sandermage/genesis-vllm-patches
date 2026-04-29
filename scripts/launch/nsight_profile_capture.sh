#!/bin/bash
# Nsight Systems first-ever PROD profile capture.
# Captures 5-minute trace of decode-heavy workload on test slot.
#
# Usage: bash scripts/launch/nsight_profile_capture.sh <run_name>
# Example: bash scripts/launch/nsight_profile_capture.sh prod_v775_2026_04_29
#
# Prereqs: server has nsys installed; container is running on test slot.
# Output: /home/sander/Genesis_Project/profiles/<run_name>.nsys-rep
#         + analyze locally with `nsys stats <run_name>.nsys-rep`

set -euo pipefail

RUN_NAME="${1:-prod_$(date -u +%Y%m%dT%H%M%SZ)}"
HOST="${HOST:-192.168.1.10}"
DURATION="${DURATION:-300}"  # 5 minutes
OUT_DIR="/home/sander/Genesis_Project/profiles"

ssh sander@${HOST} "
mkdir -p ${OUT_DIR}
nsys --version || { echo 'nsys not installed on server. Install: sudo apt install nsight-systems-2024.1'; exit 1; }
PID=\$(docker exec vllm-server-mtp-test pgrep -f 'vllm.entrypoints' | head -1)
if [ -z \"\$PID\" ]; then echo 'vLLM process not found in container'; exit 1; fi
echo \"Profiling PID \$PID for ${DURATION}s...\"
docker exec vllm-server-mtp-test bash -c \"
  nsys profile -t cuda,nvtx,osrt -o /tmp/${RUN_NAME} \\
    --duration ${DURATION} \\
    --capture-range=none --force-overwrite=true \\
    -p \$PID || echo 'attach mode requires CAP_SYS_ADMIN; falling back to launch-mode profile (start a fresh test)'
\"
docker cp vllm-server-mtp-test:/tmp/${RUN_NAME}.nsys-rep ${OUT_DIR}/${RUN_NAME}.nsys-rep
echo 'Profile saved at:' ${OUT_DIR}/${RUN_NAME}.nsys-rep
echo 'Analyze with: nsys stats ${OUT_DIR}/${RUN_NAME}.nsys-rep'
"

# Trigger workload during profile capture (separate ssh session)
echo "Profile capture in progress on server. Triggering decode workload..."
sleep 3
for i in $(seq 1 10); do
    curl -sS -X POST http://${HOST}:8000/v1/chat/completions \
        -H "Authorization: Bearer genesis-local" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Write a 500-word essay on quantum computing."}],"max_tokens":600}' \
        -o /dev/null -w "  workload req $i: %{http_code} (%{time_total}s)\n" 2>/dev/null
done
echo "Workload done. Wait for nsys to finish capture..."
echo "Then run: ssh sander@${HOST} \"nsys stats /home/sander/Genesis_Project/profiles/${RUN_NAME}.nsys-rep | head -50\""

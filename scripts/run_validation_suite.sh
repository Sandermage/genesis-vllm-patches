#!/usr/bin/env bash
# Genesis universal per-model validation runner — v7.62.x updated.
#
# Active model tags (v7.62.x):
#   qwen3_6_35b_fp8         — 35B-A3B-FP8 PROD (TQ k8v4 + MTP K=3 + PN8)
#   qwen3_6_27b_int4_short  — 27B-int4-Lorbus short-ctx (no TQ, fp8_e5m2)
#   qwen3_6_27b_int4_long   — 27B-int4-Lorbus long-ctx 256K (no TQ)
#   qwen3_6_27b_int4_TQ     — 27B-int4-Lorbus + TurboQuant k8v4 (P98 required)
# Cross-arch (compat verification, not PROD):
#   qwen3_next_awq | qwen3_32b_dense | gemma4_26b_moe
# Legacy aliases (still supported): qwen3_next_fp8 → qwen3_6_35b_fp8
#
# For performance benchmarks: tools/genesis_bench_suite.py
# This runner does CORRECTNESS validation (apply matrix, smoke tests, pytest).
#
# Usage:
#   ./scripts/run_validation_suite.sh <model_tag>
#
# Where <model_tag> is one of:
#   qwen3_next_fp8     — MoE + hybrid + TurboQuant k8v4 (prod baseline)
#   qwen3_next_awq     — MoE + hybrid + AWQ 4-bit (no TQ)
#   qwen3_32b_dense    — dense attention, no MoE, no hybrid, no TQ
#   gemma4_26b_moe     — MoE, not hybrid, no TQ (cross-arch MoE)
#
# Prereq per model: a docker-compose.<tag>.yml file + the model in HF cache.
# For prod (qwen3_next_fp8) — uses docker-compose.integration.yml.
#
# Writes structured results into:
#   benchmarks/v7_10_validation_20260424/<tag>/
#
# Each check is best-effort — the script continues on failure so you get
# partial results even if one stage breaks. Final summary.md tells you
# what passed / failed.

set -u

# ── args + validation ──────────────────────────────────────────────────
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <model_tag>"
    echo "Valid tags: qwen3_next_fp8 | qwen3_next_awq | qwen3_32b_dense | gemma4_26b_moe"
    exit 1
fi

MODEL_TAG="$1"
case "$MODEL_TAG" in
    qwen3_next_fp8|qwen3_next_awq|qwen3_32b_dense|gemma4_26b_moe) ;;
    *) echo "Unknown model tag: $MODEL_TAG"; exit 1 ;;
esac

# ── config per model ───────────────────────────────────────────────────
case "$MODEL_TAG" in
    qwen3_next_fp8|qwen3_6_35b_fp8)
        COMPOSE_FILE="docker-compose.integration.yml"
        CONTAINER="vllm-server-mtp-test"
        MODEL_NAME="qwen3.6-35b-a3b"
        MAX_CTX=262144
        SWEEP_FROM=50; SWEEP_TO=250; SWEEP_STEP=50
        STRESS_FROM=150; STRESS_TO=180; STRESS_STEP=15
        EXPECT_MOE=true; EXPECT_HYBRID=true; EXPECT_TQ=true
        EXPECT_P51_FIRES=false   # TQ active → P51 should NOT fire
        ;;
    qwen3_next_awq)
        COMPOSE_FILE="docker-compose.integration-awq.yml"
        CONTAINER="vllm-integration-awq"
        MODEL_NAME="qwen3.6-35b-a3b-awq-integration"
        MAX_CTX=262144
        SWEEP_FROM=50; SWEEP_TO=250; SWEEP_STEP=50
        STRESS_FROM=150; STRESS_TO=180; STRESS_STEP=15
        EXPECT_MOE=true; EXPECT_HYBRID=true; EXPECT_TQ=false
        EXPECT_P51_FIRES=true
        ;;
    qwen3_32b_dense)
        COMPOSE_FILE="docker-compose.qwen3-32b-dense.yml"
        CONTAINER="vllm-qwen3-32b-dense"
        MODEL_NAME="qwen3-32b-dense"
        MAX_CTX=32768
        SWEEP_FROM=4; SWEEP_TO=28; SWEEP_STEP=4
        STRESS_FROM=20; STRESS_TO=30; STRESS_STEP=5
        EXPECT_MOE=false; EXPECT_HYBRID=false; EXPECT_TQ=false
        EXPECT_P51_FIRES=true
        ;;
    gemma4_26b_moe)
        COMPOSE_FILE="docker-compose.gemma4-26b-moe.yml"
        CONTAINER="vllm-gemma4-26b-moe"
        MODEL_NAME="gemma4-26b-moe"
        MAX_CTX=32768
        SWEEP_FROM=4; SWEEP_TO=28; SWEEP_STEP=4
        STRESS_FROM=20; STRESS_TO=30; STRESS_STEP=5
        EXPECT_MOE=true; EXPECT_HYBRID=false; EXPECT_TQ=false
        EXPECT_P51_FIRES=true
        ;;
esac

HOST=${HOST:-localhost}
PORT=${PORT:-8000}
API_KEY=${API_KEY:-genesis-local}
OUT_ROOT="benchmarks/v7_10_validation_20260424/${MODEL_TAG}"
mkdir -p "$OUT_ROOT"

log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "${OUT_ROOT}/run.log"; }

log "════════════════════════════════════════════════════════════════════"
log "Genesis v7.10 validation — ${MODEL_TAG}"
log "Container: ${CONTAINER}  |  Model: ${MODEL_NAME}  |  Max ctx: ${MAX_CTX}"
log "Output: ${OUT_ROOT}/"
log "════════════════════════════════════════════════════════════════════"

# ── 1. Capture boot log ────────────────────────────────────────────────
log ""
log "=== 1. Boot log capture ==="
docker logs "${CONTAINER}" > "${OUT_ROOT}/boot.log" 2>&1 || \
    log "⚠ docker logs failed (container may not exist yet) — continuing"

grep -E "\[Genesis|Genesis Results:|\[P5[0-3]|model_detect" "${OUT_ROOT}/boot.log" \
    > "${OUT_ROOT}/apply_all.log" 2>/dev/null || true

BOOT_SUMMARY=$(grep -E "Genesis Results:" "${OUT_ROOT}/boot.log" | tail -1 || echo "NOT FOUND")
log "  Genesis summary: ${BOOT_SUMMARY}"

# ── 2. Dispatch profile from inside container ─────────────────────────
log ""
log "=== 2. Dispatch profile (model_detect) ==="
docker exec "${CONTAINER}" python3 -c "
import json
from vllm._genesis.model_detect import get_model_profile
print(json.dumps(get_model_profile(), indent=2, default=str))
" > "${OUT_ROOT}/dispatch_profile.json" 2>&1 || \
    log "⚠ dispatch_profile dump failed (container may not have v7.10 yet)"

if [[ -s "${OUT_ROOT}/dispatch_profile.json" ]]; then
    log "  Profile saved (see ${OUT_ROOT}/dispatch_profile.json)"
    grep -E '"(moe|hybrid|turboquant|model_type)":' "${OUT_ROOT}/dispatch_profile.json" | head -5 | sed 's/^/    /'
fi

# ── 3. Smoke test ──────────────────────────────────────────────────────
log ""
log "=== 3. Smoke test (10 requests @ 4k ctx) ==="
SMOKE_FILE="${OUT_ROOT}/smoke.jsonl"
: > "${SMOKE_FILE}"

for i in $(seq 1 10); do
    RESP=$(curl -s -w '\n{"http_status":%{http_code}}\n' \
        -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "{
            \"model\": \"${MODEL_NAME}\",
            \"messages\": [{\"role\":\"user\",\"content\":\"Count to 5 in English. Reply with only the numbers.\"}],
            \"max_tokens\": 64,
            \"temperature\": 0
        }" 2>&1)
    STATUS=$(echo "${RESP}" | tail -1 | python3 -c 'import sys,json; print(json.load(sys.stdin).get("http_status","?"))' 2>/dev/null || echo "?")
    HAS_OUTPUT=$(echo "${RESP}" | head -n -1 | grep -c '"content"' || echo 0)
    echo "{\"run\":${i},\"http_status\":\"${STATUS}\",\"has_output\":${HAS_OUTPUT}}" >> "${SMOKE_FILE}"
done

SMOKE_PASS=$(grep -c '"http_status":"200"' "${SMOKE_FILE}" || echo 0)
log "  Smoke: ${SMOKE_PASS}/10 OK"

# ── 4. Context sweep (full range) ──────────────────────────────────────
log ""
log "=== 4. Context sweep (${SWEEP_FROM}k..${SWEEP_TO}k step ${SWEEP_STEP}k) ==="
if [[ -f "./genesis_context_sweep.py" ]]; then
    python3 ./genesis_context_sweep.py \
        --host "http://${HOST}:${PORT}" --api-key "${API_KEY}" \
        --model "${MODEL_NAME}" \
        --from-k "${SWEEP_FROM}" --to-k "${SWEEP_TO}" --step-k "${SWEEP_STEP}" --runs 1 \
        --label "v7_10_${MODEL_TAG}_sweep" \
        --out "${OUT_ROOT}/context_sweep_full.jsonl" 2>&1 | tail -10 | sed 's/^/  /'
else
    log "⚠ genesis_context_sweep.py not found — skipping"
fi

# ── 5. Stress (Probe M — preempts upstream #40420 class) ──────────────
log ""
log "=== 5. Stress / Probe M (${STRESS_FROM}k..${STRESS_TO}k 3× each) ==="
if [[ -f "./genesis_context_sweep.py" ]]; then
    python3 ./genesis_context_sweep.py \
        --host "http://${HOST}:${PORT}" --api-key "${API_KEY}" \
        --model "${MODEL_NAME}" \
        --from-k "${STRESS_FROM}" --to-k "${STRESS_TO}" --step-k "${STRESS_STEP}" --runs 3 \
        --label "v7_10_${MODEL_TAG}_probe_m" \
        --out "${OUT_ROOT}/stress_probe_m.jsonl" 2>&1 | tail -10 | sed 's/^/  /'
fi

# ── 6. Speed bench @ 100k (only where applicable) ──────────────────────
log ""
log "=== 6. Speed bench @ 100k ==="
if [[ "${MAX_CTX}" -ge 100000 ]]; then
    if [[ -f "./genesis_context_sweep.py" ]]; then
        python3 ./genesis_context_sweep.py \
            --host "http://${HOST}:${PORT}" --api-key "${API_KEY}" \
            --model "${MODEL_NAME}" \
            --from-k 100 --to-k 100 --step-k 1 --runs 3 \
            --label "v7_10_${MODEL_TAG}_speed_100k" \
            --out "${OUT_ROOT}/speed_100k.jsonl" 2>&1 | tail -5 | sed 's/^/  /'
    fi
else
    log "  Skipped (model max_ctx=${MAX_CTX} < 100k) — using ctx=${STRESS_TO}k instead"
    if [[ -f "./genesis_context_sweep.py" ]]; then
        python3 ./genesis_context_sweep.py \
            --host "http://${HOST}:${PORT}" --api-key "${API_KEY}" \
            --model "${MODEL_NAME}" \
            --from-k "${STRESS_TO}" --to-k "${STRESS_TO}" --step-k 1 --runs 3 \
            --label "v7_10_${MODEL_TAG}_speed_low" \
            --out "${OUT_ROOT}/speed_100k.jsonl" 2>&1 | tail -5 | sed 's/^/  /'
    fi
fi

# ── 7. Memory profile (nvidia-smi before + after) ─────────────────────
log ""
log "=== 7. Memory profile ==="
MEM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")
log "  VRAM before stress: ${MEM_BEFORE} MiB"

# Trigger some load
for i in $(seq 1 50); do
    curl -s -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":4,\"temperature\":0}" \
        > /dev/null 2>&1
done

MEM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "N/A")
log "  VRAM after 50 reqs: ${MEM_AFTER} MiB"

cat > "${OUT_ROOT}/memory_profile.json" <<EOF
{
  "model_tag": "${MODEL_TAG}",
  "vram_before_mib": "${MEM_BEFORE}",
  "vram_after_50_reqs_mib": "${MEM_AFTER}",
  "delta_mib": "$((MEM_AFTER - MEM_BEFORE)) (may be N/A if nvidia-smi unavailable)"
}
EOF

# ── 8. Dispatch correctness cross-check ────────────────────────────────
log ""
log "=== 8. Dispatch correctness ==="
P51_COUNT=$(grep -c "\[P51 TQ-active\]" "${OUT_ROOT}/boot.log" 2>/dev/null || echo 0)
P52_SKIPS=$(grep -c "Genesis v7.9 dispatch.* skipped.*MoE\|no MoE" "${OUT_ROOT}/boot.log" 2>/dev/null || echo 0)
P53_SKIPS=$(grep -c "Genesis v7.9 dispatch.* skipped.*attention\|no GDN\|no hybrid" "${OUT_ROOT}/boot.log" 2>/dev/null || echo 0)

log "  P51 TQ-active fired: ${P51_COUNT} times (expected fires=${EXPECT_P51_FIRES})"
log "  P52 MoE-active skips: ${P52_SKIPS} (expected on dense model only)"
log "  P53 Hybrid-active skips: ${P53_SKIPS} (expected on pure-attention model only)"

# ── 9. Write summary ───────────────────────────────────────────────────
log ""
log "=== 9. Summary ==="
cat > "${OUT_ROOT}/summary.md" <<EOF
# Genesis v7.10 validation — ${MODEL_TAG}

**Date**: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
**Container**: ${CONTAINER}
**Model**: ${MODEL_NAME}
**Max ctx**: ${MAX_CTX}

## Results at a glance

| Check | Result |
|---|---|
| Boot (Genesis applied) | ${BOOT_SUMMARY} |
| Smoke 10/10 | ${SMOKE_PASS}/10 |
| Context sweep | see \`context_sweep_full.jsonl\` |
| Stress (Probe M) | see \`stress_probe_m.jsonl\` |
| Speed bench | see \`speed_100k.jsonl\` |
| Memory delta | before=${MEM_BEFORE} MiB, after=${MEM_AFTER} MiB |
| P51 fires | ${P51_COUNT} (expected: ${EXPECT_P51_FIRES}) |
| P52 skips | ${P52_SKIPS} |
| P53 skips | ${P53_SKIPS} |

## Expected profile

| Attr | Expected |
|---|---|
| moe | ${EXPECT_MOE} |
| hybrid | ${EXPECT_HYBRID} |
| turboquant | ${EXPECT_TQ} |

## Raw files

- [boot.log](./boot.log)
- [apply_all.log](./apply_all.log)
- [dispatch_profile.json](./dispatch_profile.json)
- [smoke.jsonl](./smoke.jsonl)
- [context_sweep_full.jsonl](./context_sweep_full.jsonl)
- [stress_probe_m.jsonl](./stress_probe_m.jsonl)
- [speed_100k.jsonl](./speed_100k.jsonl)
- [memory_profile.json](./memory_profile.json)
- [run.log](./run.log)
EOF

log ""
log "✅ Validation suite finished for ${MODEL_TAG}"
log "   Summary: ${OUT_ROOT}/summary.md"
log "   Raw: ${OUT_ROOT}/"

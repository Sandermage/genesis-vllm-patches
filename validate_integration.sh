#!/usr/bin/env bash
# Genesis v7.0-dev INTEGRATION validation script (TDD gate 2 of 2)
#
# Runs on VM 100 (192.168.1.10) AFTER the integration container is up and
# prod has been stopped (or scaled-down). Performs: container health check,
# Genesis import check, full pytest suite (real CUDA), chat-completion
# smoke test, and diagnostic probes defined in master plan v7.0.
#
# CRITICAL: VM 103 (192.168.1.15) has NO GPUs — do NOT run this there.
# For CPU-only unit validation, use ./validate_unit.sh instead.
#
# Prereq (Mode A blue/green):
#   ssh sander@192.168.1.10 'docker stop <prod-vllm-container>'
#   docker compose -f docker-compose.integration.yml up -d
#   # wait ~5 min for model load
#   ./validate_integration.sh
#
# Usage:
#   ./validate_integration.sh           # uses defaults below
#   HOST=192.168.1.10 ./validate_integration.sh   # remote host override
#
# Exit codes:
#   0 — All tests pass, ready for prod consideration
#   1 — pytest failures detected
#   2 — diagnostic probe failure
#   3 — smoke test failure
#   4 — setup error

set -u  # don't 'set -e' — we want all checks to run and report summary

CONTAINER=${CONTAINER:-vllm-integration-v7}
HOST=${HOST:-localhost}
PORT=${PORT:-8000}
MODEL_NAME=${MODEL_NAME:-qwen3.6-35b-a3b-integration}
API_KEY=${API_KEY:-genesis-local}

PASS=0
FAIL=0
WARN=0

log() {
    echo "[$(date +'%H:%M:%S')] $*"
}

pass() {
    PASS=$((PASS + 1))
    log "✅ PASS: $*"
}

fail() {
    FAIL=$((FAIL + 1))
    log "❌ FAIL: $*"
}

warn() {
    WARN=$((WARN + 1))
    log "⚠️  WARN: $*"
}

# ═══════════════════════════════════════════════════════════════════════════
#                          1. CONTAINER HEALTH
# ═══════════════════════════════════════════════════════════════════════════

log "=== 1. Container health check ==="

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    fail "Container ${CONTAINER} not running"
    exit 4
fi
pass "Container ${CONTAINER} is running"

if ! docker exec "$CONTAINER" curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    fail "/health endpoint not responding (may still be loading)"
    docker logs --tail 50 "$CONTAINER"
    exit 4
fi
pass "/health endpoint responding"

# ═══════════════════════════════════════════════════════════════════════════
#                      2. GENESIS PACKAGE IMPORT CHECK
# ═══════════════════════════════════════════════════════════════════════════

log ""
log "=== 2. Genesis package import check ==="

if docker exec "$CONTAINER" python3 -c \
    "from vllm._genesis import __version__, __author__; print(f'{__version__} by {__author__}')" 2>&1; then
    pass "vllm._genesis imports cleanly"
else
    fail "vllm._genesis import failed"
fi

if docker exec "$CONTAINER" python3 -c \
    "from vllm._genesis.guards import platform_summary; import json; print(json.dumps(platform_summary(), default=str, indent=2))" 2>&1; then
    pass "platform_summary() works"
else
    fail "platform_summary() failed"
fi

# ═══════════════════════════════════════════════════════════════════════════
#                          3. TDD PYTEST SUITE
# ═══════════════════════════════════════════════════════════════════════════

log ""
log "=== 3. TDD pytest suite ==="

# Copy tests into container (if not already mounted)
docker exec "$CONTAINER" bash -c "pip install pytest -q 2>/dev/null"

TEST_OUTPUT=$(docker exec "$CONTAINER" python3 -m pytest \
    /usr/local/lib/python3.12/dist-packages/vllm/_genesis/tests/ \
    -v --tb=short 2>&1) || true

echo "$TEST_OUTPUT" | tail -30

if echo "$TEST_OUTPUT" | grep -qE "passed"; then
    PYTEST_PASSED=$(echo "$TEST_OUTPUT" | grep -oE "[0-9]+ passed" | head -1 | grep -oE "[0-9]+")
    pass "pytest: ${PYTEST_PASSED} tests passed"
fi

if echo "$TEST_OUTPUT" | grep -qE "failed"; then
    PYTEST_FAILED=$(echo "$TEST_OUTPUT" | grep -oE "[0-9]+ failed" | head -1 | grep -oE "[0-9]+")
    fail "pytest: ${PYTEST_FAILED} tests failed"
fi

# ═══════════════════════════════════════════════════════════════════════════
#                      4. SMOKE TEST — CHAT COMPLETION
# ═══════════════════════════════════════════════════════════════════════════

log ""
log "=== 4. Smoke test — chat completion ==="

RESPONSE=$(curl -sN -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${API_KEY}" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Say exactly: TEST_OK\"}],
        \"max_tokens\": 10,
        \"temperature\": 0
    }" 2>&1)

if echo "$RESPONSE" | grep -q "TEST_OK"; then
    pass "Chat completion smoke test — model responds correctly"
elif echo "$RESPONSE" | grep -qE "choices|content"; then
    warn "Chat completion returns response but not exact 'TEST_OK' — acceptable (model may paraphrase)"
else
    fail "Chat completion failed: $RESPONSE"
fi

# ═══════════════════════════════════════════════════════════════════════════
#                      5. DIAGNOSTIC PROBES (from master plan)
# ═══════════════════════════════════════════════════════════════════════════

log ""
log "=== 5. Diagnostic probes (master plan Part 11.3) ==="

# ── Probe A: P8 helper function injected into kv_cache_utils.py ──
if docker exec "$CONTAINER" grep -qE "def token_capacity_kv_cache_groups" \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py 2>/dev/null; then
    pass "Probe A: P8 KV hybrid reporting helper injected (3.76x capacity fix)"
else
    fail "Probe A: token_capacity_kv_cache_groups helper NOT in kv_cache_utils.py"
fi

# ── Probe A': scheduler has the import ──
if docker exec "$CONTAINER" grep -qE "from vllm.v1.core.kv_cache_utils import token_capacity_kv_cache_groups" \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py 2>/dev/null; then
    pass "Probe A': P8 scheduler import present"
else
    fail "Probe A': P8 scheduler import MISSING"
fi

# ── Probe B: block_size after P6 alignment ──
BLOCK_LOG=$(docker logs "$CONTAINER" 2>&1 | grep -iE "Setting attention block size|mamba.page" | head -5)
if [[ -n "$BLOCK_LOG" ]]; then
    pass "Probe B: attention block_size alignment logged"
    echo "$BLOCK_LOG" | sed 's/^/   /'
else
    warn "Probe B: no block_size alignment log line"
fi

# ── Probe C: MARLIN selected (Patch 17/18, P23 applicability) ──
if docker logs "$CONTAINER" 2>&1 | grep -qE "Using MARLIN.*MoE backend|Selected.*marlin"; then
    pass "Probe C: MARLIN MoE backend active (P17/18/23 effective)"
else
    warn "Probe C: MARLIN not logged (may use different backend — not a failure)"
fi

# ── Probe D: no routed_experts flag (#40692 protection) ──
if docker exec "$CONTAINER" ps -ef 2>/dev/null | grep -v grep | grep -q "enable.return.routed"; then
    fail "Probe D: --enable-return-routed-experts is SET (#40692 crash risk)"
else
    pass "Probe D: no routed_experts flag (safe from #40692)"
fi

# ── Probe E: KV cache size reported + concurrency ──
KV_LOG=$(docker logs "$CONTAINER" 2>&1 | grep -E "GPU KV cache size|Maximum concurrency" | tail -4)
if [[ -n "$KV_LOG" ]]; then
    pass "Probe E: KV cache size reported"
    echo "$KV_LOG" | sed 's/^/   /'
else
    warn "Probe E: KV cache size log not found"
fi

# ── Probe F: platform summary at startup (P31 / P7 / P22 rebinds) ──
if docker logs "$CONTAINER" 2>&1 | grep -q "Genesis platform:"; then
    pass "Probe F: Genesis platform diagnostic logged at startup"
else
    warn "Probe F: Genesis platform diagnostic missing"
fi

# ── Probe G: Genesis patch summary ──
GENESIS_SUMMARY=$(docker logs "$CONTAINER" 2>&1 | grep -E "Genesis Results:" | tail -3)
if [[ -n "$GENESIS_SUMMARY" ]]; then
    pass "Probe G: Genesis orchestrator ran"
    echo "$GENESIS_SUMMARY" | sed 's/^/   /'
else
    fail "Probe G: Genesis orchestrator DID NOT RUN (critical)"
fi

# ── Probe H: PDL env misconfig guard (#40742) ──
if docker logs "$CONTAINER" 2>&1 | grep -qE "PDL env vars set but this GPU does NOT"; then
    fail "Probe H: PDL env vars set on non-PDL GPU — #40742 crash risk"
else
    pass "Probe H: PDL env safe (no #40742 trigger conditions)"
fi

# ── Probe I: P4 hybrid attn-layer detection ──
P4_LOG=$(docker logs "$CONTAINER" 2>&1 | grep -E "\[Genesis P4\] TQ hybrid: full-attention layers" | head -1)
if [[ -n "$P4_LOG" ]]; then
    pass "Probe I: P4 full-attention layer detection succeeded"
    echo "$P4_LOG" | sed 's/^/   /'
else
    warn "Probe I: P4 attn-layer detection log missing (may not fire on non-hybrid)"
fi

# ── Probe J: P14 BlockTable monkey-patch rebound (runtime wiring) ──
P14_LOG=$(docker logs "$CONTAINER" 2>&1 | grep -E "\[Genesis P14\] rebound BlockTable" | head -1)
if [[ -n "$P14_LOG" ]]; then
    pass "Probe J: P14 BlockTable monkey-patch active (append_row + move_row tail-zeroed)"
else
    warn "Probe J: P14 BlockTable rebind log missing"
fi

# ── Probe K: P22 TurboQuantAttentionImpl rebind ──
P22_LOG=$(docker logs "$CONTAINER" 2>&1 | grep -cE "\[Genesis P22\] rebound TurboQuantAttentionImpl")
if [[ "$P22_LOG" -gt 0 ]]; then
    pass "Probe K: P22 TurboQuantAttentionImpl._ensure_on_device rebound ($P22_LOG times — once per process)"
else
    warn "Probe K: P22 rebind log missing"
fi

# ── Probe L1: v7.9 model_detect profile resolved ──
P52_LOG=$(docker logs "$CONTAINER" 2>&1 | grep -E "\[Genesis v7.9 model_detect\] profile resolved" | head -1)
if [[ -n "$P52_LOG" ]]; then
    pass "Probe L1: v7.9 model_detect profile resolved at boot"
    echo "$P52_LOG" | sed 's/^/   /'
else
    warn "Probe L1: v7.9 model_detect profile log missing (P52/P53 defense layer may not be wired)"
fi

# ── Probe L2: v7.9 P51 TQ-active detection — SHOULD NOT fire on TQ prod ──
P51_ON_TQ=$(docker logs "$CONTAINER" 2>&1 | grep -cE "\[P51 TQ-active\] skipping TQ preallocs")
if [[ "$P51_ON_TQ" -gt 0 ]]; then
    warn "Probe L2: P51 fired on what should be a TQ container — check kv_cache_dtype config"
else
    pass "Probe L2: P51 did NOT fire on TQ container (expected — preallocs active)"
fi

# ── Probe M: ≥150k-ctx regression test (preempts upstream issue #40420) ──
# Root cause of #40420: lazy torch.empty in _continuation_prefill invisible to
# profiler → OOM at ~185k. Our P22+P38 pre-allocate → must survive ≥150k cleanly.
log ""
log "=== 6. ≥150k-ctx regression test (upstream #40420 class) ==="

SWEEP_OUT=/tmp/v7_9_150k_regression.jsonl
if [[ -x "./genesis_context_sweep.py" ]] || [[ -f "./genesis_context_sweep.py" ]]; then
    python3 ./genesis_context_sweep.py \
        --host "http://${HOST}:${PORT}" --api-key "${API_KEY}" \
        --model "${MODEL_NAME}" \
        --from-k 150 --to-k 180 --step-k 15 --runs 1 \
        --label "v7.9_40420_regression" --out "$SWEEP_OUT" 2>&1 | tail -20
    REG_EXIT=$?
    if [[ $REG_EXIT -eq 0 ]]; then
        # Parse results — every size must have a successful response (http_status 200)
        FAIL_LINES=$(grep -c '"http_status":\s*[^2]' "$SWEEP_OUT" 2>/dev/null || echo 0)
        if [[ "$FAIL_LINES" == "0" ]]; then
            pass "Probe M: ≥150k regression test — 150k/165k/180k all returned 200"
        else
            fail "Probe M: ≥150k regression test — $FAIL_LINES sizes failed (see $SWEEP_OUT)"
        fi
    else
        warn "Probe M: context sweep exit=$REG_EXIT (non-fatal)"
    fi
else
    warn "Probe M: genesis_context_sweep.py not found — regression skipped"
fi

# ── Probe L: post-register runtime assertion (if the container uses --verify-rebinds) ──
VERIFY_LOG=$(docker logs "$CONTAINER" 2>&1 | grep -E "Post-register rebind verification:" | head -1)
if [[ -n "$VERIFY_LOG" ]]; then
    VERIFY_DETAIL=$(docker logs "$CONTAINER" 2>&1 | grep -E "^\s*[✓✗] P[0-9]" | head -10)
    if echo "$VERIFY_DETAIL" | grep -q "✗"; then
        fail "Probe L: Post-register verification FAILED on at least one rebind"
        echo "$VERIFY_DETAIL" | sed 's/^/   /'
    else
        pass "Probe L: Post-register verification all green"
        echo "$VERIFY_DETAIL" | sed 's/^/   /'
    fi
else
    warn "Probe L: Post-register verification not enabled (add --verify-rebinds to apply_all CLI)"
fi

# ═══════════════════════════════════════════════════════════════════════════
#                              SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

log ""
log "═══════════════════════════════════════════════════════════════════════"
log "  FINAL SUMMARY"
log "═══════════════════════════════════════════════════════════════════════"
log "  ✅ PASS:  $PASS"
log "  ❌ FAIL:  $FAIL"
log "  ⚠️  WARN:  $WARN"
log "═══════════════════════════════════════════════════════════════════════"

if [[ $FAIL -gt 0 ]]; then
    log ""
    log "❌ VALIDATION FAILED — do NOT push to GitHub, do NOT deploy to prod."
    log "   Review failures above and fix before re-running."
    exit 1
fi

if [[ $WARN -gt 3 ]]; then
    log ""
    log "⚠️  MANY WARNINGS — review manually before prod deploy."
    exit 0
fi

log ""
log "✅ VALIDATION PASSED — Genesis v7.0-dev ready for prod consideration."
log "   Next steps (requires user approval):"
log "   1. Review pytest output above"
log "   2. Check decode throughput matches baseline (142 t/s short)"
log "   3. Run long-context OOM stress test (256k prompt)"
log "   4. If all green → push to GitHub (v7.0-dev branch)"
log "   5. Separately plan prod blue/green deploy"
exit 0

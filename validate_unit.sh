#!/usr/bin/env bash
# Genesis UNIT validation — CPU-only, runs anywhere with Docker.
#
# What this does (vs other validation tools):
#   - validate_unit.sh           THIS — CPU-only Python pytest suite (~30 sec)
#   - validate_integration.sh    Real CUDA + container health + chat smoke + pytest
#   - tools/genesis_bench_suite.py  Performance benchmark (TPS / TTFT / ctx probe)
#
# Use this script as your fastest sanity check after touching Genesis patch
# Python code (no GPU needed). It runs the unit pytest suite inside a
# transient Docker container.
#
# Usage:
#   ./validate_unit.sh
#
# Exit codes:
#   0 — All pytest tests pass
#   1 — pytest failures
#   2 — Docker / container setup error

set -u

log() {
    echo "[$(date +'%H:%M:%S')] $*"
}

log "=== Genesis v7.0-dev UNIT validation (CPU-only) ==="

if ! command -v docker >/dev/null 2>&1; then
    log "❌ docker not installed"
    exit 2
fi

if ! docker compose version >/dev/null 2>&1; then
    log "❌ docker compose plugin not available"
    exit 2
fi

# Run the unit suite (one-shot, --rm cleans up after)
if docker compose -f docker-compose.unit.yml run --rm genesis-unit; then
    log "✅ UNIT validation PASSED"
    log ""
    log "Next: TDD gate 2 — integration validation on a GPU host"
    log "   ./validate_integration.sh  # requires prod downtime window"
    exit 0
else
    log "❌ UNIT validation FAILED — see pytest output above"
    exit 1
fi

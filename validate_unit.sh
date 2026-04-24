#!/usr/bin/env bash
# Genesis v7.0-dev UNIT validation — CPU-only, runs anywhere (prefer VM 103).
#
# This is TDD gate 1 of 2: validates the Python layer without touching real
# vLLM engine or GPUs. Fast (~30 sec) and can run on any Docker host.
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
    log "Next: TDD gate 2 — integration validation on VM 100 (GPUs)"
    log "   ./validate_integration.sh  # requires prod downtime window"
    exit 0
else
    log "❌ UNIT validation FAILED — see pytest output above"
    exit 1
fi

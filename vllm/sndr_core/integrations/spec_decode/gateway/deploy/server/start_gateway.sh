#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Launcher: SNDR spec-decode gateway (D2b).
#
# Builds (idempotent) and runs the gateway container. Expects the
# default and structured vLLM containers to be on the same docker
# bridge network (or reachable via host networking).
#
# Env overrides:
#   GATEWAY_IMAGE=sndr-gateway:dev
#   GATEWAY_PORT=8100
#   DEFAULT_URL=http://localhost:8101
#   STRUCTURED_URL=http://localhost:8102
#   PROFILE=gemma4-tq-mtp-structured-k4
#   ADMIN_ALLOW_REMOTE=0   (set to 1 if calling /admin from outside container)
#
# This script uses host networking so the gateway can reach the two
# vLLM containers (which are also on host net) via localhost. Adapt
# to your topology if you use a docker bridge.
set -euo pipefail

GENESIS_REPO="${GENESIS_REPO:-/home/sander/genesis-vllm-patches}"
GATEWAY_IMAGE="${GATEWAY_IMAGE:-sndr-gateway:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-sndr-gateway}"
GATEWAY_PORT="${GATEWAY_PORT:-8100}"
DEFAULT_URL="${DEFAULT_URL:-http://localhost:8101}"
STRUCTURED_URL="${STRUCTURED_URL:-http://localhost:8102}"
PROFILE="${PROFILE:-gemma4-tq-mtp-structured-k4}"
ADMIN_ALLOW_REMOTE="${ADMIN_ALLOW_REMOTE:-0}"

# Build (idempotent — Docker caches layers; cheap if nothing changed).
docker build \
    -t "${GATEWAY_IMAGE}" \
    -f "${GENESIS_REPO}/vllm/sndr_core/integrations/spec_decode/gateway/deploy/Dockerfile" \
    "${GENESIS_REPO}"

docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

docker run -d \
    --name "${CONTAINER_NAME}" \
    --network host \
    -e SNDR_GATEWAY_DEFAULT_URL="${DEFAULT_URL}" \
    -e SNDR_GATEWAY_STRUCTURED_URL="${STRUCTURED_URL}" \
    -e SNDR_GATEWAY_PROFILE="${PROFILE}" \
    -e SNDR_GATEWAY_BIND_PORT="${GATEWAY_PORT}" \
    -e SNDR_GATEWAY_BIND_HOST=0.0.0.0 \
    -e SNDR_GATEWAY_HEALTH_INTERVAL=5 \
    -e SNDR_GATEWAY_TIMEOUT=120 \
    -e SNDR_GATEWAY_LOG_LEVEL=INFO \
    -e SNDR_GATEWAY_ADMIN_ALLOW_REMOTE="${ADMIN_ALLOW_REMOTE}" \
    "${GATEWAY_IMAGE}"

echo "${CONTAINER_NAME} on port ${GATEWAY_PORT}"
echo "  default upstream:    ${DEFAULT_URL}"
echo "  structured upstream: ${STRUCTURED_URL}"
echo "  profile:             ${PROFILE}"

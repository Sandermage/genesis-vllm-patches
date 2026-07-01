#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Deploy (or redeploy) the genesis-memory UNIFIED daemon with full Control Center
# visibility. Reproducible: run this and the container comes up identically.
#
# What the flags buy you:
#   -v /var/run/docker.sock  -> container management + host inventory (the daemon
#                               talks the Docker Engine API directly over the
#                               socket; no docker CLI needed in the image).
#   --device nvidia.com/gpu   -> GPU telemetry: nvidia-smi (injected via CDI) so
#                               the Hardware panel shows the cards + utilisation.
#   SNDR_OPENAI_BASE_URL/...   -> the engine the GUI auto-connects to. The MODEL,
#                               version and live KPIs are read FROM the engine
#                               (/v1/models + /metrics) — only the address is set
#                               here; everything model-dependent auto-detects.
#
# Secrets (memory API key, gateway upstreams, PG password) are NOT baked into the
# image — they are carried over from the currently-running container, or supplied
# via $ENV_FILE. Override any VAR below via the environment.
set -euo pipefail

IMAGE="${IMAGE:-genesis-memory:dev}"
NAME="${NAME:-genesis-memory}"
PORT="${PORT:-8811}"                                   # host port -> :8800 in the container
NET="${NET:-genesis_project_genesis}"                  # reach the engine + cliproxy by name
PGVOL="${PGVOL:-genesis_memory_pgdata}"
DOCKER_SOCK="${DOCKER_SOCK:-/var/run/docker.sock}"
GPU_DEVICE="${GPU_DEVICE:-nvidia.com/gpu=all}"         # CDI; set empty to disable GPU telemetry

# Engine the GUI auto-connects to (model/version/KPIs auto-detect from it).
ENGINE_BASE="${SNDR_OPENAI_BASE_URL:-http://vllm-qwen3.6-35b-balanced-k3:8102/v1}"
ENGINE_METRICS="${SNDR_METRICS_URL:-${ENGINE_BASE%/v1}/metrics}"
ENGINE_KEY="${SNDR_ENGINE_API_KEY:-genesis-local}"
# The reported engine host MUST match the base_url host (the container name the
# daemon resolves), so the chat — which proxies through the daemon — adopts a
# host the daemon can actually reach (not 127.0.0.1). Derive it from the base URL.
ENGINE_HOST="${SNDR_RUNTIME_HOST:-$(printf '%s' "$ENGINE_BASE" | sed -E 's#^[a-z]+://([^:/]+).*#\1#')}"

# Carry secrets/config forward from the running container, unless $ENV_FILE is set.
ENV_FILE="${ENV_FILE:-/tmp/genesis-memory.env}"
if [ -z "${ENV_FILE_PROVIDED:-}" ] && docker inspect "$NAME" >/dev/null 2>&1; then
    docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' \
        | grep -E '^(GENESIS_|GATEWAY_|POSTGRES_|HF_HOME=|SNDR_)' > "$ENV_FILE" || true
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

gpu_args=()
[ -n "$GPU_DEVICE" ] && gpu_args=(--device "$GPU_DEVICE")
envfile_args=()
[ -f "$ENV_FILE" ] && envfile_args=(--env-file "$ENV_FILE")

docker run -d --name "$NAME" --restart unless-stopped \
    -p "${PORT}:8800" \
    -v "${PGVOL}:/var/lib/postgresql/data" \
    -v "${DOCKER_SOCK}:/var/run/docker.sock" \
    "${gpu_args[@]}" \
    "${envfile_args[@]}" \
    -e "SNDR_OPENAI_BASE_URL=${ENGINE_BASE}" \
    -e "SNDR_METRICS_URL=${ENGINE_METRICS}" \
    -e "SNDR_ENGINE_API_KEY=${ENGINE_KEY}" \
    -e "SNDR_RUNTIME_HOST=${ENGINE_HOST}" \
    "$IMAGE" >/dev/null

docker network connect "$NET" "$NAME" 2>/dev/null || true

printf 'genesis-memory up on :%s — ' "$PORT"
for _ in $(seq 1 60); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' -m3 "http://127.0.0.1:${PORT}/api/v1/health" 2>/dev/null)" = "200" ]; then
        echo "healthy"; exit 0
    fi
    sleep 1
done
echo "did NOT become healthy in 60s — check: docker logs $NAME" >&2
exit 1

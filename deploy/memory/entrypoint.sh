#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start co-located Postgres (via the base image's entrypoint, which runs initdb
# + creates POSTGRES_USER/DB on first boot), wait for readiness, then exec the
# product-API as the container's foreground process.
set -euo pipefail

# Launch Postgres in the background using the pgvector/postgres base entrypoint.
docker-entrypoint.sh postgres &
PG_PID=$!

# Wait for Postgres to accept connections (initdb on a fresh volume takes a bit).
for _ in $(seq 1 90); do
    if pg_isready -h 127.0.0.1 -p 5432 -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$PG_PID" 2>/dev/null; then
        echo "postgres exited during startup" >&2
        exit 1
    fi
    sleep 1
done

echo "postgres ready; starting product-API on :8800"
# Schema is created idempotently by the API's PostgresStore at startup.
exec python3 -m uvicorn "sndr.product_api.server:create_app" \
    --factory --host 0.0.0.0 --port 8800

# SPDX-License-Identifier: Apache-2.0
# Unified genesis-memory image: Postgres + pgvector + the product-API (which
# serves /api/v1/memory/* and, when a build is present, the GUI) in ONE
# container — unified config, no cross-container overhead (operator's ask).
#
# The vLLM engine stays a separate GPU container; this image is CPU-only.
#
# Base = the pgvector image already on the server (Postgres 15 + pgvector 0.8).
# No supervisord: the base entrypoint initialises Postgres on first boot, then
# our entrypoint waits for it and launches uvicorn as the foreground process.
FROM pgvector/pgvector:0.8.0-pg15-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install third-party deps in their own layer (cached across source changes) —
# the [gui-api] + [memory] extras = fastapi/uvicorn/psycopg (no vllm).
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    "pyyaml>=6.0" "packaging>=23.0" \
    "fastapi>=0.110" "uvicorn[standard]>=0.27" "h11>=0.16.0" \
    "psycopg[binary]>=3.1"

# Then the package itself (--no-deps): only this fast layer re-runs on a code change.
COPY . /app
RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps .

# The API talks to the co-located Postgres over loopback. DIM must match the
# embedder; 256 = HashEmbedder default. Override at run time as needed.
ENV GENESIS_MEMORY_DSN="postgresql://genesis:genesis@127.0.0.1:5432/genesis_memory" \
    GENESIS_MEMORY_DIM="256" \
    POSTGRES_USER="genesis" \
    POSTGRES_PASSWORD="genesis" \
    POSTGRES_DB="genesis_memory"

COPY deploy/memory/entrypoint.sh /usr/local/bin/genesis-memory-entrypoint.sh
RUN chmod +x /usr/local/bin/genesis-memory-entrypoint.sh

EXPOSE 8800
ENTRYPOINT ["/usr/local/bin/genesis-memory-entrypoint.sh"]

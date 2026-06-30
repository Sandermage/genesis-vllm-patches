# SPDX-License-Identifier: Apache-2.0
# Unified genesis-memory image: Postgres + pgvector + the product-API + the GUI
# in ONE container — unified config, no cross-container overhead (operator's ask).
# The vLLM engine stays a separate GPU container; this image is CPU-only.

# ---- Stage 1: build the React/Vite GUI ----
FROM node:22-slim AS gui
WORKDIR /gui
COPY gui/web/package.json gui/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY gui/web/ ./
RUN npm run build   # -> /gui/dist (index.html + assets)

# ---- Stage 2: runtime (Postgres 15 + pgvector 0.8 base, already on the server) ----
FROM pgvector/pgvector:0.8.0-pg15-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Third-party deps in their own cached layer (no vllm).
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    "pyyaml>=6.0" "packaging>=23.0" \
    "fastapi>=0.110" "uvicorn[standard]>=0.27" "h11>=0.16.0" \
    "psycopg[binary]>=3.1"

# Then the package itself (--no-deps): only this fast layer re-runs on a code change.
COPY . /app
RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps .

# The built GUI, served same-origin by the product-API (one core).
COPY --from=gui /gui/dist /app/gui-static

ENV GENESIS_MEMORY_DSN="postgresql://genesis:genesis@127.0.0.1:5432/genesis_memory" \
    GENESIS_MEMORY_DIM="256" \
    POSTGRES_USER="genesis" \
    POSTGRES_PASSWORD="genesis" \
    POSTGRES_DB="genesis_memory" \
    SNDR_GUI_STATIC_CARBON="/app/gui-static"

COPY deploy/memory/entrypoint.sh /usr/local/bin/genesis-memory-entrypoint.sh
RUN chmod +x /usr/local/bin/genesis-memory-entrypoint.sh

EXPOSE 8800
ENTRYPOINT ["/usr/local/bin/genesis-memory-entrypoint.sh"]

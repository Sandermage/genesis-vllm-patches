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
RUN npm run build \
 && printf 'docker-%s\n' "$(ls -1 /gui/dist/assets 2>/dev/null | sha1sum | cut -c1-12)" > /gui/dist/build-id.txt
# build-id.txt (content hash of the emitted assets) is served from SNDR_GUI_STATIC
# and surfaced via /api/v1/health.gui_build_id; the SPA polls it for auto-reload.

# ---- Stage 2: runtime (Postgres 15 + pgvector 0.8 base, already on the server) ----
FROM pgvector/pgvector:0.8.0-pg15-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Third-party deps in their own cached layer (no vllm; model2vec is CPU/numpy,
# no torch at inference).
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    "pyyaml>=6.0" "packaging>=23.0" \
    "fastapi>=0.110" "uvicorn[standard]>=0.27" "h11>=0.16.0" "httpx>=0.27" \
    "psycopg[binary]>=3.1" "model2vec>=0.3"

# Bake the static embedder model into the image so runtime is fully offline.
ENV HF_HOME="/app/hf-cache"
RUN python3 -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M')"

# Then the package itself (--no-deps): only this fast layer re-runs on a code change.
COPY . /app
RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps .

# The built GUI, served same-origin by the product-API (one core).
COPY --from=gui /gui/dist /app/gui-static

# Postgres is loopback-only inside the container (port 5432 is never published),
# so trust auth on localhost is safe and avoids baking any secret into the image.
ENV GENESIS_MEMORY_DSN="postgresql://genesis@127.0.0.1:5432/genesis_memory" \
    GENESIS_MEMORY_EMBEDDER="model2vec" \
    POSTGRES_USER="genesis" \
    POSTGRES_DB="genesis_memory" \
    POSTGRES_HOST_AUTH_METHOD="trust" \
    GENESIS_MEMORY_MAINTENANCE_INTERVAL="1800" \
    GENESIS_MEMORY_MAX_NODES="20000" \
    SNDR_GUI_STATIC_CARBON="/app/gui-static" \
    SNDR_GUI_STATIC="/app/gui-static" \
    SNDR_AUTH="off"

COPY deploy/memory/entrypoint.sh /usr/local/bin/genesis-memory-entrypoint.sh
RUN chmod +x /usr/local/bin/genesis-memory-entrypoint.sh

EXPOSE 8800
ENTRYPOINT ["/usr/local/bin/genesis-memory-entrypoint.sh"]

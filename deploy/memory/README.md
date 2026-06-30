# genesis-memory — unified container

Postgres + pgvector + the product-API (serving `/api/v1/memory/*`, and the GUI
when a build is present) in **one** container. Unified config, no cross-container
overhead. The vLLM engine stays a separate GPU container.

## Build

From the repo root (the Dockerfile uses the repo as build context):

```bash
docker build -f deploy/memory/Dockerfile -t genesis-memory:dev .
```

## Run

```bash
docker run -d --name genesis-memory \
  -p 8800:8800 \
  -v genesis_memory_pgdata:/var/lib/postgresql/data \
  --restart unless-stopped \
  genesis-memory:dev
```

- API: `http://<host>:8800/api/v1/memory/...` (and `/api/v1/health`).
- Postgres data persists in the `genesis_memory_pgdata` volume.
- Owner scoping: send `X-Owner-Id: <id>` (the proxy middleware sets this).

## Smoke

```bash
curl -s localhost:8800/api/v1/health
curl -s -XPOST localhost:8800/api/v1/memory/remember \
  -H 'X-Owner-Id: 1' -H 'content-type: application/json' \
  -d '{"text":"postgres vector memory graph"}'
curl -s 'localhost:8800/api/v1/memory/search?q=postgres+memory' -H 'X-Owner-Id: 1'
curl -s localhost:8800/api/v1/memory/stats -H 'X-Owner-Id: 1'
```

## Config (env)

| Var | Default | Meaning |
|---|---|---|
| `GENESIS_MEMORY_DSN` | `postgresql://genesis@127.0.0.1:5432/genesis_memory` | co-located Postgres (loopback) |
| `GENESIS_MEMORY_DIM` | `256` | embedding dim (must match the `vector(dim)` column / embedder) |
| `POSTGRES_USER` / `POSTGRES_DB` | `genesis` / `genesis_memory` | first-boot DB init |
| `POSTGRES_HOST_AUTH_METHOD` | `trust` | loopback-only PG (port 5432 never published) → no baked secret |

## Memory gateway (universal augment for all models)

The same container exposes an OpenAI-compatible `POST /v1/chat/completions` that
transparently adds memory to ANY model — recall+inject before the upstream,
capture after. Point clients at the gateway; it forwards to the upstream you set:

| Var | Example | Meaning |
|---|---|---|
| `GATEWAY_UPSTREAM_URL` | `http://cliproxy:8317/v1` (external) · `http://127.0.0.1:8102/v1` (internal vLLM) | OpenAI-compatible upstream |
| `GATEWAY_UPSTREAM_KEY` | `<bearer>` | token the upstream expects (optional) |

Flow: `client → :8811/v1/chat/completions → recall+inject → UPSTREAM → capture → client`
(SSE streaming is teed through and the reply reassembled for capture). The
gateway stays dormant (503) until `GATEWAY_UPSTREAM_URL` is set.

**CLIProxyAPI (external models), unmodified** — run the stock image and bind it
private; our gateway is the only client:

```yaml
# CLIProxyAPI config.yaml
host: "127.0.0.1"
port: 8317
api-keys: ["<shared-secret>"]          # GATEWAY_UPSTREAM_KEY
disable-claude-cloak-mode: true          # don't let cloak clobber our system block
# providers / openai-compatibility / auth-dir as usual
```

Owner scoping: clients send `X-Owner-Id: <id>` (the per-user memory). External
models need no special support — recalled memory arrives as plain system text.

## Notes

- No supervisord: the pgvector base image's entrypoint initialises Postgres on
  first boot; `entrypoint.sh` waits for readiness, then `exec`s uvicorn as the
  foreground process. The schema is created idempotently by `PostgresStore`.
- pgvectorscale (StreamingDiskANN) is a later upgrade: swap the base image for
  `timescale/timescaledb-ha` when vectors outgrow RAM / exceed the HNSW limits.

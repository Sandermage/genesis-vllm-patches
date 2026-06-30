# Genesis Memory — Complete Manual

The persistent **neural-graph memory** for Genesis: a brain-like knowledge store
that makes every model (internal vLLM and external via the proxy) smarter over
time. Knowledge is stored as nodes that auto-form connections and cluster into
"clouds", recalled by vector + spreading activation, decays/reinforces like human
memory, and is exposed over an HTTP API, an OpenAI-compatible gateway, and a GUI
graph panel — all in one container.

> Design rationale (why Postgres+pgvector, why not Apache AGE/sqlite, the verified
> research) lives in `docs/design/memory-engine-production-design.md`,
> `memory-neural-graph-mode.md`, `persistent-memory-architecture.md`. This manual
> is the **operational + developer reference** for what is built.

## Contents
1. [Architecture](#1-architecture)
2. [Core concepts & brain mechanics](#2-core-concepts--brain-mechanics)
3. [Backends & embedders](#3-backends--embedders)
4. [HTTP API reference](#4-http-api-reference)
5. [The memory gateway (universal augment)](#5-the-memory-gateway)
6. [Programmatic use (Python)](#6-programmatic-use-python)
7. [Obsidian import](#7-obsidian-import)
8. [Maintenance & consolidation](#8-maintenance--consolidation)
9. [Configuration reference (all env vars)](#9-configuration-reference)
10. [Deployment (the unified container)](#10-deployment)
11. [Security](#11-security)
12. [GUI panel](#12-gui-panel)
13. [Operations & troubleshooting](#13-operations--troubleshooting)
14. [Scale-up paths](#14-scale-up-paths)
15. [Implementation status & tests](#15-implementation-status--tests)

---

## 1. Architecture

One CPU-only container = **Postgres + pgvector + the product-API + the GUI +
the memory gateway**. The vLLM GPU engine stays a separate container.

```
                    ┌─────────────────── genesis-memory container ───────────────────┐
   client / proxy   │  product-API (uvicorn :8800)                                    │
   ───────────────► │   ├─ /api/v1/memory/*   (CRUD + recall + graph + import)        │
                    │   ├─ /v1/chat/completions  (gateway: augment→upstream→capture)  │
                    │   └─ GUI (Sigma.js graph), served same-origin                   │
                    │         │                                                        │
                    │   MemoryEngine ── Embedder (Model2Vec / Hash)                    │
                    │         │                                                        │
                    │   MemoryStore ── Postgres+pgvector (or in-memory)                │
                    │   background: maintenance thread (consolidate + prune)           │
                    └─────────────────────────────────────────────────────────────────┘
        gateway upstream ─►  CLIProxyAPI (external models)  |  vLLM (internal 35B)
```

**Layers** (all in `sndr/memory/` + `sndr/product_api/`):
- `model.py` — `MemoryNode`, `MemoryEdge`, `SearchHit` + tuned constants.
- `store.py` — `MemoryStore` ABC (the contract) + the **shared** brain-recall
  algorithm + retention (so every backend is numerically identical).
- `inmemory.py` — pure-stdlib reference backend (tests + dev default).
- `postgres.py` — `PostgresStore` (production: psycopg + pgvector).
- `embedder.py` — `Embedder` ABC, `HashEmbedder` (dep-free), `Model2VecEmbedder` (real CPU).
- `engine.py` — `MemoryEngine` (text facade) + `run_maintenance`.
- `middleware.py` — `ConversationMemory` (augment/capture).
- `client.py` — `MemoryHTTPClient` (drive memory from another process).
- `gateway.py` — OpenAI-response helpers (assistant-text extraction + SSE
  reassembly) used by the gateway route.
- `obsidian.py` — `import_vault`.
- `product_api/routes/{memory,gateway}.py`, `schemas/memory.py`, `security.py`.

---

## 2. Core concepts & brain mechanics

A **node** (`mem_node`) is a memory atom: `id, owner_id, kind, content,
embedding, importance, strength, access_count, community_id, properties,
created_at, accessed_at`. An **edge** (`mem_edge`) connects two nodes:
`src_id, dst_id, rel, weight, properties, valid_at, invalid_at`.

| Mechanic | What it does | Tuning |
|---|---|---|
| **Semantic auto-link** | kNN over embeddings → `similar_to` edges (the connections) | cosine ≥ τ (default 0.8) |
| **Hebbian co-access** | co-recalled nodes wire together: `w ← min(1,(1−λ)w+η)` | η=0.02, λ=0.995 |
| **Ebbinghaus decay** | retention `R = exp(−age / (S·strength·(1+importance)))` at read | S=86400s (1 day) |
| **Strength reinforcement** | each retrieval: `strength = 1+ln(1+access_count)` → slower decay (spacing effect) | — |
| **Importance** | heuristic, weighted-degree + access (hub > leaf) | recomputed in consolidate |
| **Communities ("clouds")** | label propagation over the graph → `community_id` (graph colors) | deterministic, ≤20 iters |
| **Spreading-activation recall** | ANN seeds → bounded cycle-safe expand, score ×weight×β per hop | β=0.5, depth≤3 |
| **Bi-temporal invalidation** | retire an edge (`invalid_at`) without deleting — excluded from traversal | — |
| **Prune** | salience-ranked eviction to a per-owner cap (leak-bound) | salience = importance+retention+0.1·access |

**Recall** is two-phase (the same Python algorithm on both backends):
1. ANN search for the query → seed activations (cosine).
2. Bounded, cycle-safe graph expand: each hop multiplies activation by
   `edge.weight · β`; the max activation per node wins.
3. Blend with lazy decay (`activation · retention`), drop non-positive, return
   top-N. Returned nodes are **touched** (reinforced) and, optionally, their
   mutual `co_access` edges strengthened.

---

## 3. Backends & embedders

**Backends** (same `MemoryStore` contract, verified identical in CI):
- **In-memory** (`InMemoryStore`) — default when `GENESIS_MEMORY_DSN` unset;
  pure stdlib, ephemeral; the test double + dev backend.
- **Postgres + pgvector** (`PostgresStore`) — production. HNSW ANN index
  (`m=16, ef_construction=200`), lexical GIN index, `iterative_scan='relaxed_order'`
  + `ef_search=100` for filtered ANN. Vectors passed as `::vector` literals (only
  `psycopg` needed). Timestamps are epoch-seconds for deterministic decay.

**Embedders** (`Embedder` ABC — text → vector):
- **`HashEmbedder`** (default, dep-free) — deterministic feature hashing,
  configurable dim (`GENESIS_MEMORY_DIM`, default 256). Low quality but real.
- **`Model2VecEmbedder`** — real static CPU embedder (`minishlab/potion-base-8M`,
  256-dim, ~30 MB, no torch). Set `GENESIS_MEMORY_EMBEDDER=model2vec`. Measured
  semantic quality: related ≈ 0.85 vs unrelated ≈ 0.01.

The store's vector dim is **derived from the embedder**, so column and embedder
never drift. HNSW caps at 2000 dims; for larger embedders see [§14](#14-scale-up-paths).

---

## 4. HTTP API reference

Base: `http://<host>:8811`. All `/api/v1/memory/*` and `/v1/*` routes are
**owner-scoped** (`X-Owner-Id` header, default `1`) and **guarded** by the API key
when `GENESIS_MEMORY_API_KEY` is set (`Authorization: Bearer <key>` or
`X-Api-Key: <key>`). `/api/v1/health` and `/api/v1/version` are public.

Responses are enveloped: `{"data": <payload>, "meta": {request_id, timestamp, ...}}`.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| POST | `/api/v1/memory/remember` | `{text, kind?, importance?, properties?, dedup?}` | `{id}` |
| GET | `/api/v1/memory/search` | `?q=&limit=10&mode=vector\|hybrid` | `[{id,content,kind,score}]` |
| POST | `/api/v1/memory/recall` | `{query, limit?, expand_depth?, reinforce?}` | `[{id,content,kind,score}]` |
| GET | `/api/v1/memory/node/{id}` | — | full node |
| GET | `/api/v1/memory/neighbors/{id}` | — | `[{id,rel,weight}]` |
| GET | `/api/v1/memory/stats` | — | `{nodes,edges}` |
| GET | `/api/v1/memory/graph` | `?limit=200` | `{nodes:[…],edges:[…]}` |
| POST | `/api/v1/memory/link` | `{tau?,k?}` | `{created}` |
| POST | `/api/v1/memory/consolidate` | `{tau?,k?}` | `{linked,communities,nodes}` |
| POST | `/api/v1/memory/edge/invalidate` | `{src,dst,rel}` | `{invalidated}` |
| POST | `/api/v1/memory/import/obsidian` | `{path}` | `{notes,links,missing}` |
| POST | `/v1/chat/completions` | OpenAI body (+`X-Memory-Upstream?`) | OpenAI response (stream/non) |
| GET | `/v1/upstreams` | — | `{upstreams:[…],default}` |

### Examples (curl)

```bash
H='-H X-Owner-Id:1 -H content-type:application/json'
KEY='-H Authorization:Bearer YOURKEY'   # omit if GENESIS_MEMORY_API_KEY unset

# Remember (deduped by exact content)
curl -s $KEY $H -XPOST localhost:8811/api/v1/memory/remember \
  -d '{"text":"The deploy server is 192.168.1.10, memory on port 8811."}'
# -> {"data":{"id":1}, ...}

# Pure ANN search
curl -s $KEY -H X-Owner-Id:1 'localhost:8811/api/v1/memory/search?q=deploy+server&limit=5'
# Hybrid (vector + keyword — catches exact terms/IDs)
curl -s $KEY -H X-Owner-Id:1 'localhost:8811/api/v1/memory/search?q=8811&mode=hybrid'

# Brain recall (graph expand + reinforce)
curl -s $KEY $H -XPOST localhost:8811/api/v1/memory/recall \
  -d '{"query":"where do we deploy","limit":10,"expand_depth":2}'

# Build connections + clusters (the "nightly batch", on demand)
curl -s $KEY $H -XPOST localhost:8811/api/v1/memory/consolidate -d '{"tau":0.8}'
# -> {"data":{"linked":N,"communities":K,"nodes":M}}

# Graph for visualization
curl -s $KEY -H X-Owner-Id:1 'localhost:8811/api/v1/memory/graph?limit=300'

# Retire a stale/contradicted edge (kept for audit, excluded from recall)
curl -s $KEY $H -XPOST localhost:8811/api/v1/memory/edge/invalidate \
  -d '{"src":1,"dst":2,"rel":"similar_to"}'
```

---

## 5. The memory gateway

An OpenAI-compatible `POST /v1/chat/completions` that transparently adds memory to
**any** model: it recalls the owner's relevant memories, injects them as a
plain-text system block (no tool-calls / model-specific format), forwards to the
chosen upstream, returns the response (streaming or not), and captures the turn.

**Multi-upstream — choose per request.** Configure a registry of named upstreams;
select with `X-Memory-Upstream: <name>` (else the default):

```bash
# clients point at the gateway instead of the upstream
curl -s $KEY -H X-Owner-Id:1 -H 'X-Memory-Upstream: cliproxy' \
  -H content-type:application/json -XPOST localhost:8811/v1/chat/completions \
  -d '{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"what is my deploy server?"}]}'

curl -s $KEY localhost:8811/v1/upstreams      # -> {"upstreams":["cliproxy","local"],"default":"local"}
```

- **External models** → upstream = **CLIProxyAPI** (unmodified): it routes to
  Claude/Gemini/Codex/etc. The model receives recalled memory as plain context;
  no special support needed.
- **Internal models** → upstream = the **vLLM** OpenAI server (the 35B). Proven:
  the 35B answers from injected memory.

Errors map cleanly (502 on upstream failure, 504 on timeout; failed turns are not
captured). Streaming responses are teed to the client while the reply is
reassembled for capture.

See [§9](#9-configuration-reference) for `GATEWAY_UPSTREAMS` and the CLIProxyAPI
config (`config.yaml`: bind private, `api-keys:`, `disable-claude-cloak-mode: true`).

---

## 6. Programmatic use (Python)

```python
from sndr.memory import (
    MemoryEngine, InMemoryStore, HashEmbedder, Model2VecEmbedder,
    ConversationMemory, MemoryHTTPClient, import_vault,
)

# In-process engine
eng = MemoryEngine(store=InMemoryStore(), embedder=Model2VecEmbedder())
nid = eng.remember(owner_id=1, text="Genesis runs the 35B on 2x A5000.")
hits = eng.recall(owner_id=1, query="what GPUs?", limit=5)
eng.consolidate(owner_id=1)            # link + communities + importance

# Universal augment/capture for ANY model (in-process or over HTTP)
cm = ConversationMemory(engine=eng)
messages = cm.augment(owner_id=1, messages=[{"role":"user","content":"which GPUs?"}])
#   -> prepends/merges a "Relevant memory:" system block
# ... call the model with `messages` ...
cm.capture(owner_id=1, messages=messages, assistant="You run 2x A5000.")

# Drive a REMOTE memory service (the proxy path) — same shape, over HTTP:
client = MemoryHTTPClient("http://server:8811", owner_id=1, token="YOURKEY")
ConversationMemory(engine=client).augment(owner_id=1, messages=messages)
```

`MemoryEngine`: `remember`, `recall`, `search`, `search_hybrid`, `link_semantic`,
`detect_communities`, `recompute_importance`, `consolidate`, `graph`. Module
`run_maintenance(engine, max_nodes=...)` runs consolidate+prune for all owners.

---

## 7. Obsidian import

An Obsidian vault is already a graph — the importer maps it 1:1:
- each `.md` note → a memory node (`kind="note"`, `properties.title`=file stem,
  `properties.tags`=`#tags`, `properties.source="obsidian"`), deduped by content;
- each `[[wikilink]]` → a `wikilink` edge (alias `|` and `#heading` stripped;
  self-links and unknown targets skipped, counted as `missing`);
- idempotent re-import.

```bash
# requires GENESIS_MEMORY_VAULT_ROOT set + the vault mounted into the container
curl -s $KEY -H X-Owner-Id:1 -H content-type:application/json \
  -XPOST localhost:8811/api/v1/memory/import/obsidian -d '{"path":"my-vault"}'
# -> {"data":{"notes":120,"links":340,"missing":12}}
```

**Path is confined** to `GENESIS_MEMORY_VAULT_ROOT` (disabled when unset; `../`
escape → 403; in-vault symlinks resolving outside the root are skipped). After
import, run `consolidate` to add semantic links + communities on top of the
hand-built wikilink graph.

To import your real vault: `docker run ... -v /path/to/vault:/vaults -e
GENESIS_MEMORY_VAULT_ROOT=/vaults ...` then `POST {"path":"."}` (or a subdir).

---

## 8. Maintenance & consolidation

**Consolidate** (`POST /api/v1/memory/consolidate` or `engine.consolidate`) =
semantic auto-link → community detection → importance recompute. The GUI "Rebuild"
button calls it.

**Background scheduler** (the leak-bound, wired): when
`GENESIS_MEMORY_MAINTENANCE_INTERVAL` > 0, a daemon thread runs
`run_maintenance` every interval for **every** owner — consolidate **and prune to
`GENESIS_MEMORY_MAX_NODES`**. So memory stays bounded automatically and the graph
self-organizes. Logs (via the uvicorn logger) show
`memory maintenance: owners=… pruned=…`. Container defaults: interval 1800 s,
cap 20000 nodes/owner.

---

## 9. Configuration reference

All via environment variables.

### Memory engine
| Var | Default | Meaning |
|---|---|---|
| `GENESIS_MEMORY_DSN` | _(unset → in-memory)_ | Postgres DSN, e.g. `postgresql://genesis@127.0.0.1:5432/genesis_memory` |
| `GENESIS_MEMORY_EMBEDDER` | `hash` | `hash` (dep-free) or `model2vec` (real CPU) |
| `GENESIS_MEMORY_MODEL` | `minishlab/potion-base-8M` | model2vec model id |
| `GENESIS_MEMORY_DIM` | `256` | HashEmbedder dim (ignored for model2vec — derived) |
| `GENESIS_MEMORY_API_KEY` | _(unset → open)_ | require this bearer/X-Api-Key on memory+gateway routes |
| `GENESIS_MEMORY_VAULT_ROOT` | _(unset → import disabled)_ | allowed root for Obsidian import |
| `GENESIS_MEMORY_MAINTENANCE_INTERVAL` | `0` (off) | seconds between auto consolidate+prune passes |
| `GENESIS_MEMORY_MAX_NODES` | `10000` | per-owner node cap (prune target) |

### Gateway
| Var | Default | Meaning |
|---|---|---|
| `GATEWAY_UPSTREAMS` | _(unset)_ | JSON: `{"name":{"url":"…/v1","key":"…"}, …}` |
| `GATEWAY_DEFAULT_UPSTREAM` | _(first / `default`)_ | upstream used without `X-Memory-Upstream` |
| `GATEWAY_UPSTREAM_URL` / `GATEWAY_UPSTREAM_KEY` | _(unset)_ | single-upstream shortcut (registered as `default`) |

### Container / Postgres / GUI
| Var | Default | Meaning |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_DB` | `genesis` / `genesis_memory` | first-boot DB init |
| `POSTGRES_HOST_AUTH_METHOD` | `trust` | loopback-only PG (5432 not published) → no baked secret |
| `SNDR_GUI_STATIC_CARBON` | `/app/gui-static` | built GUI dir served by the API |

### Request headers
`X-Owner-Id` (owner scope) · `Authorization: Bearer`/`X-Api-Key` (when key set) ·
`X-Memory-Upstream` (gateway upstream selection).

---

## 10. Deployment

The unified image (`deploy/memory/Dockerfile`): a node stage builds the GUI; the
runtime stage is `pgvector/pgvector:0.8.0-pg15-trixie` + Python + the package +
Model2Vec baked. No supervisord — the base entrypoint inits Postgres, then
`entrypoint.sh` waits and `exec`s uvicorn.

```bash
# build (from repo root)
docker build -f deploy/memory/Dockerfile -t genesis-memory:dev .

# run — memory + GUI + gateway, all upstreams, hardened
docker run -d --name genesis-memory -p 8811:8800 \
  -v genesis_memory_pgdata:/var/lib/postgresql/data \
  --network <shared-net-with-cliproxy-and-vllm> \
  -e GENESIS_MEMORY_EMBEDDER=model2vec \
  -e GENESIS_MEMORY_API_KEY="$(openssl rand -base64 24)" \
  -e GENESIS_MEMORY_MAINTENANCE_INTERVAL=1800 \
  -e GATEWAY_UPSTREAMS='{"cliproxy":{"url":"http://cliproxyapi:8317/v1","key":"<K>"},"local":{"url":"http://vllm:8102/v1","key":"genesis-local"}}' \
  -e GATEWAY_DEFAULT_UPSTREAM=local \
  --restart unless-stopped genesis-memory:dev
```

**Networking:** to reach CLIProxyAPI / vLLM by name, put genesis-memory on the
same docker network (`docker network connect <net> genesis-memory`, and the vLLM
container too if it's on the default bridge). Smoke: `curl localhost:8811/api/v1/health`.

---

## 11. Security

- **Auth:** set `GENESIS_MEMORY_API_KEY` for any non-localhost exposure — it guards
  every memory + gateway route (constant-time compare; health stays public). The
  container publishes on the LAN, so **set it in production**.
- **Owner scoping:** every row carries `owner_id`; queries filter on it (app-layer
  enforcement) and the API derives it from `X-Owner-Id`. (Postgres RLS is **not**
  used — it conflicts with the cross-owner maintenance/stats ops; app-layer +
  the API key is the isolation model. See design §9.)
- **Vault confinement:** Obsidian import is confined to `GENESIS_MEMORY_VAULT_ROOT`
  with `../`-escape and symlink-escape protection.
- **Secrets:** the container's Postgres uses loopback `trust` (5432 not published)
  → no DB password baked. Upstream/API keys are passed via env (standard); keep
  them out of committed files.
- **SQL safety:** all identifiers composed with `psycopg.sql`; values parameterized.

---

## 12. GUI panel

The "Memory" section (Engine group, Brain icon) of the Control Center, served
same-origin by the container:
- **List view** — remember, search (toggle "Brain recall" for graph expand),
  click a result to inspect its node + connections, "Rebuild" to consolidate.
- **Graph view** — Obsidian-like force-directed graph (graphology + ForceAtlas2):
  nodes colored by community ("clouds"), sized by importance/access, edges by
  Hebbian weight; hover to label, click to inspect. Refreshes after rebuild.

---

## 13. Operations & troubleshooting

- **Health:** `GET /api/v1/health` (public). Gateway dormant → `/v1/chat/completions`
  returns 503 until an upstream is configured.
- **401 on memory routes** → `GENESIS_MEMORY_API_KEY` is set; send the bearer/X-Api-Key.
- **Gateway 502/504** → the selected upstream is unreachable/timed out; check
  `X-Memory-Upstream`, the URL/key, and that genesis-memory shares a network with it.
- **Postgres down** → the app degrades to the ephemeral in-memory backend (logged
  `postgres_unavailable_fallback_inmemory`); fix the DSN/DB and restart for persistence.
- **HNSW upkeep** (large/churny deployments): `REINDEX INDEX CONCURRENTLY` then
  `VACUUM (ANALYZE)` on `mem_node` periodically; keep the ANN index in `shared_buffers`.
- **Backup:** `pg_dump` covers nodes+edges+vectors (one DB); the restore target
  needs the `vector` extension installed first.
- **Logs:** `docker logs genesis-memory` shows uvicorn access + maintenance passes.

---

## 14. Scale-up paths

The current build targets a homelab (single owner, ≤ tens of thousands of nodes).
For larger scale, the design documents these (not wired by default):

- **Async connection pool** — `PostgresStore` is a single connection + lock
  (correct, but serializes under concurrent async load; fine at single-user scale).
  Upgrade: a `psycopg_pool` (sync pool + `def` routes in FastAPI's threadpool) or a
  full `AsyncConnectionPool`. The brain logic is backend-agnostic, so only the
  store's connection handling changes.
- **pgvectorscale / StreamingDiskANN** — for > 1M vectors or > 2000-dim embedders
  (Qwen3-Embedding-4B/8B): swap the base image to `timescale/timescaledb-ha`
  (ships pgvector + pgvectorscale) and create a `diskann` index on a full-precision
  `vector` column instead of HNSW. ≤ 2000-dim stays on HNSW.
- **alembic migrations** — the schema is created idempotently by
  `PostgresStore.ensure_schema`; for managed evolution, add an alembic baseline
  (register the `vector` type) and run migrations instead.
- **RLS** — intentionally not used (cross-owner maintenance/stats conflict); revisit
  only for true multi-tenant DB-enforced isolation, which would also require
  per-owner connections and reworking the cross-owner ops.

---

## 15. Implementation status & tests

**Built + tested** (in-memory + live Postgres, CI-gated via the `memory-postgres`
job): schema + HNSW/GIN indexes + ANN tuning; all brain mechanics
(Hebbian/decay/strength/communities/importance/recall/prune); hybrid search;
dedup; bi-temporal invalidation; consolidation + the wired maintenance scheduler;
the full HTTP API; the multi-upstream gateway (stream + non-stream); API-key auth;
graceful Postgres-down + upstream-error handling; Obsidian import; both embedders;
the unified container; the GUI panel.

**Deferred** (see [§14](#14-scale-up-paths)): async pool, pgvectorscale, alembic;
RLS decided-against; Leiden replaced by label propagation.

**Test surface:** ~90 memory tests (`tests/unit/test_memory_*`,
`test_gateway_route.py`, `test_obsidian_import.py`, `test_memory_auth.py`) + a
leak-soak (`tests/soak/`), run on both backends — the Postgres backend is exercised
against a live pgvector in CI. Run locally:

```bash
pytest tests/unit/test_memory_*.py tests/unit/test_gateway_route.py \
       tests/unit/test_obsidian_import.py tests/unit/test_memory_auth.py
# against a live Postgres+pgvector:
MEMORY_TEST_DSN=postgresql://genesis:pw@host:5432/db pytest tests/unit/test_memory_store_contract.py
```

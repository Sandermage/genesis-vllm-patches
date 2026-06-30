# Memory engine — production design (Postgres + pgvector + pgvectorscale, no graph DB)

Status: design / spec (no build yet). Supersedes the storage sections of
`memory-engine-lean-build.md` and `memory-neural-graph-mode.md §1b`. The brain-mechanics, viz,
and API of those docs are unchanged and storage-agnostic — this doc pins the concrete store.

Grounded in a verified study of how production memory systems actually persist on Postgres
(sources inline). Investigation discipline: Study → Analyze → Verify → Search → Compare → THEN design.

---

## 0. Decision

**Stack = Postgres 17 + pgvector + pgvectorscale. No sqlite-vec. No Apache AGE. No external graph DB.**
The knowledge graph is two ordinary relational tables (`mem_node`, `mem_edge`) with JSONB
properties, traversed by `WITH RECURSIVE`; vectors live in pgvector and are accelerated by
pgvectorscale StreamingDiskANN when they outgrow RAM. Graph analytics (Leiden communities,
PageRank/importance) run in an off-line batch via Python `igraph`/`leidenalg`, not in the DB.

### Why we dropped Apache AGE (revised from the earlier recommendation)

Deeper study reversed the initial "+ Apache AGE" call. Evidence:

- **The leading OSS memory systems on Postgres all avoid AGE.** Mem0 = pure pgvector
  (`id UUID, vector, payload JSONB`, HNSW default) + a second pgvector table for entity *nodes*,
  no edges, no AGE. Letta = pure relational + pgvector (`Vector(4096)` column, no graph DB, no
  Cypher — grep of the repo for `apache_age|ag_catalog|cypher|create_graph` returns zero). Cognee's
  Postgres graph backend = two relational tables `graph_node`/`graph_edge` (JSONB props, B-tree on
  source/target), and it **explicitly refuses raw Cypher** ("use Neo4j/Ladybug for that"). All can
  run vector+graph on one Postgres without AGE. Zep/Graphiti and Microsoft GraphRAG don't use AGE
  either (Neo4j and Parquet-flat-files respectively).
- **AGE is rare in shipped products** (mostly tutorials and "also-supported" secondary backends).
  The one detailed shipped-production case (Trendyol) **bypassed AGE Cypher** for native SQL writes
  and fixed-depth relational traversal because variable-length `[r*..N]` queries degraded from "low
  hundreds of ms" (Neo4j) to **"3–5 seconds"** on AGE (the `*` wildcard "bypasses indexes
  entirely"). AGE is also not available on AWS RDS, and PG17 support trailed upstream by ~a year.
- **AGE doesn't even simplify the vector half.** You cannot pgvector-index a vector stored inside an
  AGE vertex (it lives in the `agtype` column, unindexable; the pgvector-in-AGE proposal #1121 was
  closed stale). Every AGE+pgvector project keeps embeddings in a **sidecar `vector(N)` table** and
  JOINs it to `cypher()` on a business key — i.e. you maintain the same relational vector table you'd
  have without AGE, *plus* the agtype casting tax (`(x::text)::int`, JSON-quoted strings), the
  per-session `LOAD`/`search_path` that breaks under PgBouncer transaction pooling, parameterized
  Cypher needing `PREPARE` (also pooling-hostile), planner ignoring GIN-on-property (#1009), and
  **no native graph algorithms** (Leiden/PageRank are DIY regardless).
- **Our scale doesn't need it.** A documented "no-Neo4j" personal KG on plain `WITH RECURSIVE` +
  pgvector measures depth-2 expansion < 10 ms, a 100-node subgraph < 20 ms, holding to ~10K nodes /
  50K edges. Memory retrieval is overwhelmingly shallow (1–3 hop: "neighbors of this fact",
  "entities linked to this conversation") — exactly the zone where recursive SQL wins and where the
  node id is a normal `bigint` so the vector↔graph join is trivial (no agtype tax).
- **Simpler ops.** Dropping AGE means the official `timescale/timescaledb-ha` image (already ships
  pgvector + pgvectorscale) needs **no custom build layer** — one image, one process, `pg_dump`
  covers everything.

We keep a thin storage interface (so unit tests use an in-memory double and LadybugDB stays a
future escape hatch if we ever need true Cypher), but the shipped backend is Postgres only.

---

## 1. Schema (concrete DDL)

Embedding dimension **1024** (bge-m3 or Qwen3-Embedding-0.6B) — chosen to stay *under* pgvector's
hard 2000-dim HNSW ceiling (every index tuple must fit an 8 KB page; >2000 dims errors out), which
keeps full `iterative_scan` + `halfvec` available, ~5 GB index per 1M vectors, and is the Matryoshka
sweet spot before quality degrades (<1024 drops fast). Escalation path: commit to Qwen3-4B (2560) /
8B (4096) only by switching that table's index to StreamingDiskANN on full-precision `vector`
(DiskANN supports up to 16000 dims; note DiskANN does NOT support `halfvec`).

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vectorscale;   -- pgvectorscale (StreamingDiskANN, SBQ)

-- NODES: one memory atom (note, fact, entity, message-derived knowledge)
CREATE TABLE mem_node (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id     BIGINT NOT NULL,                 -- tenant / user scope (see section 5)
    kind         TEXT   NOT NULL,                 -- 'note' | 'entity' | 'fact' | 'summary'
    content      TEXT   NOT NULL,
    embedding    vector(1024),                    -- the only vector column; ANN-indexed
    importance   REAL   NOT NULL DEFAULT 0.0,     -- Generative-Agents importance (LLM-rated, batch)
    strength     REAL   NOT NULL DEFAULT 1.0,     -- Ebbinghaus retention base (decayed at read)
    access_count INT    NOT NULL DEFAULT 0,
    community_id INT,                             -- Leiden cluster ("cloud"); set by batch
    properties   JSONB  NOT NULL DEFAULT '{}',    -- tags, source model, wikilinks, etc.
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    accessed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- EDGES: the graph. relational, FK-enforced, JSONB props (Cognee graph_node/graph_edge pattern)
CREATE TABLE mem_edge (
    src_id   BIGINT NOT NULL REFERENCES mem_node(id) ON DELETE CASCADE,
    dst_id   BIGINT NOT NULL REFERENCES mem_node(id) ON DELETE CASCADE,
    rel      TEXT   NOT NULL,                      -- 'similar_to'|'co_access'|'depends_on'|'wikilink'
    weight   REAL   NOT NULL DEFAULT 0.0,          -- Hebbian weight [0,1]; updated at co-access
    properties JSONB NOT NULL DEFAULT '{}',
    valid_at   TIMESTAMPTZ NOT NULL DEFAULT now(), -- bi-temporal (Graphiti-style invalidation)
    invalid_at TIMESTAMPTZ,
    PRIMARY KEY (src_id, dst_id, rel)
);
```

### Indexes

```sql
-- Vector ANN (1024 dims -> HNSW; bump ef_construction well above the weak default of 64)
CREATE INDEX mem_node_hnsw ON mem_node
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
--  >2000-dim escalation instead:
--  CREATE INDEX mem_node_diskann ON mem_node USING diskann (embedding vector_cosine_ops);

-- Owner scoping: composite B-tree + partial indexes for hot owners (11x smaller / 20x faster build)
CREATE INDEX mem_node_owner ON mem_node (owner_id, kind);
-- CREATE INDEX mem_node_hot_owner ON mem_node USING hnsw (embedding vector_cosine_ops)
--     WHERE owner_id = <hot>;

-- Graph traversal joins (critical) + property lookups + clusters
CREATE INDEX mem_edge_src ON mem_edge (src_id) INCLUDE (dst_id, rel, weight);
CREATE INDEX mem_edge_dst ON mem_edge (dst_id) INCLUDE (src_id, rel, weight);
CREATE INDEX mem_node_props_gin ON mem_node USING gin (properties);
CREATE INDEX mem_node_community ON mem_node (owner_id, community_id);
```

---

## 2. Brain-mechanics — deterministic, in SQL, off the hot path

All four edge-formation mechanisms (semantic kNN, Hebbian co-access, typed relations, wikilinks)
and decay match `memory-neural-graph-mode.md section 2`; here is how they map to this store.

**Lazy Ebbinghaus decay at read** (no background job rewriting rows — leak-free, O(touched)):
effective retention = `strength * exp(-age_seconds / (S * importance_boost))`, computed in the
SELECT; only an UPDATE of `accessed_at`/`access_count` on actual retrieval.

**Hebbian co-access** — after a retrieval returns a set of node ids, one batched UPSERT strengthens
edges between co-retrieved nodes (HeLa-Mem: `w <- (1-lambda)*w + eta`, eta=0.02, lambda=0.995):

```sql
INSERT INTO mem_edge (src_id, dst_id, rel, weight)
SELECT a, b, 'co_access', 0.02
FROM unnest($1::bigint[]) a, unnest($1::bigint[]) b
WHERE a < b
ON CONFLICT (src_id, dst_id, rel)
DO UPDATE SET weight = LEAST(1.0, mem_edge.weight * 0.995 + 0.02);
```

This write goes through the off-hot-path write-back queue (section 4), never blocking the response.

**Semantic auto-edges** (kNN, cosine >= 0.8) and **typed relations** (LLM triples) + **Leiden
communities** + **PageRank importance** run in the nightly batch (the T3 tier on an idle A5000 or
on CPU): pull `(src,dst,weight)` into Python `igraph`, run `leidenalg`, write `community_id` back;
prune by salience cap. Leiden (not Louvain — Louvain yields disconnected communities). No graph
algorithms run inside Postgres.

---

## 3. Retrieval — two-phase (ANN first, then bounded graph expand)

The verified rule: never bury the `<=>` operator inside the same plan as a deep traversal (HNSW
won't parallelize with recursion). ANN-search in its own CTE -> feed the small id set into a
depth-bounded recursive expand -> score -> return subgraph.

```sql
SET LOCAL hnsw.iterative_scan = 'relaxed_order';   -- so owner-filter can't starve the candidate set
SET LOCAL hnsw.ef_search = 200;

WITH seeds AS (                                     -- phase 1: ANN, owner-scoped
    SELECT id, 1.0 - (embedding <=> $query_vec) AS sim
    FROM mem_node
    WHERE owner_id = $owner
    ORDER BY embedding <=> $query_vec
    LIMIT 15
),
expanded AS (                                       -- phase 2: bounded graph walk (<=3 hops)
    SELECT s.id, 0 AS depth, s.sim AS score, ARRAY[s.id] AS path
    FROM seeds s
    UNION ALL
    SELECT e.dst_id, x.depth + 1,
           x.score * e.weight * 0.5,                -- spreading activation damping beta ~ 0.5
           x.path || e.dst_id
    FROM expanded x
    JOIN mem_edge e ON e.src_id = x.id AND e.invalid_at IS NULL
    WHERE x.depth < 3
      AND NOT (e.dst_id = ANY(x.path))              -- cycle-safe
)
SELECT n.*, max(x.score) AS activation
FROM expanded x JOIN mem_node n ON n.id = x.id
GROUP BY n.id
ORDER BY activation DESC
LIMIT 30;
```

Final ranking blends activation with lazy-decay retention and importance (Generative-Agents
recency·importance·relevance) in the app layer before injection.

---

## 4. Connection pooling + async write path

- **Driver:** `psycopg3` async pool — adapts `prepare_threshold` gracefully behind a pooler
  (asyncpg is faster absolutely but its server-side prepared statements break under PgBouncer
  transaction mode unless `statement_cache_size=0` / PgBouncer >=1.21 `max_prepared_statements`).
- **Write-back queue:** embedding + node/edge writes are enqueued (in-process asyncio queue) and
  flushed in batches — they must never block the proxy's streamed LLM response. Batch/`COPY`
  inserts (AWS: up to 67x faster than row-at-a-time for embeddings).
- **Pooler mode:** transaction pooling (required by RLS, see section 5), NOT statement pooling.

---

## 5. Multi-tenant isolation (owner scoping)

- **RLS** on `mem_node`/`mem_edge` filtering by `current_setting('app.owner_id')`, **plus** an
  app-layer `WHERE owner_id = ...` (defense-in-depth — 2025 guidance warns RLS-alone is fragile under
  pooler misconfig). The composite `(owner_id, ...)` index serves both.
- **Pooler rule:** RLS depends on per-session `SET app.owner_id` -> use **transaction pooling**
  (set it at the start of each transaction); never statement pooling.
- **At scale:** partition `mem_node` by `owner_id` (partition pruning + smaller per-partition vector
  indexes). For pgvectorscale, owner can also be a `SMALLINT[] labels` column for Filtered DiskANN.

---

## 6. Ops & deployment (the melochi)

- **Docker:** base on `timescale/timescaledb-ha` (ships pgvector + pgvectorscale; no AGE build
  needed). Verify `CREATE EXTENSION vector; CREATE EXTENSION vectorscale;`.
- **Migrations (alembic):** register the type so autogenerate sees it —
  `connection.dialect.ischema_names['vector'] = pgvector.sqlalchemy.Vector`; `CREATE EXTENSION` in
  an early revision; hand-review every autogenerated revision.
- **HNSW maintenance:** HNSW degrades under churn (graph bloat + working set spilling RAM). Cadence:
  **`REINDEX INDEX CONCURRENTLY` (atomic swap, no downtime) then `VACUUM (ANALYZE)` weekly**,
  rate-limited by index size. Monitor index-in-RAM (the classic "fast on day 1, slow at month 3").
- **Resource tuning (32–128 GB box):** `shared_buffers ~ 25%` RAM, `effective_cache_size ~ 50–75%`;
  build-time `maintenance_work_mem >= 2x final index size` (min 2 GB), `max_parallel_maintenance_workers >= 4`,
  `/dev/shm >= maintenance_work_mem` in Docker. The ANN index must fit `shared_buffers`/page cache
  or latency falls off a cliff — this is the lever where StreamingDiskANN (disk-resident, ~9x
  compression) earns its place once vectors outgrow RAM.
- **Backup:** `pg_dump`/PITR covers nodes+edges+vectors in one DB (single-process advantage). Restore
  target must have `vector`+`vectorscale` installed before restore (extensions aren't in the dump).

---

## 7. Integration (unchanged — storage-agnostic)

Proxy memory-middleware (universal plain-text capture/inject for Gemma/Qwen/external — pattern (a),
not tool-calls) -> this store; owner-scoped
`/api/v1/memory/{search,subgraph,node,neighbors,communities,stream}` on the product-API (LightRAG
shape); React `Memory.tsx` panel with Sigma.js + graphology (ForceAtlas2 + community colors). The
store change does not touch any of these — they speak the thin storage interface.

---

## 8. Tonkosti checklist (must get right)

1. Embedder dim **<= 2000** for HNSW (1024 recommended); >2000 => StreamingDiskANN on full `vector`
   (never `halfvec` + diskann — unsupported), or MRL-truncate 4096->1024.
2. **Filtered ANN:** always `hnsw.iterative_scan='relaxed_order'` + `ef_search>=200` when combining
   owner-filter with ANN, or recall silently collapses to ~10%.
3. `ef_construction` 200 (not the default 64) for production recall; `m=16`.
4. **Two-phase retrieval** — ANN in its own CTE, then bounded (`depth<3`, cycle-safe) recursive
   expand. Never one giant plan.
5. **Pooling:** transaction mode (RLS needs per-tx `SET`), psycopg3 async pool; writes off the hot
   path via batched queue.
6. **RLS + app-layer WHERE** both; composite `(owner_id,...)` index; partition by owner at scale.
7. **HNSW upkeep:** REINDEX CONCURRENTLY + VACUUM weekly; keep the index RAM-resident.
8. Graph stays relational (`mem_node`/`mem_edge` + recursive CTE); algorithms (Leiden/PageRank) in
   the Python batch, not the DB.
9. Bi-temporal edges (`valid_at`/`invalid_at`) so contradicted facts are invalidated, not deleted —
   audit-friendly and leak-bounded (DELETE space is reclaimed by VACUUM).
10. `timescaledb-ha` base image; alembic must register the `vector` type for autogenerate.

---

## 9. Implementation status (2026-07) — built vs designed

Honest delta between this design and the shipped code (`sndr/memory/`, the
product-API memory/gateway routes, the unified container).

**Built + tested (in-memory + live Postgres, CI-gated):**
- `mem_node`/`mem_edge` schema, HNSW + lexical GIN indexes, `iterative_scan` tuning.
- Brain mechanics: ANN search, Hebbian co-access, lazy Ebbinghaus decay, **strength
  reinforcement** (retrieval slows decay), two-phase spreading-activation recall,
  salience prune. recall/decay live ONCE in the base class → both backends identical.
- Hybrid search (vector + keyword), exact-content dedup, communities (label
  propagation, not Leiden), importance heuristic, `consolidate`, **wired maintenance
  scheduler** (consolidate + prune per owner → the leak-bound), Obsidian import.
- **Bi-temporal invalidation** — `invalidate_edge` writer + `POST /edge/invalidate`;
  reads exclude `invalid_at`, records kept for audit.
- `/api/v1/memory/*` + the multi-upstream OpenAI gateway; **API-key guard**;
  graceful Postgres-down fallback + upstream-error 502/504; visible maintenance logs.
- Embedders: HashEmbedder (dep-free) + Model2Vec (real CPU). Container: one image
  (Postgres+pgvector+API+GUI+gateway+maintenance).

Full operational + developer reference: **`docs/memory/MANUAL.md`**.

**Decided against (design over-promised) — with rationale:**
- **RLS** — NOT used. It conflicts with the cross-owner ops (`owner_ids`,
  `count_edges`, the maintenance scheduler) and the single-connection model, and
  app-layer `WHERE owner_id` + the API-key guard already enforce isolation. Revisit
  only for true multi-tenant DB-enforced isolation (would also need per-owner
  connections + reworking the cross-owner ops).

**Deferred (scale-up; documented in MANUAL §14, not wired by default):**
- **Async connection pool** — PostgresStore is a single connection + lock (correct,
  but serializes under concurrent async load; fine at homelab single-user scale).
- **pgvectorscale / StreamingDiskANN / halfvec** — stock pgvector HNSW only (≤2000
  dim); the escalation (>1M vectors or >2000-dim embedders) swaps the base image.
- **Leiden** — replaced by deterministic label propagation (adequate for homelab).
- **alembic migrations / partitioning / REINDEX-VACUUM cadence** — schema is created
  idempotently by `ensure_schema`; ops tooling is documented, not automated.

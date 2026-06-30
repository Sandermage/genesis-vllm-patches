# Neural knowledge-graph mode (Obsidian-like) — hardware tiers + auto-linking + GUI + API

**Status:** design / discussion (2026-06-30). Third companion doc (after
`persistent-memory-architecture.md` = universal landscape, `memory-engine-lean-build.md` =
lean engine). This one nails the operator's ask: **knowledge that forms connections &
dependencies and clusters into "clouds" like Obsidian, visualized in the GUI, working with
all models, connected to our API — with a CHOICE of how much of the Ryzen-370 box + the
A5000s to use.** Grounded in a live research sweep; sources inline.

## 0. The shape (what we're building)

A **brain-like knowledge graph**: each memory/note is a node; edges form automatically
(semantic + co-access + relations) → entities cluster into colored "clouds" (communities) →
rendered as a live, zoomable, force-directed graph in the GUI, exactly like Obsidian's graph
view. The same graph is the universal memory (plain-text injection serves every model) and is
exposed over our product-API.

**The blueprint already exists and matches our stack:** **LightRAG** (HKUDS, MIT) is FastAPI +
**React + graphology + Sigma.js** with a KG web UI (local-graph, Expand/Prune node, community
coloring) and works against a local vLLM/OpenAI-compatible endpoint — i.e. our exact GUI stack
(React+Vite) and serving stack. We copy its API + viz shape and back it with our lean
sqlite-vec engine (no per-write LLM, deterministic Hebbian — the leanness LightRAG lacks).
(https://github.com/HKUDS/LightRAG · graph_routes.py · LightRAG-API-Server.md)

## 1. Hardware — use the Ryzen box + A5000s, with a CHOICE (light → heavier)

Principle: **the 2×A5000 stay 100% on the 35B engine; the memory layer lives on the Ryzen-370
box over LAN.** The A5000s are touched only by an optional nightly batch.

> NB: "Ryzen 370" is most likely an **AMD Ryzen AI 9 HX 370** (Strix Point: 12-core Zen5,
> Radeon 890M iGPU, XDNA2 NPU ~50 TOPS). **Please confirm the exact box (APU model + RAM, or
> desktop Ryzen + discrete GPU)** — it decides T2. T0/T1 (CPU) work regardless.

**Verified HX 370 reality (mid-2026) — the NPU & iGPU are NOT embedding accelerators:**
- **NPU (XDNA2):** embeddings effectively **don't work** — VitisAI hangs compiling BERT/encoders
  (RyzenAI-SW #312) and the public onnxruntime-genai lacks the NPU EP (#333). The NPU runs
  *decoder LLMs* (FastFlowLM ~20–80 tok/s) + Whisper, not encoder embedders.
- **iGPU 890M (gfx1150):** **not** in the ROCm support matrix; ROCm is override-dependent and
  **broke on a real ONNX/CLIP embedding** workload (immich #22874). Vulkan/llama.cpp works for
  *decoder LLMs* but is unproven for encoders. APU inference is **RAM-bandwidth-bound (~128 GB/s)**.
- **⇒ On the HX 370 the only mature embedding path is the CPU** (fastembed ONNX-INT8 / static
  Model2Vec). The box's real value = its **12 Zen5 cores + hosting the memory service/store** over
  LAN — not NPU/iGPU acceleration. So the tier ladder below is CPU-first by necessity, not just
  by preference.

| Tier | Where / accelerator | Embedder | Quality | Cost | When to pick |
|---|---|---|---|---|---|
| **T0 (default)** | Ryzen box **CPU**, static | potion-base-8M / -retrieval-32M (~30 MB, numpy, ~200–500× faster than a transformer on CPU) | ~82–93% of MiniLM | **0 GPU, ~30 MB RAM** | always-on path; start here |
| **T1** | Ryzen box **CPU** (ONNX-INT8, fastembed/Infinity) | bge-small / multilingual-e5 | solid (62 MTEB-v1) | CPU only (NOT NPU/iGPU — see above) | when T0 recall is short |
| **T2** | a **dedicated/spare discrete GPU** (NOT the 35B A5000s, NOT the HX 370 accel) | Qwen3-Embedding-0.6B→4B (Apache-2.0, MRL, multilingual, Russian) | top-tier (64–74 MTEB) | ~1.5–9 GB VRAM | multilingual / serious recall |
| **T3 (batch)** | **idle A5000 at night** only | graph-build / consolidation / reranker on vLLM `run_batch` | best graph quality | A5000 off-hours | entity/relation extraction, dedup, Leiden, reflection |

- The **read path is always CPU + 0 LLM** (vector search + optional FlashRank Nano ~4 MB CPU
  rerank). The 35B/A5000s are touched only by the **T3 nightly batch**.
- **Don't co-locate embeddings on the 35B GPU:** vLLM pre-allocates its VRAM fraction per process,
  so at ~95% util a second process = OOM risk + KV-cache shrink + compute contention on the hot
  path. LAN RTT (sub-ms) ≪ embed compute (2–20 ms), so the memory service on the Ryzen box is fine.
- Embedding microservice options (OpenAI `/v1/embeddings`): **Infinity** (CPU/ROCm), **lemonade**
  (AMD-native), **TEI** (CPU; AMD support is Instinct-only, not the 890M). **The GUI exposes the
  tier as a dropdown** so you choose.

## 1b. Storage backend — Postgres + pgvector + pgvectorscale (sqlite-vec AND Apache AGE DROPPED)

> **DECISION (operator, 2026-06-30): drop sqlite-vec; on deeper study, also drop Apache AGE.** The
> single backend is **Postgres + pgvector + pgvectorscale**, with the graph as two relational tables
> (`mem_node`/`mem_edge`) + `WITH RECURSIVE`. **Full concrete schema, indexing, brain-mechanics-in-SQL,
> pooling / RLS / migrations and every subtlety live in `memory-engine-production-design.md`.** A thin
> storage interface is kept (unit-test double + LadybugDB future option); the shipped engine is
> Postgres-only.

- **Why sqlite-vec rejected:** brute-force only (no stable ANN, ceiling ~hundreds-of-thousands),
  single-writer, "graph" = DIY edges-table — fine for a zero-ops MVP, not for the structurality wanted.
- **Why Apache AGE also dropped (revised):** the leading Postgres-memory systems (Mem0, Cognee, Letta)
  all avoid it; AGE is rare in shipped products (the one detailed production case bypassed its Cypher
  for native SQL after variable-length `[r*..N]` degraded from ~100s-of-ms to **3–5 s**); you still
  need a sidecar `vector(N)` table (vectors can't be indexed inside an AGE `agtype` node); `LOAD`/
  `search_path` break under transaction pooling; and AGE has **no native graph algorithms** (Leiden/
  PageRank are DIY regardless). Our retrieval is shallow (≤3 hops) — the zone where plain recursive
  SQL wins (<10 ms depth-2, <20 ms 100-node subgraph, to ~10K nodes). Dropping AGE also lets us use
  the stock `timescale/timescaledb-ha` image (pgvector + pgvectorscale built in) with no custom build.
- **Graph-native escape hatch (only if we ever truly need Cypher): LadybugDB** — live MIT fork of the
  now-dead Kùzu; pre-1.0. Avoid FalkorDB (SSPL), Neo4j CE (GPLv3+JVM), Memgraph (BSL) for a clean OSI build.
- The brain-mechanics (§2) and viz/API (§3–4) are storage-agnostic by design.

## 2. How the "neurons / clouds" form (auto-linking + clustering) — mostly no-LLM

Edges are created automatically by four cheap mechanisms (Obsidian only has the first kind —
manual wikilinks; we add three automatic kinds):

1. **Semantic edges (no LLM):** on insert, kNN over embeddings; cosine ≥ τ (≈0.8) → `similar_to`
   edge. The "related notes" web. (A-MEM uses top-k=10; HippoRAG adds synonymy edges at τ=0.8.)
2. **Hebbian co-access edges (no LLM):** co-retrieved nodes wire together. Verified formula
   (HeLa-Mem): `w_ij ← (1−λ)·w_ij + η·[both in retrieved set]`, **η=0.02, λ=0.995**; decay + prune.
   Optional spreading-activation recall (β=0.1, θ=0.6). (Genuinely novel — a
   documented-but-largely-unimplemented pattern, survey arXiv:2602.05665.)
3. **Typed relation edges (LLM, T3 batch only):** the nightly job extracts `(subject, rel, object)`
   triples — the "dependencies." Never on the write path. (cognee/LightRAG/Mem0 pattern.)
4. **Explicit links/tags:** Obsidian-style `[[wikilinks]]`/tags from ingested notes → edges, with
   **backlinks** (bidirectional), exactly like Obsidian.

> **Canonical algorithm = A-MEM (Zettelkasten, arXiv 2502.12110)** — our closest reference: each
> note `{content, keywords, tags, context, embedding, links}`; semantic-kNN (#1) *proposes*
> candidates, the T3-batch LLM (#3) *confirms* typed links, Hebbian (#2) is the always-on no-LLM
> layer, and "memory evolution" lets the batch LLM refine a new note's nearest-4 neighbors. This
> is exactly the "knowledge auto-forms connections + refines itself" behaviour requested.

**Clouds = community detection.** Run **Louvain/Leiden** (`graphology-communities-louvain`, MIT)
to assign each node a community id → color → the force-directed layout naturally groups them into
the colored "clouds" you see in Obsidian. Re-run incrementally in the T3 batch as the graph grows.

## 3. Visualization in the GUI (the Obsidian look)

- **Library: Sigma.js + graphology** (WebGL, MIT, actively maintained) — built for thousands–tens
  of thousands of nodes; ForceAtlas2 layout (the Obsidian "spring" feel) + Louvain community
  coloring (the "clouds"). Cytoscape.js (Canvas, MIT) is the alternative for smaller analysis-rich
  graphs. (LightRAG uses Sigma+graphology — copy it.)
- **New React panel `gui/web/src/Memory.tsx`** (sibling of Engine/Fleet/Planner), reusing our
  `api.ts`. It's the natural evolution of the **existing Obsidian-vault RAG in `Engine.tsx`**
  (which already indexes Obsidian/notes folders) — now you SEE the graph of that knowledge.
- **Interaction = local-graph + expand (the LOD pattern, mandatory for 50k nodes):**
  - never load the global graph; default to a **local subgraph** around a seed (depth≤2,
    max_nodes≈500), `is_truncated` flag shows "more available";
  - **Expand/Prune on click** (LightRAG's exact buttons) pulls neighbors on demand;
  - Sigma `nodeReducer`/`edgeReducer` dim/hide low-degree nodes when zoomed out (zoom = detail);
  - **cache node positions** so adding a node doesn't reshuffle the whole layout (Obsidian behavior);
  - **layout split:** precompute `x,y` server-side for big/global views; run ForceAtlas2 in a
    **Web Worker** client-side for the live local subgraph.
- Features matching Obsidian: search box (`/memory/labels/popular`), node detail panel, local-graph
  depth slider, cluster colors, live updates.

## 4. API (on our product-API, owner-scoped, copies LightRAG)

New `/api/v1/memory/*` routes (registered like the existing `@app.get("/api/v1/...")`):

```
GET  /api/v1/memory/search?q=&k=20            # vector seed nodes (entry points)
GET  /api/v1/memory/node/{id}                 # node + immediate edges (detail panel)
GET  /api/v1/memory/subgraph?seed=&depth=2&max_nodes=500   # LOCAL graph for the viz (workhorse)
GET  /api/v1/memory/node/{id}/neighbors?limit=50           # expand-on-click (delta only)
GET  /api/v1/memory/communities               # cluster membership → colors
GET  /api/v1/memory/labels/popular?limit=300  # high-degree entry points for the search box
GET  /api/v1/memory/stats                     # counts/density/last-updated (poll/header)
POST /api/v1/memory/add                        # ingest node(s)/edge(s)
GET  /api/v1/memory/stream  (SSE)             # incremental add_node/add_edge events
```
- **JSON shape (LightRAG `KnowledgeGraph`):** `{nodes:[{id, labels[], properties:{name,
  description, degree, cluster, x, y, owner_id}}], edges:[{id, type, source, target,
  properties:{weight}}], is_truncated}`.
- **Live updates: SSE** (`/memory/stream`) — one-directional server→client; client **batches/throttles**
  events and appends to the graphology instance (never refetch the whole graph, never re-render
  per event). Fallback: poll `/memory/stats`, refetch subgraph only on change.
- **Isolation (critical):** every route injects `owner`/`group_id` from the authenticated session
  (never a client param) and filters the query — including `neighbors`/`subgraph` expansion and the
  SSE feed, so expanding a node can't leak another tenant's nodes. Mirror our `_auth_guard`;
  store an indexed `owner_id` on node+edge tables. (Graphiti `group_id` pattern; LightRAG
  `Depends(combined_auth)`.)

## 5. "Connected to our API or feeds it data" — both directions

- **Serve (read):** the GUI + any client read the graph via the `/memory/*` API above.
- **Feed (write, universal):** the **proxy memory-middleware** (the universal layer all models
  cross) captures each turn and `POST`s to `/memory/add` async — so Gemma, Qwen, and external
  GPT/Claude all populate the SAME graph. The proxy also injects retrieved memory (plain text) on
  read → every model benefits, none is special.
- **MCP (optional):** expose the same store as an MCP server too, so agents that want explicit
  memory control can read/write — secondary to the always-on proxy capture.

## 6. Bringing it together — the assembled mode

```
 turn (any model) ─▶ proxy middleware ─▶ POST /memory/add (async)         [universal capture]
                          │ inject retrieved text on read
 Ryzen-370 box: memory service ── sqlite-vec nodes+edges ── CPU embedder (T0) ── /memory/* API
        │ auto-edges: semantic(kNN) + Hebbian(co-access) + wikilinks   [no LLM, hot path]
        │ nightly on idle A5000 (T3): relation-extract + dedup + Louvain communities + prune
        ▼
 GUI Memory.tsx (React + Sigma.js + graphology): force-graph, colored clouds, local-graph+expand, SSE-live
```

**Why this is the best/quality version:** leak-free (sqlite-vec recycles, bounded caps + decay +
salience prune), low-load (CPU read path, 0 GPU on hot path, A5000 untouched except nightly),
brain-like (auto semantic + Hebbian + relation edges → Louvain clouds), universal (proxy
plain-text injection for all models), and an Obsidian-grade live graph UI — all on MIT/Apache
building blocks we control, not a heavy framework.

## 7. Open decision for the operator
- **Confirm the Ryzen-370 box** (Ryzen AI HX 370 APU + RAM? or desktop Ryzen + discrete GPU?) →
  pins Tier 1/2. Tier 0 (CPU-static) is the safe default regardless.
- **Pick the default tier** (recommend T0 read path + T3 nightly graph-build on idle A5000).
- Greenlight Phase 1 (engine + API) then Phase 1b (GUI graph panel).

### Key sources
- Blueprint: https://github.com/HKUDS/LightRAG (graph_routes.py, types.py, LightRAG-API-Server.md) — FastAPI+React+Sigma+graphology, MIT
- Viz: https://www.sigmajs.org/ · https://graphology.github.io/standard-library/ (forceatlas2 + communities-louvain, MIT) · https://js.cytoscape.org/
- Embedders: https://github.com/MinishLab/model2vec (potion, MIT) · https://hf.co/Qwen/Qwen3-Embedding-0.6B (Apache-2.0) · https://github.com/PrithivirajDamodaran/FlashRank
- Ryzen AI: https://qwenlm.github.io/ (embedding) · AMD Ryzen AI SW / ONNX Runtime VitisAI EP (NPU) — validate at build
- Isolation/incremental: Graphiti group_id · SSE-vs-WS (https://ably.com/blog/websockets-vs-sse)
- Obsidian graph behavior: https://obsidian.md/help/plugins/graph

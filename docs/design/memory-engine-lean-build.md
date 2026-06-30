# Lean custom memory engine — leak-free, low-overhead, brain-like (build spec)

**Status:** design / discussion (2026-06-30). Companion to
`persistent-memory-architecture.md` (the universal landscape). This doc is the
*implementation* spec for the operator's explicit requirements: **our own,
quality, no memory leaks, low load on hardware + the model, works like a brain.**
Grounded in the research sweep + our codebase's proven leak-free patterns.

## 0. The four hard constraints → design rules

| Constraint (operator) | Hard design rule |
|---|---|
| **Своё / качественное** (our own, quality) | Build a lean ~1-2k-LOC service we control; *borrow proven algorithms*, don't adopt a heavy framework (cognee/Graphiti pull in Neo4j + per-write LLM extraction). |
| **Без утечек памяти** | Bounded-by-construction: hard capacity cap + decay eviction; on-disk store (RSS stays flat); reuse our `ResponseCacheLRU` (LRU+TTL) and Track-A `forget()`/page-cap discipline. No unbounded dict/cache anywhere. |
| **Без нагрузки на железо/модель** | Zero GPU contention with the 35B engine: CPU/static embeddings; **NO LLM on the read path**; extraction/consolidation deferred to **idle-GPU batches**, never per-turn. |
| **Как мозг** | Weighted, decaying, self-reinforcing connections (Hebbian) + salience-driven pruning — all **deterministic (no LLM)** so it's cheap and leak-bounded. |

## 1. Build-our-own vs adopt — the decision

Adopt-wholesale is the wrong fit here: cognee/Graphiti are excellent but (a) require a
**graph DB (Neo4j/FalkorDB)** — extra service + RAM, and a graph that **grows unbounded**
unless externally capped (called out for A-MEM/LightRAG); (b) do **per-episode LLM
extraction**, which on our single-engine rig means hammering the 35B on the write path.

So: **build a lean custom service**, borrowing the *proven algorithms* (decay formulas,
Hebbian reweighting, embedding-blocking dedup, sleep-time consolidation) but on our own
bounded, CPU-first substrate. This is exactly "своё, качественное, без утечек, без нагрузки."

There is **no existing knowledge-memory module** in the repo (`product_api/legacy/memory.py`
is a VRAM hardware-fit report), so this is greenfield → a new `sndr/memory/` package.

## 2. No-leak engineering (bounded by construction)

The leak class we already fixed in Track A (unbounded page-tracking dict → page-cap +
`forget()`; bare-namespace `pop` no-op → delete-all-keys; pool-budget accounting) is the
exact discipline this service must carry from day one.

- **On-disk store, not in-RAM:** memories live in **SQLite (`sqlite-vec`) or pgvector**, not
  Python dicts → process RSS stays flat regardless of corpus size. The ANN index is the only
  in-RAM structure and is itself capacity-capped.
- **Hard capacity cap** (mirrors chat_rag's `_VAULT_MAX_FILES`/`_MAX_TOTAL_BYTES`): a fixed
  ceiling on rows + total bytes; writes past it trigger eviction (§4), never unbounded growth.
- **Reuse `ResponseCacheLRU`** (`sndr/cache/response_cache.py`, LRU + TTL) for the embedding
  cache and the hot-query cache — proven, bounded, TTL-evicted.
- **Explicit `forget(id)` / `prune()`** (mirrors Track-A `tier_manager.forget()` /
  `_enforce_page_cap()`): the only ways memory leaves; both are O(1)/O(cap), never scans.
- **No unbounded accumulators:** every counter/edge-table is bounded by the row cap; co-access
  edges are capped per node (top-N strongest, weakest pruned).
- **Leak gate in CI/tests:** a soak test that adds N≫cap memories and asserts row count,
  index size, and RSS all plateau at the cap (the soak-as-leak-detector pattern from the
  club-3090 methodology).

## 3. Low hardware / model load

The 35B engine already uses ~95% of the 2× A5000 VRAM. The memory layer must not touch it.

- **Embeddings on CPU (no GPU contention):** static/CPU embedder — **Model2Vec / potion-base**
  (sub-ms CPU, ~30 MB) for the bulk, or **bge-small / bge-m3 on CPU** for higher quality. The
  GPU Qwen3-Embedding is optional and only if a spare slice exists. Embedder is pinned + version-
  tagged in the store (one embedder per index — switching = re-embed).
- **NO LLM on the read path:** retrieval = vector search (+ optional CPU rerank, FlashRank
  ONNX). Measured cost for this class: search p95 ~0.2 s (Mem0 Table 2), context ~1.6 k tokens
  vs ~115 k full-context (Zep) — i.e. the *read* path is cheap and never calls the 35B.
- **Extraction/consolidation = deferred idle-GPU BATCH, never per-turn.** Per-conversation
  extraction is ~7 k tokens (Mem0 §4.5); doing it per-turn would load the engine on the hot
  path. Instead: capture turns cheaply (append raw), then a **sleep-time batch job**
  (`LLM.generate()` / vLLM `run_batch` on idle GPU) extracts/dedups/consolidates off the user
  path. We already have idle GPU at night.
- **The ConvoMem gate (don't build heavy until needed):** below ~150 repeated conversations,
  long-context + the existing `chat_rag` vector RAG already beats RAG-memory (70-82% vs
  30-45%). So Phase 0 = measure; the brain layer only switches on past the threshold — cheapest
  possible until it earns its keep.

## 4. Brain-like — but cheap and deterministic (no per-write LLM)

All "brain" behavior is closed-form math over the store; the only LLM use is the deferred
batch consolidation (§3).

- **Decay / forgetting (Ebbinghaus):** retention `R = e^(−Δt / S)`, strength `S` starts at 1,
  **`S += 1` and `Δt → 0` on each recall** (MemoryBank) — recalled memories persist, unused
  ones fade. Equivalent: Generative-Agents `score = recency·importance·relevance` with recency
  decay `0.995^hours` (LangChain `TimeWeightedVectorStoreRetriever`, `decay_rate≈0.01`). Pure
  arithmetic on `last_access`/`access_count` columns.
- **Hebbian co-access reinforcement (the "neurons that wire together"):** when two memories are
  retrieved/used in the same turn, **increment their edge weight** (`edges.weight += 1`,
  `last_coaccess = now`). Frequently co-used memories form strong connections that surface
  together later. Pure counting — no LLM. Edge weights themselves decay, and each node keeps
  only its top-N strongest edges (bounded).
- **Salience-driven pruning (the brain prunes → bounds the store):** `salience = f(recency,
  access_count, importance, edge_strength)`; when the store hits its capacity cap, evict the
  lowest-salience memories (MemoryOS "heat threshold" / SCM "intentional forgetting"). This is
  what keeps RAM/disk bounded over months of running.
- **Dedup / entity-resolution without LLM (structural integrity):** on write, embedding cosine
  ≥ threshold (≈0.9, SemHash-style) → merge/update instead of insert; only genuinely ambiguous
  cases are queued for the batch LLM-judge. Keeps the graph clean cheaply.
- **Consolidation/reflection:** the deferred batch job synthesizes higher-level "concept" nodes
  from clusters of episodic memories (Generative-Agents reflection tree) and resolves
  contradictions (Graphiti-style edge invalidation: mark superseded, don't delete — preserves
  audit). Runs nightly on idle GPU.

## 5. Minimal data model (sqlite-vec / pgvector)

```
memories(
  id            INTEGER PK,
  kind          TEXT,        -- episodic | semantic | concept | procedural
  content       TEXT,
  embedding     BLOB,        -- one pinned embedder; version-tagged
  importance    REAL,        -- 0..1 (cheap heuristic at write; LLM-rated in batch)
  created_at    REAL,
  last_access   REAL,        -- updated on recall (decay reset)
  access_count  INTEGER,     -- recall frequency (strength)
  salience      REAL,        -- derived; drives eviction
  source_id     TEXT,        -- provenance (which turn/doc) — for audit + trust-scoring
  owner         TEXT         -- per-user/project isolation key (RLS / filter)
)
edges(
  src INTEGER, dst INTEGER,
  type TEXT,                 -- relation label (batch-extracted) or "coaccess"
  weight REAL,               -- Hebbian; decays; top-N per node kept
  valid_at REAL, invalid_at REAL,  -- bi-temporal (supersede, don't delete)
  last_coaccess REAL
)
```
Both tables hard-capped; eviction by `salience` (memories) and weight/recency (edges).
Isolation via `owner` + Postgres RLS (or SQLite-per-tenant) — never metadata-filter-only.

## 6. Universal integration (unchanged from rev2)

Model-agnostic by construction: the **proxy memory-middleware** retrieves and injects
**plain text** into any model's prompt (Gemma/Qwen/Llama/GPT/Claude alike). The embedder is
an internal detail, not a model lock-in. The optional weights-loop is one LoRA per local
architecture, served by the existing per-engine + aggregator topology.

## 7. Reuse map (our codebase)

| Need | Reuse |
|---|---|
| Bounded LRU+TTL caches | `sndr/cache/response_cache.py` (`ResponseCacheLRU`) |
| Hard size/count caps + enforced checks | `chat_rag` `_VAULT_MAX_*` pattern |
| `forget()` / capacity enforcement / no-leak discipline | Track-A `tier_manager.forget()`, `_enforce_page_cap()`, `prealloc.release()` |
| Bootstrap corpus | existing `chat_rag` project-knowledge index |
| Per-request auth/isolation | `product_api` auth (`owner` ↔ session user) |
| New code | greenfield `sndr/memory/` package |

## 8. Phased build (each phase shippable + measured)

- **Phase 0 — measure (no build):** stand up **LongMemEval** self-hosted as the scoreboard;
  count real repeated-conversation volume. Below ~150, keep plain `chat_rag` — don't build.
- **Phase 1 — lean read/write core:** `sqlite-vec` store + CPU embedder + bounded caps +
  deterministic decay/salience + proxy memory-middleware (inject plain text on read, append
  raw turn on write, async). Hebbian co-access on retrieve. Leak soak-test in CI. **No GPU,
  no per-turn LLM.**
- **Phase 2 — idle-GPU consolidation:** nightly batch (vLLM `run_batch`) extracts entities,
  dedups (embedding-blocking → LLM-judge only for ambiguous), builds concept nodes, resolves
  contradictions, recomputes salience + prunes to cap. Add Presidio PII scrub at ingestion.
- **Phase 3 — per-architecture weights loop (optional):** distill high-salience validated
  memory → one LoRA per local arch (Qwen3.6, Gemma-4) → hot-swap, eval-gated.

## 9. Bottom line

A lean, on-disk, CPU-embedded, deterministic-brain memory service we build ourselves:
**leak-free** (bounded by construction + our proven eviction patterns), **low-load**
(no GPU contention, no read-path LLM, extraction batched on idle GPU), **brain-like**
(Hebbian-reinforced, decaying, self-pruning weighted graph), and **universal** (plain-text
injection at the proxy serves every model). Borrow the proven algorithms; own the
bounded substrate.

## 10. Appendix — verified implementation specifics (deep research, 2026-06-30)

Concrete numbers/formulas/code pulled from primary sources, so this is build-ready.

### A. Storage choice — sqlite-vec (leak-bounded by construction)
- **HNSW (hnswlib/usearch/DuckDB-VSS/pgvector-HNSW) NEVER frees on delete** — `mark_deleted` is a
  tombstone; RAM is reclaimed only by a full rebuild. usearch `compact()` is historically unreliable
  (issue #355). So a churning memory store on HNSW grows unbounded until rebuilt.
- **sqlite-vec is the clean fit:** in-process single file, and its delete path (PR #243) **zeros the
  rowid slot + recycles the space** for future inserts → storage is *bounded*, not leaking, with no
  daemon. Brute-force search is fine for 10k–few-hundred-k vectors (our scale). Sizes (raw f32 =
  dim×4): **100k @ 256-dim (potion) ≈ 102 MB; @ 384-dim ≈ 154 MB**; int8 = ¼, bit = 1/32.
- If we ever need ANN speed past ~1M vectors: usearch (uint40 links + i8/b1 quant + on-disk view) but
  **schedule periodic rebuilds** to reclaim deletes. Avoid hnswlib as primary under high churn.
- **No graph DB** (Neo4j JVM ~5 GB heap, Milvus 3-container, Qdrant ~1.2 GB daemon — all too heavy on
  the shared box). The "graph" is just an `edges` table in the same SQLite file.

### B. Python long-process leak discipline (RSS stays flat) — mandatory checklist
- **glibc arena fragmentation** is the #1 false "leak" (RSS climbs, heap stable): set
  `MALLOC_ARENA_MAX=2` or preload **jemalloc** (`background_thread:true,dirty_decay_ms:1000`) — a real
  service went 1.25 → 0.12 MB/hr after this. Or call `malloc_trim()` periodically.
- **Every cache bounded:** `lru_cache(maxsize=N)` or `cachetools.TTLCache` — never `maxsize=None`; and
  **never `lru_cache` on instance methods** (captures `self`, never frees). Reuse our `ResponseCacheLRU`.
- **asyncio:** keep a strong-ref `set` for tasks AND `task.add_done_callback(set.discard)` (else either
  silent GC-cancel or unbounded set growth).
- **DB:** bound `pool_size`/`max_overflow`, always close via context managers, alarm on
  `pool.checkedout()` pinned at max.
- **CI soak test (leak gate):** add N≫cap memories, assert row count, index size, AND RSS plateau.

### C. Brain-like mechanisms — exact, deterministic, no-LLM (verified code/formulas)
- **Hebbian co-access edge bump** (EngramAI `storage.rs`, AGPL → borrow the *rule*, not code):
  on co-retrieval of nodes i,j → `w ← min(1.0, w + 0.1)`; canonicalize `id1<id2` (order-free); edges
  decay by a sweep `w *= (1−λ)` and prune `w < 0.1`. Soft-bound variant (EPFL Neuronal Dynamics
  §19.2): `w += η·(1−w)` on co-active, `w *= (1−λ)` otherwise → stays in [0,1], self-bounding.
  **Note: true Hebbian co-retrieval edge-bumping is a documented-but-largely-unimplemented pattern
  (survey arXiv:2602.05665) — so this is filling a real gap, not duplicating an OSS project.**
- **Ebbinghaus decay** (MemoryBank, MIT): `R = e^(−t/S)`; on recall `S += 1`, reset `t→0`; evict when
  `random() > R` (probabilistic) or `R < threshold` (deterministic). **FIX the upstream
  operator-precedence bug** — use `math.exp(-t / (5*S))`, not `-t / 5*S`.
- **Generative-Agents retrieval** (Apache-2.0, ~50 LOC): normalize recency/importance/relevance to
  [0,1], `score = w_r·recency + w_i·importance + w_rel·relevance`; recency `= decay^hours_since_access`
  (**0.995** paper / 0.99 code). **Re-tune weights** — shipped `gw=[0.5,3,2]` is arbitrary.
- **Spreading activation = brain pattern-completion, cheap** (SYNAPSE 2026 / ACT-R / PPR): iterate
  `a ← α·seed + (1−α)·decay·(W_norm @ a)`, threshold-prune to keep sparse, **K=2–5 hops**; each hop is
  one **sparse mat-vec = O(#edges)**, no eigensolve. PPR (damping 0.5, HippoRAG) is the same thing.
- **Salience pruning bounds the store** (MemoryOS heat): `salience = α·access_count +
  β·dialogue_len + γ·exp(−Δt/μ)`; evict lowest when over the capacity cap. This is "the brain prunes."

### D. Low-load embedding/rerank stack (0 GB VRAM — never touches the 35B)
- First-stage: **`minishlab/potion-retrieval-32M`** (~30 MB, numpy-only, ~500× faster than its
  transformer teacher on CPU, ~82–87% of all-MiniLM retrieval quality) — or **fastembed
  `bge-small-en-v1.5`** (ONNX-INT8, 134 sent/s on 4 vCPU, stronger recall) if quality matters more.
- Rerank: **FlashRank Nano** (`ms-marco-TinyBERT-L-2-v2`, ~4 MB, ~0.1 s/100 docs CPU) recovers most of
  the precision lost vs a big GPU embedder. (Qwen3-Embedding-4B = ~18 GB VRAM — unaffordable here.)
- Pin ONE embedder + version-tag the store (switching ⇒ full re-embed).

### E. Adopt is too heavy (confirms build-lean)
- Mem0: 59 MB, **2 LLM calls per write** (extract + ADD/UPDATE/DELETE/NOOP), needs vector DB (+Neo4j
  for graph), drags LangChain. Letta: **287 MB FastAPI/Postgres server**, 60+ deps, LLM tool-call per
  write. Both make every write an LLM round-trip — the opposite of low-load. Borrow their *algorithms*.
- Reference impls to read (not adopt): MemoryBank (~40 LOC forget curve, MIT), Generative-Agents
  (~644 LOC memory+retrieval, Apache-2.0) — permissive, copyable; EngramAI (AGPL — algorithm only);
  nano-graphrag/A-MEM (~1.1k LOC each but LLM-heavy writes — borrow the data model, drop per-write LLM).

### Key sources
- Overhead numbers: Mem0 arXiv 2504.19413 (§4.5 ~7k tok/conv, Table 2 p95 0.2s search) · Zep arXiv 2501.13956 (1.6k vs 115k tokens, ~90% latency cut)
- Decay/Hebbian: MemoryBank arXiv 2305.10250 (R=e^(−t/S)) · Generative Agents arXiv 2304.03442 (recency×importance×relevance, 0.995) · LangChain TimeWeightedVectorStoreRetriever
- Cheap dedup: SemHash (MinishLab, MIT) · embedding-blocking→LLM-judge (Graphiti `resolve_extracted_nodes`)
- Lean refs to borrow from: nano-graphrag (~1100 LOC) · A-MEM (Zettelkasten evolution) · EngramAI (ACT-R + Ebbinghaus + Hebbian, single SQLite) · MemoryBank-SiliconFriend (MIT)
- CPU embedding: Model2Vec/potion · BGE-M3 · FlashRank (CPU rerank)
- Gate: ConvoMem arXiv 2511.10523 ("<150 conversations don't need RAG")
- Async/idle batch: sleep-time arXiv 2504.13171 · vLLM run_batch

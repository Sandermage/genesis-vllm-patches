# Persistent, brain-like, self-evolving memory — architecture & engine selection

**Status:** design / recommendation (2026-06-30). Grounded in a live web + HuggingFace
research sweep of 15+ systems; every option below was checked for license, maturity,
self-hostability, and OpenAI-compatible/vLLM fit. Sources inline.

## 1. Goal (restated)

A persistent knowledge + memory layer for the Genesis stack that is:

1. **Brain-like** — entities as nodes, relationships as edges, connections that
   *strengthen with use and prune when stale* (Hebbian-ish), not a flat vector blob.
2. **Structurally sound** — ontology / coreference so the same entity isn't
   duplicated; contradictions are resolved, not stacked.
3. **Self-evolving** — the store reorganizes over time; the system *gets smarter*
   the longer it runs.
4. **Shared** — one memory that **external models (via the proxy/aggregator)** AND
   the **internal vLLM models** both read and write.
5. **Capability-growing for the internal models** — not just retrieval; a path where
   accumulated memory turns into new *weights-level* ability for the local Qwen3.6.

These map to two distinct mechanisms that must NOT be conflated:

- **(A) A memory/KG service** → makes *any* model smarter *at inference* by injecting
  connected, consolidated knowledge. Works identically for external and internal models.
- **(B) A self-improvement loop** → Reflexion/ExpeL accumulate reusable insights (works
  for all models); **memory→LoRA distillation** is the *only* path that makes the
  **internal** model smarter *in its weights* (external API models can't take adapters).

## 2. Recommended architecture

```
                         ┌──────────────────────────────────────────┐
   user / agents ───────▶│  OpenAI-compatible PROXY  (Genesis_proxy) │
                         │  memory-middleware:                       │
                         │   • READ  → inject retrieved memory       │
                         │   • WRITE → capture turn (async)          │
                         └───────┬───────────────────────┬──────────┘
                                 │ routes to             │ calls on every request
                  ┌──────────────▼─────┐        ┌────────▼──────────────────────┐
                  │ external model APIs │        │  MEMORY SERVICE (self-hosted)  │
                  │ (via aggregator)    │        │  brain = temporal/onto KG +    │
                  └─────────────────────┘        │  vector; self-evolving         │
                  ┌─────────────────────┐        │  exposes REST + MCP            │
                  │ internal vLLM 35B    │◀───────┤  embed/rerank via vLLM         │
                  │ (chat) + LoRA hot-   │        └────────┬───────────────────────┘
                  │ swap                 │                 │ nightly
                  └──────────▲───────────┘        ┌────────▼──────────┐
                             └───────────────────-│ memory→LoRA       │  (Phase 3: internal
                                                   │ distiller         │   model gains weights)
                                                   └───────────────────┘
```

**Why the hook lives in the proxy:** the aggregator already routes *both* external
APIs and the internal vLLM through one OpenAI-compatible proxy — that is the only
layer every request crosses, so a memory hook there is shared *by construction* and
provider-agnostic (the model never knows). An MCP surface is exposed *in addition*,
as an optional explicit tool for agents that want to manage memory directly — but it
is **not** the primary capture path, because external API models you don't control
won't reliably call MCP tools. (Pattern corroborated by the agent-memory surveys:
memory should be a standalone component layered on the LLM, not embedded per-agent.)

## 3. Engine selection — the "brain"

Two front-runners, both Apache-2.0, self-hostable, native vLLM/OpenAI-compatible,
and actively maintained (live GitHub metrics, 2026-06-30):

| Engine | Brain mechanism | Self-evolution | Shared store | vLLM fit | Stars / release |
|---|---|---|---|---|---|
| **cognee** (topoteretes) | ECL graph+vector; ontology + cross-doc coreference | **`memify`: strengthens frequent connections, reweights edges by usage, prunes stale nodes** + `improve()` feedback | **FastAPI API-mode + MCP — many models share one graph** | LiteLLM `hosted_vllm/` + `LLM_ENDPOINT` | 26k / v1.2.2 (2026-06-26) |
| **Graphiti** (getzep) | bi-temporal KG; LLM-extracted entities+edges | **temporal edge invalidation** — facts get validity windows; contradictions close old edges, don't delete | `group_id` namespacing; MCP + REST | `OpenAIGenericClient` + `base_url`, JSON-schema decode | 28k / v0.29.2 (2026-06-08) |

- **Primary recommendation: cognee.** It is the closest literal match to "connections
  like neurons that strengthen over time + structural integrity": `memify` does
  usage-weighted edge reinforcement + stale-node pruning (Hebbian-like consolidation),
  the ontology/coreference step keeps the graph structurally clean, and `improve()`
  is explicit feedback-driven learning. Its **API-mode** lets external + internal
  models hit one shared graph — exactly requirement (4). Backends are all self-hostable
  (Postgres-graph or Neo4j/Kuzu/FalkorDB + pgvector/LanceDB/Qdrant). vLLM via LiteLLM
  is first-class.
- **Alternative: Graphiti**, when rigorous *temporal correctness* matters more than
  Hebbian consolidation — bi-temporal validity windows make "what did we believe, and
  when" auditable; superseded facts are invalidated, never silently overwritten.
- **Neurobiological note: HippoRAG 2** ("From RAG to Memory: Non-Parametric Continual
  Learning", arXiv 2502.14802) is the most *literally* brain-modeled — artificial
  neocortex (LLM) + hippocampus (open KG) + Personalized PageRank, first-class vLLM
  (online + 3× faster offline batch indexing). But it's a research library with no
  service/multi-tenant layer — best used as the *retrieval engine behind* the memory
  service, not the service itself.

Lighter agent-memory layers (consider if a full KG is overkill): **Mem0** (59k stars,
ADD/UPDATE/DELETE/NOOP self-consolidation, simplest to adopt), **MemOS** (richest
evolution + `user_id` sharing, heaviest stack), **Supermemory** (best contradiction
handling + time-based forgetting). All Apache-2.0/MIT, all vLLM-compatible.

> Honesty note: most headline benchmark numbers (LongMemEval/LoCoMo %, token-savings)
> are vendor-reported. The hard facts above (license, stars, releases, backends, vLLM
> config) are verified against GitHub/docs; the comparative quality claims are not
> independently reproduced and should be validated on our own data before committing.

## 4. Retrieval models (self-hosted, Apache-2.0, vLLM-served)

| Role | Pick | HF id | VRAM (fp16) | Why |
|---|---|---|---|---|
| Embedding (primary) | Qwen3-Embedding-4B | `Qwen/Qwen3-Embedding-4B` | ~8 GB | MTEB-multilingual top tier (~69.5), 32k seq, instruction-aware, Russian; sweet spot for the 2× A5000 budget |
| Embedding (max quality) | Qwen3-Embedding-8B | `Qwen/Qwen3-Embedding-8B` | ~16 GB | MTEB-multilingual #1 (~70.6) if VRAM allows |
| Embedding (light fallback) | BGE-M3 | `BAAI/bge-m3` | ~2 GB | hybrid dense+sparse+ColBERT in one, battle-tested, MIT |
| Reranker | Qwen3-Reranker-4B | `Qwen/Qwen3-Reranker-4B` | ~8 GB | same family, vLLM `/v1/rerank` |
| Reranker (light) | bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | ~2 GB | CPU-friendly, multilingual |

vLLM serves embeddings (`/v1/embeddings`) and rerank (`/v1/rerank`) as first-class
endpoints. Caveat: the OpenAI-compatible server does not yet pass a per-request
`instruction` field to the Qwen3 instruction-aware models — prepend the task
instruction to the query text inside the memory service before embedding.

## 5. Self-improvement — making models smarter over time (ranked by practicality)

1. **Reflexion** — verbal self-correction stored as episodic memory; zero training,
   works for external + internal alike. First to wire in.
2. **ExpeL** — abstracts cross-task natural-language *insights/rules* from successful
   + failed trajectories; stored in the memory service, injected as few-shot. Pure
   memory layer.
3. **Voyager-style skill library** — verified executable skills indexed by NL
   description; compounding capability for tool/code agents (needs a sandbox to verify).
4. **Self-RAG** — train (or prompt) a critic to gate retrieval + self-validate.
5. **Memory → LoRA distillation** — periodically distill accumulated high-value
   memory/trajectories into a LoRA adapter via continual instruction tuning;
   **vLLM hot-swaps LoRA adapters natively.** This is the realistic *"internal model
   gains new capabilities in weights"* path (requirement 5). External API models can't
   take adapters → for them the memory layer is the ceiling.
6. **SEAL** (Self-Adapting LLMs, MIT 2025) — model writes its own training data +
   self-edits weights. Highest ceiling, but research-grade: needs SFT+RL infra and has
   documented catastrophic forgetting on sequential edits. Roadmap target, not a near-term build.

**Practical sequence:** Reflexion + ExpeL first (no training, immediate, universal) →
Voyager skills if there are tool agents → memory→LoRA as the weights-level loop for the
internal Qwen3.6 → SEAL only as a long-horizon experiment.

## 6. Phased rollout

- **Phase 1 — shared retrieval brain (1–2 weeks).** Stand up the memory service
  (cognee API-mode) on the homelab: Postgres + pgvector (+ Neo4j or Kuzu for the
  graph). Point its LLM/embedding at the proxy/vLLM (`hosted_vllm/`). Serve
  Qwen3-Embedding-4B + Qwen3-Reranker-4B on vLLM. Add the proxy memory-middleware
  (inject on read, async capture on write). Seed the graph from the existing
  `chat_rag` corpus (the current simple project-knowledge RAG becomes the bootstrap
  dataset; the service supersedes it). → external + internal models share one
  connected memory immediately.
- **Phase 2 — self-evolution (1 week).** Turn on `memify` consolidation (edge
  reweighting + stale pruning) + ontology; add Reflexion/ExpeL insight capture. →
  the brain reorganizes and improves with use.
- **Phase 3 — internal weights loop (2–4 weeks).** Nightly distiller selects
  high-value, validated memory → builds a small SFT set → trains a LoRA adapter →
  hot-swaps into vLLM. Guard against forgetting (held-out eval gate before swap). →
  the internal model gains new capability, not just recall.

## 7. Integration constraints / risks (verified)

- **Tool-calling models only:** memory-tool frameworks (Letta etc.) require the routed
  model to support function calling — fine for Qwen3.6 + major external APIs, but the
  middleware approach (inject-into-prompt) avoids this dependency entirely and is why
  it's preferred over a tool-only design.
- **Embeddings endpoint required:** the graph engines need an embeddings endpoint
  alongside chat — ensure the proxy/aggregator exposes one (or point the service at the
  vLLM embed server directly).
- **Ingestion is LLM-heavy:** graph extraction (cognee `cognify`, Graphiti per-episode)
  costs tokens/latency on every write — run it on the internal 35B (off the user path,
  async) and budget for it; APC (now enabled) makes the repeated-context part cheap.
- **API churn:** cognee (121 releases) and Mem0 (v3 dropped external graph DBs)
  iterate fast — pin a version.
- **Separate-repo touch:** the proxy middleware + aggregator wiring live in
  `Genesis_proxy_ai` / `Genesis_agregator_ai` (separate projects) — those changes are
  scoped there, not in this engine repo.

## 8. Bottom line

Stand up **cognee** (API-mode, Postgres+pgvector+graph) as the shared brain behind the
**proxy memory-middleware**, with **Qwen3-Embedding-4B + Qwen3-Reranker-4B** on vLLM;
layer **Reflexion + ExpeL** for immediate cross-model learning; and add the
**memory→LoRA nightly distiller** as the one path that makes the *internal* Qwen3.6
smarter in weights. Use **Graphiti** instead of cognee if auditable bi-temporal fact
validity outranks Hebbian consolidation. Validate the comparative quality claims on
our own data before locking the choice.

### Key sources
- cognee: https://github.com/topoteretes/cognee · https://docs.cognee.ai/setup-configuration/llm-providers
- Graphiti: https://github.com/getzep/graphiti · https://arxiv.org/abs/2501.13956
- HippoRAG 2: https://arxiv.org/abs/2502.14802 · https://github.com/OSU-NLP-Group/HippoRAG
- Mem0: https://github.com/mem0ai/mem0 · https://arxiv.org/abs/2504.19413
- MemOS: https://github.com/MemTensor/MemOS · https://arxiv.org/abs/2505.22101
- LightRAG: https://github.com/HKUDS/LightRAG · https://arxiv.org/abs/2410.05779
- SEAL: https://arxiv.org/abs/2506.10943 · ExpeL: https://arxiv.org/abs/2308.10144
- Qwen3-Embedding/Reranker: https://github.com/QwenLM/Qwen3-Embedding · https://hf.co/Qwen/Qwen3-Embedding-4B

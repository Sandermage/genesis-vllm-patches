# Preset catalog — operator guide

How to find, inspect, and choose between Genesis V2 presets using
`sndr preset list / show / explain / recommend`. Companion to
[`CLI_REFERENCE.md`](CLI_REFERENCE.md) §8 (flag-level surface) and
[`MODELS.md`](MODELS.md) (model-side catalog).

> Quick reference table of the 14 currently-annotated production
> presets is in §6 below. Card schema reference lives next to the
> Python types in [`preset_schema.py`](../vllm/sndr_core/model_configs/preset_schema.py).

## What a preset is

A **preset** is the operator-facing entrypoint that resolves to a
specific (model × hardware × profile) triplet plus a piece of
operator-product metadata called a `PresetCard`. The card describes
**what the preset is for** — workload, hardware envelope, K/MTP
policy, evidence, fallback, do-not-use conditions — not how it is
implemented. Implementation lives in the V2 layered config tree
(`model_configs/builtin/{model,hardware,profile}/`).

```text
                ┌───────────────────────────────┐
                │  preset YAML                  │
                │    model    : <ModelDef.id>   │  ← implementation
                │    hardware : <HardwareDef.id>│
                │    profile  : <ProfileDef.id> │
                │    card:    │
                │      title  │
                │      status │  ← operator-product surface
                │      workload_allow / deny
                │      evidence_refs
                │      fallback_preset
                │      ...
                └───────────────────────────────┘
```

The card is optional: legacy 3-pointer presets continue to compose
and run. Annotated cards unlock `list` / `show` / `explain` /
`recommend` operator surfaces.

## When to use which command

Decision tree for the four `sndr preset` leaves:

| You want to... | Use |
|---|---|
| ...see every preset (or filter by status / family / workload / mode / hardware) | `sndr preset list` |
| ...inspect ONE preset's card in full (workload contract, evidence, tradeoffs, do-not-use) | `sndr preset show <id>` |
| ...validate that the YAML triplet composes to the runtime claimed in the card | `sndr preset explain <id>` |
| ...describe a workload and let the CLI pick the right preset for you | `sndr preset recommend --workload <W>` |

Quick distinctions vs neighbouring commands:

- `sndr config explain <id>` dumps the raw composed YAML. `sndr preset show` renders the operator-product card view.
- `sndr profile show <id>` inspects the patches-delta layer. `sndr preset show` reads the higher-level card metadata.
- `sndr routing-table` emits the workload→preset routing policy at the project level. `sndr preset recommend` queries it per-operator-question with extra filters.

## Anatomy of a card

The card sections an operator most often reads:

| Section | What it answers |
|---|---|
| `status` | How production-ready this preset is (see §4 below). |
| `audience` | Who the preset is for (operator / dev / bench / qa / internal). |
| `mode` | Runtime mode shape (throughput / structured_throughput / latency / long_context / tool_agent). |
| `workload_allow` / `workload_deny` | Which workload classes the preset is tuned for (allow) and explicitly hostile to (deny). |
| `concurrency.{min,canonical,max}` | Tested concurrency envelope. Recommend filters on `[min..max]`. |
| `K` | Spec-decode draft tokens (1 = no draft / no speculation). |
| `context.max_model_len` | Context-window cap for this preset's sizing. |
| `routing_family` | Family the preset belongs to (shared cross-preset routing). |
| `default_for_family` | One preset per family is the "if in doubt, use me" default. |
| `fallback_preset` | The conservative alternative if this preset doesn't fit (K>1 presets require one). |
| `primary_metric` + `evidence_refs` | Headline number plus where it came from. Visibility is public / private / mixed. |
| `tradeoffs` | Operator-language statements about cost vs benefit. |
| `do_not_use` | Anti-patterns: condition + reason pairs. |

## Status semantics

`card.status` is the operator-product lifecycle. The ladder, from
strictest to most permissive:

| Status | Meaning |
|---|---|
| `production` | Public baseline cross-validates the preset's runtime; ready for operator-facing production traffic. |
| `production_candidate` | All production-grade fields present but at least one is private-only evidence; promote to `production` once a matching public bench lands. |
| `internal_validated` | Validated by the maintainer for a specific purpose; not a public production claim. |
| `bench_pending` | Loadable preset, primary metric not yet measured. |
| `experimental` | Loose validation; for exploration. |
| `qa` | Lives in the QA harness, not for end-user. |
| `example` | Demonstration / onboarding preset. |
| `historical` | Kept for reproducibility / regression diffs only. |
| `tombstone` | Empirically broken or superseded; `get()` raises with reason. |

At the time of writing, all 14 prod-\* presets sit at
`production_candidate` — public baseline JSONs in
[`tests/integration/baselines/`](../tests/integration/baselines/) cover
the model family but lack the per-preset `config` block cross-validation
that `production` requires. Promotion to `production` is a CONFIG-UX.4
audit-driven decision (separate from card annotation).

### Card status vs profile `override_policy` — two separate ladders

The card's `status` describes **operator-product maturity** of the
*preset*. The profile's `override_policy.class` describes **evidence
maturity** of any `sizing_override` block on the underlying *profile*
(see `docs/CONFIGURATION.md` → "Override policy"). They are independent:

- A `production_candidate` preset CAN compose a profile carrying
  `override_policy.class: bench` (sizing override has bench evidence
  but not full production cross-validation). This is the common state
  for the 14 prod-\* presets today.
- A `production` preset MUST compose a profile carrying
  `override_policy.class: production` (or no `sizing_override` at all).
  This is the gating contract for the promotion step.

`audit_override_policy.py` (default-on at Stage 1) flags any
`sizing_override` block lacking a matching `override_policy`. Class-4
forbidden override predicates fire unconditionally — see
`docs/CONFIGURATION.md` → "Class-4 forbidden overrides".

### Browse cards via the derived catalog

The derived config catalog (`sndr config-catalog`,
[`CLI_REFERENCE.md`](CLI_REFERENCE.md) §13) exposes the same card data
as `sndr preset show` plus the underlying profile / model / hardware
/ baseline rows. Use it for **read-only inspection** when scripting or
when the operator-facing CLI is too narrow:

```bash
# Show a preset's full card + composed config
sndr config-catalog show preset/prod-qwen3.6-35b-balanced

# Find every profile with bench-class override_policy expiring soon
sndr config-catalog query --row-type profile \
                          --field override_expires_at \
                          --expires-before 2026-09-01
```

The catalog is a derived API (rebuilt on demand, no committed JSON);
`sndr preset` remains the operator-facing surface for daily use.

## Evidence visibility

`card.evidence_visibility` (and per-`evidence_refs[].visibility`)
classifies how reproducible the claim is:

- `public` — committed artefact (e.g. `tests/integration/baselines/*.json`); any operator can re-run.
- `private` — lives in the maintainer-private archive (gitignored; see [`CORE_ENGINE_BOUNDARY.md`](CORE_ENGINE_BOUNDARY.md) for the three-zone namespace policy). Operator-side reproduction requires the maintainer's bench rig.
- `mixed` — at least one ref of each visibility.

Operator rule (per [`CORE_ENGINE_BOUNDARY.md`](CORE_ENGINE_BOUNDARY.md)
+ [`LICENSE_POLICY.md`](LICENSE_POLICY.md)): public docs may not link
into the private maintainer tree; the audit gate
[`audit_public_docs.py`](../scripts/audit_public_docs.py) enforces.
Private evidence in a card is fine because cards live next to source.

## Quick reference — 14 production-facing presets (21 carded total)

Manually curated from the cards (last refresh 2026-05-30). After
CONFIG-UX.2b closure (2026-05-30), **all 21 builtin presets carry
operator cards**:

- 14 `production_candidate` — listed below by family
- 7 non-production (2 `qa`, 3 `example`, 1 `experimental`,
  1 `bench_pending`) — listed at the end of this section under
  [Non-production presets](#non-production-presets-7-carded)

Updated when new presets are annotated; auto-generation deferred to a
future generator phase.

### Qwen 3.6 27B INT4 TQ family — `qwen3_6_27b_int4_tq`

| Preset | K | Concurrency | Mode | Best for |
|---|---:|---:|---|---|
| `prod-qwen3.6-27b-tq-k8v4` ★default | 3 | 1..4 | throughput | Long-context single-stream; 262K context cap. |
| `prod-qwen3.6-27b-tq-multiconc` | 3 | 1..8 | throughput | Multi-conc throughput (379 TPS @ conc=8); 131K context. |

### Qwen 3.6 27B DFlash family — `qwen3_6_27b_int4_dflash`

| Preset | K | Concurrency | Mode | Best for |
|---|---:|---:|---|---|
| `prod-qwen3.6-27b-dflash` ★default | 5 | 1 | throughput | Single-stream code-gen; bf16 separate drafter; 185K context. |
| `prod-qwen3.6-27b-dflash-multiconc` | 5 | 1..8 | throughput | Multi-conc 385 TPS @ conc=8 (3.76x scaling). |

### Qwen 3.6 35B-A3B FP8 family — `qwen3_6_35b_a3b_fp8`

| Preset | K | Concurrency | Mode | Best for |
|---|---:|---:|---|---|
| `prod-qwen3.6-35b-balanced` ★default | 3 | 1..2 | throughput | Balanced single-stream; 280K context. |
| `prod-qwen3.6-35b-multiconc` | 3 | 1..8 | throughput | **Reference free-chat multi-conc** — 689 TPS @ conc=8. |

### Qwen 3.6 35B-A3B FP8 DFlash family — `qwen3_6_35b_a3b_fp8_dflash`

| Preset | K | Concurrency | Mode | Best for |
|---|---:|---:|---|---|
| `prod-qwen3.6-35b-dflash` ★default | 3 | 1 | latency | Single-stream DFlash N=3; 65K context. |
| `prod-qwen3.6-35b-dflash-multiconc` | 3 | 1..8 | throughput | TTFT-tuned multi-conc (562 TPS, TTFT 162 ms — vs 689 TPS / 243 ms on `prod-qwen3.6-35b-multiconc`). |

### Gemma 4 26B-A4B MoE family — `gemma4_moe_26b_a4b`

| Preset | K | Concurrency | Mode | Best for |
|---|---:|---:|---|---|
| `prod-gemma4-26b-default` ★default | 1 | 1..2 | throughput | K=1 control / MTP off baseline; serves as fallback for K=4 siblings. |
| `prod-gemma4-26b-mtp-k4` | 4 | 1..2 | structured_throughput | K=4 single-stream structured (tool_call / structured_json.short). |
| `prod-gemma4-26b-multiconc` | 4 | 1..8 | structured_throughput | K=4 multi-conc structured (235.9 TPS @ conc=8 Mode A). |
| `prod-gemma4-26b-multiconc-k1` | 1 | 1..8 | throughput | K=1 multi-conc B4 comparator (diagnostic baseline). |

### Gemma 4 31B dense family — `gemma4_dense_31b_tq`

| Preset | K | Concurrency | Mode | Best for |
|---|---:|---:|---|---|
| `prod-gemma4-31b-tq-default` ★default | 1 | 1..2 | throughput | Dense 31B, MTP off, broad workload coverage. |
| `prod-gemma4-31b-tq-mtp-structured-k4` | 4 | 1 | structured_throughput | β'-A control: K=4 structured + acceptance artefact gate. |

### Non-production presets (7 carded)

These cards exist so the operator product surface (`sndr preset
list/show/explain/recommend`) covers the full builtin catalog. They
are NOT eligible for production routing — `card.status` and
`workload_deny` keep them out of the production allow-set.

| Preset | Status | Audience | Mode | Best for |
|---|---|---|---|---|
| `qa-qwen3.6-27b-tested` | qa | qa | throughput | Regression baseline vs `prod-qwen3.6-27b-tq-k8v4` (fp8 KV variant). |
| `qa-qwen3.6-27b-tq-1x` | qa | qa | throughput | Single-card 78K context; 27B TQ k8v4 QA. |
| `example-2x-tier-aware` | example | dev | throughput | Path C tier-aware cache demo (PN95). |
| `example-3090-dense-cpu-offload` | example | dev | latency | club-3090 Path A dense + CPU offload demo. |
| `example-3090-tier-aware` | example | dev | long_context | club-3090 Path C tier-aware demo (145K ctx). |
| `experimental-qwen3.6-27b-tq-dflash-ab` | experimental | bench | structured_throughput | dflash A/B experimental variant. |
| `long-ctx-qwen3.6-27b` | bench_pending | bench | long_context | 280K context probe (fp8 KV); bench refresh pending. |

## Workload taxonomy

The `card.workload_allow` / `workload_deny` fields use a frozen
canonical taxonomy. `sndr preset recommend --workload <W>` accepts
only these strings (or a `custom:<slug>` escape):

| Workload | What it covers |
|---|---|
| `free_chat` | Open-ended conversational generation. |
| `structured_json.short` | JSON / structured output ≤ ~200 tokens (tool args, small schemas). |
| `structured_json.long` | JSON output > ~200 tokens (large schema responses). |
| `tool_call.short` | Single-step tool invocation. |
| `tool_call.long` | Multi-step / nested tool-call sequences. |
| `summarization` | Document summarization. |
| `code_gen` | Code generation / autocompletion. |
| `long_context_qa` | Long-context retrieval / QA. |

Custom workloads use `custom:<slug>` (slug matches `[a-z0-9._-]+`).
Recommend ranks them only when a card's `workload_allow` lists the
exact same string.

## Examples

Find the best preset for short structured JSON at concurrency 4 on
the homelab rig:

```bash
sndr preset recommend \
  --workload structured_json.short \
  --hardware a5000-2x-24gbvram-16cpu-128gbram \
  --concurrency 4
```

Inspect what `prod-qwen3.6-35b-multiconc` actually does at runtime (composed
config + fallback diff vs `prod-qwen3.6-35b-balanced`):

```bash
sndr preset explain prod-qwen3.6-35b-multiconc
```

Drill into a specific evidence path:

```bash
sndr preset show prod-qwen3.6-35b-balanced --field card.evidence_refs.0.path
# tests/integration/baselines/35b_v11_wave9.json
```

List all production_candidate presets for a specific family:

```bash
sndr preset list --status production_candidate --family qwen3_6_35b_a3b_fp8
```

Confirm a workload-deny is honoured (Gemma4 K=4 must NOT appear for
free_chat queries — its `workload_deny` lists `free_chat`):

```bash
sndr preset recommend --workload free_chat \
                      --hardware a5000-2x-24gbvram-16cpu-128gbram \
                      --concurrency 8
# Returns Qwen 35B multi-conc presets first; Gemma4 K=4 is excluded.
```

## See also

- [`CLI_REFERENCE.md`](CLI_REFERENCE.md) §8 — flag-level surface for `sndr preset`
- [`CLI_REFERENCE.md`](CLI_REFERENCE.md) §13 — derived config catalog (`sndr config-catalog`)
- [`CONFIGURATION.md`](CONFIGURATION.md) — `override_policy`, `SNDR_V1_ROLLOUT_STAGE`, Class-4 forbidden overrides
- [`MODELS.md`](MODELS.md) — model-side catalog (one row per ModelDef)
- [`PATCHES.md`](PATCHES.md) — patch taxonomy (referenced by `profile.patches_delta`)
- [`CORE_ENGINE_BOUNDARY.md`](CORE_ENGINE_BOUNDARY.md) — three-zone namespace policy that governs evidence visibility
- [`BENCHMARKS.md`](BENCHMARKS.md) — bench methodology that produces the evidence_refs targets

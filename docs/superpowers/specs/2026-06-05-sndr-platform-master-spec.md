# sndr-platform — Master Engineering Specification

**Document version**: 1.0.0
**Date**: 2026-06-05
**Status**: APPROVED — execution started
**Owner**: Sander Barzov (Sandermage)
**Audience**: Principal engineers, future maintainers, security reviewers, contributors
**Repository**: `sndr-platform` (public, Apache 2.0) + `sndr-engine` (private, proprietary)
**Document type**: Living Master Specification — updates through ADRs

---

## How to read this document

This is **not** a sales pitch or a product brief. It is a corporate engineering
specification written at the level a Principal Engineer would expect when joining
the team. Read it in this order:

1. **Part 1 (Vision)** — what we are building and why
2. **Part 2 (Principles)** — non-negotiable rules every contribution follows
3. **Part 3 (Architecture)** — the physical and logical structure
4. **Part 16 (Migration Plan)** — how we get from current state to the target

Everything else is reference material to consult when you encounter the relevant
problem (error handling, observability, security, etc.).

The document is **self-contained**: a new engineer should be able to understand
the entire system from this file alone. External references (ADRs, code) are
linked but never required for high-level understanding.

---

## Table of Contents

- [Part 1: Vision and Goals](#part-1-vision-and-goals)
- [Part 2: Engineering Principles](#part-2-engineering-principles)
- [Part 3: Repository Architecture](#part-3-repository-architecture)
- [Part 4: Module Architecture and Dependency Rules](#part-4-module-architecture-and-dependency-rules)
- [Part 5: Algorithmic Chains (boot, apply, drift, upgrade, license)](#part-5-algorithmic-chains)
- [Part 6: GUI Architecture (Carbon Design System)](#part-6-gui-architecture-carbon-design-system)
- [Part 7: API Design Rules](#part-7-api-design-rules)
- [Part 8: Error Handling Strategy](#part-8-error-handling-strategy)
- [Part 9: Observability Standards](#part-9-observability-standards)
- [Part 10: Security Model](#part-10-security-model)
- [Part 11: Performance Budgets](#part-11-performance-budgets)
- [Part 12: Testing Strategy](#part-12-testing-strategy)
- [Part 13: CI/CD Pipeline](#part-13-cicd-pipeline)
- [Part 14: Documentation Framework](#part-14-documentation-framework)
- [Part 15: Repository Hygiene](#part-15-repository-hygiene)
- [Part 16: Migration Execution Plan](#part-16-migration-execution-plan)
- [Part 17: Glossary](#part-17-glossary)
- [Part 18: Frequently Asked Questions](#part-18-frequently-asked-questions)
- [Part 19: Risk Register](#part-19-risk-register)
- [Part 20: Success Metrics](#part-20-success-metrics)
- [Appendices](#appendices)

---

# Part 1: Vision and Goals

## 1.1 The problem we are solving

`genesis-vllm-patches` started as a personal patch overlay for one engine (vLLM)
on one operator's hardware (2× RTX A5000). It grew into:

- 333 patches across 26 subdirectories
- A commercial paid tier (`sndr_engine`) with 14-16 engine-licensed patches
- A full enterprise React GUI (11,633 LOC `App.tsx`) with multi-host fleet, auth,
  container management
- An automation surface (CLI, REST API, SSE streams)
- A bench harness, drift detection, manifest tooling
- Operator scripts, config builders, doctor commands

The current architecture has structural problems that block further growth:

1. **`sndr_core` lives under `vllm/sndr_core/`** — our code masquerades as part
   of the vLLM package. There is no clean separation between engine and
   orchestrator.
2. **No engine abstraction** — adding sglang as a second engine would require
   rewriting most of the codebase.
3. **`integrations/attention/` has 101 files in one flat directory** — the
   organization broke down a long time ago.
4. **`product_api/` has 45 files at the root** — no domain/route separation.
5. **`gui/web/src/App.tsx` is 11,633 lines** — the largest file in the project
   is the GUI entry point, which is unmaintainable.
6. **Drift detection is manual** — every pin upgrade is a multi-step playbook
   that can be forgotten.
7. **The GUI styling is bespoke** — no design system, inconsistent components,
   no accessibility audit.

## 1.2 What we are building

**`sndr-platform`** is a multi-engine inference-stack orchestration platform with
three components:

| Component | License | Audience |
|---|---|---|
| `sndr` Python package | Apache 2.0 (public) | Operators of any open-source engine |
| `sndr-engine` Python wheel | Proprietary (commercial) | Paid customers needing licensed kernels |
| `sndr Control Center` GUI | Apache 2.0 (public) | DevOps operators managing inference fleets |

The platform supports:

- **Multi-engine**: vLLM today; sglang next; TensorRT-LLM/Triton later
- **Multi-pin**: track 2–3 pins per engine simultaneously (current, previous,
  staging) with per-pin patch compatibility
- **Drift-aware**: automated detection and reporting when upstream code changes
  break our anchors
- **Manifest-based**: per-pin YAML manifests separating "what code to find" from
  "what to do with it"
- **Enterprise-ready**: structured logging, Prometheus metrics, OpenTelemetry
  traces, RBAC, audit logs, license gating
- **GUI-first**: every action surfaceable through Carbon Design System UI
- **Internationalized**: English + Russian at launch; i18n framework ready for
  more languages

## 1.3 Non-goals

We are explicitly **not** building:

- A model server (vLLM and sglang already do this)
- A model registry or model card system
- A workload orchestrator (Kubernetes operators)
- A general-purpose patch-management framework (we are specifically for inference
  engines)
- A desktop application via Tauri (web-only for v12; reconsider for v13+)

## 1.4 Success looks like

By the end of this refactor (v12.0.0):

- New maintainer can onboard in 2 hours from `docs/QUICKSTART.md`
- Pin upgrade is a single command (`sndr pin upgrade --to X`)
- Drift detection produces issues automatically every day
- Adding a new patch requires creating one file and one registry entry
- GUI loads in < 2 seconds (cold), < 500 ms (warm)
- 85%+ test coverage on critical paths
- Zero "magic" environment variables — every behavior is documented
- sglang skeleton is in place for future contributions
- Bench reproducibility CV < 8% sustained across runs

---

# Part 2: Engineering Principles

These principles are **non-negotiable**. Every pull request, every review, every
architectural decision is checked against them. If a principle seems to be in the
way, propose changing it through an ADR — do not violate it silently.

## 2.1 Code-level principles

### Principle: Explicit over implicit

No hidden behavior. Every environment variable, every default value, every
"convention" must be documented in code and reference docs.

**Concrete enforcement**:
- All env vars referenced in code must have a corresponding entry in
  `docs/reference/ENV_VARS.md`
- All defaults must be expressed as named constants, never inline literals
- "It just works" magic auto-imports are forbidden — use explicit registration

**Why**: Implicit behavior accumulates as folk knowledge. New maintainers cannot
discover what they cannot see. Audit becomes impossible. Production incidents
become "I didn't know X was happening" stories.

**Example of violation**:
```python
# BAD: implicit auto-discovery
def apply_all():
    for module in os.listdir("patches/"):  # who knew patches/ was scanned?
        import_module(module).apply()
```

**Example of compliance**:
```python
# GOOD: explicit registration via entry points
def apply_all():
    for entry in entry_points(group="sndr.engines.vllm.patches"):
        entry.load().apply()  # entry points are documented contracts
```

### Principle: Composition over inheritance

Inheritance is allowed only for ABC contracts (interfaces). Behavior is composed
from small functions and small dataclasses.

**Concrete enforcement**:
- `class X(EngineAdapter):` — OK, implementing a contract
- `class VllmEngineV2(VllmEngine):` — NOT OK, inheritance for code reuse
- Patches are not classes hierarchies; they are functions or dataclasses

**Why**: Inheritance couples implementations. Future engines and future patches
have nothing in common with each other except the contract they expose. Coupling
through inheritance prevents independent evolution.

### Principle: Single source of truth (SSOT)

Every fact has exactly one authoritative location:

| Fact | SSOT location |
|---|---|
| Patch metadata | `sndr/engines/<engine>/patches/registry.py` |
| Type definitions | `sndr/product_api/schemas/` (Pydantic) |
| Configuration | `sndr.config.SndrConfig` (typed dataclass) |
| Per-pin anchors | `sndr/engines/<engine>/pins/<pin>/manifest.yaml` |
| Brand colors | `gui/web/src/theme/tokens.ts` |
| API contract | FastAPI OpenAPI generation, never hand-maintained |

**Why**: Duplication invariably drifts. When two locations disagree, neither is
trustworthy. Bugs hide in the divergence.

### Principle: Fail fast, fail loud, fail safe

- **Fail fast**: detect errors at boot, not at first request
- **Fail loud**: structured log + metric + (where relevant) operator alert
- **Fail safe**: degraded mode preserves correctness (community-only on license
  failure; previous pin on upgrade failure)

**Concrete enforcement**:
- Boot validates configuration, manifest md5s, license, all patches' applies_to
- All errors logged with `level=error`, structured fields, full stack trace
- Catastrophic errors raise typed exceptions; non-catastrophic degrade with WARN

### Principle: Layered architecture, downward dependencies only

Modules are organized into numbered layers. Layer N may import from layers 0..N-1,
**never** from N+1.

The layers are:
- 0: `kernel/` — engine-agnostic primitives (text_patch, manifest, multi_file)
- 1: `engines/` — engine adapters
- 2: `dispatcher/` — registry, applies_to gating, tier check, version range
- 3: `apply/` — orchestrator that drives the boot-time patch loop
- 4: `product_api/` — REST/SSE/WS layer
- 5: `cli/` — command-line interface

**Enforcement**: `tools/ci/verify_layer_imports.py` parses every Python file's
imports and fails the build if a forbidden upward import is found.

### Principle: Versioned contracts

Anything other code depends on is versioned:

- REST API: `/api/v1/`, `/api/v2/`
- PatchSpec schema: explicit `version: int` field
- Manifest format: `version: 1` at top of file
- Engine adapter ABC: minor versions extend, major versions break

**Why**: Without versions, consumers cannot know what to expect. Breaking changes
silently corrupt clients. With versions, deprecation cycles become possible.

### Principle: Deprecation discipline

Removed code goes through:
1. **Deprecation announcement**: changelog entry, runtime warning, deprecation
   marker in docstring
2. **One release cycle** during which old + new coexist
3. **Removal**: clean delete, changelog entry confirming removal

**Why**: Customers and downstream contributors have running systems. Breaking
them without warning destroys trust.

## 2.2 Architectural principles

### Principle: Twelve-Factor App compliance

Standard cloud-native discipline:
1. Codebase: single repo per deployable artifact
2. Dependencies: explicit declarations, isolated env (pyproject.toml)
3. Config: environment variables only (no config files committed)
4. Backing services: treat databases/caches as attached resources
5. Build, release, run: strictly separated stages
6. Processes: stateless and share-nothing
7. Port binding: self-contained, no runtime injection
8. Concurrency: horizontal scaling via process model
9. Disposability: fast startup, graceful shutdown
10. Dev/prod parity: dev mirrors prod as closely as possible
11. Logs: streams of events, not files
12. Admin processes: one-off tasks run in identical environment

### Principle: Strangler-fig migration

We migrate by adding new structure alongside old, deprecating old, removing old
last. Never rip-and-replace. Always reversible at phase boundaries.

**Why**: A 12-week refactor cannot be "go live" on one Friday. Phases must be
shippable independently. If phase 5 reveals a flaw in phase 3's design, we
rollback phase 5 only.

### Principle: Adapter pattern for engines

Every engine implements `EngineAdapter` ABC. Core orchestrator code is
engine-agnostic. Engine-specific code is fully contained in `engines/<name>/`.

**Why**: This is the foundation of multi-engine support. Without it, sglang
adoption is a rewrite. With it, sglang adoption is a port.

### Principle: Plugin discovery via entry points

The commercial wheel `sndr-engine` registers patches through `setuptools` entry
points, not hardcoded imports. The community `sndr` package does not know that
`sndr-engine` exists at build time.

```python
# pyproject.toml of sndr-engine
[project.entry-points."sndr.engines.vllm.patches"]
p67 = "sndr_engine.vllm.patches.p67:patch"
pn21 = "sndr_engine.vllm.patches.pn21:patch"
```

```python
# sndr/plugins/loader.py
from importlib.metadata import entry_points

def discover_engine_patches(engine: str) -> list[PatchSpec]:
    """Load patches registered by external wheels."""
    patches = []
    for entry in entry_points(group=f"sndr.engines.{engine}.patches"):
        patches.append(entry.load())
    return patches
```

**Why**: Plugin discovery decouples the public package from proprietary content.
A public user without a license can install `sndr` and get full community
functionality. A paying customer adds `sndr-engine` and gets more patches with
zero code changes in `sndr`.

### Principle: Repository pattern for state

All persistence is abstracted behind repositories:

```python
class SessionRepository(Protocol):
    def save(self, session: Session) -> None: ...
    def find(self, token: str) -> Session | None: ...
    def delete(self, token: str) -> None: ...

class JsonSessionRepository:
    """Default: JSON files in $SNDR_HOME/sessions/"""
    ...

class SqliteSessionRepository:
    """For multi-process deployments."""
    ...
```

**Why**: We need flexibility to swap storage backends. Today JSON works. Tomorrow
we may want SQLite for concurrent multi-host control. The application code does
not need to change.

### Principle: Command pattern for CLI

Each CLI command is an isolated class implementing a `Command` protocol:

```python
class Command(Protocol):
    name: str
    description: str
    def configure_parser(self, parser): ...
    def execute(self, args) -> int: ...

class PatchesListCommand:
    name = "patches list"
    description = "List patches with optional filters"
    def configure_parser(self, parser):
        parser.add_argument("--engine", default="vllm")
        parser.add_argument("--filter")
    def execute(self, args):
        ...
```

Commands are testable in isolation. Adding a new command requires no changes to
the dispatcher.

## 2.3 Operational principles

### Principle: Zero-downtime upgrades

Every change preserves rollback capability for at least one previous version.

- **Pin upgrade**: previous image kept as `nightly-previous` for instant rollback
- **Manifest updates**: backed up as `manifest.yaml.bak` before regeneration
- **Database migrations** (when introduced): forward + backward compatible for one
  release

### Principle: Observable by default

Every operation produces:
- A **structured log** event
- A **metric** counter or histogram
- A **trace span** with attributes

Silent failures are forbidden. If something can fail, its failure is observable.

### Principle: Defense in depth (security)

License gating uses multiple layers:
1. Token signature (Ed25519)
2. Token expiry
3. Token version compatibility
4. Package import check (`sndr_engine` installed?)
5. Dispatcher tier gate
6. Audit log of every gate evaluation

Compromising any single layer does not grant access. All five must align.

### Principle: Reproducible benchmarks

Bench methodology is versioned. Every bench report includes:
- vLLM pin (full SHA)
- Genesis version
- Hardware (GPU model, count, NCCL config)
- Methodology version (e.g., "v2: 7-warm-run sustained")
- Model config hash
- Container image digest

A report from January and a report from June must be comparable if they share the
same methodology version.

### Principle: Forward-compatible schemas

When extending schemas:
- ✅ Add optional fields
- ✅ Add new enum values (consumer must accept unknown)
- ❌ Remove fields (use deprecation cycle)
- ❌ Rename fields (add new, deprecate old)
- ❌ Change field types

Unknown fields are ignored gracefully. Unknown enum values fall back to a
documented "unknown" handler.

---

# Part 3: Repository Architecture

## 3.1 Top-level repository layout

```
sndr-platform/                          # PUBLIC repository, Apache 2.0
├── sndr/                               # Python package (top-level, not under vllm/)
├── gui/                                # React Control Center
├── tools/                              # Python maintenance tools
├── scripts/                            # Legacy audit scripts (to be migrated to tools/)
├── tests/                              # Python tests (unit, integration, e2e)
├── docs/                               # Public documentation
├── .github/                            # CI/CD workflows
├── pyproject.toml                      # Python package metadata
├── README.md                           # Public-facing entry
├── LICENSE                             # Apache 2.0 text
├── SECURITY.md                         # Security policy + disclosure procedure
├── CONTRIBUTING.md                     # How to contribute
├── CHANGELOG.md                        # Release history
├── .gitignore                          # Strict: sndr_engine, sndr_private
├── .pre-commit-config.yaml             # Lint hooks
├── ruff.toml                           # Python linter config
└── mypy.ini                            # Type checker config
```

**Companion private repository** (separate, not in this tree):

```
sndr-engine/                            # PRIVATE repository, proprietary
├── pyproject.toml                      # Entry points: sndr.engines.vllm.patches
├── sndr_engine/                        # Package code
│   ├── __init__.py                     # Registration helpers
│   └── vllm/                           # Per-engine extensions
│       └── patches/                    # 14-16 engine-tier patches
├── tests/                              # Private tests
└── docs/                               # Customer-facing docs
```

**Local-only files** (gitignored within the public repo):

```
sndr_private/                           # Maintainer-only, never committed
├── planning/                           # Audit notes, design drafts
├── audits/                             # Pin-bump reports, decision logs
├── snapshots/                          # State captures
├── runs/                               # Bench outputs
└── pyproject-engine-template.toml      # Template for sndr-engine wheel
```

## 3.2 Inside `sndr/` — Python package

```
sndr/
├── __init__.py                         # Entry: sndr.init()
├── version.py                          # Single source: __version__ = "12.0.0"
├── config.py                           # Typed SndrConfig dataclass
├── license.py                          # Ed25519 verifier (commercial gate)
├── exceptions.py                       # Typed exception hierarchy
│
├── kernel/                             # LAYER 0 — engine-agnostic primitives
│   ├── __init__.py
│   ├── text_patch.py                   # TextPatch, TextPatcher, marker-based idempotency
│   ├── multi_file.py                   # Multi-file atomic transactions
│   ├── manifest.py                     # Per-pin manifest loader and validator
│   ├── orchestrator.py                 # Apply loop (engine-parametrized)
│   └── types.py                        # Shared dataclasses (PatchResult, ApplyOutcome)
│
├── detection/                          # LAYER 0 — engine-agnostic hardware detection
│   ├── __init__.py
│   ├── gpu_arch.py                     # GPU profile (Ampere/Hopper/Blackwell predicates)
│   └── perf_model.py                   # Roofline math (compute/memory ridge)
│
├── dispatcher/                         # LAYER 2 — registry + decision
│   ├── __init__.py
│   ├── registry.py                     # PATCH_REGISTRY single source of truth
│   ├── decision.py                     # applies_to, version_range, tier gating
│   ├── spec.py                         # PatchSpec dataclass
│   ├── audit.py                        # Registry invariants (CI gate)
│   └── lifecycle.py                    # Lifecycle state machine
│
├── apply/                              # LAYER 3 — orchestrator
│   ├── __init__.py
│   ├── boot.py                         # apply_all() entry called by sndr.init()
│   ├── runtime.py                      # Runtime patch hooks (lazy retry, etc.)
│   └── transaction.py                  # Cross-patch atomicity
│
├── engines/                            # LAYER 1 — engine adapters
│   ├── __init__.py                     # ENGINE_REGISTRY + get_engine()
│   ├── base.py                         # EngineAdapter ABC
│   │
│   ├── vllm/
│   │   ├── __init__.py
│   │   ├── adapter.py                  # class VllmEngine(EngineAdapter)
│   │   ├── detection/                  # vllm-specific probes
│   │   │   ├── __init__.py
│   │   │   ├── config.py               # Reads vllm.config.LlmConfig
│   │   │   ├── model.py                # Reads model architecture
│   │   │   ├── guards.py               # Version checks, resolve_file
│   │   │   └── runtime.py              # Live runtime introspection
│   │   ├── pins/                       # Per-pin manifests
│   │   │   ├── 0.21.1.dev354_626fa9bba/
│   │   │   │   ├── manifest.yaml
│   │   │   │   ├── anchors.yaml
│   │   │   │   └── snapshot.md5sums
│   │   │   ├── 0.22.0_0b3ba88f/
│   │   │   └── 0.22.1_da1daf40b/
│   │   ├── patches/                    # 252 community patches
│   │   │   ├── __init__.py
│   │   │   ├── registry.py             # Per-engine patch list
│   │   │   ├── attention/
│   │   │   │   ├── flash_attn/
│   │   │   │   ├── gdn/
│   │   │   │   └── turboquant/
│   │   │   ├── spec_decode/
│   │   │   ├── moe/
│   │   │   ├── kv_cache/
│   │   │   ├── reasoning/
│   │   │   ├── tool_parsing/
│   │   │   ├── model_compat/
│   │   │   └── observability/
│   │   └── kernels/                    # Triton kernels (vllm-tied)
│   │       ├── gdn/
│   │       ├── marlin/
│   │       └── turboquant/
│   │
│   └── sglang/                         # SKELETON for future port
│       ├── __init__.py
│       ├── adapter.py                  # class SglangEngine(EngineAdapter) - stub
│       ├── README.md                   # Porting guide
│       ├── pins/                       # (empty)
│       └── patches/                    # (empty)
│
├── product_api/                        # LAYER 4 — FastAPI backend
│   ├── __init__.py
│   ├── server.py                       # Main FastAPI app
│   ├── middleware.py                   # CORS, auth, rate limit, request-id
│   ├── routes/                         # REST endpoints (one file per resource)
│   │   ├── engines.py
│   │   ├── patches.py
│   │   ├── pins.py
│   │   ├── drift.py
│   │   ├── containers.py
│   │   ├── fleet.py
│   │   ├── hosts.py
│   │   ├── chat.py
│   │   ├── bench.py
│   │   ├── doctor.py
│   │   ├── licensing.py
│   │   ├── auth.py
│   │   └── health.py
│   ├── domain/                         # Business logic (engine-aware)
│   │   ├── memory_estimator.py
│   │   ├── preset_recommender.py
│   │   ├── container_ops.py
│   │   ├── deployment.py
│   │   ├── patch_explain.py
│   │   └── license_status.py
│   ├── auth/                           # KEPT existing structure
│   │   ├── pam.py
│   │   ├── oauth.py
│   │   ├── totp.py
│   │   ├── sessions.py
│   │   ├── ratelimit.py
│   │   └── store.py
│   ├── streaming/
│   │   ├── sse.py                      # Server-Sent Events
│   │   └── websocket.py                # WebSocket terminal/logs
│   ├── schemas/                        # Pydantic models (API contracts)
│   │   ├── patches.py
│   │   ├── engines.py
│   │   ├── pins.py
│   │   ├── containers.py
│   │   └── errors.py
│   └── web_static/                     # Built GUI assets (from gui/web/dist/)
│
├── cli/                                # LAYER 5 — command-line interface
│   ├── __init__.py
│   ├── main.py                         # Entry point: sndr [command] [args]
│   ├── commands/                       # One file per command group
│   │   ├── apply.py
│   │   ├── patches.py
│   │   ├── pins.py
│   │   ├── drift.py
│   │   ├── engines.py
│   │   ├── bench.py
│   │   ├── doctor.py
│   │   ├── gui.py
│   │   ├── license.py
│   │   └── start.py
│   ├── shared/
│   │   ├── output.py                   # JSON / YAML / table formatting
│   │   ├── filters.py                  # Common filter logic
│   │   └── interactive.py              # Confirmation prompts
│   └── completion/                     # Shell completion (bash, zsh, fish)
│
├── model_configs/                      # YAML configs (engine field per config)
│   ├── builtin/
│   └── schema.py                       # Config schema validation
│
├── observability/                      # Engine-agnostic
│   ├── __init__.py
│   ├── logging.py                      # Structured JSON logging setup
│   ├── metrics.py                      # Prometheus metrics
│   └── trace.py                        # OpenTelemetry tracing
│
├── compat/                             # Cross-version helpers
│   ├── predicates.py
│   └── version_check.py
│
├── cache/                              # KV cache tier_manager (NOT license tier)
│
├── runtime/                            # Worker/engine lifecycle helpers
│
└── plugins/                            # Entry-point loader for sndr-engine
    ├── __init__.py
    └── loader.py                       # discover_engine_patches()
```

## 3.3 Inside `gui/` — React Control Center

```
gui/
└── web/
    ├── package.json                    # Node dependencies + scripts
    ├── tsconfig.json                   # TypeScript strict mode config
    ├── vite.config.ts                  # Vite build config
    ├── playwright.config.ts            # E2E test config
    ├── eslint.config.js                # ESLint flat config
    ├── prettier.config.js              # Prettier rules
    ├── public/                         # Static assets (favicon, brand)
    ├── src/
    │   ├── main.tsx                    # Entry point
    │   ├── App.tsx                     # Thin shell (≤200 LOC after refactor)
    │   ├── router.ts                   # Hash router config
    │   ├── theme/                      # Carbon Design System integration
    │   │   ├── tokens.ts               # Brand color tokens
    │   │   ├── ThemeProvider.tsx       # Carbon Theme + brand overlay
    │   │   └── styles.scss             # Global styles
    │   ├── features/                   # Feature modules (one folder per route)
    │   │   ├── patches/
    │   │   │   ├── PatchesView.tsx
    │   │   │   ├── PatchInventoryControl.tsx
    │   │   │   ├── PatchSummaryPanel.tsx
    │   │   │   ├── PatchLifecycleGraph.tsx
    │   │   │   ├── PatchRegistryInsight.tsx
    │   │   │   ├── PatchModelSupport.tsx
    │   │   │   └── api.ts              # Feature-scoped API client
    │   │   ├── engines/                # Engine selector view
    │   │   ├── pins/                   # Pin manager (current/previous/staging)
    │   │   ├── drift/                  # Drift dashboard
    │   │   ├── fleet/                  # Multi-host overview
    │   │   ├── hosts/                  # Per-host management
    │   │   ├── containers/             # Container lifecycle
    │   │   ├── chat/                   # Engine chat interface
    │   │   ├── bench/                  # Benchmark history + run
    │   │   ├── doctor/                 # Diagnostic dashboard
    │   │   ├── auth/                   # Login, 2FA, security panel
    │   │   ├── licensing/              # License status, upload, expiry
    │   │   └── settings/               # User preferences, theme, language
    │   ├── components/                 # Shared UI primitives
    │   │   ├── SegmentBar.tsx
    │   │   ├── KpiCard.tsx
    │   │   ├── DataTable.tsx           # Carbon DataTable wrapper
    │   │   ├── Skeleton.tsx
    │   │   ├── Modal.tsx
    │   │   ├── Dropdown.tsx
    │   │   ├── StatusBadge.tsx
    │   │   └── ErrorBoundary.tsx
    │   ├── api/                        # API client layer
    │   │   ├── client.ts               # Base fetch wrapper with auth + errors
    │   │   ├── schema.gen.ts           # OpenAPI auto-generated types
    │   │   └── errors.ts               # Error normalization
    │   ├── hooks/                      # Reusable hooks
    │   │   ├── useFetch.ts
    │   │   ├── usePoll.ts
    │   │   ├── useSSE.ts
    │   │   ├── useStore.ts             # Zustand selector helper
    │   │   └── useDebounce.ts
    │   ├── stores/                     # Zustand stores (lightweight state)
    │   │   ├── auth.ts
    │   │   ├── engine.ts               # Currently selected engine
    │   │   ├── pin.ts                  # Currently selected pin
    │   │   ├── ui.ts                   # UI state (sidebar, theme density)
    │   │   └── patches.ts              # Patch list cache
    │   ├── i18n/                       # Localization
    │   │   ├── index.ts                # Lingui setup
    │   │   ├── locales/
    │   │   │   ├── en/messages.po
    │   │   │   └── ru/messages.po
    │   │   └── messages.compiled.js    # Build artifact
    │   ├── types/                      # Shared TypeScript types
    │   └── lib/                        # Utility functions
    │       ├── format.ts               # Date/number formatting
    │       ├── validation.ts
    │       └── tokens.ts
    ├── tests/                          # Vitest unit tests (colocated also OK)
    └── e2e/                            # Playwright E2E tests
        ├── smoke.spec.ts
        ├── patches_view.spec.ts
        ├── pin_upgrade.spec.ts
        └── auth_flow.spec.ts
```

## 3.4 Inside `tools/` — maintenance scripts

```
tools/
├── manifest_gen.py                     # Auto-snapshot upstream files into manifest
├── drift_check.py                      # MD5 verification against manifest
├── upgrade_pin.py                      # Orchestrate full pin upgrade pipeline
├── license_keygen.py                   # Generate signed customer tokens (offline)
├── migrate_imports.py                  # Phase 2/3/4 import path rewriter
├── extract_anchors.py                  # Migration: extract _OLD strings into manifests
└── ci/                                 # CI helper scripts
    ├── verify_layer_imports.py         # Enforce layered architecture
    ├── verify_no_engine_leak.py        # Block tier=engine patches in public registry
    ├── check_test_coverage.py
    └── performance_budget.py
```

## 3.5 Recommendation: monorepo or multi-repo?

We chose **single repo** for `sndr-platform` (Python + GUI together).

Reasons:
- Single CI pipeline, simpler release coordination
- GUI built assets bundled into Python wheel (sndr-control-center)
- Tight version coupling (GUI v12 only works with sndr v12)
- No need for npm workspaces overhead

The `sndr-engine` (commercial) is a separate repository because:
- Different license (proprietary)
- Different access (private team only)
- Different release cadence (customer-driven)
- Different security model (signed releases per customer)

---

# Part 4: Module Architecture and Dependency Rules

## 4.1 Layered architecture diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 5: GUI (React + Carbon Design System)                         │
│  Location: gui/web/                                                  │
│  Imports: HTTP API (Layer 4) over REST + SSE + WebSocket             │
│  Restrictions: never imports Python code; communicates via JSON      │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 5 (Python): CLI                                               │
│  Location: sndr/cli/                                                 │
│  Imports: layers 0-4                                                 │
│  Purpose: Operator-facing commands                                   │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 4: HTTP API (FastAPI)                                         │
│  Location: sndr/product_api/                                         │
│  Imports: layers 0-3 + license                                       │
│  Purpose: REST/SSE/WS endpoints for GUI and external clients         │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 3: Apply Orchestrator                                         │
│  Location: sndr/apply/                                               │
│  Imports: layers 0-2                                                 │
│  Purpose: Boot-time patch application loop                           │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 2: Dispatcher (Decision Logic)                                │
│  Location: sndr/dispatcher/                                          │
│  Imports: layers 0-1 + license                                       │
│  Purpose: Registry, applies_to gating, version checks, tier gates    │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Engine Adapters                                            │
│  Location: sndr/engines/                                             │
│  Imports: layer 0 only                                               │
│  Purpose: Engine-specific detection, paths, patches, kernels         │
└──────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 0: Kernel (Engine-Agnostic Primitives)                        │
│  Location: sndr/kernel/, sndr/detection/, sndr/observability/        │
│  Imports: stdlib + approved third-party only                         │
│  Purpose: Generic text patching, manifests, GPU profile, logging     │
└──────────────────────────────────────────────────────────────────────┘
```

## 4.2 Approved third-party dependencies (layer 0)

Layer 0 may import only:
- Python stdlib
- `pydantic` (data validation)
- `prometheus_client` (metrics)
- `opentelemetry-api` (tracing)
- `python-json-logger` (structured logging)
- `pyyaml` (manifest format)
- `cryptography` (license verification, lazy import)

Any addition to this list requires an ADR.

## 4.3 Forbidden import patterns

```yaml
# tools/ci/layer_rules.yaml
forbidden_patterns:
  # No upward imports
  - rule: "kernel imports engines"
    pattern: "sndr.kernel.* imports sndr.engines"
  - rule: "dispatcher imports apply"
    pattern: "sndr.dispatcher.* imports sndr.apply"
  - rule: "engines imports product_api"
    pattern: "sndr.engines.* imports sndr.product_api"
  
  # No cross-engine imports
  - rule: "vllm engine imports sglang"
    pattern: "sndr.engines.vllm.* imports sndr.engines.sglang"
  - rule: "sglang engine imports vllm"
    pattern: "sndr.engines.sglang.* imports sndr.engines.vllm"
  
  # No engine-specific in agnostic locations
  - rule: "kernel imports vllm package"
    pattern: "sndr.kernel.* imports vllm"
  - rule: "detection imports vllm"
    pattern: "sndr.detection.* imports vllm"
```

## 4.4 Communication patterns between layers

### Pattern: Adapter contract

Layer 2 (dispatcher) needs engine-specific info but cannot import engine code
directly (would couple). Solution: call `EngineAdapter` methods.

```python
# sndr/dispatcher/decision.py
def check_applies_to(engine: EngineAdapter, meta: dict) -> tuple[bool, str]:
    """Check if a patch applies, using engine adapter for detection."""
    profile = engine.get_model_profile()  # adapter method, not direct vllm call
    if "model_arch" in meta:
        if not any(arch in meta["model_arch"] for arch in profile.architectures):
            return False, "model_arch mismatch"
    return True, "applies"
```

### Pattern: Event bus

Layers 4 (API) and 5 (GUI) need real-time updates from layers 1-3. Solution:
publish-subscribe via in-process event bus.

```python
# sndr/observability/events.py
class EventBus:
    def emit(self, event: str, payload: dict): ...
    def subscribe(self, event: str, handler: Callable): ...

# Layer 3 emits
bus.emit("patch.applied", {"patch_id": "PN286", "engine": "vllm"})

# Layer 4 (SSE route) subscribes
bus.subscribe("patch.applied", lambda evt: sse_broadcast(evt))
```

### Pattern: Lazy retry

Some operations need data not available at boot but only at first runtime call
(e.g., PN302 model profile needs vllm_config). Solution: lazy retry hook.

```python
# sndr/apply/runtime.py
def install_lazy_retry(patch_id: str, target_class, method: str):
    """Wrap a target method so it runs the patch on first invocation."""
    original = getattr(target_class, method)
    fired = [False]
    
    def wrapper(self, *args, **kwargs):
        if not fired[0]:
            fired[0] = True
            run_deferred_patch(patch_id, vllm_config=self.vllm_config)
        return original(self, *args, **kwargs)
    
    setattr(target_class, method, wrapper)
```

---

# Part 5: Algorithmic Chains

The most important behaviors of the system, fully specified.

## 5.1 Boot sequence (deterministic)

The boot sequence is **deterministic**: given the same inputs (env vars,
manifest, license), the boot produces the same patches applied in the same
order. This is critical for reproducibility, debugging, and security audit.

### Phases of boot

```
T+0   Container starts, Python begins
T+0   vllm package imports (or sglang, depending on engine)
T+0   sndr package imports
T+0   sndr.__init__ runs:
        - Load sndr/version.py
        - Configure observability (logger, metrics, trace)
        - Read SNDR_ENGINE, SNDR_ENGINE_PIN, SNDR_CONFIG env
        - Parse SndrConfig (typed)
        - emit "sndr.lifecycle.imported"

T+1   sndr.init() called explicitly (idempotent):
        Phase A: license
          - Locate license token (env or file)
          - Verify Ed25519 signature
          - Check expiry, version match
          - Cache result for process lifetime
          - emit "sndr.license.verified" with status
        
        Phase B: engine adapter
          - ENGINE_REGISTRY[config.engine] → VllmEngine class
          - Instantiate adapter
          - Adapter.detect_version() → "0.22.1rc1.dev195+gda1daf40b"
          - Adapter.get_pin_manifest(version) → load manifest.yaml
          - Verify manifest md5sums against live files:
              for each file in manifest:
                  computed = md5(file)
                  if computed != manifest.files[file].md5:
                      emit "sndr.drift.detected"
                      if config.strict_drift:
                          raise DriftDetectedError
                      else:
                          log.warning
          - emit "sndr.engine.ready"
        
        Phase C: patch discovery
          - Community patches: enumerate sndr/engines/vllm/patches/
          - Engine patches: load via entry_points if license valid
          - Build PATCH_REGISTRY in deterministic order:
              sort by (priority, ordinal, id)
        
        Phase D: dispatcher decision
          - For each patch:
              - Check env_flag (operator opt-in)
              - Check applies_to (model_arch, version_range, ...)
              - Check tier gate (community always pass; engine needs license)
              - Decide: apply | skip | block
          - Build ordered apply plan
        
        Phase E: orchestrator apply
          - For each patch in plan:
              span = tracer.start_span("apply_patch", attributes={"patch_id": id})
              try:
                  result = patch.apply()
                  span.set_attribute("outcome", result.status)
                  metrics.patches_applied.inc(labels)
                  log.info("patch.applied", extra={"patch_id": id, ...})
              except Exception as e:
                  span.set_attribute("outcome", "failed")
                  metrics.patches_failed.inc(labels)
                  log.error("patch.failed", extra={"patch_id": id, "error": ...})
                  if config.strict_apply:
                      raise
                  # otherwise continue with next patch
              finally:
                  span.end()
          - Summary log: "sndr.apply.complete" with counts
        
        Phase F: runtime hooks
          - Install lazy retry hooks (PN302-style late binding)
          - Install runtime patches (orchestrator hooks)
          - emit "sndr.runtime.ready"

T+N   sndr.lifecycle.ready emitted
T+N   Engine continues normal startup (vllm serve...)
```

### Recommendation: instrument every phase

```python
def init():
    with tracer.start_as_current_span("sndr.boot") as boot_span:
        with tracer.start_as_current_span("phase_a_license"):
            verify_license()
        with tracer.start_as_current_span("phase_b_engine"):
            engine = bootstrap_engine()
        # ... etc
        boot_span.set_attribute("total_duration_ms", ...)
```

Telemetry collected during boot is invaluable for debugging slow starts in
production.

## 5.2 Patch application algorithm

The decision tree for a single patch:

```
patch_request_received
    ↓
is patch enabled? (env_flag, config.patches.disable)
    ↓ yes
    is patch in valid tier? (community | engine + license)
        ↓ yes
        does applies_to match runtime profile?
            (model_arch, version_range, compute_capability, ...)
            ↓ yes
            does the engine adapter support this patch?
                ↓ yes
                resolve target file via adapter.resolve_file()
                    ↓ found
                    check anchor matches in target file
                        ↓ match
                        apply text replacement
                        verify marker written
                        emit "patch.applied"
                        return SUCCESS
                        ↓ no match
                        check upstream_drift_markers
                            ↓ upstream merged
                            emit "patch.upstream_merged"
                            return SKIPPED
                            ↓ unknown drift
                            emit "patch.anchor_drift"
                            return FAILED (or skip per policy)
                    ↓ not found
                    emit "patch.target_missing"
                    return SKIPPED
                ↓ no
                emit "patch.engine_unsupported"
                return SKIPPED
            ↓ no
            emit "patch.applies_to_mismatch"
            return SKIPPED
        ↓ no
        emit "patch.tier_blocked"
        return SKIPPED
    ↓ no
    return SKIPPED ("disabled by operator")
```

### Recommendation: never raise on patch failure by default

The default behavior should be: log structured error, continue. Only raise if
operator opted into `SNDR_STRICT_APPLY=1`. The reason: in production, one
broken patch should not prevent boot. The operator can investigate the error
and decide whether to fix or disable.

## 5.3 Pin upgrade algorithm (`sndr pin upgrade --to X`)

```
Step 1: Pre-flight validation
    - Current pin matches an existing manifest in pins/
    - Sufficient disk space for new image
    - No critical containers actively serving requests
    - sndr_engine wheel (if installed) is compatible with target sndr version

Step 2: Image acquisition
    - docker pull vllm/vllm-openai:<target>
    - Verify image signature (cosign, if configured)
    - Tag current `:nightly` → `:nightly-previous`
    - Tag new image as `:nightly` AND `:nightly-<full-sha>`

Step 3: Manifest generation
    - Boot vanilla smoke container (no sndr mount, no env flags)
    - Wait for ready
    - For each file referenced by community patches:
        docker cp container:/path → tmp/
        compute md5sum
        extract anchor regions (snippet matching)
        compute snippet md5
    - Write engines/vllm/pins/<target>/manifest.yaml
    - Write engines/vllm/pins/<target>/snapshot.md5sums
    - Stop smoke container

Step 4: Drift analysis
    - Diff new manifest against current pin manifest
    - Per-patch classification:
        STABLE: anchor unchanged
        BENIGN: md5 changed but text identical (whitespace, comments)
        DRIFT: anchor text changed, needs manual review
        BLOCKED: file removed upstream
    - Generate drift_report.md with diffs

Step 5: Apply matrix validation
    - Boot test container (sndr mount + new pin + all patches enabled)
    - Run apply_all() with strict=False
    - Collect per-patch outcome
    - Categorize: applied, skipped, failed
    - Compare against previous-pin baseline

Step 6: Bench smoke
    - Quick bench (1-2 runs) per critical model
    - Compare TPS against previous baseline
    - Acceptable: ±2% per model

Step 7: Verdict and report
    - If all STABLE + bench within budget: ELIGIBLE for auto-promote
    - Else: BLOCKED, report only
    - Generate upgrade_report.md
    - Output: docs/pin_upgrades/<timestamp>_<target>.md

Step 8: Manual promotion (separate command)
    - Operator runs: sndr pin promote --to <target>
    - Updates KNOWN_GOOD_VLLM_PINS allowlist
    - Updates docs/concepts/PINS.md compatibility table
    - Commits manifest + report
    - emit "sndr.pin.promoted"
```

### Recommendation: never auto-promote

Even when all checks pass, never promote without operator confirmation. Pin
upgrades affect production performance and correctness. A second pair of eyes
on the report catches edge cases the automation missed.

## 5.4 Drift detection algorithm (daily cron)

```
Trigger: GitHub Actions cron at 06:00 UTC daily

Step 1: Pull latest nightly
    - docker pull vllm/vllm-openai:nightly
    - Skip if image hash unchanged since last run

Step 2: Per-pin verification
    - For each pin manifest in engines/vllm/pins/:
        if pin == "latest_nightly":
            run drift_check for this snapshot
        else:
            skip (only latest is dynamic)

Step 3: Drift check per file
    - For each file in pin manifest:
        live_md5 = compute_md5(docker_cp(file))
        if live_md5 != manifest.files[file].md5:
            for anchor in manifest.files[file].anchors:
                if live_anchor_present:
                    if md5(live_anchor) != anchor.md5_of_snippet:
                        record DRIFT
                else:
                    record ANCHOR_MISSING
        else:
            record OK

Step 4: Report generation
    - drift_report.json with structured findings
    - drift_report.md with human-readable summary

Step 5: Notification
    - If any DRIFT or ANCHOR_MISSING:
        gh issue create --label "genesis-drift" --body drift_report.md
        if slack_webhook configured:
            send notification
    - Else: silent (no spam)

Step 6: Dashboard update
    - Append to docs/_internal/drift_history.json
    - GUI Drift dashboard auto-refreshes from this
```

## 5.5 License verification chain

This algorithm is documented in detail because it is security-critical.

```
License verification (called once at sndr.init()):

Input: license token (string)
    from env SNDR_ENGINE_LICENSE_KEY
    OR from file ~/.sndr/license.json
    OR from /etc/sndr/license.json

Step 1: Locate
    if env present: use env (highest priority)
    elif file exists: use file
    else: return LicenseStatus.NO_KEY

Step 2: Parse
    parts = token.split('.')
    if len(parts) != 2:
        return LicenseStatus.BAD_FORMAT
    payload_b64, signature_b64 = parts
    payload = base64url_decode(payload_b64)
    signature = base64url_decode(signature_b64)
    if any decode fails: return LicenseStatus.BAD_FORMAT

Step 3: Verify signature
    using embedded Ed25519 trust anchor public key
    if Ed25519.verify(public_key, signature, payload) raises:
        emit "sndr.license.invalid_signature"
        return LicenseStatus.BAD_SIGNATURE

Step 4: Parse payload
    data = JSON.parse(payload)
    required fields: customer_id, issued_at, expires_at, engine_major

Step 5: Check expiry
    if data.expires_at < now():
        emit "sndr.license.expired" with customer_id
        return LicenseStatus.EXPIRED

Step 6: Check version compatibility
    if data.engine_major != sndr.__version_major__:
        emit "sndr.license.version_mismatch"
        return LicenseStatus.VERSION_MISMATCH

Step 7: Check sndr_engine package
    try:
        import sndr_engine
    except ImportError:
        return LicenseStatus.NO_PACKAGE

Step 8: All checks passed
    cache LicenseStatus.LICENSED with customer_id
    emit "sndr.license.verified"
    return LicenseStatus.LICENSED
```

### Recommendation: never log token contents

```python
# BAD
log.debug(f"License token: {token}")

# GOOD
log.info("license.token.loaded", extra={
    "source": "env" | "file",
    "length": len(token),
    "customer_id_hash": sha256(customer_id)[:8],  # for correlation, not identity
})
```

The token is bearer credentials. Logging it is equivalent to logging passwords.

---

# Part 6: GUI Architecture (Carbon Design System)

## 6.1 Why Carbon?

Carbon is IBM's open-source design system. We chose it because:

1. **Enterprise credibility**: IBM, Microsoft, Google internal tools use it. It
   signals "professional" to enterprise buyers.
2. **Dark-first**: Carbon's g100 theme is built for productivity (data tables,
   monitoring dashboards). Our use case is operator monitoring — Carbon is a
   perfect fit.
3. **Comprehensive components**: 60+ React components, 4000+ icons. We rarely
   need to build from scratch.
4. **Accessibility**: WCAG 2.1 AA compliant out of the box.
5. **Active maintenance**: IBM funds it. Long-term stability.
6. **TypeScript-first**: full types, no `@types/` packages needed.
7. **CSS tokens**: every color, spacing, and typography decision is a token.
   Easy to override for brand without touching component code.

Alternatives considered and rejected:
- **Material UI**: too "consumer", less productivity-focused
- **Ant Design**: large bundle, less accessibility
- **Chakra**: less mature for enterprise
- **Custom design system**: huge ongoing cost

## 6.2 Theme configuration

We use the **g100 theme** (Carbon's darkest, highest-contrast theme) as the
base. We layer brand colors on top.

```scss
// gui/web/src/theme/styles.scss
@use '@carbon/react/scss/themes' as themes;
@use '@carbon/react/scss/theme' as theme;

// Brand overlay on Carbon g100
$sndr-overlay: (
  // Override Carbon primary blue with sndr brand accent (kept blue-ish for compat)
  background-brand: #0f62fe,
  link-primary: #4589ff,
  link-secondary: #78a9ff,
  
  // SNDR signature accents
  support-success: #24a148,        // patch applied
  support-warning: #f1c21b,        // drift detected
  support-error: #fa4d56,          // patch failed
  support-info: #4589ff,           // informational
  support-caution-major: #ff832b,  // critical config
);

:root {
  @include theme.theme(themes.$g100, $sndr-overlay);
}
```

## 6.3 Typography

We use IBM Plex (Carbon's default) because:
- Designed for screens with high information density
- Excellent Cyrillic support (essential for Russian)
- Monospace variant for code/anchors/SHA display
- Free, open-source

```scss
// gui/web/src/theme/typography.scss
@import '@ibm/plex/css/ibm-plex.css';

:root {
  --font-family-sans: 'IBM Plex Sans', system-ui, sans-serif;
  --font-family-mono: 'IBM Plex Mono', 'SF Mono', Consolas, monospace;
}
```

## 6.4 Layout grid (Carbon 16-column responsive)

```tsx
// gui/web/src/App.tsx
import { Theme, Grid, Column } from '@carbon/react';
import { UIShell } from './layouts/UIShell';

export function App() {
  return (
    <Theme theme="g100">
      <UIShell>
        <Grid fullWidth>
          <Column lg={16} md={8} sm={4}>
            <RouterOutlet />
          </Column>
        </Grid>
      </UIShell>
    </Theme>
  );
}
```

The 16-column grid scales:
- **sm** (≤671px, mobile): 4 columns
- **md** (672–1055px, tablet): 8 columns
- **lg** (1056–1311px, desktop): 16 columns
- **xlg** (1312–1583px, wide): 16 columns
- **max** (≥1584px): 16 columns

Power users sit at 1080p+ displays, so the 16-column wide grid is the primary
target.

## 6.5 Information density modes

Carbon supports comfortable (default) and compact density. We expose both:

```tsx
// gui/web/src/stores/ui.ts
import { create } from 'zustand';

interface UIStore {
  density: 'comfortable' | 'compact';
  setDensity: (d: 'comfortable' | 'compact') => void;
}

export const useUIStore = create<UIStore>((set) => ({
  density: 'comfortable',
  setDensity: (density) => set({ density }),
}));

// Used in App.tsx
const density = useUIStore((s) => s.density);
<div className={density === 'compact' ? 'cds--compact' : ''}>
  {children}
</div>
```

Compact mode reduces row heights, padding, and font scaling. Power users with
many patches/containers benefit from compact.

## 6.6 Component mapping

| sndr feature | Carbon component | Notes |
|---|---|---|
| Patches inventory table | `DataTable` + `TableToolbar` + `TableBatchActions` | Sortable, filterable, searchable, bulk actions |
| Engine + pin selector | `Dropdown` (paired) | Top bar |
| Drift status alert | `InlineNotification` (kind=warning) | Persistent until acknowledged |
| Bench results timeline | `@carbon/charts-react` `LineChart` | Time series |
| Container cards grid | `ClickableTile` | Grid layout with `Tile` |
| Confirmation dialogs | `ComposedModal` + `ModalHeader` + `ModalBody` + `ModalFooter` | Standard pattern |
| Forms | `Form` + `TextInput` + `Select` + `Checkbox` + `ToggleSmall` | All Carbon |
| Toast notifications | `ToastNotification` (via `<ActionableNotification>` for events with action) | Auto-dismiss + actionable |
| Loading skeletons | `DataTableSkeleton`, `SkeletonText`, `SkeletonPlaceholder` | One per layout |
| Status badges (tier, lifecycle) | `Tag` (type=blue/green/red/...) | Color encodes meaning |
| Top navigation | `Header` + `HeaderName` + `HeaderGlobalBar` + `HeaderGlobalAction` | UI Shell |
| Side navigation | `SideNav` + `SideNavItems` + `SideNavLink` | Persistent rail |
| Code snippets (anchors) | `CodeSnippet` (type="multi") | Syntax highlight via Prism |
| Empty states | Custom following Carbon Empty States spec | Illustration + message + CTA |
| Date/time pickers | `DatePicker` | For bench filter ranges |
| Search inputs | `Search` (size="lg") | Top of inventory tables |
| Pagination | `Pagination` | Below tables |
| Tabs | `Tabs` + `TabList` + `Tab` + `TabPanels` + `TabPanel` | Patch detail views |
| Tooltips | `Tooltip` | Help text on icons |
| Breadcrumbs | `Breadcrumb` + `BreadcrumbItem` | Navigation context |
| Progress | `ProgressBar`, `ProgressIndicator` | Long-running ops |

## 6.7 i18n with Lingui

We use Lingui (not react-intl, not i18next) because:
- Smaller bundle (~20kb vs 60kb)
- ICU MessageFormat support (plurals, gender)
- TypeScript-first
- `.po` file format compatible with Crowdin/Phrase/Lokalise

```tsx
// gui/web/src/features/patches/PatchesView.tsx
import { Trans, t } from '@lingui/macro';
import { i18n } from '@lingui/core';

export function PatchesView() {
  return (
    <div>
      <h2><Trans>Available patches</Trans></h2>
      <p>
        <Trans>
          {patches.length} patches registered, {applied} applied
        </Trans>
      </p>
      <DataTable
        headers={[
          { key: 'id', header: i18n._(t`ID`) },
          { key: 'tier', header: i18n._(t`Tier`) },
          { key: 'lifecycle', header: i18n._(t`Lifecycle`) },
          { key: 'family', header: i18n._(t`Family`) },
        ]}
      />
    </div>
  );
}
```

```po
# gui/web/src/i18n/locales/ru/messages.po
msgid "Available patches"
msgstr "Доступные патчи"

msgid "{0} patches registered, {1} applied"
msgstr "{0} патчей зарегистрировано, {1} применено"

msgid "ID"
msgstr "ID"

msgid "Tier"
msgstr "Уровень"

msgid "Lifecycle"
msgstr "Жизненный цикл"

msgid "Family"
msgstr "Семейство"
```

## 6.8 State management strategy

We use **Zustand** (not Redux, not Context) for global state.

Reasons:
- ~3kb gzipped
- No boilerplate (no reducers, actions, dispatch)
- TypeScript-first with strong inference
- Devtools support
- No `Provider` wrapper needed

```typescript
// gui/web/src/stores/engine.ts
import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface EngineStore {
  selected: 'vllm' | 'sglang';
  pin: string | null;
  setEngine: (engine: 'vllm' | 'sglang') => void;
  setPin: (pin: string) => void;
}

export const useEngineStore = create<EngineStore>()(
  devtools(
    (set) => ({
      selected: 'vllm',
      pin: null,
      setEngine: (engine) => set({ selected: engine, pin: null }),
      setPin: (pin) => set({ pin }),
    }),
    { name: 'engine-store' }
  )
);
```

For server state (data from API), we use **TanStack Query** (formerly React
Query):

```typescript
// gui/web/src/features/patches/usePatches.ts
import { useQuery } from '@tanstack/react-query';
import { useEngineStore } from '@/stores/engine';
import { patchesApi } from './api';

export function usePatches() {
  const engine = useEngineStore((s) => s.selected);
  const pin = useEngineStore((s) => s.pin);
  
  return useQuery({
    queryKey: ['patches', engine, pin],
    queryFn: () => patchesApi.list({ engine, pin }),
    staleTime: 30_000,           // 30 seconds before refetch
    refetchOnWindowFocus: true,   // refetch when user returns to tab
  });
}
```

## 6.9 Routing

We use hash routing (not React Router, not Next router) to keep things simple:

```typescript
// gui/web/src/router.ts
export type Route =
  | { name: 'overview' }
  | { name: 'patches'; filter?: string }
  | { name: 'pins' }
  | { name: 'drift' }
  | { name: 'fleet' }
  | { name: 'hosts'; hostId?: string }
  | { name: 'containers'; name?: string }
  | { name: 'chat' }
  | { name: 'bench' }
  | { name: 'doctor' }
  | { name: 'settings' };

export function parseHash(hash: string): Route {
  const [name, search] = hash.slice(1).split('?');
  const params = new URLSearchParams(search);
  // ...
}
```

Why hash routing:
- Works without server-side route configuration
- Easy to bookmark
- No history API issues with browser back button on legacy pages
- Simple enough that we don't need a full router library

## 6.10 Recommendation: GUI refactor strategy

The current `App.tsx` is 11,633 lines. We do not rewrite it in one go. Strategy:

1. **Phase 12a**: Adopt Carbon Theme + IBM Plex fonts; existing components stay
   but visual style starts to match
2. **Phase 12b**: Extract the `Patches` feature module (already has clear
   subcomponents inside App.tsx)
3. **Phase 12c**: Extract `Engines`, `Pins`, `Drift` (new features)
4. **Phase 12d**: Extract `Fleet`, `Hosts`, `Containers`
5. **Phase 12e**: Extract `Chat`, `Bench`, `Doctor`
6. **Phase 12f**: Extract `Auth`, `Licensing`, `Settings`
7. **Phase 12g**: App.tsx becomes a thin shell (~200 LOC)

Each extraction is reviewed independently. If 12c reveals a flaw in 12a's
approach, only 12c is reworked, not the whole GUI.

---

# Part 7: API Design Rules

## 7.1 URL convention

```
/api/v{N}/{resource}                          # List
/api/v{N}/{resource}/{id}                     # Detail
/api/v{N}/{resource}/{id}/{subresource}       # Nested
/api/v{N}/{resource}/{id}/actions/{action}    # Action (non-CRUD)

Examples:
GET    /api/v1/engines
GET    /api/v1/engines/vllm
GET    /api/v1/engines/vllm/pins
GET    /api/v1/engines/vllm/pins/0.22.1
POST   /api/v1/engines/vllm/pins/0.22.1/actions/promote
GET    /api/v1/engines/vllm/patches
GET    /api/v1/engines/vllm/patches/PN286
POST   /api/v1/engines/vllm/patches/PN286/actions/override
GET    /api/v1/engines/vllm/drift
GET    /api/v1/engines/vllm/containers
GET    /api/v1/engines/vllm/containers/qwen3-27b/logs
GET    /api/v1/engines/vllm/containers/qwen3-27b/stats
POST   /api/v1/engines/vllm/containers/qwen3-27b/actions/restart
POST   /api/v1/engines/vllm/containers/qwen3-27b/actions/exec
GET    /api/v1/licensing/status
POST   /api/v1/licensing/upload
GET    /api/v1/fleet
GET    /api/v1/hosts
GET    /api/v1/health
GET    /api/v1/version
```

Streaming:
```
GET    /api/v1/events                         # SSE event stream
WS     /api/v1/terminal/{host_id}             # WebSocket PTY
```

## 7.2 Response envelope

Every successful response has this shape:

```json
{
  "data": { ... },
  "meta": {
    "api_version": "v1",
    "request_id": "abc123def456",
    "engine": "vllm",
    "pin": "0.22.1",
    "timestamp": "2026-06-05T14:30:00Z"
  }
}
```

Listing responses include pagination metadata:

```json
{
  "data": [ ... ],
  "meta": {
    "api_version": "v1",
    "request_id": "...",
    "page": { "number": 1, "size": 50, "total_items": 252, "total_pages": 6 }
  }
}
```

## 7.3 Error response (RFC 7807 Problem Details)

```json
{
  "type": "https://docs.sndr-platform.io/errors/drift-detected",
  "title": "Drift detected on pin upgrade",
  "status": 409,
  "detail": "Patch P3 anchor changed in upstream 0.22.2",
  "instance": "/api/v1/engines/vllm/pins/upgrade",
  "extensions": {
    "patch_id": "P3",
    "previous_pin": "0.22.1",
    "target_pin": "0.22.2",
    "anchor_file": "v1/attention/ops/triton_turboquant_store.py",
    "expected_md5": "7f23a48c92cd414a",
    "actual_md5": "9b34c5e6f7a89102"
  }
}
```

## 7.4 Pagination, filtering, sorting (consistent)

```
GET /api/v1/engines/vllm/patches?
  filter[tier]=community&
  filter[family]=attention&
  filter[lifecycle]=stable&
  sort=-id&                              # - prefix = descending
  page[size]=50&
  page[number]=1
```

Multiple values for same filter: comma-separated
```
filter[tier]=community,engine
```

## 7.5 OpenAPI auto-generation

FastAPI auto-generates `/openapi.json`. We add a build step:

```bash
# gui/web/package.json scripts
"gen:api": "openapi-typescript http://localhost:8800/openapi.json --output src/api/schema.gen.ts"
```

This generates TypeScript types from the Pydantic schemas:

```typescript
// gui/web/src/api/schema.gen.ts (auto-generated, DO NOT EDIT)
export interface Patch {
  id: string;
  tier: 'community' | 'engine';
  lifecycle: 'experimental' | 'stable' | 'retired';
  family: string;
  title: string;
  applies_to?: Record<string, unknown>;
}
```

CI check: regenerate and diff. If diff is non-empty, the contract has drifted.

## 7.6 Backward compatibility window

When we change an API endpoint:
1. Introduce v2 alongside v1
2. v1 returns deprecation header for one release
3. v2 becomes default
4. v1 removed in next major

```
HTTP/1.1 200 OK
Sunset: Sat, 31 Dec 2026 23:59:59 GMT
Deprecation: true
Link: </api/v2/patches>; rel="successor-version"
```

---

# Part 8: Error Handling Strategy

## 8.1 Exception hierarchy

```python
# sndr/exceptions.py

class SndrError(Exception):
    """Base for all sndr-platform errors."""
    code: str = "sndr.error"
    http_status: int = 500
    
    def __init__(self, message: str = "", **context):
        super().__init__(message)
        self.message = message
        self.context = context  # structured for logging + RFC 7807

# Configuration
class ConfigError(SndrError):
    code = "sndr.config.invalid"
    http_status = 400

# License
class LicenseError(SndrError):
    code = "sndr.license.error"
    http_status = 402

class LicenseExpiredError(LicenseError):
    code = "sndr.license.expired"

class LicenseVersionMismatchError(LicenseError):
    code = "sndr.license.version_mismatch"

class LicenseBadSignatureError(LicenseError):
    code = "sndr.license.bad_signature"

# Engine
class EngineError(SndrError):
    code = "sndr.engine.error"
    http_status = 500

class EngineNotInstalledError(EngineError):
    code = "sndr.engine.not_installed"
    http_status = 503

class EngineUnsupportedError(EngineError):
    code = "sndr.engine.unsupported"
    http_status = 400

# Patches
class PatchError(SndrError):
    code = "sndr.patch.error"

class PatchAnchorDriftError(PatchError):
    code = "sndr.patch.anchor_drift"
    http_status = 409

class PatchTargetMissingError(PatchError):
    code = "sndr.patch.target_missing"
    http_status = 404

class PatchApplyFailedError(PatchError):
    code = "sndr.patch.apply_failed"
    http_status = 500

# Pins
class PinError(SndrError):
    code = "sndr.pin.error"

class PinNotSupportedError(PinError):
    code = "sndr.pin.not_supported"
    http_status = 400

# Drift
class DriftDetectedError(SndrError):
    code = "sndr.drift.detected"
    http_status = 409

# Auth
class AuthError(SndrError):
    code = "sndr.auth.error"
    http_status = 401

class AuthForbiddenError(AuthError):
    code = "sndr.auth.forbidden"
    http_status = 403
```

## 8.2 Error propagation chain

```
Layer 0 (kernel) raises typed exception
    ↓
Layer 1 (engine adapter) catches, may wrap with engine-specific context
    ↓
Layer 2 (dispatcher) logs event, decides retry/abort per policy
    ↓
Layer 3 (apply) decides patch-level continuation
    ↓
Layer 4 (HTTP API) maps to RFC 7807 Problem Details + sets headers
    ↓
Layer 5 (GUI) renders Carbon InlineNotification with action button
```

## 8.3 Logging on errors

Every exception MUST be logged with structured fields:

```python
try:
    apply_patch(patch)
except PatchApplyFailedError as e:
    log.error(
        "patch.apply_failed",  # event name (machine-parseable)
        extra={
            "patch_id": patch.id,
            "engine": engine.name,
            "pin": engine.current_pin,
            "exception_type": "PatchApplyFailedError",
            "exception_code": e.code,
            "trace_id": current_trace_id(),
            **e.context,
        },
        exc_info=True,  # full stack trace
    )
    metrics.patches_failed.labels(
        engine=engine.name,
        patch_id=patch.id,
    ).inc()
    raise  # propagate up
```

## 8.4 Recommendation: never swallow exceptions silently

```python
# BAD: silent failure
try:
    do_something()
except Exception:
    pass  # we will never know this happened

# BAD: silent failure with comment
try:
    do_something()
except Exception:
    pass  # ignore, not critical

# GOOD: log and continue
try:
    do_something()
except Exception as e:
    log.warning(
        "operation.failed_recoverable",
        extra={"operation": "do_something", "error": str(e)},
    )
    # continue with fallback
    return fallback_value()

# GOOD: log and reraise
try:
    do_something()
except Exception as e:
    log.error("operation.failed", exc_info=True)
    raise
```

---

# Part 9: Observability Standards

## 9.1 Structured logging (JSON, single line per event)

```python
# sndr/observability/logging.py
import logging
import json
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger

trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)

class SndrJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['service'] = 'sndr'
        log_record['version'] = __version__
        log_record['logger'] = record.name
        log_record['level'] = record.levelname
        
        # Inject trace context
        if (tid := trace_id_var.get()) is not None:
            log_record['trace_id'] = tid
        
        # Map "event" from logger.info(event, extra={...})
        if not isinstance(log_record.get('message'), dict):
            log_record['event'] = log_record.pop('message', 'unknown')

def configure_logging(level: str = "INFO"):
    handler = logging.StreamHandler()
    handler.setFormatter(SndrJsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])
```

Every log line looks like:

```json
{
  "ts": "2026-06-05T14:30:00.123Z",
  "level": "INFO",
  "service": "sndr",
  "version": "12.0.0",
  "logger": "sndr.dispatcher",
  "event": "patch.applied",
  "trace_id": "abc123def456",
  "patch_id": "PN286",
  "engine": "vllm",
  "pin": "0.22.1",
  "duration_ms": 4.2
}
```

## 9.2 Event naming convention

Format: `{noun}.{verb}` in past tense (for things that happened) or present
imperative (for requests).

Good event names:
- `patch.applied`
- `patch.failed`
- `patch.skipped`
- `pin.upgraded`
- `pin.promoted`
- `drift.detected`
- `license.verified`
- `license.expired`
- `engine.ready`
- `boot.complete`

Bad event names:
- `applying_patch` (gerund, not declarative)
- `PatchApplied` (camelCase, not snake)
- `patch applied` (space, not dot)
- `success` (verb without subject)

## 9.3 Metrics (Prometheus format)

```python
# sndr/observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Counters: monotonically increasing
patches_applied = Counter(
    'sndr_patches_applied_total',
    'Total patches successfully applied',
    ['engine', 'pin', 'tier', 'patch_id'],
)

patches_failed = Counter(
    'sndr_patches_failed_total',
    'Total patches that failed to apply',
    ['engine', 'pin', 'tier', 'patch_id', 'error_code'],
)

# Histograms: distributions
patch_apply_duration = Histogram(
    'sndr_patch_apply_duration_seconds',
    'Time spent applying a single patch',
    ['engine', 'patch_id'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

api_request_duration = Histogram(
    'sndr_api_request_duration_seconds',
    'HTTP request duration',
    ['method', 'route', 'status'],
)

# Gauges: current state
license_valid = Gauge(
    'sndr_license_valid',
    '1 if license is valid, 0 otherwise',
)

active_containers = Gauge(
    'sndr_active_containers',
    'Number of managed containers running',
    ['engine'],
)

# Info: build-time metadata
build_info = Info(
    'sndr_build',
    'Build information',
)
build_info.info({
    'version': '12.0.0',
    'commit': 'abc123',
    'python_version': '3.12',
})
```

Exposed at `/api/v1/metrics` in Prometheus scrape format.

## 9.4 Distributed tracing (OpenTelemetry)

```python
# sndr/observability/trace.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def configure_tracing(otlp_endpoint: str | None = None):
    provider = TracerProvider()
    if otlp_endpoint:
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        )
    trace.set_tracer_provider(provider)

# Usage
tracer = trace.get_tracer("sndr.dispatcher")

with tracer.start_as_current_span("apply_patch") as span:
    span.set_attribute("patch.id", patch_id)
    span.set_attribute("engine", current_engine())
    try:
        result = patch.apply()
        span.set_attribute("outcome", result.status)
    except Exception as e:
        span.set_attribute("outcome", "failed")
        span.record_exception(e)
        raise
```

## 9.5 SLI/SLO targets

| SLI | Target SLO | Measurement |
|---|---|---|
| Boot duration P95 | < 5 sec | `sndr_boot_duration_seconds` |
| Patch apply success rate | > 99% (excl. skip-by-design) | applied / (applied + failed) |
| Drift detection latency | < 24h after upstream change | drift report timestamp |
| GUI page load P95 (cold) | < 2 sec | browser performance API |
| GUI page load P95 (warm) | < 500 ms | browser performance API |
| API response P95 | < 200 ms (non-streaming) | `sndr_api_request_duration_seconds` |
| License verification | < 100 ms | per-call timer |
| Bench reproducibility CV | < 8% sustained | bench tool output |

---

# Part 10: Security Model

## 10.1 Threat model

| Threat | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Unauthorized engine-tier patch usage | Medium | High (revenue loss) | Ed25519 signed tokens + expiry + version match |
| License token leakage | Low | High | Never logged, never in metrics, hashed in audit |
| Malicious patch injection | Low | Critical | Only registered patches loaded; entry_points from trusted packages only |
| Container escape via patches | Very low | Critical | Patches modify Python files only; no syscall surface |
| GUI XSS | Medium | High | React auto-escape; CSP headers; no innerHTML/dangerouslySetInnerHTML |
| API auth bypass | Low | Critical | All operations behind auth middleware; integration-tested |
| Multi-tenant data leak | Medium | High | Per-host secrets store, encrypted at rest, scoped by host_id |
| Container exec abuse | High (if enabled) | High | SNDR_ENABLE_EXEC gate (default off) + audit log + RBAC |
| Pin downgrade attack | Low | Medium | KNOWN_GOOD_VLLM_PINS allowlist + manual promotion |
| Supply chain compromise | Medium | Critical | pip-audit + CodeQL + dependency review on every PR |
| Brute-force auth | Medium | Medium | Per-IP rate limit + account lockout after N failures |

## 10.2 Authentication flow

```
Browser              GUI                    API                    Auth Backend
  │                   │                      │                         │
  │ ── HTTPS GET /    ─────────────────────→ │                         │
  │ ←── 200 (login form) ─────────────────── │                         │
  │ ── POST /api/v1/auth/login (user/pass) → │                         │
  │                                          ├── scrypt verify ───────→│
  │                                          │←── verified ──────────── │
  │                                          ├── issue session token ─→│
  │                                          │←── token ─────────────── │
  │ ←── Set-Cookie: session=... ──────────── │                         │
  │ ── GET /api/v1/patches (cookie) ───────→ │                         │
  │                                          ├── verify session ──────→│
  │                                          ├── check authz ─────────→│
  │                                          ├── execute query ────────│
  │ ←── data ──────────────────────────────── │                         │
```

## 10.3 Authorization model (RBAC)

Three roles:
- **viewer**: read-only access to all resources
- **operator**: viewer + execute actions (start/stop containers, override patches)
- **admin**: operator + manage users, licenses, sensitive config

Per-route authorization decorator:

```python
@router.post("/patches/{patch_id}/actions/override")
@requires_role("operator")
async def override_patch(patch_id: str, ...):
    ...

@router.delete("/users/{user_id}")
@requires_role("admin")
async def delete_user(user_id: str, ...):
    ...
```

## 10.4 Audit log (immutable, append-only)

Every sensitive action produces an audit entry:

```json
{
  "ts": "2026-06-05T14:30:00Z",
  "event": "patch.override",
  "actor": {
    "type": "user",
    "id": "user:sander",
    "role": "operator"
  },
  "source": {
    "ip": "192.168.1.5",
    "user_agent": "Mozilla/5.0 ...",
    "session_id": "sess_abc123"
  },
  "target": {
    "resource": "patch",
    "patch_id": "PN286",
    "engine": "vllm"
  },
  "action": "enable",
  "outcome": "success",
  "trace_id": "trace_xyz789"
}
```

Audit logs:
- Written to dedicated file `$SNDR_HOME/audit/audit-YYYY-MM-DD.jsonl`
- Rotated daily, archived 1 year minimum
- NEVER overwritten — append-only
- GUI exposure for admin role only
- Optionally forwarded to SIEM (configurable webhook)

## 10.5 Data protection

| Data class | Protection |
|---|---|
| License tokens | Never logged, never in metrics |
| Customer IDs | Hashed (SHA-256, first 8 chars) in logs |
| SSH credentials | Encrypted at rest in secrets store (Fernet, key from env) |
| API tokens | Rotatable, configurable scope, hashed in DB |
| Session cookies | httpOnly + Secure + SameSite=strict |
| Passwords | scrypt hashed (Carbon's TextInput type="password" disables autofill in dev) |
| 2FA secrets | Encrypted at rest |

## 10.6 Recommendation: implement security headers

```python
# sndr/product_api/middleware.py
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "  # Carbon needs inline styles
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none';"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response
```

---

# Part 11: Performance Budgets

## 11.1 Hard limits (enforced via CI)

```yaml
# tools/ci/performance_budget.yaml
python:
  boot_duration_ms: 5000        # sndr.init() to "ready"
  patch_apply_p95_ms: 50        # single patch apply
  api_response_p95_ms: 200      # non-streaming endpoints
  api_response_p99_ms: 500
  memory_overhead_mb: 100       # sndr's overhead above engine baseline

gui:
  bundle_size_kb_gzipped: 350   # main bundle
  bundle_chunks_max: 12         # code splitting
  first_contentful_paint_ms: 1500
  largest_contentful_paint_ms: 2500
  cumulative_layout_shift: 0.1
  time_to_interactive_ms: 3500

inference (bench regression):
  35b_fp8_tps_min: 207          # vs current baseline (Genesis stack)
  27b_int4_tps_min: 115
  regression_threshold_pct: 2
```

CI fails if any budget exceeded by > 5% without justification ADR.

## 11.2 Bundle analysis (GUI)

```bash
# After build, analyze
npm run build
npm run analyze  # opens visualizer

# CI assertion
node tools/ci/check_bundle_size.js
```

## 11.3 Recommendation: lazy load features

```typescript
// gui/web/src/router.ts
const PatchesView = lazy(() => import('@/features/patches/PatchesView'));
const FleetView = lazy(() => import('@/features/fleet/FleetView'));
// ...
```

Each feature module loaded only when its route is visited. Reduces initial
bundle. Critical for our 24-route GUI.

---

# Part 12: Testing Strategy

## 12.1 Testing pyramid

```
              ┌──────────────┐
              │   E2E (10%)  │   Playwright: 4 critical user paths
              ├──────────────┤
              │ Integration  │
              │    (30%)     │   pytest: pin matrix, apply chains, API
              ├──────────────┤
              │              │
              │  Unit (60%)  │   Vitest + pytest, isolated modules
              │              │
              └──────────────┘
```

## 12.2 Coverage requirements

| Code area | Coverage min | Test type |
|---|---|---|
| `sndr/kernel/` | 90% | unit |
| `sndr/dispatcher/` | 90% | unit |
| `sndr/engines/*/adapter.py` | 85% | unit |
| `sndr/engines/*/detection/` | 80% | unit |
| `sndr/engines/*/patches/` | 70% functional | integration |
| `sndr/product_api/routes/` | 85% | unit + integration |
| `sndr/product_api/domain/` | 90% | unit |
| `sndr/license.py` | 95% | unit (security-critical) |
| `gui/web/src/features/` | 70% | unit + e2e |
| `gui/web/src/components/` | 80% | unit |
| `gui/web/src/api/` | 75% | unit |

## 12.3 Test types and patterns

### Unit test (isolated, fast)

```python
# tests/unit/dispatcher/test_decision.py
def test_check_applies_to_model_arch_wildcard():
    profile = {"architectures": ["UnknownArchType"]}
    meta = {"applies_to": {"model_arch": ["KnownArch", "*"]}}
    ok, reason = check_applies_to(profile, meta)
    assert ok
    assert "wildcard" in reason.lower()

def test_check_applies_to_model_arch_exact_match():
    profile = {"architectures": ["Qwen3_5MoeForConditionalGeneration"]}
    meta = {"applies_to": {"model_arch": ["Qwen3_5MoeForConditionalGeneration"]}}
    ok, _ = check_applies_to(profile, meta)
    assert ok

def test_check_applies_to_model_arch_no_match():
    profile = {"architectures": ["UnknownArch"]}
    meta = {"applies_to": {"model_arch": ["KnownArch"]}}
    ok, reason = check_applies_to(profile, meta)
    assert not ok
    assert "model_arch" in reason
```

### Integration test (pin matrix)

```python
# tests/integration/pin_matrix/vllm/test_apply_matrix.py
@pytest.mark.gpu
@pytest.mark.parametrize("pin", ["0.21.1.dev354", "0.22.0", "0.22.1.dev195"])
def test_all_community_patches_apply_on_pin(pin):
    """Apply all community patches on each supported pin; expect no failures."""
    container = boot_smoke_container(pin)
    try:
        engine = VllmEngine(config=SndrConfig(engine="vllm", engine_pin=pin))
        engine.bootstrap()
        results = engine.apply_all(strict=False)
        
        failures = [r for r in results if r.status == "failed"]
        assert len(failures) == 0, f"Failures: {failures}"
        
        applied = [r for r in results if r.status == "applied"]
        assert len(applied) > 100, "Suspiciously few patches applied"
    finally:
        container.stop()
```

### E2E test (Playwright)

```typescript
// gui/web/e2e/pin_upgrade.spec.ts
import { test, expect } from '@playwright/test';

test('operator can initiate pin upgrade', async ({ page }) => {
  await page.goto('/');
  await page.fill('[data-testid=login-username]', 'operator');
  await page.fill('[data-testid=login-password]', 'test-pass');
  await page.click('[data-testid=login-submit]');
  
  await page.click('text=Pins');
  await page.click('text=Upgrade');
  await page.selectOption('[data-testid=target-pin]', '0.22.2');
  await page.click('[data-testid=confirm-upgrade]');
  
  await expect(page.locator('[data-testid=upgrade-progress]')).toBeVisible();
  await expect(page.locator('text=Image acquired')).toBeVisible({ timeout: 60_000 });
});
```

## 12.4 Test fixtures

```
tests/
├── fixtures/
│   ├── pristine_vllm/              # Pre-extracted vllm files for offline tests
│   │   ├── 0.21.1.dev354/
│   │   ├── 0.22.0/
│   │   └── 0.22.1.dev195/
│   ├── sample_manifests/
│   │   └── valid.yaml
│   ├── sample_configs/
│   │   ├── community_only.yaml
│   │   └── with_license.yaml
│   ├── license_tokens/
│   │   ├── valid_token.txt
│   │   ├── expired_token.txt
│   │   └── invalid_signature_token.txt
│   └── responses/                  # Mocked API responses for unit tests
│       └── ...
└── ...
```

## 12.5 Recommendation: snapshot testing for manifests

```python
# tests/integration/pin_matrix/test_manifest_stability.py
def test_manifest_for_pin_matches_snapshot(pin, snapshot):
    """Regenerate manifest and compare against committed snapshot."""
    manifest = generate_manifest(pin)
    snapshot.assert_match(yaml.dump(manifest), f"{pin}.yaml")
```

If a manifest changes unintentionally, the test fails with a diff. Operator
reviews and either commits the new snapshot (intentional) or fixes the bug.

---

# Part 13: CI/CD Pipeline

## 13.1 Main CI workflow

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, refactor/**]
  pull_request:

jobs:
  python-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.10' }
      - run: pip install -e .[dev]
      - run: ruff check sndr/ tools/
      - run: mypy sndr/
      - run: bandit -r sndr/ -c .bandit.yaml

  layer-rules:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python tools/ci/verify_layer_imports.py

  no-engine-leak:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python tools/ci/verify_no_engine_leak.py

  python-tests:
    needs: [python-lint, layer-rules]
    strategy:
      matrix:
        python: ['3.10', '3.11', '3.12']
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '${{ matrix.python }}' }
      - run: pip install -e .[dev]
      - run: pytest tests/unit --cov=sndr --cov-report=xml
      - run: pytest tests/integration -m "not slow and not gpu"
      - uses: codecov/codecov-action@v4
        with: { files: coverage.xml }

  gui-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: cd gui/web && npm ci
      - run: cd gui/web && npm run lint
      - run: cd gui/web && npm run typecheck

  gui-tests:
    needs: gui-lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: cd gui/web && npm ci
      - run: cd gui/web && npm run test:unit -- --coverage

  gui-e2e:
    needs: gui-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: cd gui/web && npm ci
      - run: cd gui/web && npx playwright install --with-deps
      - run: cd gui/web && npm run test:e2e

  gui-bundle-size:
    needs: gui-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: cd gui/web && npm ci
      - run: cd gui/web && npm run build
      - run: node tools/ci/check_bundle_size.js

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install pip-audit
      - run: pip-audit
      - uses: github/codeql-action/analyze@v3
        with: { languages: 'python,javascript' }
```

## 13.2 Pin matrix workflow

```yaml
# .github/workflows/pin_matrix.yml
name: Pin Matrix Tests

on:
  pull_request:
    paths:
      - 'sndr/engines/vllm/patches/**'
      - 'sndr/engines/vllm/pins/**'
      - 'sndr/dispatcher/**'

jobs:
  smoke:
    strategy:
      fail-fast: false
      matrix:
        pin: ['0.21.1.dev354', '0.22.0', '0.22.1']
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: |
          docker run --rm --gpus all \
            -v $PWD/sndr:/usr/local/lib/python3.12/dist-packages/sndr:ro \
            -e SNDR_ENGINE=vllm \
            -e SNDR_ENGINE_PIN=${{ matrix.pin }} \
            vllm/vllm-openai:nightly-${{ matrix.pin }} \
            python3 -c "from sndr import init; init()"
```

## 13.3 Daily drift detection

```yaml
# .github/workflows/drift_check.yml
name: Daily Drift Detection

on:
  schedule:
    - cron: '0 6 * * *'
  workflow_dispatch:

jobs:
  detect:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install -e .
      - run: sndr drift check --engine vllm --all-pins --output drift.json
      - id: check
        run: |
          if [ -s drift.json ]; then
            echo "drift_found=true" >> $GITHUB_OUTPUT
          fi
      - if: steps.check.outputs.drift_found == 'true'
        run: |
          gh issue create \
            --label "genesis-drift" \
            --title "Drift detected $(date +%F)" \
            --body-file drift_report.md
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## 13.4 Release workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  build-python:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install build
      - run: python -m build
      - uses: actions/upload-artifact@v4
        with: { name: python-wheel, path: dist/ }

  build-gui:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: cd gui/web && npm ci && npm run build
      - uses: actions/upload-artifact@v4
        with: { name: gui-dist, path: gui/web/dist }

  bundle:
    needs: [build-python, build-gui]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with: { name: python-wheel, path: dist/ }
      - uses: actions/download-artifact@v4
        with: { name: gui-dist, path: gui-dist/ }
      - run: python tools/bundle_gui_into_wheel.py dist/ gui-dist/

  publish-pypi:
    needs: bundle
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/download-artifact@v4
      - uses: pypa/gh-action-pypi-publish@release/v1

  publish-docker:
    needs: bundle
    runs-on: ubuntu-latest
    steps:
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: |
            sndr/control-center:${{ github.ref_name }}
            sndr/control-center:latest

  github-release:
    needs: [publish-pypi, publish-docker]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: gh release create ${{ github.ref_name }} --notes-file docs/changelog/${{ github.ref_name }}.md
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

# Part 14: Documentation Framework

## 14.1 Diátaxis taxonomy

We use the Diátaxis framework for documentation:

```
docs/
├── tutorials/                  # Learning-oriented (lead by the hand)
│   ├── getting_started.md
│   ├── your_first_patch.md
│   └── your_first_pin_upgrade.md
├── how_to/                     # Problem-oriented (cookbook)
│   ├── debug_anchor_drift.md
│   ├── add_new_engine.md
│   ├── customize_gui_theme.md
│   └── enable_2fa.md
├── reference/                  # Information-oriented (lookup)
│   ├── API.md                  # REST endpoints
│   ├── CLI.md                  # Command reference
│   ├── CONFIG.md               # Config file format
│   ├── ENV_VARS.md             # All environment variables
│   ├── ERRORS.md               # Error codes catalog
│   └── METRICS.md              # Prometheus metric catalog
├── concepts/                   # Understanding-oriented (theory)
│   ├── ENGINES.md
│   ├── PINS.md
│   ├── PATCHES.md
│   ├── DRIFT.md
│   ├── LICENSING.md
│   └── ARCHITECTURE.md
└── _adr/                       # Architectural Decision Records
    ├── 0001-multi-engine-refactor.md
    ├── 0002-engine-adapter-pattern.md
    └── ...
```

## 14.2 ADR template

```markdown
# ADR-XXX: Title

**Date**: YYYY-MM-DD
**Status**: Proposed | Accepted | Deprecated | Superseded by ADR-YYY
**Deciders**: @author1, @author2

## Context

What problem are we solving? What constraints exist? What was the trigger?

## Decision

What did we decide? Be specific. Use imperative voice.

## Consequences

### Positive
- Outcome 1
- Outcome 2

### Negative
- Trade-off 1
- Trade-off 2

### Neutral
- Side effect 1

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ... | ... | ... | ... |

## Alternatives considered

### Option A: [name]
Description. Why rejected.

### Option B: [name]
Description. Why rejected.

## References

- Related ADRs: ADR-XXX, ADR-YYY
- External docs: links
- PRs: links
```

## 14.3 First 10 ADRs to write

| ADR | Title |
|---|---|
| 0001 | Multi-engine architecture (sndr-platform refactor) |
| 0002 | Engine adapter pattern (EngineAdapter ABC) |
| 0003 | Per-pin manifest format (YAML with md5 anchors) |
| 0004 | Commercial wheel via entry_points (sndr-engine) |
| 0005 | Carbon Design System adoption (GUI) |
| 0006 | Zustand + TanStack Query for state management |
| 0007 | Lingui for i18n (en + ru) |
| 0008 | Layered architecture with CI-enforced rules |
| 0009 | Structured JSON logging via python-json-logger |
| 0010 | Repository pattern for state persistence |

## 14.4 Documentation quality requirements

Every public API function MUST have:
- Type hints on all parameters and return value
- Google-style docstring with: summary, args, returns, raises, example
- Listing in `docs/reference/API.md`

Every concept that operators need to understand MUST have:
- A markdown file in `docs/concepts/`
- A short "What is X?" first paragraph
- A "Why does it matter?" section
- Diagrams or examples
- "See also" links

Every error code MUST have:
- Entry in `docs/reference/ERRORS.md`
- Cause section
- Resolution section
- Examples

---

# Part 15: Repository Hygiene

## 15.1 Pre-commit hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        files: ^sndr/
        additional_dependencies: [pydantic, pyyaml]
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: detect-private-key
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-merge-conflict
  
  - repo: local
    hooks:
      - id: verify-layer-imports
        name: Verify layered architecture
        entry: python tools/ci/verify_layer_imports.py
        language: system
        files: ^sndr/.*\.py$
      
      - id: verify-no-engine-leak
        name: No engine-tier in public registry
        entry: python tools/ci/verify_no_engine_leak.py
        language: system
        files: ^sndr/engines/.*/registry\.py$
      
      - id: gui-eslint
        name: GUI ESLint
        entry: bash -c 'cd gui/web && npx eslint --max-warnings=0'
        language: system
        files: ^gui/web/src/.*\.(ts|tsx)$
```

## 15.2 PR template

```markdown
<!-- .github/PULL_REQUEST_TEMPLATE.md -->

## Description

What does this PR do? Why is it needed?

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation
- [ ] Refactor (non-functional code change)
- [ ] Performance improvement
- [ ] Test addition or improvement

## Quality checklist

- [ ] Tests added (unit + integration where applicable)
- [ ] Coverage ≥ 85% for new code in `sndr/`
- [ ] Documentation updated (reference, concepts, changelog)
- [ ] Changelog entry added (`docs/changelog/`)
- [ ] No security regressions (bandit + pip-audit clean)
- [ ] Performance impact assessed (< 2% bench regression)
- [ ] ADR created if architectural change
- [ ] Layered architecture rules respected (CI verifies)

## Linked issues

Closes #XXX

## Screenshots (if GUI change)

| Before | After |
|---|---|
| ... | ... |

## Migration notes (if breaking change)

Steps users need to take to upgrade.
```

## 15.3 CODEOWNERS

```
# .github/CODEOWNERS

# Default: project owner
*                                @sander

# Security-critical
sndr/license.py                  @sander @security
sndr/product_api/auth/           @sander @security

# Architecture
sndr/kernel/                     @sander @architecture
sndr/dispatcher/                 @sander @architecture
sndr/engines/base.py             @sander @architecture

# Per-engine
sndr/engines/vllm/               @sander @vllm-team
sndr/engines/sglang/             @sander @sglang-team

# GUI
gui/                             @sander @frontend

# Docs
docs/                            @sander @docs

# CI/CD
.github/                         @sander @platform
tools/ci/                        @sander @platform
```

## 15.4 .gitignore enforcement

```gitignore
# .gitignore (key entries)

# Commercial wheel (separate repo)
sndr_engine/
sndr-engine/

# Private docs (local-only)
sndr_private/
Genesis_internal_docs/

# Build artifacts
dist/
build/
*.egg-info/
__pycache__/
.mypy_cache/
.pytest_cache/
.ruff_cache/

# Node
node_modules/
gui/web/dist/
gui/web/.vite/

# Environment
.env
.env.local

# IDE
.vscode/
.idea/

# Secrets
*.pem
*.key
license.json
```

CI verifies on every PR that `.gitignore` patterns work:

```python
# tools/ci/verify_no_engine_leak.py
import subprocess
import sys

PROHIBITED_PATHS = [
    "sndr_engine/",
    "sndr-engine/",
    "sndr_private/",
]

PROHIBITED_FIELDS = ['tier: "engine"', "'tier': 'engine'"]

def main():
    # Check no prohibited paths in repo
    for path in PROHIBITED_PATHS:
        result = subprocess.run(["git", "ls-files", path], capture_output=True, text=True)
        if result.stdout.strip():
            print(f"ERROR: {path} should not be in public repo")
            sys.exit(1)
    
    # Check no engine-tier patches in registry
    result = subprocess.run(
        ["git", "grep", "-r", "tier.*engine"],
        capture_output=True, text=True,
    )
    # ... filter false positives, fail on real matches
    
    print("No engine leak detected.")

if __name__ == "__main__":
    main()
```

---

# Part 16: Migration Execution Plan

## 16.1 Phases overview

| # | Phase | Days | Risk | Key deliverables |
|---|---|---|---|---|
| 0 | Pre-flight + baseline | 2 | None | ADR-001, baseline tag, refactor branch |
| 1 | sndr skeleton + ABC + compat shims | 5 | Low | ADR-002, EngineAdapter contract, imports work both ways |
| 2 | Move engine-agnostic kernel | 5 | Low | text_patch, manifest, multi_file in sndr/ |
| 3 | Split detection (agnostic vs vllm) | 4 | Low | ADR-003, detection layer correctly split |
| 4 | Move 252 patches to engines/vllm/patches/ | 7 | Medium | Apply matrix unchanged, registry updated |
| 4.5 | License + commercial boundary | 4 | Medium | ADR-004, COMMERCIAL_TIER.md |
| 4.6 | sndr_private restructure | 1 | Low | Clean private docs, .gitignore audit |
| 5 | product_api refactor (routes + domain) | 7 | Medium | OpenAPI regenerated, backward compat shims |
| 6 | CLI refactor + `sndr` entry point | 4 | Low | CLI ref docs, shell completion |
| 6.5 | License GUI bridge | 2 | Low | Licensing panel in GUI |
| 7 | Per-pin manifests + drift cron | 7 | Medium | 3 baseline manifests, CI live |
| 8 | Container migration (production cut-over) | 3 | High | Universal launcher, production validated |
| 9 | GUI engine-aware top bar (Phase 1) | 7 | Low | Engine selector, pin manager, drift dashboard |
| 10 | SGLang skeleton | 4 | None | ADR-005, EngineAdapter skeleton doc |
| 11 | Cleanup + v12.0.0 release | 4 | Medium | All docs, changelog, release |
| 12 | GUI Carbon + features split (Phase 2) | 8 | Low | 24 feature modules, ≤300 LOC each |
| 12b | i18n implementation (en + ru) | 3 | Low | 100% strings translated |

Total: ~77 days focused work (~12 weeks)

## 16.2 Quality gate per phase

Before merging any phase to main:

```yaml
gate:
  tests:
    unit: pass + coverage ≥ target
    integration: pass
    e2e (if GUI change): pass
  lint:
    python: ruff clean
    typescript: eslint clean
  security:
    bandit: clean
    pip-audit: clean
  performance:
    bench_regression: < 2%
    gui_bundle: within budget
  documentation:
    api_docs: updated
    architecture_doc: updated (if structural change)
    changelog: entry added
  reviews:
    code: ≥ 1 approval
    architecture: ≥ 1 approval (if structural)
```

If any gate fails, merge is blocked.

## 16.3 Recommendation: per-phase branch + squash merge

```
main
 ├── refactor/sndr-platform (long-running base)
 │   ├── refactor/phase-0-baseline
 │   ├── refactor/phase-1-skeleton
 │   ├── refactor/phase-2-kernel
 │   └── ...
```

Each phase is a separate branch off the base. Reviewed independently. Squash
merged when ready. Visible per-phase history on main.

## 16.4 Rollback procedures

Every phase MUST be rollback-able:

| Phase | Rollback procedure |
|---|---|
| 0-3 | `git revert` — no production impact, shims preserve behavior |
| 4 | `git revert`; if production already running new patches, redeploy old via launcher rollback |
| 4.5 | `git revert` + restart with `vllm/sndr_core/license.py` shim |
| 5 | `git revert` + GUI continues using v1 URLs (shims kept until phase 11) |
| 6 | `git revert` + old CLI continues to work (entry point shim) |
| 7 | Skip generated manifests; fall back to old apply_matrix |
| 8 | Switch back to old launcher script; production restored |
| 9 | GUI changes are additive; revert removes new views |
| 10 | sglang skeleton is empty; revert is no-op |
| 11 | DO NOT REVERT after this phase; new release published |
| 12 | GUI changes are isolated by feature; revert one feature at a time |

## 16.5 Production validation per phase

After each phase that touches production:

```bash
# Health check
sndr health check --engine vllm
# Expected: all green

# Apply matrix
sndr patches list --engine vllm --filter applied
# Compare against pre-phase snapshot
# Expected: no unexpected changes

# Smoke bench
sndr bench quick --engine vllm --model qwen3.6-35b-fp8
# Compare TPS against baseline
# Expected: within ±2%

# GUI smoke
curl http://localhost:8800/api/v1/health
# Expected: 200 OK

# E2E from CI
cd gui/web && npm run test:e2e -- --grep smoke
# Expected: all pass
```

---

# Part 17: Glossary

| Term | Definition |
|---|---|
| **Adapter** | A class implementing `EngineAdapter` ABC, providing engine-specific behavior to engine-agnostic core |
| **Anchor** | A snippet of upstream code that a patch finds and modifies |
| **Apply matrix** | The set of patches that were applied (or skipped/failed) during a boot |
| **Bench** | Benchmark — measurement of inference performance (TPS, TPOT, TTFT) |
| **Boot** | The process of initializing sndr — config load, engine bootstrap, patch apply |
| **Carbon** | IBM's open-source design system used for the GUI |
| **CV** | Coefficient of Variation — std/mean, used to measure bench stability |
| **Drift** | A change in upstream code that breaks an anchor |
| **Engine** | An inference runtime (vLLM, sglang, ...) that sndr orchestrates |
| **Family** | A group of related patches (attention, moe, kv_cache, ...) |
| **g100** | Carbon's darkest theme, used as base for sndr Control Center |
| **GDN** | Gated Delta Net — a linear-attention layer in Qwen3.6 models |
| **Genesis** | Old internal codename, retained in some external references |
| **Ironrule #11** | "Deep-diff verification before retiring a patch as upstream-merged" — established discipline |
| **Lifecycle** | A patch's state: experimental / stable / retired |
| **License tier** | community (free) or engine (paid) |
| **Manifest** | YAML file describing per-pin file paths and anchor checksums |
| **MTP** | Multi-Token Prediction — a speculative decoding method |
| **Pin** | A specific vLLM version (commit SHA) we support |
| **Plugin** | An external Python package that registers patches via entry points |
| **SLI/SLO** | Service Level Indicator / Service Level Objective |
| **Strangler-fig** | Migration pattern: new code grows alongside old until old is removed |
| **TQ** | TurboQuant — KV cache compression (k8v4 = 8-bit key, 4-bit value) |
| **TTFT** | Time To First Token — latency from request to first response token |
| **TPOT** | Time Per Output Token — latency per generated token after first |
| **TPS** | Tokens Per Second — output throughput |
| **Wall TPS** | Throughput measured as wall-clock time / output tokens |

---

# Part 18: Frequently Asked Questions

## 18.1 Why not use Kubernetes operators / Helm?

We considered. Decided against because:
- Operators add complexity for small teams (1-3 operators)
- Most users do not have a k8s cluster
- Container management with Docker socket + SSH covers our scenarios
- Operators can be added in v14+ if demand emerges

## 18.2 Why not gRPC / protobuf?

We considered. Decided against because:
- Browser GUI is the primary consumer; gRPC-web is awkward
- REST + OpenAPI gives us TypeScript types for free
- SSE handles streaming use cases
- gRPC's binary efficiency does not matter for our request rates

## 18.3 Why not full Tauri for desktop?

We considered. Decided against for v12 because:
- Adds 3-4 days of work + significant CI complexity
- Most operators already have a browser
- Web app works on macOS, Linux, Windows uniformly
- Can be added in v13+ if users request

## 18.4 Why Lingui instead of react-intl?

- Smaller bundle (~20kb vs ~60kb)
- Better TypeScript support
- ICU MessageFormat (handles Russian plurals correctly)
- Better tooling for translator workflow (.po files)

## 18.5 Why Zustand instead of Redux?

- 3kb gzipped vs 20kb+ for Redux Toolkit
- No reducer/action/dispatch boilerplate
- Better TypeScript inference
- Less mental overhead for new contributors
- Sufficient for our state complexity

## 18.6 Why hash routing instead of React Router?

- Works without server-side rewrite rules
- Smaller bundle
- Easier to bookmark / link
- 24 routes do not need React Router's advanced features

## 18.7 Why Apache 2.0 for public + proprietary for private?

- Apache 2.0 maximizes adoption of the community tier
- Patent grant protects contributors and users
- Commercial wheel with proprietary license enforces paid tier
- Standard SaaS/dual-license pattern (e.g., Redis, Elastic, Mongo)

## 18.8 What happens if a license expires in production?

- Engine-tier patches are no longer applied at next boot
- Community-tier patches continue to work normally
- GUI shows a prominent license warning banner
- Engine-tier features degrade gracefully (no crashes)
- Customer receives email + GUI alert ahead of expiry

## 18.9 How do we handle a security vulnerability in a patch?

- Yank the patch via lifecycle: `retired_security`
- Issue advisory in `docs/security/` (date-numbered)
- Bump patch revision
- Notify customers via mailing list
- See SECURITY.md for full disclosure procedure

## 18.10 Can we contribute new engines (TensorRT-LLM, Triton)?

Yes, but:
- Engines must reach 80% feature parity with existing adapters
- Maintainer commitment required (active contributor)
- Initial implementation must include 5+ working patches
- ADR required for design decisions

---

# Part 19: Risk Register

| ID | Risk | Probability | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| R-01 | Refactor takes 50% longer than estimated | High | Medium | Phase-by-phase execution with quality gates; can pause at any phase boundary | Sander |
| R-02 | Production regression during phase 8 cut-over | Medium | High | Strangler-fig pattern; old launcher kept; instant rollback | Sander |
| R-03 | Upstream vLLM regression masks our refactor regression | Medium | Medium | Vanilla baseline measured per pin; isolate via diff | Sander |
| R-04 | Carbon Design System version churn breaks GUI | Low | Low | Pin major version; review minor updates | Frontend |
| R-05 | Commercial customers reject license rotation | Low | High | Maintain old tokens compat for 1 release cycle | Sander |
| R-06 | sglang adapter never written (skeleton stays empty) | Medium | Low | Doc skeleton clearly; not a release blocker | Sander |
| R-07 | i18n adds maintenance burden | High | Low | Lingui CLI extracts strings; .po files easy to update | Frontend |
| R-08 | Drift detection cron generates noise | Medium | Low | Tune thresholds; allow ack/silence per issue | Platform |
| R-09 | Security vulnerability in license verification | Low | Critical | Code review by security; pen-test; bandit; CodeQL | Security |
| R-10 | Performance regression from observability overhead | Low | Medium | Bench with/without observability; budgets enforced | Sander |
| R-11 | Engine adapter ABC too rigid for sglang | Medium | Medium | ABC reviewed during sglang first port; can extend before v13 | Sander |
| R-12 | Manifest format becomes unwieldy at 10+ pins | Medium | Low | Lazy load per-pin; YAML supports anchors and merge | Platform |
| R-13 | Two phases (12, 12b GUI) compete with platform work | High | Medium | Sequence: platform first (phase 11), GUI last (phase 12) | Sander |

---

# Part 20: Success Metrics

We will measure success of the refactor by these metrics, evaluated 3 months
post v12.0.0:

## 20.1 Engineering velocity

- **Patch authoring time**: from idea to PR
  - Baseline: ~4 hours (current; involves multiple checklists)
  - Target: < 1 hour (with scaffolding + clear docs)

- **Pin upgrade time**: from new image available to validated promotion
  - Baseline: ~8 hours (manual; this session)
  - Target: < 2 hours (one command + report review)

- **Drift response time**: from upstream merge to maintainer notified
  - Baseline: ~weeks (manual)
  - Target: < 24 hours (daily cron)

## 20.2 Code quality

- **Test coverage**: ≥ 85% on critical paths (kernel, dispatcher, license)
- **Bench reproducibility**: CV < 8% sustained
- **Lint cleanliness**: 0 ruff warnings, 0 mypy errors
- **Bundle size**: GUI main bundle < 350kb gzipped

## 20.3 Operational excellence

- **Boot duration P95**: < 5 seconds
- **GUI page load P95 (warm)**: < 500 ms
- **API response P95**: < 200 ms (non-streaming)
- **License verification**: < 100 ms

## 20.4 Adoption

- **External contributors**: ≥ 2 non-Sander contributors with merged PRs
- **GitHub stars**: ≥ 100 (engagement signal, not a goal in itself)
- **Documentation completeness**: every public API has a docstring + ref entry

## 20.5 Business

- **Commercial license adoption**: track per customer (private metric)
- **Support tickets**: reduction in "how do I X?" tickets via better docs
- **Onboarding time**: new maintainer to first merged PR < 1 week

---

# Appendices

## Appendix A: ADR-001 (full text — to be created)

The first ADR will document the multi-engine refactor decision. Template:

```markdown
# ADR-001: Multi-engine architecture refactor to sndr-platform

**Date**: 2026-06-05
**Status**: Accepted
**Deciders**: @sander

## Context

The `genesis-vllm-patches` codebase has grown to 333 patches, a full enterprise
GUI, a commercial paid tier, and a multi-host fleet management surface. The
current structure has problems:

1. sndr_core lives under vllm/sndr_core/ — false parent-child
2. No engine abstraction — sglang adoption requires rewrite
3. Monolithic directories (attention/ has 101 files)
4. 11,633-line App.tsx is unmaintainable
5. Drift detection is manual

The strategic direction is multi-engine (vllm + sglang). The current
architecture blocks this.

## Decision

Refactor the codebase to a multi-engine architecture under the name
`sndr-platform`. Key elements:

- sndr package at top level (not under vllm/)
- engines/ directory with per-engine adapters
- Per-pin YAML manifests for drift detection
- Carbon Design System for GUI
- Lingui for i18n (en + ru)
- Layered architecture (CI-enforced)
- Strangler-fig migration (12-week plan, see Master Spec)

## Consequences

### Positive
- Multi-engine ready (sglang, future engines)
- Drift detection automatable
- Patch authoring simplified
- GUI maintainable (feature split)
- Enterprise look (Carbon)
- i18n unblocks RU market

### Negative
- 12 weeks of focused work
- Risk of regression during phases 4-8
- New contributors must learn Carbon + Zustand

### Risks
See Risk Register in Master Spec (Part 19).

## Alternatives considered

### Option A: Incremental cleanup (no refactor)
Reject: does not unblock sglang, does not solve App.tsx monolith.

### Option B: Full rewrite from scratch
Reject: throws away working code, no business value.

### Option C: Keep vllm-only forever
Reject: contradicts strategic direction.

## References

- Master Spec: docs/superpowers/specs/2026-06-05-sndr-platform-master-spec.md
- Risk Register: Master Spec Part 19
- Migration Plan: Master Spec Part 16
```

## Appendix B: Code style examples

### Python: good vs bad

```python
# BAD: implicit, untyped, no docstring
def apply(patch, cfg):
    if cfg.strict:
        result = patch.apply()
        if not result:
            raise Exception("Failed")
        return result
    try:
        return patch.apply()
    except:
        pass
    return None

# GOOD: explicit, typed, documented
def apply_patch(
    patch: Patch,
    config: SndrConfig,
) -> PatchResult:
    """Apply a single patch with the given configuration policy.
    
    Args:
        patch: The patch to apply.
        config: Runtime configuration; controls strict mode.
    
    Returns:
        PatchResult indicating success, skip, or failure.
    
    Raises:
        PatchApplyFailedError: If strict mode is enabled and apply fails.
    """
    try:
        result = patch.apply()
        log.info(
            "patch.applied",
            extra={"patch_id": patch.id, "outcome": result.status},
        )
        return result
    except Exception as e:
        log.error(
            "patch.failed",
            extra={"patch_id": patch.id, "error": str(e)},
            exc_info=True,
        )
        if config.strict_apply:
            raise PatchApplyFailedError(
                f"Patch {patch.id} failed in strict mode",
                patch_id=patch.id,
                cause=str(e),
            ) from e
        return PatchResult(status="failed", patch_id=patch.id, error=str(e))
```

### TypeScript: good vs bad

```typescript
// BAD: any types, no error handling, mutation
function getPatches(filter) {
  const r = fetch('/api/patches?filter=' + filter);
  return r.then(r => r.json()).then(data => {
    data.patches.forEach(p => p.loaded = true);
    return data.patches;
  });
}

// GOOD: typed, handled, immutable
import type { Patch } from '@/api/schema.gen';
import { apiClient } from '@/api/client';

interface PatchFilter {
  tier?: 'community' | 'engine';
  family?: string;
  lifecycle?: 'experimental' | 'stable' | 'retired';
}

export async function listPatches(
  engine: string,
  filter: PatchFilter = {},
): Promise<readonly Patch[]> {
  try {
    const response = await apiClient.get<{ data: Patch[] }>(
      `/api/v1/engines/${engine}/patches`,
      { params: { filter } },
    );
    return response.data.data;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return [];
    }
    throw error;
  }
}
```

## Appendix C: Phase 0 checklist

```markdown
- [ ] Git tag baseline: `pre-sndr-refactor-baseline`
- [ ] Branch created: `refactor/sndr-platform`
- [ ] Master Spec committed
- [ ] ADR-001 committed
- [ ] Work journal initialized
- [ ] Success criteria documented
```

## Appendix D: Common patterns

### Pattern: Idempotent operation

```python
_done: bool = False

def init():
    global _done
    if _done:
        return
    # ... do work
    _done = True
```

### Pattern: Lazy import

```python
def expensive_operation():
    # Import lazily — only when this function is called
    import heavy_dependency
    return heavy_dependency.do_thing()
```

### Pattern: Context-aware logger

```python
from contextvars import ContextVar

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)

def with_trace_id(trace_id: str):
    token = _trace_id.set(trace_id)
    try:
        yield
    finally:
        _trace_id.reset(token)

# Usage
with with_trace_id("abc123"):
    log.info("event")  # automatically includes trace_id=abc123
```

---

## Document control

| Date | Version | Author | Change |
|---|---|---|---|
| 2026-06-05 | 1.0.0 | Sander | Initial master spec |

---

End of document. Total length: ~2200 lines. This is a living document; updates
proceed through ADRs referenced from this spec.

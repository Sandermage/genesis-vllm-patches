# ADR-0001: Multi-engine architecture refactor to sndr-platform

**Date**: 2026-06-05
**Status**: Accepted
**Deciders**: @sander
**Related**: Master Spec and Execution Journal (maintainer-internal working notes, not published)

## Context

`genesis-vllm-patches` started as a personal patch overlay for vLLM on one
operator's hardware. It grew to:

- 333 patches across 26 subdirectories
- A commercial paid tier (`sndr_engine`) with Ed25519 license verification
- A 11,633-line monolithic React GUI with multi-host fleet management
- An automation surface (CLI, REST API, SSE streams)
- A drift detection / manifest tooling pipeline
- Operator scripts, config builders, doctor commands

The structural problems blocking further growth:

1. **`sndr_core` lives under `vllm/sndr_core/`** — our code masquerades as part
   of the vLLM package. The mount path is inside the upstream package
   (`vllm/sndr_core/`), which creates artificial dependency on a specific
   namespace that we do not own.

2. **No engine abstraction** — every detection, every patch, every CLI command
   hardcodes `from vllm.*`. Adding sglang (planned next strategic step) would
   require rewriting most of the codebase.

3. **Monolithic directories** — `integrations/attention/` has 101 files in one
   flat folder. `product_api/` has 45 root files. `cli/` has 40 root files. The
   organization broke down a long time ago.

4. **GUI is an 11,633-line monolith** — `App.tsx` is the largest file in the
   project. Adding a new feature requires modifying this one file. Cognitive
   load on new contributors is prohibitive.

5. **Drift detection is manual** — every pin upgrade is a multi-step playbook
   (8+ steps) that can be forgotten. Upstream changes that break our anchors
   are discovered only when production fails.

6. **GUI styling is bespoke** — no design system, inconsistent components, no
   accessibility audit. Cannot present an enterprise UX.

7. **Engineering principles not enforced** — naming conventions, import rules,
   coverage targets are folklore. CI does not block violations.

The strategic direction is **multi-engine support** (vLLM + sglang now,
TensorRT-LLM/Triton later). The current architecture is the primary blocker.

## Decision

Execute a 12-week refactor to `sndr-platform`. Key architectural elements:

1. **`sndr` Python package at top level** (not under `vllm/`).
2. **`engines/` directory** with per-engine adapters (`vllm`, `sglang`).
3. **`EngineAdapter` ABC** as the contract for engine-specific behavior.
4. **Per-pin YAML manifests** for drift detection and anchor resolution.
5. **Carbon Design System** (IBM, g100 theme) for the GUI.
6. **Lingui i18n** (English + Russian at launch).
7. **Layered architecture** (kernel → engines → dispatcher → apply → API → CLI)
   with CI-enforced dependency rules.
8. **Strangler-fig migration** in 16 phases (see Master Spec Part 16).
9. **Commercial wheel** (`sndr-engine`) as a separate private repository,
   loaded via setuptools entry points.
10. **Quality gates** enforced per phase: coverage, lint, security, performance.

Full architectural specification lives in the maintainer-internal
master engineering spec (not published).

## Consequences

### Positive

- **Multi-engine ready**: sglang adoption becomes a port (weeks) rather than a
  rewrite (months).
- **Drift detection automatable**: daily cron + per-pin manifests catch upstream
  changes within 24 hours.
- **Patch authoring simplified**: clear conventions, scaffolding, focused
  directories.
- **GUI maintainable**: 24 feature modules instead of 1 monolith.
- **Enterprise presentation**: Carbon Design System signals professional
  engineering to commercial buyers.
- **i18n unblocks RU market**: Russian-speaking operators get first-class
  experience.
- **Onboarding accelerated**: new maintainer can read the Master Spec in 2 hours
  and contribute within a week.
- **Quality measurable**: CI-enforced budgets prevent regression.

### Negative

- **12 weeks of focused work** (or 5-6 months calendar at partial allocation).
- **Risk of regression** during phases 4-8 (patches move, container migrates).
- **New contributors must learn** Carbon + Zustand + Lingui + layered
  architecture conventions.
- **Migration effort** for existing operators (one launcher script update).
- **Document maintenance** is non-trivial (10+ ADRs, reference docs, concepts).

### Neutral

- **Repository renamed conceptually** (still `genesis-vllm-patches` git remote,
  but logical name is `sndr-platform` for v12.0.0).
- **Backward compatibility** is preserved for one release cycle via shims.

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Refactor takes 50% longer than estimated | High | Medium | Phase-by-phase with quality gates; pause at any boundary |
| Production regression during phase 8 cut-over | Medium | High | Strangler-fig pattern; old launcher kept; instant rollback |
| Upstream vLLM regression masks our refactor regression | Medium | Medium | Vanilla baseline per pin; isolate via diff |
| Commercial customers reject license token rotation | Low | High | Maintain old tokens compat for 1 release cycle |
| sglang adapter never written (skeleton stays empty) | Medium | Low | Doc skeleton clearly; not a release blocker |
| Engine adapter ABC too rigid for sglang | Medium | Medium | Reviewed during first sglang port; can extend before v13 |

Full risk register in Master Spec Part 19.

## Alternatives considered

### Option A: Incremental cleanup (no refactor)

Make small improvements within current structure: split a few large files,
add some tests, document conventions.

**Rejected because**:
- Does not unblock sglang (the strategic goal)
- Does not solve the 11,633-line App.tsx
- Drift detection stays manual
- Does not give enterprise look (Carbon vs bespoke CSS)

### Option B: Full rewrite from scratch

Write `sndr-platform` v1.0 from scratch as a new repository. Port patches one
by one.

**Rejected because**:
- Throws away working production code (333 patches that work)
- No business value during the rewrite window (weeks of nothing shippable)
- Risk of "second-system syndrome" (over-engineering)
- Migration path for existing operators is harder
- Strangler-fig pattern delivers the same outcome with lower risk

### Option C: Keep vllm-only forever

Decline multi-engine ambition. Stay focused on vLLM.

**Rejected because**:
- Contradicts strategic direction
- Reduces commercial value proposition
- Sglang and TensorRT-LLM are increasingly common in enterprise
- We have already committed to multi-engine in commercial conversations

### Option D: Adapter pattern in current structure (no top-level move)

Add `engines/` subdirectory under `vllm/sndr_core/` without renaming. Get most
of the multi-engine benefit without the import path migration.

**Rejected because**:
- Keeps the false parent-child (sndr_core under vllm/)
- Confuses contributors ("why is sglang adapter under vllm/?")
- Does not solve the namespace masquerade problem
- Mount path remains inside upstream package — risk of conflicts

## References

- **Master Spec**: maintainer-internal master engineering spec (not published)
- **Execution Journal**: maintainer-internal refactor execution log (not published)
- **Risk Register**: Master Spec Part 19
- **Migration Plan**: Master Spec Part 16
- **Carbon Design System**: https://carbondesignsystem.com/

## Decision log

| Date | Change | Author |
|---|---|---|
| 2026-06-05 | Initial decision, status: Accepted | @sander |

# Core / Engine / Private namespace boundary

> Codifies the three-zone namespace policy first surfaced in
> P0.PROJECT-STRUCTURE.R+ (2026-05-24). Lives next to
> [`SPONSORS.md`](SPONSORS.md), [`LICENSE_POLICY.md`](LICENSE_POLICY.md),
> and [`RELEASE_POLICY.md`](RELEASE_POLICY.md).

Genesis ships from a single source tree that contains code at three
different distribution tiers. The boundary between them is enforced
by tests + an audit gate, not by convention. This document is the
written contract those gates check against.

## The three zones

| Zone | Path | Distribution | Visible in public clone? | Ships in core wheel? |
| --- | --- | --- | --- | --- |
| **Core (community)** | `vllm/sndr_core/` | Apache-2.0 community wheel `vllm-sndr-core` | Yes | Yes |
| **Engine (commercial)** | `vllm/sndr_engine/` | Separate commercial wheel via `pyproject-engine.toml` + license gate | Reserved, currently empty | **No** (explicit `pyproject.toml` exclude) |
| **Private (maintainer)** | repo-root `sndr_private/` | Never distributed | **No** (gitignored) | **No** (top-level, outside `vllm/`) |

The three are mutually exclusive. Any directory matching the
pattern `vllm/**/sndr_private` is a category error — it would mean
maintainer-private content sitting inside the public Apache namespace.
The error is enforced as a hard rule (#27).

## Core — `vllm/sndr_core/`

The whole Apache-2.0 community wheel. After the 2026-05-08 strict-AND
audit and 2026-05-24 P0.PROJECT-STRUCTURE.R+ relocation, this zone
contains all production patches, including ones previously parked in
"engine" (P67/P67b/P67c, PN21..PN24, PN26, PN29, PN38, PN40, PN57,
P82, PN16, PN65, and the legacy P* family). They are all community
patches now.

Layout invariants:

- `vllm/sndr_core/integrations/<family>/` — runtime overlays
- `vllm/sndr_core/integrations/_retired/` — env-gated dormant patches
  preserved for back-compat (e.g. retired G4 upstream work-in-progress)
- `vllm/sndr_core/dispatcher/`, `apply/`, `model_configs/`, etc. —
  framework

No `sndr_private/` subdirectory anywhere under `vllm/sndr_core/`.

## Engine — `vllm/sndr_engine/`

The reserved single private/commercial code namespace. Built as a
separate wheel from `pyproject-engine.toml`, gated by a signed
license token. Loaded — when present — through `engine_available()`
optional-discovery imports from `sndr_core`; never a hard import.

Currently the namespace is **reserved but empty**. After the
2026-05-08 strict-AND audit, the only previous candidate (PN72) was
reclassified to community because its real algorithm ships in
`vllm/sndr_core/kernels/ngram_frequency_filter.py`.

Boundary rule for new patches — `tier="engine"` only when **all four**
conditions hold (Sander's strict-AND rule, see
[`PATCHES.md`](PATCHES.md#engine-tier-the-strict-and-boundary)):

1. NOT present on the public GitHub repo
2. NO external author credit in title / credit text
3. NO PR link / PR number in title / credit text
4. NO `upstream_pr` / `related_upstream_prs` field

If any of the four fail → community tier. A patch with a public PR
link or external co-author can never be engine, even if Genesis adds
substantial new work on top — that's an Apache-2.0 derivative,
not a clean-room maintainer-original.

## Private — repo-root `sndr_private/`

Maintainer-private archive: planning notes, audit reports,
abandoned experiments, run logs. Sits at the **repo root**, not
inside `vllm/`. Two protections keep it from ever shipping:

1. `.gitignore` excludes `sndr_private/` from commits → never reaches
   GitHub. Audit script `scripts/audit_private_namespace.py` (rule 3)
   verifies this on every pre-commit run.
2. `pyproject.toml` package discovery is `include = ["vllm.sndr_core*"]`,
   so top-level directories are outside the wheel by construction.

Typical contents (all gitignored):

```
sndr_private/
├── planning/        # roadmaps, master plans, P0 deliverables
├── audits/          # internal audit reports, deep-diff findings
├── archived/        # dormant code preserved for history (e.g. genesis_tq_abandoned/)
├── runs/            # bench results, log captures
└── research/        # speculative experiments, RFCs
```

The only allowed location for the `sndr_private` namespace is
this repo-root directory.

## What is forbidden

**Hard rule #27 — no `sndr_private` directory anywhere under `vllm/`.**

Forbidden patterns (all blocked by `scripts/audit_private_namespace.py`):

- `vllm/sndr_core/sndr_private/`
- `vllm/sndr_core/<anything>/sndr_private/`
- `vllm/sndr_engine/sndr_private/`
- Any other `vllm/**/sndr_private`

Historical note: P0.PROJECT-STRUCTURE.R+ found `vllm/sndr_core/sndr_private/`
shipping in the wheel (17 files split across `genesis_tq_abandoned/`
dormant exploration and `g4_upstream_tq_wip/` env-gated retired
patches). P0.1 M.3a-d relocated them:

- `genesis_tq_abandoned/` → `sndr_private/archived/genesis_tq_abandoned/` (top-level, gitignored)
- `g4_upstream_tq_wip/` → `vllm/sndr_core/integrations/_retired/g4_upstream_tq_wip/` (proper public namespace for retired patches)

## How the boundary is enforced

Three independent gates, each catching a different failure mode:

| Gate | Where | What it checks |
| --- | --- | --- |
| `tests/unit/test_wheel_contents.py` (M.2) | pytest | Black-box wheel contract — builds the wheel, asserts no `sndr_private` and no `vllm/sndr_engine/*` paths anywhere inside |
| `scripts/audit_private_namespace.py` (M.7) | pre-commit + `make audit-private-namespace` | Source-tree gate — forbids `vllm/**/sndr_private` directories, cross-checks M.2 still asserts the wheel invariant, verifies top-level `sndr_private/` stays gitignored |
| `tests/unit/test_edition_boundary.py` | pytest | Strict-AND rule for `tier="engine"` — every engine-tier patch must satisfy all four conditions |

Together they make the boundary regression-proof: changing the
implementation of the wheel build (recursive globs, MANIFEST.in,
src-layout, hatchling, etc.) is fine; the three contracts above stay
the same.

## See also

- [`PATCHES.md`](PATCHES.md#engine-tier-the-strict-and-boundary) — strict-AND rule and the 2026-05-08 audit reclassification
- [`SPONSORS.md`](SPONSORS.md) — relationship between sponsorship and the open / commercial split
- [`LICENSE_POLICY.md`](LICENSE_POLICY.md) — Apache-2.0 / commercial license terms per zone
- [`RELEASE_POLICY.md`](RELEASE_POLICY.md) — proof-artefact ratchet that depends on the boundary

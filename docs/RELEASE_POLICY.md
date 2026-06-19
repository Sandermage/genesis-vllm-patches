# Release Policy — Genesis vLLM Patches

This document is the **single source of truth** for which patch-proof
mode gates a public Genesis release. It exists to remove the
ambiguity flagged by the 2026-05-15 production-readiness audit (C1):
two release-check modes existed (`require-static` and `require-baseline`)
without an explicit decision which one is the current release gate.

## TL;DR

| Mode | Status | Where it runs |
| --- | --- | --- |
| `require-static` | **Active release gate** | `make evidence --release`, CI |
| `require-bench` | Optional ratchet | `make audit-release-check-bench-attached` |
| `require-baseline` | Optional ratchet (strict) | `make audit-release-check-baseline-optional` |

A release ships when `require-static` passes for every registry entry.
The two stricter modes are run on demand by operators preparing a
hardened deploy.

## Mode definitions

### `require-static` — current release gate

Every entry in `PATCH_REGISTRY` must have a static-proof JSON under
`evidence/patch_proof/<id>__*.json` that records:

- registry presence (id resolves)
- `apply_module` resolution (file exists OR `_retired/` exemption)
- env-flag declaration matches `sndr/env.py::Flags`
- dispatcher wiring matches expectations (legacy `@register_patch` OR
  spec-driven dispatch via `dispatcher/spec.py`)

No bench data is required. The proof artifact is produced by
`sndr patches release-check --mode require-static` and persists across
runs under `evidence/patch_proof/`.

Why this is the gate: every patch the registry advertises must at least
*resolve* before release; bench data quality varies per patch and is
hardware-bound, so making it mandatory would block every release until
operators re-bench all 318 entries on their rig. See
`audit_release_check_baseline-optional` rationale in the Makefile for
the historical decision.

### `require-bench` — optional ratchet (default-on subset)

Same as `require-static` PLUS at least one bench attachment per patch
(any `bench_attached` artifact, regardless of baseline comparison).
Operators preparing a hardened deploy run this against the default-on
subset of their preset first, then expand coverage incrementally.

### `require-baseline` — optional ratchet (strict)

Same as `require-bench` PLUS the bench must carry a baseline
comparison (`bench_with_baseline`). This is the strictest mode and
takes the longest to satisfy because every patch needs a reference
bench run.

Currently 0/318 entries carry `bench_with_baseline`. Operators who
want to adopt this gate should:

1. Promote the **default-on patches in production presets** first
   (smallest set that actually matters in steady-state traffic).
2. Then **all default-on** patches across all presets.
3. Then **stability-critical** patches (PN95 tier-aware cache,
   PN116/118/119 TurboQuant backports, PN132/133 correctness fixes).
4. Then everything else (community / research / drift-sensitive).

## How to run each gate

```bash
# Release-gate (current public policy) — used by CI + make evidence
make audit-release-check
python3 -m vllm.sndr_core.cli patches release-check --mode require-static

# Hardened ratchet 1 — at least one bench attached per patch
make audit-release-check-bench-attached
python3 -m vllm.sndr_core.cli patches release-check --mode require-bench

# Hardened ratchet 2 — bench with baseline comparison per patch
make audit-release-check-baseline-optional
python3 -m vllm.sndr_core.cli patches release-check --mode require-baseline
```

The `make_evidence.py --release` aggregate wires the **release gate**
into a 43-of-43 audit panel. Operators who want one of the ratchets
to participate in that aggregate should add the matching target to
`scripts/make_evidence.py` and document the change in this file.

## Lifecycle of a policy decision

1. **Proposal.** A maintainer opens a PR that bumps `make_evidence.py`
   to a stricter default mode. The PR description must show the
   percentage of registry entries that pass under the proposed gate
   today.
2. **Coverage runway.** The PR remains open until at least 95% of
   default-on entries in **all** production presets pass under the
   proposed gate.
3. **Cutover.** The PR merges, `RELEASE_POLICY.md` is updated to
   reflect the new gate, and any operator-facing scripts are
   refreshed (Makefile target description, README badge, CI YAML).

Inverse direction (loosening the gate) is allowed only when the
stricter mode causes a release-blocking regression that cannot be
fixed in the short window of a release candidate; loosen-then-fix is
preferred over no-release. Such loosenings must reference the
upstream bug, the regression-test that catches it next time, and the
target date for re-enabling the strict gate.

## Related artefacts

- `Makefile` targets: `audit-release-check*`
- CLI: `sndr/cli/legacy/patches.py::release_check`
- Implementation: `sndr/proof/release_check.py`
- CI workflow: `.github/workflows/test.yml` (`make evidence --release`)
- Audit history is captured per-release in the production-readiness
  audit reports that maintainers attach to each release artifact.

## Operator runbook — promoting the production subset to `bench_with_baseline`

This section is the step-by-step recipe for going from the public
release gate (`require-static`, 318/308 covered out-of-the-box) to
the hardened ratchet (`require-bench` or `require-baseline`) on the
practical subset that actually ships in production presets.

### What is the "production subset"?

`vllm.sndr_core.proof.production_subset.get_production_subset()` is
the canonical source. It returns the frozenset of patches that are
either:

- enabled (`GENESIS_ENABLE_*=1`) by any V2 preset matching
  `prod-*` under `vllm/sndr_core/model_configs/builtin/presets/`, or
- flagged `default_on=True` in `PATCH_REGISTRY` (so they load
  implicitly even without an explicit preset opt-in).

Current state: **109 patches** across 8 production presets
(prod-{27b,35b}-{dflash,dflash-multiconc,tq,tq-multiconc} —
some combinations omitted). The remaining ~60 patches in the
registry are experimental/research opt-in or retired — they stay
under `require-static` even on hardened deploys.

> **Note** (2026-06-02): The bench-attachment percentages in this section are a
> Wave 10-era snapshot (registry count = 169, bench-with-baseline = 81). The
> registry has grown to 236 entries since (Phase 6 P3.7 Product API
> materialization, Phase 10 V1 sunset retirements, and ongoing patch additions).
> Re-deriving the percentages requires re-running the bench-attachment workflow
> against the current preset roster (`sndr patches release-check --mode
> require-bench-attached` against all 16 `prod-*` presets on the rig) —
> deferred until next bench window. Trend direction (most patches are
> `static_only` proof, no measurement attached) still holds; the absolute
> denominators have shifted upward.
>
> See Phase 6.D operator deferral note in commit `297e09f7` for the original
> rationale.

> **Update** (2026-06-03): partial re-derivation completed against the active
> `prod-gemma4-31b-tq-mtp-structured-k4` container on pin
> `0.21.1rc1.dev354+g626fa9bba` (registry 240 entries, production_subset 152):
>
> | Bucket | Before bench-attach | After bench-attach |
> |---|---:|---:|
> | dead (no proof) | 152 (100%) | 115 (75.7%) |
> | static_only (proof, no bench) | 0 (0%) | 37 (24.3%) |
> | bench_with_baseline | 0 (0%) | 0 (0%) |
>
> Workflow performed: `bench_multiturn_tps`-derived bench JSON attached via
> `sndr patches bench-attach G4_NN .../gemma_prod.json` to each of the 37
> active `G4_*` patches in the production-subset. Bench-attach without
> `--baseline` populates the bench_delta block with timestamp + measured
> headline metrics, but *_pct deltas remain null — these patches are now
> in `static_only` bucket (proof + bench timestamp but no comparison
> baseline). To promote to `bench_with_baseline`, the bench needs a
> baseline JSON (run with K_001=OFF for example, then attach with
> `--baseline /path/to/baseline.json`).
>
> The remaining 115 patches are in `dead` because the Gemma preset's
> active env doesn't enable them (the production-subset includes ALL
> patches enabled by ANY prod-* preset, not just the currently-running
> one). Promoting them requires booting each of the other 15 prod-*
> presets in turn and running the same attach workflow.
>
> Trend direction (most patches are `static_only` or `dead`, very few in
> `bench_with_baseline`) holds. Path to enterprise-grade gating:
> automate the per-preset boot + bench + attach cycle as a CI workflow.
> See `tools/bench_multiturn_tps.py` for the reusable measurement harness
> + `scripts/attach_bench_proof.py` (legacy) for the attach helper.

### Current attachment state (2026-05-22)

After running the workflow below against the `prod-qwen3.6-27b-dflash-multiconc`
preset (the active homelab container):

- **95/226 patches** (42.0%) are in `bench_with_baseline`
  - 45 from `prod-qwen3.6-27b-dflash-multiconc.genesis_env`
  - 50 default-on patches running implicitly in the same container

- **28 production-subset patches** remain in `static_only` because
  they are enabled by **other** prod-* presets that have not yet been
  bench-attached on this rig:

  ```text
  P37  P70  P71  P78  P81  P82  P94  P95   P98  P99
  P101 P103 P107 P67b
  PN8  PN12 PN14 PN16 PN17 PN22 PN30 PN52
  PN56 PN66 PN67 PN82 PN90
  ```

### How to promote one more preset

For each remaining prod preset, repeat the cycle:

```bash
# 1. Boot the preset's container (replaces the previous running one)
sndr launch <preset>             # e.g., sndr launch prod-qwen3.6-27b-tq-multiconc

# 2. Regenerate static proofs (per-host, per-pin)
sndr patches prove --all

# 3. Bench the live endpoint
python3 sndr/extras/tools/genesis_bench_suite.py \
    --host localhost --port 8000 --api-key genesis-local \
    --model <model_id> --quick \
    --out tools/bench_results/<preset>_$(date +%Y%m%d_%H%M%S).json

# 4. Save the bench as the new canonical baseline for this preset
cp tools/bench_results/<preset>_*.json \
   tests/integration/baselines/<preset>.json

# 5. Attach to every patch the preset enables — explicit env + default_on
python3 scripts/attach_bench_proof.py \
    --bench tools/bench_results/<preset>_*.json \
    --baseline tests/integration/baselines/<preset>.json \
    --preset <preset> \
    --include-default-on
```

After cycling all 8 prod presets, `proof-status` should report
~122/230 (~53%) in `bench_with_baseline` — the by-design ceiling
for the current registry (subset of patches enabled by any
`prod-*` preset). The remaining ~106 stay `static_only` by design —
they are experimental opt-in patches that no production preset
enables.

Note on denominators in this section: **109** = production-subset
(line 137 — patches eligible for bench gates), **169** = eligible
count at the 2026-05-16 R-01 audit snapshot (CHANGELOG entry below),
**226** = total registry count at the 2026-05-22 attachment-state
snapshot (line 148), **230** = current registry count. Historical
snapshot numbers (109/169/226) are dated and preserved; only the
forward projection above tracks the live registry total.

### Per-preset coverage cheatsheet

The list below shows which patches each preset uniquely contributes
(not already covered by the prod-qwen3.6-27b-dflash-multiconc bench). A
single bench run per preset closes its row.

| Preset | New patches it covers | Model |
| --- | --- | --- |
| `prod-qwen3.6-27b-tq-k8v4` | 22 (P101 P103 P107 P67b P70 P82 P94 P95 P98 P99 PN8 PN12 PN14 PN16 PN17 PN30 PN52 PN56 PN66 PN67 PN82 PN90) | Qwen3.6-27B-int4-AutoRound (TQ k8v4 KV) |
| `prod-qwen3.6-27b-tq-multiconc` | same 22 (max_num_seqs=8) | same |
| `prod-qwen3.6-35b-balanced` | +P37 +P71 +P81 (25 total) | Qwen3.6-35B-A3B-FP8 (single-conc) |
| `prod-qwen3.6-35b-multiconc` | same 25 (max_num_seqs=8) | same |
| `prod-qwen3.6-35b-dflash` | 13 (P37 P67b P70 P78 P81 P82 P98 P99 PN8 PN17 PN22 P101 P103) | 35B + DFlash drafter |
| `prod-qwen3.6-35b-dflash-multiconc` | same 13 | same |

### Patches with no test coverage at all (work needed)

After R-04 triage (audit 2026-05-16) **6 patches in the production
subset have no test coverage** — neither a dedicated `test_pNN_*.py`
nor a family-contract test:

```text
P60  P60b  P67b  P78   →  attention.gdn / .turboquant (kernel-bound)
PN122  PN202              →  observability / streaming (cross-cutting)
```

These are genuine gaps. They stay `review_required` (correctly)
under derived metadata. A contributor who wants to close them
should follow [CONTRIBUTING.md § "How to add a new patch"](CONTRIBUTING.md)
step 4 ("Add a unit test").

The remaining `review_required=39` count is correct semantics for
experimental opt-in patches outside the production subset — those
require an operator review before enabling because they have no
empirical evidence of safety on the operator's specific rig.

## History

| Date | Change |
| --- | --- |
| 2026-05-11 | Initial `require-static` policy adopted as the release gate; `require-baseline` introduced as a strict ratchet. |
| 2026-05-15 | Audit flagged ambiguity (C1): `audit-release-check-strict` Makefile target name suggested it was a release-blocker, but `make_evidence.py --release` only ran `require-static`. |
| 2026-05-16 | Audit C1 closure — renamed strict target to `audit-release-check-baseline-optional`, added `RELEASE_POLICY.md` (this file) as canonical source of truth, added `bench-attached` ratchet as the bridge between static-only and full-baseline coverage. |
| 2026-05-16 | Audit R-01 + R-02 + R-04 closure — introduced `vllm.sndr_core.proof.production_subset`, `scripts/attach_bench_proof.py --include-default-on`, `sndr patches release-check --scope production-subset`. First bench cycle against `prod-qwen3.6-27b-dflash-multiconc` brought 81/169 patches (47.9%) to `bench_with_baseline`. R-04 triage promoted 37 family-contract-covered patches from `review_required` to `eligible`, leaving 39 honest gaps (33 experimental opt-in outside subset, 6 genuine in-subset test gaps documented above). |

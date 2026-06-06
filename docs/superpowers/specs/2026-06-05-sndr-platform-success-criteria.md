# sndr-platform Refactor — Success Criteria

**Date**: 2026-06-05
**Owner**: Sander Barzov
**Related**: [Master Spec](2026-06-05-sndr-platform-master-spec.md), [ADR-001](../../_adr/0001-multi-engine-refactor.md)

This document defines **measurable** success criteria for the sndr-platform
refactor. Each phase has its own quality gates, but these are the criteria for
the refactor as a whole.

The refactor is considered **complete and successful** when ALL of the following
are true.

## 1. Architecture criteria

- [ ] `sndr/` package exists at top level of repository
- [ ] `vllm/sndr_core/` is empty or removed (Phase 11 cleanup)
- [ ] `sndr/engines/vllm/` contains the vllm adapter with all 252 community
      patches migrated
- [ ] `sndr/engines/sglang/` exists as skeleton with adapter ABC stub and
      README documenting how to port patches
- [ ] `sndr/kernel/` contains engine-agnostic primitives only (no `from vllm`
      imports)
- [ ] `sndr/dispatcher/` uses `EngineAdapter` ABC, not direct vllm imports
- [ ] Layered architecture rules enforced by
      `tools/ci/verify_layer_imports.py`
- [ ] `tools/ci/verify_no_engine_leak.py` blocks engine-tier patches in public
      registry

## 2. Code quality criteria

- [ ] Coverage: ≥ 85% on `sndr/kernel/`, `sndr/dispatcher/`, `sndr/license.py`
- [ ] Coverage: ≥ 80% on `sndr/engines/vllm/adapter.py`
- [ ] Coverage: ≥ 70% on GUI feature modules
- [ ] Ruff: 0 warnings on `sndr/` and `tools/`
- [ ] Mypy: 0 errors on `sndr/` (strict mode)
- [ ] ESLint: 0 warnings on `gui/web/src/`
- [ ] TypeScript: strict mode, no `any` types except documented exceptions

## 3. Performance criteria

- [ ] Boot duration P95 < 5 seconds (measured by
      `sndr_boot_duration_seconds`)
- [ ] Patch apply P95 < 50 ms per patch
- [ ] API response P95 < 200 ms (non-streaming endpoints)
- [ ] License verification < 100 ms
- [ ] GUI main bundle ≤ 350kb gzipped
- [ ] GUI cold page load P95 < 2 seconds
- [ ] GUI warm page load P95 < 500 ms
- [ ] Bench TPS regression < 2% per model
  - 27B Lorbus INT4: ≥ 115 TPS (Genesis stack)
  - 35B-A3B FP8: ≥ 207 TPS (Genesis stack)

## 4. Operational criteria

- [ ] `sndr drift check` works automatically (daily CI cron)
- [ ] `sndr pin upgrade --to X` works as single command
- [ ] Per-pin manifests exist for at least 3 pins:
  - `0.21.1.dev354_626fa9bba/`
  - `0.22.0/`
  - `0.22.1_da1daf40b/`
- [ ] License Ed25519 verification preserves backward compatibility with
      existing customer tokens
- [ ] Production 35B + 27B containers can be deployed via universal launcher
- [ ] Old launcher scripts work as deprecated alternates for 1 release cycle

## 5. GUI criteria

- [ ] Carbon Design System (g100 theme) adopted
- [ ] IBM Plex fonts loaded
- [ ] App.tsx ≤ 200 LOC (thin shell)
- [ ] 24 feature modules in `gui/web/src/features/`, each ≤ 300 LOC
- [ ] Engine selector dropdown works
- [ ] Pin manager view shows current + previous + staging
- [ ] Drift dashboard shows per-patch drift status
- [ ] Lingui i18n covers 100% of UI strings in en + ru
- [ ] Accessibility: WCAG 2.1 AA pass (axe-core in CI)
- [ ] Bundle size budget enforced in CI

## 6. Documentation criteria

- [ ] `docs/README.md` updated
- [ ] `docs/QUICKSTART.md` written (5-minute guide)
- [ ] `docs/concepts/ENGINES.md` written
- [ ] `docs/concepts/PINS.md` written
- [ ] `docs/concepts/PATCHES.md` written
- [ ] `docs/concepts/DRIFT.md` written
- [ ] `docs/concepts/LICENSING.md` written
- [ ] `docs/guides/PATCH_AUTHORING.md` written
- [ ] `docs/guides/ENGINE_ADAPTER.md` written
- [ ] `docs/guides/PIN_UPGRADE.md` written
- [ ] `docs/guides/COMMERCIAL_TIER.md` written
- [ ] `docs/reference/API.md` auto-generated from OpenAPI
- [ ] `docs/reference/CLI.md` complete
- [ ] `docs/reference/ENV_VARS.md` complete
- [ ] `docs/reference/ERRORS.md` catalog of all error codes
- [ ] `docs/changelog/v12.0.0.md` written
- [ ] At least 10 ADRs written for major decisions

## 7. Security criteria

- [ ] All endpoints behind auth middleware (verified by integration tests)
- [ ] Bandit: 0 high-severity issues
- [ ] pip-audit: 0 known vulnerabilities
- [ ] CodeQL: 0 critical findings
- [ ] License tokens never logged (verified by grep + manual review)
- [ ] Audit log writes for all sensitive actions
- [ ] CSP headers + X-Frame-Options + HSTS configured

## 8. Testing criteria

- [ ] 60% of tests are unit tests (fast, isolated)
- [ ] 30% are integration tests (pin matrix, apply chains)
- [ ] 10% are E2E tests (Playwright critical paths)
- [ ] Pin matrix tests pass on all 3 baseline pins
- [ ] GUI E2E tests pass for: smoke, auth, patches view, pin upgrade
- [ ] CI runs all of the above on every PR

## 9. Migration criteria

- [ ] Existing operators can upgrade by changing only mount path + env var
- [ ] Backward compatibility shims work for 1 release (verified by tests)
- [ ] Production cut-over (Phase 8) completed with no downtime
- [ ] Apply matrix identical before/after Phase 4 (modulo intentional fixes)
- [ ] Customer license tokens continue to work

## 10. Release criteria (v12.0.0)

- [ ] All above criteria checked
- [ ] CHANGELOG.md entry written
- [ ] All ADRs accepted (no "proposed" status)
- [ ] PyPI release published (`sndr` package)
- [ ] Docker Hub release published (`sndr/control-center`)
- [ ] GitHub release created with notes
- [ ] Migration guide for v11 users
- [ ] No known critical bugs
- [ ] Maintainer sign-off

## How we measure

| Criterion type | Measurement |
|---|---|
| Architecture | CI scripts (verify_layer_imports.py, etc.) |
| Code quality | Ruff + Mypy + Bandit + coverage report |
| Performance | Prometheus metrics + GitHub Actions benchmarks |
| Operational | Manual smoke tests + scripted checks |
| GUI | Lighthouse + axe-core + Playwright |
| Documentation | Diff against template; manual review |
| Security | bandit + pip-audit + CodeQL |
| Testing | pytest --cov + Vitest --coverage + Playwright |
| Migration | Per-phase rollback drills |
| Release | Manual sign-off after checklist |

## Tracking

Status of each criterion is tracked in the [Execution Journal](../journal/2026-06-05-sndr-platform-execution-log.md).

When a criterion is met, the corresponding checkbox in this document is marked
in a journal entry referencing the PR that satisfied it.

## Stretch criteria (nice-to-have, not required for v12.0.0)

- [ ] sglang adapter has at least 1 working patch ported (demonstrates
      pluggability)
- [ ] Bench tool publishes results to a queryable history dashboard
- [ ] OpenTelemetry traces exported to a real backend (Jaeger or Tempo)
- [ ] Prometheus metrics scraped by a real Prometheus instance
- [ ] GUI dark/light theme toggle (Carbon supports it natively)
- [ ] Tauri desktop wrapper as experimental release artifact

# SPDX-License-Identifier: Apache-2.0
"""Metadata overlay for PATCH_REGISTRY (audit P1-2 closure, 2026-05-12).

Why
---
The registry in `registry.py` holds 136 entries, and many metadata
fields (implementation_status, test_status, production_default) are
identical across whole groups of patches. To avoid duplicating them
in every entry, this overlay declares the groups:

  - All `lifecycle=stable` → `implementation_status=full`,
    `production_default=eligible`.
  - All `lifecycle=experimental` + `default_on=True` → `full`,
    `eligible`. With default_on=False the impl is usually still
    `full`, but treated more cautiously.
  - `lifecycle=legacy` → `live` (pre-dispatcher, working),
    `eligible`.
  - `lifecycle=retired` → `retired`, `blocked`.
  - `lifecycle=research` → `research`, `research_only`.

Explicit overrides per-patch are layered on top (e.g.
PN95.implementation_status=partial — wiring incomplete).

API
---

- `derive_metadata(patch_id, registry_meta)` → dict with fields
  `implementation_status`, `test_status`, `production_default`.
- `EXPLICIT_OVERRIDES`: dict[patch_id, dict] — pinpoint exceptions.

Related
-------

- `dispatcher/spec.py::infer_implementation_status` (old inference,
  now delegates here).
- `cli/patches.py::_run_plan --profile production` (uses
  production_default for blocking).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = REPO_ROOT / "tests" / "unit" / "integrations"


ImplStatus = Literal[
    "full", "partial", "scaffold", "placeholder",
    "retired", "research", "live", "coordinator",
]
TestStatus = Literal["unit", "integration", "bench", "none"]
# Phase 0.3 (audit 2026-05-12): `review_required` added for patches
# whose impl_status=stable/full/live but test_status=none. Previously
# such patches received `eligible` (production-ready), which overstated
# readiness and could let untested code slip into production.
ProductionDefault = Literal[
    "eligible", "blocked", "research_only", "review_required",
]


class DerivedMetadata(TypedDict):
    implementation_status: ImplStatus
    test_status: TestStatus
    production_default: ProductionDefault


# Phase 0.3: single mapping impl_status × test_status → production_default.
# Previously this rule was scattered across 6 branches of derive_metadata;
# now a single function is the source of truth, and tests cover it independently.
#
# M.1.1.T0 (2026-05-27): constants centralised in
# ``dispatcher/_constants.py``; re-imported under the historical names.
from ._constants import (  # noqa: F401
    _BLOCKED_STATUSES,
    _RESEARCH_STATUSES,
)


def _production_default_for(
    impl_status: str, test_status: str,
) -> ProductionDefault:
    """Compute production_default from (implementation_status, test_status).

    Rules:
      - partial/placeholder/retired → blocked (known broken / stale)
      - research → research_only (requires an explicit research flag)
      - everything else (full/live/scaffold/coordinator):
          - test_status=none → review_required (needs test coverage or
            an audited override via EXPLICIT_OVERRIDES)
          - otherwise → eligible
    """
    if impl_status in _BLOCKED_STATUSES:
        return "blocked"
    if impl_status in _RESEARCH_STATUSES:
        return "research_only"
    if test_status == "none":
        return "review_required"
    return "eligible"


# Audit closure (2026-05-16): explicit overrides for default-on patches
# without a discoverable per-patch unit-test file. Each entry below documents
# why the patch is production-eligible despite the file-based test
# discovery returning `test_status="none"`. The override fields are
# audited per-patch — they MUST point to a real evidence artefact
# (upstream PR, bench baseline JSON, family-contract coverage, etc).
#
# Three closure categories below:
#
#  1. wave10_backport — upstream PRs merged into vllm dev371. Validated
#     via tests/integration/baselines/{27b,35b}_v11_wave9.json + matching
#     proof artefacts in docs/proofs/. Test coverage is integration-tier
#     (full bench cycle) rather than per-patch unit-tier.
#
#  2. legacy_pre_dispatcher — P1-P46 era patches written before the
#     dispatcher/registry layer existed. They're covered by
#     tests/integration/test_patch_regression_bounds.py (TPS/CV bounds
#     enforced against the v11_wave9 baselines) plus the family-contract
#     factories in tests/unit/integrations/<family>/. Each patch has a
#     bench-history attestation in docs/proofs/ documenting the Wave at
#     which it was validated.
#
#  3. marker_only_advisory — registry markers (no runtime wiring); they
#     exist to record decisions/tunings made elsewhere (start-script env,
#     vllm CLI flag, kernel autotune). Test coverage is the registry
#     contract test itself — there's no executable Genesis code path to
#     unit-test.
EXPLICIT_OVERRIDES: dict[str, DerivedMetadata] = {
    # ── Known-partial wiring ─────────────────────────────────────────
    "PN95": {
        "implementation_status": "partial",
        "test_status": "unit",
        "production_default": "blocked",
    },
    "PN64": {
        # Marlin MoE SM 12.0 placeholder — no real tuning data yet.
        "implementation_status": "placeholder",
        "test_status": "none",
        "production_default": "blocked",
    },
    "PN26b": {
        # Sparse-V research kernel — code exists, no production
        # validation on Ampere.
        "implementation_status": "scaffold",
        "test_status": "unit",
        "production_default": "research_only",
    },
    # Coordinator-only entries (no actual wiring file):
    "P5b": {
        "implementation_status": "coordinator",
        "test_status": "unit",
        "production_default": "eligible",
    },

    # ── Wave 10 backports — upstream merged, integration-tested ──────
    # All entries below are validated against the v11_wave9 bench
    # baseline + carry a docs/proofs/ artefact citing the upstream PR.
    "PN96b": {
        "implementation_status": "full",
        "test_status": "integration",  # bench cycle + family contract
        "production_default": "eligible",
    },
    "P108": {
        "implementation_status": "full",
        "test_status": "integration",  # vllm#42603 + bench validation
        "production_default": "eligible",
    },
    "P109": {
        "implementation_status": "full",
        "test_status": "integration",  # vllm#42614 + bench validation
        "production_default": "eligible",
    },
    "PN110": {
        "implementation_status": "full",
        "test_status": "integration",  # vllm#42615 + bench validation
        "production_default": "eligible",
    },
    "PN116": {
        "implementation_status": "full",
        "test_status": "integration",  # TQ prefill — bench + manual A/B
        "production_default": "eligible",
    },
    "PN118": {
        "implementation_status": "full",
        "test_status": "integration",  # vllm#42551 — bench validation
        "production_default": "eligible",
    },
    "PN119": {
        "implementation_status": "full",
        "test_status": "integration",  # vllm#40792 GQA grouping
        "production_default": "eligible",
    },

    # ── stable-lifecycle backports without per-patch unit test ───────
    # Covered by family contracts + bench regression bounds.
    "PN33": {
        "implementation_status": "full",
        "test_status": "integration",  # spec-decode warmup K-aware
        "production_default": "eligible",
    },
    "PN35": {
        "implementation_status": "full",
        "test_status": "integration",  # vllm#35 inputs_embeds skip
        "production_default": "eligible",
    },

    # ── Legacy pre-dispatcher era (P1-P46) — bench-history attested ──
    # These predate the dispatcher/registry layer. Coverage is provided
    # by tests/integration/test_patch_regression_bounds.py (TPS/CV bounds
    # against v11_wave9 baselines) + family-contract factories.
    "P3":   {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P4":   {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P5":   {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P6":   {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P7":   {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P14":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P15":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P22":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P24":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P26":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P27":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P28":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P31":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P34":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P36":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P38":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P39a": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P44":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P46":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},

    # ── Marker-only advisory entries (registry annotations only) ─────
    # No runtime Genesis code — these record upstream-CLI flags, env
    # toggles or autotune knobs. Coverage = registry contract test.
    "P1":   {"implementation_status": "live", "test_status": "unit", "production_default": "eligible"},
    "P17":  {"implementation_status": "live", "test_status": "unit", "production_default": "eligible"},
    "P18b": {"implementation_status": "live", "test_status": "unit", "production_default": "eligible"},
    "P20":  {"implementation_status": "live", "test_status": "unit", "production_default": "eligible"},
    "P23":  {"implementation_status": "live", "test_status": "unit", "production_default": "eligible"},
    "P29":  {"implementation_status": "live", "test_status": "unit", "production_default": "eligible"},
    "P32":  {"implementation_status": "live", "test_status": "unit", "production_default": "eligible"},
    "PN60": {"implementation_status": "live", "test_status": "unit", "production_default": "eligible"},
    "PN63": {"implementation_status": "live", "test_status": "unit", "production_default": "eligible"},

    # ── R-04 closure (audit 2026-05-16): patches in the production
    # subset (any ``prod-*`` preset enables them) that are covered by
    # their family-contract auto-discovery test in
    # ``tests/unit/integrations/<family>/test_<family>_family_contract.py``.
    # The contract verifies per-patch invariants — anchor exists,
    # marker present in source, env_flag references resolve, no
    # top-level torch import, family field matches registry. That is
    # honest integration-level coverage even when no dedicated
    # ``test_pNN_*.py`` file exists. ``_file_based_test_status``
    # only sees dedicated files, so without the override these would
    # silently degrade to ``review_required``. Operators reading
    # release-readiness dashboards saw a misleading "0 tests" signal.
    "P37":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P61b": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P62":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P64":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P66":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P68":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P69":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P70":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P72":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P74":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P81":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "P91":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN8":  {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN12": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN21": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN22": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN23": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN24": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN38": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN40": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN67": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN71": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN73": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN77": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN91": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN92": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN96": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN106": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN125": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN126": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN127": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN128": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN129": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN130": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN132": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "PN133": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
    "SNDR_WORKSPACE_001": {"implementation_status": "full", "test_status": "integration", "production_default": "eligible"},
}


def _file_based_test_status(patch_id: str, family: str = "") -> TestStatus:
    """Best-effort: look for `tests/unit/integrations/<family>/test_<id>_*.py`
    or `tests/legacy/test_<id>*.py`. Return `unit` if found, otherwise
    `none`. Integration / bench tiers require a manual override via
    EXPLICIT_OVERRIDES.
    """
    pid_lower = patch_id.lower()
    # Direct hit in integrations
    if family:
        fam_dir = TESTS_DIR / family.replace(".", "/")
        if fam_dir.is_dir():
            for f in fam_dir.rglob(f"test_{pid_lower}_*.py"):
                if f.is_file():
                    return "unit"
            for f in fam_dir.rglob(f"test_{pid_lower}.py"):
                if f.is_file():
                    return "unit"
    # Global search in integrations/
    if TESTS_DIR.is_dir():
        for f in TESTS_DIR.rglob(f"test_{pid_lower}_*.py"):
            return "unit"
        for f in TESTS_DIR.rglob(f"test_{pid_lower}.py"):
            return "unit"
    # Legacy bucket
    legacy = REPO_ROOT / "tests" / "legacy"
    if legacy.is_dir():
        for f in legacy.rglob(f"test_{pid_lower}*.py"):
            return "unit"
        # Numeric forms (test_pn33_* vs test_pN33_*)
        for f in legacy.rglob(f"test_p{pid_lower.lstrip('p')}*.py"):
            return "unit"
    return "none"


_LIFECYCLE_TO_IMPL: dict[str, ImplStatus] = {
    "retired":     "retired",
    "deprecated":  "retired",
    "research":    "research",
    "stable":      "full",
    "coordinator": "coordinator",
    "legacy":      "live",
    # experimental / unknown → fallback `live` (see derive_metadata).
}


def derive_metadata(
    patch_id: str, registry_meta: dict,
) -> DerivedMetadata:
    """Return derived metadata for one patch.

    Resolution order:

      1. EXPLICIT_OVERRIDES — audited per-patch escape hatches.
      2. Lifecycle hard rules — retired/deprecated short-circuit to
         ``production_default=blocked`` regardless of registry
         ``implementation_status``. A retired patch must not load even
         if its wiring file still reports ``full`` impl status.
      3. Research lifecycle hard rule — research patches always map to
         ``production_default=research_only`` regardless of explicit
         ``implementation_status``. See note below.
      4. Registry ``implementation_status`` (when explicitly set).
         test_status + production_default flow through
         ``_production_default_for``.
      5. Lifecycle-based fallback (same routing through
         ``_production_default_for``).

    Audit R-01 closure (2026-05-16): research lifecycle now short-
    circuits to research_only. Previously a research patch with
    ``implementation_status=full`` and a unit test (P82/P83) derived
    to ``production_default=eligible``, which misled production-
    readiness dashboards. Research code is not "production
    candidate" by definition — eligibility requires the experimental
    or stable lifecycle. EXPLICIT_OVERRIDES still wins above this
    rule (audited escape hatch).

    Audit C5 closure (2026-05-16): retired/deprecated lifecycle now
    short-circuits to blocked. Previously a retired patch with
    ``implementation_status=full`` ended up as ``review_required``
    because the inference looked at impl_status before lifecycle —
    that wrongly suggested the retired wiring was still production-
    eligible. Tests in tests/unit/dispatcher/ cover this contract.

    Etap 0.3 (2026-05-12): production_default takes test_status into
    account. Stable/full/live patches without tests get
    ``review_required`` instead of ``eligible`` so an untested
    overlay cannot silently slip into the production matrix.
    """
    # 1. Explicit override (point-of-truth, audited)
    override = EXPLICIT_OVERRIDES.get(patch_id)
    if override is not None:
        return override

    test = _file_based_test_status(
        patch_id, str(registry_meta.get("family", "")),
    )

    # 2. Lifecycle hard rules — retired/deprecated always block.
    lc = str(registry_meta.get("lifecycle", "")).lower()
    if lc in ("retired", "deprecated"):
        return {
            "implementation_status": "retired",
            "test_status": test,
            "production_default": "blocked",
        }

    # 3. Lifecycle=research is a hard rule too — it must NEVER derive
    # to production_default=eligible, even when the registry declares
    # implementation_status="full" or unit tests exist. Research code
    # may be runtime-complete (impl_status=full is a factual statement
    # about the code path), but it has not been validated as production
    # candidate. Reporting downstream of derive_metadata() previously
    # showed P82/P83 (lifecycle=research, impl=full) as eligible, which
    # misled production-readiness dashboards. EXPLICIT_OVERRIDES still
    # wins above this rule (audited escape hatch); everything else
    # flowing through derive_metadata respects research as terminal.
    if lc == "research":
        return {
            "implementation_status": "research",
            "test_status": test,
            "production_default": "research_only",
        }

    # 4. Registry already specifies an implementation_status — honour it.
    explicit = registry_meta.get("implementation_status")
    if isinstance(explicit, str) and explicit:
        return {
            "implementation_status": explicit,  # type: ignore[typeddict-item]
            "test_status": test,
            "production_default": _production_default_for(explicit, test),
        }

    # 5. Lifecycle-based fallback.
    impl: ImplStatus = _LIFECYCLE_TO_IMPL.get(lc, "live")
    return {
        "implementation_status": impl,
        "test_status": test,
        "production_default": _production_default_for(impl, test),
    }


__all__ = [
    "EXPLICIT_OVERRIDES",
    "DerivedMetadata",
    "ImplStatus",
    "TestStatus",
    "ProductionDefault",
    "derive_metadata",
]

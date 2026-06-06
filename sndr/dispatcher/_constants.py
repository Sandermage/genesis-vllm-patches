# SPDX-License-Identifier: Apache-2.0
"""Dispatcher validator constants — single source of truth.

M.1.1.T0 (2026-05-27): centralized the module-private constants
previously inlined in ``dispatcher/audit.py`` and
``dispatcher/registry_metadata.py``. Pure data; no behavior change.
Each consumer module re-imports the original name via
``from ._constants import <NAME>``.

Why this exists
───────────────
The audit + metadata layers both classify ``PATCH_REGISTRY`` entries
along the same axes (lifecycle, implementation status, env-flag
prefixes). When those enumerations drifted across the two modules in
the past, the only signal was test failure deep inside the audit
script. Centralising the constants here makes the policy enumerations
greppable in one place and lets future audits import them without
reaching into another module's private namespace.

Scope discipline
────────────────
- ONLY pure-data constants belong here.
- No functions, no dataclasses, no logic.
- Constants stay module-private (leading underscore) so they do NOT
  enter ``dispatcher/__init__.py`` ``__all__``. Consumers go through
  their original module's import alias for back-compat with any
  external tooling that reaches into private attributes.
"""
from __future__ import annotations


# ─── Tier (audit closure 2026-05-08, Stage 5) ─────────────────────────
# Patches are tagged as either ``community`` (Apache wheel) or
# ``engine`` (commercial overlay). Anything else is a registry typo.
_VALID_TIERS = frozenset({"community", "engine"})


# ─── Lifecycle (operational role of a patch) ──────────────────────────
_VALID_LIFECYCLES = frozenset({
    "stable",          # production-ready, default-on or operator-controlled
    "experimental",    # opt-in only; behavior may change
    "deprecated",      # empirically disproven; kept for reproducibility
    "legacy",          # default-on but operator-controllable via SNDR_LEGACY_*
    "research",        # research-only artifact; never default-on
    "merged_upstream", # self-retiring marker (vllm has the fix natively now)
    "retired",         # superseded by another patch; kept as historical alias
    "coordinator",     # umbrella entry that orchestrates sub-patches (e.g. P5b)
})


# ─── Implementation status (P2-1, audit 2026-05-08) ───────────────────
# Orthogonal axis to lifecycle. `lifecycle` answers "what's the
# operational role of this patch?", whereas `implementation_status`
# answers "how complete is the actual code?". A patch can be
# `lifecycle="stable"` AND `implementation_status="full"` (the boring
# happy path), but it can also be `lifecycle="experimental",
# implementation_status="marker_only"` (PN62 today: marker is set, no
# downstream code reads it yet). The new field surfaces in CLI output
# (`sndr patches`) and lets production presets refuse to enable
# `marker_only` / `placeholder` patches.
_VALID_IMPLEMENTATION_STATUSES = frozenset({
    "full",          # implementation complete; tests cover the fast path
    "partial",       # main path implemented; some sub-features stubbed
    "marker_only",   # boot-time marker set, no downstream consumer (e.g. PN62)
    "placeholder",   # registry entry exists; impl pending future PR
    "experimental",  # impl exists but lacks A/B / cross-rig validation
    "retired",       # impl moved to archive; only stub remains
    # Note: `unknown` is the implicit default for entries that don't yet
    # set this field; the validator emits an INFO suggesting one of
    # the explicit values.
})


# ─── Canonical env_flag prefixes (recognized by runtime + env.py) ─────
# env.py knows both Sander-IP and community brands. Semantic prefix
# groups:
#   *_ENABLE_  — gate opt-in; default OFF
#   *_DISABLE_ — opt-out for default-on legacy patches
#   *_LEGACY_  — restore legacy behavior on default-on flips
#   *_ALLOW_   — gate a feature that is otherwise blocked by policy;
#                semantically distinct from ENABLE_ (operator
#                permission, not feature switch) — used by coordinator
#                patches like PN274. Added 2026-05-22 (Phase 3A.6) to
#                close the false-positive doctor warning on PN274's
#                SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER.
# Anything outside these prefix groups surfaces as a WARNING.
_CANONICAL_ENV_PREFIXES = (
    "SNDR_ENABLE_", "GENESIS_ENABLE_",
    "SNDR_DISABLE_", "GENESIS_DISABLE_",
    "SNDR_LEGACY_", "GENESIS_LEGACY_",
    "SNDR_ALLOW_", "GENESIS_ALLOW_",
    # Info-marker semantic (Phase 10.5 2026-06-01): operator-visible
    # flag that documents an external state (e.g. G4_T1
    # GENESIS_INFO_G4_T1_PR42006_OVERLAY_MOUNTED — reports whether
    # the operator has bind-mounted the vendored upstream tool-call
    # overlay file). No toggle semantics, no boot-time apply gate.
    # Distinct from ENABLE/DISABLE because the operator does not
    # "enable" the overlay — it is either bind-mounted at container
    # launch or it is not, and this flag surfaces that condition
    # via the audit/explain tooling without implying patch control.
    "SNDR_INFO_", "GENESIS_INFO_",
)


# ─── Production-default status buckets (Etap 0.3) ─────────────────────
# Used by ``registry_metadata._production_default_for`` to derive a
# per-entry ``production_default`` field from the pair
# ``(implementation_status, test_status)``. Previously inlined in
# ``registry_metadata.py``; centralised here so the audit + derive
# layers share the same enumeration.
_BLOCKED_STATUSES = frozenset({"partial", "placeholder", "retired"})
_RESEARCH_STATUSES = frozenset({"research"})

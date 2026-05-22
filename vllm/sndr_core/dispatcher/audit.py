# SPDX-License-Identifier: Apache-2.0
"""SNDR Core dispatcher — registry validation + audit.

Boot-time invariant checks for PATCH_REGISTRY:
  - Every patch_id is unique
  - requires_patches references valid patch_ids (no cycles)
  - conflicts_with bidirectional + valid
  - env_flag in canonical form
  - lifecycle field valid
  - applies_to schema valid

Used by `apply/orchestrator` at boot to fail fast on registry typos.
Also exposed via CLI `sndr audit` for offline review.

Migration history:
  - Original location: vllm/_genesis/dispatcher.py (Stage 0).
  - Stage 3 (CURRENT): split into dispatcher/audit.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .registry import PATCH_REGISTRY  # noqa: F401  (re-exported)
from .spec import VALID_UPSTREAM_PR_RELATIONSHIPS

log = logging.getLogger("genesis.dispatcher")


def _live_registry() -> dict[str, dict[str, Any]]:
    """Resolve the registry through the canonical SNDR Core dispatcher.

    PR38 cleanup (2026-05-08): `vllm.sndr_core.dispatcher.__init__.py`
    re-exports `PATCH_REGISTRY` from `.registry` at package level.
    Tests now monkey-patch the canonical package directly:

        monkeypatch.setattr(
            vllm.sndr_core.dispatcher, "PATCH_REGISTRY", fake_registry,
        )

    `_live_registry()` reads the same attribute on the same package
    module, so the patch propagates without going through any legacy
    shim. The previous Stage 3 indirection through `_genesis.dispatcher`
    is gone now that `_genesis/` is being removed.
    """
    from vllm.sndr_core import dispatcher as _canonical
    return _canonical.PATCH_REGISTRY


# ─── Validation issue dataclass ──────────────────────────────────────────
# Reported by validate_registry() and validate_apply_plan(). Severity
# levels: "ERROR" (operator must fix), "WARNING" (likely-wrong, allow boot
# to proceed), "INFO" (informational only).

@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "ERROR" | "WARNING" | "INFO"
    patch_id: str
    message: str


def _coerce_list(value: Any) -> list[str]:
    """Normalize a metadata field into list[str]. Tolerates None / scalar."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return []


# Canonical enum-like sets for registry-field validation. Keep in sync with
# the registry.py docstring. Adding a new tier/lifecycle? Update both.
_VALID_TIERS = frozenset({"community", "engine"})
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

# P2-1 (audit 2026-05-08): orthogonal axis to lifecycle. `lifecycle`
# answers "what's the operational role of this patch?", whereas
# `implementation_status` answers "how complete is the actual code?".
# A patch can be `lifecycle="stable"` AND `implementation_status="full"`
# (the boring happy path), but it can also be `lifecycle="experimental",
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

# Canonical env_flag prefixes recognized by the runtime (env.py knows
# both Sander-IP and community brands). Semantic prefix groups:
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
)


def _is_canonical_env_flag(flag: str) -> bool:
    """Registry env_flag must use one of the canonical full-prefix forms.

    See env.py for the alias logic — SNDR_* takes precedence over
    GENESIS_* for the same suffix. Both prefixes work. ALLOW_ is
    semantically distinct from ENABLE_ (operator permission gate vs
    feature switch); both are recognized.
    """
    return any(flag.startswith(p) for p in _CANONICAL_ENV_PREFIXES)


def validate_registry(
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Static validation of PATCH_REGISTRY shape.

    F-009 expansion (audit 2026-05-07): now covers every contract field
    the registry docstring promises, not just requires/conflicts graph.

    Per-entry checks:
      - `tier` is one of {community, engine}
      - `lifecycle` is one of the canonical set (see `_VALID_LIFECYCLES`)
      - `env_flag` uses canonical full-prefix form (SNDR_ENABLE_/GENESIS_ENABLE_)
      - `apply_module` (when present) is dotted-path import-resolvable
      - `applies_to` shape is dict-or-absent (loose schema; the runtime
        gate in `_check_applies_to` does the field-level checks)

    Cross-entry checks:
      - `requires_patches` references valid patch_ids (no self-ref, no cycle)
      - `conflicts_with` references valid patch_ids (no self-ref)

    Returns a list of `ValidationIssue` (empty list = clean).
    """
    if registry is None:
        registry = _live_registry()

    issues: list[ValidationIssue] = []
    keys = set(registry.keys())

    # 1. Per-entry contract checks (tier, lifecycle, env_flag, applies_to)
    for pid, meta in registry.items():
        # tier
        tier = meta.get("tier")
        if tier is not None and tier not in _VALID_TIERS:
            issues.append(ValidationIssue(
                "ERROR", pid,
                f"tier={tier!r} is not in {sorted(_VALID_TIERS)}",
            ))

        # lifecycle
        lifecycle = meta.get("lifecycle")
        if lifecycle is not None and lifecycle not in _VALID_LIFECYCLES:
            issues.append(ValidationIssue(
                "ERROR", pid,
                f"lifecycle={lifecycle!r} is not in {sorted(_VALID_LIFECYCLES)}",
            ))
        # PR38 §5.5 ratchet (2026-05-08): patches without an explicit
        # lifecycle drift into ambiguity over time. Surface as INFO so
        # the registry self-documents which patches need a decision.
        # Not raised to WARNING because 91 patches today have no
        # lifecycle and we don't want to drown out real issues.
        if lifecycle is None:
            issues.append(ValidationIssue(
                "INFO", pid,
                "lifecycle field unset — pick one of "
                f"{sorted(_VALID_LIFECYCLES)}. Promoting to "
                "lifecycle='stable' triggers anchor manifest "
                "requirements; see "
                "docs/upstream/STABLE_PROMOTION_CHECKLIST.md.",
            ))

        # P2-1 (audit 2026-05-08): implementation_status validation.
        impl_status = meta.get("implementation_status")
        if (impl_status is not None
                and impl_status not in _VALID_IMPLEMENTATION_STATUSES):
            issues.append(ValidationIssue(
                "ERROR", pid,
                f"implementation_status={impl_status!r} is not in "
                f"{sorted(_VALID_IMPLEMENTATION_STATUSES)}",
            ))

        # Phase 5.1.C (2026-05-22): upstream_pr_relationship enum check.
        # After the 5.1.B migration every upstream_pr-bearing entry
        # carries an explicit relationship value, so missing-when-set
        # is now an ERROR (escalated from silent in 5.1.A). When the
        # field is present it MUST be one of the canonical values.
        # The reverse case (relationship set without upstream_pr) stays
        # WARNING — likely a copy-paste mistake but not fatal.
        rel = meta.get("upstream_pr_relationship")
        upstream_pr_value = meta.get("upstream_pr")
        if rel is not None and rel not in VALID_UPSTREAM_PR_RELATIONSHIPS:
            issues.append(ValidationIssue(
                "ERROR", pid,
                f"upstream_pr_relationship={rel!r} is not in "
                f"{sorted(VALID_UPSTREAM_PR_RELATIONSHIPS)}",
            ))
        if rel is None and isinstance(upstream_pr_value, int):
            issues.append(ValidationIssue(
                "ERROR", pid,
                f"upstream_pr is set (#{upstream_pr_value}) but "
                f"upstream_pr_relationship is missing — pick one of "
                f"{sorted(VALID_UPSTREAM_PR_RELATIONSHIPS)}. Default "
                f"choice for plain backports is 'backport'.",
            ))
        if rel is not None and upstream_pr_value is None:
            issues.append(ValidationIssue(
                "WARNING", pid,
                f"upstream_pr_relationship={rel!r} is set but "
                f"upstream_pr is None — relationship field has no "
                f"target; either set upstream_pr or remove the "
                f"relationship field",
            ))

        # env_flag canonical form. WARNING (not ERROR) because the
        # runtime decision now strips the prefix and delegates to
        # env.is_enabled — so the registry can be drift-fixed gradually
        # without breaking apply behavior.
        env_flag = meta.get("env_flag")
        if env_flag and not _is_canonical_env_flag(env_flag):
            issues.append(ValidationIssue(
                "WARNING", pid,
                f"env_flag={env_flag!r} lacks canonical SNDR_ENABLE_/"
                f"GENESIS_ENABLE_ prefix — operators may not realize "
                f"the alias works",
            ))

        # apply_module (optional today; will become required when the
        # parking-lot _per_patch_dispatch.py is retired). When present,
        # must import-resolve so we fail fast on typo'd paths.
        apply_module = meta.get("apply_module")
        if apply_module:
            try:
                import importlib
                importlib.import_module(apply_module)
            except Exception as e:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"apply_module={apply_module!r} fails to import: "
                    f"{type(e).__name__}: {e}",
                ))

        # applies_to: dict or absent. Field-level type checks happen at
        # _check_applies_to call time; this layer only catches "someone
        # accidentally wrote a list/string here".
        applies_to = meta.get("applies_to")
        if applies_to is not None and not isinstance(applies_to, dict):
            issues.append(ValidationIssue(
                "ERROR", pid,
                f"applies_to must be dict or absent, got "
                f"{type(applies_to).__name__}",
            ))

    # 2. Reference existence + self-reference (graph layer)
    for pid, meta in registry.items():
        for ref in _coerce_list(meta.get("requires_patches")):
            if ref == pid:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"requires_patches contains self-reference {ref!r}",
                ))
            elif ref not in keys:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"requires_patches references unknown patch_id {ref!r}",
                ))
        for ref in _coerce_list(meta.get("conflicts_with")):
            if ref == pid:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"conflicts_with contains self-reference {ref!r}",
                ))
            elif ref not in keys:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"conflicts_with references unknown patch_id {ref!r}",
                ))

    # 3. Cycle detection on requires_patches graph (DFS three-color).
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {pid: WHITE for pid in registry}

    def _walk(pid: str, path: list[str]) -> None:
        if color[pid] == GRAY:
            cycle = path[path.index(pid):] + [pid]
            issues.append(ValidationIssue(
                "ERROR", pid,
                f"requires_patches cycle detected: {' → '.join(cycle)}",
            ))
            return
        if color[pid] == BLACK:
            return
        color[pid] = GRAY
        for ref in _coerce_list(registry.get(pid, {}).get("requires_patches")):
            if ref in color:
                _walk(ref, path + [pid])
        color[pid] = BLACK

    for pid in list(registry):
        if color[pid] == WHITE:
            _walk(pid, [])

    return issues


def validate_apply_plan(
    applied: set[str],
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[ValidationIssue]:
    """Runtime validation: given the live APPLY set, surface dependency /
    conflict violations.

    Args:
        applied: set of patch_ids that the dispatcher actually decided to
            APPLY this boot (from `get_apply_matrix()` filtered by
            applied=True, or computed externally).
        registry: optional override for testing; defaults to PATCH_REGISTRY.

    Returns:
        list of ValidationIssue. Severities:
          - ERROR  : missing required, conflict-pair both applied
          - WARNING: applied set contains a patch_id not in registry

    Conflict pairs are reported once (canonicalized — sorted ids) even when
    the conflict is declared symmetrically on both sides.
    """
    if registry is None:
        registry = _live_registry()

    issues: list[ValidationIssue] = []

    # Unknown ids in applied set
    for pid in applied:
        if pid not in registry:
            issues.append(ValidationIssue(
                "WARNING", pid,
                f"applied set contains unknown patch_id {pid!r}",
            ))

    # Required dependencies — only check patches that ARE applied
    for pid in applied:
        meta = registry.get(pid)
        if meta is None:
            continue  # already reported as unknown above
        for ref in _coerce_list(meta.get("requires_patches")):
            if ref not in applied:
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"missing required dependency: {pid} requires {ref!r} "
                    f"to also be APPLY (currently SKIP)",
                ))

    # Conflicts — canonicalize pairs to avoid double-reporting
    seen_pairs: set[tuple[str, str]] = set()
    for pid in applied:
        meta = registry.get(pid)
        if meta is None:
            continue
        for ref in _coerce_list(meta.get("conflicts_with")):
            if ref in applied:
                pair = tuple(sorted([pid, ref]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                issues.append(ValidationIssue(
                    "ERROR", pid,
                    f"conflict: {pair[0]} and {pair[1]} are both APPLY but "
                    f"declared mutually exclusive — pick one",
                ))

    return issues


def log_validation_issues(issues: list[ValidationIssue]) -> None:
    """Emit issues at appropriate log severity. Operator-readable summary."""
    if not issues:
        log.info("[Genesis Dispatcher v2] validator: clean (no issues)")
        return
    for i in issues:
        msg = f"[Genesis Dispatcher v2] validator {i.severity}: {i.patch_id} — {i.message}"
        if i.severity == "ERROR":
            log.error(msg)
        elif i.severity == "WARNING":
            log.warning(msg)
        else:
            log.info(msg)


# ─── CLI entry-point ──────────────────────────────────────────────────────

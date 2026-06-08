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

    PR38 cleanup (2026-05-08): `sndr.dispatcher.__init__.py`
    re-exports `PATCH_REGISTRY` from `.registry` at package level.
    Tests now monkey-patch the canonical package directly:

        monkeypatch.setattr(
            sndr.dispatcher, "PATCH_REGISTRY", fake_registry,
        )

    `_live_registry()` reads the same attribute on the same package
    module, so the patch propagates without going through any legacy
    shim. The previous Stage 3 indirection through `_genesis.dispatcher`
    is gone now that `_genesis/` is being removed.
    """
    from sndr import dispatcher as _canonical
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
# M.1.1.T0 (2026-05-27): validator constants centralised in
# ``dispatcher/_constants.py``. Re-imported here under their historical
# module-private names so existing call sites continue to resolve.
from ._constants import (  # noqa: F401
    _CANONICAL_ENV_PREFIXES,
    _VALID_IMPLEMENTATION_STATUSES,
    _VALID_LIFECYCLES,
    _VALID_TIERS,
)


def _is_canonical_env_flag(flag: str) -> bool:
    """Registry env_flag must use one of the canonical full-prefix forms.

    See env.py for the alias logic — SNDR_* takes precedence over
    GENESIS_* for the same suffix. Both prefixes work. ALLOW_ is
    semantically distinct from ENABLE_ (operator permission gate vs
    feature switch); both are recognized.
    """
    return any(flag.startswith(p) for p in _CANONICAL_ENV_PREFIXES)


def _audit_tier(pid: str, meta: dict[str, Any]) -> list[ValidationIssue]:
    """Per-entry: ``tier`` must be one of ``_VALID_TIERS`` when set."""
    tier = meta.get("tier")
    if tier is not None and tier not in _VALID_TIERS:
        return [ValidationIssue(
            "ERROR", pid,
            f"tier={tier!r} is not in {sorted(_VALID_TIERS)}",
        )]
    return []


def _audit_lifecycle(pid: str, meta: dict[str, Any]) -> list[ValidationIssue]:
    """Per-entry: ``lifecycle`` must be one of ``_VALID_LIFECYCLES`` when
    set; unset is INFO-class drift signal.

    PR38 §5.5 ratchet (2026-05-08): patches without an explicit
    lifecycle drift into ambiguity over time. Surface as INFO so the
    registry self-documents which patches need a decision. Not raised
    to WARNING because 91 patches today have no lifecycle and we don't
    want to drown out real issues.
    """
    out: list[ValidationIssue] = []
    lifecycle = meta.get("lifecycle")
    if lifecycle is not None and lifecycle not in _VALID_LIFECYCLES:
        out.append(ValidationIssue(
            "ERROR", pid,
            f"lifecycle={lifecycle!r} is not in {sorted(_VALID_LIFECYCLES)}",
        ))
    if lifecycle is None:
        out.append(ValidationIssue(
            "INFO", pid,
            "lifecycle field unset — pick one of "
            f"{sorted(_VALID_LIFECYCLES)}. Promoting to "
            "lifecycle='stable' triggers anchor manifest "
            "requirements; see "
            "docs/upstream/STABLE_PROMOTION_CHECKLIST.md.",
        ))
    return out


def _audit_implementation_status(
    pid: str, meta: dict[str, Any],
) -> list[ValidationIssue]:
    """P2-1 (audit 2026-05-08): ``implementation_status`` enum check."""
    impl_status = meta.get("implementation_status")
    if (impl_status is not None
            and impl_status not in _VALID_IMPLEMENTATION_STATUSES):
        return [ValidationIssue(
            "ERROR", pid,
            f"implementation_status={impl_status!r} is not in "
            f"{sorted(_VALID_IMPLEMENTATION_STATUSES)}",
        )]
    return []


def _audit_upstream_pr_relationship(
    pid: str, meta: dict[str, Any],
) -> list[ValidationIssue]:
    """Phase 5.1.C (2026-05-22): ``upstream_pr_relationship`` enum check.

    After the 5.1.B migration every upstream_pr-bearing entry carries
    an explicit relationship value, so missing-when-set is now an
    ERROR (escalated from silent in 5.1.A). When the field is present
    it MUST be one of the canonical values. The reverse case
    (relationship set without upstream_pr) stays WARNING — likely a
    copy-paste mistake but not fatal.
    """
    out: list[ValidationIssue] = []
    rel = meta.get("upstream_pr_relationship")
    upstream_pr_value = meta.get("upstream_pr")
    if rel is not None and rel not in VALID_UPSTREAM_PR_RELATIONSHIPS:
        out.append(ValidationIssue(
            "ERROR", pid,
            f"upstream_pr_relationship={rel!r} is not in "
            f"{sorted(VALID_UPSTREAM_PR_RELATIONSHIPS)}",
        ))
    if rel is None and isinstance(upstream_pr_value, int):
        out.append(ValidationIssue(
            "ERROR", pid,
            f"upstream_pr is set (#{upstream_pr_value}) but "
            f"upstream_pr_relationship is missing — pick one of "
            f"{sorted(VALID_UPSTREAM_PR_RELATIONSHIPS)}. Default "
            f"choice for plain backports is 'backport'.",
        ))
    if rel is not None and upstream_pr_value is None:
        out.append(ValidationIssue(
            "WARNING", pid,
            f"upstream_pr_relationship={rel!r} is set but "
            f"upstream_pr is None — relationship field has no "
            f"target; either set upstream_pr or remove the "
            f"relationship field",
        ))
    return out


def _audit_env_flag_canonical(
    pid: str, meta: dict[str, Any],
) -> list[ValidationIssue]:
    """``env_flag`` canonical form. WARNING (not ERROR) because the
    runtime decision now strips the prefix and delegates to
    ``env.is_enabled`` — so the registry can be drift-fixed gradually
    without breaking apply behavior."""
    env_flag = meta.get("env_flag")
    if env_flag and not _is_canonical_env_flag(env_flag):
        return [ValidationIssue(
            "WARNING", pid,
            f"env_flag={env_flag!r} lacks canonical SNDR_ENABLE_/"
            f"GENESIS_ENABLE_ prefix — operators may not realize "
            f"the alias works",
        )]
    return []


def _audit_apply_module_importable(
    pid: str, meta: dict[str, Any],
) -> list[ValidationIssue]:
    """``apply_module`` is optional today; will become required when
    the parking-lot ``_per_patch_dispatch.py`` is retired. When
    present, must import-resolve so we fail fast on typo'd paths."""
    apply_module = meta.get("apply_module")
    if not apply_module:
        return []
    try:
        import importlib
        importlib.import_module(apply_module)
    except ModuleNotFoundError as e:
        # Distinguish a genuinely-broken patch path from a missing engine-runtime
        # dependency. The control-plane host (GUI/CLI daemon) has no GPU stack, so
        # patch modules that `import torch`/`triton`/`vllm` fail to import here —
        # that is an environment limitation, NOT a patch defect. Only a missing
        # module under the patch's OWN top-level package (typo'd apply_module
        # path) is a real error operators must fix.
        missing_top = (e.name or "").split(".")[0]
        apply_top = apply_module.split(".")[0]
        if missing_top and missing_top != apply_top:
            return [ValidationIssue(
                "INFO", pid,
                f"apply_module={apply_module!r} not import-verified on this host: "
                f"needs engine runtime dependency {missing_top!r} (present only on "
                f"the GPU/engine host).",
            )]
        return [ValidationIssue(
            "ERROR", pid,
            f"apply_module={apply_module!r} fails to import: "
            f"{type(e).__name__}: {e}",
        )]
    except Exception as e:
        return [ValidationIssue(
            "ERROR", pid,
            f"apply_module={apply_module!r} fails to import: "
            f"{type(e).__name__}: {e}",
        )]
    return []


def _audit_applies_to_shape(
    pid: str, meta: dict[str, Any],
) -> list[ValidationIssue]:
    """``applies_to``: dict or absent. Field-level type checks happen
    at ``_check_applies_to`` call time; this layer only catches
    "someone accidentally wrote a list/string here"."""
    applies_to = meta.get("applies_to")
    if applies_to is not None and not isinstance(applies_to, dict):
        return [ValidationIssue(
            "ERROR", pid,
            f"applies_to must be dict or absent, got "
            f"{type(applies_to).__name__}",
        )]
    return []


def _audit_entry_contract(
    pid: str, meta: dict[str, Any],
) -> list[ValidationIssue]:
    """Run every per-entry contract check in canonical order.

    Order matters: existing tests + ``audit_registry_contract.py``
    consume the issue list and ``test_iron_rule_11_enforcement.py``
    cross-references the order of severities. Preserve the original
    Phase 1 sequence (tier → lifecycle → impl_status → upstream_pr_*
    → env_flag → apply_module → applies_to).
    """
    out: list[ValidationIssue] = []
    out.extend(_audit_tier(pid, meta))
    out.extend(_audit_lifecycle(pid, meta))
    out.extend(_audit_implementation_status(pid, meta))
    out.extend(_audit_upstream_pr_relationship(pid, meta))
    out.extend(_audit_env_flag_canonical(pid, meta))
    out.extend(_audit_apply_module_importable(pid, meta))
    out.extend(_audit_applies_to_shape(pid, meta))
    return out


def _audit_references(
    pid: str, meta: dict[str, Any], keys: set[str],
) -> list[ValidationIssue]:
    """Graph layer per-entry: ``requires_patches`` + ``conflicts_with``
    must reference valid patch_ids and may not self-reference."""
    out: list[ValidationIssue] = []
    for ref in _coerce_list(meta.get("requires_patches")):
        if ref == pid:
            out.append(ValidationIssue(
                "ERROR", pid,
                f"requires_patches contains self-reference {ref!r}",
            ))
        elif ref not in keys:
            out.append(ValidationIssue(
                "ERROR", pid,
                f"requires_patches references unknown patch_id {ref!r}",
            ))
    for ref in _coerce_list(meta.get("conflicts_with")):
        if ref == pid:
            out.append(ValidationIssue(
                "ERROR", pid,
                f"conflicts_with contains self-reference {ref!r}",
            ))
        elif ref not in keys:
            out.append(ValidationIssue(
                "ERROR", pid,
                f"conflicts_with references unknown patch_id {ref!r}",
            ))
    return out


def _audit_requires_cycles(
    registry: dict[str, dict[str, Any]],
) -> list[ValidationIssue]:
    """DFS three-color cycle detection on the ``requires_patches`` graph.

    Reports each cycle once at the entry where the back-edge was
    detected, with the cycle path joined by ``→``.
    """
    out: list[ValidationIssue] = []
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {pid: WHITE for pid in registry}

    def _walk(pid: str, path: list[str]) -> None:
        if color[pid] == GRAY:
            cycle = path[path.index(pid):] + [pid]
            out.append(ValidationIssue(
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
    return out


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

    M.1.1.T2 restructure (2026-05-27): the original 190-LOC body is
    split into private named helpers above. Error message wording,
    severity, and check ordering are preserved byte-identical;
    ``tests/unit/scripts/test_audit_registry_contract.py``,
    ``tests/unit/dispatcher/test_iron_rule_11_enforcement.py`` and the
    live ``scripts/audit_registry_contract.py`` invocation are the
    invariance guards.

    Returns a list of `ValidationIssue` (empty list = clean).
    """
    if registry is None:
        registry = _live_registry()

    issues: list[ValidationIssue] = []
    keys = set(registry.keys())

    # 1. Per-entry contract checks (tier, lifecycle, env_flag, applies_to)
    for pid, meta in registry.items():
        issues.extend(_audit_entry_contract(pid, meta))

    # 2. Reference existence + self-reference (graph layer)
    for pid, meta in registry.items():
        issues.extend(_audit_references(pid, meta, keys))

    # 3. Cycle detection on requires_patches graph (DFS three-color).
    issues.extend(_audit_requires_cycles(registry))

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

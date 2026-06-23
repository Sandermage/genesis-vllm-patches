# SPDX-License-Identifier: Apache-2.0
"""Pure-API layer for ``sndr patches plan`` — M.6.3.

Runs the dispatcher's ``should_apply`` decision for every registry
entry against a preset's environment, returning a structured
:class:`PlanReport`. The preset's ``system_env`` + ``genesis_env``
overlay is applied through a context manager so the calling process's
``os.environ`` is restored on **every** exit path — success, exception,
or generator close. The legacy CLI ``test_plan_restores_env_after_run``
invariant is preserved.

CLI rendering (banners, skip-reason grouping, resolver block, advisory
block) stays in :mod:`sndr.cli.legacy.patches`; this module never
prints.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional


class PresetNotFoundError(LookupError):
    """Raised when ``simulate_plan`` cannot resolve the preset key.

    Carries both the operator-supplied key and the underlying resolver
    error message so the CLI can render the same ``preset {key!r} not
    found (...)`` line operators see pre-M.6.3.
    """

    def __init__(self, preset_key: str, reason: str) -> None:
        super().__init__(f"preset {preset_key!r} not found ({reason})")
        self.preset_key = preset_key
        self.reason = reason


@dataclass(frozen=True)
class PlanReport:
    """Outcome of a dispatcher plan-simulation for one preset.

    ``apply`` / ``skip`` / ``errors`` are lists of plain dicts (one per
    registry entry) so the CLI can iterate them verbatim into JSON
    output. ``resolver_payload`` is present only when the caller
    requested a full policy resolution via the ``policy`` argument;
    ``advisory_warnings`` is the cheap compat-only resolver pass we
    always run so legacy operators still see conflict / candidate_when
    mismatches.
    """

    preset: str
    profile: str
    apply: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    skip: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    errors: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    profile_violations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    resolver_payload: Optional[dict[str, Any]] = None
    advisory_warnings: tuple[str, ...] = field(default_factory=tuple)


@contextlib.contextmanager
def preset_env_overlay(cfg: Any) -> Iterator[None]:
    """Context manager that overlays ``cfg.system_env`` + ``cfg.genesis_env``
    onto :data:`os.environ` and restores every touched key on exit.

    Restoration runs in ``finally``, so the overlay is reverted whether
    the body returned, raised, or was abandoned via generator close.
    Exposed as a public helper so callers running ``should_apply`` or
    bench scripts under a preset's env can reuse the same primitive.
    """
    overlay: dict[str, str] = {}
    overlay.update(getattr(cfg, "system_env", {}) or {})
    overlay.update(getattr(cfg, "genesis_env", {}) or {})
    saved: dict[str, Optional[str]] = {}
    for k, v in overlay.items():
        saved[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def _decision_dict(d: Any, *, explain: bool) -> dict[str, Any]:
    base = {
        "patch_id": d.patch_id,
        "env_flag": d.env_flag,
        "value": d.value,
        "decision": d.decision,
        "role": d.role,
        "reason": d.reason,
    }
    if explain:
        base["note"] = d.note
        base["bench_evidence"] = d.bench_evidence
    return base


def simulate_plan(
    preset_key: str,
    *,
    profile: str = "any",
    policy: Optional[str] = None,
    explain: bool = False,
) -> PlanReport:
    """Simulate dispatcher decisions for ``preset_key`` and return the
    structured report.

    Raises :class:`PresetNotFoundError` when the preset is unknown. All
    other failures from the underlying resolver propagate as their
    original exceptions; the env overlay is restored regardless.
    """
    from sndr.cli.legacy.memory import _resolve_preset_v1_or_v2

    try:
        cfg = _resolve_preset_v1_or_v2(preset_key)
    except Exception as e:
        raise PresetNotFoundError(preset_key, str(e)) from e
    if cfg is None:
        raise PresetNotFoundError(preset_key, "no matching V1 or V2 entry")

    apply_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    with preset_env_overlay(cfg):
        from sndr.dispatcher import PATCH_REGISTRY, should_apply

        for pid in sorted(PATCH_REGISTRY):
            meta = PATCH_REGISTRY[pid]
            if not isinstance(meta, dict):
                continue
            try:
                applied, reason = should_apply(pid)
            except Exception as e:
                error_rows.append({
                    "patch_id": pid,
                    "title": meta.get("title", ""),
                    "tier": meta.get("tier", ""),
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
            row = {
                "patch_id": pid,
                "title": (meta.get("title") or "")[:80],
                "tier": meta.get("tier", "community"),
                "default_on": bool(meta.get("default_on", False)),
                "lifecycle": meta.get("lifecycle"),
                "reason": reason[:160],
            }
            if applied:
                apply_rows.append(row)
            else:
                skip_rows.append(row)

    # Production profile gate. ``partial``/``placeholder`` impl_status =
    # wiring stub; ``research``/``retired`` lifecycle = should not reach
    # production. Read PATCH_REGISTRY outside the env overlay (it's a
    # module-level dict — env doesn't affect it).
    from sndr.dispatcher import PATCH_REGISTRY

    profile_violations: list[dict[str, Any]] = []
    if profile == "production":
        forbidden_status = {"partial", "placeholder"}
        forbidden_lifecycle = {"research", "retired"}
        for r in apply_rows:
            meta = PATCH_REGISTRY.get(r["patch_id"]) or {}
            impl = meta.get("implementation_status")
            lc = meta.get("lifecycle")
            reasons = []
            if impl in forbidden_status:
                reasons.append(f"implementation_status={impl}")
            if lc in forbidden_lifecycle:
                reasons.append(f"lifecycle={lc}")
            if reasons:
                profile_violations.append({
                    "patch_id": r["patch_id"],
                    "title": r["title"],
                    "reasons": reasons,
                })

    # patch_plan resolver layer — runs in advisory mode when no policy
    # was requested, full resolution otherwise. Failure is non-fatal:
    # advisory warnings are surfaced opportunistically, not gated.
    resolver_payload: Optional[dict[str, Any]] = None
    advisory_warnings: tuple[str, ...] = ()
    if policy is None:
        try:
            from sndr.model_configs.patch_plan import resolve_patch_plan

            advisory_plan = resolve_patch_plan(cfg, policy="compat")
            advisory_warnings = tuple(advisory_plan.warnings)
        except Exception:
            advisory_warnings = ()
    else:
        from sndr.model_configs.patch_plan import resolve_patch_plan

        plan = resolve_patch_plan(cfg, policy=policy)
        resolver_payload = {
            "policy": plan.policy,
            "included": [_decision_dict(d, explain=explain) for d in plan.included],
            "excluded": [_decision_dict(d, explain=explain) for d in plan.excluded],
            "warnings": list(plan.warnings),
            "passthrough": dict(plan.passthrough),
            "env": plan.env,
        }

    return PlanReport(
        preset=preset_key,
        profile=profile,
        apply=tuple(apply_rows),
        skip=tuple(skip_rows),
        errors=tuple(error_rows),
        profile_violations=tuple(profile_violations),
        resolver_payload=resolver_payload,
        advisory_warnings=advisory_warnings,
    )

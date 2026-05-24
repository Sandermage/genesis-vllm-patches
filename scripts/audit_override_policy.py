#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.audit — profile OverridePolicy gate.

Audits every profile under `builtin/profile/` for the presence + shape
of `override_policy:` when the profile carries `sizing_override:`. At
Stage 1 (CONFIG-UX.audit phase scope) missing policy is a warning, not
an error — the field is brand-new in CONFIG-UX.1 and existing 21
profiles haven't been retrofitted yet.

Stage-1 contract (CONFIG-UX.audit phase):

  sizing_override + no override_policy        → warning
  invalid override_policy shape               → error (validation)
  effective_class=production + no reason      → error
  effective_class=production + no evidence    → error
  non-production class + no reason            → error
  override_policy without sizing_override     → warning (unusual)

Deferred to CONFIG-UX.4:

  - Class-4 forbidden-override hard enforcement
    (FORBIDDEN_OVERRIDES placeholder constant declared here as
    empty tuple — CONFIG-UX.4 fills + flips to hard reject).
  - Stage 2/3 severity escalation per `SNDR_V1_ROLLOUT_STAGE`.
  - `expires_at` past-date escalation.
  - Cross-validation of evidence_refs paths against
    tests/integration/baselines/ config blocks.

Exit codes:
  0 — clean (default) OR clean (--strict)
  1 — errors found (always) OR warnings found (--strict only)
  2 — usage / IO error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Placeholder for CONFIG-UX.4 hard-enforcement list.
# Operator decision (CONFIG_UX_R §3.2 Class 4) lists candidate forbidden
# overrides; this audit phase does NOT reject them — semantic enforcement
# lives in CONFIG-UX.4. Constant declared here so CONFIG-UX.4 can fill
# without structural change to the script.
FORBIDDEN_OVERRIDES: tuple[str, ...] = ()


@dataclass
class Finding:
    profile_id: str
    severity: str  # info | warning | error
    rule: str
    message: str

    def as_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
        }


@dataclass
class OverrideReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, profile_id: str, severity: str, rule: str, message: str) -> None:
        self.findings.append(Finding(profile_id, severity, rule, message))

    def count_by_severity(self) -> dict[str, int]:
        out = {"info": 0, "warning": 0, "error": 0}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.findings)


def _profile_dir() -> Path:
    return (
        REPO_ROOT / "vllm" / "sndr_core" / "model_configs"
        / "builtin" / "profile"
    )


def _community_profile_dir() -> Path:
    return (
        REPO_ROOT / "vllm" / "sndr_core" / "model_configs"
        / "community" / "profile"
    )


def _list_profile_ids() -> list[str]:
    out: set[str] = set()
    for d in (_profile_dir(), _community_profile_dir()):
        if d.is_dir():
            out.update(
                p.stem for p in d.glob("*.yaml")
                if p.is_file() and not p.stem.startswith("_")
            )
    return sorted(out)


def _audit_one_profile(profile_id: str, report: OverrideReport) -> None:
    """Audit one profile; append findings to report."""
    from vllm.sndr_core.model_configs.registry_v2 import load_profile
    from vllm.sndr_core.model_configs.schema import SchemaError

    try:
        profile = load_profile(profile_id)
    except SchemaError as e:
        report.add(
            profile_id, "error", "schema_load",
            f"YAML parse / shape validation failed: {e}",
        )
        return
    except Exception as e:  # pragma: no cover — guard against unexpected
        report.add(
            profile_id, "error", "schema_load",
            f"unexpected loader error ({type(e).__name__}): {e}",
        )
        return

    sizing = profile.sizing_override
    policy = profile.override_policy

    # ── Rule: override_policy must be sibling of sizing_override ─────
    if sizing is not None and policy is None:
        report.add(
            profile_id, "warning", "missing_override_policy",
            (
                f"profile.sizing_override present but no override_policy: "
                f"add an OverridePolicy block (CONFIG-UX.1 schema) — "
                f"warning at Stage 1, will escalate in CONFIG-UX.4"
            ),
        )
        # Don't return — the audit still checks for any half-shape if user
        # added policy but with bugs. Falls through to None checks below.

    if policy is None:
        # No policy → no further policy-level checks. Sizing alone is OK
        # at Stage 1; CONFIG-UX.4 will tighten.
        if sizing is None:
            # Neither sizing_override nor override_policy → cleanest path,
            # nothing to audit further. Info-only.
            pass
        return

    # ── Rule: override_policy shape ──────────────────────────────────
    # The loader already calls policy.validate() but if something slipped
    # through (e.g. monkey-patched), re-validate here for the audit.
    try:
        policy.validate()
    except SchemaError as e:
        report.add(
            profile_id, "error", "override_policy_shape",
            f"override_policy shape invalid: {e}",
        )
        return

    # ── Rule: override_policy без sizing_override (unusual) ──────────
    if sizing is None and policy is not None:
        report.add(
            profile_id, "warning", "policy_without_sizing",
            (
                "override_policy declared but no sizing_override present — "
                "policy applies to nothing"
            ),
        )

    # ── Rule: class derivation + per-class requirements ──────────────
    effective_class = policy.effective_class(profile.role)

    # Reason is required across all override classes other than
    # safe_per_launch (the trivial pass-through). Even for safe_per_launch
    # we don't require reason (it's the explicit "no policy needed" class).
    if effective_class != "safe_per_launch":
        if not policy.reason:
            report.add(
                profile_id, "error", "missing_reason",
                (
                    f"override_policy.reason required (effective_class="
                    f"{effective_class!r}); operators reading `sndr "
                    f"profile show` need a written justification"
                ),
            )

    # Production class — additionally require evidence references.
    if effective_class == "production":
        if not policy.evidence_refs:
            report.add(
                profile_id, "error", "production_missing_evidence",
                (
                    "effective_class=production requires at least one "
                    "evidence_ref backing the override (path strings; "
                    "audit_override_policy at Stage 1 only checks "
                    "presence — content cross-validation is CONFIG-UX.4)"
                ),
            )

    # ── Rule: forbidden overrides (PLACEHOLDER — Stage 1 no-op) ──────
    # CONFIG-UX.4 will check sizing_override fields against
    # FORBIDDEN_OVERRIDES list (e.g. gpu_memory_utilization > 1.0,
    # kv_cache_dtype downgrade, spec_decode_method change). For now the
    # tuple is empty so this loop is a no-op.
    for _ in FORBIDDEN_OVERRIDES:  # pragma: no cover — empty until CONFIG-UX.4
        pass


def run_audit(profile_ids: Optional[list[str]] = None) -> OverrideReport:
    """Run the override-policy audit.

    Args:
        profile_ids: optional restricted list (testing hook).

    Returns:
        OverrideReport with findings.
    """
    if profile_ids is None:
        profile_ids = _list_profile_ids()
    report = OverrideReport()
    for pid in profile_ids:
        _audit_one_profile(pid, report)
    return report


def _print_table(report: OverrideReport, total_profiles: int) -> None:
    counts = report.count_by_severity()
    print("audit-override-policy: profile sizing_override + OverridePolicy")
    print("─" * 70)
    print(f"  scanned: {total_profiles} profile(s)")
    print(
        f"  findings: {counts.get('error', 0)} error, "
        f"{counts.get('warning', 0)} warning, "
        f"{counts.get('info', 0)} info"
    )
    print()
    if not report.findings:
        print("  ✓ no findings")
        return
    by_severity = {"error": [], "warning": [], "info": []}
    for f in report.findings:
        by_severity[f.severity].append(f)
    for sev in ("error", "warning", "info"):
        items = by_severity[sev]
        if not items:
            continue
        marker = {"error": "✗", "warning": "⚠", "info": "•"}[sev]
        print(f"  {marker} {sev.upper()} ({len(items)}):")
        for f in items:
            print(f"      [{f.rule}] {f.profile_id}: {f.message}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON instead of the table view",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=("treat warnings as fatal (CI/release gate). Default mode "
              "exits 0 on warnings — only errors are fatal."),
    )
    parser.add_argument(
        "--profile", action="append", default=None,
        help="limit audit to one profile id (repeatable). Default: all.",
    )
    args = parser.parse_args()

    try:
        profile_ids = args.profile or _list_profile_ids()
        if not profile_ids:
            print("audit-override-policy: no profiles found", file=sys.stderr)
            return 2
        report = run_audit(profile_ids)
    except Exception as e:  # pragma: no cover
        print(
            f"audit-override-policy: internal error: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 2

    if args.json:
        payload = {
            "scanned": len(profile_ids),
            "counts": report.count_by_severity(),
            "findings": [f.as_dict() for f in report.findings],
            "has_errors": report.has_errors(),
            "has_warnings": report.has_warnings(),
            "strict": args.strict,
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_table(report, len(profile_ids))

    if report.has_errors():
        return 1
    if args.strict and report.has_warnings():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

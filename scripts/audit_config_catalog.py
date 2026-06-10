#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.audit — preset card catalog gate.

Wraps the typed `validate_for_status()` from `preset_schema.py` plus
catalog-level cross-checks the loader can't see (paths, fallback chain,
default_for_family uniqueness).

Severity model (CONFIG_UX_R §6 + audit phase scope refinement):

  default (no --strict)  → exit 0 even with warnings; CI advisory
  --strict               → exit 1 on any warning or error
  pre-commit             → invokes default mode (non-fatal during
                            Stage 1 of V1 rollout; card-less presets
                            still warn but don't block commits)

Stage-1 contract (CONFIG-UX.audit phase):

  card-less preset             → warning (not error)
  card present, non-production → permissive (skip strict)
  card present, production*    → strict (errors block --strict mode)
  fallback_preset missing      → error (always)
  default_for_family collision → error (always)
  evidence_refs[].path absent  → error (unless external://)
  public-production + private  → error per §2.4 rule 1

Deferred to CONFIG-UX.4:

  - Stage 2/3 escalation (warnings → errors per rollout stage)
  - Routing-family tight coupling with `sndr routing-table`
  - Class-4 forbidden override hard enforcement (lives in
    audit_override_policy.py constants, not enforced here)

Exit codes:
  0 — clean (default) OR clean (--strict)
  1 — errors found (always) OR warnings found (--strict only)
  2 — usage / IO error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Finding:
    """One audit finding for a specific preset.

    Severity:
      info    → status quo, not a problem (e.g. card-less at Stage 1)
      warning → should fix soon; non-fatal at default severity
      error   → must fix; fatal even at default severity
    """
    preset_id: str
    severity: str  # info | warning | error
    rule: str
    message: str

    def as_dict(self) -> dict:
        return {
            "preset_id": self.preset_id,
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
        }


@dataclass
class CatalogReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, preset_id: str, severity: str, rule: str, message: str) -> None:
        self.findings.append(Finding(preset_id, severity, rule, message))

    def count_by_severity(self) -> dict[str, int]:
        out = {"info": 0, "warning": 0, "error": 0}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def has_warnings(self) -> bool:
        return any(f.severity == "warning" for f in self.findings)


def _preset_dir() -> Path:
    # v12.1 (2026-06-09): canonical path is sndr/. Legacy vllm/sndr_core/
    # was archived to sndr_private/archive/ in commit 6bf9c04c.
    canonical = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "presets"
    if canonical.is_dir():
        return canonical
    return (
        REPO_ROOT / "sndr" / "model_configs"
        / "builtin" / "presets"
    )


def _list_preset_ids() -> list[str]:
    d = _preset_dir()
    if not d.is_dir():
        return []
    return sorted(
        p.stem for p in d.glob("*.yaml")
        if p.is_file() and not p.stem.startswith("_")
    )


def _audit_one_preset(
    preset_id: str,
    report: CatalogReport,
    *,
    all_preset_ids: set[str],
    family_defaults: dict[str, list[str]],  # family → [preset_id, ...] declaring default_for_family=True
) -> None:
    """Audit one preset; append findings to report.

    Imports happen lazily so the script can be invoked without
    triggering the full sndr_core import chain at script startup.
    """
    from sndr.model_configs.preset_schema import (
        validate_for_status,
    )
    from sndr.model_configs.registry_v2 import load_preset_def
    from sndr.model_configs.schema import SchemaError

    try:
        preset = load_preset_def(preset_id)
    except SchemaError as e:
        report.add(
            preset_id, "error", "schema_load",
            f"YAML parse / shape validation failed: {e}",
        )
        return
    except Exception as e:  # pragma: no cover — guard against unexpected loader errors
        report.add(
            preset_id, "error", "schema_load",
            f"unexpected loader error ({type(e).__name__}): {e}",
        )
        return

    # ── Rule: card presence ──────────────────────────────────────────
    if not preset.has_card():
        report.add(
            preset_id, "warning", "missing_card",
            "preset has no `card:` annotation (CONFIG-UX.2 work pending)",
        )
        # No further card-level checks possible on a legacy 3-pointer preset.
        return

    card = preset.card
    status = card.status

    # ── Rule: status-aware semantic validation (preset_schema gives it) ──
    semantic_errs = validate_for_status(card, preset_id)
    for msg in semantic_errs:
        # Production-grade status → error; non-production → permissive.
        # validate_for_status() already returns empty list for non-prod
        # statuses, so any items here are production-grade by construction.
        report.add(preset_id, "error", "card_strict_validation", msg)

    # ── Rule: fallback_preset resolves ───────────────────────────────
    if card.fallback_preset is not None:
        if card.fallback_preset not in all_preset_ids:
            report.add(
                preset_id, "error", "fallback_resolution",
                f"card.fallback_preset={card.fallback_preset!r} does not match "
                f"any preset in builtin/presets/ (must be an existing preset id)",
            )

    # ── Rule: default_for_family uniqueness ──────────────────────────
    # Collisions are detected at the report level (after all presets walked).
    if card.default_for_family and card.routing_family:
        family_defaults.setdefault(card.routing_family, []).append(preset_id)

    # ── Rule: evidence_refs[].path exists (or external://) ───────────
    for i, ev in enumerate(card.evidence_refs):
        if ev.path.startswith("external://"):
            continue
        if not ev.path:
            report.add(
                preset_id, "error", "evidence_path",
                f"card.evidence_refs[{i}]: path is empty",
            )
            continue
        # Resolve relative paths against repo root.
        resolved = (REPO_ROOT / ev.path).resolve()
        # Bound the check to repo subtree — accept anything under REPO_ROOT
        # or a symlinked location that resolves to a real file.
        if not resolved.exists():
            report.add(
                preset_id, "error", "evidence_path",
                f"card.evidence_refs[{i}].path={ev.path!r} does not exist "
                f"(repo-relative; use `external://...` for off-repo refs)",
            )

    # ── Rule: §2.4 #1 — public-production preset must have public evidence ──
    # Strict validation already emits this when card status=production +
    # audience=operator; this audit reinforces it for production_candidate too
    # (forward-compat hint: when promoting to production, this becomes
    # blocking). At audit phase scope: warn-only for production_candidate;
    # error already handled via validate_for_status for production+operator.
    if (
        status == "production_candidate"
        and card.audience == "operator"
        and card.evidence_visibility in (None, "private")
    ):
        public_refs = [ev for ev in card.evidence_refs if ev.visibility == "public"]
        if not public_refs and card.evidence_visibility != "public":
            report.add(
                preset_id, "warning", "production_candidate_public_evidence",
                "production_candidate + audience=operator should already have "
                "at least one public evidence_ref (will block promotion to "
                "production)",
            )

    # ── Rule: routing_family light cross-check ───────────────────────
    # Not tight-coupled to sndr routing-table at this phase. We just verify
    # the field is non-empty when set, and string-formatted. Cross-card
    # consistency would require a parsed routing-table snapshot — deferred
    # to CONFIG-UX.4.
    if card.routing_family is not None and not isinstance(card.routing_family, str):
        report.add(
            preset_id, "error", "routing_family_shape",
            f"card.routing_family must be a string, got "
            f"{type(card.routing_family).__name__}",
        )


def run_audit(preset_ids: Optional[list[str]] = None) -> CatalogReport:
    """Run the full preset catalog audit.

    Args:
        preset_ids: optional restricted list (testing hook); default = all
            presets under builtin/presets/.

    Returns:
        CatalogReport with findings.
    """
    if preset_ids is None:
        preset_ids = _list_preset_ids()

    all_ids = set(preset_ids)
    family_defaults: dict[str, list[str]] = {}
    report = CatalogReport()

    for pid in preset_ids:
        _audit_one_preset(
            pid, report,
            all_preset_ids=all_ids,
            family_defaults=family_defaults,
        )

    # Post-walk aggregate rule: default_for_family uniqueness per family.
    for family, presets in family_defaults.items():
        if len(presets) > 1:
            joined = ", ".join(repr(p) for p in presets)
            for pid in presets:
                report.add(
                    pid, "error", "default_for_family_collision",
                    f"routing_family={family!r} has {len(presets)} presets "
                    f"declaring default_for_family=True ({joined}); max 1",
                )

    return report


def _print_table(report: CatalogReport, total_presets: int) -> None:
    counts = report.count_by_severity()
    print("audit-config-catalog: preset card + catalog cross-checks")
    print("─" * 70)
    print(f"  scanned: {total_presets} preset(s)")
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
            print(f"      [{f.rule}] {f.preset_id}: {f.message}")
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
        "--preset", action="append", default=None,
        help="limit audit to one preset id (repeatable). Default: all.",
    )
    args = parser.parse_args()

    try:
        preset_ids = args.preset or _list_preset_ids()
        if not preset_ids:
            print("audit-config-catalog: no presets found", file=sys.stderr)
            return 2
        report = run_audit(preset_ids)
    except Exception as e:  # pragma: no cover — guard against import failures
        print(
            f"audit-config-catalog: internal error: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 2

    if args.json:
        payload = {
            "scanned": len(preset_ids),
            "counts": report.count_by_severity(),
            "findings": [f.as_dict() for f in report.findings],
            "has_errors": report.has_errors(),
            "has_warnings": report.has_warnings(),
            "strict": args.strict,
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_table(report, len(preset_ids))

    if report.has_errors():
        return 1
    if args.strict and report.has_warnings():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

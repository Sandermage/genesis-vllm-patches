#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 6 P3.4 — Dispatcher migration readiness audit.

Master plan §3.4: migrate from legacy apply_patch_* iteration to
spec-driven iter_patch_specs() iteration. Per scout (2026-06-03):
  - Registry is already 92% spec-driven (221/240 have apply_module)
  - 19 entries intentionally unmapped (lifecycle=legacy / marker_only /
    research / retired / coordinator)
  - Real gaps: 0
  - No migration script needed for Tier A (auto-derivation handles it)
  - 95 legacy apply_patch_* functions can be retired once the
    orchestrator switches to iter_patch_specs()

This script audits the live registry + apply dispatch state and reports
whether the orchestrator switch is safe to make in the current release.

Output: human-readable audit + JSON shape suitable for CI gates.

Usage
-----
    python3 scripts/audit_dispatcher_migration_readiness.py
    python3 scripts/audit_dispatcher_migration_readiness.py --json
    python3 scripts/audit_dispatcher_migration_readiness.py --strict
        # exit 1 if migration is not ready (unmapped gaps > 0)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
Status: v11.2.0+ P3.4 audit (orchestrator switch deferred to v12.0.0)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_registry():
    """Import PATCH_REGISTRY without requiring vllm to be installed."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
        return PATCH_REGISTRY
    except ImportError as exc:
        raise SystemExit(
            f"could not import PATCH_REGISTRY: {exc}\n"
            f"hint: run from repo root with PYTHONPATH=. or install in editable mode"
        )


def _categorize_entries(registry: dict) -> dict:
    """Bucket every registry entry by spec-readiness."""
    buckets = {
        "ready_explicit_apply_module": [],
        "ready_auto_derived": [],
        "intentionally_unmapped_legacy": [],
        "intentionally_unmapped_marker_only": [],
        "intentionally_unmapped_research": [],
        "intentionally_unmapped_retired": [],
        "intentionally_unmapped_coordinator": [],
        "REAL_GAPS": [],
    }
    for pid, entry in registry.items():
        apply_module = entry.get("apply_module")
        lifecycle = entry.get("lifecycle", "unknown")
        impl_status = entry.get("implementation_status", "unknown")

        if apply_module:
            # Explicit apply_module field — fully spec-driven
            buckets["ready_explicit_apply_module"].append(pid)
            continue

        # No explicit apply_module — could be auto-derived (file on disk),
        # intentional (legacy/marker_only/etc.), or a real gap.
        if lifecycle == "legacy":
            buckets["intentionally_unmapped_legacy"].append(pid)
        elif lifecycle == "retired":
            buckets["intentionally_unmapped_retired"].append(pid)
        elif lifecycle == "research":
            buckets["intentionally_unmapped_research"].append(pid)
        elif lifecycle == "coordinator":
            buckets["intentionally_unmapped_coordinator"].append(pid)
        elif impl_status in ("marker_only", "placeholder", "metadata_only"):
            buckets["intentionally_unmapped_marker_only"].append(pid)
        else:
            # Try to auto-derive — check if any file matches the pattern
            # in vllm/sndr_core/integrations/<family>/p<id>_*.py
            family = entry.get("family", "").replace(".", "/")
            integrations_dir = (
                REPO_ROOT / "sndr" / "engines" / "vllm" / "patches"
            )
            if family:
                family_dir = integrations_dir / family
                if family_dir.exists():
                    # Try to find a matching file
                    pid_lower = pid.lower()
                    matches = list(family_dir.glob(f"{pid_lower}_*.py"))
                    if matches:
                        buckets["ready_auto_derived"].append(pid)
                        continue
            # No file found, no lifecycle override → real gap
            buckets["REAL_GAPS"].append(pid)

    return buckets


def _make_summary(buckets: dict, total: int) -> dict:
    ready = (
        len(buckets["ready_explicit_apply_module"])
        + len(buckets["ready_auto_derived"])
    )
    intentional = sum(
        len(buckets[k])
        for k in (
            "intentionally_unmapped_legacy",
            "intentionally_unmapped_marker_only",
            "intentionally_unmapped_research",
            "intentionally_unmapped_retired",
            "intentionally_unmapped_coordinator",
        )
    )
    gaps = len(buckets["REAL_GAPS"])

    return {
        "total_entries": total,
        "spec_ready": ready,
        "spec_ready_pct": round(100.0 * ready / total, 1) if total else 0.0,
        "intentionally_unmapped": intentional,
        "intentionally_unmapped_pct": (
            round(100.0 * intentional / total, 1) if total else 0.0
        ),
        "real_gaps": gaps,
        "real_gaps_pct": round(100.0 * gaps / total, 1) if total else 0.0,
        "migration_safe": gaps == 0,
        "tier_breakdown": {
            "Tier_A_explicit": len(buckets["ready_explicit_apply_module"]),
            "Tier_F_auto_derived": len(buckets["ready_auto_derived"]),
            "Tier_E_retired": len(buckets["intentionally_unmapped_retired"]),
            "Tier_legacy_unmapped": len(
                buckets["intentionally_unmapped_legacy"]
            ),
            "Tier_marker_only": len(buckets["intentionally_unmapped_marker_only"]),
            "Tier_research": len(buckets["intentionally_unmapped_research"]),
            "Tier_coordinator": len(buckets["intentionally_unmapped_coordinator"]),
        },
    }


def _print_human(buckets: dict, summary: dict) -> None:
    print("=" * 70)
    print("Phase 6 P3.4 — Dispatcher migration readiness audit")
    print("=" * 70)
    print()
    print(f"Total PATCH_REGISTRY entries:    {summary['total_entries']}")
    print(
        f"Spec-ready (explicit + derived): {summary['spec_ready']} "
        f"({summary['spec_ready_pct']}%)"
    )
    print(
        f"Intentionally unmapped:          {summary['intentionally_unmapped']} "
        f"({summary['intentionally_unmapped_pct']}%)"
    )
    print(
        f"REAL GAPS:                       {summary['real_gaps']} "
        f"({summary['real_gaps_pct']}%)"
    )
    print()
    print("Tier breakdown:")
    for k, v in summary["tier_breakdown"].items():
        print(f"  {k}: {v}")
    print()
    if summary["real_gaps"] > 0:
        print(f"⚠  REAL GAPS — these entries lack any apply_module + lifecycle override:")
        for pid in buckets["REAL_GAPS"]:
            print(f"    - {pid}")
        print()
        print(
            "→ NOT READY for v12.0.0 orchestrator switch. Either map them, "
            "give them a lifecycle override, or retire them."
        )
    else:
        print("✓ Migration ready — all 240 entries have a known apply path.")
        print()
        print("To switch the orchestrator iteration (deferred to v12.0.0):")
        print(
            "  1. Change vllm/sndr_core/apply/orchestrator.py to iterate "
            "iter_patch_specs() instead of PATCH_REGISTRY."
        )
        print(
            "  2. For each spec with apply_module, importlib.import_module "
            "+ call its apply() method."
        )
        print("  3. For each spec without apply_module, honor the lifecycle:")
        print("     legacy/marker_only/research/retired/coordinator → no-op skip.")
        print(
            "  4. Drop the 95 legacy apply_patch_* functions from "
            "vllm/sndr_core/apply/_per_patch_dispatch.py."
        )
        print()
        print("Estimated effort: 2-3 days (less than master plan's 8-9 day estimate)")
        print("because auto-derivation is already done — no per-patch migration work needed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--json", action="store_true",
        help="emit machine-readable JSON (buckets + summary)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit 1 when real_gaps > 0",
    )
    args = parser.parse_args()

    registry = _import_registry()
    buckets = _categorize_entries(registry)
    summary = _make_summary(buckets, len(registry))

    if args.json:
        print(json.dumps(
            {"buckets": buckets, "summary": summary},
            indent=2, sort_keys=True,
        ))
    else:
        _print_human(buckets, summary)

    if args.strict and summary["real_gaps"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

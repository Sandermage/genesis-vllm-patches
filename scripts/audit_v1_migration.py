#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CONFIG-UX.4.1 — V1 monolithic key migration audit.

Separate from `audit_no_new_v1.py` (which freezes the V1 baseline at 12
keys). This audit is about **runtime resolution semantics**: for every
V1 key actually present in the repo, look up its migration bucket
(transparent / needs_operator_choice / deprecated / tombstone) and
emit a per-stage severity finding.

Severity model is driven by `_rollout.py::effective_severity()`:

  Stage 0/1 (default at CONFIG-UX.4.1 ship)  →  all warnings non-fatal
  Stage 2  →  --strict turns warnings into exit 1
  Stage 3  →  ERROR by default for non-transparent buckets

Operator overrides:
  --stage N          explicit stage selector (testing hook)
  --strict           treat warnings as fatal
  --json             machine-readable output
  SNDR_V1_ROLLOUT_STAGE=N  env-driven stage selector
  GENESIS_DISABLE_V1_DEPRECATION_WARNING=1  silences informational output

Exit codes:
  0 — clean (no errors at the chosen severity)
  1 — at least one error (or warning with --strict / Stage 3)
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


_MIGRATION_TABLE_PATH = (
    REPO_ROOT
    / "sndr" / "model_configs"
    / "_v1_migration_table.json"
)


@dataclass
class MigrationEntry:
    bucket: str
    v2_preset: Optional[str]
    rationale: str

    def as_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "v2_preset": self.v2_preset,
            "rationale": self.rationale,
        }


@dataclass
class Finding:
    v1_key: str
    bucket: str
    severity: str  # info | warn | error
    v2_preset: Optional[str]
    rationale: str

    def as_dict(self) -> dict:
        return {
            "v1_key": self.v1_key,
            "bucket": self.bucket,
            "severity": self.severity,
            "v2_preset": self.v2_preset,
            "rationale": self.rationale,
        }


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    table_entry_count: int = 0
    v1_keys_seen: int = 0

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def count_by_severity(self) -> dict[str, int]:
        out = {"info": 0, "warn": 0, "error": 0}
        for f in self.findings:
            out[f.severity] = out.get(f.severity, 0) + 1
        return out

    def count_by_bucket(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.bucket] = out.get(f.bucket, 0) + 1
        return out

    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def has_warnings(self) -> bool:
        return any(f.severity == "warn" for f in self.findings)


def load_migration_table() -> dict[str, MigrationEntry]:
    """Load `_v1_migration_table.json` into typed entries."""
    if not _MIGRATION_TABLE_PATH.is_file():
        raise FileNotFoundError(
            f"V1 migration table not found: {_MIGRATION_TABLE_PATH}"
        )
    with _MIGRATION_TABLE_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("schema_version") != 1:
        raise ValueError(
            f"unsupported migration table schema_version: "
            f"{data.get('schema_version')!r} (expected 1)"
        )
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        raise ValueError("migration table `entries` must be a mapping")
    out: dict[str, MigrationEntry] = {}
    for key, body in entries.items():
        if not isinstance(body, dict):
            raise ValueError(
                f"migration table entry for {key!r} must be a mapping"
            )
        bucket = body.get("bucket")
        from sndr.model_configs._rollout import BUCKETS
        # Only V1-source buckets allowed in this table (synthetic
        # card_less_* / missing_override_policy buckets live elsewhere).
        v1_allowed = {"transparent", "needs_operator_choice", "deprecated", "tombstone"}
        if bucket not in v1_allowed:
            raise ValueError(
                f"migration table entry {key!r}: bucket={bucket!r} must be one of "
                f"{sorted(v1_allowed)}"
            )
        out[key] = MigrationEntry(
            bucket=bucket,
            v2_preset=body.get("v2_preset"),
            rationale=body.get("rationale", ""),
        )
    return out


def list_v1_keys_on_disk() -> list[str]:
    """Read the actual V1 monolithic top-level keys present at
    `sndr/model_configs/builtin/*.yaml`.

    Same scan path as `audit_no_new_v1.py` — but this audit doesn't
    care about freeze; it reads the `key:` field from each YAML so
    we can resolve the migration bucket.
    """
    builtin = REPO_ROOT / "sndr" / "model_configs" / "builtin"
    if not builtin.is_dir():
        return []
    keys: list[str] = []
    for yaml_path in sorted(builtin.glob("*.yaml")):
        if yaml_path.stem.startswith("_") or not yaml_path.is_file():
            continue
        try:
            text = yaml_path.read_text(encoding="utf-8")
        except OSError:
            continue
        # Minimal `key:` extraction — avoid full YAML load to keep the
        # audit torch-free + fast.
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("key:"):
                value = line.split(":", 1)[1].strip()
                # Strip optional quotes
                if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                    value = value[1:-1]
                if value:
                    keys.append(value)
                break
    return sorted(keys)


def run_audit(
    *,
    stage: Optional[int] = None,
    strict_mode: bool = False,
) -> Report:
    """Run the V1 migration audit.

    Args:
        stage: explicit stage override; None reads env.
        strict_mode: if True, --strict semantics (warnings → fatal at Stage 2+).

    Returns:
        Report with per-key findings + counts.
    """
    from sndr.model_configs._rollout import effective_severity

    table = load_migration_table()
    v1_keys = list_v1_keys_on_disk()
    report = Report(table_entry_count=len(table), v1_keys_seen=len(v1_keys))

    # Optional: missing-from-table or extra-in-table sanity.
    table_keys = set(table.keys())
    on_disk = set(v1_keys)
    # On-disk keys not in table → emit a warn finding (defensive — should
    # not happen because audit_no_new_v1.py freezes the baseline).
    for key in sorted(on_disk - table_keys):
        report.add(Finding(
            v1_key=key,
            bucket="needs_operator_choice",  # safest default
            severity="warn",
            v2_preset=None,
            rationale=(
                "V1 key present on disk but not in migration table — "
                "add an entry to `_v1_migration_table.json` (defensive default: "
                "needs_operator_choice)."
            ),
        ))

    # Per-key finding for everything that IS in the table.
    for key, entry in sorted(table.items()):
        sev = effective_severity(
            bucket=entry.bucket,  # type: ignore[arg-type]
            stage=stage,
            strict_mode=strict_mode,
        )
        # Skip on-disk presence check — entry is informational regardless.
        if key not in on_disk:
            # Table-only entry → record as info (operator removed the V1
            # YAML but migration table still references it).
            report.add(Finding(
                v1_key=key,
                bucket=entry.bucket,
                severity="info",
                v2_preset=entry.v2_preset,
                rationale=(
                    f"Migration entry exists but V1 YAML not present on disk "
                    f"(table-only). Originally: {entry.rationale}"
                ),
            ))
            continue
        report.add(Finding(
            v1_key=key,
            bucket=entry.bucket,
            severity=sev,
            v2_preset=entry.v2_preset,
            rationale=entry.rationale,
        ))

    return report


def _print_table(report: Report) -> None:
    counts_s = report.count_by_severity()
    counts_b = report.count_by_bucket()
    from sndr.model_configs._rollout import rollout_stage
    print("audit-v1-migration: V1 monolithic key migration bucket resolution")
    print("─" * 70)
    stage = rollout_stage()
    print(f"  stage: {stage} (env SNDR_V1_ROLLOUT_STAGE; default = 0)")
    print(f"  v1 keys on disk: {report.v1_keys_seen}")
    print(f"  migration table entries: {report.table_entry_count}")
    print(
        f"  findings: {counts_s.get('error', 0)} error, "
        f"{counts_s.get('warn', 0)} warn, "
        f"{counts_s.get('info', 0)} info"
    )
    print(
        f"  bucket distribution: "
        f"transparent={counts_b.get('transparent', 0)} "
        f"needs_choice={counts_b.get('needs_operator_choice', 0)} "
        f"deprecated={counts_b.get('deprecated', 0)} "
        f"tombstone={counts_b.get('tombstone', 0)}"
    )
    print()
    if not report.findings:
        print("  ✓ no findings")
        return

    by_severity = {"error": [], "warn": [], "info": []}
    for f in report.findings:
        by_severity[f.severity].append(f)
    for sev in ("error", "warn", "info"):
        items = by_severity[sev]
        if not items:
            continue
        marker = {"error": "✗", "warn": "⚠", "info": "•"}[sev]
        print(f"  {marker} {sev.upper()} ({len(items)}):")
        for f in items:
            v2 = f.v2_preset or "(no V2 alias)"
            print(f"      [{f.bucket}] {f.v1_key}  →  {v2}")
            if f.rationale:
                print(f"          {f.rationale}")
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
        "--stage", type=int, default=None,
        help=("explicit stage override (testing hook). None reads "
              "SNDR_V1_ROLLOUT_STAGE env."),
    )
    args = parser.parse_args()

    try:
        report = run_audit(stage=args.stage, strict_mode=args.strict)
    except (FileNotFoundError, ValueError) as e:
        print(f"audit-v1-migration: {e}", file=sys.stderr)
        return 2

    if args.json:
        payload = {
            "stage": (
                args.stage
                if args.stage is not None
                else __import__(
                    "sndr.model_configs._rollout",
                    fromlist=["rollout_stage"],
                ).rollout_stage()
            ),
            "strict": args.strict,
            "v1_keys_on_disk": report.v1_keys_seen,
            "table_entries": report.table_entry_count,
            "counts": report.count_by_severity(),
            "bucket_distribution": report.count_by_bucket(),
            "findings": [f.as_dict() for f in report.findings],
            "has_errors": report.has_errors(),
            "has_warnings": report.has_warnings(),
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_table(report)

    if report.has_errors():
        return 1
    if args.strict and report.has_warnings():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

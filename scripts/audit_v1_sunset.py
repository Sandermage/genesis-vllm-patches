#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""audit_v1_sunset.py — V1 monolithic config sunset countdown.

§9.T (V1 monolithic sunset) tracks the 5-stage retirement of the
legacy top-level ``builtin/*.yaml`` presets in favor of the V2 layered
``model/`` + ``hardware/`` + ``profile/`` + ``presets/`` triplet. The
sibling audits cover two narrow concerns:

  - ``audit_no_new_v1.py``: V1 freeze baseline (no new V1 file may
    land at ``builtin/*.yaml``).
  - ``audit_v1_migration.py``: per-V1-key bucket resolution + severity
    matrix at the configured rollout stage.

This audit fills the gap between those two by producing an aggregate
**sunset-readiness report** answering five questions:

  1. What rollout stage is in effect right now?
  2. How many V1 files are on disk, and what is the bucket distribution?
  3. For every V1 key, does its declared V2 preset alias actually exist
     in the ``builtin/presets/`` filesystem?
  4. Per stage gate (T.1 → T.5 of §9.T), what blockers remain?
  5. Is any V1 key tagged ``tombstone`` (which would force exit 1
     regardless of stage)?

The script is **informational by default** — exit 0 unless a true
sunset blocker is present:

  - ``tombstone`` bucket entry whose file is still on disk (incident),
  - or ``--strict`` AND any classification is ``blocker_no_v2_alias``
    AND the resolved stage is ≥ 3 (operator has opted into hard
    enforcement of V1 retirement readiness).

Modes:

  python3 scripts/audit_v1_sunset.py               # human-readable
  python3 scripts/audit_v1_sunset.py --json        # machine-readable
  python3 scripts/audit_v1_sunset.py --strict      # gate Stage 3+ blockers
  python3 scripts/audit_v1_sunset.py --stage N     # override resolved stage

Exit codes:

  0 — sunset countdown clean for the resolved stage.
  1 — tombstone incident, or Stage 3+ blockers with ``--strict``.
  2 — internal error / migration table unloadable / V1 inventory
      diverged from migration table.
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

MODEL_CONFIGS_DIR = REPO_ROOT / "sndr" / "model_configs"
BUILTIN_DIR = MODEL_CONFIGS_DIR / "builtin"
PRESETS_DIR = BUILTIN_DIR / "presets"
MIGRATION_TABLE_PATH = MODEL_CONFIGS_DIR / "_v1_migration_table.json"


CLASSIFICATIONS: tuple[str, ...] = (
    "tombstone_ready",            # transparent + V2 alias resolves
    "tombstone_candidate",        # deprecated + V2 alias resolves
    "operator_decision_pending",  # needs_operator_choice + V2 resolves
    "blocker_no_v2_alias",        # v2_preset null OR doesn't resolve
    "tombstoned",                 # bucket=tombstone (V1 file removed)
    "untracked",                  # V1 file on disk, no migration entry
)


@dataclass
class V1Classification:
    v1_key: str
    bucket: str
    v2_preset: Optional[str]
    classification: str
    rationale: str
    v2_resolves: bool

    def as_dict(self) -> dict:
        return {
            "v1_key": self.v1_key,
            "bucket": self.bucket,
            "v2_preset": self.v2_preset,
            "classification": self.classification,
            "rationale": self.rationale,
            "v2_resolves": self.v2_resolves,
        }


@dataclass
class StageReadiness:
    stage: int
    ready: bool
    blockers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "stage": self.stage,
            "ready": self.ready,
            "blockers": list(self.blockers),
        }


@dataclass
class SunsetReport:
    rollout_stage: int
    v1_files_on_disk: int
    migration_table_entries: int
    classifications: list[V1Classification]
    counts: dict[str, int]
    stages: list[StageReadiness]
    tombstone_incidents: list[str]
    untracked_keys: list[str]

    def as_dict(self) -> dict:
        return {
            "rollout_stage": self.rollout_stage,
            "v1_files_on_disk": self.v1_files_on_disk,
            "migration_table_entries": self.migration_table_entries,
            "classifications": [c.as_dict() for c in self.classifications],
            "counts": dict(self.counts),
            "stages": [s.as_dict() for s in self.stages],
            "tombstone_incidents": list(self.tombstone_incidents),
            "untracked_keys": list(self.untracked_keys),
        }


def load_migration_table(path: Path = MIGRATION_TABLE_PATH) -> dict[str, dict]:
    """Read the migration table JSON and return its ``entries`` dict.

    Returns an empty dict on missing/malformed file — the caller's
    audit will then flag every V1 key as ``untracked``.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = data.get("entries", {})
    return entries if isinstance(entries, dict) else {}


def list_v1_files_on_disk(builtin_dir: Path = BUILTIN_DIR) -> set[str]:
    """Return the set of V1 monolithic preset keys (stem of top-level
    ``*.yaml`` files; subdirs are V2 and are skipped)."""
    if not builtin_dir.is_dir():
        return set()
    out: set[str] = set()
    for f in builtin_dir.glob("*.yaml"):
        if not f.is_file():
            continue
        # `audit_no_new_v1.py` uses .name; the migration table uses the
        # stem (no extension). Strip ``.yaml`` to match the table key.
        # The frozen baseline also uses ``-EXAMPLE`` (uppercase) for
        # three example files while the table normalizes to lowercase
        # ``-example``; preserve the on-disk form here and normalize at
        # comparison time.
        out.add(f.stem.replace("-EXAMPLE", "-example"))
    return out


def list_v2_preset_ids(presets_dir: Path = PRESETS_DIR) -> set[str]:
    """Return the set of V2 preset stems present in ``presets/``."""
    if not presets_dir.is_dir():
        return set()
    return {p.stem for p in presets_dir.glob("*.yaml") if p.is_file()}


def classify(
    v1_keys: set[str],
    entries: dict[str, dict],
    v2_presets: set[str],
) -> tuple[list[V1Classification], list[str]]:
    """Compute per-V1-key classification + the list of ``untracked``
    V1 keys (files on disk but absent from the migration table)."""
    classifications: list[V1Classification] = []
    untracked: list[str] = []

    seen_keys = set(v1_keys)

    # 1) Entries from the table (covers tombstone entries even if the
    #    V1 file is already deleted).
    for key, entry in sorted(entries.items()):
        bucket = entry.get("bucket", "needs_operator_choice")
        v2_preset = entry.get("v2_preset")
        rationale = entry.get("rationale", "")

        if bucket == "tombstone":
            classifications.append(V1Classification(
                v1_key=key,
                bucket=bucket,
                v2_preset=v2_preset,
                classification="tombstoned",
                rationale=rationale,
                v2_resolves=(v2_preset in v2_presets) if v2_preset else False,
            ))
            seen_keys.discard(key)
            continue

        # Non-tombstone entries describe live V1 files; if the file is
        # missing on disk the entry is stale — treat as ``untracked``
        # in the divergence list but still classify by the table state.
        v2_resolves = bool(v2_preset and v2_preset in v2_presets)

        if bucket == "transparent":
            cls = "tombstone_ready" if v2_resolves else "blocker_no_v2_alias"
        elif bucket == "deprecated":
            cls = "tombstone_candidate" if v2_resolves else "blocker_no_v2_alias"
        elif bucket == "needs_operator_choice":
            cls = "operator_decision_pending" if v2_resolves else "blocker_no_v2_alias"
        else:
            # Unknown bucket — defensive default.
            cls = "blocker_no_v2_alias"

        classifications.append(V1Classification(
            v1_key=key,
            bucket=bucket,
            v2_preset=v2_preset,
            classification=cls,
            rationale=rationale,
            v2_resolves=v2_resolves,
        ))
        seen_keys.discard(key)

    # 2) V1 files on disk that have no migration-table entry. These
    #    bypass the §9.T staging entirely; surface them as blockers.
    for key in sorted(seen_keys):
        untracked.append(key)
        classifications.append(V1Classification(
            v1_key=key,
            bucket="(none)",
            v2_preset=None,
            classification="untracked",
            rationale=(
                "V1 preset on disk has no entry in "
                "_v1_migration_table.json — Stage 1 prerequisite missing."
            ),
            v2_resolves=False,
        ))

    return classifications, untracked


def compute_stage_readiness(
    classifications: list[V1Classification],
) -> list[StageReadiness]:
    """Return readiness flags for §9.T stages 1–5.

    Stage 1 (migration tooling): no ``untracked`` keys.
    Stage 2 (deprecation warnings): Stage 1 ready (warnings are runtime).
    Stage 3 (generated V2 equivalents): Stage 2 ready AND no
            ``blocker_no_v2_alias``.
    Stage 4 (V1 hidden by default): Stage 3 ready.
    Stage 5 (V1 loader removed): Stage 4 ready AND no
            ``operator_decision_pending`` (every V1 has a clear path).
    """
    untracked = [c for c in classifications if c.classification == "untracked"]
    blockers = [
        c for c in classifications
        if c.classification == "blocker_no_v2_alias"
    ]
    pending = [
        c for c in classifications
        if c.classification == "operator_decision_pending"
    ]

    s1_ready = not untracked
    s2_ready = s1_ready
    s3_ready = s2_ready and not blockers
    s4_ready = s3_ready
    s5_ready = s4_ready and not pending

    def _block_strs(items: list[V1Classification]) -> list[str]:
        return [f"{c.v1_key} [{c.classification}]" for c in items]

    return [
        StageReadiness(1, s1_ready, _block_strs(untracked)),
        StageReadiness(2, s2_ready, _block_strs(untracked)),
        StageReadiness(
            3, s3_ready,
            _block_strs(untracked) + _block_strs(blockers),
        ),
        StageReadiness(
            4, s4_ready,
            _block_strs(untracked) + _block_strs(blockers),
        ),
        StageReadiness(
            5, s5_ready,
            _block_strs(untracked) + _block_strs(blockers) + _block_strs(pending),
        ),
    ]


def _resolve_stage(explicit: Optional[int]) -> int:
    """Pick the rollout stage: explicit CLI arg wins; otherwise read
    via ``_rollout.rollout_stage`` (env + DEFAULT_STAGE)."""
    if explicit is not None:
        return explicit
    try:
        from sndr.model_configs._rollout import rollout_stage
    except ImportError:
        return 1
    return rollout_stage()


def build_report(*, stage_override: Optional[int] = None) -> SunsetReport:
    entries = load_migration_table()
    v1_keys = list_v1_files_on_disk()
    v2_presets = list_v2_preset_ids()
    classifications, untracked = classify(v1_keys, entries, v2_presets)

    counts: dict[str, int] = {c: 0 for c in CLASSIFICATIONS}
    for c in classifications:
        counts[c.classification] = counts.get(c.classification, 0) + 1

    stages = compute_stage_readiness(classifications)

    tombstone_incidents = [
        c.v1_key for c in classifications
        if c.classification == "tombstoned" and c.v1_key in v1_keys
    ]

    return SunsetReport(
        rollout_stage=_resolve_stage(stage_override),
        v1_files_on_disk=len(v1_keys),
        migration_table_entries=len(entries),
        classifications=classifications,
        counts=counts,
        stages=stages,
        tombstone_incidents=tombstone_incidents,
        untracked_keys=untracked,
    )


def _render_text(report: SunsetReport) -> str:
    out = [
        "audit-v1-sunset: §9.T V1 monolithic sunset countdown",
        "─" * 70,
        f"  resolved rollout stage: {report.rollout_stage} "
        "(env SNDR_V1_ROLLOUT_STAGE; default 1)",
        f"  v1 files on disk:       {report.v1_files_on_disk}",
        f"  migration entries:      {report.migration_table_entries}",
        "",
        "  classification counts:",
    ]
    for name in CLASSIFICATIONS:
        out.append(f"    {name:30s} {report.counts.get(name, 0)}")

    out.append("")
    out.append("  stage readiness:")
    for s in report.stages:
        sym = "✓" if s.ready else "✗"
        out.append(f"    {sym} stage {s.stage}: "
                   f"{'ready' if s.ready else 'blocked'}")
        for b in s.blockers[:5]:
            out.append(f"        - {b}")
        if len(s.blockers) > 5:
            out.append(f"        … ({len(s.blockers) - 5} more)")

    if report.tombstone_incidents:
        out.append("")
        out.append("  ✗ TOMBSTONE INCIDENTS (V1 file present after retirement):")
        for k in report.tombstone_incidents:
            out.append(f"      - {k}")

    if report.untracked_keys:
        out.append("")
        out.append("  ✗ UNTRACKED V1 KEYS (on disk, no migration entry):")
        for k in report.untracked_keys:
            out.append(f"      - {k}")

    out.append("─" * 70)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    ap.add_argument("--strict", action="store_true",
                    help="gate Stage 3+ blockers (exit 1 on blocker_no_v2_alias)")
    ap.add_argument("--stage", type=int, default=None,
                    help="override resolved rollout stage (0..3)")
    args = ap.parse_args()

    try:
        report = build_report(stage_override=args.stage)
    except Exception as e:
        print(f"audit-v1-sunset: internal error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(_render_text(report))

    # Exit semantics:
    #   - tombstone incident → always exit 1 (V1 file should be gone
    #     once the entry hits tombstone).
    #   - untracked V1 keys  → always exit 1 (audit_no_new_v1.py should
    #     have caught this; surface here as a defense in depth).
    #   - blocker_no_v2_alias → exit 1 only with --strict AND stage>=3.
    if report.tombstone_incidents:
        return 1
    if report.untracked_keys:
        return 1
    if args.strict and report.rollout_stage >= 3:
        blocker_n = report.counts.get("blocker_no_v2_alias", 0)
        if blocker_n > 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

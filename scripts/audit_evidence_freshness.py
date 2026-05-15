#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evidence ledger freshness audit (Consolidated Roadmap §10.3 item 3).

Per §10.3 contract: the operator's `ROADMAP_EVIDENCE_LEDGER` must have
a recent entry — either within 7 days OR pointing at the current HEAD
commit. Stale ledger = aspirational "done" claims unbacked by repro.

The ledger lives under `docs/_internal/ROADMAP_EVIDENCE_LEDGER_*.md`
which is gitignored (Q4 internal-docs policy). In CI / fresh-checkout
contexts the file is absent — that's expected, NOT a violation. This
script's audit semantics:

  • Ledger absent  → emit "skipped" (rc=0). CI / clone contexts pass.
  • Ledger present → check that the newest dated entry is within
    `--max-age-days` (default 7) of today OR contains a literal
    reference to `HEAD` short SHA. Stale ledger = rc=1.

Entry header format (per template at top of every ledger file):

    ### YYYY-MM-DDTHH:MM±ZZZZ — short title

So freshness = max(parsed date) across all `### YYYY-MM-DD...` headers.

Exit codes:
  0 — fresh, OR ledger absent (operator-tier check skipped on CI).
  1 — ledger present but stale beyond --max-age-days.
  2 — ledger present but no dated entries found (malformed).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_GLOB = "docs/_internal/ROADMAP_EVIDENCE_LEDGER_*.md"

# Match `### 2026-05-13T18:25+0300 — short title` style headers (only date matters here).
ENTRY_RE = re.compile(r"^###\s*(\d{4}-\d{2}-\d{2})")


def _find_ledger() -> Path | None:
    matches = list(REPO_ROOT.glob(LEDGER_GLOB))
    if not matches:
        return None
    # newest by name (date in filename) — operators rotate ledger per quarter
    return sorted(matches)[-1]


def _newest_entry_date(text: str) -> _dt.date | None:
    newest: _dt.date | None = None
    for line in text.splitlines():
        m = ENTRY_RE.match(line)
        if not m:
            continue
        try:
            d = _dt.date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if newest is None or d > newest:
            newest = d
    return newest


def _current_short_sha() -> str | None:
    try:
        rc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if rc.returncode == 0:
            return rc.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def audit(max_age_days: int) -> dict:
    ledger = _find_ledger()
    if ledger is None:
        return {
            "ledger_path": None,
            "skipped": True,
            "reason": "no ledger file present (CI / fresh clone)",
        }
    text = ledger.read_text(encoding="utf-8")
    newest = _newest_entry_date(text)
    if newest is None:
        return {
            "ledger_path": ledger.relative_to(REPO_ROOT).as_posix(),
            "skipped": False,
            "newest_entry": None,
            "rc": 2,
            "reason": "ledger has no dated entries",
        }
    age_days = (_dt.date.today() - newest).days
    fresh_by_age = age_days <= max_age_days
    short_sha = _current_short_sha()
    fresh_by_sha = bool(short_sha and short_sha in text)
    return {
        "ledger_path": ledger.relative_to(REPO_ROOT).as_posix(),
        "skipped": False,
        "newest_entry": newest.isoformat(),
        "age_days": age_days,
        "max_age_days": max_age_days,
        "current_short_sha": short_sha,
        "fresh_by_age": fresh_by_age,
        "fresh_by_sha": fresh_by_sha,
        "rc": 0 if (fresh_by_age or fresh_by_sha) else 1,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-age-days", type=int, default=7,
        help="freshness threshold in days (default: 7).",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = audit(args.max_age_days)
    if args.json:
        import json
        print(json.dumps(report, indent=2, sort_keys=True))
        return int(report.get("rc", 0))

    print("audit-evidence-freshness")
    print("─" * 70)
    if report.get("skipped"):
        print(f"  · {report['reason']}")
        print("  ✓ skipped (operator-tier check)")
        return 0
    rc = report.get("rc", 0)
    print(f"  ledger:          {report['ledger_path']}")
    if report.get("newest_entry") is None:
        print("  newest entry:    (none — malformed)")
        print("  ✗ FAIL — ledger has no dated entries")
        return 2
    print(f"  newest entry:    {report['newest_entry']} ({report['age_days']} days old)")
    print(f"  threshold:       ≤{report['max_age_days']} days OR contains HEAD sha")
    print(f"  current sha:     {report['current_short_sha'] or '(unknown)'}")
    print(f"  fresh by age:    {report['fresh_by_age']}")
    print(f"  fresh by sha:    {report['fresh_by_sha']}")
    print()
    if rc == 0:
        print("  ✓ ledger fresh")
    else:
        print(
            f"  ✗ FAIL — ledger stale: last entry {report['age_days']} days old, "
            f"sha {report['current_short_sha']} not referenced"
        )
        print()
        print("  Fix: append a fresh entry per the ledger template,")
        print("       OR include the current short SHA in the most recent entry.")
    return rc


if __name__ == "__main__":
    sys.exit(main())

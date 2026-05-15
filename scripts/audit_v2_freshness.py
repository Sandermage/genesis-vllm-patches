#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 freshness gate — `make audit-v2-freshness`.

Each V2 model YAML carries a `last_validated: 'YYYY-MM-DD'` field —
the date when the model+patches+bench combination was last verified
end-to-end. Stale dates mean the published bench claims (TPS, decode
latency, tool-call quality) were measured against an older patch
set / vllm pin / methodology.

This gate enforces:
  • The field is parseable as an ISO date.
  • The date is not older than `max_age_days` (default: 180).
  • The date is not in the future (catches `'2099-...'` typos).

Why 180 days default: Genesis publishes a new bench wave every
~3 months; 180 days = 2 wave-cycles' grace. Operator overrides via
`--max-age-days` for tighter / looser policies.

Scope: only `kind: model` YAMLs carry `last_validated`. Hardware
configs are not bench-claim carriers (no freshness check). Profiles
carry `created` (immutable provenance), not validation timestamps.

Exit codes:
  0 — every model YAML has a parseable, fresh, non-future date
  1 — at least one model is stale / unparseable / future-dated
  2 — internal error
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "model"

DEFAULT_MAX_AGE_DAYS = 180


@dataclass
class FreshnessCheck:
    path: Path
    model_id: str
    last_validated_raw: str = ""
    last_validated: Optional[dt.date] = None
    age_days: Optional[int] = None
    status: str = "ok"   # ok | stale | future | unparseable | missing
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "ok"


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_iso_date(s) -> Optional[dt.date]:
    """Tolerate `'2026-05-12'` string, `2026-05-12` date object, or
    datetime."""
    if isinstance(s, dt.datetime):
        return s.date()
    if isinstance(s, dt.date):
        return s
    if isinstance(s, str):
        try:
            return dt.date.fromisoformat(s.strip())
        except ValueError:
            return None
    return None


def check_one_model(
    path: Path, *,
    today: dt.date,
    max_age_days: int,
) -> FreshnessCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return FreshnessCheck(
            path=path, model_id="?",
            status="unparseable",
            error=f"YAML parse error: {e}",
        )

    model_id = data.get("id", path.stem)
    raw = data.get("last_validated")
    if raw is None:
        return FreshnessCheck(
            path=path, model_id=model_id,
            status="missing",
            error="last_validated field absent",
        )

    parsed = _parse_iso_date(raw)
    if parsed is None:
        return FreshnessCheck(
            path=path, model_id=model_id,
            last_validated_raw=str(raw),
            status="unparseable",
            error=f"last_validated={raw!r} not a valid ISO date",
        )

    age = (today - parsed).days
    if age < 0:
        status = "future"
    elif age > max_age_days:
        status = "stale"
    else:
        status = "ok"

    return FreshnessCheck(
        path=path,
        model_id=model_id,
        last_validated_raw=str(raw),
        last_validated=parsed,
        age_days=age,
        status=status,
    )


def audit_v2_freshness(
    *,
    model_dir: Path = MODEL_DIR,
    today: Optional[dt.date] = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> list[FreshnessCheck]:
    today = today or dt.date.today()
    if not model_dir.is_dir():
        return []
    return [
        check_one_model(p, today=today, max_age_days=max_age_days)
        for p in sorted(model_dir.glob("*.yaml"))
    ]


# ─── Renderers ────────────────────────────────────────────────────────


def _render_text(
    results: list[FreshnessCheck], *,
    today: dt.date, max_age_days: int,
) -> str:
    lines = []
    lines.append(
        f"audit-v2-freshness: {len(results)} model YAML(s), "
        f"today={today.isoformat()}, max_age={max_age_days}d"
    )
    lines.append("─" * 70)
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.passed:
            lines.append(
                f"  {sym} {r.model_id:36s} "
                f"validated {r.last_validated_raw} "
                f"({r.age_days}d old)"
            )
        elif r.status == "stale":
            lines.append(
                f"  {sym} {r.model_id:36s} "
                f"STALE — validated {r.last_validated_raw} "
                f"({r.age_days}d > {max_age_days}d)"
            )
        elif r.status == "future":
            lines.append(
                f"  {sym} {r.model_id:36s} "
                f"FUTURE-DATED — {r.last_validated_raw} (age {r.age_days}d)"
            )
        else:
            lines.append(
                f"  {sym} {r.model_id:36s}  [{r.status}] {r.error}"
            )

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} model(s) fresh")
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: re-validate stale models and bump last_validated, "
            "or raise --max-age-days if the policy needs loosening."
        )
    return "\n".join(lines)


def _render_json(
    results: list[FreshnessCheck], *,
    today: dt.date, max_age_days: int,
) -> str:
    return json.dumps({
        "today": today.isoformat(),
        "max_age_days": max_age_days,
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "by_status": {
            status: sum(1 for r in results if r.status == status)
            for status in ("ok", "stale", "future", "unparseable", "missing")
        },
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.path),
                "last_validated": r.last_validated_raw or None,
                "age_days": r.age_days,
                "status": r.status,
                "passed": r.passed,
                "error": r.error or None,
            }
            for r in results
        ],
    }, indent=2, sort_keys=True)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON.")
    ap.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
                    help=f"Days before a model is considered stale "
                         f"(default: {DEFAULT_MAX_AGE_DAYS}).")
    ap.add_argument("--today", default=None,
                    help="Override today's date (ISO format, for testing).")
    args = ap.parse_args()

    if args.today:
        try:
            today = dt.date.fromisoformat(args.today)
        except ValueError:
            sys.stderr.write(f"--today: invalid ISO date {args.today!r}\n")
            return 2
    else:
        today = dt.date.today()

    results = audit_v2_freshness(today=today, max_age_days=args.max_age_days)
    if args.json:
        print(_render_json(results, today=today, max_age_days=args.max_age_days))
    else:
        print(_render_text(results, today=today, max_age_days=args.max_age_days))

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

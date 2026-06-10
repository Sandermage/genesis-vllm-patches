#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 / §6.10 V2 license + maintainer coverage gate.

Every V2 `kind: model` YAML must declare:

  • `license:` — one of the recognized SPDX identifiers (case-insensitive
    match against the ALLOWED_LICENSES frozen list).
  • `maintainer:` — non-empty string. We don't enforce a format (operator
    may use github username, full name, or email); we just require the
    field is present and non-blank.

Why model-only: hardware + profile don't carry distribution-licensable
content (they're shape/runtime config). Models carry the bench-claim +
patches matrix; license + maintainer have legal/attribution weight here.

Adding to ALLOWED_LICENSES requires explicit operator decision (this
file is the source of truth for what licenses the project accepts).

Exit codes:
  0 — every model has a valid license + non-empty maintainer
  1 — at least one missing/unknown
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "model"


# ─── Allowed SPDX-style identifiers ──────────────────────────────────
#
# Lowercase comparison after normalization. Add to this set when a new
# license becomes acceptable — that's an explicit operator decision,
# tracked via ledger entry.

ALLOWED_LICENSES: frozenset[str] = frozenset({
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "gpl-3.0",
    "lgpl-3.0",
    "mpl-2.0",
    # Project-local license tokens for non-SPDX checkpoints (Phase 5.2.B
    # 2026-05-22). The V2 schema currently uses bare lowercase strings,
    # not SPDX `LicenseRef-…` syntax, so a project-local token suffices.
    # Promote to formal SPDX `LicenseRef-Gemma-Terms-of-Use` if/when the
    # schema gains explicit support for that form.
    "gemma-license",
})


@dataclass
class LicenseCheck:
    path: Path
    model_id: str
    license_raw: Optional[str] = None
    license_ok: bool = False
    maintainer_raw: Optional[str] = None
    maintainer_ok: bool = False
    parse_error: str = ""
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            not self.parse_error
            and self.license_ok
            and self.maintainer_ok
        )


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _normalize_license(raw) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.strip().lower()


def check_one_model(path: Path) -> LicenseCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return LicenseCheck(
            path=path, model_id="?",
            parse_error=f"YAML parse error: {e}",
        )

    model_id = data.get("id", path.stem)
    lic_raw = data.get("license")
    maint_raw = data.get("maintainer")

    r = LicenseCheck(
        path=path,
        model_id=model_id,
        license_raw=lic_raw,
        maintainer_raw=maint_raw,
    )

    norm = _normalize_license(lic_raw)
    if not norm:
        r.reasons.append("license field missing or non-string")
    elif norm not in ALLOWED_LICENSES:
        r.reasons.append(
            f"license={lic_raw!r} not in ALLOWED_LICENSES "
            f"({sorted(ALLOWED_LICENSES)})"
        )
    else:
        r.license_ok = True

    if not isinstance(maint_raw, str) or not maint_raw.strip():
        r.reasons.append("maintainer field missing or empty")
    else:
        r.maintainer_ok = True

    return r


def audit_v2_license_coverage(
    model_dir: Path = MODEL_DIR,
) -> list[LicenseCheck]:
    if not model_dir.is_dir():
        return []
    return [check_one_model(p) for p in sorted(model_dir.glob("*.yaml"))]


def _render_text(results: list[LicenseCheck]) -> str:
    lines = [
        f"audit-v2-license-coverage: {len(results)} V2 model YAML(s)",
        f"  allowed licenses: {sorted(ALLOWED_LICENSES)}",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.parse_error:
            lines.append(f"  {sym} {r.model_id}: {r.parse_error}")
            continue
        lines.append(
            f"  {sym} {r.model_id:36s} "
            f"license={r.license_raw!r:18s} maintainer={r.maintainer_raw!r}"
        )
        for reason in r.reasons:
            lines.append(f"      - {reason}")

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} models have valid license + maintainer")
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: set `license:` to a known SPDX id (or add new license "
            "to ALLOWED_LICENSES with operator approval) and ensure "
            "`maintainer:` is a non-empty string."
        )
    return "\n".join(lines)


def _render_json(results: list[LicenseCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "allowed_licenses": sorted(ALLOWED_LICENSES),
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.path),
                "license": r.license_raw,
                "license_ok": r.license_ok,
                "maintainer": r.maintainer_raw,
                "maintainer_ok": r.maintainer_ok,
                "passed": r.passed,
                "reasons": r.reasons,
                "parse_error": r.parse_error or None,
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
    args = ap.parse_args()

    results = audit_v2_license_coverage()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

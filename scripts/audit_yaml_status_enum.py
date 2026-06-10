#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit gate — every builtin model YAML must declare `Status:` and
(for non-✅) `Caveats:` in its header block.

Pattern adopted from noonghunna/club-3090 compose convention
(see sndr_private/planning/audits/CLUB3090_CROSS_REFERENCE_2026-05-29_RU.md).
Their *mandatory + enumerated* lifecycle marker kills the previous
ambiguity where the absence of an explicit Status line could mean either
"validated production" or "forgot to fill in."

Status values (in header comment block, line-starts `# Status: <value>`):
  ✅ Production             — verify-full + bench + soak PASS
  ⚠️ Production w/ caveats — validated but constrained (Caveats REQUIRED)
  🧪 Experimental           — under validation (Caveats REQUIRED)
  👁️ Preview                — known quality issues (Caveats REQUIRED)
  ⏸️ Upstream-gated         — blocked on external dep (Caveats REQUIRED)
  🗑️ Deprecated             — removal pending (Caveats REQUIRED)

Invariants enforced
-------------------
  1. Header block (first 40 lines, comment-only) contains `# Status: <one of enum>`
  2. If Status ≠ ✅ Production, header block contains `# Caveats: <text ≥ 10 chars>`
  3. Enum value matches one of the canonical strings exactly (zero-tolerance to
     prevent emoji drift / lookalike characters; new-style ⚠ matches ⚠️
     because Pango/terminal render them identically).

Usage
-----
  python3 scripts/audit_yaml_status_enum.py           # human-readable report
  python3 scripts/audit_yaml_status_enum.py --strict  # exit 1 on any violation
  python3 scripts/audit_yaml_status_enum.py --json    # machine-readable

No torch / pyyaml / transformers imports — runs in CI on bare Python.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Canonical enum. Variation-Selector-16 ('️') after the symbol is OPTIONAL —
# both '⚠' and '⚠️' render as ⚠ in most terminals; we accept both.
_STATUS_OK = "✅ Production"
_STATUS_PROD_CAVEAT = "⚠️ Production w/ caveats"
_STATUS_PROD_CAVEAT_PLAIN = "⚠ Production w/ caveats"
_STATUS_EXPERIMENTAL = "\U0001F9EA Experimental"
_STATUS_PREVIEW = "\U0001F441️ Preview"
_STATUS_PREVIEW_PLAIN = "\U0001F441 Preview"
_STATUS_UPSTREAM_GATED = "⏸️ Upstream-gated"
_STATUS_UPSTREAM_GATED_PLAIN = "⏸ Upstream-gated"
_STATUS_DEPRECATED = "\U0001F5D1️ Deprecated"
_STATUS_DEPRECATED_PLAIN = "\U0001F5D1 Deprecated"

VALID_STATUSES = frozenset({
    _STATUS_OK,
    _STATUS_PROD_CAVEAT, _STATUS_PROD_CAVEAT_PLAIN,
    _STATUS_EXPERIMENTAL,
    _STATUS_PREVIEW, _STATUS_PREVIEW_PLAIN,
    _STATUS_UPSTREAM_GATED, _STATUS_UPSTREAM_GATED_PLAIN,
    _STATUS_DEPRECATED, _STATUS_DEPRECATED_PLAIN,
})

CAVEAT_REQUIRED_PREFIXES = (
    "⚠",            # ⚠ both variants
    "\U0001F9EA",        # 🧪
    "\U0001F441",        # 👁
    "⏸",            # ⏸
    "\U0001F5D1",        # 🗑
)

# Regex captures the value after `# Status:` until end-of-line; leading whitespace
# tolerated; comment marker `#` followed by 1+ spaces required.
_STATUS_RE = re.compile(r"^#\s+Status:\s*(.+?)\s*$", re.MULTILINE)
_CAVEAT_RE = re.compile(r"^#\s+Caveats?:\s*(.+?)\s*$", re.MULTILINE)

_HEADER_WINDOW_LINES = 40  # Status must land in first N lines


@dataclass(slots=True)
class Violation:
    yaml_path: Path
    code: str
    detail: str


def _header_block(text: str) -> str:
    """Return first N lines (header window)."""
    return "\n".join(text.splitlines()[:_HEADER_WINDOW_LINES])


def _audit_one(path: Path) -> list[Violation]:
    text = path.read_text(encoding="utf-8")
    header = _header_block(text)

    out: list[Violation] = []
    status_match = _STATUS_RE.search(header)
    if not status_match:
        out.append(Violation(
            path, "missing_status",
            f"no `# Status: <enum>` in first {_HEADER_WINDOW_LINES} lines",
        ))
        return out

    status_val = status_match.group(1).strip()
    if status_val not in VALID_STATUSES:
        out.append(Violation(
            path, "invalid_status",
            f"Status value {status_val!r} not in enum; "
            f"valid = ['✅ Production', '⚠️ Production w/ caveats', "
            f"'\U0001F9EA Experimental', '\U0001F441️ Preview', "
            f"'⏸️ Upstream-gated', '\U0001F5D1️ Deprecated']",
        ))
        return out

    # Caveat-required statuses must also carry `# Caveats:` line ≥ 10 chars.
    if status_val.startswith(CAVEAT_REQUIRED_PREFIXES):
        caveat_match = _CAVEAT_RE.search(header)
        if not caveat_match:
            out.append(Violation(
                path, "missing_caveats",
                f"Status {status_val!r} requires `# Caveats: <≥10 chars>` "
                f"in header window",
            ))
        elif len(caveat_match.group(1).strip()) < 10:
            out.append(Violation(
                path, "caveats_too_short",
                f"Caveats: {caveat_match.group(1)!r} <10 chars — be explicit",
            ))
    return out


def _enumerate_yamls(root: Path) -> list[Path]:
    """Find all builtin model YAMLs (excluding `_retired/`, `_deprecated/`,
    inherits-only profiles, hardware/profile/preset configs).

    v12.1 (2026-06-09): canonical path is ``sndr/model_configs/builtin/
    model``. Legacy ``vllm/sndr_core/model_configs/...`` was archived to
    ``sndr_private/archive/`` (commit 6bf9c04c). Falls back to the
    legacy path only if the canonical doesn't exist — keeps the script
    portable across the v11→v12 transition window.
    """
    canonical = root / "sndr" / "model_configs" / "builtin" / "model"
    if canonical.is_dir():
        return sorted(p for p in canonical.glob("*.yaml") if p.is_file())
    legacy = root / "sndr" / "model_configs" / "builtin" / "model"
    if legacy.is_dir():
        return sorted(p for p in legacy.glob("*.yaml") if p.is_file())
    return []


def _format_report(violations: list[Violation], total: int) -> str:
    lines: list[str] = []
    by_path: dict[Path, list[Violation]] = {}
    for v in violations:
        by_path.setdefault(v.yaml_path, []).append(v)
    if violations:
        lines.append(
            f"audit_yaml_status_enum: {len(violations)} violation(s) "
            f"across {len(by_path)}/{total} YAMLs"
        )
        for p, vs in sorted(by_path.items()):
            lines.append(f"  {p.relative_to(p.parents[5])}:")
            for v in vs:
                lines.append(f"    - [{v.code}] {v.detail}")
    else:
        lines.append(
            f"audit_yaml_status_enum: PASS — {total} YAMLs carry valid "
            f"Status enum"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any violation found (CI gate mode)")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of human-readable")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                    help="repo root (auto-detected)")
    args = ap.parse_args(argv)

    yamls = _enumerate_yamls(args.root)
    if not yamls:
        sys.stderr.write(
            "ERROR: no builtin model YAMLs found at "
            f"{args.root}/vllm/sndr_core/model_configs/builtin/model/\n"
        )
        return 2

    violations: list[Violation] = []
    for p in yamls:
        violations.extend(_audit_one(p))

    if args.json:
        print(json.dumps({
            "total_yamls": len(yamls),
            "violations": [
                {"path": str(v.yaml_path.relative_to(args.root)),
                 "code": v.code, "detail": v.detail}
                for v in violations
            ],
            "pass": not violations,
        }, indent=2))
    else:
        print(_format_report(violations, len(yamls)))

    if args.strict and violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 versions pin-format gate.

Verifies that `versions.vllm_pin_required` and `versions.genesis_pin_min`
in every V2 model YAML match the canonical pin-format regex:

  • `vllm_pin_required` — vllm pre-release pin form, e.g.:
        '0.20.2rc1.dev9+g01d4d1ad3'
        '0.20.2rc1.dev209+g5536fc0c0'
    Regex: `^\\d+\\.\\d+\\.\\d+(?:rc\\d+)?(?:\\.dev\\d+)?\\+g[0-9a-f]+$`
  • `genesis_pin_min` — genesis semver tag, e.g.:
        'v11.0.0'
        'v11.0.0+wave8'
    Regex: `^v\\d+\\.\\d+\\.\\d+(?:[+-][\\w.]+)?$`

Catches typos in pins (e.g. dropping the `+g<sha>` suffix, mistyping
the version number, mixing v/no-v prefix on genesis).

Exit codes:
  0 — every pin in every V2 model matches format
  1 — at least one typo'd pin
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "model"


VLLM_PIN_RE    = re.compile(r"^\d+\.\d+\.\d+(?:rc\d+)?(?:\.dev\d+)?\+g[0-9a-f]+$")
GENESIS_PIN_RE = re.compile(r"^v\d+\.\d+\.\d+(?:[+-][\w.]+)?$")


@dataclass
class PinCheck:
    path: Path
    model_id: str
    vllm_pin: Optional[str] = None
    genesis_pin: Optional[str] = None
    violations: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and not self.violations


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_one_model(path: Path) -> PinCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return PinCheck(path=path, model_id="?",
                      error=f"YAML parse error: {e}")
    model_id = data.get("id", path.stem)
    versions = data.get("versions") or {}
    vp = versions.get("vllm_pin_required")
    gp = versions.get("genesis_pin_min")

    violations: list[str] = []
    if vp is None:
        violations.append("vllm_pin_required is missing/None")
    elif not isinstance(vp, str) or not VLLM_PIN_RE.match(vp):
        violations.append(
            f"vllm_pin_required={vp!r} does not match "
            f"{VLLM_PIN_RE.pattern!r}"
        )

    if gp is None:
        violations.append("genesis_pin_min is missing/None")
    elif not isinstance(gp, str) or not GENESIS_PIN_RE.match(gp):
        violations.append(
            f"genesis_pin_min={gp!r} does not match "
            f"{GENESIS_PIN_RE.pattern!r}"
        )

    return PinCheck(
        path=path, model_id=model_id,
        vllm_pin=vp if isinstance(vp, str) else None,
        genesis_pin=gp if isinstance(gp, str) else None,
        violations=violations,
    )


def audit_v2_versions_pin_format(
    model_dir: Path = MODEL_DIR,
) -> list[PinCheck]:
    if not model_dir.is_dir():
        return []
    return [check_one_model(p) for p in sorted(model_dir.glob("*.yaml"))]


def _render_text(results: list[PinCheck]) -> str:
    lines = [
        f"audit-v2-versions-pin-format: {len(results)} model YAML(s)",
        f"  vllm pin regex:    {VLLM_PIN_RE.pattern}",
        f"  genesis pin regex: {GENESIS_PIN_RE.pattern}",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.model_id}: {r.error}")
            continue
        lines.append(
            f"  {sym} {r.model_id:36s}  "
            f"vllm={r.vllm_pin!r}  genesis={r.genesis_pin!r}"
        )
        for v in r.violations:
            lines.append(f"      ⚠ {v}")
    passed = sum(1 for r in results if r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} models clean")
    return "\n".join(lines)


def _render_json(results: list[PinCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "regex": {
            "vllm_pin_required": VLLM_PIN_RE.pattern,
            "genesis_pin_min":   GENESIS_PIN_RE.pattern,
        },
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.path),
                "vllm_pin": r.vllm_pin,
                "genesis_pin": r.genesis_pin,
                "passed": r.passed,
                "violations": r.violations,
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
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = audit_v2_versions_pin_format()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

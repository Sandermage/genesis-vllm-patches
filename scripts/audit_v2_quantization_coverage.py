#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 quantization + dtype coverage gate.

V2 model.{quantization, dtype} are passed straight to vLLM at launch.
A typo silently makes vLLM fall back to default (or fail to start, if
the value is rejected at the argparse layer). Either way: the operator
intent diverges from what runs.

Frozen allowed sets:

  • `quantization`: None / auto_round / gptq_marlin / awq / fp8 /
    bitsandbytes / awq_marlin (case-sensitive)
  • `dtype`: float16 / bfloat16 / float32 / auto

Exit codes:
  0 — every model's quantization + dtype in allowed set
  1 — at least one unknown value
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "model"


ALLOWED_QUANTIZATION: frozenset = frozenset({
    None,
    "auto_round",
    "gptq_marlin",
    "awq",
    "awq_marlin",
    "fp8",
    "bitsandbytes",
})

ALLOWED_DTYPE: frozenset = frozenset({
    "float16",
    "bfloat16",
    "float32",
    "auto",
})


@dataclass
class QuantCheck:
    path: Path
    model_id: str
    quantization: object = None
    dtype: object = None
    violations: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and not self.violations


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_one_model(path: Path) -> QuantCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return QuantCheck(path=path, model_id="?",
                         error=f"YAML parse error: {e}")
    model_id = data.get("id", path.stem)
    q = data.get("quantization", "__MISSING__")
    d = data.get("dtype", "__MISSING__")

    r = QuantCheck(path=path, model_id=model_id, quantization=q, dtype=d)
    if q == "__MISSING__":
        r.violations.append("quantization field absent")
    elif q not in ALLOWED_QUANTIZATION:
        r.violations.append(
            f"quantization={q!r} not in {sorted([v for v in ALLOWED_QUANTIZATION if v is not None])}"
        )

    if d == "__MISSING__":
        r.violations.append("dtype field absent")
    elif d not in ALLOWED_DTYPE:
        r.violations.append(
            f"dtype={d!r} not in {sorted(ALLOWED_DTYPE)}"
        )
    return r


def audit_v2_quantization_coverage(
    model_dir: Path = MODEL_DIR,
) -> list[QuantCheck]:
    if not model_dir.is_dir():
        return []
    return [check_one_model(p) for p in sorted(model_dir.glob("*.yaml"))]


def _render_text(results: list[QuantCheck]) -> str:
    lines = [
        f"audit-v2-quantization-coverage: {len(results)} model YAML(s)",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.model_id}: {r.error}")
            continue
        lines.append(
            f"  {sym} {r.model_id:36s}  "
            f"quant={r.quantization!r:18s}  dtype={r.dtype!r}"
        )
        for v in r.violations:
            lines.append(f"      ⚠ {v}")
    passed = sum(1 for r in results if r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} models clean")
    return "\n".join(lines)


def _render_json(results: list[QuantCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "allowed_quantization": sorted(
            [v for v in ALLOWED_QUANTIZATION if v is not None],
        ),
        "allowed_dtype": sorted(ALLOWED_DTYPE),
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.path),
                "quantization": r.quantization,
                "dtype": r.dtype,
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
    results = audit_v2_quantization_coverage()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 context-length sanity gate.

Verifies `hardware.sizing.max_model_len` and
`hardware.sizing.max_num_batched_tokens` are within plausible bounds
and consistent with each other:

  • max_model_len ∈ [1_024, 2_097_152]    (1K..2M tokens)
  • max_num_batched_tokens ∈ [256, 65_536] (chunked prefill chunk size)
  • max_num_batched_tokens ≤ max_model_len (a single chunk can't
    exceed the model's context window)

Out-of-range values catch typos like swapping K/M magnitudes
(`max_model_len: 320` instead of `320000`) or 0-byte values.

Exit codes:
  0 — every hardware YAML's sizing values are sane
  1 — at least one violation
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HARDWARE_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "hardware"


MAX_MODEL_LEN_MIN     = 1_024
MAX_MODEL_LEN_MAX     = 2_097_152      # 2M tokens — well above any current model
MAX_BATCH_TOK_MIN     = 256
MAX_BATCH_TOK_MAX     = 65_536


@dataclass
class CtxCheck:
    path: Path
    hardware_id: str
    max_model_len: object = None
    max_num_batched_tokens: object = None
    violations: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and not self.violations


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def check_one_hardware(path: Path) -> CtxCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return CtxCheck(path=path, hardware_id="?",
                       error=f"YAML parse error: {e}")
    hardware_id = data.get("id", path.stem)
    sz = data.get("sizing") or {}
    mml = sz.get("max_model_len")
    mbt = sz.get("max_num_batched_tokens")

    r = CtxCheck(
        path=path, hardware_id=hardware_id,
        max_model_len=mml, max_num_batched_tokens=mbt,
    )

    if not isinstance(mml, int) or isinstance(mml, bool):
        r.violations.append(f"max_model_len={mml!r} not int")
    elif mml < MAX_MODEL_LEN_MIN or mml > MAX_MODEL_LEN_MAX:
        r.violations.append(
            f"max_model_len={mml} outside [{MAX_MODEL_LEN_MIN}..{MAX_MODEL_LEN_MAX}]"
        )

    if not isinstance(mbt, int) or isinstance(mbt, bool):
        r.violations.append(f"max_num_batched_tokens={mbt!r} not int")
    elif mbt < MAX_BATCH_TOK_MIN or mbt > MAX_BATCH_TOK_MAX:
        r.violations.append(
            f"max_num_batched_tokens={mbt} outside "
            f"[{MAX_BATCH_TOK_MIN}..{MAX_BATCH_TOK_MAX}]"
        )

    if (
        isinstance(mml, int) and isinstance(mbt, int)
        and not isinstance(mml, bool) and not isinstance(mbt, bool)
    ):
        if mbt > mml:
            r.violations.append(
                f"max_num_batched_tokens={mbt} > max_model_len={mml} "
                "(chunk can't exceed context)"
            )
    return r


def audit_v2_context_length_sanity(
    hw_dir: Path = HARDWARE_DIR,
) -> list[CtxCheck]:
    if not hw_dir.is_dir():
        return []
    return [check_one_hardware(p) for p in sorted(hw_dir.glob("*.yaml"))]


def _render_text(results: list[CtxCheck]) -> str:
    lines = [
        f"audit-v2-context-length-sanity: {len(results)} hardware YAML(s)",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.hardware_id}: {r.error}")
            continue
        lines.append(
            f"  {sym} {r.hardware_id:36s}  "
            f"ctx={r.max_model_len}  batch={r.max_num_batched_tokens}"
        )
        for v in r.violations:
            lines.append(f"      ⚠ {v}")
    passed = sum(1 for r in results if r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} hardware files clean")
    return "\n".join(lines)


def _render_json(results: list[CtxCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "bounds": {
            "max_model_len": [MAX_MODEL_LEN_MIN, MAX_MODEL_LEN_MAX],
            "max_num_batched_tokens": [MAX_BATCH_TOK_MIN, MAX_BATCH_TOK_MAX],
        },
        "results": [
            {
                "hardware_id": r.hardware_id,
                "path": _rel(r.path),
                "max_model_len": r.max_model_len,
                "max_num_batched_tokens": r.max_num_batched_tokens,
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
    results = audit_v2_context_length_sanity()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

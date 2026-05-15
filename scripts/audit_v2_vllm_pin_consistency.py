#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 vllm-pin consistency gate.

For every V2 `kind: model` YAML that names a baseline JSON via
`versions.reference_metrics_ref`, the model's `versions.vllm_pin_required`
must match the baseline's recorded vllm version. Otherwise the bench
claims live at a different pin than the model declares — the F-018
class of drift that the V1 yaml comment in `a5000-2x-35b-prod.yaml`
explicitly called out.

Baseline JSON schema variant tolerance: vllm_version may live at the
top level (`vllm_version: "..."`), inside an object
(`vllm_version: {parsed: {vllm_version: "..."}}`), or under config
(`config.vllm_pin`). We walk known paths in order.

Exit codes:
  0 — every model with a baseline ref agrees on the vllm pin
  1 — at least one model.vllm_pin_required ≠ baseline.vllm_version
  2 — internal error (baseline file missing / unreadable)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "model"


@dataclass
class PinCheck:
    path: Path
    model_id: str
    yaml_pin: Optional[str] = None
    baseline_ref: Optional[str] = None
    baseline_pin: Optional[str] = None
    skipped_reason: str = ""    # populated when there's no baseline to compare
    error: str = ""

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if self.skipped_reason:
            return True   # vacuously — no claim to check
        return self.yaml_pin == self.baseline_pin


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _walk_string(d, path: list[str]):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, str) else None


def _extract_baseline_vllm_version(d: dict) -> Optional[str]:
    """Try known schema variants in order. Return first non-None hit."""
    for path in (
        ["vllm_version"],
        ["vllm_pin"],
        ["vllm_version", "parsed", "vllm_version"],
        ["parsed", "vllm_version"],
        ["config", "vllm_pin"],
        ["headline", "vllm_pin"],
        ["summary", "vllm_pin"],
    ):
        v = _walk_string(d, path)
        if v:
            return v
    return None


def check_one_model(path: Path) -> PinCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return PinCheck(
            path=path, model_id="?",
            error=f"YAML parse error: {e}",
        )
    model_id = data.get("id", path.stem)
    versions = data.get("versions") or {}
    yaml_pin = versions.get("vllm_pin_required")
    ref = versions.get("reference_metrics_ref")

    r = PinCheck(
        path=path,
        model_id=model_id,
        yaml_pin=yaml_pin,
        baseline_ref=ref,
    )

    if not ref:
        r.skipped_reason = "no reference_metrics_ref — no baseline to compare"
        return r

    ref_path = (REPO_ROOT / str(ref)).resolve()
    if not ref_path.is_file():
        r.error = f"baseline file not found: {ref}"
        return r

    try:
        bdata = json.loads(ref_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        r.error = f"baseline parse error: {e}"
        return r

    baseline_pin = _extract_baseline_vllm_version(bdata)
    if not baseline_pin:
        r.error = "baseline file has no recognized vllm version field"
        return r
    r.baseline_pin = baseline_pin
    return r


def audit_v2_vllm_pin_consistency(
    model_dir: Path = MODEL_DIR,
) -> list[PinCheck]:
    if not model_dir.is_dir():
        return []
    return [check_one_model(p) for p in sorted(model_dir.glob("*.yaml"))]


def _render_text(results: list[PinCheck]) -> str:
    lines = [f"audit-v2-vllm-pin-consistency: {len(results)} model YAML(s)",
             "─" * 70]
    checked = skipped = 0
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.model_id}: {r.error}")
            continue
        if r.skipped_reason:
            skipped += 1
            lines.append(
                f"  · {r.model_id:36s} [skipped] {r.skipped_reason}"
            )
            continue
        checked += 1
        if r.passed:
            lines.append(
                f"  {sym} {r.model_id:36s} pin={r.yaml_pin} == baseline"
            )
        else:
            lines.append(
                f"  {sym} {r.model_id:36s} "
                f"yaml={r.yaml_pin!r} vs baseline={r.baseline_pin!r}"
            )
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(
        f"  {passed}/{len(results)} models pass  "
        f"({checked} compared, {skipped} skipped — no baseline)"
    )
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: either bump model.versions.vllm_pin_required to "
            "match the baseline, or re-snap baseline against the current pin."
        )
    return "\n".join(lines)


def _render_json(results: list[PinCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.path),
                "yaml_pin": r.yaml_pin,
                "baseline_ref": r.baseline_ref,
                "baseline_pin": r.baseline_pin,
                "skipped_reason": r.skipped_reason or None,
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
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = audit_v2_vllm_pin_consistency()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

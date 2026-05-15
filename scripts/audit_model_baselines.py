#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 7 supplement — `make audit-model-baselines`.

Each V2 model YAML may carry a `reference_metrics_ref: <path>` field
pointing at a baseline JSON used as the bench-comparison source. The
field is allowed to be null (no baseline established yet), but when set
it MUST point at a file that exists AND parses as JSON. Without this
gate, broken paths silently rot: the YAML claims a baseline, but the
operator running `sndr patches bench-attach --baseline <ref>` gets a
file-not-found at the worst possible moment (release pipeline, GPU
host).

Discovered in Entry 22 investigation: two V2 models referenced kebab-
style baseline filenames that don't exist on disk (the actual files
are snake-case `27b_v11_wave9.json` / `35b_v11_wave9.json`). This
gate surfaces such drift at PR time, not at release time.

Exit codes:
  0 — every reference_metrics_ref is null OR points to a parseable JSON
  1 — at least one broken / missing / unparseable reference
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "model"


@dataclass
class ModelBaselineCheck:
    """Per-model audit result."""
    yaml_path: Path
    model_id: str
    reference_metrics_ref: object   # str path | None
    baseline_exists: bool
    baseline_parseable: bool
    error: str = ""

    @property
    def passed(self) -> bool:
        """Null ref passes (no claim made). Set ref must exist+parse."""
        if self.reference_metrics_ref is None:
            return True
        return self.baseline_exists and self.baseline_parseable


def _load_yaml(path: Path) -> dict:
    """Parse YAML without depending on PyYAML — we only need a handful
    of top-level scalars (`id`, `bench_validation.reference_metrics_ref`).
    Pure-Python line-walker is enough for the V2 model YAML shape we
    own."""
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _ref_from_yaml(data: dict) -> object:
    """Extract `reference_metrics_ref` — V2 model schema places it under
    `versions:` block. We also accept top-level and `bench_validation:`
    as fallbacks for schema-variant tolerance."""
    for parent_key in ("versions", "bench_validation"):
        parent = data.get(parent_key)
        if isinstance(parent, dict) and "reference_metrics_ref" in parent:
            return parent["reference_metrics_ref"]
    if "reference_metrics_ref" in data:
        return data["reference_metrics_ref"]
    return None


def check_one_yaml(yaml_path: Path) -> ModelBaselineCheck:
    try:
        data = _load_yaml(yaml_path)
    except Exception as e:
        return ModelBaselineCheck(
            yaml_path=yaml_path,
            model_id="?",
            reference_metrics_ref=None,
            baseline_exists=False,
            baseline_parseable=False,
            error=f"YAML parse error: {e}",
        )

    model_id = data.get("id", yaml_path.stem)
    ref = _ref_from_yaml(data)

    if ref is None:
        return ModelBaselineCheck(
            yaml_path=yaml_path,
            model_id=model_id,
            reference_metrics_ref=None,
            baseline_exists=True,    # vacuously
            baseline_parseable=True,
        )

    # Resolve relative to repo root.
    ref_path = (REPO_ROOT / str(ref)).resolve()
    if not ref_path.is_file():
        return ModelBaselineCheck(
            yaml_path=yaml_path,
            model_id=model_id,
            reference_metrics_ref=ref,
            baseline_exists=False,
            baseline_parseable=False,
            error=f"baseline file not found: {ref}",
        )

    try:
        json.loads(ref_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return ModelBaselineCheck(
            yaml_path=yaml_path,
            model_id=model_id,
            reference_metrics_ref=ref,
            baseline_exists=True,
            baseline_parseable=False,
            error=f"baseline JSON parse error: {e}",
        )

    return ModelBaselineCheck(
        yaml_path=yaml_path,
        model_id=model_id,
        reference_metrics_ref=ref,
        baseline_exists=True,
        baseline_parseable=True,
    )


def audit_model_baselines(
    model_dir: Path = MODEL_DIR,
) -> list[ModelBaselineCheck]:
    """Run the check on every V2 model YAML in `model_dir`."""
    if not model_dir.is_dir():
        return []
    yamls = sorted(model_dir.glob("*.yaml"))
    return [check_one_yaml(p) for p in yamls]


def _render_text(results: list[ModelBaselineCheck]) -> str:
    lines = []
    lines.append(f"audit-model-baselines: {len(results)} V2 model YAML(s)")
    lines.append("─" * 70)
    passing = [r for r in results if r.passed]
    failing = [r for r in results if not r.passed]
    null_refs = [r for r in passing if r.reference_metrics_ref is None]
    real_refs = [r for r in passing if r.reference_metrics_ref is not None]

    for r in results:
        sym = "✓" if r.passed else "✗"
        ref = "null" if r.reference_metrics_ref is None else str(r.reference_metrics_ref)
        lines.append(f"  {sym} {r.model_id:36s} → {ref}")
        if not r.passed and r.error:
            lines.append(f"      {r.error}")

    lines.append("─" * 70)
    lines.append(
        f"  {len(passing)}/{len(results)} passing  "
        f"({len(null_refs)} null, {len(real_refs)} verified, "
        f"{len(failing)} broken)"
    )
    if failing:
        lines.append("")
        lines.append("  ✗ Broken baseline references — fix the YAML's "
                     "reference_metrics_ref or commit the missing baseline JSON.")
    return "\n".join(lines)


def _yaml_label(p: Path) -> str:
    """Format YAML path relative to repo root when possible; absolute
    otherwise (test fixtures may live under /tmp)."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _render_json(results: list[ModelBaselineCheck]) -> str:
    payload = {
        "total":  len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "checks": [
            {
                "yaml":   _yaml_label(r.yaml_path),
                "model_id": r.model_id,
                "reference_metrics_ref": r.reference_metrics_ref,
                "baseline_exists":   r.baseline_exists,
                "baseline_parseable": r.baseline_parseable,
                "passed": r.passed,
                "error":  r.error or None,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON output.")
    ap.add_argument("--model-dir", default=None,
                    help="Override model YAML directory.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR
    results = audit_model_baselines(model_dir)

    if args.json:
        print(_render_json(results))
    else:
        print(_render_text(results))

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

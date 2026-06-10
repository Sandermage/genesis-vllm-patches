#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 cross-reference gate — `make audit-v2-cross-reference`.

V2 layered config has TWO sources of cross-references between files:

  1. `profile.parent_model: <id>` — every profile names a model it's
     a delta of. The named id must exist as `model/<id>.yaml`.
  2. `preset.{model, hardware, profile}: <id>` — every alias preset
     names a triplet of refs that must all resolve.

If a referenced id doesn't exist on disk, `load_alias` will fail at
compose time (caught by E25's `audit-configs`) — but only for THAT
preset. Profile `parent_model` errors are silent unless someone reads
the profile (no compose surface). This gate covers both classes
upfront, with per-field diagnostics.

Exit codes:
  0 — every cross-reference resolves
  1 — at least one broken reference
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin"

MODEL_DIR    = BUILTIN_DIR / "model"
HARDWARE_DIR = BUILTIN_DIR / "hardware"
PROFILE_DIR  = BUILTIN_DIR / "profile"
PRESETS_DIR  = BUILTIN_DIR / "presets"


@dataclass
class RefCheck:
    layer: str               # "profile" | "preset"
    label: str               # source file identifier
    field_name: str          # which field carries the ref
    ref_value: str           # the id that was referenced
    target_layer: str        # "model" | "hardware" | "profile"
    resolved: bool = False
    parse_error: str = ""

    @property
    def passed(self) -> bool:
        return not self.parse_error and self.resolved


# ─── Helpers ──────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _collect_ids(layer_dir: Path) -> set[str]:
    if not layer_dir.is_dir():
        return set()
    return {p.stem for p in layer_dir.glob("*.yaml")}


def _check_profiles(model_ids: set[str]) -> list[RefCheck]:
    out: list[RefCheck] = []
    if not PROFILE_DIR.is_dir():
        return out
    for yp in sorted(PROFILE_DIR.glob("*.yaml")):
        try:
            data = _load_yaml(yp)
        except Exception as e:
            out.append(RefCheck(
                layer="profile", label=yp.stem,
                field_name="parent_model",
                ref_value="", target_layer="model",
                parse_error=f"YAML parse error: {e}",
            ))
            continue
        pm = data.get("parent_model")
        out.append(RefCheck(
            layer="profile",
            label=yp.stem,
            field_name="parent_model",
            ref_value=str(pm) if pm else "",
            target_layer="model",
            resolved=bool(pm) and pm in model_ids,
        ))
    return out


def _check_presets(
    model_ids: set[str],
    hw_ids: set[str],
    profile_ids: set[str],
) -> list[RefCheck]:
    out: list[RefCheck] = []
    if not PRESETS_DIR.is_dir():
        return out
    for ap in sorted(PRESETS_DIR.glob("*.yaml")):
        try:
            data = _load_yaml(ap)
        except Exception as e:
            for field_name, target in (
                ("model", "model"), ("hardware", "hardware"),
                ("profile", "profile"),
            ):
                out.append(RefCheck(
                    layer="preset", label=ap.stem,
                    field_name=field_name,
                    ref_value="", target_layer=target,
                    parse_error=f"YAML parse error: {e}",
                ))
            continue
        for field_name, target_set, target_layer in (
            ("model",    model_ids,   "model"),
            ("hardware", hw_ids,      "hardware"),
            ("profile",  profile_ids, "profile"),
        ):
            v = data.get(field_name)
            out.append(RefCheck(
                layer="preset",
                label=ap.stem,
                field_name=field_name,
                ref_value=str(v) if v else "",
                target_layer=target_layer,
                resolved=bool(v) and v in target_set,
            ))
    return out


def audit_v2_cross_reference() -> list[RefCheck]:
    model_ids   = _collect_ids(MODEL_DIR)
    hw_ids      = _collect_ids(HARDWARE_DIR)
    profile_ids = _collect_ids(PROFILE_DIR)
    return _check_profiles(model_ids) + _check_presets(
        model_ids, hw_ids, profile_ids,
    )


# ─── Renderers ────────────────────────────────────────────────────────


def _render_text(results: list[RefCheck]) -> str:
    lines = [
        f"audit-v2-cross-reference: {len(results)} ref(s) checked",
        "─" * 70,
    ]
    by_layer: dict[str, list[RefCheck]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r)
    for layer in ("profile", "preset"):
        rs = by_layer.get(layer, [])
        if not rs:
            continue
        lines.append(f"  ── {layer} refs ({len(rs)}) ──")
        for r in rs:
            sym = "✓" if r.passed else "✗"
            if r.parse_error:
                lines.append(
                    f"    {sym} {r.label}: {r.parse_error}"
                )
            elif r.resolved:
                lines.append(
                    f"    {sym} {r.label} {r.field_name}={r.ref_value}"
                )
            else:
                lines.append(
                    f"    {sym} {r.label} {r.field_name}={r.ref_value!r} "
                    f"→ no such {r.target_layer}"
                )
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} refs resolve")
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: either correct the ref id or create the target file."
        )
    return "\n".join(lines)


def _render_json(results: list[RefCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "refs": [
            {
                "layer": r.layer,
                "label": r.label,
                "field": r.field_name,
                "ref_value": r.ref_value,
                "target_layer": r.target_layer,
                "resolved": r.resolved,
                "passed": r.passed,
                "parse_error": r.parse_error or None,
            }
            for r in results
        ],
    }, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    results = audit_v2_cross_reference()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

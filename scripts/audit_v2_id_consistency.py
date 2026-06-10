#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 id-filename consistency gate — `make audit-v2-id-consistency`.

V2 alias presets resolve files by id:

    preset.model    → model_configs/builtin/model/<id>.yaml
    preset.hardware → model_configs/builtin/hardware/<id>.yaml
    preset.profile  → model_configs/builtin/profile/<id>.yaml

If the YAML's `id:` field doesn't match its filename stem, the alias
resolver looks up the WRONG file (or fails). The drift is silent at
compose time — `load_alias` may resolve to an unrelated file with the
matching stem.

This gate enforces: `data["id"] == path.stem` for every V2
model/hardware/profile YAML. (Preset YAMLs don't have an `id` field —
they're alias triplets.)

Exit codes:
  0 — every (id, filename stem) pair matches
  1 — at least one mismatch
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

LAYERS_WITH_ID: tuple[str, ...] = ("model", "hardware", "profile")


@dataclass
class IdCheck:
    layer: str
    path: Path
    filename_stem: str
    yaml_id: str = ""
    parse_error: str = ""

    @property
    def passed(self) -> bool:
        return (
            not self.parse_error
            and self.yaml_id == self.filename_stem
        )


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _check_file(path: Path, layer: str) -> IdCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return IdCheck(
            layer=layer, path=path,
            filename_stem=path.stem,
            parse_error=f"YAML parse error: {e}",
        )
    if not isinstance(data, dict):
        return IdCheck(
            layer=layer, path=path,
            filename_stem=path.stem,
            parse_error="top-level is not a mapping",
        )
    return IdCheck(
        layer=layer,
        path=path,
        filename_stem=path.stem,
        yaml_id=str(data.get("id", "")),
    )


def audit_v2_id_consistency() -> list[IdCheck]:
    out: list[IdCheck] = []
    for layer in LAYERS_WITH_ID:
        layer_dir = BUILTIN_DIR / layer
        if not layer_dir.is_dir():
            continue
        for yp in sorted(layer_dir.glob("*.yaml")):
            out.append(_check_file(yp, layer))
    return out


def _render_text(results: list[IdCheck]) -> str:
    lines = [
        f"audit-v2-id-consistency: {len(results)} V2 YAML(s) across "
        f"{len(LAYERS_WITH_ID)} id-carrying layers",
        "─" * 70,
    ]
    by_layer: dict[str, list[IdCheck]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r)
    for layer in LAYERS_WITH_ID:
        rs = by_layer.get(layer, [])
        if not rs:
            continue
        lines.append(f"  ── {layer} ({len(rs)} entries) ──")
        for r in rs:
            sym = "✓" if r.passed else "✗"
            if r.parse_error:
                lines.append(f"    {sym} {r.filename_stem}: {r.parse_error}")
            elif r.yaml_id == r.filename_stem:
                lines.append(f"    {sym} {r.filename_stem}")
            else:
                lines.append(
                    f"    {sym} {r.filename_stem}: id={r.yaml_id!r} "
                    f"(mismatch — should equal stem)"
                )
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} entries match")
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: either rename the file to match `id:` or update "
            "the `id:` field to match the filename stem."
        )
    return "\n".join(lines)


def _render_json(results: list[IdCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "results": [
            {
                "layer": r.layer,
                "path": _rel(r.path),
                "filename_stem": r.filename_stem,
                "yaml_id": r.yaml_id,
                "passed": r.passed,
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

    results = audit_v2_id_consistency()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

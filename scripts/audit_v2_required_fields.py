#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 schema fields gate — `make audit-v2-required-fields`.

Each V2 layer (model / hardware / profile / preset-alias) has a stable
set of top-level fields that ALL committed YAMLs in that layer share.
This gate codifies that set as a frozen schema. Any V2 YAML missing one
of those fields fails the gate at PR time, before the field gets
silently treated as `None` at compose time.

What this catches that other gates don't:

  • `audit-configs`           — compose succeeds even with missing fields
    (the composer fills defaults).
  • `audit-launch-coverage`   — mounts + env only, not top-level schema.
  • `audit-v2-env-keys`       — patch keys only, not structural fields.

Per-layer required field set was derived from analysis of every
committed V2 YAML (Entry 27 survey). The current state IS the
canonical schema; this gate is the regression anchor.

Layers:

  • **model**    — `builtin/model/*.yaml`,    13 required top-level fields
  • **hardware** — `builtin/hardware/*.yaml`,  9 required top-level fields
  • **profile**  — `builtin/profile/*.yaml`,   9 required top-level fields
  • **preset**   — `builtin/presets/*.yaml`,   3 required top-level fields

Exit codes:
  0 — every YAML in every layer satisfies its required-field schema
  1 — at least one missing required field
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin"


# ─── Per-layer required field schemas (frozen by Entry 27) ────────────
#
# Each set is the intersection of fields present in EVERY committed
# YAML of that layer at E27 freeze time. Adding a new file to the layer
# implicitly enforces the schema; removing a required field requires
# updating this set + a ledger entry explaining why.

REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    # V2 model YAML schema. Drops `notes` (operator-optional commentary).
    "model": frozenset({
        "schema_version", "kind", "id", "title", "maintainer",
        "last_validated", "license",
        "model_path", "served_model_name", "dtype",
        "quantization",   # may be `null`, but the field itself is required
        "trust_remote_code",
        "capabilities", "requires", "versions", "patches",
    }),
    # V2 hardware YAML schema.
    "hardware": frozenset({
        "schema_version", "kind", "id", "title", "maintainer",
        "hardware", "sizing", "runtime", "system_env",
    }),
    # V2 profile YAML schema. `created` is the immutable provenance
    # timestamp (vs. `last_validated` which is a model bench date).
    "profile": frozenset({
        "schema_version", "kind", "id", "maintainer",
        "parent_model", "status", "created",
        "patches_delta", "promotion",
    }),
    # V2 preset/alias schema: just a 3-tuple of references. The whole
    # design choice for V2 is "alias = pointer triplet".
    "preset": frozenset({
        "model", "hardware", "profile",
    }),
}


# ─── Result types ─────────────────────────────────────────────────────


@dataclass
class FileFieldsCheck:
    layer: str
    path: Path
    yaml_id: str
    present_fields: set[str] = field(default_factory=set)
    missing_fields: list[str] = field(default_factory=list)
    parse_error: str = ""

    @property
    def passed(self) -> bool:
        return not self.parse_error and not self.missing_fields


# ─── Walkers ──────────────────────────────────────────────────────────


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _check_file(path: Path, layer: str) -> FileFieldsCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return FileFieldsCheck(
            layer=layer, path=path, yaml_id="?",
            parse_error=f"YAML parse error: {e}",
        )
    if not isinstance(data, dict):
        return FileFieldsCheck(
            layer=layer, path=path, yaml_id="?",
            parse_error="top-level is not a mapping",
        )

    required = REQUIRED_FIELDS[layer]
    present = set(data.keys())
    yaml_id = data.get("id", path.stem)
    missing = sorted(required - present)
    return FileFieldsCheck(
        layer=layer,
        path=path,
        yaml_id=yaml_id,
        present_fields=present,
        missing_fields=missing,
    )


def audit_v2_required_fields() -> list[FileFieldsCheck]:
    out: list[FileFieldsCheck] = []
    for layer, subdir in (
        ("model",    BUILTIN_DIR / "model"),
        ("hardware", BUILTIN_DIR / "hardware"),
        ("profile",  BUILTIN_DIR / "profile"),
        ("preset",   BUILTIN_DIR / "presets"),
    ):
        if not subdir.is_dir():
            continue
        for yp in sorted(subdir.glob("*.yaml")):
            out.append(_check_file(yp, layer))
    return out


# ─── Renderers ────────────────────────────────────────────────────────


def _render_text(results: list[FileFieldsCheck]) -> str:
    lines = []
    lines.append(f"audit-v2-required-fields: {len(results)} V2 YAML(s) "
                 f"across 4 layers")
    lines.append("─" * 70)

    by_layer: dict[str, list[FileFieldsCheck]] = {}
    for r in results:
        by_layer.setdefault(r.layer, []).append(r)

    for layer in ("model", "hardware", "profile", "preset"):
        rs = by_layer.get(layer, [])
        if not rs:
            continue
        req = sorted(REQUIRED_FIELDS[layer])
        lines.append(
            f"  ── {layer} layer ({len(rs)} entries, "
            f"{len(req)} required fields) ──"
        )
        for r in rs:
            sym = "✓" if r.passed else "✗"
            lines.append(f"    {sym} {r.yaml_id}")
            if r.parse_error:
                lines.append(f"        ! {r.parse_error}")
            if r.missing_fields:
                lines.append(
                    f"        missing: {r.missing_fields}"
                )

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} entries satisfy schema")
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: add the missing field(s) to the YAML. See "
            "audit_v2_required_fields.py:REQUIRED_FIELDS for canonical schema."
        )
    return "\n".join(lines)


def _render_json(results: list[FileFieldsCheck]) -> str:
    payload = {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "required_fields": {
            layer: sorted(REQUIRED_FIELDS[layer])
            for layer in ("model", "hardware", "profile", "preset")
        },
        "by_layer": {
            layer: {
                "entries": sum(1 for r in results if r.layer == layer),
                "failed":  sum(1 for r in results
                               if r.layer == layer and not r.passed),
            }
            for layer in ("model", "hardware", "profile", "preset")
        },
        "results": [
            {
                "layer": r.layer,
                "path": _rel(r.path),
                "yaml_id": r.yaml_id,
                "passed": r.passed,
                "missing_fields": r.missing_fields,
                "parse_error": r.parse_error or None,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON.")
    ap.add_argument("--layer", default=None,
                    choices=["model", "hardware", "profile", "preset"],
                    help="Limit to one layer.")
    args = ap.parse_args()

    results = audit_v2_required_fields()
    if args.layer:
        results = [r for r in results if r.layer == args.layer]

    if args.json:
        print(_render_json(results))
    else:
        print(_render_text(results))

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

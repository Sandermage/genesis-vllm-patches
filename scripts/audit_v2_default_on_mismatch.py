#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 default-on mismatch detector (informational).

PATCH_REGISTRY entries carry `default_on: bool`. When `True`, the
patch is enabled at runtime even without an explicit
`GENESIS_ENABLE_<X>_*: '1'` flag. If a V2 model EXPLICITLY disables
such a patch via `GENESIS_ENABLE_<X>_*: '0'`, that's an
operator-intent signal worth surfacing — it means «I want this OFF
even though the registry says ON by default».

This is **informational**, not a violation: operators may legitimately
disable default-on patches (e.g. to test without a particular patch
contribution, to bisect bench regressions, or to work around a
hardware-specific issue). The audit just makes those overrides
operator-visible at PR review time.

Exit codes:
  0 — every model surveyed (informational; never blocks)
  1 — reserved for future strict mode (currently unused)
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "model"


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class MismatchCheck:
    path: Path
    model_id: str
    overrides: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        # Informational gate — always passes unless there's a hard error.
        return not self.error


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _norm_value(v) -> str:
    return str(v).strip().strip("'").strip('"')


def _build_flag_to_default_on_pids() -> dict:
    """Map env_flag → list of pids that are default_on=True and use this flag."""
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except ImportError:
        return {}
    out: dict[str, list[str]] = defaultdict(list)
    for pid, meta in PATCH_REGISTRY.items():
        if not meta.get("default_on"):
            continue
        f = meta.get("env_flag")
        if f:
            out[f].append(pid)
    return dict(out)


def check_one_model(
    path: Path, flag_to_default_on_pids: dict,
) -> MismatchCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return MismatchCheck(
            path=path, model_id="?",
            error=f"YAML parse error: {e}",
        )
    model_id = data.get("id", path.stem)
    patches = data.get("patches") or {}
    overrides: list[dict] = []
    for k, v in patches.items():
        if _norm_value(v) != "0":
            continue
        pids = flag_to_default_on_pids.get(k)
        if not pids:
            continue
        for pid in pids:
            overrides.append({
                "env_flag": k,
                "patch_id": pid,
                "registry_default": "on",
                "model_value": "0",
            })
    return MismatchCheck(
        path=path,
        model_id=model_id,
        overrides=overrides,
    )


def audit_v2_default_on_mismatch(
    model_dir: Path = MODEL_DIR,
) -> list[MismatchCheck]:
    if not model_dir.is_dir():
        return []
    flag_idx = _build_flag_to_default_on_pids()
    return [check_one_model(p, flag_idx) for p in sorted(model_dir.glob("*.yaml"))]


def _render_text(results: list[MismatchCheck]) -> str:
    lines = [
        f"audit-v2-default-on-mismatch: {len(results)} model YAML(s)  [informational]",
        "─" * 70,
    ]
    total_overrides = 0
    for r in results:
        if r.error:
            lines.append(f"  ! {r.model_id}: {r.error}")
            continue
        if not r.overrides:
            lines.append(f"  ✓ {r.model_id} (no default-on overrides)")
            continue
        total_overrides += len(r.overrides)
        lines.append(
            f"  · {r.model_id}: {len(r.overrides)} operator override(s)"
        )
        for o in r.overrides[:5]:
            lines.append(
                f"      operator-disabled  {o['patch_id']:10s}  "
                f"(registry: default_on=True)"
            )
        if len(r.overrides) > 5:
            lines.append(f"      ... ({len(r.overrides) - 5} more)")
    lines.append("─" * 70)
    lines.append(
        f"  {total_overrides} explicit default-on override(s) across {len(results)} model(s)"
    )
    return "\n".join(lines)


def _render_json(results: list[MismatchCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "total_overrides": sum(len(r.overrides) for r in results),
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.path),
                "overrides": r.overrides,
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
    results = audit_v2_default_on_mismatch()
    print(_render_json(results) if args.json else _render_text(results))
    # Informational: pass unless internal error.
    return 0 if all(r.passed for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())

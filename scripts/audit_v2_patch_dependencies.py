#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 patch-dependency gate.

For every V2 `kind: model` YAML, walks `model.patches` matrix, resolves
each `GENESIS_ENABLE_<X>_*: '1'` env_flag to the set of patch_ids that
use it, and verifies:

  • `requires_patches`: every patch_id required by an enabled patch
    must itself be enabled (either via the same flag or another).
  • `conflicts_with`: no two enabled patches may name each other in
    their conflicts list.

Multi-pid env_flag handling: PATCH_REGISTRY currently has 2 env_flags
shared by multiple pids (`GENESIS_ENABLE_P67_*` → {P67, P67b}, and
`GENESIS_ENABLE_PN40_DFLASH_OMNIBUS` → {PN40, PN40-classifier}).
Setting one such flag enables BOTH pids — the audit uses a 1:N
flag→pids reverse map so `P67b.requires=['P67']` is satisfied when
the shared flag is set.

Exit codes:
  0 — every enabled patch's requires/conflicts satisfied
  1 — at least one violation
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


# Dependencies satisfied OUTSIDE the model-YAML `patches:` env map.
# The audit's "enabled" universe is built from `GENESIS_ENABLE_*: '1'`
# entries, but legacy auto-apply patches (synthetic `GENESIS_LEGACY_*`
# env_flags) are applied by the boot bundle / per-patch dispatch, never
# via a YAML flag — a requires_patches edge pointing at one of them is
# satisfied at boot even though no YAML flag exists. Keyed allowlist
# with operator rationale (same convention as ALLOWED_RETIRED_PATCHES
# in audit_v2_patch_lifecycle.py). Add entries ONLY with boot evidence.
IMPLICITLY_SATISFIED_DEPS: dict[str, str] = {
    "P27": (
        "Preflight residual triage 2026-06-11 §2 — PN71 declares "
        "requires_patches=['P27'] (its anchor contains P27-injected "
        "comments). P27 is a pre-dispatcher legacy patch "
        "(env_flag=GENESIS_LEGACY_P27, never set in model YAMLs) applied "
        "by the boot bundle: deduped PROD boot line 20 shows "
        "'applied: P27 Qwen3 BEFORE-THINK fallback'. The dependency is "
        "boot-satisfied, not YAML-satisfied."
    ),
}


@dataclass
class DepCheck:
    path: Path
    model_id: str
    enabled_pids: list[str] = field(default_factory=list)
    missing_requires: list[dict] = field(default_factory=list)
    conflicts_active: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return (
            not self.error
            and not self.missing_requires
            and not self.conflicts_active
        )


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _norm_value(v) -> str:
    return str(v).strip().strip("'").strip('"')


def _build_flag_index() -> tuple[dict, dict]:
    """Return (flag→list-of-pids, pid→meta) maps from PATCH_REGISTRY."""
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except ImportError:
        return {}, {}
    flag_to_pids: dict[str, list[str]] = defaultdict(list)
    pid_meta: dict[str, dict] = {}
    for pid, meta in PATCH_REGISTRY.items():
        pid_meta[pid] = meta
        f = meta.get("env_flag")
        if f:
            flag_to_pids[f].append(pid)
    return dict(flag_to_pids), pid_meta


def check_one_model(
    path: Path, *,
    flag_to_pids: dict,
    pid_meta: dict,
) -> DepCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return DepCheck(
            path=path, model_id="?",
            error=f"YAML parse error: {e}",
        )
    model_id = data.get("id", path.stem)
    patches = data.get("patches") or {}

    # Collect enabled patch ids — every pid using a `'1'`-set env_flag.
    enabled_pids: set[str] = set()
    for k, v in patches.items():
        if _norm_value(v) != "1":
            continue
        for pid in flag_to_pids.get(k, []):
            enabled_pids.add(pid)

    # Per-pid requires + conflicts checks.
    missing_requires: list[dict] = []
    conflicts_active: list[dict] = []
    for pid in sorted(enabled_pids):
        meta = pid_meta.get(pid, {})
        for req in meta.get("requires_patches") or []:
            if req in IMPLICITLY_SATISFIED_DEPS:
                # Boot-satisfied dependency (legacy auto-apply) — see
                # the allowlist rationale at the top of this script.
                continue
            if req not in enabled_pids:
                missing_requires.append({
                    "patch_id": pid,
                    "missing_dependency": req,
                })
        for conf in meta.get("conflicts_with") or []:
            if conf in enabled_pids and conf != pid:
                # Avoid double-reporting: emit only when pid < conf.
                if pid < conf:
                    conflicts_active.append({
                        "patch_a": pid,
                        "patch_b": conf,
                    })

    return DepCheck(
        path=path,
        model_id=model_id,
        enabled_pids=sorted(enabled_pids),
        missing_requires=missing_requires,
        conflicts_active=conflicts_active,
    )


def audit_v2_patch_dependencies(
    model_dir: Path = MODEL_DIR,
) -> list[DepCheck]:
    if not model_dir.is_dir():
        return []
    flag_to_pids, pid_meta = _build_flag_index()
    return [
        check_one_model(p, flag_to_pids=flag_to_pids, pid_meta=pid_meta)
        for p in sorted(model_dir.glob("*.yaml"))
    ]


def _render_text(results: list[DepCheck]) -> str:
    lines = [
        f"audit-v2-patch-dependencies: {len(results)} model YAML(s)",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.model_id}: {r.error}")
            continue
        lines.append(
            f"  {sym} {r.model_id:36s}  "
            f"enabled={len(r.enabled_pids)}  "
            f"req_viol={len(r.missing_requires)}  "
            f"conf_viol={len(r.conflicts_active)}"
        )
        for mr in r.missing_requires:
            lines.append(
                f"      ⚠ {mr['patch_id']} requires {mr['missing_dependency']} — not enabled"
            )
        for c in r.conflicts_active:
            lines.append(
                f"      ⚠ {c['patch_a']} ⨯ {c['patch_b']} — both enabled (conflict)"
            )
    passed = sum(1 for r in results if r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} models clean")
    return "\n".join(lines)


def _render_json(results: list[DepCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.path),
                "enabled_count": len(r.enabled_pids),
                "missing_requires": r.missing_requires,
                "conflicts_active": r.conflicts_active,
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
    results = audit_v2_patch_dependencies()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

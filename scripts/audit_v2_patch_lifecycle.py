#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 patch-lifecycle coverage gate.

For each V2 `kind: model` YAML, every `GENESIS_ENABLE_<X>_*: '1'` flag
maps via PATCH_REGISTRY to a `lifecycle` field
(stable / experimental / legacy / retired / research / coordinator).
Enabling a `lifecycle: retired` patch in production config is a
hygiene issue — the code path runs but is no longer actively maintained.

This gate surfaces enabled-retired patches per V2 model. Default mode
is **informational** (warn, don't block) because:

  • Retired patches may still ship working code (retirement is a
    maintenance signal, not a "removed" signal).
  • Operator may legitimately keep one enabled until a known
    replacement lands.

To make it gating, change the GATES entry severity, OR add specific
patch_ids to the `ALLOWED_RETIRED_PATCHES` allowlist with a comment
explaining why.

Survey at E30 time found three enabled retired patches in committed
V2 models: PN19, PN52, P94. These are added to the allowlist below
with operator context. Future drift (a NEW retired patch enabled
without allowlist entry) flags.

Exit codes:
  0 — no out-of-allowlist retired patches enabled
  1 — at least one out-of-allowlist retired/disallowed patch enabled
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "model"

# Ensure repo root on path so `vllm.sndr_core.*` resolves when run as
# `python3 scripts/audit_v2_patch_lifecycle.py` (Python only adds the
# script's directory to sys.path by default, not the parent).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Lifecycles we treat as "hygiene-violation if enabled". Default:
# `retired` only — other lifecycles (experimental, legacy, stable,
# research, coordinator) are all legitimate in production V2 models.
DISALLOWED_LIFECYCLES: frozenset[str] = frozenset({"retired"})


# Operator-known retired patches that are still enabled by design at
# E30 freeze. Each entry records why. Add a comment when expanding.
ALLOWED_RETIRED_PATCHES: dict[str, str] = {
    "PN19": "carry-over from W-A; replacement is part of PN-series consolidation work",
    "PN52": "still actively consumed by 27B INT4 / 35B FP8 prod path",
    "P94":  "enabled in 27B INT4 TQ + 35B FP8 prod — operator review pending",
}


@dataclass
class LifecycleCheck:
    model_path: Path
    model_id: str
    enabled_patches: int = 0
    by_lifecycle: dict = field(default_factory=dict)
    violations: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and not self.violations


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _build_flag_to_meta() -> dict:
    """Map env_flag → (patch_id, lifecycle, tier, family)."""
    try:
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    except ImportError:
        return {}
    out = {}
    for pid, meta in PATCH_REGISTRY.items():
        flag = meta.get("env_flag")
        if flag:
            out[flag] = {
                "patch_id": pid,
                "lifecycle": meta.get("lifecycle", "?"),
                "tier": meta.get("tier", "?"),
                "family": meta.get("family", "?"),
            }
    return out


def _norm_value(v) -> str:
    return str(v).strip().strip("'").strip('"')


def check_one_model(path: Path, flag_to_meta: dict) -> LifecycleCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return LifecycleCheck(
            model_path=path, model_id="?",
            error=f"YAML parse error: {e}",
        )
    model_id = data.get("id", path.stem)
    patches = data.get("patches") or {}

    by_lifecycle: dict[str, int] = {}
    violations: list[dict] = []
    enabled_count = 0

    for k, v in patches.items():
        if k not in flag_to_meta:
            continue
        if _norm_value(v) != "1":
            continue
        meta = flag_to_meta[k]
        lc = meta["lifecycle"]
        by_lifecycle[lc] = by_lifecycle.get(lc, 0) + 1
        enabled_count += 1
        if lc in DISALLOWED_LIFECYCLES:
            pid = meta["patch_id"]
            if pid in ALLOWED_RETIRED_PATCHES:
                continue   # operator-allowlisted
            violations.append({
                "env_flag": k,
                "patch_id": pid,
                "lifecycle": lc,
            })

    return LifecycleCheck(
        model_path=path,
        model_id=model_id,
        enabled_patches=enabled_count,
        by_lifecycle=by_lifecycle,
        violations=violations,
    )


def audit_v2_patch_lifecycle(
    model_dir: Path = MODEL_DIR,
) -> list[LifecycleCheck]:
    if not model_dir.is_dir():
        return []
    flag_to_meta = _build_flag_to_meta()
    return [
        check_one_model(p, flag_to_meta)
        for p in sorted(model_dir.glob("*.yaml"))
    ]


def _render_text(results: list[LifecycleCheck]) -> str:
    lines = [
        f"audit-v2-patch-lifecycle: {len(results)} model YAML(s)",
        f"  disallowed lifecycles: {sorted(DISALLOWED_LIFECYCLES)}",
        f"  allowlist (retired patches OK to enable): "
        f"{sorted(ALLOWED_RETIRED_PATCHES.keys())}",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.model_id}: {r.error}")
            continue
        bl = ", ".join(f"{k}={v}" for k, v in sorted(r.by_lifecycle.items()))
        lines.append(f"  {sym} {r.model_id:36s}  enabled={r.enabled_patches}  {bl}")
        for v in r.violations:
            lines.append(
                f"      ⚠ {v['patch_id']} ({v['lifecycle']}) — "
                f"not in allowlist"
            )
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} models clean")
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: either disable the retired patch in the model, "
            "or add the patch_id to ALLOWED_RETIRED_PATCHES with operator rationale."
        )
    return "\n".join(lines)


def _render_json(results: list[LifecycleCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "disallowed_lifecycles": sorted(DISALLOWED_LIFECYCLES),
        "allowed_retired_patches": ALLOWED_RETIRED_PATCHES,
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.model_path),
                "enabled_patches": r.enabled_patches,
                "by_lifecycle": r.by_lifecycle,
                "violations": r.violations,
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
    results = audit_v2_patch_lifecycle()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

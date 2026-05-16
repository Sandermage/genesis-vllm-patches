#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§Phase D — `make audit-patch-plan-resolves` — resolver health gate.

Every committed V2 preset MUST resolve cleanly under all three
policies (compat / safe / minimal). This gate iterates the full
``model_configs/builtin/presets/`` directory and runs the
``resolve_patch_plan(cfg, policy=…)`` triple for each. Any preset
that raises, or produces a non-zero ``warnings`` tuple under safe
(the conservative default we'll likely promote), is flagged.

The gate has two thresholds:

  GATING       — every preset must compose + resolve without raising.
                 Raised exception = build break.
  INFORMATIONAL — every preset's safe-policy warnings tuple is empty.
                 Non-empty = informational warning, not gating, because
                 conflict / candidate_when warnings can be legitimate
                 (operator opted into a config that the resolver flags
                 for visibility but doesn't block).

Exit codes:
  0 — every preset resolved cleanly under every policy
  1 — at least one preset failed to resolve (GATING violation)
  2 — internal error (registry / file loader broken)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
PRESETS_DIR = (
    REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "presets"
)
POLICIES: tuple[str, ...] = ("compat", "safe", "minimal")


@dataclass
class ResolveCheck:
    preset: str
    path: Path
    by_policy: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str = ""

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        for p, payload in self.by_policy.items():
            if "error" in payload:
                return False
        return True

    @property
    def total_warnings(self) -> int:
        return sum(
            len(payload.get("warnings", []))
            for payload in self.by_policy.values()
        )


def _check_preset(preset_name: str, path: Path) -> ResolveCheck:
    """Compose + resolve a single preset under every policy."""
    chk = ResolveCheck(preset=preset_name, path=path)
    # Ensure the repo root is on sys.path so `import vllm.sndr_core …`
    # works from `scripts/` even when run outside the package's
    # installed environment (CI, fresh clone).
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from vllm.sndr_core.model_configs.registry_v2 import load_alias
        from vllm.sndr_core.model_configs.patch_plan import (
            resolve_patch_plan,
        )
    except Exception as e:
        chk.error = f"resolver import failed: {type(e).__name__}: {e}"
        return chk

    try:
        cfg = load_alias(preset_name)
    except Exception as e:
        chk.error = f"compose failed: {type(e).__name__}: {e}"
        return chk

    for policy in POLICIES:
        slot: dict[str, Any] = {}
        try:
            plan = resolve_patch_plan(cfg, policy=policy)
            slot["included_count"] = len(plan.included)
            slot["excluded_count"] = len(plan.excluded)
            slot["passthrough_count"] = len(plan.passthrough)
            slot["warnings"] = list(plan.warnings)
        except Exception as e:
            slot["error"] = f"{type(e).__name__}: {e}"
        chk.by_policy[policy] = slot
    return chk


def audit_patch_plan_resolves() -> list[ResolveCheck]:
    if not PRESETS_DIR.is_dir():
        return []
    presets = sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))
    return [_check_preset(name, PRESETS_DIR / f"{name}.yaml") for name in presets]


# ─── Renderers ───────────────────────────────────────────────────────────


def _render_text(results: list[ResolveCheck]) -> tuple[str, bool]:
    lines = [
        f"audit-patch-plan-resolves: {len(results)} V2 preset(s) "
        f"× {len(POLICIES)} policies = "
        f"{len(results) * len(POLICIES)} resolutions",
        "─" * 70,
    ]
    total_warnings = 0
    all_passed = True
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.preset}: {r.error}")
            all_passed = False
            continue
        per_policy = " ".join(
            (
                f"{p}=ERR" if "error" in r.by_policy[p]
                else f"{p}=in{r.by_policy[p]['included_count']}/"
                     f"ex{r.by_policy[p]['excluded_count']}/"
                     f"warn{len(r.by_policy[p].get('warnings', []))}"
            )
            for p in POLICIES
        )
        lines.append(f"  {sym} {r.preset:<40} {per_policy}")
        for p in POLICIES:
            slot = r.by_policy[p]
            if "error" in slot:
                lines.append(f"      {p} error: {slot['error']}")
                all_passed = False
                continue
            for w in slot.get("warnings", []):
                total_warnings += 1
                lines.append(f"      {p} ⚠ {w}")
    lines.append("─" * 70)
    lines.append(
        f"  resolutions clean: "
        f"{sum(1 for r in results if r.passed)} / {len(results)} preset(s)"
    )
    if total_warnings:
        lines.append(
            f"  resolver warnings (informational): {total_warnings} "
            f"across {sum(1 for r in results if r.total_warnings > 0)} preset(s)"
        )
    return "\n".join(lines), all_passed


def _render_json(results: list[ResolveCheck]) -> tuple[str, bool]:
    all_passed = all(r.passed for r in results)
    payload = {
        "presets_scanned": len(results),
        "policies": list(POLICIES),
        "all_passed": all_passed,
        "results": [
            {
                "preset": r.preset,
                "path": str(r.path.relative_to(REPO_ROOT)),
                "passed": r.passed,
                "error": r.error or None,
                "by_policy": r.by_policy,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True), all_passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON output.")
    ap.add_argument(
        "--strict-warnings", action="store_true",
        help=(
            "Treat any non-empty warnings tuple as a gating failure. "
            "Default: warnings are informational only; the gate only "
            "fails on a real exception during compose / resolve."
        ),
    )
    args = ap.parse_args()

    try:
        results = audit_patch_plan_resolves()
    except Exception as e:
        sys.stderr.write(f"audit-patch-plan-resolves: {e}\n")
        return 2

    if args.json:
        out, ok = _render_json(results)
    else:
        out, ok = _render_text(results)
    print(out)
    if args.strict_warnings:
        any_warns = any(r.total_warnings > 0 for r in results)
        if any_warns and ok:
            sys.stderr.write(
                "audit-patch-plan-resolves: --strict-warnings set and "
                "at least one preset produced resolver warnings\n"
            )
            return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

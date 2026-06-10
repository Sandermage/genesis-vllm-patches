#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 P2.3 — lifecycle ratchet for runtime-hook patches.

Genesis's STABLE lifecycle was originally designed for text-patch
contributions: each promotion required `register_for_manifest()` +
pristine `anchor_manifest.json` coverage of every wrapped callable.
That mechanism blocks runtime-hook patches (e.g. PN35, PN96, PN33)
from ever being promoted to STABLE because their apply path doesn't
go through TextPatcher — they install runtime monkey-patches that
don't have line-level anchors.

Per Consolidated Roadmap §8.3 (REMAINING_WORK_PLAN P2.3, 1-2d), the
ratchet expands to BOTH patch classes while keeping the production-ready
promise.

Schema additions (registry.py entries):

  • `stable_kind: Literal["text-patch", "runtime-hook"]` — REQUIRED for
    every `lifecycle: stable` entry. Decides which ratchet rules apply.
  • `production_validated_pins: list[tuple[genesis_pin, vllm_pin]]` —
    REQUIRED when `stable_kind: runtime-hook`. Minimum 2 entries to
    prove the runtime-hook fires correctly across at least two
    independently-validated (genesis_pin, vllm_pin) combinations.

This gate validates:

  • Every `lifecycle: stable` patch declares `stable_kind`.
  • `stable_kind` ∈ {"text-patch", "runtime-hook"}.
  • If `runtime-hook`: `production_validated_pins` is a list with ≥2
    tuples; each tuple is (genesis_pin_str, vllm_pin_str) with both
    non-empty.

Exit codes:
  0 — every stable patch's ratchet metadata is valid
  1 — at least one stable patch is missing/malformed metadata
  2 — internal error (PATCH_REGISTRY not importable)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ALLOWED_STABLE_KINDS: frozenset[str] = frozenset({"text-patch", "runtime-hook"})
MIN_RUNTIME_HOOK_PINS: int = 2


@dataclass
class StablePatchCheck:
    patch_id: str
    lifecycle: str
    stable_kind: object = None
    production_validated_pins: object = None
    violations: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.violations


def _load_registry() -> dict:
    try:
        from sndr.dispatcher.registry import PATCH_REGISTRY
    except ImportError as e:
        raise RuntimeError(
            f"PATCH_REGISTRY not importable from repo_root={REPO_ROOT}: {e}"
        ) from e
    return PATCH_REGISTRY


def check_one_patch(pid: str, meta: dict) -> StablePatchCheck:
    r = StablePatchCheck(
        patch_id=pid,
        lifecycle=meta.get("lifecycle", "?"),
        stable_kind=meta.get("stable_kind"),
        production_validated_pins=meta.get("production_validated_pins"),
    )

    if r.stable_kind is None:
        r.violations.append(
            "missing required field `stable_kind` (must be "
            f"one of {sorted(ALLOWED_STABLE_KINDS)})"
        )
        return r

    if r.stable_kind not in ALLOWED_STABLE_KINDS:
        r.violations.append(
            f"stable_kind={r.stable_kind!r} not in "
            f"{sorted(ALLOWED_STABLE_KINDS)}"
        )
        return r

    if r.stable_kind == "runtime-hook":
        pins = r.production_validated_pins
        if pins is None:
            r.violations.append(
                "stable_kind='runtime-hook' requires "
                "`production_validated_pins` (list of "
                "(genesis_pin, vllm_pin) tuples, min 2)"
            )
        elif not isinstance(pins, list):
            r.violations.append(
                f"production_validated_pins={type(pins).__name__}; expected list"
            )
        elif len(pins) < MIN_RUNTIME_HOOK_PINS:
            r.violations.append(
                f"production_validated_pins has {len(pins)} entries; "
                f"runtime-hook ratchet requires ≥ {MIN_RUNTIME_HOOK_PINS}"
            )
        else:
            for i, entry in enumerate(pins):
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    r.violations.append(
                        f"production_validated_pins[{i}]={entry!r} "
                        "must be (genesis_pin, vllm_pin) tuple"
                    )
                    continue
                g_pin, v_pin = entry
                if not isinstance(g_pin, str) or not g_pin.strip():
                    r.violations.append(
                        f"production_validated_pins[{i}] genesis_pin "
                        f"empty/non-string: {g_pin!r}"
                    )
                if not isinstance(v_pin, str) or not v_pin.strip():
                    r.violations.append(
                        f"production_validated_pins[{i}] vllm_pin "
                        f"empty/non-string: {v_pin!r}"
                    )

    return r


def audit_runtime_hook_ratchet(
    registry: dict | None = None,
) -> list[StablePatchCheck]:
    if registry is None:
        registry = _load_registry()
    return [
        check_one_patch(pid, meta)
        for pid, meta in registry.items()
        if meta.get("lifecycle") == "stable"
    ]


def _render_text(results: list[StablePatchCheck]) -> str:
    lines = []
    lines.append(
        f"audit-runtime-hook-ratchet: {len(results)} stable patch(es) checked"
    )
    lines.append("─" * 70)
    for r in results:
        sym = "✓" if r.passed else "✗"
        kind_str = r.stable_kind if r.stable_kind is not None else "<unset>"
        lines.append(f"  {sym} {r.patch_id:8s}  stable_kind={kind_str}")
        if r.stable_kind == "runtime-hook":
            pins = r.production_validated_pins or []
            lines.append(
                f"      production_validated_pins: {len(pins) if isinstance(pins, list) else 'N/A'} entries"
            )
        for v in r.violations:
            lines.append(f"      ⚠ {v}")
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} stable patches pass ratchet")
    if failed:
        lines.append("")
        lines.append(
            "  ✗ Fix: add `stable_kind` (and `production_validated_pins` "
            "for runtime-hook) to PATCH_REGISTRY entry. "
            "See audit_runtime_hook_ratchet.py docstring."
        )
    return "\n".join(lines)


def _render_json(results: list[StablePatchCheck]) -> str:
    return json.dumps({
        "total_stable": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "allowed_stable_kinds": sorted(ALLOWED_STABLE_KINDS),
        "min_runtime_hook_pins": MIN_RUNTIME_HOOK_PINS,
        "patches": [
            {
                "patch_id": r.patch_id,
                "lifecycle": r.lifecycle,
                "stable_kind": r.stable_kind,
                "production_validated_pins": r.production_validated_pins,
                "passed": r.passed,
                "violations": r.violations,
            }
            for r in results
        ],
    }, indent=2, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON.")
    args = ap.parse_args()

    try:
        results = audit_runtime_hook_ratchet()
    except RuntimeError as e:
        sys.stderr.write(f"audit-runtime-hook-ratchet: {e}\n")
        return 2

    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

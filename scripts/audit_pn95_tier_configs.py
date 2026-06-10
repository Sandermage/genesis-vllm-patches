#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PN95 tier_configs audit gate — Phase 10.5 D.2 (2026-06-01).

Validates every YAML file under
``sndr/cache/pn95/tier_configs/`` against the
``_PN95TierConfigAdapter`` shape that ``make_tier_manager`` consumes.

Background: the PN95 architectural unblock (V1 sunset cascade
2026-06-01) moved tier_specs out of the V1 monolithic ModelConfig
into PN95-internal YAMLs. The loader
(``sndr.cache.pn95.tier_config_loader.load_by_key``) already
raises ``ValueError`` on schema mismatch at load time, but until this
gate landed there was no CI-friendly way to verify the whole catalog
loads cleanly without dragging a torch/vllm import chain. This gate
exercises ``load_by_key`` against every file at audit time so PRs
that drift the schema break the gate, not first-touch on an operator
container restart.

Invariants asserted:

  1. Every ``tier_configs/*.yaml`` file loads via ``load_by_key``
     without raising (the loader itself enforces device + capacity_gib
     required fields, tier list shape, top-level mapping shape).
  2. Tier ``device`` values use the known set {``gpu``, ``cpu``,
     ``disk``} — PN95's TierManager only knows these three. Free-form
     device names would silently behave as "cpu" downstream and waste
     the operator's debugging cycle.
  3. Total per-rig GPU capacity_gib does not exceed the rig name's
     advertised total (e.g. ``a5000-2x-*`` ≤ 2 × 24 GiB = 48 GiB).
     This is a sanity check, not a runtime gate — TierManager itself
     accepts any number and trusts the operator.

Exit code:

  0 — all configs load cleanly + invariants hold.
  1 — at least one config violates a rule (CI gate fires).
  2 — internal error (filesystem / import issue).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TIER_DIR = REPO_ROOT / "sndr" / "cache" / "pn95" / "tier_configs"

KNOWN_DEVICES = {"gpu", "cpu", "disk"}

# Rig-name → advertised GPU capacity GiB (per TP worker × tp_size).
# Matches the rig-name convention `<gpu>-<tp>x-<purpose>`.
_RIG_GPU_VRAM = {
    "a5000-1x": 24.0 * 1,
    "a5000-2x": 24.0 * 2,
    "a5000-4x": 24.0 * 4,
    "rtx3090-1x": 24.0 * 1,
    "rtx3090-2x": 24.0 * 2,
}


@dataclass
class _Finding:
    yaml_id: str
    severity: str  # "error" | "warning"
    rule_id: str
    message: str


def _rig_prefix(yaml_id: str) -> str | None:
    """Return the rig prefix (e.g. 'a5000-2x') from a tier_config id."""
    m = re.match(r"^(rtx3090|a5000)-(\d+)x", yaml_id)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}x"


def audit() -> list[_Finding]:
    findings: list[_Finding] = []

    if not TIER_DIR.is_dir():
        findings.append(_Finding(
            "(directory)", "error", "PN95-T-000",
            f"tier_configs dir not found: {TIER_DIR}",
        ))
        return findings

    # Import the loader lazily so a torch-free CI gate still works.
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from sndr.cache.pn95.tier_config_loader import (
            load_by_key, known_keys,
        )
    finally:
        sys.path.pop(0)

    keys = known_keys()
    for key in sorted(keys):
        # PN95-T-001: file loads cleanly via load_by_key.
        try:
            adapter = load_by_key(key)
        except ValueError as e:
            findings.append(_Finding(
                key, "error", "PN95-T-001",
                f"loader rejected schema: {e}",
            ))
            continue
        if adapter is None:  # pragma: no cover — known_keys lists only present files
            findings.append(_Finding(
                key, "error", "PN95-T-001",
                "known_keys() returned key but load_by_key returned None",
            ))
            continue

        # PN95-T-002: every tier.device is in KNOWN_DEVICES.
        unknown_devices = sorted({
            t.device for t in adapter.cache_config.tiers
            if t.device not in KNOWN_DEVICES
        })
        if unknown_devices:
            findings.append(_Finding(
                key, "error", "PN95-T-002",
                f"unknown tier.device value(s) {unknown_devices!r}; "
                f"PN95 TierManager only routes {sorted(KNOWN_DEVICES)!r}",
            ))

        # PN95-T-003: rig-name GPU capacity sanity check.
        rig = _rig_prefix(key)
        if rig is not None and rig in _RIG_GPU_VRAM:
            advertised = _RIG_GPU_VRAM[rig]
            gpu_total = sum(
                t.capacity_gib for t in adapter.cache_config.tiers
                if t.device == "gpu"
            )
            if gpu_total > advertised:
                findings.append(_Finding(
                    key, "error", "PN95-T-003",
                    f"declared GPU tier capacity {gpu_total} GiB exceeds "
                    f"rig advertised total {advertised} GiB ({rig})",
                ))
            elif gpu_total > advertised * 0.97:  # warn near hard limit
                findings.append(_Finding(
                    key, "warning", "PN95-T-003",
                    f"declared GPU tier capacity {gpu_total} GiB is "
                    f">{int(0.97 * 100)}% of rig total {advertised} GiB "
                    f"({rig}) — leaves little headroom for runtime/CG capture",
                ))

    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="emit JSON payload instead of human-readable summary")
    args = ap.parse_args()

    findings = audit()
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]

    if args.json:
        print(json.dumps({
            "errors": [f.__dict__ for f in errors],
            "warnings": [f.__dict__ for f in warnings],
            "counts": {
                "error": len(errors),
                "warning": len(warnings),
            },
            "passed": not errors,
        }, indent=2, sort_keys=True))
    else:
        print(f"audit-pn95-tier-configs: scanned {len(list(TIER_DIR.glob('*.yaml')))} file(s)")
        print("─" * 70)
        if errors:
            print(f"  ✗ ERROR ({len(errors)}):")
            for f in errors:
                print(f"      [{f.rule_id}] {f.yaml_id}: {f.message}")
        if warnings:
            print(f"  ⚠ WARNING ({len(warnings)}):")
            for f in warnings:
                print(f"      [{f.rule_id}] {f.yaml_id}: {f.message}")
        if not findings:
            print("  ✓ all PN95 tier_configs load cleanly + invariants hold")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())

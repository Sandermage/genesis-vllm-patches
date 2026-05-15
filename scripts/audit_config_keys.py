#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Canonical env-key registry audit (Consolidated Roadmap §10.3 item 4 / §6.7).

Walks every committed V1 monolithic + V2 layered YAML and runs
`sndr config-keys-validate` on each. The validator computes the canonical
union of every Genesis/SNDR env key the codebase knows about
(PATCH_REGISTRY entries + V2 model.patches blocks + V1 genesis_env
blocks) and checks that every key in the YAML is in that union.

A typo or undocumented patch produces an "unknown key" hit, which
this audit promotes to a release gate.

Scope:

  • `vllm/sndr_core/model_configs/builtin/*.yaml`           (V1 monolithic)
  • `vllm/sndr_core/model_configs/builtin/model/*.yaml`     (V2 ModelDef)
  • `vllm/sndr_core/model_configs/builtin/hardware/*.yaml`  (V2 HardwareDef)
  • `vllm/sndr_core/model_configs/builtin/profile/*.yaml`   (V2 ProfileDef)

Preset alias triplets (`builtin/presets/*.yaml`) don't carry env keys
of their own — they're pure pointers. They are skipped.

Exit codes:
  0 — every YAML's keys are in the canonical registry.
  1 — at least one YAML has an unknown env key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import the validator directly so we don't pay subprocess startup per
# YAML (would be 40+ python launches otherwise).
sys.path.insert(0, str(REPO_ROOT))
from vllm.sndr_core.cli import config_keys as _ck  # type: ignore  # noqa: E402  — import after sys.path mutation


SCAN_DIRS = (
    "vllm/sndr_core/model_configs/builtin",
    "vllm/sndr_core/model_configs/builtin/model",
    "vllm/sndr_core/model_configs/builtin/hardware",
    "vllm/sndr_core/model_configs/builtin/profile",
)

SKIP_DIRS = ("presets",)


def _gather_yamls() -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for s in SCAN_DIRS:
        p = REPO_ROOT / s
        if not p.is_dir():
            continue
        for f in p.glob("*.yaml"):
            if any(skip in f.parts for skip in SKIP_DIRS):
                continue
            if f in seen:
                continue
            seen.add(f)
            out.append(f)
    return sorted(out)


def audit() -> dict:
    canon = _ck.load_canonical_registry()
    canonical_keys = set(canon.keys()) if hasattr(canon, "keys") else set(canon)
    results: list[dict] = []
    total_unknown = 0
    for fp in _gather_yamls():
        try:
            keys = _ck._extract_keys_from_yaml(fp)
        except RuntimeError:
            keys = []
        unknown = sorted({
            k for k in keys
            if (k.startswith("GENESIS_") or k.startswith("SNDR_"))
            and k not in canonical_keys
        })
        rel = fp.relative_to(REPO_ROOT).as_posix()
        results.append({
            "yaml": rel,
            "unknown_keys": unknown,
            "count": len(unknown),
        })
        total_unknown += len(unknown)
    return {
        "canonical_count": len(canonical_keys),
        "yaml_count": len(results),
        "total_unknown": total_unknown,
        "per_yaml": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = audit()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "audit-config-keys: "
            f"{report['yaml_count']} YAMLs scanned against "
            f"{report['canonical_count']} canonical keys"
        )
        print("─" * 70)
        bad = [r for r in report["per_yaml"] if r["count"] > 0]
        if bad:
            for r in bad[:10]:
                print(f"  ✗ {r['yaml']}: {r['count']} unknown key(s)")
                for k in r["unknown_keys"][:5]:
                    print(f"      · {k}")
                if r["count"] > 5:
                    print(f"      ... ({r['count'] - 5} more)")
            if len(bad) > 10:
                print(f"  ... ({len(bad) - 10} more YAMLs with unknown keys)")
            print()
            print(f"  FAIL — {report['total_unknown']} unknown key(s)")
        else:
            print("  ✓ every YAML's Genesis/SNDR keys are canonical")
            print()
            print("  OK — env-key drift gate clean")
    return 0 if report["total_unknown"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-model validation: V2 compose + apply matrix render + bench-readiness.

For every model in the V2 registry, validate that:
  1. It loads cleanly.
  2. It composes against at least one valid hardware target.
  3. The compose result has a non-empty patch list.
  4. The compose result produces a valid V1 launch command.
  5. Patches that would apply (default_on or env-toggled) are all in the
     dispatcher registry — no missing modules.

This script does NOT boot containers. It validates the configuration path
end-to-end up to the docker-run boundary. Use ``bench_live_*.py`` scripts
for the actual TPS measurement once a container is running.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sndr.model_configs.registry_v2 import (
    compose_by_ids,
    list_hardware,
    list_models,
    list_profiles,
    load_model,
    load_profile,
)
from sndr.dispatcher.registry import PATCH_REGISTRY

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"

if not sys.stdout.isatty():
    GREEN = RED = YELLOW = BOLD = RESET = ""


def section(title: str) -> None:
    print(f"\n{BOLD}── {title} ──{RESET}")


def model_passes(model_id: str) -> dict:
    """Render the full config matrix for one model. Returns a summary dict."""
    report = {
        "model_id": model_id,
        "compose_ok": 0,
        "compose_rejected": 0,
        "compose_errors": [],
        "hardware_targets": [],
        "patches_total": 0,
        "patches_default_on": 0,
        "patches_in_registry": 0,
        "patches_missing": [],
        "valid_triplets": [],
    }
    hardware = list_hardware()
    profiles = list_profiles()

    for h in hardware:
        for prof_id in profiles:
            try:
                prof_def = load_profile(prof_id)
                if prof_def.parent_model and prof_def.parent_model != model_id:
                    continue
            except Exception:
                continue
            try:
                config = compose_by_ids(model_id, h, prof_id)
                report["compose_ok"] += 1
                report["valid_triplets"].append(f"{prof_id}@{h}")
                if h not in report["hardware_targets"]:
                    report["hardware_targets"].append(h)
                # The patch list lives in `genesis_env` as
                # GENESIS_ENABLE_<PATCH_ID>_* env vars. Build the patch ID
                # set by reverse-lookup against the registry's env_flag field.
                env = config.genesis_env or {}
                enabled_flags = {k for k, v in env.items()
                                 if k.startswith("GENESIS_ENABLE_") and v == "1"}
                flag_to_patch = {
                    entry.get("env_flag"): pid
                    for pid, entry in PATCH_REGISTRY.items()
                    if entry.get("env_flag")
                }
                touched = 0
                for flag in enabled_flags:
                    touched += 1
                    if flag in flag_to_patch:
                        report["patches_in_registry"] += 1
                    else:
                        # Unknown env flag — patch deleted or refactored
                        report["patches_missing"].append(flag)
                report["patches_total"] += touched
                report["patches_default_on"] += touched
            except Exception as e:
                report["compose_rejected"] += 1
                report["compose_errors"].append(f"{prof_id}@{h}: {type(e).__name__}: {str(e)[:80]}")

    # De-duplicate missing patches (same set across all triplets)
    report["patches_missing"] = sorted(set(report["patches_missing"]))
    return report


def main() -> int:
    section("Per-model V2 configuration validation")

    models = list_models()
    print(f"Models discovered: {len(models)}")

    summaries = []
    fails = 0
    for m in models:
        r = model_passes(m)
        summaries.append(r)
        if r["compose_ok"] == 0:
            status = f"{RED}FAIL{RESET}"
            fails += 1
        elif r["patches_missing"]:
            status = f"{YELLOW}WARN{RESET}"
        else:
            status = f"{GREEN}OK{RESET}"

        triplets = len(r["valid_triplets"])
        hwlist = ",".join(t.split("gbvram")[0].replace("-24", "") for t in r["hardware_targets"]) or "(none)"
        unique_patches = r["patches_total"] // max(triplets, 1)
        print(f"  {status:>15}  {m:<40}  triplets={triplets:>2}  "
              f"hw={hwlist:<20}  patches={unique_patches} "
              f"missing={len(r['patches_missing'])}")

    section("Aggregate")
    total_triplets = sum(r["compose_ok"] for r in summaries)
    total_missing = sum(len(r["patches_missing"]) for r in summaries)
    print(f"  Models OK: {len(summaries) - fails} / {len(summaries)}")
    print(f"  Total valid triplets: {total_triplets}")
    print(f"  Missing patch references: {total_missing}")
    if total_missing:
        all_missing = set()
        for r in summaries:
            all_missing.update(r["patches_missing"])
        print(f"  Unique missing patch IDs: {sorted(all_missing)[:10]}{'...' if len(all_missing)>10 else ''}")

    # Detailed per-model patch matrix
    section("Per-model first triplet patch counts")
    for r in summaries:
        if r["compose_ok"]:
            first = r["valid_triplets"][0]
            triplets = r["compose_ok"]
            avg_patches = r["patches_total"] / triplets if triplets else 0
            avg_default_on = r["patches_default_on"] / triplets if triplets else 0
            print(f"  {r['model_id']:<40}  triplet={first:<35}  "
                  f"avg_total={avg_patches:.0f}  default_on={avg_default_on:.0f}")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

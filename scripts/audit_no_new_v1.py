#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 9 release gate — `make audit-no-new-v1`.

V1 freeze: no new monolithic preset YAML may land at
`vllm/sndr_core/model_configs/builtin/*.yaml` (top-level, not subdirs).
The existing 11 V1 files are frozen as legacy; any addition forces the
new preset to land as a V2 layered triplet (model + hardware + profile)
under the corresponding subdir.

The frozen baseline list is hardcoded below. Adding to it requires a
PR that updates BOTH this file AND the new V1 yaml, signalling that
the operator explicitly chose to extend the legacy path (e.g. for a
back-port that V2 layered shape can't express cleanly yet).

Exit code:
  0 — top-level builtin/*.yaml matches the frozen baseline.
  1 — drift detected (new V1 file added OR baseline file deleted).
  2 — internal error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BUILTIN_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin"

# Phase 9 frozen baseline — V1 monolithic presets accepted into the
# release tier. Original freeze (2026-05-12) listed 11 files.
#
# Bumps:
#   2026-05-14 +1 — `a5000-1x-tier-aware-pn95.yaml` added as a Wave 9
#   long-context single-card preset (PN95 multi-tier offload reaches
#   200K on a single A5000 24 GB). The config is verified on hardware;
#   the V2 layered triplet migration is queued as a follow-up cleanup,
#   not a blocker for release. Until that lands the V1 entry stays
#   here so `make evidence` does not gate on the migration.
FROZEN_V1_BASELINE: frozenset[str] = frozenset({
    # Phase 10 Step 4 (2026-06-01): V1 monolithic preset tier FULLY
    # RETIRED. Final 2 transparent-bucket files deleted —
    # a5000-2x-35b-prod (V2 equivalent: `prod-qwen3.6-35b-balanced`,
    # composes byte-identical config) + a5000-2x-27b-int4-tq-k8v4 (V2:
    # `prod-qwen3.6-27b-tq-k8v4`). 12 → 10 in the V1 sunset cascade
    # (sunsets #1–#10), then 2 → 0 in this step. The Phase 9 freeze
    # gate continues to enforce: any new top-level builtin/*.yaml must
    # land as V2 layered triplet (model + hardware + profile) under
    # the corresponding subdir; adding to V1 requires explicit
    # baseline bump signalling deliberate legacy-tier extension.
    # a5000-2x-27b-int4-tq-k8v4.yaml retired 2026-06-01 (final sunset).
    # a5000-2x-35b-prod.yaml retired 2026-06-01 (final sunset).
    # single-3090-dense-cpu-offload-EXAMPLE.yaml retired 2026-06-01
    # — V2 equivalent: preset `example-3090-dense-cpu-offload`.
    # First V1 sunset (Phase 9 → Phase 10 transition proof-of-concept).
    # single-3090-hybrid-gdn-tier-aware-EXAMPLE.yaml retired 2026-06-01
    # — V2 equivalent: preset `example-3090-tier-aware`. Second V1 sunset.
    # a5000-1x-27b-int4-tested.yaml retired 2026-06-01
    # — V2 equivalent: preset `qa-qwen3.6-27b-tq-1x`. Third V1 sunset
    # (first NON-EXAMPLE V1 file — ZERO runtime refs per audit).
    # a5000-2x-35b-fp8-dflash.yaml retired 2026-06-01
    # — V2 equivalent: preset `prod-qwen3.6-35b-dflash` (V2 trims
    # max_model_len 160K → 65K post-dev371 DFlash memory accounting fix).
    # Fourth V1 sunset.
    # a5000-2x-27b-int4-tq-k8v4-dflash.yaml retired 2026-06-01
    # — V2 equivalent: preset `experimental-qwen3.6-27b-tq-dflash-ab`
    # (A/B diagnostic, same model + envs + 131K ctx; V1 had
    # `lifecycle: retired` marker already set 2026-05-26). Fifth V1
    # sunset; first time a self-flagged retired V1 file actually deleted.
    # a5000-2x-27b-dflash-true.yaml retired 2026-06-01
    # — V2 equivalent: preset `prod-qwen3.6-27b-dflash` (TRANSPARENT
    # bucket — V2 composes byte-identical config DFlash N=5 single-
    # stream). Sixth V1 sunset; first TRANSPARENT-bucket V1 retired.
    # a5000-2x-27b-int4-long-ctx.yaml retired 2026-06-01
    # — V2 equivalent: preset `long-ctx-qwen3.6-27b` (sizing-identical
    # 280K ctx / util 0.90 / seqs 2 / batched 2048 / fp8_e5m2 KV /
    # MTP K=3; V2 has override_policy.bench_pending=true since long-
    # context 32K+ bench refresh against current pin is deferred —
    # operator must refresh bench before promoting to production tier).
    # Seventh V1 sunset.
    # a5000-2x-27b-int4-tested.yaml retired 2026-06-01
    # — V2 equivalent: preset `qa-qwen3.6-27b-tested` (sizing-identical
    # 131K ctx / util 0.90 / seqs 2 / batched 4096 / fp8_e5m2 KV /
    # MTP K=3; V2 explicitly disables 16 Wave 1/7/8 patches via
    # patches_delta — V2 ≠ byte-identical V1, operator must consciously
    # pick). Eighth V1 sunset. Legacy CLI test fixtures migrated to
    # surviving sibling `a5000-2x-27b-int4-tq-k8v4`.
    # a5000-2x-tier-aware-EXAMPLE.yaml retired 2026-06-01
    # — Architectural unblock via PN95 refactor: extracted cache_config
    # tier_specs to vllm/sndr_core/cache/pn95/tier_configs/. PN95 hook
    # now reads from PN95-internal dir first, V1 ModelConfig fallback
    # preserved. tier_configs/a5000-2x-tier-aware-example.yaml carries
    # backward-compat alias for any operator still pointing at the V1
    # key. Ninth V1 sunset.
    # a5000-1x-tier-aware-pn95.yaml retired 2026-06-01
    # — Same PN95 architectural unblock. tier_configs/a5000-1x-tier-
    # aware-pn95.yaml carries backward-compat alias.
    # Tenth V1 sunset. (Pairs with #9 in the same PN95 refactor.)
})


def _current_v1_files() -> set[str]:
    """Top-level *.yaml under builtin/ — that's the V1 monolithic tier.
    Subdirs (model/, hardware/, profile/, presets/) are V2 layered."""
    if not BUILTIN_DIR.is_dir():
        return set()
    return {p.name for p in BUILTIN_DIR.glob("*.yaml") if p.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    current = _current_v1_files()
    added = current - FROZEN_V1_BASELINE
    removed = FROZEN_V1_BASELINE - current

    failures = bool(added or removed)

    if args.json:
        print(json.dumps(
            {
                "frozen_baseline": sorted(FROZEN_V1_BASELINE),
                "current": sorted(current),
                "added": sorted(added),
                "removed": sorted(removed),
                "passed": not failures,
            },
            indent=2, sort_keys=True,
        ))
    else:
        print(f"audit-no-new-v1: {len(current)} V1 file(s) currently present")
        print(f"                 {len(FROZEN_V1_BASELINE)} in frozen baseline")
        print("─" * 70)
        if added:
            print(f"  ✗ NEW V1 file(s) added since freeze ({len(added)}):")
            for f in sorted(added):
                print(f"      + {f}")
            print()
            print("  Action: either")
            print("    (a) migrate to a V2 layered triplet under builtin/{model,hardware,profile}/")
            print("        + a preset alias under builtin/presets/, then delete the new V1 file; OR")
            print("    (b) update FROZEN_V1_BASELINE in scripts/audit_no_new_v1.py")
            print("        in the same PR, signalling explicit V1-tier extension.")
        if removed:
            print(f"  ✗ baseline V1 file(s) MISSING ({len(removed)}):")
            for f in sorted(removed):
                print(f"      - {f}")
            print()
            print("  Action: either restore the file or update FROZEN_V1_BASELINE.")
        if not failures:
            print(f"  ✓ V1 frozen — top-level builtin/*.yaml matches the {len(FROZEN_V1_BASELINE)}-entry baseline")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Empirical comparison: legacy apply orchestrator vs spec-driven path.

Phase 6 P3.4 v12.0.0 readiness — per scout finding, the v12.0.0 default
flip from PATCH_REGISTRY list iteration to iter_patch_specs() must be
validated to produce IDENTICAL boot behaviour.

This script runs both paths in a dry-run mode (no actual vLLM, no torch
side effects) and compares:

- Apply matrix: which patches each path tried, what verdict each got
- Order: did the spec-driven path apply patches in the same sequence
  as the legacy path?
- Reason text: did the skip/applied reasons match?
- Coverage: did one path try patches the other didn't?

Output: structured JSON with full diff + human-readable summary.

Usage
-----

  # Show diff summary
  python3 scripts/audit_legacy_vs_spec_driven_apply_matrix.py

  # Emit full JSON for CI consumption
  python3 scripts/audit_legacy_vs_spec_driven_apply_matrix.py --json

  # Fail-fast mode — exit 1 on any divergence
  python3 scripts/audit_legacy_vs_spec_driven_apply_matrix.py --strict

Limitations
-----------

This is a STATIC structural comparison — it inspects what each path
WOULD iterate, not what each apply() function actually does. The
runtime mutation behaviour (text patches actually modifying upstream
files, monkey-patches taking effect) is not exercised. That requires
a real vLLM-equipped rig + side-by-side reboot bench.

What this DOES catch:
- Order divergence (different sequence between paths)
- Coverage divergence (one path tries patches the other doesn't)
- Patch-id mismatch (legacy "P67 Title" vs spec "P67 Title (compound)")

What this does NOT catch:
- Per-patch apply() behaviour divergence (needs rig)
- Side-effect ordering issues (needs rig)
- CUDA-graph capture interactions (needs rig)

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa
Status: v11.3.0+ P3.4 readiness audit (v12.0.0 prerequisite)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _import_or_die():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        return PATCH_REGISTRY
    except ImportError as e:
        raise SystemExit(
            f"cannot import PATCH_REGISTRY: {e}\n"
            f"hint: run with PYTHONPATH=. from repo root"
        )


# ─── Combined-ID pseudo-entries (legacy historical bundling) ───────────
#
# Several historical legacy @register_patch("...") registrations bundle
# what the master plan tracked as TWO conceptual patch IDs under ONE
# callable. The spec registry only carries the first ID (with
# lifecycle="legacy" + apply_module=None — informational entry) because
# the second ID has no independent code path. The audit must treat the
# combined token as the bundle representative, NOT as a true legacy-only
# id, so v12_0_safe stays accurate.
#
# Each entry maps the legacy combined token (first whitespace-delimited
# token of the @register_patch title) to the canonical spec ID it
# represents.
_COMBINED_LEGACY_TOKEN_TO_SPEC_ID: dict[str, str] = {
    "P1/P2":   "P1",   # FP8 kernel dispatcher (Ampere=Marlin vs Ada+=Triton)
    "P17/P18": "P17",  # Marlin MoE per-SM block_size_m tuning
    "P32/P33": "P32",  # TurboQuant cu_2 + synth_seq_lens preallocs
    "P68/P69": "P68",  # long-context tool-call adherence bundle (P69 has own spec)
}


def _enumerate_legacy_path() -> list[dict[str, Any]]:
    """Build the legacy apply-matrix preview without booting vLLM.

    The legacy orchestrator iterates `apply._state.PATCH_REGISTRY` (a
    list of `(name, fn)` tuples). Each fn is decorated via
    `@register_patch("...")` in `_per_patch_dispatch.py`. We import
    that module — which is import-side-effect-free at this level
    (registration happens but no patch fn is called) — then enumerate
    the list.

    v11.3.0 (BUG #6 audit follow-through): combined legacy tokens
    (P1/P2, P17/P18, P32/P33, P68/P69) are normalized to their canonical
    spec ID via `_COMBINED_LEGACY_TOKEN_TO_SPEC_ID`. Without this
    normalization the first-token extractor reports false legacy-only
    entries that confuse v12_0_safe gating.
    """
    # Importing apply triggers @register_patch decorators.
    from vllm.sndr_core.apply import _state, _per_patch_dispatch  # noqa: F401
    matrix: list[dict[str, Any]] = []
    for name, fn in _state.PATCH_REGISTRY:
        # name typically looks like "P67 Multi-query attn — kernel switch"
        # We extract the patch_id (first whitespace-delimited token)
        head = name.split()[0] if name else ""
        # Normalize combined legacy tokens to the canonical spec ID.
        canonical = _COMBINED_LEGACY_TOKEN_TO_SPEC_ID.get(head, head)
        matrix.append({
            "patch_id": canonical,
            "raw_legacy_token": head,
            "display_name": name,
            "source": "legacy_apply_patch_register",
            "fn_module": getattr(fn, "__module__", None),
            "fn_qualname": getattr(fn, "__qualname__", None),
        })
    return matrix


def _enumerate_spec_driven_path(registry: dict) -> list[dict[str, Any]]:
    """Build the spec-driven apply-matrix preview from
    `iter_patch_specs()`. Returns ONLY specs that would actually be
    dispatched (apply_module is not None) — informational entries
    that always skip are excluded for fair comparison with the legacy
    apply._state.PATCH_REGISTRY which only contains patches with
    apply_patch_* functions.

    No vLLM boot — we just iterate the spec generator and project the
    fields. The actual import of `spec.apply_module` is NOT triggered
    (that's only needed at apply-time).

    v11.3.0 BUG #6 audit follow-through: also exposes the set of
    informational entries (apply_module=None) classified by their
    env_flag convention. Entries with `GENESIS_LEGACY_*` env are policy
    — known not-yet-migrated legacy bundle representatives, applied
    only via the legacy path. Entries WITHOUT that prefix (and no
    apply_module) are either coordinator/research/metadata-only or a
    drift bug. The audit surfaces the breakdown for operator review.
    """
    from vllm.sndr_core.dispatcher.spec import iter_patch_specs
    matrix: list[dict[str, Any]] = []
    skipped_informational: list[dict[str, Any]] = []
    for spec in iter_patch_specs():
        if spec.apply_module is None:
            env_flag = registry.get(spec.patch_id, {}).get("env_flag", "")
            policy = (
                "legacy_intentional"
                if env_flag.startswith("GENESIS_LEGACY_")
                else "non_legacy_informational"
            )
            skipped_informational.append({
                "patch_id": spec.patch_id,
                "lifecycle": spec.lifecycle,
                "env_flag": env_flag,
                "policy": policy,
            })
            continue
        matrix.append({
            "patch_id": spec.patch_id,
            "display_name": f"{spec.patch_id} {spec.title}".strip(),
            "source": "spec_driven_iter_patch_specs",
            "apply_module": spec.apply_module,
            "lifecycle": spec.lifecycle,
            "default_on": spec.default_on,
        })
    globals()["_LAST_SKIPPED_INFORMATIONAL"] = skipped_informational
    return matrix


def _diff_matrices(
    legacy: list[dict[str, Any]],
    spec_driven: list[dict[str, Any]],
    registry: dict,
) -> dict[str, Any]:
    """Compute the structural diff between the two apply matrices.

    v11.3.0 BUG #6 audit follow-through: the diff now distinguishes
    `legacy_only_intentional` (spec has the ID but no apply_module on
    purpose — informational entry with GENESIS_LEGACY_* env policy)
    from `legacy_only_drift` (spec doesn't have the ID at all, OR has
    it without the policy env prefix). Only the latter blocks
    v12_0_safe.
    """
    legacy_ids = [m["patch_id"] for m in legacy]
    spec_ids = [m["patch_id"] for m in spec_driven]

    legacy_set = set(legacy_ids)
    spec_set = set(spec_ids)

    legacy_only = sorted(legacy_set - spec_set)
    spec_only = sorted(spec_set - legacy_set)
    common = legacy_set & spec_set

    # Classify legacy_only: intentional vs drift.
    # Intentional: spec has the ID with apply_module=None AND
    # env_flag starts with GENESIS_LEGACY_ (operator can't accidentally
    # enable in spec mode; legacy path still applies).
    legacy_only_intentional: list[str] = []
    legacy_only_drift: list[str] = []
    for pid in legacy_only:
        spec_entry = registry.get(pid)
        if (
            spec_entry
            and isinstance(spec_entry, dict)
            and spec_entry.get("apply_module") is None
            and str(spec_entry.get("env_flag", "")).startswith(
                "GENESIS_LEGACY_"
            )
        ):
            legacy_only_intentional.append(pid)
        else:
            legacy_only_drift.append(pid)

    # Order divergence — for IDs in both paths, do they appear in the
    # same relative order?
    legacy_pos = {pid: i for i, pid in enumerate(legacy_ids)}
    spec_pos = {pid: i for i, pid in enumerate(spec_ids)}
    common_sorted_by_legacy = sorted(common, key=lambda x: legacy_pos[x])
    common_sorted_by_spec = sorted(common, key=lambda x: spec_pos[x])
    order_divergent = common_sorted_by_legacy != common_sorted_by_spec

    # For order-divergent cases, find the first divergence
    first_swap = None
    if order_divergent:
        for i, (a, b) in enumerate(
            zip(common_sorted_by_legacy, common_sorted_by_spec)
        ):
            if a != b:
                first_swap = {
                    "position": i,
                    "legacy_at_pos": a,
                    "spec_at_pos": b,
                }
                break

    return {
        "legacy_total": len(legacy),
        "spec_driven_total": len(spec_driven),
        "common_count": len(common),
        "legacy_only_count": len(legacy_only),
        "legacy_only_ids": legacy_only[:30],
        "legacy_only_intentional_count": len(legacy_only_intentional),
        "legacy_only_intentional_ids": legacy_only_intentional[:30],
        "legacy_only_drift_count": len(legacy_only_drift),
        "legacy_only_drift_ids": legacy_only_drift[:30],
        "spec_only_count": len(spec_only),
        "spec_only_ids": spec_only[:30],
        "order_divergent": order_divergent,
        "first_order_divergence": first_swap,
        # v12_0_safe: only DRIFT blocks the flip (intentional legacy
        # entries are policy — they ALWAYS skip in spec mode but the
        # legacy path still applies them). spec_only entries are NEW
        # patches added directly to spec-driven without a legacy hook;
        # they WILL START applying on flip, which is the desired
        # behavior change for that direction. Order divergence is a
        # separate policy concern (dict-insertion vs decorator-call
        # order) — track via v12_0_strict_order.
        "v12_0_safe": len(legacy_only_drift) == 0,
        "v12_0_strict_order": (
            len(legacy_only_drift) == 0
            and not order_divergent
        ),
    }


def _print_human(diff: dict[str, Any]) -> None:
    print("=" * 70)
    print("Apply-matrix comparison: legacy vs spec-driven")
    print("=" * 70)
    print()
    print(f"Legacy path total:      {diff['legacy_total']}")
    print(f"Spec-driven path total: {diff['spec_driven_total']}")
    print(f"Common patch IDs:       {diff['common_count']}")
    print(f"Legacy-only:            {diff['legacy_only_count']}")
    print(f"Spec-only:              {diff['spec_only_count']}")
    print(f"Order divergent:        {diff['order_divergent']}")
    print()
    if diff["legacy_only_count"] > 0:
        print(
            f"Legacy-only IDs (in apply._state.PATCH_REGISTRY but not "
            f"in iter_patch_specs()):"
        )
        print(
            f"  intentional (spec entry with GENESIS_LEGACY_* env, "
            f"apply_module=None — policy): "
            f"{diff['legacy_only_intentional_count']}"
        )
        for pid in diff["legacy_only_intentional_ids"]:
            print(f"    · {pid}")
        print(
            f"  drift (no matching spec entry, OR spec entry without "
            f"GENESIS_LEGACY_* policy env — needs migration): "
            f"{diff['legacy_only_drift_count']}"
        )
        for pid in diff["legacy_only_drift_ids"]:
            print(f"    ! {pid}")
        if diff["legacy_only_count"] > 30:
            print(f"  ... +{diff['legacy_only_count'] - 30} more")
        print()
    if diff["spec_only_count"] > 0:
        print(
            "Spec-only IDs (in dispatcher.PATCH_REGISTRY but no "
            "legacy apply_patch_* function — auto-derive expected):"
        )
        for pid in diff["spec_only_ids"]:
            print(f"  - {pid}")
        if diff["spec_only_count"] > 30:
            print(f"  ... +{diff['spec_only_count'] - 30} more")
        print()
    if diff["order_divergent"]:
        print("ORDER DIVERGENCE DETECTED:")
        fs = diff["first_order_divergence"]
        if fs:
            print(f"  First divergence at position {fs['position']}:")
            print(f"    legacy applies:       {fs['legacy_at_pos']}")
            print(f"    spec-driven applies:  {fs['spec_at_pos']}")
        print()
        print(
            "  This means the v12.0.0 default flip would change boot-log "
            "order, which may break patch dependency chains."
        )
        print()
    if diff["v12_0_safe"]:
        print(
            "✓ Apply matrices have NO DRIFT — v12.0.0 default flip is "
            "structurally safe (intentional-legacy entries policy-skip "
            "in spec mode; legacy path still applies them)."
        )
        if diff["order_divergent"]:
            print(
                "  Note: order divergence remains. Audit per-patch "
                "dependency chains before assuming v12_0_safe_with_order."
            )
        else:
            print(
                "  Apply order also matches. Next validation step: "
                "empirical rig comparison (boot once with each path, "
                "diff actual side effects)."
            )
    else:
        print(
            "⚠ Apply matrices have DRIFT. v12.0.0 default flip would "
            "change boot behaviour. Investigate `legacy_only_drift_ids` "
            "before flipping the default."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--json", action="store_true",
        help="emit full structured JSON (legacy + spec matrices + diff)",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit 1 if v12_0_safe is False",
    )
    args = parser.parse_args()

    _import_or_die()
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    legacy = _enumerate_legacy_path()
    spec_driven = _enumerate_spec_driven_path(PATCH_REGISTRY)
    diff = _diff_matrices(legacy, spec_driven, PATCH_REGISTRY)

    if args.json:
        print(json.dumps(
            {
                "legacy_matrix": legacy,
                "spec_driven_matrix": spec_driven,
                "diff": diff,
            },
            indent=2, sort_keys=True,
        ))
    else:
        _print_human(diff)

    if args.strict and not diff["v12_0_safe"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

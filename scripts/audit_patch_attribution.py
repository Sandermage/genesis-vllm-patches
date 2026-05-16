#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§Phase A patch-attribution audit — `make audit-patch-attribution`.

`ModelDef.patches_attribution` is the Phase A schema extension that
stores WHY a patch sits in the model's canonical set (role +
note/bench_evidence/candidate_when). Schema validators in
`schema_v2.PatchAttribution.validate()` catch *syntactic* drift
(unknown role, role-conditional required fields missing). This audit
gate catches *semantic* drift that schema validation cannot:

  AT-1  patches_attribution key references a patch ID that exists
        in PATCH_REGISTRY (catches typos like `PN204b` vs `PN204`).
  AT-2  if patches_attribution[ID] declares a role that asserts the
        patch is part of the model's active set (load_bearing,
        defensive, optional_perf), the matching env_flag must also
        exist in model.patches. Roles that explain intentional
        ABSENCE (suspected_regression, no_op, unknown) are exempt —
        the whole point of suspected_regression is to document a
        patch that was excluded from this model.
  AT-3  coverage statistic: report what fraction of model.patches
        env flags have an attribution entry. Not gating — informational
        only, but trends matter (Phase B/C consume this data).

The gate is GATING on AT-1 and AT-2, INFORMATIONAL on AT-3.

Exit codes:
  0 — every model YAML is attribution-consistent
  1 — at least one AT-1 / AT-2 violation
  2 — internal error
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = (
    REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "model"
)


@dataclass
class AttributionCheck:
    model_id: str
    path: Path
    total_patches: int = 0
    total_attributions: int = 0
    unknown_patch_ids: list[str] = field(default_factory=list)
    attribution_without_flag: list[str] = field(default_factory=list)
    parse_error: str = ""

    @property
    def coverage_pct(self) -> float:
        if self.total_patches == 0:
            return 100.0
        # Attributions covering patches that exist in model.patches.
        covered = self.total_attributions - len(self.attribution_without_flag)
        return 100.0 * max(0, covered) / self.total_patches

    @property
    def passed(self) -> bool:
        return (
            not self.parse_error
            and not self.unknown_patch_ids
            and not self.attribution_without_flag
        )


def _load_registry_index() -> dict[str, str]:
    """Return mapping {patch_id → env_flag} from PATCH_REGISTRY."""
    sys.path.insert(0, str(REPO_ROOT))
    from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
    out: dict[str, str] = {}
    for pid, meta in PATCH_REGISTRY.items():
        flag = meta.get("env_flag")
        if isinstance(flag, str) and flag:
            out[pid] = flag
    return out


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _check_one(
    path: Path,
    registry_flags: dict[str, str],
) -> AttributionCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return AttributionCheck(
            model_id=path.stem, path=path,
            parse_error=f"YAML parse error: {e}",
        )
    if not isinstance(data, dict):
        return AttributionCheck(
            model_id=path.stem, path=path,
            parse_error="top-level is not a mapping",
        )

    patches = data.get("patches") or {}
    attributions = data.get("patches_attribution") or {}
    chk = AttributionCheck(
        model_id=str(data.get("id", path.stem)),
        path=path,
        total_patches=len(patches),
        total_attributions=len(attributions),
    )

    # AT-1: every attribution key must be a known registry patch ID.
    for pid in attributions:
        if pid not in registry_flags:
            chk.unknown_patch_ids.append(pid)

    # AT-2: roles that assert presence (load_bearing/defensive/
    # optional_perf) must have the matching env_flag in model.patches.
    # Roles that explain ABSENCE (suspected_regression/no_op/unknown)
    # are exempt — they intentionally document patches kept out of this
    # model's active set.
    _ASSERTS_PRESENCE = {"load_bearing", "defensive", "optional_perf"}
    for pid, attr in attributions.items():
        if not isinstance(attr, dict):
            continue
        role = attr.get("role")
        if role not in _ASSERTS_PRESENCE:
            continue
        flag = registry_flags.get(pid)
        if flag and flag not in patches:
            chk.attribution_without_flag.append(pid)

    return chk


def audit_patch_attribution() -> tuple[dict[str, str], list[AttributionCheck]]:
    registry_flags = _load_registry_index()
    results: list[AttributionCheck] = []
    if MODEL_DIR.is_dir():
        for yp in sorted(MODEL_DIR.glob("*.yaml")):
            results.append(_check_one(yp, registry_flags))
    return registry_flags, results


# ─── Renderers ──────────────────────────────────────────────────────────


def _render_text(
    registry_flags: dict[str, str],
    results: list[AttributionCheck],
) -> tuple[str, bool]:
    lines = [
        f"audit-patch-attribution: {len(results)} model YAML(s) scanned",
        f"  registry size: {len(registry_flags)} patches with env_flag",
        "─" * 70,
    ]
    total_patches = sum(r.total_patches for r in results)
    total_attribs = sum(r.total_attributions for r in results)
    lines.append(
        f"  coverage: {total_attribs} attributions over {total_patches} "
        f"patches across {len(results)} models"
    )
    lines.append("")
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.parse_error:
            lines.append(f"  {sym} {r.model_id}: {r.parse_error}")
            continue
        lines.append(
            f"  {sym} {r.model_id}: "
            f"{r.total_attributions}/{r.total_patches} attributed "
            f"({r.coverage_pct:.0f}% coverage)"
        )
        for pid in r.unknown_patch_ids:
            lines.append(f"      AT-1 unknown patch id: {pid}")
        for pid in r.attribution_without_flag:
            lines.append(
                f"      AT-2 attribution without matching env_flag in "
                f"model.patches: {pid}"
            )
    lines.append("")
    passed = all(r.passed for r in results)
    if not passed:
        lines.append(
            "  ✗ Fix: align patches_attribution keys with PATCH_REGISTRY "
            "ids and the env_flags listed in model.patches."
        )
    else:
        lines.append(
            "  ✓ All model YAMLs are attribution-consistent."
        )
    return "\n".join(lines), passed


def _render_json(
    registry_flags: dict[str, str],
    results: list[AttributionCheck],
) -> tuple[str, bool]:
    passed = all(r.passed for r in results)
    payload = {
        "registry_flag_count": len(registry_flags),
        "models_scanned": len(results),
        "total_patches": sum(r.total_patches for r in results),
        "total_attributions": sum(r.total_attributions for r in results),
        "passed": passed,
        "results": [
            {
                "model_id": r.model_id,
                "path": str(r.path.relative_to(REPO_ROOT)),
                "total_patches": r.total_patches,
                "total_attributions": r.total_attributions,
                "coverage_pct": round(r.coverage_pct, 1),
                "unknown_patch_ids": r.unknown_patch_ids,
                "attribution_without_flag": r.attribution_without_flag,
                "parse_error": r.parse_error or None,
                "passed": r.passed,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True), passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON output.")
    ap.add_argument(
        "--min-coverage", type=float, default=None,
        help=(
            "Optional informational threshold (percent). When set, the "
            "gate also fails when the overall coverage % is below this "
            "value. Use to ratchet attribution coverage upward over time. "
            "Default: no coverage floor — only consistency is gating."
        ),
    )
    args = ap.parse_args()

    try:
        registry_flags, results = audit_patch_attribution()
    except Exception as e:
        sys.stderr.write(f"audit-patch-attribution: {e}\n")
        return 2

    if args.json:
        out, ok = _render_json(registry_flags, results)
    else:
        out, ok = _render_text(registry_flags, results)
    print(out)

    # Optional coverage floor — operator-set ratchet.
    if args.min_coverage is not None and ok:
        total_patches = sum(r.total_patches for r in results)
        total_attribs = sum(r.total_attributions for r in results)
        coverage = (
            100.0 * total_attribs / total_patches if total_patches else 100.0
        )
        if coverage < args.min_coverage:
            sys.stderr.write(
                f"audit-patch-attribution: coverage {coverage:.1f}% < "
                f"--min-coverage {args.min_coverage}% — ratchet failed\n"
            )
            return 1

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

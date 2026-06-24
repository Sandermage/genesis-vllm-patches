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
MODEL_DIR = REPO_ROOT / "sndr" / "model_configs" / "builtin" / "model"

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
    "PN394": (
        "Pin bump dev148 -> dev301 retire 2026-06-24 — vllm#46047 (qwen3 "
        "partial-param `<`-truncation fix) is IN dev301 (anchor drifted per "
        "the dev301 anchor-SOT regen). Retired + capped <0.23.1rc1.dev301. "
        "Env flag GENESIS_ENABLE_PN394_QWEN3_PARTIAL_PARAM_LT_FIX=1 is still "
        "set in the 27B/35B PROD YAMLs INTENTIONALLY: on dev301 the wiring "
        "self-skips (version cap + the post-fix `>(.*)$` drift marker) so it "
        "is harmless, AND it stays correct on a dev148 rollback (the bug is "
        "live on dev148, the previous/rollback pin). Leave enabled until "
        "dev148 rollback is no longer possible, then clean from YAMLs."
    ),
    "PN22": (
        "0.23.1 reverify 2026-06-17 — vllm#39419 (LocalArgmaxMixin) merged at "
        "bd2d83ff, ancestor of 0.23.1rc1.dev101; mixin native on the deployed "
        "pin (live-verified). Retired + capped <0.23.1rc1.dev101. Env flag "
        "still set in 27B/35B prod YAMLs; the wiring self-skips (default_off + "
        "drift marker), harmless. Cleaned at next config audit cycle."
    ),
    "PN90": (
        "0.23.1 reverify 2026-06-17 — vllm#40269 merged at f51f6844 (present on "
        "both dev491 and 0.23.1), proposer symbols native. Retired; range "
        "already <0.22.0 so it self-skips on the deployed pin. Env flag still "
        "set in prod YAMLs; harmless (does NOT enable upstream probabilistic "
        "draft — that path is the -5.9% TPS regression, intentionally left off)."
    ),
    "PN19": "carry-over from W-A; replacement is part of PN-series consolidation work",
    "PN52": "still actively consumed by 27B INT4 / 35B FP8 prod path",
    "P94":  "enabled in 27B INT4 TQ + 35B FP8 prod — operator review pending",
    "PN82": (
        "K.1.R 2026-05-28 — vllm#41873 merged at 39d5fa96 in window "
        "dev371→626fa9bb, byte-equivalent retire. Env flag still set in "
        "9 model_config YAMLs (27B + 35B prod path); the wiring now "
        "self-skips with a retirement-stub return so leaving the env "
        "set is harmless. Will be cleaned out of YAMLs at next config "
        "audit cycle, not blocking the pin bump."
    ),
    "PN132": (
        "Iron-rule-#11 retire 2026-05-30 (session commit 6082d8a4) — "
        "upstream vllm#42739 merged 2026-05-23 at d19db10974587 in window "
        "dev371→626fa9bb, root-cause fix (stride-aware Triton kernel) is "
        "strictly better than PN132's `.contiguous()` workaround. Env flag "
        "still set in 8 model_config YAMLs (gemma4 26B/31B + qwen3.6 27B "
        "dflash/tq + 35B fp8 dflash/full); the wiring's apply() now "
        "self-skips via inspect.signature check for the post-merge "
        "`mask_value` kwarg, so leaving the env set is harmless on the new "
        "pin. Will be cleaned out of YAMLs at next config audit cycle."
    ),
    "P78": (
        "Preflight residual triage retire 2026-06-11 §3 — upstream "
        "absorbed Sites B/C/D/E (CPU-mirror metadata native in pristine "
        "turboquant_attn.py:190-193/237-238/486-489/601-610 on pin "
        "0.22.1rc1.dev259; the buggy GPU .tolist() pattern is gone). "
        "Env flag still set =1 in qwen3.6-35b-a3b-fp8-dflash YAML; "
        "lifecycle='retired' makes the dispatcher skip it, so leaving "
        "the env set is harmless on the new pin. Will be cleaned out of "
        "the YAML at next config audit cycle."
    ),
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
        from sndr.dispatcher.registry import PATCH_REGISTRY
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

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§4.2 V2 capability-coverage gate.

V2 model.capabilities carries operational capability strings:

  • `attention_arch`         — model's attention shape, drives PN59/P98/PN11 routing
  • `tool_call_parser`       — vllm-side tool-call parser name
  • `reasoning_parser`       — vllm-side reasoning parser name (qwen3 / etc.)
  • `kv_cache_dtype`         — KV cache quantization mode
  • `spec_decode.method`     — speculative-decode strategy

A typo in any of these strings silently selects the WRONG code path
at runtime (or `None` fallback, also wrong). This gate freezes the
allowed value set for each field, derived from a survey of every
committed V2 model.

Adding a new value requires explicit operator decision + ledger
entry. The frozen schema is the regression anchor.

Exit codes:
  0 — every capability value is in the allowed set
  1 — at least one unknown / typo'd value
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


# ─── Allowed value sets (frozen as code) ──────────────────────────────
#
# Each entry maps a capability path to its allowed value set. `None` is
# always allowed for optional capabilities. Path uses dot notation for
# nested keys (e.g. `spec_decode.method`).

# Phase 5.2.B (2026-05-22) — allowlists extended to cover the Gemma 4
# workstream. The Gemma family does not share Qwen's attention
# architecture or parser conventions, so its values were missing from
# the original (Qwen-only) allowlist. Adding them is "audit semantics
# catch up to production shape", not a relaxation of the contract —
# every new value is an explicit canonical token used by a shipped
# ModelDef under `builtin/model/gemma-4-*.yaml`.
#
# `auto` is the engine's "pick a sensible default" sentinel for
# kv_cache_dtype — vLLM honors it natively and gemma-4 ModelDefs use
# it to opt out of explicit fp8/fp16 selection.
ALLOWED_CAPABILITIES: dict[str, frozenset] = {
    "attention_arch":         frozenset({
        "dense", "hybrid_gdn_moe",
        "gemma4_dense", "gemma4_moe",
    }),
    "tool_call_parser":       frozenset({"qwen3_coder", "gemma4"}),
    # `None` is the canonical value for ModelDefs without a thinking-tag
    # parser (gemma-4 does not emit `</think>`-style traces). The
    # docstring's "None is always allowed" promise is not auto-applied
    # by the audit; explicit membership is required.
    "reasoning_parser":       frozenset({"qwen3", None}),
    "kv_cache_dtype":         frozenset({
        "fp8_e5m2", "fp8_e4m3", "turboquant_k8v4", "fp16", "auto", None,
    }),
    "spec_decode.method":     frozenset({
        "mtp", "dflash", "ngram", None,
    }),
}


@dataclass
class CapCheck:
    path: Path
    model_id: str
    violations: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return not self.error and not self.violations


def _load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _walk(d, path: str):
    """Walk dot-notation path. Returns (found, value)."""
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return False, None
        cur = cur[key]
    return True, cur


def check_one_model(path: Path) -> CapCheck:
    try:
        data = _load_yaml(path)
    except Exception as e:
        return CapCheck(path=path, model_id="?",
                       error=f"YAML parse error: {e}")
    model_id = data.get("id", path.stem)
    caps = data.get("capabilities") or {}
    if not isinstance(caps, dict):
        return CapCheck(path=path, model_id=model_id,
                       error="capabilities block is not a mapping")
    violations: list[dict] = []
    for cap_path, allowed in ALLOWED_CAPABILITIES.items():
        found, val = _walk(caps, cap_path)
        if not found:
            # Absent fields are OK — required-fields gate handles presence.
            continue
        # For spec_decode.method, the parent `spec_decode` may be None
        # (model has no spec-decode capability). `_walk` returns
        # (True, None) in that case → skipped silently below.
        if val is None and None in allowed:
            continue
        if val not in allowed:
            violations.append({
                "capability": cap_path,
                "value": val,
                "allowed": sorted(
                    [v for v in allowed if v is not None],
                ),
            })
    return CapCheck(
        path=path, model_id=model_id, violations=violations,
    )


def audit_v2_capability_coverage(
    model_dir: Path = MODEL_DIR,
) -> list[CapCheck]:
    if not model_dir.is_dir():
        return []
    return [check_one_model(p) for p in sorted(model_dir.glob("*.yaml"))]


def _render_text(results: list[CapCheck]) -> str:
    lines = [
        f"audit-v2-capability-coverage: {len(results)} model YAML(s)",
        "─" * 70,
    ]
    for r in results:
        sym = "✓" if r.passed else "✗"
        if r.error:
            lines.append(f"  {sym} {r.model_id}: {r.error}")
            continue
        lines.append(f"  {sym} {r.model_id}")
        for v in r.violations:
            lines.append(
                f"      ⚠ {v['capability']}={v['value']!r} not in {v['allowed']}"
            )
    passed = sum(1 for r in results if r.passed)
    lines.append("─" * 70)
    lines.append(f"  {passed}/{len(results)} models clean")
    return "\n".join(lines)


def _render_json(results: list[CapCheck]) -> str:
    return json.dumps({
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "allowed_capabilities": {
            k: sorted([v for v in vals if v is not None])
            for k, vals in ALLOWED_CAPABILITIES.items()
        },
        "models": [
            {
                "model_id": r.model_id,
                "path": _rel(r.path),
                "passed": r.passed,
                "violations": r.violations,
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
    results = audit_v2_capability_coverage()
    print(_render_json(results) if args.json else _render_text(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

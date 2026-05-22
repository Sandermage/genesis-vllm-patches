#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V2 ModelDef ↔ hardware canonical-pin drift audit.

Phase 5.2.E (2026-05-22) — close the silent-drift gap that allowed
qwen3.6-7b-dense to ship with `vllm_pin_required: dev338+gbf0d2dc6d`
while every hardware YAML on the rig was already on dev371+gbf610c2f5
(see Phase 5.2 V2 config completeness audit, finding F-Q7B).

The existing `audit_v2_runtime_pins.py` R-PIN-4 rule accepts BOTH
dev338 and dev371 as legitimate ModelDef pin values because per-model
P2.4d promotion happens one at a time. That's the right granularity
for migration-state tracking. This new gate is stricter and orthogonal:
it asks "does this ModelDef's pin actually match the rig you'd deploy
it on, or is the operator carrying an implicit hold?"

Rules:

  R-MD-HW-1 — hardware self-consistency.
    All hardware YAMLs under `builtin/hardware/` must pin the same
    `runtime.docker.image` SHA fragment. A split-rig scenario where
    one card runs dev338 and another dev371 is out of scope and
    rejected here (the operator should resolve the split before
    re-running this audit).

  R-MD-HW-2 — ModelDef-vs-hardware SHA equality, waiver-aware.
    For each `builtin/model/*.yaml`:
      * Extract the SHA fragment after the `+g` token in
        `versions.vllm_pin_required` (e.g. `dev371+gbf610c2f5`
        → `bf610c2f5`).
      * Compare against the canonical hardware SHA prefix.
      * On mismatch: PASS if `versions.pin_hold` is a non-empty
        string; ERROR otherwise.

The `pin_hold` field was added in Phase 5.2.C (schema_v2.py) and
populated on qwen3.6-7b-dense to carry the placeholder-checkpoint
rationale. Any future ModelDef that intentionally lags the rig pin
must declare its hold reason inline.

The script is read-only — never modifies YAML, registry, or git state.

Exit codes:
  0 — all rules pass.
  1 — at least one violation found.
  2 — audit tooling itself failed (missing file, no hardware images).

Usage:
  python3 scripts/audit_v2_modeldef_vs_hardware_pin.py
  python3 scripts/audit_v2_modeldef_vs_hardware_pin.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HARDWARE_DIR = (
    REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "hardware"
)
MODEL_DIR = (
    REPO_ROOT / "vllm" / "sndr_core" / "model_configs" / "builtin" / "model"
)


# ─── YAML scalar extractors (regex; no PyYAML dependency) ──────────────────

_HW_IMAGE_RE = re.compile(
    r"^\s{4}image:\s*(?P<value>\S+)", re.MULTILINE,
)
_MODEL_PIN_RE = re.compile(
    r"^\s{2}vllm_pin_required:\s*(?P<value>\S+)", re.MULTILINE,
)
_MODEL_PIN_HOLD_RE = re.compile(
    r"^\s{2}pin_hold:\s*(?P<value>.+?)(?:\s*#.*)?$",
    re.MULTILINE,
)

# SHA fragment in a hardware image tag: `:nightly-<sha40>`
_IMAGE_SHA_RE = re.compile(r"nightly-(?P<sha>[0-9a-f]{40})")
# SHA fragment in a ModelDef pin: `+g<sha10>`
_PIN_SHA_RE = re.compile(r"\+g(?P<sha>[0-9a-f]{6,40})")


def _strip_yaml_value(raw: str) -> str:
    """Strip trailing inline comment and surrounding quotes from a scalar."""
    if " #" in raw:
        raw = raw.split(" #", 1)[0]
    return raw.strip().strip('"').strip("'")


def _read_hardware_sha(path: Path) -> str | None:
    """Return the 40-hex SHA fragment of the image tag, or None."""
    m = _HW_IMAGE_RE.search(path.read_text())
    if not m:
        return None
    image = _strip_yaml_value(m.group("value"))
    s = _IMAGE_SHA_RE.search(image)
    return s.group("sha") if s else None


def _read_model_pin(path: Path) -> str | None:
    m = _MODEL_PIN_RE.search(path.read_text())
    return _strip_yaml_value(m.group("value")) if m else None


def _read_model_pin_hold(path: Path) -> str | None:
    """Return pin_hold scalar if present + non-empty + non-`null`/`none`."""
    src = path.read_text()
    # Match only inside the `versions:` block — top-level scalar pin_hold
    # would otherwise be ambiguous. The regex requires 2-space indent
    # which the ModelVersions block uses.
    m = _MODEL_PIN_HOLD_RE.search(src)
    if not m:
        return None
    val = _strip_yaml_value(m.group("value"))
    if not val or val.lower() in ("null", "none", "~"):
        return None
    return val


def _pin_sha(pin: str) -> str | None:
    """Extract the short SHA fragment after `+g` from a vLLM pin string."""
    m = _PIN_SHA_RE.search(pin)
    return m.group("sha") if m else None


# ─── Rule implementations ─────────────────────────────────────────────────


@dataclass
class RuleResult:
    rule_id: str
    passed: bool
    violations: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)


def rule_md_hw_1() -> tuple[RuleResult, str | None]:
    """Hardware SHA self-consistency. Returns (result, canonical_sha)."""
    rr = RuleResult(rule_id="R-MD-HW-1", passed=True)
    seen: dict[str, list[str]] = {}
    for p in sorted(HARDWARE_DIR.glob("*.yaml")):
        sha = _read_hardware_sha(p)
        if sha is None:
            rr.violations.append(
                f"{p.name}: runtime.docker.image lacks `:nightly-<sha40>` "
                f"suffix"
            )
            continue
        seen.setdefault(sha, []).append(p.name)
    if not seen:
        rr.passed = False
        rr.violations.append(
            "no hardware YAML had an extractable image SHA — "
            "audit can't determine canonical rig pin"
        )
        return rr, None
    if len(seen) > 1:
        rr.passed = False
        details = ", ".join(
            f"{sha[:10]}…={files}" for sha, files in sorted(seen.items())
        )
        rr.violations.append(
            f"hardware split-rig drift detected: {details}"
        )
        return rr, None
    canonical_sha = next(iter(seen))
    rr.info.append(
        f"canonical hardware SHA: {canonical_sha[:10]}… "
        f"({len(seen[canonical_sha])} hardware YAML(s))"
    )
    if rr.violations:
        rr.passed = False
    return rr, canonical_sha


def rule_md_hw_2(canonical_sha: str) -> RuleResult:
    """ModelDef-vs-hardware SHA equality, waiver-aware."""
    rr = RuleResult(rule_id="R-MD-HW-2", passed=True)
    canonical_prefix = canonical_sha  # full 40 hex
    for p in sorted(MODEL_DIR.glob("*.yaml")):
        pin = _read_model_pin(p)
        if pin is None:
            rr.violations.append(
                f"{p.name}: no `versions.vllm_pin_required` field"
            )
            continue
        sha = _pin_sha(pin)
        if sha is None:
            rr.violations.append(
                f"{p.name}: vllm_pin_required={pin!r} has no `+g<sha>` token"
            )
            continue
        # Match short pin SHA against the 40-char canonical prefix.
        if canonical_prefix.startswith(sha):
            rr.info.append(f"{p.name:48s}  match  ({sha})")
            continue
        # Mismatch — check for pin_hold waiver.
        hold = _read_model_pin_hold(p)
        if hold:
            rr.info.append(
                f"{p.name:48s}  HOLD   ({sha} vs {canonical_prefix[:10]}…)\n"
                f"      reason: {hold[:120]}{'…' if len(hold) > 120 else ''}"
            )
            continue
        rr.passed = False
        rr.violations.append(
            f"{p.name}: vllm_pin_required SHA {sha!r} != canonical "
            f"{canonical_prefix[:10]}…; no `versions.pin_hold` rationale "
            f"set. Either promote the ModelDef to the canonical pin OR "
            f"add a `pin_hold:` annotation explaining the deliberate hold."
        )
    return rr


def run_all() -> tuple[list[RuleResult], int]:
    results: list[RuleResult] = []
    rr1, canonical_sha = rule_md_hw_1()
    results.append(rr1)
    if not rr1.passed or canonical_sha is None:
        return results, 1
    results.append(rule_md_hw_2(canonical_sha))
    failed = any(not r.passed for r in results)
    return results, (1 if failed else 0)


# ─── CLI ──────────────────────────────────────────────────────────────────


def _print_report(results: list[RuleResult]) -> None:
    print()
    print("╭──────────────────────────────────────────────────────────╮")
    print("│  V2 ModelDef ↔ hardware canonical-pin drift audit       │")
    print("╰──────────────────────────────────────────────────────────╯")
    for r in results:
        mark = "✓" if r.passed else "✗"
        print(f"  {mark} {r.rule_id:10s}  {len(r.violations)} violation(s)")
        for v in r.violations:
            print(f"      ✗ {v}")
        for i in r.info:
            print(f"      · {i}")
    print()
    if all(r.passed for r in results):
        print("  ✓ All rules clean")
    else:
        print(
            "  ✗ Fix: either bump model.versions.vllm_pin_required to the "
            "canonical hardware SHA, or add a `versions.pin_hold` rationale "
            "explaining the deliberate hold (see qwen3.6-7b-dense.yaml for "
            "an example)."
        )


def _print_json(results: list[RuleResult]) -> None:
    out = [
        {
            "rule_id": r.rule_id,
            "passed": r.passed,
            "violations": r.violations,
            "info": r.info,
        }
        for r in results
    ]
    print(json.dumps(out, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json", action="store_true",
                    help="JSON output for CI / dashboards")
    args = ap.parse_args()
    try:
        results, code = run_all()
    except Exception as e:
        print(f"audit tooling failure: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2
    if args.json:
        _print_json(results)
    else:
        _print_report(results)
    return code


if __name__ == "__main__":
    sys.exit(main())

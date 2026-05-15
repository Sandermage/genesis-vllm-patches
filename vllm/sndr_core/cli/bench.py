# SPDX-License-Identifier: Apache-2.0
"""Phase 6 — `sndr bench` subcommand tree (methodology contract).

`sndr bench-compare` (S2.5) already exists as a top-level alias for the
A/B harness. Phase 6 adds a structured `sndr bench` parent with:

  sndr bench validate <result.json> [--methodology <yaml>] [--json]
      Verify that a bench artefact JSON carries every mandatory field
      named in `vllm/sndr_core/tools/bench_methodology.yaml`, matches the methodology
      fingerprint, and respects the warmup/measure/CV protocol.
      Exit 0 on pass, 1 on validation errors, 2 on internal errors.

  sndr bench methodology [--json]
      Print the active methodology contract — operator-readable summary
      of what bench artefacts must contain.

The compare command is left as-is at the top level for backward compat;
a future deprecation may consolidate it under `sndr bench compare`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Optional


__all__ = [
    "add_argparser",
    "run_validate",
    "run_methodology",
    "load_methodology",
    "methodology_sha",
]


# Wave 10 (2026-05-15) refactor: canonical location moved INSIDE sndr_core
# package to make it self-contained. parents[0]=cli, [1]=sndr_core,
# [2]=vllm, [3]=repo. Internal path: parents[1] / tools / yaml.
# Legacy repo-root path kept as fallback for older checkouts.
_DEFAULT_METHODOLOGY_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "bench_methodology.yaml"
)
_LEGACY_METHODOLOGY_PATH = (
    Path(__file__).resolve().parents[3] / "tools" / "bench_methodology.yaml"
)


def add_argparser(subparsers: Any) -> None:
    """Phase 6 wires TWO top-level commands instead of a `bench` parent —
    the `bench` name is already bridged to `compat.cli bench`
    (the legacy genesis_bench_suite.py runner). Top-level aliases match
    the existing `bench-compare` (S2.5) pattern."""

    p_val = subparsers.add_parser(
        "bench-validate",
        help="Validate a bench result JSON against the methodology contract (Phase 6).",
        description=(
            "Verify that a bench artefact JSON carries every mandatory field "
            "named in `vllm/sndr_core/tools/bench_methodology.yaml`, matches the methodology "
            "fingerprint, and respects the warmup/measure/CV protocol."
        ),
    )
    p_val.add_argument("artefact",
                       help="Path to a genesis_bench_suite.py JSON result.")
    p_val.add_argument("--methodology", default=None,
                       help="Path to methodology YAML (default: vllm/sndr_core/tools/bench_methodology.yaml).")
    p_val.add_argument("--json", action="store_true",
                       help="Emit machine-readable JSON.")
    p_val.set_defaults(func=run_validate)

    p_m = subparsers.add_parser(
        "bench-methodology",
        help="Print the active bench methodology contract (Phase 6).",
        description=(
            "Operator-readable summary of what bench artefacts must contain "
            "for release-tier gating to accept them."
        ),
    )
    p_m.add_argument("--methodology", default=None,
                     help="Path to methodology YAML.")
    p_m.add_argument("--json", action="store_true")
    p_m.set_defaults(func=run_methodology)


# ─── Helpers ───────────────────────────────────────────────────────────


def load_methodology(path: Optional[Path] = None) -> dict:
    """Load methodology YAML. Returns parsed dict; raises on file/parse errors.

    Wave 10 path resolution:
      1. Explicit `path` argument wins if provided.
      2. Canonical: vllm/sndr_core/tools/bench_methodology.yaml.
      3. Operator-side fallback: <repo-root>/tools/bench_methodology.yaml
         (legacy location for dev checkouts where the file was not yet
         moved into the package).
    """
    if path is not None:
        fp = Path(path)
    elif _DEFAULT_METHODOLOGY_PATH.is_file():
        fp = _DEFAULT_METHODOLOGY_PATH
    elif _LEGACY_METHODOLOGY_PATH.is_file():
        fp = _LEGACY_METHODOLOGY_PATH
    else:
        fp = _DEFAULT_METHODOLOGY_PATH  # surface canonical path in the error
    if not fp.is_file():
        raise FileNotFoundError(
            f"methodology file not found: {fp}. "
            f"Phase 6 expects vllm/sndr_core/tools/bench_methodology.yaml "
            f"(canonical sndr_core path). Operator-side fallback: "
            f"repo-root tools/bench_methodology.yaml."
        )
    import yaml
    with fp.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"methodology {fp} did not parse to a dict")
    if data.get("schema_version") != 1:
        raise ValueError(
            f"methodology schema_version={data.get('schema_version')!r} unsupported "
            f"(expected 1)"
        )
    return data


def methodology_sha(path: Optional[Path] = None) -> str:
    """SHA-256 of methodology YAML — the fingerprint bench artefacts carry."""
    fp = Path(path) if path else _DEFAULT_METHODOLOGY_PATH
    h = hashlib.sha256()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── validate ──────────────────────────────────────────────────────────


def _validate_artefact(artefact: dict, methodology: dict, mfile: Path) -> list[dict]:
    """Return a list of issue dicts (each {rule, severity, message})."""
    issues: list[dict] = []
    expected_sha = methodology_sha(mfile)
    expected_id = methodology.get("methodology_id")
    expected_warmup = methodology.get("measurement", {}).get("warmup_runs")
    expected_measure = methodology.get("measurement", {}).get("measure_runs")
    cv_warn = methodology.get("measurement", {}).get("cv_warn_pct", 5.0)
    cv_fail = methodology.get("measurement", {}).get("cv_fail_pct", 10.0)
    required = methodology.get("required_artefact_fields", [])

    # M-1: required fields present.
    for field_name in required:
        if field_name not in artefact:
            issues.append({
                "rule": "M-1",
                "severity": "error",
                "message": f"missing required artefact field: {field_name!r}",
            })

    # M-2: methodology id matches.
    if "methodology_id" in artefact and artefact["methodology_id"] != expected_id:
        issues.append({
            "rule": "M-2",
            "severity": "error",
            "message": (
                f"methodology_id mismatch: artefact={artefact['methodology_id']!r}, "
                f"contract={expected_id!r}"
            ),
        })

    # M-3: methodology fingerprint matches the YAML on disk.
    if "methodology_sha" in artefact and artefact["methodology_sha"] != expected_sha:
        issues.append({
            "rule": "M-3",
            "severity": "error",
            "message": (
                f"methodology_sha mismatch: artefact carries "
                f"{artefact['methodology_sha'][:16]}..., contract is "
                f"{expected_sha[:16]}... — bench ran against a different "
                f"methodology revision; re-run or update contract"
            ),
        })

    # M-4: warmup_runs + measure_runs match protocol.
    aw = artefact.get("warmup_runs")
    if aw is not None and expected_warmup is not None and aw != expected_warmup:
        issues.append({
            "rule": "M-4",
            "severity": "error",
            "message": (
                f"warmup_runs mismatch: artefact={aw}, contract={expected_warmup}"
            ),
        })
    am = artefact.get("measure_runs")
    if am is not None and expected_measure is not None and am != expected_measure:
        issues.append({
            "rule": "M-4",
            "severity": "error",
            "message": (
                f"measure_runs mismatch: artefact={am}, contract={expected_measure}"
            ),
        })

    # M-5: CV within tolerance.
    cv = artefact.get("cv_pct")
    if isinstance(cv, (int, float)):
        if cv > cv_fail:
            issues.append({
                "rule": "M-5",
                "severity": "error",
                "message": (
                    f"cv_pct={cv} exceeds cv_fail_pct={cv_fail} — bench too noisy "
                    f"for release; rerun with more measure_runs or fix the GPU "
                    f"clock state"
                ),
            })
        elif cv > cv_warn:
            issues.append({
                "rule": "M-5",
                "severity": "warning",
                "message": (
                    f"cv_pct={cv} exceeds cv_warn_pct={cv_warn} — within "
                    f"release tolerance but worth investigating"
                ),
            })

    # M-6: tool-call score baseline.
    tc_min = methodology.get("tolerances", {}).get("tool_call_min_score", 0)
    raw_score = artefact.get("tool_call_score")
    if raw_score is not None:
        parsed_ok = False
        if isinstance(raw_score, str) and "/" in raw_score:
            try:
                numer = int(raw_score.split("/")[0])
                parsed_ok = True
                if numer < tc_min:
                    issues.append({
                        "rule": "M-6",
                        "severity": "error",
                        "message": (
                            f"tool_call_score {raw_score} below contract minimum "
                            f"{tc_min}/10"
                        ),
                    })
            except ValueError:
                pass
        if not parsed_ok:
            issues.append({
                "rule": "M-6",
                "severity": "warning",
                "message": f"tool_call_score {raw_score!r} not parseable as 'N/10'",
            })

    return issues


def run_validate(opts: argparse.Namespace) -> int:
    artefact_path = Path(opts.artefact)
    mpath = Path(opts.methodology) if opts.methodology else _DEFAULT_METHODOLOGY_PATH

    if not artefact_path.is_file():
        msg = f"artefact not found: {artefact_path}"
        if opts.json:
            print(json.dumps({"error": msg}, indent=2))
        else:
            print(f"sndr bench validate: {msg}")
        return 2

    try:
        artefact = json.loads(artefact_path.read_text(encoding="utf-8"))
        methodology = load_methodology(mpath)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        msg = f"{type(e).__name__}: {e}"
        if opts.json:
            print(json.dumps({"error": msg}, indent=2))
        else:
            print(f"sndr bench validate: {msg}")
        return 2

    issues = _validate_artefact(artefact, methodology, mpath)
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    passed = not errors

    if opts.json:
        payload = {
            "artefact": str(artefact_path),
            "methodology": str(mpath),
            "methodology_sha": methodology_sha(mpath),
            "issues": issues,
            "errors": len(errors),
            "warnings": len(warnings),
            "passed": passed,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if passed else 1

    print(f"sndr bench validate {artefact_path}")
    print(f"  methodology: {mpath} (sha={methodology_sha(mpath)[:16]}...)")
    print(f"  errors:      {len(errors)}")
    print(f"  warnings:    {len(warnings)}")
    print()
    if issues:
        for sev in ("error", "warning"):
            rows = [i for i in issues if i["severity"] == sev]
            if not rows:
                continue
            sym = "✗" if sev == "error" else "⚠"
            print(f"  {sym} {sev.upper()} ({len(rows)}):")
            for i in rows:
                print(f"    [{i['rule']}] {i['message']}")
            print()
    if passed:
        print("  ✓ artefact passes methodology contract")
    else:
        print(f"  ✗ artefact FAILED methodology contract ({len(errors)} errors)")
    return 0 if passed else 1


# ─── methodology ───────────────────────────────────────────────────────


def run_methodology(opts: argparse.Namespace) -> int:
    mpath = Path(opts.methodology) if opts.methodology else _DEFAULT_METHODOLOGY_PATH
    try:
        methodology = load_methodology(mpath)
    except (FileNotFoundError, ValueError) as e:
        print(f"sndr bench methodology: {e}")
        return 2

    sha = methodology_sha(mpath)

    if opts.json:
        payload = dict(methodology)
        payload["_path"] = str(mpath)
        payload["_sha"] = sha
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"sndr bench methodology — {mpath}")
    print(f"  sha:               {sha}")
    print(f"  methodology_id:    {methodology.get('methodology_id')}")
    print(f"  schema_version:    {methodology.get('schema_version')}")
    m = methodology.get("measurement", {})
    print()
    print("  Measurement protocol:")
    print(f"    warmup_runs:     {m.get('warmup_runs')}")
    print(f"    measure_runs:    {m.get('measure_runs')}")
    print(f"    cv_warn_pct:     {m.get('cv_warn_pct')}")
    print(f"    cv_fail_pct:     {m.get('cv_fail_pct')}")
    print()
    t = methodology.get("tolerances", {})
    print("  Tolerances:")
    for k, v in sorted(t.items()):
        print(f"    {k}: {v}")
    print()
    required = methodology.get("required_artefact_fields", [])
    print(f"  Mandatory artefact fields ({len(required)}):")
    for f in required:
        print(f"    • {f}")
    return 0

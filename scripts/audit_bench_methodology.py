#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""§6.8 + §5 — `make audit-bench-methodology` — stale-bench detector.

Genesis bench methodology contract lives at
`sndr/extras/tools/bench_methodology.yaml` (canonical, v12 tree;
post-Wave-10 self-containment). A legacy operator-side mirror at
repo-root `tools/bench_methodology.yaml` is supported as a fallback
when the canonical file is missing.

Every bench artefact under `evidence/patch_proof/*.json` carries a
`bench_delta.methodology_sha` fingerprint of the methodology that was
in force when the bench was measured.

If the contract changes (warmup count bumped, CV tolerance tightened,
prompt corpus updated), every existing bench artefact's `methodology_sha`
becomes stale — its claims no longer apply to the current methodology.
Release-gate consumers (`sndr patches release-check`) should NOT trust
such artefacts at face value.

This gate enforces the invariant: every bench artefact's
`methodology_sha` must equal the current
`sha256(sndr/extras/tools/bench_methodology.yaml)`.

Modes:

  python3 scripts/audit_bench_methodology.py         # human report
  python3 scripts/audit_bench_methodology.py --json  # CI-readable
  python3 scripts/audit_bench_methodology.py --no-bench-allow-empty
      # Without this flag, an empty `evidence/patch_proof/` is OK
      # (vacuously passes — happy default since GPU bench is operator-gated).
      # Pass to force-fail on empty directory.

Exit codes:
  0 — every bench artefact's methodology_sha matches the current contract
  1 — at least one stale or missing methodology_sha
  2 — internal error (contract file missing, etc.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
# Canonical location: package-local under sndr/extras/tools/ (v12 tree;
# post-Wave-10 self-containment lived at vllm/sndr_core/tools/).
# Legacy operator-side mirror at repo-root tools/ is supported as fallback.
_CANONICAL_METHODOLOGY = (
    REPO_ROOT / "sndr" / "extras" / "tools" / "bench_methodology.yaml"
)
_LEGACY_METHODOLOGY = REPO_ROOT / "tools" / "bench_methodology.yaml"


def _resolve_methodology_file() -> Path:
    if _CANONICAL_METHODOLOGY.is_file():
        return _CANONICAL_METHODOLOGY
    if _LEGACY_METHODOLOGY.is_file():
        return _LEGACY_METHODOLOGY
    return _CANONICAL_METHODOLOGY


METHODOLOGY_FILE = _resolve_methodology_file()
PROOF_DIR = REPO_ROOT / "evidence" / "patch_proof"


@dataclass
class ArtefactCheck:
    path: Path
    patch_id: str
    has_bench_delta: bool
    methodology_sha: Optional[str]
    canonical_sha: str
    status: str   # "match" | "stale" | "missing_sha" | "no_bench_delta" | "error"
    error: str = ""

    @property
    def passed(self) -> bool:
        # Three states pass: full match, OR no bench_delta yet (static-only
        # artefact — methodology not relevant), OR missing_sha (we warn but
        # don't block; operator may have ingested via an older bench-suite
        # that didn't yet stamp methodology_sha).
        return self.status in ("match", "no_bench_delta")


# ─── Helpers ──────────────────────────────────────────────────────────


def _canonical_methodology_sha(methodology_file: Path = METHODOLOGY_FILE) -> str:
    if not methodology_file.is_file():
        raise FileNotFoundError(
            f"bench methodology contract missing: {methodology_file}"
        )
    digest = hashlib.sha256(methodology_file.read_bytes()).hexdigest()
    return digest


def _audit_one_artefact(path: Path, canonical_sha: str) -> ArtefactCheck:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return ArtefactCheck(
            path=path, patch_id="?",
            has_bench_delta=False,
            methodology_sha=None,
            canonical_sha=canonical_sha,
            status="error",
            error=f"could not load artefact: {e}",
        )

    patch_id = data.get("patch_id", path.stem.split("__")[0])
    bench_delta = data.get("bench_delta")
    if not isinstance(bench_delta, dict) or not bench_delta:
        return ArtefactCheck(
            path=path, patch_id=patch_id,
            has_bench_delta=False,
            methodology_sha=None,
            canonical_sha=canonical_sha,
            status="no_bench_delta",
        )

    msha = bench_delta.get("methodology_sha")
    if msha is None or msha == "":
        return ArtefactCheck(
            path=path, patch_id=patch_id,
            has_bench_delta=True,
            methodology_sha=None,
            canonical_sha=canonical_sha,
            status="missing_sha",
        )

    if str(msha) == canonical_sha:
        return ArtefactCheck(
            path=path, patch_id=patch_id,
            has_bench_delta=True,
            methodology_sha=str(msha),
            canonical_sha=canonical_sha,
            status="match",
        )

    return ArtefactCheck(
        path=path, patch_id=patch_id,
        has_bench_delta=True,
        methodology_sha=str(msha),
        canonical_sha=canonical_sha,
        status="stale",
    )


def audit_bench_methodology(
    *,
    proof_dir: Path = PROOF_DIR,
    methodology_file: Path = METHODOLOGY_FILE,
) -> tuple[str, list[ArtefactCheck]]:
    canonical_sha = _canonical_methodology_sha(methodology_file)
    if not proof_dir.is_dir():
        return canonical_sha, []
    artefacts = sorted(proof_dir.glob("*.json"))
    return canonical_sha, [
        _audit_one_artefact(p, canonical_sha) for p in artefacts
    ]


# ─── Renderers ────────────────────────────────────────────────────────


def _render_text(
    canonical_sha: str, results: list[ArtefactCheck], *, allow_empty: bool,
) -> tuple[str, bool]:
    lines = []
    lines.append("audit-bench-methodology — stale-bench detector")
    lines.append("─" * 70)
    lines.append(f"  canonical SHA: {canonical_sha[:16]}…")
    lines.append(f"  artefacts:     {len(results)}")
    lines.append("")

    if not results:
        if allow_empty:
            lines.append(
                "  · evidence/patch_proof/ is empty — "
                "vacuously passing (operator-gated GPU bench not run yet)"
            )
            return "\n".join(lines), True
        lines.append(
            "  ✗ evidence/patch_proof/ is empty — "
            "no bench artefacts to verify"
        )
        return "\n".join(lines), False

    stale = [r for r in results if r.status == "stale"]
    missing = [r for r in results if r.status == "missing_sha"]
    errors = [r for r in results if r.status == "error"]
    matches = [r for r in results if r.status == "match"]
    no_bench = [r for r in results if r.status == "no_bench_delta"]

    for r in results:
        sym = "✓" if r.passed else "✗"
        lines.append(f"  {sym} {r.patch_id:8s} [{r.status:13s}] {r.path.name}")
        if r.status == "stale":
            lines.append(
                f"      got={r.methodology_sha[:16]}… "
                f"want={canonical_sha[:16]}…"
            )
        elif r.status == "error":
            lines.append(f"      {r.error}")

    lines.append("─" * 70)
    lines.append(
        f"  match={len(matches)}  "
        f"no_bench={len(no_bench)}  "
        f"stale={len(stale)}  "
        f"missing_sha={len(missing)}  "
        f"error={len(errors)}"
    )
    passed = all(r.passed for r in results)
    if not passed:
        lines.append("")
        lines.append(
            "  ✗ Fix: re-run the bench against the current methodology "
            "and re-ingest via `sndr patches bench-attach` (E19)."
        )
    return "\n".join(lines), passed


def _render_json(
    canonical_sha: str, results: list[ArtefactCheck], *, allow_empty: bool,
) -> tuple[str, bool]:
    if not results and allow_empty:
        payload = {
            "canonical_sha": canonical_sha,
            "total_artefacts": 0,
            "by_status": {"empty_allowed": True},
            "passed": True,
            "release_blocked": False,
            "artefacts": [],
        }
        return json.dumps(payload, indent=2, sort_keys=True), True

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    # `(results or allow_empty)` short-circuits to `results` (the list of
    # ArtefactCheck dataclasses) when results is non-empty, which then
    # leaks into the JSON payload via the `and` and crashes json.dumps
    # with "Object of type ArtefactCheck is not JSON serializable". Force
    # the right operand to bool so `passed` is always a bool.
    passed = all(r.passed for r in results) and bool(results or allow_empty)
    payload = {
        "canonical_sha": canonical_sha,
        "total_artefacts": len(results),
        "by_status": by_status,
        "passed": passed,
        "release_blocked": not passed,
        "artefacts": [
            {
                "patch_id": r.patch_id,
                "path": str(r.path.relative_to(REPO_ROOT))
                    if r.path.is_absolute() and REPO_ROOT in r.path.parents
                    else str(r.path),
                "status": r.status,
                "methodology_sha": r.methodology_sha,
                "passed": r.passed,
                "error": r.error or None,
            }
            for r in results
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True), passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true",
                    help="Machine-readable JSON output.")
    ap.add_argument("--proof-dir", default=None,
                    help="Override evidence/patch_proof directory.")
    ap.add_argument("--methodology", default=None,
                    help="Override methodology YAML path (default: "
                         "sndr/extras/tools/bench_methodology.yaml "
                         "with fallback to tools/bench_methodology.yaml).")
    ap.add_argument("--no-bench-allow-empty",
                    dest="allow_empty", action="store_false",
                    default=True,
                    help="Force-fail when proof_dir is empty (default: pass).")
    args = ap.parse_args()

    proof_dir = Path(args.proof_dir) if args.proof_dir else PROOF_DIR
    methodology = Path(args.methodology) if args.methodology else METHODOLOGY_FILE

    try:
        canonical_sha, results = audit_bench_methodology(
            proof_dir=proof_dir,
            methodology_file=methodology,
        )
    except FileNotFoundError as e:
        sys.stderr.write(f"audit-bench-methodology: {e}\n")
        return 2

    if args.json:
        out, ok = _render_json(canonical_sha, results,
                               allow_empty=args.allow_empty)
    else:
        out, ok = _render_text(canonical_sha, results,
                               allow_empty=args.allow_empty)
    print(out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

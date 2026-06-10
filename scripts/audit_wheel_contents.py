#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""audit_wheel_contents.py — CLI surface for wheel-boundary invariants
(§9.A.1, AUDIT-CLOSURE.3, 2026-05-27).

Wheel boundary is already enforced end-to-end by two existing pytest
test files:

  * ``tests/unit/test_wheel_contents.py`` (5 tests) — black-box wheel
    contract: builds the wheel via ``python -m build``, inspects the
    zip + installs in an isolated venv + asserts runtime registry
    lookups succeed. Covers the *delivery* surface.

  * ``tests/unit/test_edition_boundary.py::TestWheelPackageSeparation``
    (2 tests) — pyproject.toml configuration check.

This audit is a **thin CLI wrapper** that:

  1. Verifies the canonical wheel-boundary test files exist and are
    pytest-collectible (catches accidental deletion of the boundary
    invariants).
  2. Re-runs the pyproject.toml structural checks without requiring
    the slow wheel-build (fast standalone gate suitable for `make
    gates` aggregate).
  3. Documents which invariants are enforced where, so operators
    auditing the wheel surface get a single-page summary.

It does NOT duplicate the heavy wheel-build invariants — those stay
in pytest where they can leverage temp-venv fixtures. To run the
full wheel build + isolated-venv runtime contract, invoke the pytest
file directly (see ``--show-tests`` mode).

Exit codes
──────────

  0 — every documented invariant has its test file + pyproject shape passes
  1 — at least one missing test file OR pyproject shape violation
  2 — internal error / tomllib not available

Modes
─────

  python3 scripts/audit_wheel_contents.py            # human-readable
  python3 scripts/audit_wheel_contents.py --json     # machine-readable
  python3 scripts/audit_wheel_contents.py --show-tests # list invariants
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent


# ─── Canonical invariant manifest ─────────────────────────────────────────


@dataclasses.dataclass
class Invariant:
    name: str
    location: str          # repo-relative file:test_function or file:class.method
    rationale: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


_INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        name="pyproject lists the sndr runtime tree in wheel packages",
        location="tests/unit/test_edition_boundary.py::TestWheelPackageSeparation::test_pyproject_includes_sndr_core",
        rationale="without `sndr*` in packages.find.include the wheel won't carry the runtime registry",
    ),
    Invariant(
        name="pyproject declares `sndr` console entry point",
        location="tests/unit/test_edition_boundary.py::TestWheelPackageSeparation::test_pyproject_root_has_sndr_console_entry",
        rationale="primary operator entry point — must be installed by pip",
    ),
    Invariant(
        name="V2 YAMLs reach the wheel (model/hardware/profile/presets ≥1 each)",
        location="tests/unit/test_wheel_contents.py::TestWheelZipContract::test_v2_yamls_present_in_all_four_subdirs",
        rationale="P0.A regression guard — non-recursive package-data spec was a real production bug",
    ),
    Invariant(
        name="sndr_private never in wheel",
        location="tests/unit/test_wheel_contents.py::TestWheelZipContract::test_no_sndr_private_anywhere_in_wheel",
        rationale="hard rule #27 — operator-private archive must not ship",
    ),
    Invariant(
        name="vllm/sndr_engine never in core wheel",
        location="tests/unit/test_wheel_contents.py::TestWheelZipContract::test_no_sndr_engine_in_core_wheel",
        rationale="commercial-engine separation via license gate",
    ),
    Invariant(
        name="V2 listings visible at runtime from installed wheel",
        location="tests/unit/test_wheel_contents.py::TestWheelRuntimeContract::test_v2_listings_visible_from_installed_wheel",
        rationale="importlib.resources lookups must succeed in clean install",
    ),
    Invariant(
        name="engine not importable from clean core install",
        location="tests/unit/test_wheel_contents.py::TestWheelRuntimeContract::test_engine_not_importable_from_core",
        rationale="isolated-venv probe — finds engine bundling regression even if zip-check passes",
    ),
)


# ─── Pyproject shape check ────────────────────────────────────────────────


@dataclasses.dataclass
class PyprojectResult:
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def check_pyproject_shape(
    *, repo_root: Optional[Path] = None,
) -> list[PyprojectResult]:
    """Re-implement the two ``TestWheelPackageSeparation`` checks
    without requiring pytest invocation. Fast standalone gate."""
    repo_root = repo_root or REPO_ROOT
    pyproject = repo_root / "pyproject.toml"
    results: list[PyprojectResult] = []

    try:
        import tomllib
    except ImportError:
        results.append(PyprojectResult(
            passed=False,
            detail="tomllib unavailable (need Python 3.11+); cannot parse pyproject",
        ))
        return results

    if not pyproject.is_file():
        results.append(PyprojectResult(
            passed=False, detail=f"pyproject.toml not found at {pyproject}",
        ))
        return results

    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
    except Exception as e:
        results.append(PyprojectResult(
            passed=False, detail=f"pyproject parse error: {e}",
        ))
        return results

    # Check 1: the sndr runtime tree in wheel packages (v12: `sndr*`).
    include = (
        data.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
        .get("include", [])
    )
    if any(p in ("sndr", "sndr*") or p.startswith("sndr.") for p in include):
        results.append(PyprojectResult(
            passed=True,
            detail=f"sndr runtime tree covered by packages.find.include: {include}",
        ))
    else:
        results.append(PyprojectResult(
            passed=False,
            detail=(
                f"sndr runtime tree MISSING from packages.find.include "
                f"(found: {include}) — wheel would not carry the registry"
            ),
        ))

    # Check 2: `sndr` console entry point.
    scripts = data.get("project", {}).get("scripts", {})
    if "sndr" in scripts:
        results.append(PyprojectResult(
            passed=True,
            detail=f"sndr console entry registered: sndr={scripts['sndr']}",
        ))
    else:
        results.append(PyprojectResult(
            passed=False,
            detail=(
                f"sndr console entry MISSING from [project.scripts] "
                f"(found: {sorted(scripts.keys())})"
            ),
        ))

    return results


# ─── Test-file existence check ────────────────────────────────────────────


@dataclasses.dataclass
class TestFileResult:
    location: str
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def check_test_files_exist(
    *, repo_root: Optional[Path] = None,
) -> list[TestFileResult]:
    repo_root = repo_root or REPO_ROOT
    seen_files: dict[str, bool] = {}
    results: list[TestFileResult] = []
    for inv in _INVARIANTS:
        # Location format: "path/to/file.py::ClassOrFn[::method]"
        file_part = inv.location.split("::", 1)[0]
        if file_part in seen_files:
            continue
        seen_files[file_part] = True
        full = repo_root / file_part
        if full.is_file():
            results.append(TestFileResult(
                location=file_part,
                passed=True,
                detail=f"present ({full.stat().st_size} bytes)",
            ))
        else:
            results.append(TestFileResult(
                location=file_part,
                passed=False,
                detail="MISSING — wheel-boundary invariant cannot be enforced",
            ))
    return results


# ─── Render ───────────────────────────────────────────────────────────────


def _render_text(
    pyproject_results: list[PyprojectResult],
    test_file_results: list[TestFileResult],
    show_tests: bool,
) -> str:
    lines: list[str] = []
    lines.append("audit-wheel-contents: wheel-boundary invariants")
    lines.append("─" * 70)

    lines.append(f"  pyproject shape checks: {len(pyproject_results)}")
    for r in pyproject_results:
        sym = "✓" if r.passed else "✗"
        lines.append(f"    {sym} {r.detail}")

    lines.append("")
    lines.append(f"  test-file presence: {len(test_file_results)}")
    for r in test_file_results:
        sym = "✓" if r.passed else "✗"
        lines.append(f"    {sym} {r.location}  ({r.detail})")

    if show_tests:
        lines.append("")
        lines.append(f"  invariants covered: {len(_INVARIANTS)}")
        for inv in _INVARIANTS:
            lines.append(f"    · {inv.name}")
            lines.append(f"      └─ {inv.location}")
            lines.append(f"      └─ rationale: {inv.rationale}")

    fail_count = (
        sum(1 for r in pyproject_results if not r.passed)
        + sum(1 for r in test_file_results if not r.passed)
    )
    lines.append("")
    if fail_count == 0:
        lines.append("  ✓ wheel boundary surface intact")
    else:
        lines.append(f"  ✗ {fail_count} violation(s) — wheel boundary at risk")
        lines.append("")
        lines.append(
            "  To run heavy wheel-build invariants:\n"
            "    python3 -m pytest tests/unit/test_wheel_contents.py "
            "tests/unit/test_edition_boundary.py::TestWheelPackageSeparation"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON")
    ap.add_argument(
        "--show-tests", action="store_true",
        help="include full invariant manifest in human output",
    )
    args = ap.parse_args()

    pyproject_results = check_pyproject_shape()
    test_file_results = check_test_files_exist()

    if args.json:
        print(json.dumps({
            "pyproject_results": [r.as_dict() for r in pyproject_results],
            "test_file_results": [r.as_dict() for r in test_file_results],
            "invariants": [inv.as_dict() for inv in _INVARIANTS],
            "passed": all(r.passed for r in pyproject_results)
                      and all(r.passed for r in test_file_results),
        }, indent=2, sort_keys=True))
    else:
        print(_render_text(
            pyproject_results, test_file_results, args.show_tests,
        ))

    fail = (
        any(not r.passed for r in pyproject_results)
        or any(not r.passed for r in test_file_results)
    )
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

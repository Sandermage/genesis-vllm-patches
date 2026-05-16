# SPDX-License-Identifier: Apache-2.0
"""Tests for constraints.txt scaffold — T1.7 / audit §P1-2.

Verifies that the constraints file at the repo root is shape-correct
(parseable, references our declared deps, contains no malformed
specifiers) and stays in sync with pyproject.toml's direct dependency
declaration.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONSTRAINTS = REPO_ROOT / "constraints.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _parse_constraints() -> list[str]:
    """Return non-comment, non-blank lines from constraints.txt."""
    if not CONSTRAINTS.is_file():
        return []
    out: list[str] = []
    for line in CONSTRAINTS.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped)
    return out


class TestConstraintsExist:
    def test_constraints_file_present(self):
        assert CONSTRAINTS.is_file(), (
            "constraints.txt must live at repo root for `pip install -c` "
            "to find it."
        )

    def test_constraints_not_empty(self):
        rows = _parse_constraints()
        assert len(rows) > 0, "constraints.txt has no specifier lines"


class TestConstraintsShape:
    def test_each_line_is_valid_specifier(self):
        """Every non-comment line must parse as a PEP 440 requirement."""
        from packaging.requirements import Requirement
        for line in _parse_constraints():
            try:
                Requirement(line)
            except Exception as e:
                pytest.fail(f"constraints.txt line {line!r}: {e}")

    def test_pinned_packages_have_upper_bound(self):
        """Avoid unbounded specifiers — they defeat the purpose."""
        for line in _parse_constraints():
            # We allow simple `>=` if it pairs with a `<` upper bound.
            if ">=" in line and "<" not in line:
                pytest.fail(
                    f"constraints.txt: {line!r} has a lower bound but no "
                    "upper bound — pin a ceiling so future breaking releases "
                    "don't silently install."
                )


class TestConstraintsCoversDirectDeps:
    """Every dependency declared in pyproject.toml [project] must appear
    in constraints.txt — otherwise the constraints file is incomplete."""

    def _read_pyproject_deps(self) -> set[str]:
        """Return canonical names (lowercased) of declared direct deps."""
        try:
            import tomllib  # py>=3.11
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        deps = data.get("project", {}).get("dependencies", []) or []
        names: set[str] = set()
        from packaging.requirements import Requirement
        for spec in deps:
            try:
                req = Requirement(spec)
                names.add(req.name.lower())
            except Exception:
                pass
        return names

    def _constraints_names(self) -> set[str]:
        from packaging.requirements import Requirement
        out: set[str] = set()
        for line in _parse_constraints():
            try:
                req = Requirement(line)
                out.add(req.name.lower())
            except Exception:
                pass
        return out

    def test_every_direct_dep_appears_in_constraints(self):
        direct = self._read_pyproject_deps()
        constrained = self._constraints_names()
        missing = direct - constrained
        assert not missing, (
            f"constraints.txt is missing direct deps: {sorted(missing)}"
        )


def _find_clean_venv_script():
    """Locate the clean-venv smoke-test script wherever it lives in
    this checkout. It is a maintainer convenience tool — public clones
    may not ship it, in which case the tests below skip."""
    candidates = [REPO_ROOT / "scripts" / "run_clean_venv_test.sh"]
    candidates.extend(REPO_ROOT.glob("*/scripts/run_clean_venv_test.sh"))
    for c in candidates:
        if c.is_file():
            return c
    return None


class TestCleanVenvScript:
    """Ensure the clean-venv smoke-test script is shipped + executable
    when this checkout carries it."""

    def test_script_exists(self):
        path = _find_clean_venv_script()
        if path is None:
            pytest.skip("clean-venv smoke script not present in this checkout")
        assert path.is_file()

    def test_script_is_executable(self):
        import stat
        path = _find_clean_venv_script()
        if path is None:
            pytest.skip("clean-venv smoke script not present in this checkout")
        mode = path.stat().st_mode
        assert mode & stat.S_IXUSR, "script must be marked executable"

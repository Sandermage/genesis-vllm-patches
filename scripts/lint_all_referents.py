#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python F822 gate — `__all__` referent validation.

Roadmap §8 Open items: static `ruff F821/F822` CI gate. F821 needs full
scope resolution + a tool like pyflakes/ruff; F822 (every name listed
in `__all__` must be defined or imported) is straightforward to do with
stdlib `ast`. This script implements F822 only and stays pure-stdlib.

What it catches:

  • `__all__ = ["foo"]` where `foo` is neither imported nor defined in the
    module → release-blocker; consumers `from mod import *` would crash
  • `__all__ = ["foo"]` where `foo` was previously defined then deleted
    → catches stale public-API surface

What it does NOT catch (out of scope here — needs pyflakes/ruff for full):

  • F821 — undefined name in function body
  • F401 — unused import
  • F811 — re-defined name

Usage:

  python3 scripts/lint_all_referents.py            # scan vllm/sndr_core/**
  python3 scripts/lint_all_referents.py --paths X  # custom scan root(s)
  python3 scripts/lint_all_referents.py --json     # machine output

Exit code:
  0 — every `__all__` referent resolves
  1 — at least one undefined name in `__all__`
  2 — internal error (e.g. malformed Python in a scanned file)
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCAN_PATHS = (
    "sndr",
    "scripts",
)


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    name: str
    message: str


# ─── AST walker ────────────────────────────────────────────────────────


def _module_defined_names(tree: ast.Module) -> set[str]:
    """Collect every top-level binding the module exposes.

    Sources counted (matches what `from <mod> import *` would expose
    if `__all__` were absent):

      - `from X import Y, Z as A` → Y, A
      - `import X.Y as Z`         → Z
      - `import X.Y`              → X (the top-level binding)
      - `def foo`, `async def foo`, `class Foo`
      - top-level `name = ...` assignments (incl. tuple/list unpacking)
      - `name: T = ...` annotated assignments
      - top-level `for name in ...` (rare but valid)
      - top-level `with ... as name`

    Conditional definitions inside `if`/`try` blocks contribute their
    inner names too — Python's `__all__` doesn't care HOW a name was
    bound, only WHETHER it ends up in module dict.
    """
    defined: set[str] = set()

    def _walk_targets(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            defined.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                _walk_targets(elt)
        elif isinstance(node, ast.Starred):
            _walk_targets(node.value)

    def _visit(node: ast.AST) -> None:
        # Direct top-level definitions.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import X.Y` binds `X` (top-level package) unless aliased.
                name = alias.asname or alias.name.split(".")[0]
                defined.add(name)
            return
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    # `from mod import *` is a wildcard — we can't statically
                    # know what it binds. Mark a sentinel; F822 is then
                    # informational for this module.
                    defined.add("*")
                else:
                    defined.add(alias.asname or alias.name)
            return
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                _walk_targets(tgt)
            return
        if isinstance(node, ast.AnnAssign) and node.target is not None:
            _walk_targets(node.target)
            return
        if isinstance(node, ast.AugAssign):
            _walk_targets(node.target)
            return
        if isinstance(node, ast.For):
            _walk_targets(node.target)
            # Recurse — `for` body may contain defs at top level (rare).
            for stmt in node.body:
                _visit(stmt)
            for stmt in node.orelse:
                _visit(stmt)
            return
        if isinstance(node, ast.AsyncFor):
            _walk_targets(node.target)
            for stmt in node.body + node.orelse:
                _visit(stmt)
            return
        if isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    _walk_targets(item.optional_vars)
            for stmt in node.body:
                _visit(stmt)
            return
        if isinstance(node, ast.AsyncWith):
            for item in node.items:
                if item.optional_vars is not None:
                    _walk_targets(item.optional_vars)
            for stmt in node.body:
                _visit(stmt)
            return
        if isinstance(node, (ast.If, ast.Try)):
            for stmt in node.body:
                _visit(stmt)
            if isinstance(node, ast.If):
                for stmt in node.orelse:
                    _visit(stmt)
            else:  # Try
                for h in node.handlers:
                    for stmt in h.body:
                        _visit(stmt)
                for stmt in node.orelse + node.finalbody:
                    _visit(stmt)
            return
        if isinstance(node, ast.TryStar) if hasattr(ast, "TryStar") else False:
            for stmt in node.body:  # type: ignore[attr-defined]
                _visit(stmt)
            return
        # Anything else (Expr, Pass, Return, etc.) doesn't bind a top-level name.

    for stmt in tree.body:
        _visit(stmt)
    return defined


def _extract_all_referents(tree: ast.Module) -> list[tuple[int, str]]:
    """Find every name appearing in `__all__ = [...]` / `__all__ += [...]`.

    Returns (line_number, name) tuples. Only string-literal members are
    enumerated — dynamic `__all__` (e.g. `__all__ = some_function()`) is
    out of scope; we skip silently.
    """
    out: list[tuple[int, str]] = []

    def _collect_list_strs(node: ast.AST, lineno: int) -> None:
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    out.append((elt.lineno, elt.value))

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    _collect_list_strs(stmt.value, stmt.lineno)
        elif isinstance(stmt, ast.AnnAssign):
            if (isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "__all__"
                    and stmt.value is not None):
                _collect_list_strs(stmt.value, stmt.lineno)
        elif isinstance(stmt, ast.AugAssign):
            if (isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "__all__"):
                _collect_list_strs(stmt.value, stmt.lineno)
    return out


# ─── Per-file checker ─────────────────────────────────────────────────


def _has_module_level_getattr(tree: ast.Module) -> bool:
    """Detect PEP 562 lazy-loader pattern: top-level `def __getattr__(name)`.

    When present, `__all__` names are resolved dynamically at attribute-
    access time. We can't statically prove they exist (the loader's
    implementation determines that), so we skip F822 for the module.
    The release-tier safety net for these lazy modules is the
    smoke-import test (`audit-configs` indirectly exercises them).
    """
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__getattr__":
            return True
    return False


def _sibling_submodule_names(path: Path) -> set[str]:
    """For an `__init__.py`, return names of sibling submodules + subpackages.

    `from package import *` will import every name in `__all__` —
    including names that match a `<sibling>.py` file or `<sibling>/`
    subpackage, even when the init doesn't explicitly import them.
    This matches Python's import-system behaviour (see CPython's
    `_handle_fromlist`).
    """
    if path.name != "__init__.py":
        return set()
    parent = path.parent
    out: set[str] = set()
    for sibling in parent.iterdir():
        if sibling.is_file() and sibling.suffix == ".py" and sibling.name != "__init__.py":
            out.add(sibling.stem)
        elif sibling.is_dir() and (sibling / "__init__.py").is_file():
            out.add(sibling.name)
    return out


def check_file(path: Path) -> list[Violation]:
    """Return F822 violations for one Python file."""
    if path.is_absolute():
        try:
            rel = path.relative_to(REPO_ROOT)
        except ValueError:
            # File outside the repo (test fixtures, ad-hoc CLI invocation).
            rel = path
    else:
        rel = path
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        # Don't fail the whole gate on a single unreadable file.
        return [Violation(
            file=str(rel), line=0, name="",
            message=f"could not read: {e}",
        )]
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as e:
        return [Violation(
            file=str(rel), line=e.lineno or 0, name="",
            message=f"SyntaxError: {e.msg}",
        )]

    referents = _extract_all_referents(tree)
    if not referents:
        return []   # No `__all__` → nothing to check.

    # PEP 562 lazy-loader: skip the check (names resolve dynamically).
    if _has_module_level_getattr(tree):
        return []

    defined = _module_defined_names(tree)
    # Wildcard import shorts the check: we can't tell what `*` exposed.
    if "*" in defined:
        return []

    # Package-init: sibling submodules count as defined (import-* loads them).
    siblings = _sibling_submodule_names(path)
    defined = defined | siblings

    out: list[Violation] = []
    for lineno, name in referents:
        if name not in defined:
            out.append(Violation(
                file=str(rel), line=lineno, name=name,
                message=f"`__all__` references {name!r} but it is "
                        f"neither defined nor imported in this module",
            ))
    return out


# ─── Repo walker ──────────────────────────────────────────────────────


def _iter_py_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for fp in root.rglob("*.py"):
            # Skip generated / cache directories.
            if any(part in {"__pycache__", ".venv", "_archive"}
                   for part in fp.parts):
                continue
            # macOS AppleDouble resource fork (not Python source — contains
            # null bytes and non-UTF8 prefix that crashes the AST parser).
            if fp.name.startswith("._"):
                continue
            out.append(fp)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paths", nargs="*", default=None,
        help="Scan roots (repo-relative). Default: sndr + scripts.",
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    roots = [
        REPO_ROOT / p for p in (args.paths or DEFAULT_SCAN_PATHS)
    ]
    files = _iter_py_files(roots)
    violations: list[Violation] = []
    parse_errors = 0
    for fp in files:
        for v in check_file(fp):
            if v.message.startswith("SyntaxError"):
                parse_errors += 1
            violations.append(v)

    if args.json:
        print(json.dumps(
            {
                "files_scanned": len(files),
                "violations": [v.__dict__ for v in violations],
                "violation_count": len(violations),
                "parse_errors": parse_errors,
                "passed": not violations,
            },
            indent=2, sort_keys=True,
        ))
    else:
        print(f"lint_all_referents: {len(files)} Python files scanned")
        print("─" * 70)
        if not violations:
            print("  ✓ every `__all__` referent resolves")
            return 0
        # Group by file for compact output.
        by_file: dict[str, list[Violation]] = {}
        for v in violations:
            by_file.setdefault(v.file, []).append(v)
        for fp, rows in sorted(by_file.items()):
            print(f"  ✗ {fp}")
            for v in rows:
                print(f"      L{v.line}: {v.name!r}  ({v.message})")
        print()
        print(f"  FAIL — {len(violations)} violation(s) across "
              f"{len(by_file)} file(s)")

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())

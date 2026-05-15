#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CI gate: forbid legacy imports in active code.

Etap 5.4 / 5.5 (audit 2026-05-12): AST-based replacement for the
previous shell `grep` gate. Catches every import shape we care about:

  • `import vllm.sndr_core.patches.foo`
  • `import vllm.sndr_core.patches.foo as bar`
  • `from vllm.sndr_core.patches.foo import x`
  • `from vllm.sndr_core import patches`        (was missed by old regex)
  • `from vllm.sndr_core.patches import x`      (was missed by old regex)
  • `vllm._genesis.<anything>` in any of the above forms.

Also greps non-Python files (YAML/TOML/JSON/workflow .yml) for the
same legacy strings — those slipped past the shell-only globs.

What we catch:
  1. `vllm.sndr_core.patches.*` — pre-v10 namespace renamed to
     `vllm.sndr_core.integrations` (PROJECT_STATE_AUDIT P0-1).
  2. `vllm._genesis.*` — pre-v11 namespace removed; allowed only in
     historical docs and explicit back-compat aliases (allowlist below).

Exit codes:
  0 — clean.
  1 — violations present (script prints file:line — fix and rerun).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SEARCH_DIRS = ("tests", "vllm/sndr_core", "scripts", "tools")
EXCLUDE_DIR_NAMES = {
    "__pycache__", ".git", ".ruff_cache", ".mypy_cache", ".pytest_cache",
}

# File suffixes to scan.
PY_SUFFIXES = (".py",)
TEXT_SUFFIXES = (".sh", ".md", ".yml", ".yaml", ".toml", ".json")

# Legacy roots — `import` checks (AST + grep) hit on either.
LEGACY_ROOTS = (
    "vllm.sndr_core.patches",
    "vllm._genesis",
)

# Path substrings where legacy refs are accepted (back-compat aliases,
# historical provenance, schema descriptions, the gate itself, archives).
ALLOWLIST_SUBSTRINGS = (
    "vllm/sndr_core/__init__.py",
    "vllm/sndr_core/schemas/patch_entry.schema.json",
    "vllm/sndr_core/compat/categories.py",
    "vllm/sndr_core/version.py",
    "vllm/sndr_core/locations/project_paths.py",
    "vllm/sndr_core/integrations/upstream_compat.py",
    "vllm/sndr_core/apply/",
    "vllm/sndr_core/compat/migrate.py",
    "tools/genesis_vllm_plugin/README.md",
    "scripts/check_no_legacy_imports",
    # Historical archives — intentional preservation of the v7/v8 layout.
    "scripts/_archive/",
    "scripts/launch/_archive/",
    # Frozen baselines from the v8 test era; values reference the
    # legacy module path verbatim and must not be rewritten.
    "tests/integration/baselines/",
)


def _is_allowlisted(path: Path) -> bool:
    rel = str(path.relative_to(REPO_ROOT))
    return any(s in rel for s in ALLOWLIST_SUBSTRINGS)


def _iter_files() -> list[Path]:
    out: list[Path] = []
    for root in SEARCH_DIRS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for f in base.rglob("*"):
            if not f.is_file():
                continue
            if any(part in EXCLUDE_DIR_NAMES for part in f.parts):
                continue
            if f.suffix in PY_SUFFIXES + TEXT_SUFFIXES:
                out.append(f)
    return out


def _check_python_imports(path: Path) -> list[tuple[int, str]]:
    """AST-walk the file; report `(lineno, message)` for each legacy
    import shape. Tolerates files with syntax errors (skip silently —
    that's a separate concern, not this gate's job)."""
    violations: list[tuple[int, str]] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return violations
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name or ""
                for root in LEGACY_ROOTS:
                    if name == root or name.startswith(root + "."):
                        violations.append((
                            node.lineno,
                            f"import {name}",
                        ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for root in LEGACY_ROOTS:
                if module == root or module.startswith(root + "."):
                    names = ", ".join(a.name for a in node.names)
                    violations.append((
                        node.lineno,
                        f"from {module} import {names}",
                    ))
                    break
            else:
                # `from vllm.sndr_core import patches` style — catch by
                # looking at imported names when the prefix is the
                # parent of a legacy root.
                if module == "vllm.sndr_core":
                    for alias in node.names:
                        if alias.name == "patches":
                            violations.append((
                                node.lineno,
                                "from vllm.sndr_core import patches",
                            ))
                elif module == "vllm":
                    for alias in node.names:
                        if alias.name == "_genesis":
                            violations.append((
                                node.lineno,
                                "from vllm import _genesis",
                            ))
    return violations


# For non-Python files we accept that we don't have full grammar —
# just look for the legacy strings outright. False-positives on text
# documentation are handled by the allowlist.
_TEXT_RE = re.compile(
    r"\b(?:vllm\.sndr_core\.patches|vllm\._genesis)\b"
)


def _check_text(path: Path) -> list[tuple[int, str]]:
    violations: list[tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return violations
    for idx, line in enumerate(text.splitlines(), start=1):
        if _TEXT_RE.search(line):
            violations.append((idx, line.strip()))
    return violations


def main(argv: list[str] | None = None) -> int:
    files = _iter_files()
    print("=== check_no_legacy_imports.py ===")
    print(f"scanning {len(files)} files in {', '.join(SEARCH_DIRS)}")

    total = 0
    for f in files:
        if _is_allowlisted(f):
            continue
        if f.suffix in PY_SUFFIXES:
            vs = _check_python_imports(f)
        else:
            vs = _check_text(f)
        if not vs:
            continue
        rel = f.relative_to(REPO_ROOT)
        for lineno, msg in vs:
            print(f"{rel}:{lineno}: {msg}")
        total += len(vs)

    print()
    if total == 0:
        print("✓ legacy-import gate: clean")
        return 0
    print(f"✗ legacy-import gate: {total} violation(s)")
    print()
    print("Fix: rename the imports:")
    print("  vllm.sndr_core.patches.<X>  →  vllm.sndr_core.integrations.<X>")
    print("  vllm._genesis.<X>           →  vllm.sndr_core.<X>")
    print("If the reference really is historical — add the file to "
          "ALLOWLIST_SUBSTRINGS in this script.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

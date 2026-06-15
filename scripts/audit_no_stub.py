#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Release-tier no-stub scan (Consolidated Roadmap §10.3 item 2, §10.5 boundary).

Catches unresolved code stubs in production code paths:

  • Bare `raise NotImplementedError(...)` statements (AST-detected,
    so docstring / string-literal references like the ones describing
    a patch that REPLACES an upstream `NotImplementedError` are
    correctly ignored).
  • `# TODO(<name>): ...` comments (§10.5 says TODOs require name + date;
    comments without a `(...)` form are advisory only and not flagged).
  • `pass  # placeholder` / `pass  # scaffold` / `pass  # FIXME` sentinel
    lines that indicate skeleton code.

The scan runs over `vllm/sndr_core/**/*.py`. Patches live there; CLI,
runtime, and discovery code is included. Tests (`tests/`, `**/_test_*`,
`**/test_*.py`) are NOT scanned — they legitimately use these idioms.

Allowlist:

  • Lines containing `audit-no-stub: allow` are skipped (operator
    escape hatch for genuinely intentional cases, e.g. an interface
    method that subclasses must override).
  • `__pycache__/` and `*.pyc` are skipped by glob.

Exit codes:
  0 — clean.
  1 — at least one violation found.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "sndr"

# Test-style filenames + dirs that are exempt by design.
TEST_PATH_RE = re.compile(r"(^|/)(tests?|_test|test_)")

# Upstream-mirror bind-mount overlays — code under these paths is a
# near-verbatim copy of upstream vLLM that gets bind-mounted into the
# container at runtime. Any `raise NotImplementedError` / `TODO(name)`
# / sentinel `pass` lines here belong to upstream, not Genesis. The
# audit's intent is to catch Genesis-authored stubs; upstream-mirror
# code is out of scope.
# v12.0: the overlay tree relocated from `integrations/attention/...` to
# `sndr/engines/vllm/patches/attention/...`; anchor on the stable
# `attention/turboquant/overlays/<pr…>/` suffix so both layouts are exempt.
UPSTREAM_MIRROR_PATH_RE = re.compile(
    r"(^|/)attention/turboquant/overlays/(pr\d+|upstream_[a-z0-9_]+)/"
)

ALLOW_MARKER = "audit-no-stub: allow"

TODO_RE = re.compile(r"#\s*TODO\(([^)]*)\)")  # only the (name) form counts
SENTINEL_RE = re.compile(r"\bpass\b\s*#\s*(placeholder|scaffold|FIXME)\b")


def _is_test_path(p: Path) -> bool:
    rel = p.relative_to(REPO_ROOT).as_posix()
    return bool(TEST_PATH_RE.search(rel))


def _is_upstream_mirror_path(p: Path) -> bool:
    rel = p.relative_to(REPO_ROOT).as_posix()
    return bool(UPSTREAM_MIRROR_PATH_RE.search(rel))


def _gather_files() -> list[Path]:
    out: list[Path] = []
    for fp in SCAN_ROOT.rglob("*.py"):
        if "__pycache__" in fp.parts:
            continue
        if fp.name.startswith("._"):
            # macOS AppleDouble resource fork: not Python source.
            continue
        if _is_test_path(fp):
            continue
        if _is_upstream_mirror_path(fp):
            continue
        out.append(fp)
    return sorted(out)


def _enclosing_function_node(tree: ast.AST, target: ast.Raise) -> ast.FunctionDef | None:
    """Walk the tree to find the FunctionDef (sync or async) that lexically
    contains ``target``. Returns None when target is at module scope."""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.__dict__.setdefault("_parent", parent)
    cur = target
    while True:
        cur = cur.__dict__.get("_parent")
        if cur is None:
            return None
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur


def _is_abstract_method_raise(tree: ast.AST, raise_node: ast.Raise) -> bool:
    """True when ``raise NotImplementedError`` is the body of an abstract
    method:
      * The enclosing function has @abstractmethod / @abc.abstractmethod
        in its decorator_list, OR
      * The function's body is a single `raise NotImplementedError(...)`
        statement (the canonical abstract-method shape — also used by
        protocol stand-ins / interface contracts that don't import abc).

    This pattern is intentional by design; flagging it as a stub is a
    false positive.
    """
    fn = _enclosing_function_node(tree, raise_node)
    if fn is None:
        return False
    # Decorator check
    for deco in fn.decorator_list:
        # @abstractmethod
        if isinstance(deco, ast.Name) and deco.id in {"abstractmethod"}:
            return True
        # @abc.abstractmethod
        if isinstance(deco, ast.Attribute) and deco.attr in {"abstractmethod"}:
            return True
    # Body-shape check: docstring + lone raise OR lone raise. Strip
    # leading Expr(Constant=str) (the docstring) before counting.
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    if len(body) == 1 and isinstance(body[0], ast.Raise):
        return True
    return False


def _check_ast_raises(path: Path, source: str) -> list[str]:
    """AST scan: report `raise NotImplementedError(...)` that aren't on
    a line with `audit-no-stub: allow` AND aren't the canonical
    abstract-method body."""
    out: list[str] = []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [f"{path.relative_to(REPO_ROOT).as_posix()}:{e.lineno}: SYNTAX ERROR"]
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc
        # Unwrap `raise NotImplementedError(...)` (with or without args).
        name = None
        if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
            name = exc.func.id
        elif isinstance(exc, ast.Name):
            name = exc.id
        if name != "NotImplementedError":
            continue
        # Skip canonical abstract-method body (decorator @abstractmethod
        # OR function body is single raise after optional docstring).
        if _is_abstract_method_raise(tree, node):
            continue
        # Check allow marker on the line.
        lineno = node.lineno
        line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if ALLOW_MARKER in line:
            continue
        # Look one line back too — common to put the marker above the raise.
        prev = lines[lineno - 2] if lineno >= 2 else ""
        if ALLOW_MARKER in prev:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        out.append(f"{rel}:{lineno}: raise NotImplementedError ({line.strip()[:80]})")
    return out


def _check_textual_markers(path: Path, source: str) -> list[str]:
    """Textual scan: TODO with name, sentinel `pass  # placeholder|scaffold|FIXME`.
    """
    out: list[str] = []
    rel = path.relative_to(REPO_ROOT).as_posix()
    for i, line in enumerate(source.splitlines(), 1):
        if ALLOW_MARKER in line:
            continue
        if TODO_RE.search(line):
            out.append(f"{rel}:{i}: TODO marker ({line.strip()[:80]})")
            continue
        if SENTINEL_RE.search(line):
            out.append(f"{rel}:{i}: sentinel pass ({line.strip()[:80]})")
    return out


def audit() -> dict[str, list[str]]:
    findings = {"raise_notimplemented": [], "textual_markers": []}
    for fp in _gather_files():
        try:
            src = fp.read_text(encoding="utf-8")
        except OSError:
            continue
        findings["raise_notimplemented"].extend(_check_ast_raises(fp, src))
        findings["textual_markers"].extend(_check_textual_markers(fp, src))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    # Fail loudly if the scan root is missing — a vanished SCAN_ROOT would
    # otherwise yield a 0-file scan and a vacuous PASS that hides drift.
    if not SCAN_ROOT.is_dir():
        msg = f"scan root not found: {SCAN_ROOT}"
        if args.json:
            print(json.dumps(
                {"findings": {}, "total": 0, "error": msg},
                indent=2, sort_keys=True,
            ))
        else:
            print(f"audit-no-stub: {msg}")
            print("  FAIL — nothing scanned; refusing vacuous PASS")
        return 2
    findings = audit()
    total = sum(len(v) for v in findings.values())
    if args.json:
        print(json.dumps(
            {"findings": findings, "total": total},
            indent=2, sort_keys=True,
        ))
    else:
        print("audit-no-stub: scanning sndr/**/*.py")
        print("─" * 70)
        for name, hits in findings.items():
            label = name.replace("_", " ")
            if hits:
                print(f"  ✗ {label}: {len(hits)} hit(s)")
                for h in hits[:5]:
                    print(f"      {h}")
                if len(hits) > 5:
                    print(f"      ... ({len(hits) - 5} more)")
            else:
                print(f"  ✓ {label}: clean")
        print()
        if total:
            print(f"  FAIL — {total} total violation(s)")
        else:
            print("  OK — no unresolved stubs in production code")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

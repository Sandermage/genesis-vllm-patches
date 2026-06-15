#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Engine boundary check (Consolidated Roadmap §10.3 item 5).

`vllm.sndr_core/` is the public OSS layer. `vllm.sndr_engine/` is the
commercial overlay distributed under a separate license. The boundary
contract:

  • `vllm.sndr_core/**/*.py` MAY reference `vllm.sndr_engine` ONLY in
    optional-discovery helpers — `import vllm.sndr_engine` wrapped in
    `try / except ImportError` blocks that return False / None when
    the engine is absent. Tier-gate / capability probes / license
    check use this pattern.

  • Unguarded `from vllm.sndr_engine import X` or `import vllm.sndr_engine`
    at module top-level (not inside try / except) is forbidden. Such an
    import would crash the public install when sndr_engine isn't
    available, breaking the community tier.

This audit uses AST to find `ImportFrom` and `Import` nodes targeting
`vllm.sndr_engine`. Each occurrence is then walked up the parent chain
to determine whether it sits inside a `try / except ImportError` block.
If so, the reference is OK; otherwise it's a violation.

Allowlist:

  • Lines with `audit-engine-boundary: allow` are exempt (operator
    escape hatch).

Exit codes:
  0 — clean.
  1 — at least one unguarded reference.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOT = REPO_ROOT / "sndr"

ALLOW_MARKER = "audit-engine-boundary: allow"


def _gather_files() -> list[Path]:
    out: list[Path] = []
    for fp in SCAN_ROOT.rglob("*.py"):
        if "__pycache__" in fp.parts:
            continue
        if fp.name.startswith("._"):
            # macOS AppleDouble resource fork: not Python source.
            continue
        out.append(fp)
    return sorted(out)


def _import_target(node: ast.AST) -> str | None:
    """Return 'vllm.sndr_engine' if this import node references the
    commercial overlay (incl. submodules); else None."""
    if isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        if mod == "vllm.sndr_engine" or mod.startswith("vllm.sndr_engine."):
            return mod
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == "vllm.sndr_engine" or alias.name.startswith(
                "vllm.sndr_engine."
            ):
                return alias.name
    return None


def _in_try_except_importerror(node: ast.AST, parent_map: dict) -> bool:
    """Walk up parents — return True if a Try node above has ImportError
    in any of its handlers (or a bare except)."""
    cur = parent_map.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.Try):
            for h in cur.handlers:
                if h.type is None:
                    return True
                # Direct: except ImportError / ModuleNotFoundError
                if isinstance(h.type, ast.Name) and h.type.id in {
                    "ImportError", "ModuleNotFoundError", "Exception",
                    "BaseException",
                }:
                    return True
                # Tuple: except (ImportError, X)
                if isinstance(h.type, ast.Tuple):
                    for elt in h.type.elts:
                        if isinstance(elt, ast.Name) and elt.id in {
                            "ImportError", "ModuleNotFoundError",
                        }:
                            return True
        cur = parent_map.get(id(cur))
    return False


def _build_parent_map(tree: ast.AST) -> dict:
    parents: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _check_file(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [f"{path.relative_to(REPO_ROOT).as_posix()}:{e.lineno}: SYNTAX ERROR"]
    lines = source.splitlines()
    parents = _build_parent_map(tree)
    out: list[str] = []
    rel = path.relative_to(REPO_ROOT).as_posix()
    for node in ast.walk(tree):
        target = _import_target(node)
        if target is None:
            continue
        lineno = node.lineno
        line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if ALLOW_MARKER in line:
            continue
        if _in_try_except_importerror(node, parents):
            continue
        out.append(f"{rel}:{lineno}: unguarded {target} ({line.strip()[:80]})")
    return out


def audit() -> list[str]:
    hits: list[str] = []
    for fp in _gather_files():
        hits.extend(_check_file(fp))
    return hits


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
                {"violations": [], "total": 0, "error": msg},
                indent=2, sort_keys=True,
            ))
        else:
            print(f"audit-engine-boundary: {msg}")
            print("  FAIL — nothing scanned; refusing vacuous PASS")
        return 2
    hits = audit()
    if args.json:
        print(json.dumps(
            {"violations": hits, "total": len(hits)},
            indent=2, sort_keys=True,
        ))
    else:
        print("audit-engine-boundary: scanning sndr/**/*.py")
        print("─" * 70)
        if hits:
            print(f"  ✗ unguarded vllm.sndr_engine references: {len(hits)} hit(s)")
            for h in hits[:10]:
                print(f"      {h}")
            if len(hits) > 10:
                print(f"      ... ({len(hits) - 10} more)")
            print()
            print(f"  FAIL — {len(hits)} unguarded reference(s)")
            print("  Fix: wrap the import in try/except ImportError, OR")
            print("  add 'audit-engine-boundary: allow' on the line if it's")
            print("  a genuine first-party engine-tier file.")
        else:
            print("  ✓ no unguarded vllm.sndr_engine imports in sndr_core")
            print()
            print("  OK — engine boundary preserved")
    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())

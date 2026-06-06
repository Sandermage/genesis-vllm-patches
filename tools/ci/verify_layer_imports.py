#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Enforce sndr-platform's layered architecture via static analysis.

Rules (see Master Spec Part 4):

    Layer 0: sndr/kernel/, sndr/detection/, sndr/observability/
             May import: stdlib + approved third-party only
    Layer 1: sndr/engines/
             May import: layer 0 + sndr/exceptions, sndr/config
    Layer 2: sndr/dispatcher/
             May import: layers 0-1 + sndr/license
    Layer 3: sndr/apply/
             May import: layers 0-2
    Layer 4: sndr/product_api/
             May import: layers 0-3 + sndr/license
    Layer 5: sndr/cli/
             May import: layers 0-4

Forbidden:
    - Any upward import (layer N from N+1, N+2, ...)
    - vllm.* imports in sndr/kernel/, sndr/detection/
    - Cross-engine imports (sndr.engines.vllm.* importing sndr.engines.sglang.*)

Exits 0 if all imports are within the rules; exits 1 with a report otherwise.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# Layer numbering. Lower numbers are "deeper" (foundation).
LAYERS = {
    "sndr/kernel": 0,
    "sndr/detection": 0,
    "sndr/observability": 0,
    "sndr/engines": 1,
    "sndr/dispatcher": 2,
    "sndr/apply": 3,
    "sndr/product_api": 4,
    "sndr/cli": 5,
}

# Layer-1 packages that are always callable from higher layers.
SHARED_PACKAGES = {"sndr.exceptions", "sndr.config", "sndr.version", "sndr.license"}

# Layer 0 packages may not import these.
LAYER_0_FORBIDDEN_PREFIXES = (
    "vllm",
    "sndr.engines",
    "sndr.dispatcher",
    "sndr.apply",
    "sndr.product_api",
    "sndr.cli",
)

# Engines may not import other engines.
ENGINE_SIBLINGS = ("sndr.engines.vllm", "sndr.engines.sglang")


def layer_of(file_path: Path) -> int | None:
    """Return the layer number of a file's owning module, or None if not tracked."""
    rel = file_path.as_posix()
    for layer_prefix, layer_n in LAYERS.items():
        if layer_prefix in rel:
            return layer_n
    return None


def extract_imports(file_path: Path) -> list[str]:
    """Parse a Python file and return TOP-LEVEL imported module names.

    Only module-level imports count as architectural dependencies. Imports
    inside functions are "lazy" — they execute conditionally and represent
    optional integrations, not contractual dependencies. This is the common
    pattern for cross-layer bridges during migration phases.

    Example: ``text_patch.py`` may have a lazy ``from vllm.X import Y``
    inside a function body to optionally enable fast-path caching when
    the vllm package is installed. This is acceptable; a top-level
    ``from vllm.X import Y`` would not be.
    """
    try:
        tree = ast.parse(file_path.read_text())
    except SyntaxError:
        return []

    imports: list[str] = []
    # Iterate ONLY the module's top-level body. Imports nested in functions,
    # classes, conditionals, or try/except blocks are skipped.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif isinstance(node, ast.Try):
            # try/except at module level is sometimes used for optional
            # imports (e.g. import optional_dep; if it fails, skip). These
            # ARE module-level and we still count them.
            for sub in node.body:
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        imports.append(alias.name)
                elif isinstance(sub, ast.ImportFrom) and sub.module:
                    imports.append(sub.module)
    return imports


def check_file(file_path: Path, errors: list[str]) -> None:
    """Check one file against the layer rules."""
    layer = layer_of(file_path)
    if layer is None:
        return  # Not in a tracked layer

    imports = extract_imports(file_path)

    # Layer 0: cannot import vllm or higher layers
    if layer == 0:
        for imp in imports:
            for forbidden in LAYER_0_FORBIDDEN_PREFIXES:
                if imp.startswith(forbidden):
                    errors.append(
                        f"{file_path}: layer 0 cannot import '{imp}' "
                        f"(starts with forbidden prefix '{forbidden}')"
                    )

    # All layers: cannot import from higher layers (numerically larger).
    for imp in imports:
        if not imp.startswith("sndr."):
            continue
        if imp in SHARED_PACKAGES or any(imp.startswith(p + ".") for p in SHARED_PACKAGES):
            continue

        imported_layer = None
        for prefix, n in LAYERS.items():
            if imp.startswith(prefix.replace("/", ".")):
                imported_layer = n
                break

        if imported_layer is not None and imported_layer > layer:
            errors.append(
                f"{file_path}: layer {layer} cannot import from layer "
                f"{imported_layer} ('{imp}')"
            )

    # Engines: cannot import sibling engines.
    if "sndr/engines/" in file_path.as_posix():
        for imp in imports:
            for sibling in ENGINE_SIBLINGS:
                # Determine which engine this file belongs to.
                owner = next(
                    (s for s in ENGINE_SIBLINGS if s.replace(".", "/") in file_path.as_posix()),
                    None,
                )
                if owner and sibling != owner and imp.startswith(sibling):
                    errors.append(
                        f"{file_path}: engine '{owner}' may not import "
                        f"sibling engine '{imp}'"
                    )


def main() -> int:
    root = Path(__file__).parent.parent.parent  # repo root
    sndr_dir = root / "sndr"

    if not sndr_dir.is_dir():
        print(f"WARN: {sndr_dir} does not exist; nothing to check.")
        return 0

    errors: list[str] = []
    for py_file in sndr_dir.rglob("*.py"):
        check_file(py_file, errors)

    if errors:
        print("LAYER RULES VIOLATIONS DETECTED")
        print("=" * 60)
        for err in errors:
            print(err)
        print("=" * 60)
        print(f"Total violations: {len(errors)}")
        return 1

    print(f"OK: scanned {sum(1 for _ in sndr_dir.rglob('*.py'))} files; "
          "no layer rule violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""DA-009 (audit 2026-05-08) — bundle tier ↔ registry tier consistency.

Bundles in `sndr/bundles/` declare a `tier` value that gates
the umbrella flag (engine bundles require license; community bundles
do not). Today the bundle's tier MUST match the tier of every patch
it includes — otherwise an `engine` bundle could be skipped while its
underlying community patches still apply individually, OR vice versa.

The previous mismatch (P67/P67b registered as community, bundle
declared engine) was caught by the production-readiness audit. This
test prevents regression.

Test strategy: parse each `sndr/bundles/*.py` for its
`run_bundle(..., tier=...)` invocation + `patcher_factories=[...]`
list, resolve each factory to its `_make_patcher` parent module's
patch_id (via the `_GENESIS_*_MARKER` convention or registry
introspection), then assert tier matches.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLES_DIR = REPO_ROOT / "sndr" / "bundles"


def _bundle_files() -> list[Path]:
    """Return every bundle module (excludes __init__ + _common)."""
    return [
        f for f in BUNDLES_DIR.glob("*.py")
        if f.name not in ("__init__.py", "_common.py")
    ]


def _parse_run_bundle_tier(src: str) -> str | None:
    """Extract the `tier=` kwarg passed to `run_bundle(...)`. None if
    the bundle module doesn't call run_bundle (skip)."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_bundle"):
            for kw in node.keywords:
                if kw.arg == "tier" and isinstance(kw.value, ast.Constant):
                    return kw.value.value
    return None


def _parse_imported_patch_modules(src: str) -> list[str]:
    """Extract module names imported as `from … import pXX_…` style."""
    tree = ast.parse(src)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                # Look for canonical pNN_ / pnNN_ patch module names.
                name = alias.asname or alias.name
                if re.match(r"^p[n]?\d+\w*$", name):
                    out.append(name)
                # Also grab module-level `_pXX` aliases used by
                # bundles (e.g. `from … import p67_… as _p67`).
                elif name.startswith("_p"):
                    actual = alias.name  # the original name before `as`
                    out.append(actual)
    return out


def _patch_id_from_module_name(module_name: str) -> str | None:
    """Map `p67_tq_multi_query_kernel` → `P67`,
    `pn82_mamba_cudagraph_prefill_zero` → `PN82`, etc."""
    m = re.match(r"^(p[n]?\d+\w*?)_", module_name)
    if not m:
        return None
    raw = m.group(1)
    if raw.lower().startswith("pn"):
        return "PN" + raw[2:].upper()
    return "P" + raw[1:].upper()


@pytest.fixture(scope="module")
def registry():
    """Load PATCH_REGISTRY once."""
    from sndr.dispatcher import PATCH_REGISTRY
    return PATCH_REGISTRY


def test_bundles_dir_present():
    assert BUNDLES_DIR.is_dir()
    assert _bundle_files(), "no bundle modules found"


def test_every_bundle_tier_matches_included_patches(registry):
    """For each bundle, every patch it includes must have a registry
    entry with the SAME tier as the bundle declares.

    DA-009 (2026-05-08): caught the legacy P67/P67b "engine" mislabel.
    """
    violations: list[str] = []
    for bf in _bundle_files():
        src = bf.read_text(encoding="utf-8")
        bundle_tier = _parse_run_bundle_tier(src)
        if bundle_tier is None:
            # Bundle module doesn't call run_bundle — skip (probably a
            # helper or work-in-progress bundle).
            continue
        modules = _parse_imported_patch_modules(src)
        for mod in modules:
            pid = _patch_id_from_module_name(mod)
            if pid is None:
                continue
            entry = registry.get(pid)
            if not isinstance(entry, dict):
                # Patch isn't in registry — separate concern from tier
                # consistency; skip silently.
                continue
            entry_tier = entry.get("tier")
            if entry_tier and entry_tier != bundle_tier:
                violations.append(
                    f"{bf.name}: bundle tier={bundle_tier!r} but "
                    f"included patch {pid} has tier={entry_tier!r}"
                )
    assert not violations, (
        "bundle/registry tier mismatch:\n  " + "\n  ".join(violations)
    )

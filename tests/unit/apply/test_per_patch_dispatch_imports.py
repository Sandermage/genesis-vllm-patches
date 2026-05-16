# SPDX-License-Identifier: Apache-2.0
"""DA-002 (audit 2026-05-08): regression guard for `_per_patch_dispatch.py`
NameError-class bugs.

The legacy parking-lot module hand-codes 124+ wrappers shaped like:

    @register_patch("P4 TurboQuant hybrid model support")
    def apply_patch_4_tq_hybrid() -> PatchResult:
        ...
        from vllm.sndr_core.integrations.scheduler import p4_tq_hybrid
        ...
        status, reason = p4_tq_hybrid.apply()    # ← MUST match imported name

Until 2026-05-08 audit, several wrappers had the imported symbol
(`p4_tq_hybrid`) named differently from the call (`patch_4_tq_hybrid`),
producing `NameError` only at dry-run time. Because dry-run is the
installer smoke path, this surfaced as `install --dry-run` exit 2 in
production with a spurious "wiring import failed" reason.

This test enforces the contract by:

  1. Walking every `apply_patch_<N>_<...>()` wrapper via AST.
  2. Extracting its `from ... import <X>` statements.
  3. Asserting that any `<symbol>.apply()` call inside the wrapper
     uses one of the imported names — never `patch_<N>_<...>` unless
     that exact name is what was imported.

Why AST instead of just running every wrapper:
  - Wrappers may try to import torch-heavy modules; AST is cold-import safe.
  - We're checking the SHAPE of the code, not its runtime behavior — the
    legacy parking lot is being phased out via spec-driven dispatch
    (PR38 Day 6-8), so a runtime-execution test would be retiring the
    same time as the parking lot itself.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DISPATCH = REPO_ROOT / "vllm" / "sndr_core" / "apply" / "_per_patch_dispatch.py"


def _collect_wrappers() -> list[ast.FunctionDef]:
    """Return every `def apply_patch_<...>()` wrapper in the dispatch file."""
    if not DISPATCH.is_file():
        pytest.skip(f"dispatch file not present: {DISPATCH}")
    src = DISPATCH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(DISPATCH))
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("apply_patch_")
    ]


def _imported_names(fn: ast.FunctionDef) -> set[str]:
    """Walk the wrapper body for `from X import Y` and `import Y` statements,
    return the set of bound local names."""
    names: set[str] = set()
    for sub in ast.walk(fn):
        if isinstance(sub, ast.ImportFrom):
            for alias in sub.names:
                names.add(alias.asname or alias.name)
        elif isinstance(sub, ast.Import):
            for alias in sub.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _called_modules(fn: ast.FunctionDef) -> list[tuple[str, int]]:
    """Find `<NAME>.apply(...)` and `callable(<NAME>.apply)` references.

    Returns list of (name, lineno).
    """
    out: list[tuple[str, int]] = []
    for sub in ast.walk(fn):
        # `X.apply(...)` call
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "apply"
                and isinstance(sub.func.value, ast.Name)):
            out.append((sub.func.value.id, sub.lineno))
        # `callable(X.apply)` reference
        if (isinstance(sub, ast.Attribute)
                and sub.attr == "apply"
                and isinstance(sub.value, ast.Name)):
            out.append((sub.value.id, sub.lineno))
    return out


def test_dispatch_file_present():
    assert DISPATCH.is_file(), f"missing {DISPATCH}"


_DELEGATION_HELPERS = {
    "_wiring_text_patch",  # generic dispatcher
    "_skipped",            # retirement stub returning early skip
    "_failed",             # explicit-error stub
    "_applied",            # marker-only success stub
    "result_to_wiring_status",  # status helper
}


def _delegates_to_wiring_helper(fn: ast.FunctionDef) -> bool:
    """True if the wrapper body delegates to one of the well-known
    helpers (`_wiring_text_patch`, `_skipped`, `_failed`, `_applied`,
    `result_to_wiring_status`). Such wrappers don't need a per-wrapper
    import statement — they either resolve wiring through the runtime
    stem index OR are intentional retirement / marker-only stubs."""
    for sub in ast.walk(fn):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id in _DELEGATION_HELPERS):
            return True
    return False


_DEPRECATED_ALIAS_WRAPPERS: frozenset[str] = frozenset({
    # 2026-05-14 PN96 rename: PN96 → PN96b. The original wrapper name
    # is preserved as a one-release deprecation alias that simply
    # invokes the new wrapper. By design it has no imports and no
    # helper delegation — it's just `return apply_patch_N96b_*()`.
    "apply_patch_N96_marlin_persistent_workspace",
})


def _calls_another_wrapper(fn: ast.FunctionDef) -> bool:
    """True if the wrapper body invokes another `apply_patch_*` function
    by name. Used to whitelist deprecation aliases."""
    for sub in ast.walk(fn):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id.startswith("apply_patch_")):
            return True
    return False


def test_every_wrapper_either_imports_or_delegates_to_helper():
    """Every `apply_patch_*` wrapper either:
       (a) imports a wiring module via `from ... import ...`, OR
       (b) delegates to `_wiring_text_patch(name, stem)` which resolves
           the wiring module through the runtime stem index, OR
       (c) is an audited deprecation alias that invokes the renamed
           wrapper (see `_DEPRECATED_ALIAS_WRAPPERS`).

    Wrappers doing none of the above are dead code (no path to apply).
    """
    wrappers = _collect_wrappers()
    assert wrappers, "no apply_patch_* wrappers found"
    dead: list[str] = []
    for w in wrappers:
        if w.name in _DEPRECATED_ALIAS_WRAPPERS:
            assert _calls_another_wrapper(w), (
                f"{w.name} listed as deprecation alias but does not "
                f"call any apply_patch_* function — remove from "
                f"_DEPRECATED_ALIAS_WRAPPERS or fix the body."
            )
            continue
        has_imports = bool(_imported_names(w))
        delegates = _delegates_to_wiring_helper(w)
        if not has_imports and not delegates:
            dead.append(w.name)
    assert not dead, (
        f"wrappers with no imports AND no _wiring_text_patch delegation: "
        f"{dead}. These cannot apply anything."
    )


def test_every_apply_call_resolves_to_imported_name():
    """DA-002 (audit 2026-05-08): every `<symbol>.apply()` in a wrapper
    must reference a NAME that was imported at the local function scope.

    This catches the P4/P5 NameError class systemically — `from X import
    p4_tq_hybrid` followed by `patch_4_tq_hybrid.apply()` previously
    only surfaced at runtime as "wiring import failed: name
    'patch_4_tq_hybrid' is not defined". Now caught statically by AST.
    """
    wrappers = _collect_wrappers()
    violations: list[str] = []
    for fn in wrappers:
        imported = _imported_names(fn)
        # Globals ALSO count — some wrappers reference module-level helpers.
        for name, lineno in _called_modules(fn):
            if name in imported:
                continue
            # Allowed module-level helpers in the dispatch file itself.
            allowed_globals = {
                "decision",  # dispatcher decision module
                "log",       # logging
                "result_to_wiring_status",  # status helper
            }
            if name in allowed_globals:
                continue
            violations.append(
                f"{fn.name}() at line {lineno} calls `{name}.apply()` "
                f"but `{name}` is not imported (locals: "
                f"{sorted(imported)})"
            )
    assert not violations, (
        "DA-002 contract violations:\n  " + "\n  ".join(violations)
    )


def test_no_legacy_patch_NN_symbols_remain():
    """Belt-and-braces: zero `patch_NN_<...>` symbols followed by `.apply`
    or `.apply()` in the dispatch file. The canonical naming is
    `pNN_<...>` matching the wiring module file basenames.
    """
    src = DISPATCH.read_text(encoding="utf-8")
    import re
    # Match `patch_<digits><optional letter>_<word>.apply` but EXCLUDE
    # `apply_patch_<...>` (function names — those keep the `apply_patch_`
    # prefix per the original taxonomy) and `register_patch(`.
    pattern = re.compile(
        r"(?<!apply_)(?<!register_)\bpatch_(\d+[a-z]?)_(\w+)\.apply"
    )
    hits = pattern.findall(src)
    assert not hits, (
        f"found {len(hits)} legacy `patch_NN_` symbol references that "
        "should be `pNN_`. Run scripts/audit/da001_typo_fix.py to fix."
    )

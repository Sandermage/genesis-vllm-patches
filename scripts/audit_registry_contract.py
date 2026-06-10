#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit registry contract — aggregate validator for PATCH_REGISTRY drift.

Single CI-friendly gate that asserts eight invariants in one pass:

  1. conflicts_with symmetry — if A conflicts_with B, then B conflicts_with A.
  2. category enum — every patch.category in VALID_CATEGORIES.
  3. family ↔ patches path consistency — wiring file lives under
     `sndr/engines/vllm/patches/<family>/` (`.` in family → `/` in path).
  4. apply_module coverage — every non-retired patch with on-disk wiring
     has a resolvable apply_module reference.
  5. retired provenance — retired patches have superseded_by + vllm_version_range
     (or explicit waiver flag).
  6. pin gate — patch.applies_to.vllm_pin entries are in KNOWN_GOOD_VLLM_PINS.
  7. dict_dup_keys — PATCH_REGISTRY dict literal has no duplicate string
     keys (Python dict silently shadows on collision; PN96 incident).
  8. docstring lifecycle sync — integration-module docstring TOMBSTONED/
     RETIRED markers agree with registry `lifecycle` field.

Exit codes:
  0 — all invariants hold (CI green)
  1 — at least one drift case detected (CI red)
  2 — internal error / registry not loadable

Modes:
  python3 scripts/audit_registry_contract.py            # human-readable
  python3 scripts/audit_registry_contract.py --json     # machine-readable
  python3 scripts/audit_registry_contract.py --strict   # exit 1 on warnings too

The script is intentionally read-only — no edits to registry.py or tests.
Add this to `make evidence` after audit-no-stub for an aggregate gate.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _check_conflicts_symmetry(registry: dict[str, dict[str, Any]]) -> list[str]:
    """Invariant 1: conflicts_with must be symmetric."""
    issues: list[str] = []
    for pid, meta in registry.items():
        conflicts = meta.get("conflicts_with") or []
        for other in conflicts:
            other_meta = registry.get(other)
            if other_meta is None:
                issues.append(
                    f"conflicts: {pid}.conflicts_with includes {other!r} "
                    "which is not in registry"
                )
                continue
            reverse = other_meta.get("conflicts_with") or []
            if pid not in reverse:
                issues.append(
                    f"conflicts: {pid} → {other} not symmetric "
                    f"({other}.conflicts_with = {reverse})"
                )
    return issues


def _check_category_enum(registry: dict[str, dict[str, Any]]) -> list[str]:
    """Invariant 2: every category in VALID_CATEGORIES."""
    try:
        from sndr.dispatcher.spec import VALID_CATEGORIES
    except ImportError as e:
        return [f"category: cannot import VALID_CATEGORIES ({e})"]
    issues: list[str] = []
    for pid, meta in registry.items():
        cat = meta.get("category")
        if cat is None:
            continue  # category is optional in some entries
        if cat not in VALID_CATEGORIES:
            issues.append(
                f"category: {pid}.category={cat!r} not in VALID_CATEGORIES "
                f"({len(VALID_CATEGORIES)} valid values)"
            )
    return issues


# Family-name → on-disk subdir aliases for cases where the family
# label is a model lineage (kept for `sndr family <X> status` ergonomics)
# but the wiring lives under a technical-area bucket. Phase 2.2
# (2026-05-22) relocated the 18 real Gemma compat patches to
# `model_compat/gemma4/` while keeping `family='gemma4'` so the
# model-family semantic stays intact. The audit honors that by
# mapping the family label through this alias table before computing
# the expected subdir.
_FAMILY_TO_SUBDIR_ALIAS: dict[str, str] = {
    "gemma4": "model_compat/gemma4",
}


def _check_family_path(registry: dict[str, dict[str, Any]]) -> list[str]:
    """Invariant 3: registry family ↔ patches subdir consistency.

    Uses the registry's canonical `apply_module` directly — not the
    filesystem-walking `module_for()` — so that one-release relocation
    shims (which duplicate the wiring file at the old path) don't
    confuse the check. The registry is the single source of truth for
    where a patch's real implementation lives; the audit follows it.

    A patch's `apply_module` may live deeper than one level under the
    family directory (e.g. probes/ under spec_decode/). Acceptance
    rule: the apply_module's prefix path must START with
    `patches/<expected>/` (v12 tree: `sndr/engines/vllm/patches/`,
    previously `vllm/sndr_core/integrations/`), where `<expected>` is
    either the family label converted from dotted to slash form, or
    the alias-table value for cases where family is a model lineage
    (Phase 2.2 — see `_FAMILY_TO_SUBDIR_ALIAS`).
    """
    issues: list[str] = []
    for pid, meta in registry.items():
        fam = meta.get("family")
        if not fam or fam == "model_specific":
            continue
        # Retired patches live in `_retired/` by design — family stays
        # informational (original wiring family), location is `_retired/`.
        if meta.get("lifecycle") == "retired":
            continue
        mod = meta.get("apply_module")
        if not mod or ".patches." not in mod:
            continue
        after_int = mod.split(".patches.", 1)[1]
        subdir = after_int.rsplit(".", 1)[0].replace(".", "/")
        expected = _FAMILY_TO_SUBDIR_ALIAS.get(fam, fam.replace(".", "/"))
        # subdir may be a deeper path under the family (e.g.
        # `spec_decode/probes`); accept when it starts with the
        # expected family path.
        if subdir != expected and not subdir.startswith(expected + "/"):
            issues.append(
                f"family-path: {pid} registry family={fam!r} "
                f"(→ patches/{expected}/) but apply_module at "
                f"patches/{subdir}/"
            )
    return issues


def _check_apply_module_coverage(
    registry: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Invariant 4: every active patch resolves to importable apply_module.

    Returns (errors, warnings) — errors for missing modules, warnings
    for patches lacking apply_module entirely (informational).
    """
    errors: list[str] = []
    warnings: list[str] = []
    # implementation_status values that legitimately have no apply_module
    # (advisory entries, placeholders, exploratory wrappers).
    # `marker_only` is the canonical enum value (per dispatcher validator);
    # other strings (metadata_only/advisory/research) are accepted for
    # backward compat from earlier drafts.
    NO_WIRING_OK = {"marker_only", "metadata_only", "placeholder", "advisory", "research"}
    for pid, meta in registry.items():
        lifecycle = meta.get("lifecycle", "experimental")
        if lifecycle == "retired":
            continue
        impl_status = meta.get("implementation_status", "")
        am = meta.get("apply_module") or ""
        if not am:
            # Suppress warning when entry is explicitly informational.
            if impl_status in NO_WIRING_OK:
                continue
            warnings.append(
                f"apply_module: {pid} has no apply_module"
                f" (consider implementation_status: metadata_only/placeholder)"
            )
            continue
        try:
            importlib.import_module(am)
        except (ImportError, ModuleNotFoundError) as e:
            errors.append(
                f"apply_module: {pid}.apply_module={am!r} not importable ({e})"
            )
    return errors, warnings


def _check_retired_provenance(registry: dict[str, dict[str, Any]]) -> list[str]:
    """Invariant 5: retired patches have provenance (superseded_by + version range
    or waiver)."""
    issues: list[str] = []
    for pid, meta in registry.items():
        if meta.get("lifecycle") != "retired":
            continue
        has_superseded = bool(meta.get("superseded_by"))
        has_version = bool(meta.get("vllm_version_range"))
        has_waiver = bool(meta.get("retired_waiver"))
        if has_waiver:
            continue
        if has_superseded and not has_version:
            issues.append(
                f"retired: {pid}.superseded_by set but vllm_version_range "
                "missing (cannot prove safe drift)"
            )
        elif not has_superseded and not has_version:
            issues.append(
                f"retired: {pid} has no superseded_by, vllm_version_range, "
                "or retired_waiver — provenance incomplete"
            )
    return issues


def _check_docstring_lifecycle_sync(registry: dict[str, dict[str, Any]]) -> list[str]:
    """Invariant 8: docstring TOMBSTONED/RETIRED markers ↔ registry lifecycle.

    Catches drift where a patch module's docstring declares itself retired
    (e.g. "TOMBSTONED — fla recurrent kernel cannot serve single-seq
    prefill") but the registry entry still says `lifecycle: experimental`.
    This bug class was hit by PN108 (caught during Phase 2.1 manual sync).

    For each registry entry with apply_module pointing into the
    engine patches tree (`sndr/engines/vllm/patches/`):
      1. Try import the module
      2. Read its docstring (`module.__doc__`)
      3. Search for markers: TOMBSTONED, RETIRED, lifecycle.*retired
      4. If marker found but registry lifecycle != retired → drift

    Skips registry entries whose lifecycle IS retired (already in sync).

    Patterns must declare THIS patch retired — not merely mention other
    patches' retirement. Use file-level markers (TOMBSTONED at start,
    "lifecycle: retired" assertion) rather than casual word matches.
    """
    import importlib
    import re

    # File-level/self-declaration markers only. Word "RETIRED" alone is
    # not enough — docstrings frequently mention OTHER patches being
    # retired (e.g. "replaces retired P7", "intended for retired after
    # evidence", "self-retired when upstream lands"). Use phrases that
    # unambiguously assert THIS patch's state.
    DOCSTRING_RETIRED_PATTERNS = (
        r"\bTOMBSTONED\b",                         # all-caps file marker
        r"\blifecycle[\s:=]+retired\b",            # explicit lifecycle assertion
        r"\bstatus:\s*retired\b",                  # explicit status assertion
        r"\bthis\s+patch\s+is\s+retired\b",        # self-declaration prose
        r"\bharmless\s+no-op\s+now\b",             # PN108-style tombstone phrase
        r"\bduplicate\s+of\s+(?:patch|sndr_|p[n]?\d+)",  # PN34-style
    )
    issues: list[str] = []
    for pid, meta in registry.items():
        if meta.get("lifecycle") == "retired":
            continue  # already in sync
        am = meta.get("apply_module") or ""
        if not am or ".patches." not in am:
            continue
        try:
            mod = importlib.import_module(am)
        except (ImportError, ModuleNotFoundError):
            continue  # apply_module check handles importability separately
        doc = (mod.__doc__ or "").upper()
        if not doc:
            continue
        for pat in DOCSTRING_RETIRED_PATTERNS:
            if re.search(pat, doc, re.IGNORECASE):
                issues.append(
                    f"docstring-lifecycle: {pid} docstring contains "
                    f"retired/tombstoned marker ({pat!r}) but registry "
                    f"lifecycle={meta.get('lifecycle')!r}"
                )
                break  # one match per patch is enough
    return issues


def _check_dict_dup_keys() -> list[str]:
    """Invariant 7: PATCH_REGISTRY dict literal in registry.py has no
    duplicate keys.

    Python dict literals silently override on duplicate keys — last one
    wins, first is shadowed. This bug class was hit by PN96 collision
    (Marlin MoE workspace + emergency demote both keyed "PN96" — only
    second loaded in production despite default_on=True on first).

    Approach: parse registry.py via AST, walk PATCH_REGISTRY dict
    literal, count occurrences of each string key. Any count > 1 is a
    shadowing bug.
    """
    import ast

    issues: list[str] = []
    registry_path = (
        REPO_ROOT
        / "sndr" / "dispatcher"
        / "registry.py"
    )
    try:
        source = registry_path.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError) as e:
        return [f"dict-dup: cannot parse registry.py ({e})"]

    # Find the PATCH_REGISTRY = { ... } assignment. The current registry
    # uses annotated assignment: `PATCH_REGISTRY: dict[...] = {...}`
    # which is ast.AnnAssign, not ast.Assign. Handle both forms.
    patch_registry_dict = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == "PATCH_REGISTRY"
                and isinstance(node.value, ast.Dict)
            ):
                patch_registry_dict = node.value
                break
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "PATCH_REGISTRY"
                    and isinstance(node.value, ast.Dict)
                ):
                    patch_registry_dict = node.value
                    break
            if patch_registry_dict is not None:
                break

    if patch_registry_dict is None:
        return ["dict-dup: PATCH_REGISTRY = {...} not found in registry.py"]

    # Walk all string-literal keys; count occurrences.
    from collections import Counter

    keys_seen = []
    for key_node in patch_registry_dict.keys:
        # Use ast.unparse to get the literal key (Python 3.9+).
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            keys_seen.append((key_node.value, key_node.lineno))
        else:
            # Non-literal key (computed expression) — skip but note.
            issues.append(
                f"dict-dup: non-literal key at line {key_node.lineno} "
                "(audit cannot deduplicate)"
            )

    counts = Counter(k for k, _ in keys_seen)
    for key, count in counts.items():
        if count > 1:
            lines = [ln for k, ln in keys_seen if k == key]
            issues.append(
                f"dict-dup: PATCH_REGISTRY[{key!r}] appears {count}× "
                f"at lines {lines} — Python dict will silently shadow all "
                "but the LAST occurrence"
            )

    return issues


def _check_pin_gate(registry: dict[str, dict[str, Any]]) -> list[str]:
    """Invariant 6: any applies_to.vllm_pin entry is in KNOWN_GOOD_VLLM_PINS."""
    try:
        from sndr.engines.vllm.detection.guards import KNOWN_GOOD_VLLM_PINS
    except ImportError as e:
        return [f"pin-gate: cannot import KNOWN_GOOD_VLLM_PINS ({e})"]
    known = set(KNOWN_GOOD_VLLM_PINS)
    issues: list[str] = []
    for pid, meta in registry.items():
        applies_to = meta.get("applies_to") or {}
        pins = applies_to.get("vllm_pin")
        if not pins:
            continue
        if isinstance(pins, str):
            pins = [pins]
        for p in pins:
            if p not in known:
                issues.append(
                    f"pin-gate: {pid}.applies_to.vllm_pin={p!r} not in "
                    "KNOWN_GOOD_VLLM_PINS"
                )
    return issues


def run_audit(strict: bool = False) -> dict[str, Any]:
    """Run all 6 invariants. Returns dict with errors/warnings per check."""
    try:
        from sndr.dispatcher import PATCH_REGISTRY as registry
    except ImportError as e:
        return {"_internal_error": f"cannot import PATCH_REGISTRY ({e})"}

    apply_errors, apply_warnings = _check_apply_module_coverage(registry)
    results = {
        "conflicts_symmetry": _check_conflicts_symmetry(registry),
        "category_enum": _check_category_enum(registry),
        "family_path": _check_family_path(registry),
        "apply_module": apply_errors,
        "apply_module_warnings": apply_warnings,
        "retired_provenance": _check_retired_provenance(registry),
        "pin_gate": _check_pin_gate(registry),
        "dict_dup_keys": _check_dict_dup_keys(),
        "docstring_lifecycle": _check_docstring_lifecycle_sync(registry),
        "_count": len(registry),
    }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero on warnings (e.g. apply_module missing)",
    )
    args = parser.parse_args(argv)

    results = run_audit(strict=args.strict)
    if "_internal_error" in results:
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"INTERNAL ERROR: {results['_internal_error']}", file=sys.stderr)
        return 2

    total_errors = sum(
        len(v) for k, v in results.items()
        if k not in ("_count", "apply_module_warnings")
        and isinstance(v, list)
    )
    total_warnings = len(results.get("apply_module_warnings", []))

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print("╭──────────────────────────────────────────────────────────╮")
        print(f"│  audit_registry_contract: {results['_count']:>3} entries{' ' * 19}│")
        print("╰──────────────────────────────────────────────────────────╯")
        for check in (
            "conflicts_symmetry",
            "category_enum",
            "family_path",
            "apply_module",
            "retired_provenance",
            "pin_gate",
            "dict_dup_keys",
            "docstring_lifecycle",
        ):
            items = results.get(check, [])
            status = "✓" if not items else "✗"
            print(f"  {status} {check:<22} {len(items):>3} issue(s)")
            for it in items[:5]:
                print(f"      - {it}")
            if len(items) > 5:
                print(f"      ... +{len(items) - 5} more")
        if total_warnings:
            print(f"\n  ⚠ {total_warnings} apply_module warning(s) "
                  "(informational; use --strict to fail)")
        print()
        if total_errors == 0 and (not args.strict or total_warnings == 0):
            print("  ✓ REGISTRY CONTRACT CLEAN")
        else:
            print(f"  ✗ {total_errors} error(s)"
                  + (f" + {total_warnings} warning(s)" if args.strict else "")
                  + " — drift detected")

    if total_errors > 0:
        return 1
    if args.strict and total_warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — every spec entry's `apply_module` must
match the canonical module that the corresponding legacy
`@register_patch` hook actually calls `.apply()` on.

Why this matters
----------------

The v12.0.0 default-flip from legacy `@register_patch` iteration to
spec-driven `iter_patch_specs()` is only safe if BOTH paths invoke
the SAME apply module for each patch ID. Otherwise the flip silently
changes which wiring runs — a regression that the patch-id coverage
audit (`audit_legacy_vs_spec_driven_apply_matrix`) misses because it
checks coverage, not semantic agreement.

v11.3.0 BUG #8/#9/#10 discovered three such mismatches:

  PN16 — spec=pn16_v6_streaming_truncator, legacy=pn16_lazy_reasoner
         (env GENESIS_ENABLE_PN16_LAZY_REASONER would activate V6
          streaming truncator on flip, breaking lazy-reasoner gating)

  PN26 — spec=pn26_sparse_v_kernel, legacy=pn26_tq_unified_perf
         (env GENESIS_ENABLE_PN26_TQ_UNIFIED would activate the
          risky sparse-V kernel instead of the safe centroids prebake)

  PN40 — spec=pn40_workload_classifier_hook, legacy=pn40_dflash_omnibus
         (env GENESIS_ENABLE_PN40_DFLASH_OMNIBUS would activate only
          sub-D classifier, dropping sub-A/B/C K-norm fusion +
          adaptive K/N control)

All three fixed in the same commit as this test.

This guard scans every `_per_patch_dispatch.py` legacy hook, finds
the `<module>.apply()` call inside the function body, and compares
the imported module's fully-qualified path against the spec entry's
`apply_module`. Mismatch → fail with offender list.

Allowlist via `_KNOWN_PER_PATCH_DIVERGENT` for legitimate cases
(e.g. legacy hook intentionally calls a different module for
back-compat — empty at v11.3.0).

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ regression guard.
"""
from __future__ import annotations

import inspect
import re


# Allowlist for patches where spec apply_module legitimately differs
# from the legacy hook's call target (e.g. legacy is a shim, spec
# points at the underlying module). Empty at v11.3.0 — every entry
# requires a justification comment.
_KNOWN_PER_PATCH_DIVERGENT: frozenset[str] = frozenset({
    # No allowlist entries at v11.3.0.
})


def _scan_legacy_hooks_for_apply_targets() -> dict[str, set[str]]:
    """For each legacy `@register_patch`-registered function, return
    the set of fully-qualified module paths that the function body
    calls `.apply()` on.

    Returns: {patch_id: {full_module_path, ...}}
    """
    from sndr.apply import _state, _per_patch_dispatch  # noqa: F401
    from sndr.dispatcher.registry import PATCH_REGISTRY
    try:
        from scripts.audit_legacy_vs_spec_driven_apply_matrix import (
            _extract_legacy_patch_id,
        )
    except ImportError:
        # Fallback: just use first whitespace-delimited token
        def _extract_legacy_patch_id(name, _spec_set):
            return name.split()[0] if name else ""

    spec_ids = set(PATCH_REGISTRY.keys())

    simple_re = re.compile(
        r"from\s+(vllm\.sndr_core\.integrations\.[\w\.]+)\s+import\s+"
        r"(\w+)(?:\s+as\s+(_\w+))?"
    )
    paren_re = re.compile(
        r"from\s+(vllm\.sndr_core\.integrations\.[\w\.]+)\s+import\s*"
        r"\(\s*\n\s*(\w+)(?:\s+as\s+(_\w+))?",
        re.M,
    )
    apply_re = re.compile(r"(\w+)\.apply\(\)")

    out: dict[str, set[str]] = {}
    for name, fn in _state.PATCH_REGISTRY:
        pid = _extract_legacy_patch_id(name, spec_ids)
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        alias_to_mod: dict[str, str] = {}
        name_to_mod: dict[str, str] = {}
        for m in simple_re.finditer(src):
            path, mod, alias = m.group(1), m.group(2), m.group(3)
            full = f"{path}.{mod}"
            if alias:
                alias_to_mod[alias] = full
            name_to_mod[mod] = full
        for m in paren_re.finditer(src):
            path, mod, alias = m.group(1), m.group(2), m.group(3)
            full = f"{path}.{mod}"
            if alias:
                alias_to_mod[alias] = full
            name_to_mod[mod] = full
        for m in apply_re.finditer(src):
            ref = m.group(1)
            mod_path = alias_to_mod.get(ref) or name_to_mod.get(ref)
            if mod_path:
                out.setdefault(pid, set()).add(mod_path)
    return out


def test_no_spec_apply_module_mismatch_with_legacy():
    """Every legacy-registered patch's `<X>.apply()` call target must
    match the spec entry's `apply_module`, OR the patch must be in
    `_KNOWN_PER_PATCH_DIVERGENT` with a justification comment."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    legacy_targets = _scan_legacy_hooks_for_apply_targets()
    mismatches: list[tuple[str, str, list[str]]] = []
    for pid, legacy_mods in legacy_targets.items():
        if pid in _KNOWN_PER_PATCH_DIVERGENT:
            continue
        spec_entry = PATCH_REGISTRY.get(pid)
        if spec_entry is None or not isinstance(spec_entry, dict):
            continue
        spec_apply = spec_entry.get("apply_module")
        if not spec_apply:
            continue
        if spec_apply not in legacy_mods:
            mismatches.append((pid, spec_apply, sorted(legacy_mods)))
    if mismatches:
        lines = [
            f"  - {pid}: spec={spec!r} legacy_calls={legacy}"
            for pid, spec, legacy in mismatches
        ]
        raise AssertionError(
            f"{len(mismatches)} patch(es) have spec `apply_module` that "
            f"does NOT match the module their legacy `@register_patch` "
            f"hook calls `.apply()` on. On v12.0.0 default-flip these "
            f"patches would change which wiring runs — silent "
            f"behavior regression.\n\nOffenders:\n" + "\n".join(lines)
        )

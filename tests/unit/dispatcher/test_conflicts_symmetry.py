"""`conflicts_with` MUST be a symmetric relation across the registry.

If patch A declares `conflicts_with: [B]`, then patch B must also list A.
Otherwise: when user enables B first then A, the dispatcher silently allows
both — the conflict check only fires from the side that DECLARED it.

Background — 2026-05-12 audit found `PN79.conflicts_with: [PN59, PN54]`
but neither PN59 nor PN54 listed PN79 back. Both fixed in registry; this
test prevents the regression class.

Also asserts: cross-references in `conflicts_with` / `requires_patches` /
`composes_with` point to patch IDs that actually exist in the registry.
"""

from __future__ import annotations


def test_conflicts_with_is_symmetric():
    """If A.conflicts_with contains B, then B.conflicts_with must contain A."""
    from sndr.dispatcher import PATCH_REGISTRY

    asymm = []
    for pid, meta in PATCH_REGISTRY.items():
        for other in (meta.get("conflicts_with") or []):
            other_meta = PATCH_REGISTRY.get(other)
            if other_meta is None:
                continue  # caught by test_conflicts_with_targets_exist
            other_conflicts = other_meta.get("conflicts_with") or []
            if pid not in other_conflicts:
                asymm.append(f"{pid} → {other} (but {other} doesn't list {pid})")
    assert not asymm, (
        f"{len(asymm)} asymmetric conflicts_with declaration(s):\n  "
        + "\n  ".join(asymm[:10])
        + "\n\nFix: add the reverse direction to the other patch's "
        "conflicts_with list."
    )


def test_conflicts_with_targets_exist():
    """Every patch ID listed in `conflicts_with` must be in the registry."""
    from sndr.dispatcher import PATCH_REGISTRY

    broken = []
    for pid, meta in PATCH_REGISTRY.items():
        for other in (meta.get("conflicts_with") or []):
            if other not in PATCH_REGISTRY:
                broken.append(f"{pid}.conflicts_with → {other!r} (not in registry)")
    assert not broken, "Dangling conflicts_with references:\n  " + "\n  ".join(broken)


def test_requires_patches_targets_exist():
    """Every patch ID listed in `requires_patches` must be in the registry."""
    from sndr.dispatcher import PATCH_REGISTRY

    broken = []
    for pid, meta in PATCH_REGISTRY.items():
        for other in (meta.get("requires_patches") or []):
            if other not in PATCH_REGISTRY:
                broken.append(f"{pid}.requires_patches → {other!r} (not in registry)")
    assert not broken, "Dangling requires_patches references:\n  " + "\n  ".join(broken)


def test_composes_with_targets_exist():
    """Every patch ID listed in `composes_with` must be in the registry."""
    from sndr.dispatcher import PATCH_REGISTRY

    broken = []
    for pid, meta in PATCH_REGISTRY.items():
        for other in (meta.get("composes_with") or []):
            if other not in PATCH_REGISTRY:
                broken.append(f"{pid}.composes_with → {other!r} (not in registry)")
    assert not broken, "Dangling composes_with references:\n  " + "\n  ".join(broken)

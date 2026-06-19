# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — `iter_patch_specs(topo_sort=True)` yields
patches in a valid topological order respecting `requires_patches`.

Why this matters
----------------

The spec-driven apply orchestrator iterates `iter_patch_specs()` and
applies each patch's `apply_module` in iteration order. If a patch
declares `requires_patches=[<X>]`, that means `<X>` must have its
apply_module run BEFORE the dependent patch. The legacy
`@register_patch` decorator-order doesn't enforce this — and at the
v11.3.0 baseline, 6 spec entries have a `requires_patches` constraint
that the natural registry-insertion order violates:

  - PN105 requires PN104 (insertion has PN105 before PN104)
  - PN34  requires PN33  (PN34 before PN33)
  - G4_75 requires G4_74 (G4_75 before G4_74)
  - G4_70 requires G4_69 (G4_70 before G4_69)
  - PN256 requires G4_67 (PN256 before G4_67)
  - G4_69 requires G4_60K (G4_69 before G4_60K)

These would result in dependent patches running against
un-monkey-patched targets — broken/undefined behavior.

The fix: `iter_patch_specs(topo_sort=True)` uses Kahn's algorithm
with insertion-order tie-breaking to yield a stable topological
order. The spec-driven orchestrator opts in via env
`SNDR_TOPO_SORT_SPECS=1`. Default OFF to preserve current behavior
for the staged v12.0.0 rollout.

This test pins two invariants:
  1. `topo_sort=True` output has ZERO order violations
  2. `topo_sort=False` (default) reproduces the registry-insertion
     order with the known violations as documented baseline

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ regression guard.
"""
from __future__ import annotations


def _collect_violations(spec_order: list[str], registry: dict) -> list[tuple[str, str, int, int]]:
    """Return list of (dependent, required, dep_pos, req_pos) where
    `dependent` appears BEFORE its required dependency."""
    pos = {pid: i for i, pid in enumerate(spec_order)}
    violations: list[tuple[str, str, int, int]] = []
    for pid, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        if pid not in pos:
            continue
        for req in (meta.get("requires_patches") or []):
            if req not in pos:
                continue
            if pos[req] >= pos[pid]:
                violations.append((pid, req, pos[pid], pos[req]))
    return violations


def test_topo_sort_yields_zero_violations():
    """With topo_sort=True, no patch is yielded before its
    requires_patches dependency."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.dispatcher.spec import iter_patch_specs
    spec_order = [s.patch_id for s in iter_patch_specs(topo_sort=True)]
    violations = _collect_violations(spec_order, PATCH_REGISTRY)
    assert not violations, (
        f"topo_sort=True yielded {len(violations)} requires_patches "
        f"order violations:\n" + "\n".join(
            f"  {pid}@{pp} requires {req}@{rp} (delta={rp-pp})"
            for pid, req, pp, rp in violations[:20]
        )
    )


def test_default_order_baseline_violations():
    """Default `iter_patch_specs()` (topo_sort=False) reproduces the
    registry-insertion order. v11.3.0 baseline known violations: 6."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.dispatcher.spec import iter_patch_specs
    spec_order = [s.patch_id for s in iter_patch_specs()]
    violations = _collect_violations(spec_order, PATCH_REGISTRY)
    # Allow growth (someone added a new requires_patches with bad
    # order — acceptable as long as topo_sort path handles it) but
    # require non-zero (sanity: at least the v11.3.0 baseline 6
    # should persist until someone reorders the dict). If this drops
    # below baseline, someone reordered registry — update baseline.
    baseline = {
        ("PN105", "PN104"),
        ("PN34", "PN33"),
        # 2026-06-17 (commit a9c9c37b): G4_75's requires_patches was
        # corrected, so ("G4_75", "G4_74") is no longer an insertion-order
        # violation and was dropped from this baseline (bug-hunt D2).
        ("G4_70", "G4_69"),
        ("PN256", "G4_67"),
        ("G4_69", "G4_60K"),
        # 2026-06-11 (preflight residual triage par.2): PN71 now
        # declares requires_patches=["P27"] — its anchor literally
        # contains P27-injected comments. P27 sits in the legacy
        # section far after PN71 in registry insertion order; live
        # boot order is owned by _per_patch_dispatch (P27 applies
        # before PN71 there) and the topo_sort path handles the edge.
        ("PN71", "P27"),
        # 2026-06-13 (50-PR sweep batch-2 wave 1): PN388 declares
        # requires_patches=["P34"] — its DUAL ANCHOR includes a
        # post-P34-shaped variant, so P34 must dispatch first. PN388
        # sits in the PN38x cohort near the top of registry insertion
        # order while P34 is in the legacy section far below; live boot
        # order is owned by _per_patch_dispatch and the topo_sort path
        # handles the edge.
        ("PN388", "P34"),
        # 2026-06-17 (bug-hunt D13/F9): PN298/PN299* gained
        # requires_patches=['PN296'] to encode the hard runtime dep on
        # PN296 (previously only in composes_with, invisible to the
        # dep-graph validator). PN296 sits at insertion index 169, after
        # the PN298/PN299* cohort (163-168), so each is an insertion-order
        # violation the topo_sort path resolves at SNDR_TOPO_SORT_SPECS=1.
        #
        # 2026-06-19 (PN29+PN298 consolidation): the merged PN298 entry no
        # longer carries requires_patches=['PN296'] at the entry level (that
        # would over-gate the version-agnostic PN29 scale-fold sub-patch; the
        # PN296 precondition lives inside the pn298 sub-patch's injected
        # code). So ("PN298","PN296") is no longer an insertion-order
        # violation and was dropped from this baseline. PN299* still declare
        # requires_patches=['PN296'] and remain below.
        ("PN299", "PN296"),
        ("PN299B", "PN296"),
        ("PN299C", "PN296"),
        ("PN299D", "PN296"),
        ("PN299E", "PN296"),
        # 2026-06-19 (PN399 dependency audit): PN399 gained
        # requires_patches=['P101'] — its const sub-patch anchors P101's
        # APPLIED output (`_CONTINUATION_DECODE_THRESHOLD = 64` +
        # `_CONTINUATION_DECODE_MAX_CACHED_LEN = 32768`; pristine has `= 128`
        # and no MAX_CACHED_LEN), so P101 must dispatch first. PN399 sits at
        # insertion index before P101 (P101 is in the TQ section far below),
        # so this is an insertion-order violation the topo_sort path resolves
        # at SNDR_TOPO_SORT_SPECS=1. Live boot order is owned by
        # _per_patch_dispatch (P101 applies before PN399 there). Same class
        # as PN388->P34 / PN71->P27 above.
        ("PN399", "P101"),
    }
    actual = {(pid, req) for pid, req, _, _ in violations}
    new_violations = sorted(actual - baseline)
    resolved = sorted(baseline - actual)
    # Allow MORE violations (anyone may add a requires_patches the
    # registry insertion order doesn't honor — topo_sort path covers
    # it). But surface added/resolved in the failure message so the
    # CI signal is actionable.
    if resolved and not new_violations:
        # Someone reordered the registry to fix baseline violations
        # without enabling topo_sort. Acceptable but worth noting.
        # Pin a warning by failing the test — forces baseline update.
        raise AssertionError(
            f"Registry insertion order changed: previously-violating "
            f"pairs now respect order: {resolved}. Update baseline."
        )
    # If new violations added — also force baseline update so audit
    # signal stays meaningful.
    if new_violations:
        raise AssertionError(
            f"New requires_patches insertion-order violations "
            f"introduced:\n  added: {new_violations}\n"
            f"  baseline (still present): {sorted(actual & baseline)}\n"
            f"Either reorder the registry entries OR update the "
            f"baseline in this test (the topo_sort path will still "
            f"apply them correctly when SNDR_TOPO_SORT_SPECS=1)."
        )


def test_topo_sort_preserves_existing_correct_order():
    """For patches with NO requires_patches OR whose deps already
    appear earlier in insertion order, topo_sort should preserve
    relative position. Sanity check that Kahn's algorithm uses
    insertion-order tie-breaking."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    from sndr.dispatcher.spec import iter_patch_specs
    default_order = [s.patch_id for s in iter_patch_specs()]
    topo_order = [s.patch_id for s in iter_patch_specs(topo_sort=True)]
    # Same set of IDs
    assert set(default_order) == set(topo_order), (
        "topo_sort dropped or added IDs vs default order"
    )
    # Most IDs should be in the same relative position
    # (only ~6-12 reorderings expected for baseline violations).
    #
    # 2026-06-11 bound update (preflight residual triage par.2): the
    # new PN71 -> P27 edge is LONG-RANGE — PN71 sits early in
    # insertion order while P27 lives in the legacy section ~134
    # entries later. Kahn defers PN71 until P27 is emitted, which
    # shifts every entry in between by one index, so the legitimate
    # move count jumped from <30 to ~134+1. Bound raised to keep the
    # tie-breaking sentinel meaningful without flagging this edge.
    moves = sum(
        1 for i, pid in enumerate(default_order)
        if topo_order.index(pid) != i
    )
    # 2026-06-15: bound 160 -> 164. PN396 + PN352B (each with composes_with
    # edges into the existing graph: PN396->PN354/PN345/PN299, PN352B->PN368/
    # P24/PN96b/P31) add a few legitimate insertion ripples. The sentinel
    # guards against a tie-breaking CHANGE, not against registry growth.
    assert moves < 164, (
        f"topo_sort moved {moves} IDs vs default — expected <164 "
        f"(baseline violations + the long-range PN71->P27 ripple "
        f"+ transitive ripples + PN396/PN352B insertions). Did the Kahn "
        f"tie-breaking change?"
    )


def test_topo_sort_no_cycle_at_baseline():
    """At v11.3.0 baseline, requires_patches DAG has no cycles. If
    this test fails, someone introduced a cycle — must resolve."""
    from sndr.dispatcher.spec import _topological_order
    from sndr.dispatcher.registry import PATCH_REGISTRY
    try:
        order = _topological_order(PATCH_REGISTRY)
    except RuntimeError as e:
        raise AssertionError(
            f"requires_patches DAG cycle introduced: {e}"
        )
    assert len(order) > 200, (
        f"topo_sort produced {len(order)} IDs, expected >200 "
        f"(v11.3.0 baseline has 241)"
    )

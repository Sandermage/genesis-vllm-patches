"""Phase 1 — anchor_discovery R1 coverage tests.

R1: discovery must enumerate ALL anchor-bearing patches (no hand-typed subset).
Discovery requires a vLLM source tree (the patchers build against it); when
vLLM is absent (local dev box) the discovery yields ~nothing and the test
skips — the authoritative R1 proof runs on the rig / CI where dev148 vLLM is
installed (see the §-server-test in the implementation plan).
"""
import pytest

from sndr.engines.vllm.anchor_discovery import (
    AnchorTarget,
    iter_anchor_targets,
    iter_specs_with_apply_module,
    _build_patcher_for_module,
)


def _discovered():
    return list(iter_anchor_targets())


def test_R1_broad_coverage():
    targets = _discovered()
    if len(targets) < 10:
        pytest.skip(
            "vLLM not installed (discovery yields nothing) — R1 is proven on "
            "the rig/CI with dev148 vLLM present"
        )
    patch_ids = {t.patch_id for t in targets}
    # Must be FAR beyond the old 4-patch hand-typed manifest subset.
    assert len(patch_ids) >= 100, (
        f"R1: only {len(patch_ids)} patches discovered, expected ~180"
    )


def test_anchor_target_shape():
    targets = _discovered()
    if not targets:
        pytest.skip("vLLM not installed — discovery empty")
    for t in targets[:30]:
        assert isinstance(t, AnchorTarget)
        assert t.patch_id and t.target_rel and t.anchor
        assert isinstance(t.required, bool)


def test_R1_no_anchor_bearing_patch_dropped():
    """Cross-check the yield logic: every spec whose patcher has >=1 anchored
    sub-patch must appear in iter_anchor_targets (catches an over-aggressive
    skip in the yield path)."""
    import importlib

    discovered = {t.patch_id for t in _discovered()}
    if not discovered:
        pytest.skip("vLLM not installed — discovery empty")
    expected = set()
    for spec in iter_specs_with_apply_module():
        try:
            mod = importlib.import_module(spec.apply_module)
        except Exception:
            continue
        patcher, _ = _build_patcher_for_module(mod)
        if patcher is None:
            continue
        if any(
            getattr(sp, "anchor", None)
            for sp in getattr(patcher, "sub_patches", []) or []
        ):
            expected.add(getattr(spec, "patch_id", "?"))
    missing = expected - discovered
    assert not missing, f"R1 VIOLATION: discovery missed {sorted(missing)}"

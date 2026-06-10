# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — PN40 spec entry's `apply_module` must
match the canonical omnibus orchestrator that the legacy
`@register_patch("PN40 DFlash drafter omnibus...")` hook actually
calls.

Why this matters
----------------

PN40 is an omnibus patch with four sub-components:
  sub-A: fused per-layer K-norm Triton kernel (DFlash only)
  sub-B: persistent K/V buffer pool (DFlash only)
  sub-C: adaptive K/N controller (universal)
  sub-D: workload classifier hook (universal)

The orchestration lives in
`sndr.engines.vllm.patches.spec_decode.pn40_dflash_omnibus`.
Its `apply()` runs sub-A + sub-B + sub-C wirings AND ALSO calls
`pn40_workload_classifier_hook.apply()` (sub-D) internally at
pn40_dflash_omnibus.py:381-382.

v11.3.0 BUG #8 discovered: the PN40 spec entry pointed
`apply_module` at the sub-D classifier hook directly. The legacy
`@register_patch("PN40 DFlash drafter omnibus...")` hook correctly
calls the omnibus orchestrator. So on the v12.0.0 default-flip from
legacy → spec-driven, PN40-enabled operators would silently lose
sub-A/B/C — a real regression of K-norm fusion, persistent pool,
and adaptive K/N control.

Same commit fixes the spec entry; this test pins the contract so
the mismatch can't return.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ regression guard.
"""
from __future__ import annotations


def test_pn40_spec_apply_module_is_omnibus_orchestrator():
    """PN40 spec.apply_module must be the canonical omnibus
    orchestrator, NOT the sub-D classifier hook directly."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    entry = PATCH_REGISTRY.get("PN40")
    assert entry is not None, "PN40 missing from PATCH_REGISTRY"
    expected = (
        "sndr.engines.vllm.patches.spec_decode.pn40_dflash_omnibus"
    )
    actual = entry.get("apply_module")
    assert actual == expected, (
        f"PN40 spec apply_module = {actual!r}, expected {expected!r}. "
        f"The omnibus orchestrator wires sub-A K-norm + sub-B pool + "
        f"sub-C adaptive K/N + sub-D classifier (the last via internal "
        f"call to pn40_workload_classifier_hook.apply at "
        f"pn40_dflash_omnibus.py:381-382). If apply_module points at "
        f"sub-D classifier directly, v12.0.0 spec-flip drops "
        f"sub-A/B/C — silent regression of DFlash K-norm fusion and "
        f"adaptive control. BUG #8 fix at v11.3.0."
    )


def test_pn40_classifier_spec_apply_module_unchanged():
    """PN40-classifier stays pointed at the classifier hook directly
    — that's the right module for the sub-D-only opt-in. Used by
    callers who want classifier without the full omnibus."""
    from sndr.dispatcher.registry import PATCH_REGISTRY
    entry = PATCH_REGISTRY.get("PN40-classifier")
    assert entry is not None, "PN40-classifier missing from PATCH_REGISTRY"
    expected = (
        "sndr.engines.vllm.patches.spec_decode."
        "pn40_workload_classifier_hook"
    )
    actual = entry.get("apply_module")
    assert actual == expected, (
        f"PN40-classifier spec apply_module = {actual!r}, expected "
        f"{expected!r}. This entry is the standalone sub-D activator; "
        f"PN40 (omnibus) covers it via internal call. If you renamed "
        f"the classifier module, update both this test and PN40's "
        f"docstring at registry.py."
    )


def test_pn40_dflash_omnibus_apply_signature_and_calls_classifier():
    """Confirm pn40_dflash_omnibus.apply() exists with the tuple
    contract AND internally invokes the sub-D classifier (i.e. flipping
    PN40's apply_module to omnibus doesn't drop sub-D coverage)."""
    import inspect
    try:
        from sndr.engines.vllm.patches.spec_decode import (
            pn40_dflash_omnibus,
        )
    except ImportError:
        # torch/vllm not available on test host — skip
        import pytest
        pytest.skip("torch/vllm unavailable; can't import omnibus module")
        return
    assert hasattr(pn40_dflash_omnibus, "apply"), (
        "pn40_dflash_omnibus.apply() missing"
    )
    src = inspect.getsource(pn40_dflash_omnibus.apply)
    assert "pn40_workload_classifier_hook" in src, (
        "omnibus.apply() no longer imports pn40_workload_classifier_hook "
        "— sub-D coverage gone. Either re-add the internal call OR "
        "split PN40 into PN40-omnibus (sub-A/B/C only) + PN40-classifier "
        "(sub-D) as independent spec entries and update this test."
    )

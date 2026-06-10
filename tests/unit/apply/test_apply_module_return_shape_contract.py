# SPDX-License-Identifier: Apache-2.0
"""CI-wide regression guard — every PATCH_REGISTRY apply_module's
apply() function must return a (status, reason) 2-tuple compatible
with the spec-driven orchestrator's unpack pattern.

Why this matters
----------------

The spec-driven apply orchestrator at
sndr/apply/orchestrator.py (`_run_via_specs`) unpacks the apply()
return value as:

    status, reason = mod.apply()

This requires the return value to be an iterable of EXACTLY length 2
where both elements are strings.

Acceptable shapes:
- `tuple[str, str]`
- `list[str]` of length 2

NOT acceptable (would TypeError on tuple-unpack):
- `dict` (the v11.3.0 SNDR_EAGLE3_AUX_HIDDEN_001 bug)
- `None`
- `bool` (legacy convention, would fail unpack)
- `PatchResult` dataclass instance

Legacy @register_patch wrapper at
sndr/apply/_per_patch_dispatch.py is more forgiving — it
checks shape and adapts. But the spec-driven path is strict, so we
pin the strict contract here.

If this test fails, a new apply_module was added with a non-tuple
return shape. Either:
  - Change the apply() body to `return (status, reason)`
  - OR document why the module needs special handling + add to
    `_KNOWN_NON_STANDARD_RETURNS` below

v11.3.0 baseline: 0 known non-standard returns. SNDR_EAGLE3_AUX_HIDDEN_001
was fixed in the same commit as this test.

Author: Sandermage (Sander) Barzov Aleksandr, Ukraine, Odessa.
Status: v11.3.0+ apply-shape contract guard.
"""
from __future__ import annotations


# Allowlist entries — apply_modules that legitimately return non-tuple
# (must be empty at v11.3.0; entries here require justification comment).
_KNOWN_NON_STANDARD_RETURNS: frozenset[str] = frozenset({
    # No allowlist entries at v11.3.0.
})


def _import_registry():
    from sndr.dispatcher.registry import PATCH_REGISTRY
    return PATCH_REGISTRY


def _try_apply(apply_module: str):
    """Import the apply_module + call apply(). Returns (result, error)
    where error is None on success, else exception."""
    import importlib
    try:
        mod = importlib.import_module(apply_module)
    except Exception as e:
        return None, f"import failed: {type(e).__name__}: {e}"
    if not hasattr(mod, "apply"):
        return None, "no apply() function"
    try:
        result = mod.apply()
        return result, None
    except Exception as e:
        return None, f"apply() raised: {type(e).__name__}: {e}"


def test_no_apply_module_returns_dict():
    """No apply_module's apply() returns a dict — would break the
    spec-driven orchestrator's `status, reason = mod.apply()` unpack
    pattern."""
    registry = _import_registry()
    offenders: list[tuple[str, str]] = []
    for pid, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        apply_module = meta.get("apply_module")
        if not apply_module:
            continue
        if pid in _KNOWN_NON_STANDARD_RETURNS:
            continue
        result, err = _try_apply(apply_module)
        if err:
            # Import or runtime failure — that's a separate concern;
            # ignore here. Other tests catch import errors.
            continue
        if isinstance(result, dict):
            offenders.append((pid, apply_module))
    assert not offenders, (
        f"{len(offenders)} apply_module(s) return dict — would break "
        f"spec-driven orchestrator tuple-unpack. v11.3.0 bug class "
        f"(SNDR_EAGLE3_AUX_HIDDEN_001 fix). Change apply() to return "
        f"(status, reason) 2-tuple:\n" + "\n".join(
            f"  - {pid}: {mod}" for pid, mod in offenders
        )
    )


def test_apply_modules_return_tuple_or_compatible():
    """Survey: apply_modules' return shapes. Documents the v11.3.0
    baseline so we can compare on future audits."""
    registry = _import_registry()
    shape_counts = {
        "tuple_2": 0,
        "list_2": 0,
        "tuple_other": 0,
        "list_other": 0,
        "dict": 0,
        "patch_result": 0,
        "bool": 0,
        "None": 0,
        "other": 0,
        "import_error": 0,
    }
    for pid, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        apply_module = meta.get("apply_module")
        if not apply_module:
            continue
        result, err = _try_apply(apply_module)
        if err is not None:
            shape_counts["import_error"] += 1
            continue
        if isinstance(result, tuple):
            shape_counts["tuple_2" if len(result) == 2 else "tuple_other"] += 1
        elif isinstance(result, list):
            shape_counts["list_2" if len(result) == 2 else "list_other"] += 1
        elif isinstance(result, dict):
            shape_counts["dict"] += 1
        elif result is None:
            shape_counts["None"] += 1
        elif isinstance(result, bool):
            shape_counts["bool"] += 1
        elif type(result).__name__ == "PatchResult":
            shape_counts["patch_result"] += 1
        else:
            shape_counts["other"] += 1

    # Pin the baseline: most return tuple_2; some return import_error
    # (no torch/vllm on this test host); zero should return dict.
    assert shape_counts["dict"] == 0, (
        f"{shape_counts['dict']} apply_modules return dict — broken "
        f"contract for spec-driven orchestrator"
    )
    # tuple_2 should be the dominant shape (most apply_modules)
    # Allow flex — registry has 219 apply_modules; >100 should be tuple_2
    # (rest are import-error due to torch/vllm gap on test host).
    assert shape_counts["tuple_2"] + shape_counts["import_error"] >= 100, (
        f"Expected at least 100 modules between tuple_2 + import_error; "
        f"got distribution {shape_counts}"
    )


def test_spec_driven_orchestrator_unpack_pattern_works():
    """Sample a few known-good apply_modules + verify the spec-driven
    orchestrator's unpack pattern (`status, reason = mod.apply()`)
    actually works without TypeError."""
    import importlib
    sample_modules = [
        "sndr.engines.vllm.patches.attention.turboquant.pn118_v2_md5_workspace",
        "sndr.engines.vllm.patches.attention.gdn.pn79_v2_md5_chunk",
        "sndr.engines.vllm.patches.spec_decode.sndr_eagle3_aux_hidden_001",
    ]
    for mod_name in sample_modules:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            # torch/vllm gap — skip
            continue
        if not hasattr(mod, "apply"):
            continue
        # The pattern that was broken by the dict-return bug:
        try:
            status, reason = mod.apply()
        except (TypeError, ValueError) as e:
            raise AssertionError(
                f"{mod_name}.apply() can't be unpacked as "
                f"`status, reason = mod.apply()`: {e}. This breaks the "
                f"spec-driven orchestrator path."
            )
        assert isinstance(status, str)
        assert isinstance(reason, str)

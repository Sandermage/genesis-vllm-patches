# SPDX-License-Identifier: Apache-2.0
"""Test-session setup for `tests/unit/infra/` — STABLE-ratchet support.

The STABLE ratchet test (`test_stable_manifest_policy.py`) asserts that
every `lifecycle="stable"` patch has its TextPatcher registered in
`wiring/patcher_registry.py::iter_registered_patchers()`. Wiring modules
register via `register_for_manifest(pristine_root=...)` rather than at
import time — same mechanism `scripts/build_anchor_manifest.py` uses to
build the on-disk manifest.

This conftest invokes those `register_for_manifest()` calls once per
test session so the ratchet test sees the same patcher set as the
build script. Without this, the registry is empty and any STABLE
patch fails the registration check vacuously.

Discovery list mirrors `scripts/build_anchor_manifest.py::_trigger_patcher_registration`.
"""
from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PRISTINE_ROOT = REPO_ROOT / "tests" / "legacy" / "pristine_fixtures"


# Mirror of `scripts/build_anchor_manifest.py::_REGISTRY_TARGETS`.
# Keep in lockstep — they are two sides of the same contract.
_STABLE_REGISTRY_TARGETS = [
    ("PN79", "sndr.engines.vllm.patches.attention.gdn"
             ".pn79_inplace_ssm_state"),
    ("PN35", "sndr.engines.vllm.patches.worker"
             ".pn35_inputs_embeds_optional"),
    ("PN33", "sndr.engines.vllm.patches.worker"
             ".pn33_spec_decode_warmup_k"),
    # Added 2026-05-28 (STAGE-6-HARDENING.1): G4_04 graduates from
    # _MANIFEST_TRACKING_DEFERRED to first-class manifest coverage now
    # that the pristine gemma4.py fixture is in place.
    ("G4_04", "sndr.engines.vllm.patches.model_compat.gemma4"
              ".g4_04_gemma4_awq_moe_keys_remap"),
]


@pytest.fixture(autouse=True)
def _register_stable_patchers_for_each_infra_test(request):
    """Invoke `register_for_manifest(pristine_root=...)` for each STABLE-
    eligible wiring module so `iter_registered_patchers()` returns the
    same set the build script would produce.

    Function-scoped autouse — runs before every test in this directory
    tree because `test_anchor_manifest.py` has its own autouse
    `_clear_registry` that wipes the global registry between its tests.
    A session-scoped fixture would be cleared by that teardown before
    `test_stable_manifest_policy.py` runs.

    Test files that want a clean empty registry (the existing
    `test_anchor_manifest.py` tests) declare their own `_clear_registry`
    fixture which fires AFTER this one and wipes our seed — that
    preserves their isolation while still letting the STABLE ratchet
    test see the registered set.
    """
    if not PRISTINE_ROOT.is_dir():
        # No fixtures — tests that need them will fail on their own with
        # a clearer message than what we'd raise here.
        yield
        return

    # Idempotent ish: `register_text_patcher` raises ValueError if a
    # different patcher object is registered for the same id. Clear
    # first to avoid that on subsequent test runs where this fixture
    # fires repeatedly.
    try:
        from sndr.engines.vllm.wiring.patcher_registry import clear_registry
        clear_registry()
    except Exception:
        pass

    for pid, mod_path in _STABLE_REGISTRY_TARGETS:
        try:
            mod = __import__(mod_path, fromlist=["register_for_manifest"])
        except ImportError:
            # Module not present in this environment (e.g. patch removed).
            # Skip silently — the ratchet test enforces presence separately.
            continue
        register_fn = getattr(mod, "register_for_manifest", None)
        if register_fn is None:
            # Module has no build-mode entry point. Skip — the ratchet
            # test will surface this as a missing-patcher violation.
            continue
        try:
            register_fn(pristine_root=PRISTINE_ROOT)
        except Exception:
            # Don't fail collection — the ratchet test will surface the
            # specific patch_id missing from the registry.
            pass

    yield

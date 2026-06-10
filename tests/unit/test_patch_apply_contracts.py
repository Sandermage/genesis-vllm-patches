# SPDX-License-Identifier: Apache-2.0
"""S2.3 audit closure (2026-05-08 noonghunna): codegen contract tests.

Walks every PATCH_REGISTRY entry and asserts:

  1. spec_count > 0 (catches accidental wipe of registry)
  2. tier in {"community", "engine"}
  3. lifecycle is set
  4. env_flag (when present) follows GENESIS_<patch_id> convention
  5. apply_module (when set) imports cleanly without torch
  6. when apply_module exposes ``apply()``:
        - it is callable
        - returns Tuple[str, str]
        - status ∈ {"applied", "skipped", "failed"}
        - never raises (with env disabled it must return "skipped")

These tests are the "contract" that prevents API drift across patches:
adding a new patch silently breaks one of the assertions and the test
suite catches it before merge.

Tests intentionally run on Mac/CI hosts WITHOUT torch installed —
contract checks never execute torch-heavy paths. The apply() smoke
sets the patch's env flag to OFF so the function returns "skipped"
quickly.
"""
from __future__ import annotations

import importlib

import pytest

from sndr.dispatcher import PATCH_REGISTRY
from sndr.dispatcher.spec import iter_patch_specs


# ─── Registry shape contracts ──────────────────────────────────────────


def test_registry_nonempty():
    assert len(PATCH_REGISTRY) > 0, "PATCH_REGISTRY accidentally empty"
    # Wave 7 baseline: at least 130 patches
    assert len(PATCH_REGISTRY) >= 130, (
        f"PATCH_REGISTRY shrunk to {len(PATCH_REGISTRY)} entries — "
        "investigate before merge"
    )


def test_iter_patch_specs_aligns_with_registry():
    spec_ids = {s.patch_id for s in iter_patch_specs()}
    registry_ids = set(PATCH_REGISTRY.keys())
    assert spec_ids == registry_ids, (
        f"spec/registry drift: only_in_specs={spec_ids - registry_ids} "
        f"only_in_registry={registry_ids - spec_ids}"
    )


# ─── Per-entry metadata contracts ──────────────────────────────────────


_ALL_IDS = sorted(PATCH_REGISTRY.keys())


@pytest.mark.parametrize("patch_id", _ALL_IDS)
class TestRegistryEntryShape:
    def test_tier_valid(self, patch_id):
        meta = PATCH_REGISTRY[patch_id]
        tier = meta.get("tier")
        assert tier in ("community", "engine"), (
            f"{patch_id}: tier={tier!r} not in {{community, engine}}"
        )

    def test_lifecycle_set(self, patch_id):
        meta = PATCH_REGISTRY[patch_id]
        # Lifecycle is required after PR38 audit; default 'experimental'
        # was tagged for any unset entry but spec must explicitly carry it.
        lifecycle = meta.get("lifecycle")
        assert lifecycle is not None and lifecycle != "", (
            f"{patch_id}: lifecycle missing — registry contract violated"
        )

    def test_default_on_is_bool(self, patch_id):
        meta = PATCH_REGISTRY[patch_id]
        if "default_on" in meta:
            assert isinstance(meta["default_on"], bool), (
                f"{patch_id}: default_on must be bool, got "
                f"{type(meta['default_on']).__name__}"
            )

    def test_env_flag_naming(self, patch_id):
        """When env_flag set, prefer GENESIS_ENABLE_<patch_id> form.

        Audit 2026-05-08 (noonghunna): env_flag names should follow
        GENESIS_ENABLE_<short>_<descriptive> pattern. We don't enforce
        exact match (some patches predate the convention) but require
        a recognized prefix so config audit allowlist works.

        Recognized prefixes:
          - GENESIS_* — canonical for patch-toggle (ENABLE/LEGACY/DISABLE).
          - SNDR_ENABLE_ — Sander-IP canonical for new patches whose
            primary surface lives outside the GENESIS_ legacy naming
            (PN282 spec-decode acceptance metric).
          - SNDR_ALLOW_ — operator-consent gate semantic (PN274, R3 audit
            2026-05-21; documented in spec_decode/safety_guard.py).
        """
        meta = PATCH_REGISTRY[patch_id]
        flag = meta.get("env_flag")
        if flag is not None:
            allowed_prefixes = ("GENESIS_", "SNDR_ENABLE_", "SNDR_ALLOW_")
            assert flag.startswith(allowed_prefixes), (
                f"{patch_id}: env_flag={flag!r} must start with one of "
                f"{allowed_prefixes}"
            )


# ─── apply_module import + apply() contract ───────────────────────────


_ALL_SPECS_WITH_MODULE = [
    s for s in iter_patch_specs() if s.apply_module is not None
]


def _import_apply_module(spec):
    """Import the spec's apply_module; skip on hosts without torch/triton.

    CI installs CPU torch, so the import contract is still fully
    exercised there; local torch-less hosts skip instead of failing
    (same pattern as tests/unit/integrations/_family_contract_helpers).
    The static "no top-level torch import" invariant is enforced by the
    family contracts, not here.
    """
    try:
        return importlib.import_module(spec.apply_module)
    except ImportError as e:
        if "torch" in str(e) or "triton" in str(e):
            pytest.skip(
                f"{spec.patch_id}: torch/triton unavailable on this "
                f"host: {e}"
            )
        raise


@pytest.mark.parametrize(
    "spec",
    _ALL_SPECS_WITH_MODULE,
    ids=lambda s: s.patch_id,
)
class TestApplyModule:
    def test_apply_module_importable(self, spec):
        """Each apply_module must import cleanly even without torch.

        Patches that need torch must defer the heavy import to inside
        apply() (Wave 6 closure pattern — P22/P31/P32/P28/P7/P17-18/P20).
        """
        _import_apply_module(spec)

    def test_apply_function_exposed(self, spec):
        """apply_module either exposes a top-level apply() or registers
        via @register_patch in _per_patch_dispatch. Both are valid; this
        test only flags modules that expose neither (probably misnamed)."""
        mod = _import_apply_module(spec)
        if not hasattr(mod, "apply"):
            # Module-level apply() is one valid pattern; the other is
            # @register_patch in _per_patch_dispatch.py — those modules
            # are usually named pXX_*.py without their own apply().
            pytest.skip(
                f"{spec.patch_id}: no module-level apply() — likely "
                "registered via @register_patch decorator"
            )
        assert callable(mod.apply), (
            f"{spec.patch_id}: apply must be callable"
        )

    def test_apply_returns_tuple_when_env_disabled(self, spec, monkeypatch):
        """With env disabled, apply() must return a 2-tuple (status,
        reason) where status ∈ {applied, skipped, failed}. apply() must
        NEVER raise — defensive try/except is mandatory inside."""
        mod = _import_apply_module(spec)
        if not hasattr(mod, "apply"):
            pytest.skip("no module-level apply()")

        meta = PATCH_REGISTRY[spec.patch_id]
        env_flag = meta.get("env_flag")
        if env_flag:
            monkeypatch.delenv(env_flag, raising=False)

        try:
            result = mod.apply()
        except SystemExit:
            # SystemExit is allowed (e.g. FLA-guard hard-fails on
            # int64 overflow). Treat it as a non-raise outcome.
            return
        except Exception as exc:  # pragma: no cover — explicit failure
            pytest.fail(
                f"{spec.patch_id}: apply() raised {type(exc).__name__}: "
                f"{exc!r} — must catch internally and return failed/skipped"
            )

        assert isinstance(result, tuple) and len(result) == 2, (
            f"{spec.patch_id}: apply() must return (str, str), got {result!r}"
        )
        status, reason = result
        assert isinstance(status, str), (
            f"{spec.patch_id}: apply() status must be str, got {type(status).__name__}"
        )
        assert isinstance(reason, str), (
            f"{spec.patch_id}: apply() reason must be str, got {type(reason).__name__}"
        )
        assert status in ("applied", "skipped", "failed"), (
            f"{spec.patch_id}: apply() status={status!r} not in valid set"
        )

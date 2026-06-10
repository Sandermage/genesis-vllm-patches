# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.dispatcher._apply_module_overlay` — Entry 12.

Contract:

  1. APPLY_MODULE_OVERLAY is a non-empty dict of patch_id → module path.
  2. All overlay values point at sndr.apply._per_patch_dispatch
     (legacy monolithic dispatcher) by design.
  3. apply_overlay returns the count of entries patched.
  4. apply_overlay NEVER overwrites an explicit apply_module already
     set in the registry.
  5. apply_overlay skips entries with an integration-tree module on
     disk (Stage 6 migrated patches).
  6. apply_overlay silently ignores patch_ids not in the registry.
  7. apply_overlay mutates the registry in-place.
"""
from __future__ import annotations

import pytest

from sndr.dispatcher import _apply_module_overlay as ovl


# ─── Overlay shape ─────────────────────────────────────────────────────


class TestOverlayShape:
    def test_overlay_is_dict(self):
        assert isinstance(ovl.APPLY_MODULE_OVERLAY, dict)

    def test_overlay_non_empty(self):
        assert len(ovl.APPLY_MODULE_OVERLAY) > 0

    def test_keys_are_strings(self):
        for k in ovl.APPLY_MODULE_OVERLAY:
            assert isinstance(k, str) and k

    def test_all_values_point_at_legacy_dispatcher(self):
        """By design (Entry 12 metadata gap closure), every overlay entry
        names the monolithic per_patch_dispatch module."""
        canonical = "sndr.apply._per_patch_dispatch"
        for v in ovl.APPLY_MODULE_OVERLAY.values():
            assert v == canonical


# ─── apply_overlay behavior ────────────────────────────────────────────


class TestApplyOverlayFill:
    def test_fills_missing_apply_module(self):
        # Use an overlay key that we know IS in the overlay
        sample_pid = next(iter(ovl.APPLY_MODULE_OVERLAY))
        # Build a fake registry with that pid having no apply_module
        fake = {sample_pid: {"tier": "community"}}
        patched = ovl.apply_overlay(fake)
        # Skip if the patch has an integration tree module on disk
        # (apply_overlay defers to spec.py in that case)
        if ovl._has_integration_tree_module(sample_pid):
            assert patched == 0
            assert "apply_module" not in fake[sample_pid]
        else:
            assert patched == 1
            assert fake[sample_pid]["apply_module"] == (
                "sndr.apply._per_patch_dispatch"
            )

    def test_returns_zero_on_empty_registry(self):
        assert ovl.apply_overlay({}) == 0

    def test_returns_zero_when_no_overlap(self):
        """Registry contains a patch_id NOT in the overlay → 0 patches applied."""
        fake = {"NEVER_IN_OVERLAY_XYZ": {"tier": "community"}}
        assert ovl.apply_overlay(fake) == 0


class TestApplyOverlayNoOverwrite:
    def test_explicit_apply_module_preserved(self):
        """Patches that ALREADY declare apply_module are left untouched."""
        sample_pid = next(iter(ovl.APPLY_MODULE_OVERLAY))
        custom_module = "sndr.engines.vllm.patches.custom.my_module"
        fake = {sample_pid: {"tier": "community", "apply_module": custom_module}}
        ovl.apply_overlay(fake)
        assert fake[sample_pid]["apply_module"] == custom_module

    def test_empty_string_apply_module_treated_as_missing(self):
        """Empty string apply_module is filled (falsy)."""
        # Find an overlay patch that DOESN'T have an on-disk integration
        # module, so apply_overlay will fill it.
        sample_pid = None
        for pid in ovl.APPLY_MODULE_OVERLAY:
            if not ovl._has_integration_tree_module(pid):
                sample_pid = pid
                break
        if sample_pid is None:
            pytest.skip("no overlay patch without integration-tree module")
        fake = {sample_pid: {"apply_module": ""}}
        patched = ovl.apply_overlay(fake)
        assert patched == 1
        assert fake[sample_pid]["apply_module"] == (
            "sndr.apply._per_patch_dispatch"
        )


class TestApplyOverlayIntegrationTree:
    def test_skips_patches_with_integration_module(self):
        """Patches with an on-disk integration/<family>/<patch>*.py file
        are skipped so spec.py can fill in the correct path."""
        # Find one overlay patch that HAS an integration module.
        with_module = [
            pid for pid in ovl.APPLY_MODULE_OVERLAY
            if ovl._has_integration_tree_module(pid)
        ]
        if not with_module:
            pytest.skip("no overlay patches have integration-tree modules")
        sample_pid = with_module[0]
        fake = {sample_pid: {"tier": "community"}}
        patched = ovl.apply_overlay(fake)
        assert patched == 0
        assert "apply_module" not in fake[sample_pid]


class TestApplyOverlayUnknownPatch:
    def test_unknown_overlay_patches_silently_ignored(self):
        """Patches present in the overlay but absent from the registry
        do not raise — Stage 6 might retire a patch while overlay still
        carries it."""
        # No-op if the registry doesn't have an entry.
        # apply_overlay does nothing for entries not in the registry.
        sample_pid = next(iter(ovl.APPLY_MODULE_OVERLAY))
        # Empty registry → 0 patches applied (no error raised)
        assert ovl.apply_overlay({}) == 0


# ─── Live registry merge regression anchor ────────────────────────────


class TestLiveRegistryMerge:
    def test_live_registry_supports_apply_overlay(self):
        """Apply the overlay to a COPY of the live PATCH_REGISTRY and
        verify it patches at least one entry. Catches the regression
        where the overlay shape no longer matches registry shape."""
        from sndr.dispatcher.registry import PATCH_REGISTRY
        # Deep-copy each entry so we don't mutate the real registry
        snapshot = {
            pid: dict(meta) if isinstance(meta, dict) else meta
            for pid, meta in PATCH_REGISTRY.items()
        }
        # Strip apply_module from every entry so we test the fill path
        for entry in snapshot.values():
            if isinstance(entry, dict):
                entry.pop("apply_module", None)
        patched = ovl.apply_overlay(snapshot)
        # At least some patches should be filled (overlay is non-empty)
        assert patched > 0

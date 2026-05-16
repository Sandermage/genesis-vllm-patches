# SPDX-License-Identifier: Apache-2.0
"""Tests for ``vllm.sndr_core.proof.production_subset``.

The subset definition is what hardened-release policy uses to scope
``require-bench`` / ``require-baseline`` gating. Drift here silently
expands or contracts the production bench surface, so the contract is
covered by both synthetic-registry cases (deterministic) and a live
smoke check against the real registry."""
from __future__ import annotations

import pytest

from vllm.sndr_core.proof.production_subset import (
    PRODUCTION_PRESET_PATTERN,
    get_production_subset,
    production_subset_breakdown,
)


class TestSubsetSynthetic:
    """Pure-function tests with a synthetic registry — no V2 presets
    are visited (the preset-walker silently returns an empty set when
    no presets resolve, so only default_on flows through)."""

    def test_default_on_alone_forms_subset(self):
        synth = {
            "P_A": {"env_flag": "GENESIS_ENABLE_A", "default_on": True},
            "P_B": {"env_flag": "GENESIS_ENABLE_B", "default_on": False},
            "P_C": {"env_flag": "GENESIS_ENABLE_C"},  # default_on absent
        }
        s = get_production_subset(synth)
        # Without preset coverage, only default_on=True survives.
        # Real prod presets will widen this, but this is the floor.
        assert "P_A" in s

    def test_subset_is_frozenset(self):
        synth = {"P_X": {"env_flag": "GENESIS_ENABLE_X", "default_on": True}}
        s = get_production_subset(synth)
        assert isinstance(s, frozenset)
        with pytest.raises((AttributeError, TypeError)):
            s.add("rogue")  # type: ignore[attr-defined]

    def test_non_dict_entries_ignored(self):
        synth = {
            "P_OK": {"env_flag": "GENESIS_ENABLE_OK", "default_on": True},
            "P_BAD": "not-a-dict",  # garbage entry, must be skipped
        }
        s = get_production_subset(synth)
        assert "P_OK" in s
        assert "P_BAD" not in s

    def test_missing_env_flag_still_default_on(self):
        """A registry entry without ``env_flag`` can still join the
        subset via ``default_on=True``."""
        synth = {
            "P_NOENV": {"default_on": True},  # no env_flag at all
        }
        s = get_production_subset(synth)
        assert "P_NOENV" in s


class TestLivePresets:
    """Smoke checks against the actual production presets."""

    def test_live_subset_is_nonempty(self):
        subset = get_production_subset()
        # The live tree always carries 8 prod-* presets + multiple
        # default_on=True patches — subset must be substantial.
        assert len(subset) > 20, (
            f"production subset suspiciously small: {len(subset)}"
        )

    def test_live_subset_smaller_than_registry(self):
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        subset = get_production_subset()
        # Subset never exceeds full registry; if it equals registry,
        # the boundary collapsed and the gate degenerates to
        # require-bench-on-everything (which is exactly what we
        # avoided by introducing the subset).
        assert len(subset) <= len(PATCH_REGISTRY)
        assert len(subset) < len(PATCH_REGISTRY) or len(PATCH_REGISTRY) == 0

    def test_breakdown_includes_known_prod_presets(self):
        b = production_subset_breakdown()
        matched = set(b["presets_matched"])
        for expected in (
            "prod-27b-dflash-multiconc",
            "prod-35b",
            "prod-35b-multiconc",
        ):
            assert expected in matched, (
                f"prod-presets walker missed expected alias {expected!r}; "
                f"matched: {sorted(matched)}"
            )

    def test_default_on_subset_of_full_subset(self):
        b = production_subset_breakdown()
        for pid in b["default_on"]:
            assert pid in b["subset"], (
                f"default_on patch {pid!r} missing from subset"
            )

    def test_pattern_constant_is_glob_form(self):
        for p in PRODUCTION_PRESET_PATTERN:
            assert "*" in p or "?" in p or "/" not in p, (
                f"pattern {p!r} should be a glob, not a literal alias"
            )


class TestPN26bResearchExclusion:
    """Audit R-01 cross-check: research-lifecycle patches should
    surface in the production subset only via explicit preset opt-in
    (or default_on=True), never via lifecycle alone. PN26b is
    research, default_on=False, and not enabled in any prod preset →
    it should be OUT of the subset."""

    def test_pn26b_not_in_default_on_subset(self):
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        meta = PATCH_REGISTRY.get("PN26b") or {}
        # Assumes PN26b stays research + default_on=False. If a future
        # operator promotes PN26b to default_on=True, this test
        # forces an explicit review of the production-subset shift.
        if str(meta.get("lifecycle")).lower() != "research":
            pytest.skip("PN26b lifecycle changed — re-evaluate the audit")
        if meta.get("default_on") is True:
            pytest.skip("PN26b promoted to default_on=True — "
                        "re-evaluate the production subset boundary")
        b = production_subset_breakdown()
        # Lifecycle-research entries don't enter the subset by default.
        # If PN26b appears, some prod preset enabled it — that is a
        # config decision the test surfaces but does not block.
        in_any_prod_preset = any(
            "PN26b" in pids for pids in b["per_preset"].values()
        )
        if "PN26b" in b["subset"]:
            assert in_any_prod_preset, (
                "PN26b in subset but no prod preset enables it — "
                "inclusion must come from preset opt-in, not lifecycle"
            )

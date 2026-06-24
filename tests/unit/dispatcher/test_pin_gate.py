# SPDX-License-Identifier: Apache-2.0
"""Pin-gate integration tests — Phase 2 (2026-05-11).

Background: pin-gate infrastructure existed in repo since Sander 2026-05-04
(`detection.guards.assert_vllm_pin_allowed` + `compat.version_check.
check_version_constraints`), wired into `dispatcher/decision.py:117-140`
to honor `applies_to.vllm_version_range`. But 0 patches actually used
it as of 2026-05-11 — gap closed by this test + PN90 reference adoption.

What this test verifies:
1. `check_version_constraints` correctly accepts in-range version.
2. Returns (False, "...") with "violates" reason for out-of-range version.
3. Conservative-pass (None reason) when vllm version undetectable.
4. PN90's declared range admits the current pin baseline (dev93+g51f22dcfd)
   and rejects pre-dev9 versions.
5. KNOWN_GOOD_VLLM_PINS contains exactly the 4 validated entries
   (drift detector — fails loudly if someone bumps without test update).
6. PEP 440 prerelease semantics work (dev versions match).

Adding a new pin to KNOWN_GOOD_VLLM_PINS? Update `EXPECTED_PINS` below.
Adding `vllm_version_range` to a new patch? Drop a smoke-case here
mirroring the PN90 pattern.
"""
from __future__ import annotations

import pytest

from sndr.compat.version_check import (
    VersionProfile,
    check_version_constraints,
)
from sndr.engines.vllm.detection.guards import KNOWN_GOOD_VLLM_PINS


# ─── Expected allowlist (drift detector) ──────────────────────────────────
# Update when a pin is added/removed from KNOWN_GOOD_VLLM_PINS.
EXPECTED_PINS = (
    "0.20.1rc1.dev16+g7a1eb8ac2",      # v7.65 PROD baseline
    "0.20.2rc1.dev9+g01d4d1ad3",       # v7.70 pin-bump target
    "0.20.2rc1.dev60+ge47c98ef7",      # 2026-05-07 candidate (Sander PR #39931)
    "0.20.2rc1.dev93+g51f22dcfd",      # Wave 8 PROD (27B 132.28, 35B 232.36)
    "0.20.2rc1.dev209+g5536fc0c0",     # 2026-05-11 Phase 2 bump (27B 131.11, -0.88% net-neutral)
    "0.20.2rc1.dev338+gbf0d2dc6d",     # 2026-05-14 Wave 9 PROD baseline (35B 216.02 sustained, 27B 130.76 — within CV target ≥220/≥130)
    "0.21.0",                          # 2026-05-15 v0.21.0 release tag (PROMOTION_PENDING)
    "0.21.1rc0",                       # 2026-05-15 v0.21.1rc0 git tag form (PROMOTION_PENDING)
    "0.21.1rc0+gd735968f6d63",         # 2026-05-15 v0.21.1rc0 canonical dev-pin form (PROMOTION_PENDING)
    "0.21.1rc0+gbf610c2f5676",         # 2026-05-15 docker hub nightly SHA — bench validated dev371 +1.76% vs dev338
    "0.20.2rc1.dev371+gbf610c2f5",     # 2026-05-15 — real version string vllm reports for nightly-bf610c2f image
    # K.1.R 2026-05-28 pin bump target — vllm/vllm-openai:nightly @
    # sha256:674922aae790c2cbf45f4e844098d227b80d40a74bfc7797a444d213a221879f
    # = upstream 626fa9bba5663a5cf6a870debf031ee344ddb822 (2026-05-28T04:59:34Z).
    # PROMOTED 2026-05-30 (K.1.R.R.8.5) — 35B FP8 dense bench 195.74 TPS /
    # 6-7 tool-call with P67; PN286 default_on flip validated (+6.6% TPS).
    "0.21.1rc0+g626fa9bba5",                            # setuptools_scm-derived (closest tag base)
    "0.21.1rc0+g626fa9bba566",                          # setuptools_scm 12-char SHA form
    "nightly-626fa9bba5663a5cf6a870debf031ee344ddb822", # docker tag form
    "0.20.2rc1.dev733+g626fa9bba5",                     # internal-version derived (confirmed at K.1.R.R.4 first boot)
    # K.2 pin bump 2026-06-05 — jump through v0.22.0 + v0.22.1 majors.
    # Upstream da1daf40bf18e5eaae04f26a80a537c8168a8bc2 (#44571). 27B boot
    # smoke PASS; PROMOTION PENDING full bench validation.
    "0.22.1rc1.dev195+gda1daf40b",                       # setuptools_scm-derived (closest tag base)
    "0.22.1rc1.dev195+gda1daf40bf18e5eaae04f26a80a537c8168a8bc2",  # full-SHA form
    "nightly-da1daf40bf18e5eaae04f26a80a537c8168a8bc2",  # docker tag form
    # PROD pin K.1.R.R.8.5 ratified 2026-06-09 — image nightly-303916e93.
    # Sustained bench: wall_TPS 218.56, TPOT 4.464 ms, CV 0.41%.
    "0.22.1rc1.dev259+g303916e93",                       # setuptools_scm-derived
    "0.22.1rc1.dev259+g303916e93d66",                    # 12-char SHA form
    "nightly-303916e93",                                 # docker tag form (short)
    "nightly-303916e93d66",                              # docker tag form (12-char)
    # PROD pin PROMOTED 2026-06-14 — image nightly-1033ffac2. dev491 is the
    # validated streaming-tool-call pin (#45171 remapped qwen3_xml ->
    # Qwen3CoderToolParser; version-gate caps the dev259-era P64/P61c/PN56
    # streaming wraps + PN347 MarlinFP8 fix <dev491). Validated this session:
    # streaming tool-calls (args complete, no XML leak) + chat-matrix
    # (thinking_off 254 TPS) + Batch A perf config. dev259 retained as rollback.
    "0.22.1rc1.dev491+g1033ffac2",                       # setuptools_scm-derived
    "0.22.1rc1.dev491+g1033ffac2d66",                    # 12-char SHA form
    "nightly-1033ffac2",                                 # docker tag form
    # PROD pin PROMOTED 2026-06-17 — image nightly-4c626633... (0.23.1rc1.dev101
    # +g4c6266331). Full-fleet validation: 35B/27B/Gemma4-31B/DiffusionGemma all
    # apply failed=0 + smoke + tool-call PASS; 35B bench 210.7 TPS = 101% of
    # dev491. MTP fixed via P67 cap <0.24.0. dev491 retained above as rollback.
    "0.23.1rc1.dev101+g4c6266331",                       # setuptools_scm-derived
    "0.23.1rc1.dev101+g4c626633159",                     # 12-char SHA form
    "nightly-4c626633",                                  # short docker tag
    "nightly-4c626633159887b0f2c962058c17c78f1434556d",  # full docker tag
    # PROD pin PROMOTED 2026-06-19 — image nightly-b4c80ec0f (0.23.1rc1.dev148
    # +gb4c80ec0f). Live rig pin: 35B PROD runs it; all K=5 + Gemma kv-auto
    # work validated on it. 35B MTP K=5 239.7 TPS / 27B 127.4 TPS; PN394/PN399
    # promoted to PROD, PN353A enabled in Qwen YAMLs. dev101 retained as
    # previous/rollback per CLAUDE.md ≤2-pin policy.
    "0.23.1rc1.dev148+gb4c80ec0f",                       # setuptools_scm-derived
    "nightly-b4c80ec0f",                                 # docker tag form (short)
    # PROD pin PROMOTED 2026-06-24 — image nightly-04c2a8dea (0.23.1rc1.dev301
    # +g04c2a8dea). Bump from dev148. 35B 208 TPS + 31B 94.7 TPS boot+chat+
    # tool-call validated. dev301 anchor-SOT regen surfaced 5 anchor_drift
    # (P85/PN394/PN353A/PN400/PN382): PN394/PN400 retired on dev301,
    # PN353A/PN382 kept+re-anchored, P85 OFF. dev148 retained as
    # previous/rollback per CLAUDE.md ≤2-pin policy.
    "0.23.1rc1.dev301+g04c2a8dea",                       # setuptools_scm-derived
    "nightly-04c2a8dea",                                 # docker tag form (short)
)


def _profile(vllm_version: str | None) -> VersionProfile:
    """Build a VersionProfile with only vllm set — torch/triton/etc None."""
    return VersionProfile(vllm=vllm_version)


# ─── KNOWN_GOOD_VLLM_PINS allowlist drift ─────────────────────────────────


class TestKnownGoodPinsAllowlist:
    """Allowlist drift detector — every bump must update EXPECTED_PINS."""

    def test_allowlist_matches_expected(self):
        """KNOWN_GOOD_VLLM_PINS must equal EXPECTED_PINS (drift trap)."""
        assert KNOWN_GOOD_VLLM_PINS == EXPECTED_PINS, (
            "KNOWN_GOOD_VLLM_PINS drifted from EXPECTED_PINS. "
            "Update test EXPECTED_PINS when adding a validated pin, OR "
            "verify allowlist edit was intentional."
        )

    def test_allowlist_no_placeholder_entries(self):
        """No '??' / 'PENDING' / empty strings — only real pin specs."""
        for pin in KNOWN_GOOD_VLLM_PINS:
            assert "?" not in pin, f"placeholder ?? in pin: {pin}"
            assert "PENDING" not in pin, f"PENDING in pin: {pin}"
            assert pin.strip() == pin and pin, f"malformed pin: {pin!r}"


# ─── check_version_constraints smoke tests ────────────────────────────────


class TestVllmVersionRangeGate:
    """Direct unit tests on check_version_constraints semantics."""

    def test_in_range_passes(self):
        """vllm in declared range → all_ok=True."""
        constraints = {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")}
        ok, results = check_version_constraints(
            constraints, profile=_profile("0.20.2rc1.dev93+g51f22dcfd"),
        )
        assert ok is True
        r = next(r for r in results if r.key == "vllm_version_range")
        assert r.matched is True
        assert "satisfies" in r.reason

    def test_below_min_fails(self):
        """vllm below min → all_ok=False with violation reason."""
        constraints = {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")}
        ok, results = check_version_constraints(
            constraints, profile=_profile("0.20.1rc1.dev16+g7a1eb8ac2"),
        )
        assert ok is False
        r = next(r for r in results if r.key == "vllm_version_range")
        assert r.matched is False
        assert "violates" in r.reason

    def test_above_max_fails(self):
        """vllm above max → all_ok=False."""
        constraints = {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")}
        ok, results = check_version_constraints(
            constraints, profile=_profile("0.21.0"),
        )
        assert ok is False

    def test_undetectable_is_conservative_pass(self):
        """vllm=None (detection failed) → all_ok stays True (conservative)."""
        constraints = {"vllm_version_range": (">=0.20.2rc1.dev9", "<0.21.0")}
        ok, results = check_version_constraints(
            constraints, profile=_profile(None),
        )
        assert ok is True
        r = next(r for r in results if r.key == "vllm_version_range")
        assert r.matched is None
        assert "conservative" in r.reason

    def test_single_string_spec_accepted(self):
        """Spec can be `"<0.21"` (single str) not just tuple — ergonomics."""
        constraints = {"vllm_version_range": "<0.21.0"}
        ok, _ = check_version_constraints(
            constraints, profile=_profile("0.20.2rc1.dev93+g51f22dcfd"),
        )
        assert ok is True

    def test_prerelease_semantics(self):
        """PEP 440: dev-versions match `>=0.20.0` when prereleases=True."""
        constraints = {"vllm_version_range": (">=0.20.0", "<0.21.0")}
        ok, _ = check_version_constraints(
            constraints, profile=_profile("0.20.2rc1.dev93+g51f22dcfd"),
        )
        assert ok is True


# ─── Per-patch declarations (smoke — at least one patch uses the gate) ────


class TestPN90VllmVersionRange:
    """PN90 reference adoption — first patch to declare vllm_version_range."""

    def test_pn90_declares_vllm_version_range(self):
        """PN90's registry entry must have vllm_version_range in applies_to."""
        from sndr.dispatcher.registry import PATCH_REGISTRY
        assert "PN90" in PATCH_REGISTRY
        applies_to = PATCH_REGISTRY["PN90"].get("applies_to", {})
        assert "vllm_version_range" in applies_to, (
            "PN90 reference declaration removed — pin-gate adoption "
            "no longer demonstrated. Re-add or update test."
        )

    def test_pn90_range_admits_current_pin(self):
        """PN90 must apply on dev93 PROD baseline."""
        from sndr.dispatcher.registry import PATCH_REGISTRY
        constraints = {
            "vllm_version_range":
                PATCH_REGISTRY["PN90"]["applies_to"]["vllm_version_range"],
        }
        ok, _ = check_version_constraints(
            constraints, profile=_profile("0.20.2rc1.dev93+g51f22dcfd"),
        )
        assert ok is True, "PN90 must apply on current dev93 PROD pin"

    def test_pn90_range_rejects_predev9(self):
        """PN90 anchors don't exist pre-dev9 — gate must reject."""
        from sndr.dispatcher.registry import PATCH_REGISTRY
        constraints = {
            "vllm_version_range":
                PATCH_REGISTRY["PN90"]["applies_to"]["vllm_version_range"],
        }
        ok, _ = check_version_constraints(
            constraints, profile=_profile("0.20.1rc1.dev16+g7a1eb8ac2"),
        )
        assert ok is False, "PN90 must be gated off pre-dev9 pins"


# ─── Dispatcher wiring sanity (source-level — no live detection) ──────────


class TestDispatcherWiringPresent:
    """Sanity that decision.py reads vllm_version_range from applies_to."""

    def test_decision_source_references_vllm_version_range(self):
        """decision.py extracts vllm_version_range from applies_to keys."""
        from pathlib import Path
        # Resolve the canonical module rather than a hardcoded path: v12.x
        # turned vllm/sndr_core/dispatcher/decision.py into a re-export shim,
        # so the real source lives at sndr/dispatcher/decision.py. Reading via
        # __file__ follows the implementation wherever it moves.
        import sndr.dispatcher.decision as _decision
        text = Path(_decision.__file__).read_text()
        assert '"vllm_version_range"' in text, (
            "decision.py no longer treats vllm_version_range as a "
            "version-key. Pin-gate wiring regressed."
        )
        assert "check_version_constraints" in text, (
            "decision.py no longer calls check_version_constraints. "
            "Pin-gate wiring regressed."
        )

    def test_decision_returns_version_prefix_on_violation(self):
        """decision.py reason format includes 'VERSION:' prefix for the
        orchestrator log to be greppable by operators."""
        from pathlib import Path
        import sndr.dispatcher.decision as _decision
        text = Path(_decision.__file__).read_text()
        assert '"VERSION:' in text, (
            "decision.py no longer emits 'VERSION:' reason prefix on "
            "version-gate failure. Operator-grep continuity regressed."
        )

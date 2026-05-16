# SPDX-License-Identifier: Apache-2.0
"""End-to-end YAML round-trip test for profile-level attribution.

Audit gap found 2026-05-16: existing profile attribution tests
construct PatchesDelta in code, never through a real YAML payload.
This file exercises the FULL path:

  1. Write a model YAML (synthetic) with patches_attribution
  2. Write a profile YAML with patches_delta.attribution overriding
     model attribution + adding a new entry
  3. Load both through registry_v2._dataclass_from_dict (the actual
     loader used by `sndr launch` / `sndr compose render`)
  4. compose() the V1 ModelConfig
  5. resolve_patch_plan() under safe + minimal policies
  6. Verify the resolver sees the *profile-overridden* attribution,
     not the model-level one
  7. Verify the policy filter behaves correctly on the override

This catches the class of bugs where YAML → dataclass round-trip
silently drops fields (audit gap from the Phase D verification
sweep — every other profile-attribution test bypassed this path).
"""
from __future__ import annotations

import textwrap

import pytest

from vllm.sndr_core.model_configs.compose import compose
from vllm.sndr_core.model_configs.patch_plan import resolve_patch_plan
from vllm.sndr_core.model_configs.registry_v2 import _dataclass_from_dict
from vllm.sndr_core.model_configs.schema_v2 import (
    HardwareDef,
    ModelDef,
    ProfileDef,
)


# YAML payload structured the way operators actually write it. We
# don't write to disk to keep tests hermetic; we round-trip through
# the same _dataclass_from_dict loader registry_v2.load_model() uses.

_MODEL_YAML_DICT = {
    "schema_version": 2,
    "kind": "model",
    "id": "qwen3.6-yaml-rt-model",
    "title": "Round-trip model",
    "maintainer": "x",
    "last_validated": "2026-05-16",
    "license": "apache-2.0",
    "model_path": "/models/x",
    "patches": {
        "GENESIS_ENABLE_PN204": "1",
        "GENESIS_ENABLE_PN17": "1",
        # A pure parameter — must pass through every policy.
        "GENESIS_PN95_CONFIG_KEY": "some-key",
    },
    "patches_attribution": {
        # Model-level: PN204 is optional_perf with bench evidence.
        "PN204": {
            "role": "optional_perf",
            "bench_evidence": "model-level dev371 bench reference",
            "candidate_when": {"max_num_seqs_gte": 4},
        },
        # Model-level: PN17 is defensive (long-ctx safety margin).
        "PN17": {
            "role": "defensive",
            "note": "model-level defensive entry",
        },
    },
}

_HARDWARE_YAML_DICT = {
    "schema_version": 2,
    "kind": "hardware",
    "id": "rt-stub-hw",
    "title": "Round-trip HW",
    "maintainer": "x",
    "hardware": {
        "gpu_match_keys": ["rt-gpu"],
        "n_gpus": 1,
        "min_vram_per_gpu_mib": 8000,
    },
    "runtime": {
        "default": "docker",
        "supported": ["docker"],
        "docker": {"image": "rt:stub"},
    },
    "sizing": {
        "max_model_len": 8192,
        "max_num_seqs": 2,
        "max_num_batched_tokens": 4096,
        "gpu_memory_utilization": 0.9,
    },
}

_PROFILE_YAML_DICT = {
    "schema_version": 2,
    "kind": "profile",
    "id": "rt-stub-profile",
    "parent_model": "qwen3.6-yaml-rt-model",
    "maintainer": "x",
    "patches_delta": {
        # Override model's PN204 attribution: flag it as suspected_regression
        # for an A/B period. The note field is required for that role.
        "attribution": {
            "PN204": {
                "role": "suspected_regression",
                "note": "profile A/B window — minimal must drop this",
            },
            # New entry not in model attribution: PN95 defensive.
            "PN95": {
                "role": "defensive",
                "note": "long-ctx demote anchor",
            },
        },
    },
}


@pytest.fixture
def loaded_triple():
    """Load model + hardware + profile through the actual loader."""
    model = _dataclass_from_dict(ModelDef, _MODEL_YAML_DICT)
    hw = _dataclass_from_dict(HardwareDef, _HARDWARE_YAML_DICT)
    profile = _dataclass_from_dict(ProfileDef, _PROFILE_YAML_DICT)
    # Run schema validators — catches loader silently dropping fields.
    model.validate()
    hw.validate()
    profile.validate()
    return model, hw, profile


class TestYamlRoundTripPreservesAttribution:
    def test_model_attribution_loads_from_dict(self, loaded_triple):
        model, _, _ = loaded_triple
        assert "PN204" in model.patches_attribution
        assert model.patches_attribution["PN204"].role == "optional_perf"
        assert (
            model.patches_attribution["PN204"].candidate_when
            == {"max_num_seqs_gte": 4}
        )
        assert "PN17" in model.patches_attribution

    def test_profile_attribution_loads_from_dict(self, loaded_triple):
        _, _, profile = loaded_triple
        delta = profile.patches_delta
        assert "PN204" in delta.attribution
        assert delta.attribution["PN204"].role == "suspected_regression"
        assert "minimal must drop" in delta.attribution["PN204"].note
        assert "PN95" in delta.attribution


class TestComposeYamlRoundTripMergesAttribution:
    def test_composed_cfg_has_profile_overridden_attribution(
        self, loaded_triple,
    ):
        model, hw, profile = loaded_triple
        cfg = compose(model, hw, profile=profile)
        # Profile override wins for PN204: full replacement.
        attr204 = cfg.patches_attribution["PN204"]
        assert attr204.role == "suspected_regression"
        assert attr204.note is not None and "A/B" in attr204.note
        # Model attribution survives where profile didn't override.
        assert cfg.patches_attribution["PN17"].role == "defensive"
        # Profile-added entry made it through.
        assert cfg.patches_attribution["PN95"].role == "defensive"


class TestResolverHonorsYamlRoundTrippedOverride:
    def test_safe_keeps_pn204_when_role_suspected_regression(
        self, loaded_triple,
    ):
        """safe drops no_op only; suspected_regression survives safe."""
        model, hw, profile = loaded_triple
        cfg = compose(model, hw, profile=profile)
        plan = resolve_patch_plan(cfg, policy="safe")
        included_ids = {d.patch_id for d in plan.included}
        assert "PN204" in included_ids

    def test_minimal_drops_pn204_via_profile_override(self, loaded_triple):
        """End-to-end: model said PN204=optional_perf (would survive
        minimal because optional_perf is one of the kept roles), but
        the YAML profile flipped it to suspected_regression. Under
        minimal that role gets dropped — proving the YAML round-trip
        + compose merge + resolver path is wired correctly."""
        model, hw, profile = loaded_triple
        cfg = compose(model, hw, profile=profile)
        plan = resolve_patch_plan(cfg, policy="minimal")
        included_ids = {d.patch_id for d in plan.included}
        excluded_ids = {d.patch_id for d in plan.excluded}
        # Profile-overridden suspected_regression → minimal excludes it.
        assert "PN204" in excluded_ids, (
            f"profile override should have downgraded PN204 to "
            f"suspected_regression and minimal should have dropped it; "
            f"included={included_ids}, excluded={excluded_ids}"
        )

    def test_parameter_passes_through_every_policy(self, loaded_triple):
        """The non-toggle GENESIS_PN95_CONFIG_KEY parameter must survive
        every policy — the resolver classifies it as passthrough and
        the env property always includes it."""
        model, hw, profile = loaded_triple
        cfg = compose(model, hw, profile=profile)
        for policy in ("compat", "safe", "minimal"):
            plan = resolve_patch_plan(cfg, policy=policy)
            assert plan.env["GENESIS_PN95_CONFIG_KEY"] == "some-key", (
                f"GENESIS_PN95_CONFIG_KEY missing under {policy!r}"
            )


class TestCandidateWhenWarningSurfacesAfterRoundTrip:
    def test_candidate_when_warning_when_max_num_seqs_below_gte(
        self, loaded_triple,
    ):
        """The model-level PN204 entry has candidate_when {max_num_seqs_gte: 4}.
        Hardware sizing has max_num_seqs=2 → mismatch. Resolver must
        emit a candidate_when warning. The profile override flips
        PN204 to suspected_regression — meaning the surfaced attribution
        is the profile's (no candidate_when there). So the warning
        should be ABSENT after the profile override wins.

        This proves the merge order (profile overrides model) doesn't
        leak the model's candidate_when into the resolver's warning
        loop."""
        model, hw, profile = loaded_triple
        cfg = compose(model, hw, profile=profile)
        plan = resolve_patch_plan(cfg, policy="compat")
        # After profile override, PN204 attribution has NO candidate_when
        # → no warning about it.
        cw_warns = [
            w for w in plan.warnings
            if "candidate_when" in w and "PN204" in w
        ]
        assert cw_warns == [], (
            f"profile override should have replaced PN204 attribution "
            f"entirely (full replacement, not field merge); got "
            f"candidate_when warnings: {cw_warns}"
        )

    def test_model_only_candidate_when_does_warn(self):
        """Variant: profile WITHOUT attribution override → model's
        candidate_when stays in effect → warning emitted."""
        model = _dataclass_from_dict(ModelDef, _MODEL_YAML_DICT)
        hw = _dataclass_from_dict(HardwareDef, _HARDWARE_YAML_DICT)
        # Profile with empty attribution override.
        empty_profile_dict = {
            **_PROFILE_YAML_DICT,
            "patches_delta": {"attribution": {}},
        }
        profile = _dataclass_from_dict(ProfileDef, empty_profile_dict)
        model.validate()
        hw.validate()
        profile.validate()
        cfg = compose(model, hw, profile=profile)
        plan = resolve_patch_plan(cfg, policy="compat")
        cw_warns = [
            w for w in plan.warnings
            if "candidate_when" in w and "PN204" in w
        ]
        assert len(cw_warns) == 1, (
            f"expected one candidate_when warning for PN204 "
            f"(model says max_num_seqs_gte=4, hardware sizing says 2); "
            f"got warnings: {plan.warnings}"
        )

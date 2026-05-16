# SPDX-License-Identifier: Apache-2.0
"""Phase B — resolver tests for `resolve_patch_plan(cfg, policy)`.

Phase A schema work (commit 1bf8415e) put `patches_attribution` on
ModelDef. Phase B builds the resolver layer that consumes it:

  PolicyName       What it filters out
  ───────────────  ───────────────────────────────────────────────────
  "compat"         nothing — current behaviour for legacy operators
  "safe"           role == "no_op"        (drop patches that don't fire)
  "minimal"        role in {"no_op", "suspected_regression", "unknown"}

Patches without attribution default to role="unknown" — kept in
compat/safe, dropped in minimal. The asymmetry is deliberate:
"unknown" means "we don't yet know whether this is needed", so safe
keeps it (conservative), minimal drops it (lean / for advanced ops
who know what they're doing).

The compose() integration test verifies that ModelDef.patches_attribution
survives the V2→V1 collapse — without it the resolver has nothing
to read at runtime.

See `docs/_internal/PATCH_ATTRIBUTION_COMPOSE_GENERATOR_INTEGRATION_PLAN_2026-05-16_RU.md`
§ 6 for the resolver algorithm and § 7.2 for the inline-attribution
decision that makes this compose hand-off possible.
"""
from __future__ import annotations

import pytest

from vllm.sndr_core.model_configs.patch_plan import (
    PatchPlan,
    PatchDecision,
    resolve_patch_plan,
)
from vllm.sndr_core.model_configs.schema import (
    HardwareSpec,
    ModelConfig,
    PatchAttribution,
)


# ─── Builders ────────────────────────────────────────────────────────────


def _make_cfg(
    genesis_env: dict[str, str] | None = None,
    patches_attribution: dict[str, PatchAttribution] | None = None,
) -> ModelConfig:
    """Minimal ModelConfig builder for resolver tests — only the
    fields the resolver reads, everything else gets a stub default."""
    return ModelConfig(
        key="test-key",
        title="Test",
        description="Resolver test stub",
        schema_version=1,
        maintainer="x",
        model_path="/m",
        hardware=HardwareSpec(
            gpu_match_keys=["test-gpu"], n_gpus=1, min_vram_per_gpu_mib=8000,
        ),
        last_validated=None, genesis_pin=None, vllm_pin_required=None,
        served_model_name=None, quantization=None, kv_cache_dtype=None,
        max_model_len=8192, gpu_memory_utilization=0.9,
        max_num_seqs=1, max_num_batched_tokens=2048,
        enable_chunked_prefill=False, dtype="float16",
        enforce_eager=False, disable_custom_all_reduce=False,
        language_model_only=True, trust_remote_code=True,
        enable_auto_tool_choice=False,
        tool_call_parser=None, reasoning_parser=None, spec_decode=None,
        genesis_env=genesis_env or {}, system_env={},
        vllm_extra_args=[], cudagraph_mode="auto",
        docker=None,
        patches_attribution=patches_attribution or {},
    )


# ─── PatchPlan shape ─────────────────────────────────────────────────────


class TestPatchPlanShape:
    def test_empty_cfg_returns_empty_plan(self):
        cfg = _make_cfg()
        plan = resolve_patch_plan(cfg, policy="compat")
        assert plan.policy == "compat"
        assert plan.included == ()
        assert plan.excluded == ()
        assert plan.env == {}

    def test_included_decision_carries_env_flag_and_value(self):
        cfg = _make_cfg(genesis_env={"GENESIS_ENABLE_PN17": "1"})
        plan = resolve_patch_plan(cfg, policy="compat")
        assert len(plan.included) == 1
        d = plan.included[0]
        assert d.env_flag == "GENESIS_ENABLE_PN17"
        assert d.value == "1"
        assert d.decision == "include"

    def test_env_property_round_trips(self):
        cfg = _make_cfg(genesis_env={
            "GENESIS_ENABLE_PN17": "1",
            "GENESIS_ENABLE_PN132": "1",
        })
        plan = resolve_patch_plan(cfg, policy="compat")
        assert plan.env == {
            "GENESIS_ENABLE_PN17": "1",
            "GENESIS_ENABLE_PN132": "1",
        }


# ─── Policy: compat ──────────────────────────────────────────────────────


class TestCompatPolicy:
    def test_compat_includes_all_truthy_flags(self):
        cfg = _make_cfg(genesis_env={
            "GENESIS_ENABLE_PN17": "1",
            "GENESIS_ENABLE_PN204": "0",
            "GENESIS_ENABLE_PN132": "1",
        })
        plan = resolve_patch_plan(cfg, policy="compat")
        included_flags = {d.env_flag for d in plan.included}
        excluded_flags = {d.env_flag for d in plan.excluded}
        assert included_flags == {
            "GENESIS_ENABLE_PN17", "GENESIS_ENABLE_PN132",
        }
        # value "0" patches surface as excluded with reason "operator-disabled".
        assert excluded_flags == {"GENESIS_ENABLE_PN204"}

    def test_compat_keeps_no_op_patches(self):
        """compat is the legacy-bridge policy; it must not change
        behaviour just because attribution declares no_op."""
        cfg = _make_cfg(
            genesis_env={"GENESIS_ENABLE_PN32": "1"},
            patches_attribution={"PN32": PatchAttribution(role="no_op")},
        )
        plan = resolve_patch_plan(cfg, policy="compat")
        included_ids = {d.patch_id for d in plan.included}
        assert "PN32" in included_ids


# ─── Policy: safe ────────────────────────────────────────────────────────


class TestSafePolicy:
    def test_safe_drops_no_op(self):
        cfg = _make_cfg(
            genesis_env={
                "GENESIS_ENABLE_PN17": "1",
                "GENESIS_ENABLE_PN32": "1",
            },
            patches_attribution={
                "PN17": PatchAttribution(role="defensive"),
                "PN32": PatchAttribution(role="no_op"),
            },
        )
        plan = resolve_patch_plan(cfg, policy="safe")
        included_ids = {d.patch_id for d in plan.included}
        excluded_ids = {d.patch_id for d in plan.excluded}
        assert "PN17" in included_ids
        assert "PN32" in excluded_ids
        # And the excluded decision carries the role-based reason
        pn32 = [d for d in plan.excluded if d.patch_id == "PN32"][0]
        assert "no_op" in pn32.reason

    def test_safe_keeps_suspected_regression(self):
        """safe is conservative — only drops definitive no-ops.
        suspected_regression is one tier more aggressive (minimal)."""
        cfg = _make_cfg(
            genesis_env={"GENESIS_ENABLE_PN134": "1"},
            patches_attribution={
                "PN134": PatchAttribution(
                    role="suspected_regression",
                    note="-25% TPS bench regressor",
                ),
            },
        )
        plan = resolve_patch_plan(cfg, policy="safe")
        included_ids = {d.patch_id for d in plan.included}
        assert "PN134" in included_ids

    def test_safe_keeps_unknown(self):
        """No attribution → role='unknown' → kept in safe."""
        cfg = _make_cfg(genesis_env={"GENESIS_ENABLE_PN17": "1"})
        plan = resolve_patch_plan(cfg, policy="safe")
        d = plan.included[0]
        assert d.role == "unknown"
        assert "PN17" in {x.patch_id for x in plan.included}


# ─── Policy: minimal ─────────────────────────────────────────────────────


class TestMinimalPolicy:
    def test_minimal_drops_no_op_and_regression_and_unknown(self):
        cfg = _make_cfg(
            genesis_env={
                "GENESIS_ENABLE_PN17": "1",          # defensive — kept
                "GENESIS_ENABLE_PN32": "1",          # no_op — dropped
                "GENESIS_ENABLE_PN134": "1",         # suspected_regression — dropped
                "GENESIS_ENABLE_PN204": "1",         # optional_perf — kept
                "GENESIS_ENABLE_PN99_UNKNOWN": "1",  # no attribution → unknown — dropped
            },
            patches_attribution={
                "PN17": PatchAttribution(role="defensive"),
                "PN32": PatchAttribution(role="no_op"),
                "PN134": PatchAttribution(
                    role="suspected_regression", note="-25% TPS",
                ),
                "PN204": PatchAttribution(
                    role="optional_perf", bench_evidence="conc=8 +5%",
                ),
            },
        )
        plan = resolve_patch_plan(cfg, policy="minimal")
        included_ids = {d.patch_id for d in plan.included}
        excluded_ids = {d.patch_id for d in plan.excluded}
        assert included_ids == {"PN17", "PN204"}
        assert "PN32" in excluded_ids
        assert "PN134" in excluded_ids

    def test_minimal_keeps_load_bearing(self):
        cfg = _make_cfg(
            genesis_env={"GENESIS_ENABLE_PN95": "1"},
            patches_attribution={
                "PN95": PatchAttribution(
                    role="load_bearing",
                    note="Tier-aware KV cache — VRAM-critical at long ctx",
                ),
            },
        )
        plan = resolve_patch_plan(cfg, policy="minimal")
        assert "PN95" in {d.patch_id for d in plan.included}


# ─── Decision metadata ───────────────────────────────────────────────────


class TestDecisionMetadata:
    def test_decision_carries_role_and_note(self):
        cfg = _make_cfg(
            genesis_env={"PN204_FLAG": "1"},
            patches_attribution={
                "PN204": PatchAttribution(
                    role="optional_perf",
                    bench_evidence="dev371 35B conc=8: 689 TPS",
                    note="Hopper-future-proof",
                ),
            },
        )
        plan = resolve_patch_plan(cfg, policy="compat")
        # Without registry coupling the resolver falls back to env-flag
        # parsing; for unmatched flags role stays 'unknown'. The test
        # below uses a real GENESIS_ENABLE_<PID> pattern so the resolver
        # can derive PN204 → its attribution entry.
        cfg2 = _make_cfg(
            genesis_env={"GENESIS_ENABLE_PN204": "1"},
            patches_attribution={
                "PN204": PatchAttribution(
                    role="optional_perf",
                    bench_evidence="dev371 35B conc=8: 689 TPS",
                    note="Hopper-future-proof",
                ),
            },
        )
        plan2 = resolve_patch_plan(cfg2, policy="compat")
        pn204 = [d for d in plan2.included if d.patch_id == "PN204"][0]
        assert pn204.role == "optional_perf"
        assert "689" in pn204.bench_evidence
        assert pn204.note == "Hopper-future-proof"

    def test_decision_role_unknown_when_no_attribution(self):
        cfg = _make_cfg(genesis_env={"GENESIS_ENABLE_PN17": "1"})
        plan = resolve_patch_plan(cfg, policy="compat")
        d = plan.included[0]
        assert d.role == "unknown"
        assert d.note == ""
        assert d.bench_evidence == ""


# ─── Invalid policy ──────────────────────────────────────────────────────


class TestInvalidPolicy:
    def test_unknown_policy_raises(self):
        cfg = _make_cfg()
        with pytest.raises(ValueError, match="policy"):
            resolve_patch_plan(cfg, policy="bogus")


# ─── Non-toggle GENESIS_* env vars must pass through ─────────────────────


class TestNonToggleGenesisKeys:
    """Bug found during Phase B verification (2026-05-16): operator
    YAMLs carry plenty of ``GENESIS_*`` env vars that are *parameters*,
    not patch toggles:

      GENESIS_BUFFER_MODE: shared
      GENESIS_PN95_TICK_EVERY: '100'
      GENESIS_PN95_CONFIG_KEY: a5000-2x-...
      GENESIS_PROFILE_RUN_CAP_M: '0'      # 0 = no cap, not "disabled"
      GENESIS_P67_NUM_KV_SPLITS: '4'      # P67 kernel parameter
      GENESIS_P82_THRESHOLD_SINGLE: '...'

    The resolver MUST NOT filter these out — dropping
    ``GENESIS_PN95_CONFIG_KEY`` would silently noop PN95 (it relies
    on the key for its config lookup). The resolver only operates on
    toggle flags (``GENESIS_ENABLE_*`` / ``GENESIS_DISABLE_*``);
    every other ``GENESIS_*`` key passes through ``plan.env``
    untouched, regardless of policy."""

    def test_non_toggle_passes_through_compat(self):
        cfg = _make_cfg(genesis_env={
            "GENESIS_ENABLE_PN95": "1",
            "GENESIS_PN95_CONFIG_KEY": "a5000-2x-tier-aware-example",
            "GENESIS_PN95_TICK_EVERY": "100",
            "GENESIS_BUFFER_MODE": "shared",
        })
        plan = resolve_patch_plan(cfg, policy="compat")
        env = plan.env
        # Toggle made it through
        assert env["GENESIS_ENABLE_PN95"] == "1"
        # Non-toggles MUST pass through with values intact
        assert env["GENESIS_PN95_CONFIG_KEY"] == "a5000-2x-tier-aware-example"
        assert env["GENESIS_PN95_TICK_EVERY"] == "100"
        assert env["GENESIS_BUFFER_MODE"] == "shared"

    def test_non_toggle_passes_through_minimal(self):
        """Even under the most aggressive policy, non-toggle parameter
        keys are not subject to attribution filtering."""
        cfg = _make_cfg(
            genesis_env={
                "GENESIS_ENABLE_PN17": "1",  # defensive → kept
                "GENESIS_PN95_CONFIG_KEY": "x",   # parameter → kept
                "GENESIS_OBSERVABILITY": "1",      # parameter → kept
            },
            patches_attribution={
                "PN17": PatchAttribution(role="defensive"),
            },
        )
        plan = resolve_patch_plan(cfg, policy="minimal")
        assert plan.env["GENESIS_PN95_CONFIG_KEY"] == "x"
        assert plan.env["GENESIS_OBSERVABILITY"] == "1"

    def test_non_toggle_zero_value_not_treated_as_disabled(self):
        """``GENESIS_PROFILE_RUN_CAP_M=0`` means "no cap", not "disabled".
        The resolver must not move it to the excluded bucket."""
        cfg = _make_cfg(genesis_env={
            "GENESIS_PROFILE_RUN_CAP_M": "0",
            "GENESIS_TQ_MAX_MODEL_LEN": "0",
        })
        plan = resolve_patch_plan(cfg, policy="compat")
        # These are NOT toggle flags → should not appear in included
        # or excluded at all (those tuples only carry toggle decisions).
        all_decisions = list(plan.included) + list(plan.excluded)
        flag_names = {d.env_flag for d in all_decisions}
        assert "GENESIS_PROFILE_RUN_CAP_M" not in flag_names
        assert "GENESIS_TQ_MAX_MODEL_LEN" not in flag_names
        # But values still pass through via plan.env
        assert plan.env["GENESIS_PROFILE_RUN_CAP_M"] == "0"
        assert plan.env["GENESIS_TQ_MAX_MODEL_LEN"] == "0"

    def test_toggle_classification_recognises_disable_form(self):
        """``GENESIS_DISABLE_<PID>`` is the dual of ENABLE; both are
        toggles and the resolver treats both the same way (the env
        VALUE decides include/exclude, not the verb)."""
        cfg = _make_cfg(genesis_env={
            "GENESIS_ENABLE_PN17": "1",
            "GENESIS_DISABLE_PN204": "1",
        })
        plan = resolve_patch_plan(cfg, policy="compat")
        # Both go through the toggle path → both end up in `included`
        # (truthy value) with patch_id resolved via the prefix-strip
        # fallback when the registry doesn't carry them.
        flags = {d.env_flag for d in plan.included}
        assert "GENESIS_ENABLE_PN17" in flags
        assert "GENESIS_DISABLE_PN204" in flags


# ─── Conflicts_with detection ────────────────────────────────────────────


class TestConflictsWarnings:
    """The resolver consults PATCH_REGISTRY for `conflicts_with`
    declarations on each included patch. When two conflicting toggles
    survive into ``included`` together, the resolver appends a
    warning string so the operator sees the misconfig before launch.

    Warnings are advisory — they do NOT move the patch to excluded.
    The dispatcher's runtime layer is the ultimate gate; resolver
    just surfaces the problem early."""

    def test_two_conflicting_patches_emit_warning(self):
        # P65 and P67 conflict per the live registry.
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        # Sanity guard: keep this test honest if registry changes.
        assert "P67" in PATCH_REGISTRY["P65"].get("conflicts_with", [])

        flag_p65 = PATCH_REGISTRY["P65"]["env_flag"]
        flag_p67 = PATCH_REGISTRY["P67"]["env_flag"]
        cfg = _make_cfg(genesis_env={flag_p65: "1", flag_p67: "1"})
        plan = resolve_patch_plan(cfg, policy="compat")
        # Both stay included — warning surface, not exclusion.
        included_ids = {d.patch_id for d in plan.included}
        assert "P65" in included_ids and "P67" in included_ids
        # Warning mentions both patches.
        assert any(
            "P65" in w and "P67" in w for w in plan.warnings
        ), f"expected P65↔P67 conflict warning, got: {plan.warnings}"

    def test_single_patch_no_conflict_warning(self):
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        flag = PATCH_REGISTRY["P65"]["env_flag"]
        cfg = _make_cfg(genesis_env={flag: "1"})
        plan = resolve_patch_plan(cfg, policy="compat")
        assert plan.warnings == ()

    def test_conflict_pair_warned_only_once(self):
        """Avoid the (A→B) + (B→A) double-warn. Each unique (pid_a, pid_b)
        pair surfaces at most one warning regardless of which side
        declares the conflict in its registry meta.

        Note: A-19 subpatch families (P67 + P67b share env_flag) DO
        produce separate warnings — P65↔P67 and P65↔P67b are two
        distinct conflict pairs even though both fire on the same
        toggle. That's a feature: each pair stays operator-visible
        so the resolver doesn't hide structurally distinct issues.
        """
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        # P62 ↔ PN58 — single-family conflict pair, perfect for the
        # canonical "warn once per pair" check.
        flag_p62 = PATCH_REGISTRY["P62"]["env_flag"]
        flag_pn58 = PATCH_REGISTRY["PN58"]["env_flag"]
        cfg = _make_cfg(genesis_env={flag_p62: "1", flag_pn58: "1"})
        plan = resolve_patch_plan(cfg, policy="compat")
        conflict_warns = [
            w for w in plan.warnings if "P62" in w and "PN58" in w
        ]
        assert len(conflict_warns) == 1, (
            f"expected 1 warning, got {len(conflict_warns)}: "
            f"{conflict_warns}"
        )

    def test_subpatch_family_emits_pair_per_subpatch(self):
        """When a conflict pair includes an A-19 family (P67 + P67b),
        the resolver emits ONE warning per (primary, family_member)
        pair — P65 ⨯ P67 AND P65 ⨯ P67b — because each subpatch is
        an independent runtime gate and dropping just one wouldn't
        resolve the other half of the conflict."""
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        flag_p65 = PATCH_REGISTRY["P65"]["env_flag"]
        flag_p67 = PATCH_REGISTRY["P67"]["env_flag"]
        cfg = _make_cfg(genesis_env={flag_p65: "1", flag_p67: "1"})
        plan = resolve_patch_plan(cfg, policy="compat")
        # Canonical warning format is "conflict: A ⨯ B — …". Parse the
        # A and B back out of every warning to avoid substring-overlap
        # bugs (P67 is a substring of P67b otherwise).
        import re
        pair_re = re.compile(r"conflict: (\S+) ⨯ (\S+) —")
        warned_pairs = {
            tuple(sorted(m.groups()))
            for w in plan.warnings
            for m in [pair_re.search(w)] if m
        }
        assert ("P65", "P67") in warned_pairs
        assert ("P65", "P67b") in warned_pairs

    def test_excluded_patches_dont_count_for_conflicts(self):
        """If one of the conflicting patches is operator-disabled
        (value="0" → excluded), no conflict exists at runtime — no
        warning."""
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        flag_p65 = PATCH_REGISTRY["P65"]["env_flag"]
        flag_p67 = PATCH_REGISTRY["P67"]["env_flag"]
        cfg = _make_cfg(genesis_env={flag_p65: "1", flag_p67: "0"})
        plan = resolve_patch_plan(cfg, policy="compat")
        # Only one in included; the other was operator-disabled.
        assert {d.patch_id for d in plan.included} == {"P65"}
        # No conflict warning.
        conflict_warns = [
            w for w in plan.warnings if "P65" in w and "P67" in w
        ]
        assert conflict_warns == []


# ─── A-19 family attribution lookup ──────────────────────────────────────


class TestFamilyAttributionLookup:
    """A-19 subpatch families share an env_flag. The primary patch ID
    (alphabetical first) is what surfaces in PatchDecision.patch_id,
    but attribution can be authored against ANY family member —
    operators usually attribute the "main" patch, which isn't always
    the alphabetical primary.

    Example: PN40 + PN40-classifier share GENESIS_ENABLE_PN40_DFLASH_OMNIBUS.
    Alphabetical primary is "PN40"; operator-facing main is also
    "PN40". Either way, attribution keyed by either family member
    must be found by the resolver.
    """

    def test_attribution_keyed_by_family_primary_found(self):
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        flag = PATCH_REGISTRY["PN40"]["env_flag"]
        cfg = _make_cfg(
            genesis_env={flag: "1"},
            patches_attribution={
                "PN40": PatchAttribution(role="defensive", note="primary"),
            },
        )
        plan = resolve_patch_plan(cfg, policy="compat")
        d = plan.included[0]
        assert d.role == "defensive"
        assert d.note == "primary"

    def test_attribution_keyed_by_family_non_primary_found(self):
        """If operator attributes PN40-classifier (non-primary family
        member), the resolver must still find it when resolving the
        shared env flag."""
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        flag = PATCH_REGISTRY["PN40-classifier"]["env_flag"]
        cfg = _make_cfg(
            genesis_env={flag: "1"},
            patches_attribution={
                "PN40-classifier": PatchAttribution(
                    role="defensive", note="keyed by sub-id",
                ),
            },
        )
        plan = resolve_patch_plan(cfg, policy="compat")
        d = plan.included[0]
        assert d.role == "defensive"
        assert d.note == "keyed by sub-id"

    def test_attribution_on_both_family_members_picks_primary(self):
        """When attribution exists on multiple family members, the
        resolver picks the primary's metadata for the surfaced
        PatchDecision — deterministic + matches the patch_id that
        operators see in plan output."""
        from vllm.sndr_core.dispatcher.registry import PATCH_REGISTRY
        flag = PATCH_REGISTRY["PN40"]["env_flag"]
        cfg = _make_cfg(
            genesis_env={flag: "1"},
            patches_attribution={
                "PN40": PatchAttribution(
                    role="load_bearing", note="primary entry"),
                "PN40-classifier": PatchAttribution(
                    role="defensive", note="sub entry"),
            },
        )
        plan = resolve_patch_plan(cfg, policy="compat")
        d = plan.included[0]
        assert d.patch_id == "PN40"
        assert d.role == "load_bearing"
        assert d.note == "primary entry"


# ─── candidate_when evaluation ───────────────────────────────────────────


class TestCandidateWhenEvaluation:
    """`candidate_when` is operator-authored predicate metadata. When
    present and not matching the current cfg, the resolver emits a
    warning so operators see "this patch's claimed conditions don't
    match your rig — it may be a no-op at runtime regardless of the
    policy you picked".

    The resolver does NOT move the patch between included/excluded —
    candidate_when is INFORMATIONAL. Operators who set candidate_when
    are usually saying "I think this patch makes sense in this regime";
    silently filtering it out would surprise them. Warning is the
    right granularity."""

    def _cfg_with(self, **kw) -> "ModelConfig":
        """ModelConfig stub with overridable hardware/sizing knobs."""
        defaults = dict(
            max_num_seqs=2, max_model_len=8192,
        )
        defaults.update(kw)
        cfg = _make_cfg(genesis_env={"GENESIS_ENABLE_PN204": "1"})
        for k, v in defaults.items():
            setattr(cfg, k, v)
        # n_gpus lives on the embedded HardwareSpec.
        if "n_gpus" in defaults:
            cfg.hardware.n_gpus = defaults["n_gpus"]
        return cfg

    def test_no_candidate_when_no_warning(self):
        cfg = self._cfg_with()
        cfg.patches_attribution = {
            "PN204": PatchAttribution(role="defensive"),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        assert plan.warnings == ()

    def test_empty_candidate_when_dict_no_warning(self):
        cfg = self._cfg_with()
        cfg.patches_attribution = {
            "PN204": PatchAttribution(role="defensive", candidate_when={}),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        assert plan.warnings == ()

    def test_gte_match_no_warning(self):
        cfg = self._cfg_with(max_num_seqs=8)
        cfg.patches_attribution = {
            "PN204": PatchAttribution(
                role="optional_perf",
                bench_evidence="multi-conc",
                candidate_when={"max_num_seqs_gte": 4},
            ),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        assert all("candidate_when" not in w for w in plan.warnings)

    def test_gte_mismatch_emits_warning(self):
        cfg = self._cfg_with(max_num_seqs=2)
        cfg.patches_attribution = {
            "PN204": PatchAttribution(
                role="optional_perf",
                bench_evidence="multi-conc",
                candidate_when={"max_num_seqs_gte": 4},
            ),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        # Patch stays included (warning-only) but the warning is present.
        assert "PN204" in {d.patch_id for d in plan.included}
        cw_warns = [w for w in plan.warnings if "candidate_when" in w and "PN204" in w]
        assert len(cw_warns) == 1, f"missing candidate_when warning: {plan.warnings}"
        assert "max_num_seqs" in cw_warns[0]

    def test_lte_predicate(self):
        cfg = self._cfg_with(max_model_len=8192)
        cfg.patches_attribution = {
            "PN204": PatchAttribution(
                role="defensive",
                candidate_when={"max_model_len_lte": 4096},
            ),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        cw_warns = [w for w in plan.warnings if "candidate_when" in w]
        assert len(cw_warns) == 1
        assert "max_model_len" in cw_warns[0]

    def test_list_membership_match(self):
        cfg = self._cfg_with()
        cfg.tool_call_parser = "qwen3_coder"
        cfg.patches_attribution = {
            "PN204": PatchAttribution(
                role="defensive",
                candidate_when={"tool_call_parser": ["qwen3_coder", "qwen3_xml"]},
            ),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        cw_warns = [w for w in plan.warnings if "candidate_when" in w]
        assert cw_warns == []

    def test_list_membership_mismatch(self):
        cfg = self._cfg_with()
        cfg.tool_call_parser = "qwen3_coder"
        cfg.patches_attribution = {
            "PN204": PatchAttribution(
                role="defensive",
                candidate_when={"tool_call_parser": ["qwen3_xml"]},
            ),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        cw_warns = [w for w in plan.warnings if "candidate_when" in w]
        assert len(cw_warns) == 1

    def test_excluded_patch_skips_candidate_when_check(self):
        """If the patch is operator-disabled (value=0), it's in
        excluded — no point warning about candidate_when mismatch."""
        cfg = self._cfg_with()
        cfg.genesis_env = {"GENESIS_ENABLE_PN204": "0"}
        cfg.patches_attribution = {
            "PN204": PatchAttribution(
                role="defensive",
                candidate_when={"max_num_seqs_gte": 9999},  # would mismatch
            ),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        # Patch is excluded (value 0). No candidate_when warning.
        assert plan.warnings == ()

    def test_unknown_predicate_key_warns_but_passes(self):
        """An unrecognised predicate key produces a warning but the
        resolver doesn't fail closed (forward-compat: operators may
        author new predicate names that the resolver hasn't been
        updated to recognise)."""
        cfg = self._cfg_with()
        cfg.patches_attribution = {
            "PN204": PatchAttribution(
                role="defensive",
                candidate_when={"unknown_thing_xyz": 42},
            ),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        unknown_warns = [
            w for w in plan.warnings if "unknown_thing_xyz" in w
        ]
        assert len(unknown_warns) == 1

    def test_multiple_predicates_all_must_match(self):
        cfg = self._cfg_with(max_num_seqs=8, max_model_len=4096)
        cfg.patches_attribution = {
            "PN204": PatchAttribution(
                role="defensive",
                candidate_when={
                    "max_num_seqs_gte": 4,
                    "max_model_len_gte": 8192,  # this one fails
                },
            ),
        }
        plan = resolve_patch_plan(cfg, policy="compat")
        cw_warns = [w for w in plan.warnings if "candidate_when" in w]
        assert len(cw_warns) == 1, plan.warnings
        # The mismatch reason should name the FAILING predicate, not
        # the passing one.
        assert "max_model_len" in cw_warns[0]


class TestCandidateWhenOnRealPreset:
    """The 35B-prod model file has a real candidate_when on PN204
    (max_num_seqs_gte: 4). The latency profile runs with
    max_num_seqs=2, so loading the real preset must surface the
    candidate_when mismatch warning."""

    def test_pn204_emits_warning_on_latency_preset(self):
        """PN204 is value='0' (operator-disabled) on prod-35b. So it
        won't be in included → candidate_when check skipped. But
        prod-35b-multiconc enables PN204 (value='1') AND raises
        max_num_seqs to 8 → candidate_when matches, no warning.
        Either way the integration must run cleanly. Test the
        defensive scenario: synthetic compose with PN204 enabled
        + max_num_seqs=2 + the model's real attribution."""
        from vllm.sndr_core.model_configs.registry_v2 import load_alias
        cfg = load_alias("prod-35b")
        # Force PN204 enabled to exercise the candidate_when path.
        cfg.genesis_env = dict(cfg.genesis_env)
        cfg.genesis_env["GENESIS_ENABLE_PN204_DUAL_STREAM_INPROJ"] = "1"
        # max_num_seqs is 2 on prod-35b → candidate_when_gte=4 mismatches.
        assert cfg.max_num_seqs < 4
        plan = resolve_patch_plan(cfg, policy="compat")
        cw_warns = [
            w for w in plan.warnings
            if "candidate_when" in w and "PN204" in w
        ]
        assert len(cw_warns) >= 1, (
            f"expected PN204 candidate_when warning; got {plan.warnings}"
        )

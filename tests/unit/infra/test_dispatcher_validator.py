# SPDX-License-Identifier: Apache-2.0
"""Unit tests for A3/D2 — PATCH_REGISTRY dependency / conflict validator.

Two distinct levels of validation:

  1. **Static registry validation** (`validate_registry`):
     Walks PATCH_REGISTRY and verifies that every `requires_patches` /
     `conflicts_with` reference resolves to a real patch_id, no patch
     references itself, and there are no simple A→B + B→A cycles.
     Runs at import time / boot.

  2. **Dynamic apply-plan validation** (`validate_apply_plan`):
     Given the live set of patch_ids that the dispatcher decided to APPLY
     this boot, returns issues for missing-required and present-conflict.
     Runs once after `apply_all` finishes.

Tests cover both.
"""
from __future__ import annotations


# Importing this also exercises module-load — if we accidentally introduce
# a cycle in the static registry, the test will surface it via fixture.
from sndr import dispatcher


# ─── Static registry validation ─────────────────────────────────────────────


class TestValidateRegistry:
    """Static checks on PATCH_REGISTRY shape (no dynamic state)."""

    def test_clean_registry_validates(self):
        """The shipped PATCH_REGISTRY must validate without ERROR or
        WARNING. INFO-level issues (like §5.5 lifecycle hints) are
        advisory and do not constitute structural problems.

        If this fails, a recent edit introduced an unknown patch_id reference
        or a cycle. Investigate the issue list before merging.
        """
        issues = dispatcher.validate_registry()
        blocking = [i for i in issues if i.severity in ("ERROR", "WARNING")]
        assert blocking == [], (
            f"PATCH_REGISTRY has structural problems:\n"
            + "\n".join(
                f"  - {i.severity}: {i.patch_id}: {i.message}"
                for i in blocking
            )
        )

    def test_validate_registry_returns_list(self):
        """Always returns a list (never None / raises)."""
        out = dispatcher.validate_registry()
        assert isinstance(out, list)

    def test_unknown_required_patch_detected(self, monkeypatch):
        """If a patch declares requires=['P_NONEXISTENT'], surface as ERROR."""
        fake_registry = {
            "P_FOO": {
                "title": "test",
                "env_flag": "GENESIS_ENABLE_FOO",
                "default_on": False,
                "requires_patches": ["P_NONEXISTENT"],
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) == 1
        assert errors[0].patch_id == "P_FOO"
        assert "P_NONEXISTENT" in errors[0].message
        assert "requires" in errors[0].message.lower()

    def test_unknown_conflict_patch_detected(self, monkeypatch):
        """If conflicts_with references a phantom patch, surface as ERROR."""
        fake_registry = {
            "P_FOO": {
                "title": "test",
                "env_flag": "GENESIS_ENABLE_FOO",
                "default_on": False,
                "conflicts_with": ["P_GHOST"],
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        errors = [i for i in issues if i.severity == "ERROR"]
        assert len(errors) == 1
        assert errors[0].patch_id == "P_FOO"
        assert "P_GHOST" in errors[0].message
        assert "conflict" in errors[0].message.lower()

    def test_self_reference_in_requires_detected(self, monkeypatch):
        """A patch that requires itself is a clear bug."""
        fake_registry = {
            "P_FOO": {
                "title": "self-loop",
                "env_flag": "GENESIS_ENABLE_FOO",
                "default_on": False,
                "requires_patches": ["P_FOO"],
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert any("self" in i.message.lower() for i in issues), issues

    def test_self_reference_in_conflicts_detected(self, monkeypatch):
        """conflicts_with=[self] means the patch can never apply — bug."""
        fake_registry = {
            "P_FOO": {
                "title": "self-conflict",
                "env_flag": "GENESIS_ENABLE_FOO",
                "default_on": False,
                "conflicts_with": ["P_FOO"],
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert any("self" in i.message.lower() for i in issues), issues

    def test_simple_two_node_cycle_detected(self, monkeypatch):
        """A→B and B→A must be flagged as a requires-cycle."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False, "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False, "requires_patches": ["P_A"]},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert any("cycle" in i.message.lower() for i in issues), issues

    def test_three_node_cycle_detected(self, monkeypatch):
        """A→B→C→A must be flagged."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False, "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False, "requires_patches": ["P_C"]},
            "P_C": {"env_flag": "GC", "default_on": False, "requires_patches": ["P_A"]},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        assert any("cycle" in i.message.lower() for i in issues), issues

    def test_dag_no_cycle(self, monkeypatch):
        """Linear A→B→C is valid (DAG). No cycle issue.

        Filter excludes the §5.5 INFO about "lifecycle" — "lifecycle"
        contains the substring "cycle", so a naive filter false-matches.
        """
        fake_registry = {
            "P_A": {"env_flag": "GENESIS_ENABLE_A", "default_on": False, "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GENESIS_ENABLE_B", "default_on": False, "requires_patches": ["P_C"]},
            "P_C": {"env_flag": "GENESIS_ENABLE_C", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        cycle_issues = [
            i for i in issues
            if "cycle" in i.message.lower()
            and "lifecycle" not in i.message.lower()
        ]
        assert cycle_issues == [], cycle_issues


# ─── Per-entry contract checks (F-009 expansion) ───────────────────────────


class TestPerEntryContract:
    """validate_registry now also enforces per-entry contracts: tier,
    lifecycle, env_flag prefix, applies_to shape, apply_module import.

    Audit `sndr_structure_deep_audit_2026-05-07.md` F-009: previously
    the validator promised these checks in its docstring but only ran
    the requires/conflicts graph layer.
    """

    def test_invalid_tier_detected(self, monkeypatch):
        fake_registry = {
            "P_BAD": {
                "env_flag": "GENESIS_ENABLE_BAD",
                "tier": "supersonic",  # not in {community, engine}
                "default_on": False,
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        tier_issues = [i for i in issues if "tier" in i.message.lower()]
        assert any(i.severity == "ERROR" for i in tier_issues), issues

    def test_invalid_lifecycle_detected(self, monkeypatch):
        fake_registry = {
            "P_BAD": {
                "env_flag": "GENESIS_ENABLE_BAD",
                "lifecycle": "moonshot",  # not in canonical set
                "default_on": False,
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        lc_issues = [i for i in issues if "lifecycle" in i.message.lower()]
        assert any(i.severity == "ERROR" for i in lc_issues), issues

    def test_non_canonical_env_flag_warning(self, monkeypatch):
        fake_registry = {
            "P_BAD": {
                "env_flag": "JUST_FOO",  # missing canonical prefix
                "default_on": False,
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        env_issues = [i for i in issues if "env_flag" in i.message]
        # WARNING (not ERROR): runtime decision strips the prefix and
        # delegates to env.is_enabled, so the registry can be drift-fixed
        # gradually without breaking apply behavior.
        assert any(i.severity == "WARNING" for i in env_issues), issues

    def test_legacy_prefix_accepted(self, monkeypatch):
        """Legacy default-on patches use SNDR_LEGACY_/GENESIS_LEGACY_ for
        opt-out — that prefix is canonical, no warning."""
        fake_registry = {
            "P_OK": {
                "env_flag": "GENESIS_LEGACY_OK",
                "lifecycle": "legacy",
                "default_on": True,
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        env_issues = [i for i in issues if "env_flag" in i.message]
        assert env_issues == [], env_issues

    def test_apply_module_unimportable_detected(self, monkeypatch):
        fake_registry = {
            "P_BAD": {
                "env_flag": "GENESIS_ENABLE_BAD",
                "apply_module": "sndr.does.not.exist",
                "default_on": False,
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        am_issues = [i for i in issues if "apply_module" in i.message]
        assert any(i.severity == "ERROR" for i in am_issues), issues

    def test_applies_to_must_be_dict(self, monkeypatch):
        fake_registry = {
            "P_BAD": {
                "env_flag": "GENESIS_ENABLE_BAD",
                "applies_to": ["not", "a", "dict"],
                "default_on": False,
            },
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        at_issues = [i for i in issues if "applies_to" in i.message]
        assert any(i.severity == "ERROR" for i in at_issues), issues


# ─── Dynamic apply-plan validation ─────────────────────────────────────────


class TestValidateApplyPlan:
    """Runtime validation given a set of actually-applied patch_ids."""

    def test_empty_plan_no_issues(self, monkeypatch):
        """No patches applied → no issues."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied=set())
        assert issues == []

    def test_required_satisfied(self, monkeypatch):
        """P_A requires P_B, both applied → no issues."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A", "P_B"})
        assert issues == []

    def test_missing_required_detected(self, monkeypatch):
        """P_A applied but P_B (required) is not → ERROR."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A"})
        assert len(issues) == 1
        assert issues[0].severity == "ERROR"
        assert issues[0].patch_id == "P_A"
        assert "P_B" in issues[0].message
        assert "missing" in issues[0].message.lower() or \
               "requires" in issues[0].message.lower()

    def test_missing_required_skipped_if_dependent_skipped(self, monkeypatch):
        """If P_A is NOT applied, missing P_B requirement is irrelevant."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "requires_patches": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied=set())
        assert issues == []

    def test_conflict_both_applied(self, monkeypatch):
        """P_A conflicts with P_B, both applied → ERROR."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "conflicts_with": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A", "P_B"})
        assert len(issues) >= 1
        assert any(
            i.severity == "ERROR" and "conflict" in i.message.lower()
            for i in issues
        ), issues

    def test_conflict_only_one_applied_no_issue(self, monkeypatch):
        """Conflict declared but only one of pair applied → no issue."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "conflicts_with": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A"})
        assert issues == []

    def test_conflict_symmetry_not_double_reported(self, monkeypatch):
        """If P_A.conflicts=[P_B] AND P_B.conflicts=[P_A], do not report twice."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "conflicts_with": ["P_B"]},
            "P_B": {"env_flag": "GB", "default_on": False,
                    "conflicts_with": ["P_A"]},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_A", "P_B"})
        # One canonical conflict pair reported (in either direction), not two
        conflict_issues = [i for i in issues if "conflict" in i.message.lower()]
        assert len(conflict_issues) == 1, conflict_issues

    def test_unknown_patch_in_applied_set(self, monkeypatch):
        """Caller passes a patch_id not in registry → WARNING."""
        fake_registry = {"P_A": {"env_flag": "GA", "default_on": False}}
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_apply_plan(applied={"P_GHOST"})
        assert len(issues) >= 1
        assert any("unknown" in i.message.lower() for i in issues)

    def test_validation_issue_dataclass_shape(self, monkeypatch):
        """Issue objects must expose .severity, .patch_id, .message attrs."""
        fake_registry = {
            "P_A": {"env_flag": "GA", "default_on": False,
                    "requires_patches": ["P_GHOST"]},
        }
        monkeypatch.setattr(dispatcher, "PATCH_REGISTRY", fake_registry)
        issues = dispatcher.validate_registry()
        for i in issues:
            assert hasattr(i, "severity")
            assert hasattr(i, "patch_id")
            assert hasattr(i, "message")
            assert i.severity in ("ERROR", "WARNING", "INFO")


# ─── Real-registry sanity: the relationships we just declared ──────────────


class TestRealRegistryRelationships:
    """Verify the natural dependencies / conflicts we've encoded.

    These pin specific design decisions so a refactor can't silently
    weaken the validator's coverage of known constraints.
    """

    def test_p60b_requires_p60(self):
        """P60b (Phase 2 Triton kernel) requires P60 (Phase 1 SSM pre-copy)."""
        meta = dispatcher.PATCH_REGISTRY.get("P60b")
        assert meta is not None
        assert "P60" in meta.get("requires_patches", []), (
            "P60b is Phase 2 of GDN+ngram fix; P60 (Phase 1) must apply first"
        )

    def test_p85_p84_dependency_dropped_after_retire(self):
        """P85's requires_patches=["P84"] was dropped 2026-06-11 (plan
        section 5 cascade resolution): P84 retired because both its
        sites are upstream-native on pin 0.22.1rc1.dev259 (scheduler
        hash_block_size param + resolve_kv_cache_block_sizes). Fine
        hashes now come from the upstream-native --hash-block-size
        engine arg (cache_config.hash_block_size) — an operator-config
        prerequisite, not a Genesis patch dependency. P85 instead
        declares composition with PN346 (Site 2 dual anchor variants;
        PN346 boot-dispatches first)."""
        meta = dispatcher.PATCH_REGISTRY.get("P85")
        assert meta is not None
        assert meta.get("requires_patches", []) == [], (
            "P85 must not re-grow patch dependencies silently — P84 was "
            "retired (upstream-native hash_block_size) and the fine-hash "
            "prerequisite is operator config, not a patch"
        )
        assert "PN346" in meta.get("composes_with", []), (
            "P85 must declare PN346 composition (Site 2 dual anchor variants)"
        )

    def test_p74_requires_p72(self):
        """P74 chunk-clamp is the safety-net for P72-unblocked batched_tokens."""
        meta = dispatcher.PATCH_REGISTRY.get("P74")
        assert meta is not None
        assert "P72" in meta.get("requires_patches", []), (
            "P74 is 'P72 companion' per its title; chunk-clamp guards "
            "P72-unblocked >4096 batched_tokens"
        )

    def test_p56_p57_archived(self):
        """P56 + P57 archived 2026-05-05 — both were dead-end TQ spec-decode
        attempts superseded by P65 (CG downgrade). They now live in
        Genesis_internal_docs/_archive/dead_patches/p56_p57_tq_specdec_deadends/.
        """
        assert "P56" not in dispatcher.PATCH_REGISTRY
        assert "P57" not in dispatcher.PATCH_REGISTRY

    def test_p67_conflicts_p65(self):
        """P67 multi-query kernel is 'proper fix replacing P65 workaround'."""
        meta = dispatcher.PATCH_REGISTRY.get("P67")
        assert meta is not None
        assert "P65" in meta.get("conflicts_with", []), (
            "P67 credit explicitly states 'replaces P65 workaround'"
        )

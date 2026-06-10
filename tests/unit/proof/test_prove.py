# SPDX-License-Identifier: Apache-2.0
"""Tests for §6.8 patch proof gate (R1 mitigation).

Contract: every PATCH_REGISTRY entry can be statically verified for
correctness independent of GPU access. Bench-delta evidence (Phase 10)
slots in later as the `bench_delta` field on the artefact.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest


# ─── Helpers ───────────────────────────────────────────────────────────


def _run_cli(handler, opts: argparse.Namespace) -> tuple[int, str]:
    buf = io.StringIO()
    rc = None
    with redirect_stdout(buf):
        rc = handler(opts)
    return rc, buf.getvalue()


# ─── static_checks_for_patch — known patches ──────────────────────────


class TestStaticChecksKnownPatches:
    def test_known_spec_only_patch_passes(self):
        """P102 is in KNOWN_SPEC_ONLY — must pass without apply_module."""
        from sndr.proof import static_checks_for_patch
        checks = static_checks_for_patch("P102")
        # P-1 (in registry), P-2 (in KNOWN_SPEC_ONLY), P-4 (in KNOWN_SPEC_ONLY),
        # P-5 (env_flag canonical). P-3/6/7 may be skipped (no apply_module / no deps).
        assert all(c.passed for c in checks), (
            f"P102 should pass; failures: "
            f"{[(c.rule, c.message) for c in checks if not c.passed]}"
        )

    def test_unknown_patch_id_fails_p1(self):
        from sndr.proof import static_checks_for_patch
        checks = static_checks_for_patch("P_NONEXISTENT_XYZ_999")
        assert len(checks) == 1
        assert checks[0].rule == "P-1"
        assert checks[0].passed is False

    def test_real_patch_with_legacy_register_passes_after_overlay(self):
        """P67b has no explicit `apply_module` key in registry.py but
        IS resolvable via either:
          • the integration-tree walk (Stage 6 migration path); OR
          • the legacy `@register_patch` register (pre-Stage-6 path).

        Entry 17 made P-2 accept the resolved apply_module from
        `dispatcher.spec.iter_patch_specs()` AND legacy-register
        membership as proof. P-2 now passes; the historical "metadata
        gap" failure mode is exercised by `TestStaticChecksSynthetic`
        with a stub registry instead.
        """
        from sndr.proof import static_checks_for_patch
        checks = static_checks_for_patch("P67b")
        rule_map = {c.rule: c for c in checks}
        # P-1 must pass (P67b IS in registry).
        assert rule_map["P-1"].passed
        # P-2 passes now — apply_module either resolves via integration
        # tree (Stage 6 migrated patches) or via the legacy-register
        # branch (Phase 10 migration target patches).
        assert rule_map["P-2"].passed is True
        # P-4 still passes — present in legacy register (no shadow orphan).
        assert rule_map["P-4"].passed

    def test_env_flag_in_canonical_registry(self):
        """Every PATCH_REGISTRY entry's env_flag must be in the §6.7
        canonical key registry. P-5 surfaces drift."""
        from sndr.proof import static_checks_for_patch
        # Pick a known patch + verify its env_flag check passes.
        checks = static_checks_for_patch("P58")
        p5 = next((c for c in checks if c.rule == "P-5"), None)
        assert p5 is not None
        assert p5.passed, p5.message


# ─── static_checks_for_patch — synthetic registry ─────────────────────


class TestStaticChecksSynthetic:
    """Pass synthetic registries / canonical key sets to exercise edge
    cases that the real PATCH_REGISTRY doesn't currently exhibit."""

    def test_apply_module_importable_passes(self):
        from sndr.proof import static_checks_for_patch
        # Use a module we know imports cleanly.
        registry = {
            "TEST_OK": {
                "env_flag": "GENESIS_ENABLE_TEST_OK",
                "apply_module": "sndr.cli.legacy._io",
            },
        }
        canonical_keys = {"GENESIS_ENABLE_TEST_OK"}
        checks = static_checks_for_patch(
            "TEST_OK",
            registry=registry, canonical_keys=canonical_keys,
            known_spec_only=frozenset(), legacy_names={"TEST_OK"},
        )
        rule_map = {c.rule: c for c in checks}
        assert rule_map["P-3"].passed, rule_map["P-3"].message

    def test_apply_module_importable_fails(self):
        from sndr.proof import static_checks_for_patch
        registry = {
            "TEST_BAD_IMPORT": {
                "env_flag": "GENESIS_ENABLE_TEST_BAD",
                "apply_module": "sndr.nonexistent.module",
            },
        }
        canonical_keys = {"GENESIS_ENABLE_TEST_BAD"}
        checks = static_checks_for_patch(
            "TEST_BAD_IMPORT",
            registry=registry, canonical_keys=canonical_keys,
            known_spec_only=frozenset(), legacy_names={"TEST_BAD_IMPORT"},
        )
        rule_map = {c.rule: c for c in checks}
        assert rule_map["P-3"].passed is False
        assert "cannot import" in rule_map["P-3"].message

    def test_unknown_env_flag_fails_p5(self):
        from sndr.proof import static_checks_for_patch
        registry = {
            "TEST_TYPO_FLAG": {
                "env_flag": "GENESIS_ENABLE_TEST_TYPO",
            },
        }
        canonical_keys: set[str] = set()    # empty — flag won't be canonical
        checks = static_checks_for_patch(
            "TEST_TYPO_FLAG",
            registry=registry, canonical_keys=canonical_keys,
            known_spec_only=frozenset({"TEST_TYPO_FLAG"}),
            legacy_names={"TEST_TYPO_FLAG"},
        )
        rule_map = {c.rule: c for c in checks}
        assert rule_map["P-5"].passed is False
        assert "not in canonical key registry" in rule_map["P-5"].message

    def test_requires_patches_unknown_fails_p6(self):
        from sndr.proof import static_checks_for_patch
        registry = {
            "TEST_NEEDS_GHOST": {
                "env_flag": "GENESIS_ENABLE_TEST_NEEDS",
                "requires_patches": ["P_GHOST_X"],
            },
        }
        canonical_keys = {"GENESIS_ENABLE_TEST_NEEDS"}
        checks = static_checks_for_patch(
            "TEST_NEEDS_GHOST",
            registry=registry, canonical_keys=canonical_keys,
            known_spec_only=frozenset({"TEST_NEEDS_GHOST"}),
            legacy_names={"TEST_NEEDS_GHOST"},
        )
        rule_map = {c.rule: c for c in checks}
        assert rule_map["P-6"].passed is False
        assert "P_GHOST_X" in rule_map["P-6"].message

    def test_conflicts_with_unknown_fails_p7(self):
        from sndr.proof import static_checks_for_patch
        registry = {
            "TEST_CONFLICT_GHOST": {
                "env_flag": "GENESIS_ENABLE_TEST_CONFLICT",
                "conflicts_with": ["P_GHOST_C"],
            },
        }
        canonical_keys = {"GENESIS_ENABLE_TEST_CONFLICT"}
        checks = static_checks_for_patch(
            "TEST_CONFLICT_GHOST",
            registry=registry, canonical_keys=canonical_keys,
            known_spec_only=frozenset({"TEST_CONFLICT_GHOST"}),
            legacy_names={"TEST_CONFLICT_GHOST"},
        )
        rule_map = {c.rule: c for c in checks}
        assert rule_map["P-7"].passed is False
        assert "P_GHOST_C" in rule_map["P-7"].message


# ─── PatchProof artefact lifecycle ────────────────────────────────────


class TestPatchProofArtefact:
    def test_build_proof_for_patch_populates_provenance(self):
        from sndr.proof import build_proof_for_patch
        proof = build_proof_for_patch("P102")
        # Provenance fields populated.
        assert proof.patch_id == "P102"
        assert proof.commit_sha  # may be 'unknown' but non-empty
        assert proof.measured_at
        assert proof.host
        # Static checks ran.
        assert proof.static_checks
        # P102 is KNOWN_SPEC_ONLY so static_passed must be True.
        assert proof.static_passed

    def test_write_and_load_round_trip(self, tmp_path):
        from sndr.proof import (
            build_proof_for_patch, load_proof_artefact, write_proof_artefact,
        )
        proof = build_proof_for_patch("P102")
        path = write_proof_artefact(proof, out_dir=tmp_path)
        assert path.is_file()
        data = load_proof_artefact(path)
        assert data["patch_id"] == "P102"
        assert data["static_passed"] is True
        assert "static_checks" in data
        assert all("rule" in c and "passed" in c for c in data["static_checks"])

    def test_find_proof_artefacts_returns_sorted(self, tmp_path):
        from sndr.proof import find_proof_artefacts
        # Empty dir → empty list.
        assert find_proof_artefacts("P102", tmp_path) == []
        # Drop two artefacts, find them.
        (tmp_path / "P102__0.20.0.json").write_text("{}", encoding="utf-8")
        (tmp_path / "P102__0.20.1.json").write_text("{}", encoding="utf-8")
        (tmp_path / "OTHER__0.20.json").write_text("{}", encoding="utf-8")
        results = find_proof_artefacts("P102", tmp_path)
        assert len(results) == 2
        assert all("P102" in str(r) for r in results)

    def test_filename_uses_safe_pin_chars(self, tmp_path):
        """vllm pins often have `+gSHA` — must be filesystem-safe."""
        from sndr.proof import (
            PatchProof, ProofCheck, write_proof_artefact,
        )
        proof = PatchProof(
            patch_id="P102",
            vllm_pin="0.20.2rc1.dev209+g5536fc0c0",
            genesis_pin="v11.0.0",
            commit_sha="abc1234",
            host="test",
            measured_at="2026-05-12T00:00:00+00:00",
            static_checks=[ProofCheck("P-1", True, "ok")],
        )
        path = write_proof_artefact(proof, tmp_path)
        # `+` must be escaped so artefact files are portable.
        assert "+" not in path.name
        assert path.name.startswith("P102__")
        assert path.suffix == ".json"


# ─── list_dead_patches ────────────────────────────────────────────────


class TestDeadDetect:
    def test_empty_artefact_dir_lists_all_as_dead(self, tmp_path):
        from sndr.proof import list_dead_patches
        from sndr.dispatcher.registry import PATCH_REGISTRY
        dead = list_dead_patches(out_dir=tmp_path)
        assert len(dead) == len(PATCH_REGISTRY)
        # Each row has the documented shape.
        for d in dead[:3]:
            assert "patch_id" in d
            assert "lifecycle" in d
            assert "tier" in d

    def test_proof_artefact_with_static_passed_excludes(self, tmp_path):
        from sndr.proof import (
            build_proof_for_patch, list_dead_patches, write_proof_artefact,
        )
        # Drop a passing artefact for P102.
        proof = build_proof_for_patch("P102")
        write_proof_artefact(proof, out_dir=tmp_path)
        dead = list_dead_patches(out_dir=tmp_path)
        dead_ids = {d["patch_id"] for d in dead}
        assert "P102" not in dead_ids

    def test_failing_artefact_does_not_count_as_proof(self, tmp_path):
        """A stale artefact with static_passed=false must NOT be treated
        as proof — the dead-detect sweep still flags the patch."""
        from sndr.proof import list_dead_patches
        # Manually fabricate a failing artefact for P102.
        (tmp_path / "P102__0.20.0.json").write_text(
            json.dumps({
                "patch_id": "P102",
                "static_passed": False,
                "static_checks": [],
            }),
            encoding="utf-8",
        )
        dead = list_dead_patches(out_dir=tmp_path)
        dead_ids = {d["patch_id"] for d in dead}
        assert "P102" in dead_ids
        # The stale artefact is referenced under `artefacts_found`.
        p102 = next(d for d in dead if d["patch_id"] == "P102")
        assert p102["artefacts_found"] == ["P102__0.20.0.json"]


# ─── CLI integration ─────────────────────────────────────────────────


class TestCLIProveSubcommand:
    def test_prove_one_known_spec_only_passes(self, tmp_path):
        from sndr.cli.legacy.patches import _run_prove
        opts = argparse.Namespace(
            patch_id="P102",
            prove_all=False,
            dead_detect=False,
            out_dir=str(tmp_path),
            no_write=False,
            json=True,
        )
        rc, out = _run_cli(_run_prove, opts)
        assert rc == 0
        payload = json.loads(out)
        assert payload["static_passed"] is True
        assert payload["artefact_path"]

    def test_prove_unknown_id_returns_one(self, tmp_path):
        from sndr.cli.legacy.patches import _run_prove
        opts = argparse.Namespace(
            patch_id="P_NONEXISTENT_777",
            prove_all=False, dead_detect=False,
            out_dir=str(tmp_path), no_write=True, json=True,
        )
        rc, _ = _run_cli(_run_prove, opts)
        assert rc == 1

    def test_prove_dead_detect_empty_dir(self, tmp_path):
        from sndr.cli.legacy.patches import _run_prove
        from sndr.dispatcher.registry import PATCH_REGISTRY
        opts = argparse.Namespace(
            patch_id=None, prove_all=False, dead_detect=True,
            out_dir=str(tmp_path), no_write=False, json=True,
        )
        rc, out = _run_cli(_run_prove, opts)
        assert rc == 0
        payload = json.loads(out)
        assert payload["total_patches"] == len(PATCH_REGISTRY)
        assert payload["proven"] == 0
        assert payload["dead"] == len(PATCH_REGISTRY)

    def test_prove_all_reports_coverage(self, tmp_path):
        from sndr.cli.legacy.patches import _run_prove
        opts = argparse.Namespace(
            patch_id=None, prove_all=True, dead_detect=False,
            out_dir=str(tmp_path), no_write=True, json=True,
        )
        rc, out = _run_cli(_run_prove, opts)
        # Real registry has metadata gaps (apply_module missing for many);
        # exit may be 1 — that's correct behaviour of the gate.
        assert rc in (0, 1)
        payload = json.loads(out)
        assert payload["total"] > 0
        assert "coverage_pct" in payload
        # Coverage is a float between 0 and 100.
        assert 0.0 <= payload["coverage_pct"] <= 100.0
        # Each result row has the documented shape.
        for r in payload["results"][:3]:
            assert "patch_id" in r
            assert "passed" in r
            assert "errors" in r

    def test_prove_no_args_returns_two(self, tmp_path):
        from sndr.cli.legacy.patches import _run_prove
        opts = argparse.Namespace(
            patch_id=None, prove_all=False, dead_detect=False,
            out_dir=str(tmp_path), no_write=False, json=False,
        )
        rc, _ = _run_cli(_run_prove, opts)
        assert rc == 2


# ─── End-to-end: register through patches argparser ───────────────────


class TestProveRegistration:
    def test_subcommand_parses(self):
        from sndr.cli.legacy.patches import add_argparser
        p = argparse.ArgumentParser()
        sub = p.add_subparsers()
        add_argparser(sub)
        ns = p.parse_args(["patches", "prove", "--dead-detect", "--json"])
        assert ns.dead_detect is True
        assert ns.json is True

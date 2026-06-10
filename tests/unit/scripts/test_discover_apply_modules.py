# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/discover_apply_modules.py`.

Contract: the script walks the legacy register, extracts the wrapped
real-function module, and proposes apply_module values for dispatcher
PATCH_REGISTRY entries that are missing the field.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "discover_apply_modules.py"


def _import_script():
    name = "_discover_apply_modules_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Patch-id extraction regex ────────────────────────────────────────


class TestPidExtraction:
    """Parametrize the leading-P-code regex with names from the real
    legacy register so we don't drift if conventions change."""

    @pytest.mark.parametrize("name,expected", [
        ("P58 Async-scheduler -1 placeholder fix", "P58"),
        ("P67 TQ multi-query kernel", "P67"),
        ("P67b TQ kernel upstream variant", "P67b"),
        ("P15B FA varlen max_seqlen_k clamp", "P15B"),
        ("PN94 ngram fast path", "PN94"),
        ("PN94B ngram followup", "PN94B"),
        ("P107 MTP truncation detector", "P107"),
    ])
    def test_extracts_pid(self, name, expected):
        mod = _import_script()
        assert mod._extract_patch_id(name) == expected

    def test_no_match_returns_none(self):
        mod = _import_script()
        assert mod._extract_patch_id("Sprint 2.6 v2") is None
        assert mod._extract_patch_id("free-form description") is None


# ─── _real_module_for_legacy_fn ────────────────────────────────────────


class TestRealModuleResolution:
    def test_wrapped_function_module(self, genesis_registry):
        """The decorator preserves __wrapped__; we use it to find the
        original function's containing module."""
        from sndr.apply._state import PATCH_REGISTRY as LEG
        mod = _import_script()
        # Smoke: every legacy entry that has __wrapped__ resolves cleanly.
        resolved = 0
        for _name, fn in LEG:
            res = mod._real_module_for_legacy_fn(fn)
            if res:
                resolved += 1
        # Every entry should resolve in practice.
        assert resolved == len(LEG), (
            f"only {resolved}/{len(LEG)} legacy entries resolved a real module"
        )

    def test_function_without_wrapped_returns_none(self):
        mod = _import_script()
        def plain():
            pass
        assert mod._real_module_for_legacy_fn(plain) is None


# ─── build_proposals ──────────────────────────────────────────────────


class TestBuildProposals:
    def test_returns_dataclass_list(self):
        mod = _import_script()
        proposals = mod.build_proposals()
        assert proposals, "no proposals from legacy register — should be 100+"
        sample = proposals[0]
        # Documented dataclass shape.
        for attr in (
            "patch_id", "legacy_name",
            "proposed_apply_module", "current_apply_module", "needs_update",
        ):
            assert hasattr(sample, attr), f"Proposal missing {attr!r}"

    def test_majority_proposes_per_patch_dispatch(self):
        """Most patches live in the legacy monolithic dispatcher today
        — every proposal should point at `_per_patch_dispatch` for now."""
        mod = _import_script()
        proposals = mod.build_proposals()
        # ≥80% of proposals target the legacy module.
        legacy_count = sum(
            1 for p in proposals
            if "sndr.apply._per_patch_dispatch" in p.proposed_apply_module
        )
        assert legacy_count / len(proposals) >= 0.80

    def test_needs_update_flag(self, genesis_registry):
        """Proposals where current != proposed are flagged needs_update."""
        mod = _import_script()
        proposals = mod.build_proposals()
        # All entries currently missing apply_module → needs_update=True.
        for p in proposals:
            if p.current_apply_module is None:
                assert p.needs_update
            elif p.current_apply_module == p.proposed_apply_module:
                assert not p.needs_update


# ─── Emit-py snippet ──────────────────────────────────────────────────


class TestEmitPySnippet:
    def test_snippet_has_dict_assignment(self):
        mod = _import_script()
        proposals = mod.build_proposals()
        text = mod._emit_py_snippet(proposals)
        # Must be importable as Python (no syntax errors).
        compile(text, "<emit-py-test>", "exec")
        # Must contain the canonical dict name.
        assert "PROPOSED_APPLY_MODULES" in text

    def test_snippet_round_trips_through_exec(self):
        mod = _import_script()
        proposals = mod.build_proposals()
        text = mod._emit_py_snippet(proposals)
        ns = {}
        exec(text, ns)
        d = ns["PROPOSED_APPLY_MODULES"]
        # Every needs_update proposal appears in the dict.
        needing = [p for p in proposals if p.needs_update]
        assert set(d.keys()) == {p.patch_id for p in needing}
        # Every value matches.
        for p in needing:
            assert d[p.patch_id] == p.proposed_apply_module


# ─── CLI exit codes + flags ───────────────────────────────────────────


class TestCLI:
    def test_default_exit_zero(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        # Operator-readable output mentions the headline metric.
        assert "Coverage if all proposals applied" in result.stdout

    def test_json_mode_shape(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        # Documented top-level keys.
        for k in ("summary", "proposals"):
            assert k in payload
        # Summary keys.
        for k in (
            "total_patches", "patches_with_legacy_match",
            "patches_needing_update", "coverage_before_pct",
            "coverage_after_pct",
        ):
            assert k in payload["summary"]

    def test_coverage_mode_short_output(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--coverage"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        # `--coverage` skips per-patch listing; the "Proposals" header
        # should NOT appear.
        assert "Proposals (first" not in result.stdout

    def test_emit_py_writes_file(self, tmp_path):
        out = tmp_path / "out.py"
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--emit-py", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert out.is_file()
        text = out.read_text(encoding="utf-8")
        assert "PROPOSED_APPLY_MODULES" in text
        # Must be syntactically valid Python.
        compile(text, str(out), "exec")


# ─── Headline acceptance: coverage projection significantly above today ──


class TestCoverageProjection:
    def test_after_applying_proposals_coverage_above_90pct(self):
        """The whole point of this script — after applying the proposals,
        PATCH_REGISTRY apply_module coverage should jump from current
        baseline (Entry 12: 2.2%) to ≥90%."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        payload = json.loads(result.stdout)
        summary = payload["summary"]
        assert summary["coverage_after_pct"] >= 90.0, summary

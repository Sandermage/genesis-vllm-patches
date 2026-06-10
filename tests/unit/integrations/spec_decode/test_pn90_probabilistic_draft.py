# SPDX-License-Identifier: Apache-2.0
"""Tests for PN90 (probabilistic draft rejection, vllm#40269 backport).

Wave 3.1 (audit closure 2026-05-09).

Tests cover:
  1. Wiring module import (no-torch / CPU-only safe)
  2. Patcher factories return TextPatcher with correct anchors
  3. Anchor strings are well-formed (newline-terminated, contain
     expected idiom)
  4. apply() respects should_apply gate (env-disabled)
  5. apply() detects upstream-merged state when literal `None,
     # draft_probs` is absent

Doesn't test live patch application against vllm — that's covered
by the server-side integration bench.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch as _mock_patch

import pytest


# ─── Module import ──────────────────────────────────────────────────────


class TestImport:
    def test_imports_cleanly_on_cpu(self):
        """Module must import without GPU / live vllm."""
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        assert hasattr(m, "apply")
        assert callable(m.apply)
        assert hasattr(m, "PN90_MARKER_PROPOSER")
        assert hasattr(m, "PN90_MARKER_RUNNER")


# ─── Patcher factory contracts ──────────────────────────────────────────


class TestPatcherFactories:
    def test_proposer_patcher_when_target_resolvable(self, monkeypatch):
        """When `resolve_vllm_file` finds the target, patcher is built."""
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        monkeypatch.setattr(
            "sndr.engines.vllm.patches.spec_decode."
            "pn90_probabilistic_draft_rejection.resolve_vllm_file",
            lambda relpath: "/fake/" + relpath,
        )
        patcher = m._make_patcher_proposer()
        assert patcher is not None
        assert patcher.target_file.endswith("llm_base_proposer.py")
        # Three sub-patches per docstring contract
        assert len(patcher.sub_patches) == 3
        # Marker is non-empty
        assert patcher.marker
        # Drift markers populated
        assert len(patcher.upstream_drift_markers) > 0

    def test_proposer_patcher_when_target_missing(self, monkeypatch):
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        monkeypatch.setattr(
            "sndr.engines.vllm.patches.spec_decode."
            "pn90_probabilistic_draft_rejection.resolve_vllm_file",
            lambda relpath: None,
        )
        patcher = m._make_patcher_proposer()
        assert patcher is None

    def test_runner_patcher_when_target_resolvable(self, monkeypatch):
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        monkeypatch.setattr(
            "sndr.engines.vllm.patches.spec_decode."
            "pn90_probabilistic_draft_rejection.resolve_vllm_file",
            lambda relpath: "/fake/" + relpath,
        )
        patcher = m._make_patcher_runner()
        assert patcher is not None
        assert patcher.target_file.endswith("gpu_model_runner.py")
        assert len(patcher.sub_patches) == 1


# ─── Anchor strings well-formed ─────────────────────────────────────────


class TestAnchorStrings:
    def test_greedy_sample_anchor_contains_expected_idiom(self):
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        # Old anchor must contain the upstream argmax-discard pattern
        assert "compute_logits(hidden_states).argmax" in m.PN90_GREEDY_SAMPLE_OLD
        assert "use_local_argmax_reduction" in m.PN90_GREEDY_SAMPLE_OLD
        # New anchor preserves both branches
        assert "compute_logits(hidden_states)" in m.PN90_GREEDY_SAMPLE_NEW
        assert "softmax(dim=-1, dtype=torch.float32)" in m.PN90_GREEDY_SAMPLE_NEW
        assert "_pn90_step_probs_buf" in m.PN90_GREEDY_SAMPLE_NEW

    def test_propose_entry_anchor_aligned(self):
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        assert "batch_size = common_attn_metadata.batch_size()" in m.PN90_PROPOSE_ENTRY_OLD
        # New version resets buffer BEFORE batch_size assignment
        assert "_pn90_step_probs_buf = []" in m.PN90_PROPOSE_ENTRY_NEW
        assert "_pn90_draft_probs = None" in m.PN90_PROPOSE_ENTRY_NEW

    def test_propose_exit_anchor_aligned(self):
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        assert "torch.stack(draft_token_ids_list, dim=1)" in m.PN90_PROPOSE_EXIT_OLD
        assert "return draft_token_ids" in m.PN90_PROPOSE_EXIT_OLD
        # New version stacks probs and stores 2D layout
        assert ".contiguous().view(-1, _pn90_vocab)" in m.PN90_PROPOSE_EXIT_NEW
        assert "self._pn90_draft_probs" in m.PN90_PROPOSE_EXIT_NEW

    def test_runner_anchor_targets_literal_none(self):
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        # Must target the EXACT `None,  # draft_probs` line so we don't
        # accidentally match an unrelated rejection_sampler call
        assert "None,  # draft_probs" in m.PN90_RUNNER_OLD
        assert "rejection_sampler(" in m.PN90_RUNNER_OLD
        # New version reads from drafter attribute
        assert "_pn90_draft_probs" in m.PN90_RUNNER_NEW
        assert "getattr(self.drafter" in m.PN90_RUNNER_NEW


# ─── apply() gate via should_apply ──────────────────────────────────────


class TestApplyGate:
    def test_apply_skips_when_should_apply_false(self, monkeypatch):
        """When dispatcher gate returns False, apply() bails early."""
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        monkeypatch.setattr(
            "sndr.dispatcher.should_apply",
            lambda pid: (False, "opt-in only — env unset"),
        )
        # Don't even need vllm install root probe
        status, reason = m.apply()
        assert status == "skipped"
        assert "opt-in" in reason

    def test_apply_skips_when_vllm_root_missing(self, monkeypatch):
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        monkeypatch.setattr(
            "sndr.dispatcher.should_apply",
            lambda pid: (True, "env enabled"),
        )
        monkeypatch.setattr(
            "sndr.engines.vllm.patches.spec_decode."
            "pn90_probabilistic_draft_rejection.vllm_install_root",
            lambda: None,
        )
        status, reason = m.apply()
        assert status == "skipped"
        assert "vllm install root" in reason


# ─── Upstream-merged self-retire ────────────────────────────────────────


class TestUpstreamMergedDetect:
    def test_apply_self_retires_when_none_literal_absent(self, tmp_path, monkeypatch):
        """If gpu_model_runner.py no longer contains `None,  # draft_probs`,
        upstream restructured the call site → PN90 retires gracefully."""
        from sndr.engines.vllm.patches.spec_decode import (
            pn90_probabilistic_draft_rejection as m,
        )
        # Build a fake runner file WITHOUT the literal
        fake_runner = tmp_path / "gpu_model_runner.py"
        fake_runner.write_text(
            "# pretend this is gpu_model_runner.py\n"
            "self.rejection_sampler(meta, draft_probs=other_source, logits, sm)\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "sndr.dispatcher.should_apply",
            lambda pid: (True, "env enabled"),
        )
        monkeypatch.setattr(
            "sndr.engines.vllm.patches.spec_decode."
            "pn90_probabilistic_draft_rejection.vllm_install_root",
            lambda: tmp_path,
        )
        monkeypatch.setattr(
            "sndr.engines.vllm.patches.spec_decode."
            "pn90_probabilistic_draft_rejection.resolve_vllm_file",
            lambda relpath: (
                str(fake_runner) if "gpu_model_runner" in relpath else None
            ),
        )
        status, reason = m.apply()
        assert status == "skipped"
        assert "upstream_merged" in reason


# ─── Registry presence ──────────────────────────────────────────────────


class TestRegistryPresence:
    def test_pn90_in_registry(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "PN90" in PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN90"]
        assert meta["tier"] == "community"
        assert meta["family"] == "spec_decode"
        assert meta["env_flag"] == "GENESIS_ENABLE_PN90_PROBABILISTIC_DRAFT"
        assert meta["default_on"] is False
        assert meta["upstream_pr"] == 40269
        assert meta.get("implementation_status") == "full"

    def test_pn90_apply_module_resolves(self):
        from sndr.dispatcher.spec import iter_patch_specs
        for spec in iter_patch_specs():
            if spec.patch_id == "PN90":
                assert spec.apply_module is not None
                assert "pn90_probabilistic_draft_rejection" in spec.apply_module
                return
        pytest.fail("PN90 not found in iter_patch_specs()")

    def test_pn90_dispatcher_wrapper_present(self):
        from sndr.apply._per_patch_dispatch import (
            apply_patch_N90_probabilistic_draft_rejection,
        )
        assert callable(apply_patch_N90_probabilistic_draft_rejection)

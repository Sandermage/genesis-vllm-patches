# SPDX-License-Identifier: Apache-2.0
"""TDD for PN82 — vllm#41873 backport.

Zero `is_prefilling` padded CUDA-graph rows so Mamba/hybrid attention
backends don't read stale True values from `condense()` rotation.

PR38 Day 1 (2026-05-07): patch landed default OFF.

This test lives at the canonical `tests/unit/integrations/worker/` location
(Stage 9 design) — `_genesis/tests/` is being phased out per Sander's
2026-05-07 directive to remove the `_genesis/` shim entirely.
"""
from __future__ import annotations

import pytest


def _wiring():
    """Resolve PN82 wiring via the canonical SNDR Core path."""
    from sndr.engines.vllm._archive import (
        pn82_mamba_cudagraph_prefill_zero as M,
    )
    return M


# ─── Anchor / replacement shape ────────────────────────────────────────────


class TestAnchorShape:
    def test_anchor_targets_real_assignment(self):
        M = _wiring()
        assert (
            "is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu\n"
            in M.PN82_ANCHOR
        )
        assert "Used by mamba backends" in M.PN82_ANCHOR

    def test_replacement_adds_padded_zero_line(self):
        M = _wiring()
        assert "is_prefilling[num_reqs:] = False\n" in M.PN82_REPLACEMENT
        assert "Genesis PN82" in M.PN82_REPLACEMENT
        assert "vllm#41873" in M.PN82_REPLACEMENT

    def test_replacement_strictly_extends_anchor(self):
        """Replacement must contain the full anchor verbatim — patch
        is purely additive. Catches typos that would silently rewrite
        the original line."""
        M = _wiring()
        assert M.PN82_ANCHOR in M.PN82_REPLACEMENT


# ─── Idempotency on synthetic file ─────────────────────────────────────────


class TestIdempotent:
    def test_apply_twice_is_no_op(self, tmp_path):
        from sndr.kernel import (
            TextPatch, TextPatcher, TextPatchResult,
        )
        M = _wiring()
        target = tmp_path / "gpu_model_runner.py"
        target.write_text("# header\n" + M.PN82_ANCHOR + "# footer\n")
        patcher = TextPatcher(
            patch_name="PN82 test",
            target_file=str(target),
            marker=M.GENESIS_PN82_MARKER,
            sub_patches=[
                TextPatch(
                    name="pn82",
                    anchor=M.PN82_ANCHOR,
                    replacement=M.PN82_REPLACEMENT,
                    required=True,
                ),
            ],
        )
        r1, _ = patcher.apply()
        assert r1 == TextPatchResult.APPLIED
        body1 = target.read_text()
        assert "PN82" in body1
        assert "is_prefilling[num_reqs:] = False" in body1

        r2, _ = patcher.apply()
        assert r2 == TextPatchResult.IDEMPOTENT
        assert target.read_text() == body1


# ─── Env-flag gating contract ──────────────────────────────────────────────


class TestEnvFlag:
    def test_default_off(self, monkeypatch):
        from sndr.dispatcher import should_apply
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO",
            raising=False,
        )
        monkeypatch.delenv(
            "SNDR_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO",
            raising=False,
        )
        decision, _ = should_apply("PN82")
        assert decision is False

    def test_genesis_enable_engages(self, monkeypatch):
        from sndr.dispatcher import should_apply
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO", "1"
        )
        decision, _ = should_apply("PN82")
        assert decision is True

    def test_sndr_enable_engages_via_alias(self, monkeypatch):
        """F-008 alias: SNDR_ENABLE_* should work the same as GENESIS_."""
        from sndr.dispatcher import should_apply
        monkeypatch.delenv(
            "GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO",
            raising=False,
        )
        monkeypatch.setenv(
            "SNDR_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO", "1"
        )
        decision, _ = should_apply("PN82")
        assert decision is True


# ─── Registry contract ─────────────────────────────────────────────────────


class TestRegistry:
    def test_pn82_in_registry(self):
        from sndr.dispatcher import PATCH_REGISTRY
        assert "PN82" in PATCH_REGISTRY

    def test_pn82_metadata_complete(self):
        from sndr.dispatcher import PATCH_REGISTRY
        meta = PATCH_REGISTRY["PN82"]
        assert meta["upstream_pr"] == 41873
        assert meta["family"] == "worker"
        assert meta["tier"] == "community"
        assert meta["default_on"] is False
        assert meta["env_flag"] == "GENESIS_ENABLE_PN82_MAMBA_CUDAGRAPH_PREFILL_ZERO"
        # Hybrid-only — dense MoE configs should skip
        assert meta["applies_to"] == {"is_hybrid": [True]}

    def test_pn82_dispatch_function_registered(self):
        from sndr.apply import _per_patch_dispatch
        assert hasattr(
            _per_patch_dispatch,
            "apply_patch_N82_mamba_cudagraph_prefill_zero",
        )


# ─── Drift markers detect upstream merge ───────────────────────────────────


class TestUpstreamDrift:
    def test_drift_marker_includes_genesis_marker(self):
        M = _wiring()
        patcher = M._make_patcher()
        if patcher is None:
            pytest.skip("vllm install root not discoverable on this host")
        assert any("[Genesis PN82" in m for m in patcher.upstream_drift_markers)

    def test_drift_marker_detects_upstream_merge(self):
        """If vllm merges PR #41873, the modified file will literally
        contain `is_prefilling[num_reqs:] = False`. The drift markers
        must include that exact line so PN82 self-retires on pin bump."""
        M = _wiring()
        patcher = M._make_patcher()
        if patcher is None:
            pytest.skip("vllm install root not discoverable on this host")
        assert any(
            "is_prefilling[num_reqs:] = False" in m
            for m in patcher.upstream_drift_markers
        )

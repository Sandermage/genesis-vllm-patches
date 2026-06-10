# SPDX-License-Identifier: Apache-2.0
"""TDD for PN72 — frequency-based ngram draft filter.

Pure-helper tests. No torch / vllm / numba required — helper is numpy-only
and importable offline.

Strategy: post-filter ngram drafts by counting how many times the proposed
first-draft-token appears in the recent window of context. Reject draft
when count < MIN_OBS (likely spurious match from chat-template tokens).

Conservative: never invents a draft, only rejects existing weak ones.
Composes additively with P70 hardcoded `prompt_lookup_min=8`.
"""
from __future__ import annotations

import numpy as np
import pytest


# Helper API contract (will be implemented in Etap 3):
#
#   from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
#       should_accept_draft, filter_drafts_by_frequency,
#   )
#
# should_accept_draft(context: np.ndarray, first_draft_token: int,
#                     window: int = 1024, min_obs: int = 4) -> bool
# filter_drafts_by_frequency(drafts: list[list[int]],
#                            num_tokens_no_spec: np.ndarray,
#                            token_ids_cpu: np.ndarray,
#                            window: int = 1024, min_obs: int = 4) -> list[list[int]]


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ─── should_accept_draft (single-request primitive) ────────────────────


class TestShouldAcceptDraft:
    """Decision primitive: does first draft token appear ≥ min_obs in window?"""

    def test_dominant_token_accepted(self):
        """Token appearing 5 times in last 1024 → accepted."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            should_accept_draft,
        )
        # context has token 42 appearing 5 times in last 100
        ctx = np.array([1, 2, 3] * 100 + [42, 5, 42, 6, 42, 7, 42, 8, 42, 9],
                       dtype=np.int32)
        assert should_accept_draft(ctx, first_draft_token=42,
                                    window=1024, min_obs=4) is True

    def test_rare_token_rejected(self):
        """Token appearing 2 times in window → rejected (< min_obs=4)."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            should_accept_draft,
        )
        ctx = np.array([1, 2, 3] * 100 + [42, 5, 99, 6, 42, 7, 8, 9],
                       dtype=np.int32)
        assert should_accept_draft(ctx, first_draft_token=42,
                                    window=1024, min_obs=4) is False

    def test_token_outside_window_rejected(self):
        """Token at position 0 (way outside last-1024 window) → rejected."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            should_accept_draft,
        )
        # 42 only at pos 0, then 2000 of zeros
        ctx = np.concatenate([np.array([42], dtype=np.int32),
                              np.zeros(2000, dtype=np.int32)])
        assert should_accept_draft(ctx, first_draft_token=42,
                                    window=1024, min_obs=4) is False

    def test_empty_context_rejected(self):
        """Empty context → reject (no observations possible)."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            should_accept_draft,
        )
        ctx = np.array([], dtype=np.int32)
        assert should_accept_draft(ctx, first_draft_token=42,
                                    window=1024, min_obs=4) is False

    def test_window_larger_than_context_uses_full(self):
        """If window > len(context), use full context — don't crash."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            should_accept_draft,
        )
        ctx = np.array([42, 1, 42, 2, 42, 3, 42, 4, 42], dtype=np.int32)
        # 5 occurrences, window=1024, but context only 9 → all 5 count
        assert should_accept_draft(ctx, first_draft_token=42,
                                    window=1024, min_obs=4) is True

    def test_min_obs_zero_always_accepts(self):
        """min_obs=0 → always accept (filter disabled)."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            should_accept_draft,
        )
        ctx = np.array([1, 2, 3], dtype=np.int32)
        assert should_accept_draft(ctx, first_draft_token=99,
                                    window=1024, min_obs=0) is True


# ─── filter_drafts_by_frequency (batch wrapper) ────────────────────────


class TestFilterDraftsByFrequency:
    """Batch operation matching NgramProposer.propose() signature."""

    def test_empty_drafts_passthrough(self):
        """No drafts → no filtering, return as-is."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            filter_drafts_by_frequency,
        )
        drafts = [[], [], []]
        ntok = np.array([10, 20, 30], dtype=np.int32)
        token_ids = np.zeros((3, 100), dtype=np.int32)
        out = filter_drafts_by_frequency(drafts, ntok, token_ids,
                                          window=1024, min_obs=4)
        assert out == [[], [], []]

    def test_strong_draft_kept(self):
        """Draft whose first token is dominant → kept."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            filter_drafts_by_frequency,
        )
        token_ids = np.zeros((1, 100), dtype=np.int32)
        # Place token 42 five times in valid window (first 50)
        token_ids[0, :10] = [1, 42, 2, 42, 3, 42, 4, 42, 5, 42]
        ntok = np.array([10], dtype=np.int32)
        drafts = [[42, 99]]  # first-token 42 = dominant
        out = filter_drafts_by_frequency(drafts, ntok, token_ids,
                                          window=1024, min_obs=4)
        assert out == [[42, 99]]

    def test_weak_draft_rejected(self):
        """Draft whose first token is rare → replaced with empty."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            filter_drafts_by_frequency,
        )
        token_ids = np.zeros((1, 100), dtype=np.int32)
        # Token 99 only appears once in context
        token_ids[0, :5] = [1, 2, 99, 3, 4]
        ntok = np.array([5], dtype=np.int32)
        drafts = [[99, 88, 77]]  # first-token 99 only seen 1× → reject
        out = filter_drafts_by_frequency(drafts, ntok, token_ids,
                                          window=1024, min_obs=4)
        assert out == [[]]

    def test_mixed_batch_filtered_independently(self):
        """One strong + one weak in same batch → independent decisions."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            filter_drafts_by_frequency,
        )
        token_ids = np.zeros((2, 100), dtype=np.int32)
        # Request 0: token 42 is dominant
        token_ids[0, :10] = [42, 1, 42, 2, 42, 3, 42, 4, 42, 5]
        # Request 1: token 99 is rare
        token_ids[1, :5] = [99, 1, 2, 3, 4]
        ntok = np.array([10, 5], dtype=np.int32)
        drafts = [[42, 1], [99, 1]]
        out = filter_drafts_by_frequency(drafts, ntok, token_ids,
                                          window=1024, min_obs=4)
        assert out == [[42, 1], []]

    def test_helper_does_not_mutate_inputs(self):
        """Defensive: input drafts list and arrays unchanged after call."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            filter_drafts_by_frequency,
        )
        token_ids = np.zeros((1, 50), dtype=np.int32)
        token_ids[0, :5] = [1, 2, 99, 3, 4]
        ntok = np.array([5], dtype=np.int32)
        original_drafts = [[99, 88]]
        token_ids_copy = token_ids.copy()
        ntok_copy = ntok.copy()

        out = filter_drafts_by_frequency(original_drafts, ntok, token_ids,
                                          window=1024, min_obs=4)

        # Inputs untouched
        assert original_drafts == [[99, 88]]
        np.testing.assert_array_equal(token_ids, token_ids_copy)
        np.testing.assert_array_equal(ntok, ntok_copy)
        # Out is a new list
        assert out is not original_drafts


# ─── Window edge cases ───────────────────────────────────────────────


class TestWindowSizing:
    def test_window_exactly_at_boundary(self):
        """Token at position (len - window) is included."""
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            should_accept_draft,
        )
        # 10 occurrences of 42 in first 100 tokens, then 1024 zeros
        ctx = np.concatenate([
            np.tile([42, 0], 10),  # 20 elements: 10× of 42
            np.zeros(1024, dtype=np.int32),
        ]).astype(np.int32)
        # window=1024 → look at last 1024 → all zeros → 0 of 42 → reject
        assert should_accept_draft(ctx, first_draft_token=42,
                                    window=1024, min_obs=4) is False
        # window=2048 → look at all → 10 of 42 → accept
        assert should_accept_draft(ctx, first_draft_token=42,
                                    window=2048, min_obs=4) is True


# ─── Configuration via env (gated through helper) ──────────────────────


class TestEnvConfiguration:
    """Helper exposes env-reading config getters that wiring will call."""

    def test_get_min_observations_default(self, monkeypatch):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            get_min_observations,
        )
        monkeypatch.delenv("GENESIS_PN72_MIN_OBSERVATIONS", raising=False)
        assert get_min_observations() == 4

    def test_get_min_observations_env_override(self, monkeypatch):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            get_min_observations,
        )
        monkeypatch.setenv("GENESIS_PN72_MIN_OBSERVATIONS", "8")
        assert get_min_observations() == 8

    def test_get_min_observations_invalid_falls_to_default(self, monkeypatch):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            get_min_observations,
        )
        monkeypatch.setenv("GENESIS_PN72_MIN_OBSERVATIONS", "abc")
        assert get_min_observations() == 4

    def test_get_min_observations_negative_falls_to_default(self, monkeypatch):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            get_min_observations,
        )
        monkeypatch.setenv("GENESIS_PN72_MIN_OBSERVATIONS", "-3")
        assert get_min_observations() == 4

    def test_get_window_default(self, monkeypatch):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            get_frequency_window,
        )
        monkeypatch.delenv("GENESIS_PN72_FREQUENCY_WINDOW", raising=False)
        assert get_frequency_window() == 1024

    def test_get_window_env_override(self, monkeypatch):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            get_frequency_window,
        )
        monkeypatch.setenv("GENESIS_PN72_FREQUENCY_WINDOW", "2048")
        assert get_frequency_window() == 2048

    def test_is_enabled_default_off(self, monkeypatch):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import is_enabled
        monkeypatch.delenv("GENESIS_ENABLE_PN72_FREQUENCY_NGRAM_DRAFTER",
                           raising=False)
        assert is_enabled() is False

    def test_is_enabled_via_env(self, monkeypatch):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import is_enabled
        monkeypatch.setenv(
            "GENESIS_ENABLE_PN72_FREQUENCY_NGRAM_DRAFTER", "1",
        )
        assert is_enabled() is True


# ─── Defensive: never raise on weird inputs ────────────────────────────


class TestDefensiveBehavior:
    """Wrapper must never raise — graceful degradation matters more than perf."""

    def test_negative_num_tokens_safe(self):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            filter_drafts_by_frequency,
        )
        token_ids = np.zeros((1, 50), dtype=np.int32)
        ntok = np.array([-5], dtype=np.int32)
        drafts = [[42]]
        # Should not crash; clamp to 0 → empty context → reject
        out = filter_drafts_by_frequency(drafts, ntok, token_ids,
                                          window=1024, min_obs=4)
        assert out == [[]]

    def test_num_tokens_exceeds_array_safe(self):
        from sndr.engines.vllm.kernels_legacy.ngram_frequency_filter import (
            filter_drafts_by_frequency,
        )
        token_ids = np.zeros((1, 10), dtype=np.int32)
        token_ids[0, :5] = [42, 42, 42, 42, 42]
        ntok = np.array([100], dtype=np.int32)  # > shape[1]=10
        drafts = [[42]]
        # Should clamp to 10, find 5× of 42 → accept
        out = filter_drafts_by_frequency(drafts, ntok, token_ids,
                                          window=1024, min_obs=4)
        assert out == [[42]]

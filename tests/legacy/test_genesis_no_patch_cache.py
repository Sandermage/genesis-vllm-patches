# SPDX-License-Identifier: Apache-2.0
"""Tests for `genesis_no_patch_cache()` env helper — P1.3 of patcher
evolution plan (2026-05-07). The operator escape hatch that future
P2.1/P2.2 cache layers MUST honor.

The helper itself is trivial; tests document the contract:
  - default False (no env or empty)
  - truthy: 1, true, yes, on (case-insensitive, whitespace-tolerant)
  - falsy: anything else (0, false, no, off, gibberish)
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Each test starts with a clean GENESIS_NO_PATCH_CACHE env."""
    monkeypatch.delenv("GENESIS_NO_PATCH_CACHE", raising=False)


class TestGenesisNoPatchCache:

    def test_default_false_when_env_unset(self):
        from sndr.engines.vllm.detection.guards import genesis_no_patch_cache
        assert genesis_no_patch_cache() is False

    def test_default_false_when_env_empty(self, monkeypatch):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "")
        from sndr.engines.vllm.detection.guards import genesis_no_patch_cache
        assert genesis_no_patch_cache() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on",
                                     "TRUE", "Yes", "On", "tRuE"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", val)
        from sndr.engines.vllm.detection.guards import genesis_no_patch_cache
        assert genesis_no_patch_cache() is True, f"{val!r} should disable cache"

    @pytest.mark.parametrize("val", [" 1 ", "\ttrue\n", " yes  "])
    def test_truthy_values_with_whitespace(self, monkeypatch, val):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", val)
        from sndr.engines.vllm.detection.guards import genesis_no_patch_cache
        assert genesis_no_patch_cache() is True, f"{val!r} should disable cache"

    @pytest.mark.parametrize("val", ["0", "false", "no", "off",
                                     "FALSE", "No", "Off"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", val)
        from sndr.engines.vllm.detection.guards import genesis_no_patch_cache
        assert genesis_no_patch_cache() is False, f"{val!r} should keep cache"

    @pytest.mark.parametrize("val", ["enabled", "wat", "lol", "active",
                                     "y", "t", "✓"])
    def test_unrecognized_values_default_false(self, monkeypatch, val):
        """Strict whitelist — only the canonical 4 values disable cache.
        This is intentional: prevents accidental disable on typo or
        adjacent spelling."""
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", val)
        from sndr.engines.vllm.detection.guards import genesis_no_patch_cache
        assert genesis_no_patch_cache() is False, \
            f"{val!r} should NOT match canonical truthy whitelist"

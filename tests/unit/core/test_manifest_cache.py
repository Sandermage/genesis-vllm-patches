# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.kernel.manifest_cache` — Site Map fast-path.

Contract:

  1. cached_load_manifest is lazy: first call attempts load, subsequent
     calls return the cached result.
  2. Sentinel _MANIFEST_INVALID prevents retry storm — once load fails,
     subsequent calls return None without re-attempting the file IO.
  3. reset_manifest_cache_for_tests restores _MANIFEST_NOT_LOADED state.
  4. derive_rel_path_from_target strips the vllm install prefix.
  5. derive_rel_path_from_target returns None on inputs without vllm/.
  6. md5_bytes returns 32-char hex.
  7. Back-compat aliases (_cached_load_manifest etc.) point at the
     same functions.
"""
from __future__ import annotations

import hashlib

import pytest

from sndr.kernel import manifest as mc


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    """Each test runs against a fresh cache state."""
    mc.reset_manifest_cache_for_tests()
    yield
    mc.reset_manifest_cache_for_tests()


# ─── derive_rel_path_from_target ──────────────────────────────────────


class TestDeriveRelPath:
    def test_strips_vllm_install_prefix(self):
        result = mc.derive_rel_path_from_target(
            "/usr/local/lib/python3.12/dist-packages/vllm/"
            "model_executor/foo.py"
        )
        assert result == "model_executor/foo.py"

    def test_handles_deep_path(self):
        result = mc.derive_rel_path_from_target(
            "/x/y/z/vllm/v1/attention/backends/turboquant_attn.py"
        )
        assert result == "v1/attention/backends/turboquant_attn.py"

    def test_last_vllm_wins_on_repeated_segments(self):
        """When `vllm` appears twice in the path, take everything after the
        LAST occurrence."""
        result = mc.derive_rel_path_from_target(
            "/home/vllm/code/vllm/model_executor/x.py"
        )
        assert result == "model_executor/x.py"

    def test_returns_none_on_no_vllm_segment(self):
        result = mc.derive_rel_path_from_target("/tmp/foo/bar.py")
        assert result is None

    def test_returns_none_on_empty_input(self):
        assert mc.derive_rel_path_from_target("") is None

    def test_returns_none_when_vllm_is_last_segment(self):
        """Path ending in `vllm` with nothing after → no rel parts."""
        assert mc.derive_rel_path_from_target("/usr/local/lib/vllm") is None

    def test_uses_forward_slashes(self):
        """Result is posix-style for manifest key consistency."""
        result = mc.derive_rel_path_from_target(
            "/x/vllm/a/b/c.py"
        )
        assert "/" in result
        assert "\\" not in result


# ─── md5_bytes ────────────────────────────────────────────────────────


class TestMd5Bytes:
    def test_returns_32_char_hex(self):
        result = mc.md5_bytes(b"hello")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_known_value_empty(self):
        # MD5 of empty bytes
        result = mc.md5_bytes(b"")
        assert result == "d41d8cd98f00b204e9800998ecf8427e"

    def test_consistent_with_hashlib(self):
        data = b"the quick brown fox"
        assert mc.md5_bytes(data) == hashlib.md5(data).hexdigest()


# ─── cached_load_manifest semantics ──────────────────────────────────


class TestCachedLoadManifest:
    def test_first_call_attempts_load(self, monkeypatch):
        """First call triggers the load logic (which may return None)."""
        # Force load path to return a known dict via monkeypatch
        sentinel = {"pins": {"vllm": "x"}, "files": {}}
        monkeypatch.setattr(
            "sndr.engines.vllm.wiring.anchor_manifest.load_manifest_for_pins",
            lambda *a, **kw: sentinel,
        )
        result = mc.cached_load_manifest()
        assert result is sentinel

    def test_subsequent_calls_return_cached_dict(self, monkeypatch):
        """Second call returns the cached object — no reload."""
        sentinel = {"cached": True}
        call_count = {"n": 0}

        def fake_load(*a, **kw):
            call_count["n"] += 1
            return sentinel

        monkeypatch.setattr(
            "sndr.engines.vllm.wiring.anchor_manifest.load_manifest_for_pins",
            fake_load,
        )
        first = mc.cached_load_manifest()
        second = mc.cached_load_manifest()
        assert first is second is sentinel
        # load_manifest_for_pins should be called exactly once
        assert call_count["n"] == 1

    def test_failed_load_caches_invalid_sentinel(self, monkeypatch):
        """First failed load → next call returns None without retry."""
        call_count = {"n": 0}

        def fake_load(*a, **kw):
            call_count["n"] += 1
            return None

        monkeypatch.setattr(
            "sndr.engines.vllm.wiring.anchor_manifest.load_manifest_for_pins",
            fake_load,
        )
        first = mc.cached_load_manifest()
        second = mc.cached_load_manifest()
        assert first is None and second is None
        # No retry storm — second call doesn't re-attempt
        assert call_count["n"] == 1

    def test_reset_for_tests_re_enables_load(self, monkeypatch):
        """After reset, the next call re-attempts the load."""
        call_count = {"n": 0}

        def fake_load(*a, **kw):
            call_count["n"] += 1
            return None

        monkeypatch.setattr(
            "sndr.engines.vllm.wiring.anchor_manifest.load_manifest_for_pins",
            fake_load,
        )
        mc.cached_load_manifest()
        mc.cached_load_manifest()
        assert call_count["n"] == 1  # cached
        mc.reset_manifest_cache_for_tests()
        mc.cached_load_manifest()
        assert call_count["n"] == 2  # re-attempted


# ─── Back-compat aliases ──────────────────────────────────────────────


class TestBackCompatAliases:
    def test_underscore_aliases_match_public_names(self):
        assert mc._cached_load_manifest is mc.cached_load_manifest
        assert mc._reset_manifest_cache_for_tests is mc.reset_manifest_cache_for_tests
        assert mc._derive_rel_path_from_target is mc.derive_rel_path_from_target
        assert mc._md5_bytes is mc.md5_bytes

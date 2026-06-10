# SPDX-License-Identifier: Apache-2.0
"""Integration tests for TextPatcher Layer 0 fast-path (P2.2).

Verifies the persistent file_cache short-circuits Layer 1+2+3 on
warm restart, AND that Layer 0 hits produce identical-to-legacy
output (graceful equivalence).

Related test files:
  test_file_cache.py — pure file_cache module unit tests
  test_text_patcher_manifest_aware.py — Phase 3 P2.1 (Layer 4.5)
  test_anchor_manifest.py — P2.1 Phase 1+2 (manifest schema/builder)

Coverage:
  TestLayer0Hit — second apply on already-patched file uses Layer 0
  TestLayer0NoCacheFirstBoot — fresh apply populates cache for next boot
  TestLayer0EnvDisables — GENESIS_NO_PATCH_CACHE=1 forces Layer 1+
  TestLayer0FileChangedAfterCache — mtime change → cache miss → Layer 1+
  TestLayer0EquivalenceWithLegacy — Layer 0 output ≡ no-cache output
  TestLayer0DoesNotBreakIDEMPOTENT — Layer 2 still works when Layer 0 misses
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """Each test gets its own cache file in tmp_path."""
    test_cache = tmp_path / "_test_cache.json"
    monkeypatch.setenv("GENESIS_FILE_CACHE_PATH", str(test_cache))
    monkeypatch.delenv("GENESIS_NO_PATCH_CACHE", raising=False)
    from sndr.engines.vllm.wiring import file_cache, text_patch
    file_cache._reset_for_tests()
    text_patch._reset_manifest_cache_for_tests()
    yield test_cache
    file_cache._reset_for_tests()
    text_patch._reset_manifest_cache_for_tests()


def _make_patcher(target: Path, marker: str = "TEST_MARK"):
    """Build a simple single-anchor TextPatcher for tests."""
    from sndr.kernel.text_patch import TextPatch, TextPatcher
    return TextPatcher(
        patch_name="test patcher",
        target_file=str(target),
        marker=marker,
        sub_patches=[
            TextPatch(name="s1", anchor="alpha", replacement="ALPHA",
                      required=True),
        ],
    )


# ═════════════════════════════════════════════════════════════════════════
# 1. TestLayer0NoCacheFirstBoot
# ═════════════════════════════════════════════════════════════════════════


class TestLayer0NoCacheFirstBoot:
    """First boot — cache absent. Apply should fall through Layer 1+2+3,
    succeed via Layer 5 legacy, and leave cache populated for next boot."""

    def test_first_apply_returns_applied(self, tmp_path):
        from sndr.kernel.text_patch import TextPatchResult
        target = tmp_path / "f.py"
        target.write_text("alpha\n")
        result, _ = _make_patcher(target).apply()
        assert result == TextPatchResult.APPLIED
        assert "ALPHA" in target.read_text()

    def test_first_apply_populates_cache(self, tmp_path):
        from sndr.engines.vllm.wiring.file_cache import is_marker_cached_present
        target = tmp_path / "f.py"
        target.write_text("alpha\n")
        # Before apply: cache miss
        assert is_marker_cached_present(str(target), "TEST_MARK") is False
        _make_patcher(target).apply()
        # After apply: cache hit
        assert is_marker_cached_present(str(target), "TEST_MARK") is True


# ═════════════════════════════════════════════════════════════════════════
# 2. TestLayer0Hit — warm restart fast path
# ═════════════════════════════════════════════════════════════════════════


class TestLayer0Hit:
    """Apply twice on same file. Second apply hits Layer 0 — IDEMPOTENT
    without reading file."""

    def test_second_apply_is_idempotent_via_layer0(self, tmp_path):
        from sndr.kernel.text_patch import TextPatchResult
        target = tmp_path / "f.py"
        target.write_text("alpha\n")
        _make_patcher(target).apply()  # first apply

        # Second apply — Layer 0 should hit
        result2, _ = _make_patcher(target).apply()
        assert result2 == TextPatchResult.IDEMPOTENT

    def test_layer0_hit_does_not_modify_file(self, tmp_path):
        from sndr.kernel.text_patch import TextPatchResult
        target = tmp_path / "f.py"
        target.write_text("alpha\n")
        _make_patcher(target).apply()
        body_after_first = target.read_text()
        mtime_after_first = target.stat().st_mtime_ns

        # Second apply — Layer 0 hit should NOT touch file
        time.sleep(0.01)  # give mtime resolution a chance
        _make_patcher(target).apply()
        body_after_second = target.read_text()
        mtime_after_second = target.stat().st_mtime_ns

        assert body_after_first == body_after_second
        assert mtime_after_first == mtime_after_second  # no write happened


# ═════════════════════════════════════════════════════════════════════════
# 3. TestLayer0EnvDisables — escape hatch
# ═════════════════════════════════════════════════════════════════════════


class TestLayer0EnvDisables:
    """GENESIS_NO_PATCH_CACHE=1 must force Layer 1+ even when Layer 0
    would have hit. Operator-controlled override."""

    def test_env_set_skips_layer0(self, tmp_path, monkeypatch):
        from sndr.kernel.text_patch import TextPatchResult
        target = tmp_path / "f.py"
        target.write_text("alpha\n")
        _make_patcher(target).apply()  # first apply, cache populated

        # Second apply WITH env=1 — should still return IDEMPOTENT but
        # via Layer 2 (full file read), not Layer 0.
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")
        result2, _ = _make_patcher(target).apply()
        assert result2 == TextPatchResult.IDEMPOTENT
        # We can't easily detect "was Layer 0 used vs Layer 2" from
        # outside, but the test confirms env doesn't break IDEMPOTENT.


# ═════════════════════════════════════════════════════════════════════════
# 4. TestLayer0FileChangedAfterCache — mtime invalidation
# ═════════════════════════════════════════════════════════════════════════


class TestLayer0FileChangedAfterCache:
    """If file changes after cache populated (e.g., apt upgrade replaces
    file), Layer 0 mtime check fails → fall through to Layer 1+."""

    def test_file_modified_externally_falls_through(self, tmp_path):
        from sndr.kernel.text_patch import TextPatchResult
        target = tmp_path / "f.py"
        target.write_text("alpha\n")
        _make_patcher(target).apply()  # populates cache

        # External modification: revert file to pristine state (no marker)
        time.sleep(0.01)
        target.write_text("alpha\n")  # mtime now different from cached

        # Apply — Layer 0 should miss (mtime changed), Layer 1+ takes over
        # File is pristine again so it gets RE-applied.
        result, _ = _make_patcher(target).apply()
        assert result == TextPatchResult.APPLIED
        assert "ALPHA" in target.read_text()


# ═════════════════════════════════════════════════════════════════════════
# 5. TestLayer0EquivalenceWithLegacy
# ═════════════════════════════════════════════════════════════════════════


class TestLayer0EquivalenceWithLegacy:
    """The CRITICAL test: with Layer 0 enabled produces same final
    file content as with Layer 0 disabled."""

    def test_idempotent_paths_equivalent(self, tmp_path, monkeypatch):
        # Path A: cache enabled (Layer 0 + Layer 2 path)
        target_a = tmp_path / "a.py"
        target_a.write_text("alpha\n")
        _make_patcher(target_a).apply()
        # Re-apply via Layer 0
        _make_patcher(target_a).apply()
        body_a = target_a.read_text()

        # Reset cache for path B
        from sndr.engines.vllm.wiring import file_cache
        file_cache._reset_for_tests()
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")

        # Path B: cache disabled (Layer 1+2 only)
        target_b = tmp_path / "b.py"
        target_b.write_text("alpha\n")
        _make_patcher(target_b).apply()
        _make_patcher(target_b).apply()
        body_b = target_b.read_text()

        assert body_a == body_b, "Layer 0 path output ≠ legacy output"


# ═════════════════════════════════════════════════════════════════════════
# 6. TestLayer0DoesNotBreakLayer2Idempotency
# ═════════════════════════════════════════════════════════════════════════


class TestLayer0DoesNotBreakLayer2:
    """If Layer 0 misses (e.g., cache absent or env=1), Layer 2's
    marker-in-content check should still detect already-patched files.

    This guarantees backwards-compat: even with broken cache, no double-apply.
    """

    def test_layer2_idempotent_when_cache_absent(
            self, tmp_path, monkeypatch):
        from sndr.kernel.text_patch import TextPatchResult
        from sndr.engines.vllm.wiring import file_cache

        target = tmp_path / "f.py"
        target.write_text("alpha\n")

        # First apply — populates cache
        _make_patcher(target).apply()

        # Wipe cache + env-disable. Layer 0 will miss completely.
        file_cache._reset_for_tests()
        monkeypatch.setenv("GENESIS_NO_PATCH_CACHE", "1")

        # Second apply — Layer 0 skipped, Layer 2 should still detect marker
        result, _ = _make_patcher(target).apply()
        assert result == TextPatchResult.IDEMPOTENT
        # File body unchanged
        assert "ALPHA" in target.read_text()

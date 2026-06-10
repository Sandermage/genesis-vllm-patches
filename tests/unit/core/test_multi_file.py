# SPDX-License-Identifier: Apache-2.0
"""Tests for `sndr.kernel.multi_file` — MultiFilePatchTransaction.

Contract:

  1. Constructor accepts list of patchers + optional name; stores them.
  2. _dry_run: returns (False, reason) when:
       - patcher is None
       - target_file doesn't exist
       - target_file unreadable
       - required anchor missing
       - required anchor ambiguous (>1 occurrence)
  3. _dry_run: returns (True, "") when all patchers validate.
  4. _dry_run: idempotency (marker already in src) skips per-file checks.
  5. _dry_run: sequential-preview catches anchor invalidation across
     sub-patches in declared order.
  6. apply_or_skip returns ("skipped", ...) on dry-run failure WITHOUT
     touching any file.
  7. apply_or_skip pre-commit snapshot takes ALL files into memory before
     any write.
  8. _write_rollback_aid writes <path>.genesis_rollback aid; returns the
     aid path; returns None on failure.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sndr.kernel.multi_file import MultiFilePatchTransaction


def _mock_patcher(
    target_file: str,
    marker: str = "MARKER",
    sub_patches: list | None = None,
    apply_returns=("APPLIED", None),
):
    """Build a TextPatcher-shaped mock for transaction tests."""
    return SimpleNamespace(
        target_file=target_file,
        marker=marker,
        sub_patches=sub_patches or [],
        apply=lambda: apply_returns,
    )


def _mock_sub_patch(name: str, anchor: str, replacement: str,
                    required: bool = True):
    return SimpleNamespace(
        name=name, anchor=anchor, replacement=replacement, required=required,
    )


# ─── Constructor ──────────────────────────────────────────────────────


class TestConstructor:
    def test_stores_patchers_list(self):
        p1 = _mock_patcher("/x.py")
        p2 = _mock_patcher("/y.py")
        txn = MultiFilePatchTransaction([p1, p2], name="my-txn")
        assert txn.patchers == [p1, p2]
        assert txn.name == "my-txn"

    def test_default_name(self):
        txn = MultiFilePatchTransaction([])
        assert txn.name == "multi-file"


# ─── _dry_run validation paths ────────────────────────────────────────


class TestDryRunFailures:
    def test_none_patcher_fails(self, tmp_path):
        txn = MultiFilePatchTransaction([None])
        ok, reason = txn._dry_run()
        assert not ok
        assert "patcher is None" in reason

    def test_missing_file_fails(self, tmp_path):
        p = _mock_patcher(str(tmp_path / "nonexistent.py"))
        txn = MultiFilePatchTransaction([p])
        ok, reason = txn._dry_run()
        assert not ok
        assert "file missing" in reason

    def test_required_anchor_missing_fails(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("a = 1\nb = 2\n")
        sp = _mock_sub_patch("sub1", anchor="NOT_THERE", replacement="X",
                              required=True)
        p = _mock_patcher(str(f), sub_patches=[sp])
        txn = MultiFilePatchTransaction([p])
        ok, reason = txn._dry_run()
        assert not ok
        assert "required anchor" in reason
        assert "not found" in reason

    def test_required_anchor_ambiguous_fails(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("foo()\nfoo()\nfoo()\n")
        sp = _mock_sub_patch("sub1", anchor="foo()", replacement="bar()",
                              required=True)
        p = _mock_patcher(str(f), sub_patches=[sp])
        txn = MultiFilePatchTransaction([p])
        ok, reason = txn._dry_run()
        assert not ok
        assert "ambiguous" in reason

    def test_marker_already_present_idempotent(self, tmp_path):
        """If marker is in src, per-file checks skipped (already-applied)."""
        f = tmp_path / "src.py"
        f.write_text("# MARKER applied\nnope\n")
        # Even with no anchor present, marker presence short-circuits validation
        sp = _mock_sub_patch("sub1", anchor="NOT_THERE", replacement="X",
                              required=True)
        p = _mock_patcher(str(f), marker="MARKER", sub_patches=[sp])
        txn = MultiFilePatchTransaction([p])
        ok, reason = txn._dry_run()
        assert ok  # marker idempotency wins

    def test_optional_anchor_missing_ok(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("a = 1\n")
        sp = _mock_sub_patch("sub1", anchor="NOT_THERE", replacement="X",
                              required=False)
        p = _mock_patcher(str(f), sub_patches=[sp])
        txn = MultiFilePatchTransaction([p])
        ok, reason = txn._dry_run()
        assert ok


class TestDryRunSuccess:
    def test_single_patch_clean(self, tmp_path):
        f = tmp_path / "src.py"
        f.write_text("hello = 1\n")
        sp = _mock_sub_patch("sub1", anchor="hello = 1",
                              replacement="hello = 2", required=True)
        p = _mock_patcher(str(f), sub_patches=[sp])
        txn = MultiFilePatchTransaction([p])
        ok, reason = txn._dry_run()
        assert ok
        assert reason == ""

    def test_sequential_preview_resolves_replacement(self, tmp_path):
        """Second sub-patch anchors against post-replacement state from
        the first sub-patch."""
        f = tmp_path / "src.py"
        f.write_text("ALPHA\nBETA\n")
        sp1 = _mock_sub_patch("first", anchor="ALPHA", replacement="GAMMA",
                                required=True)
        sp2 = _mock_sub_patch("second", anchor="GAMMA",  # appears after sp1
                                replacement="DELTA", required=True)
        p = _mock_patcher(str(f), sub_patches=[sp1, sp2])
        txn = MultiFilePatchTransaction([p])
        ok, reason = txn._dry_run()
        assert ok, reason


# ─── apply_or_skip — skip path ────────────────────────────────────────


class TestApplyOrSkip:
    def test_dry_run_failure_skips_without_writing(self, tmp_path):
        """A failed dry-run must NOT modify any file."""
        f = tmp_path / "src.py"
        original = "a = 1\n"
        f.write_text(original)
        sp = _mock_sub_patch("sub1", anchor="NOT_THERE", replacement="X",
                              required=True)
        p = _mock_patcher(str(f), sub_patches=[sp])
        txn = MultiFilePatchTransaction([p], name="test-txn")
        result, reason = txn.apply_or_skip()
        assert result == "skipped"
        assert "test-txn" in reason
        assert "dry-run failed" in reason
        # File untouched
        assert f.read_text() == original


# ─── _write_rollback_aid ──────────────────────────────────────────────


class TestRollbackAid:
    def test_writes_aid_returns_path(self, tmp_path):
        target = tmp_path / "victim.py"
        target.write_text("original\n")
        aid_path = MultiFilePatchTransaction._write_rollback_aid(
            str(target), "snapshot content\n"
        )
        assert aid_path == str(target) + ".genesis_rollback"
        aid_file = tmp_path / "victim.py.genesis_rollback"
        assert aid_file.exists()
        assert aid_file.read_text() == "snapshot content\n"

    def test_empty_path_returns_none(self):
        assert MultiFilePatchTransaction._write_rollback_aid("", "content") is None

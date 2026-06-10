# SPDX-License-Identifier: Apache-2.0
"""Stage 8 — per-sub drift markers (TextPatch.upstream_merged_markers).

Tests the per-sub-patch drift detection added in Stage 8 (2026-05-07).
Differs from patcher-level `TextPatcher.upstream_drift_markers` (Layer 3,
all-or-nothing): per-sub drift lets ONE sub no-op while siblings continue.

Used when upstream cherry-picks part of a multi-anchor backport: e.g.
P64 has 4 sub-anchors, upstream merges only 2 → without per-sub drift,
P64 either re-applies all 4 (silent no-op since marker present) or
skips entirely; with per-sub drift, the 2 not-yet-merged anchors keep
applying.
"""
from __future__ import annotations
from pathlib import Path

import os
import tempfile

import pytest


@pytest.fixture
def fresh_target():
    """Create a temp file with predictable content for anchor tests."""
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_textpatch_defaults_back_compat(fresh_target):
    """Existing patches construct TextPatch without drift fields — defaults
    must be empty list + skip_silently for back-compat."""
    from sndr.kernel import TextPatch
    p = TextPatch(name="legacy", anchor="X", replacement="Y")
    assert p.upstream_merged_markers == []
    assert p.on_upstream_merge == "skip_silently"


def test_per_sub_drift_skip_silently(fresh_target):
    """Sub with matching upstream marker no-ops; siblings apply normally."""
    from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

    Path(fresh_target).write_text(
        "AAA\nBBB-already-fixed-by-upstream\nCCC\n"
    )
    patcher = TextPatcher(
        patch_name="test_drift_silent",
        target_file=fresh_target,
        marker="Genesis test silent",
        sub_patches=[
            TextPatch(name="s1", anchor="AAA", replacement="A1"),
            TextPatch(
                name="s2", anchor="BBB", replacement="B1",
                upstream_merged_markers=["BBB-already-fixed-by-upstream"],
            ),
            TextPatch(name="s3", anchor="CCC", replacement="C1"),
        ],
    )
    result, _failure = patcher.apply()
    assert result == TextPatchResult.APPLIED
    content = Path(fresh_target).read_text()
    assert "A1" in content
    assert "C1" in content
    assert "B1" not in content                          # sub-2 didn't apply
    assert "BBB-already-fixed-by-upstream" in content   # untouched


def test_per_sub_drift_warn_logs_warning(fresh_target, caplog):
    """on_upstream_merge='warn' logs at WARNING level but still continues."""
    import logging

    from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

    Path(fresh_target).write_text("X\nY-merged-warn-me\nZ\n")
    patcher = TextPatcher(
        patch_name="test_drift_warn",
        target_file=fresh_target,
        marker="Genesis test warn",
        sub_patches=[
            TextPatch(name="s1", anchor="X", replacement="x1"),
            TextPatch(
                name="s2", anchor="Y", replacement="y1",
                upstream_merged_markers=["Y-merged-warn-me"],
                on_upstream_merge="warn",
            ),
            TextPatch(name="s3", anchor="Z", replacement="z1"),
        ],
    )
    with caplog.at_level(logging.WARNING, logger="genesis.wiring.text_patch"):
        result, _ = patcher.apply()
    assert result == TextPatchResult.APPLIED
    # WARNING-level log emitted for the upstream-merged sub
    has_warn = any(
        "upstream-merged" in r.message and "Y-merged" in r.message
        for r in caplog.records
        if r.levelno == logging.WARNING
    )
    assert has_warn, (
        f"Expected WARNING about upstream-merged sub-2, "
        f"got: {[r.message for r in caplog.records]}"
    )


def test_per_sub_drift_abort_bundle(fresh_target):
    """on_upstream_merge='abort_bundle' aborts the entire patcher with
    SKIPPED — file unchanged even for siblings whose anchors WOULD apply."""
    from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

    original = "PRE\nMERGED-INCOMPATIBLE\nPOST\n"
    Path(fresh_target).write_text(original)
    patcher = TextPatcher(
        patch_name="test_abort",
        target_file=fresh_target,
        marker="Genesis abort test",
        sub_patches=[
            TextPatch(name="s1", anchor="PRE", replacement="pre1", required=True),
            TextPatch(
                name="s2", anchor="X-not-found", replacement="x2", required=True,
                upstream_merged_markers=["MERGED-INCOMPATIBLE"],
                on_upstream_merge="abort_bundle",
            ),
            TextPatch(name="s3", anchor="POST", replacement="post1", required=True),
        ],
    )
    result, failure = patcher.apply()
    assert result == TextPatchResult.SKIPPED
    assert failure is not None
    assert "abort_bundle" in failure.detail
    # File MUST be byte-identical to original (no partial commit)
    assert Path(fresh_target).read_text() == original


def test_per_sub_drift_works_with_required_sub(fresh_target):
    """A required=True sub that hits its drift marker is OK to skip
    silently — per-sub drift takes precedence over required-anchor check."""
    from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

    Path(fresh_target).write_text("KEEP\nMERGED-MARKER\nKEEP2\n")
    patcher = TextPatcher(
        patch_name="test_required_drift",
        target_file=fresh_target,
        marker="Genesis required drift test",
        sub_patches=[
            TextPatch(
                # required=True but its anchor is gone (upstream merged)
                name="s_required_but_merged",
                anchor="MISSING-ANCHOR",
                replacement="never",
                required=True,
                upstream_merged_markers=["MERGED-MARKER"],
            ),
            TextPatch(name="s_other", anchor="KEEP2", replacement="K2-fixed"),
        ],
    )
    result, _ = patcher.apply()
    assert result == TextPatchResult.APPLIED
    content = Path(fresh_target).read_text()
    assert "K2-fixed" in content
    assert "MERGED-MARKER" in content  # untouched


def test_per_sub_drift_no_match_proceeds_normally(fresh_target):
    """When the upstream marker is NOT present, sub applies normally."""
    from sndr.kernel import TextPatch, TextPatcher, TextPatchResult

    Path(fresh_target).write_text("ALPHA\nBETA\n")
    patcher = TextPatcher(
        patch_name="test_no_drift",
        target_file=fresh_target,
        marker="Genesis no-drift test",
        sub_patches=[
            TextPatch(
                name="s1", anchor="BETA", replacement="BETA-PATCHED",
                upstream_merged_markers=["DIFFERENT-MARKER-NOT-PRESENT"],
            ),
        ],
    )
    result, _ = patcher.apply()
    assert result == TextPatchResult.APPLIED
    content = Path(fresh_target).read_text()
    assert "BETA-PATCHED" in content

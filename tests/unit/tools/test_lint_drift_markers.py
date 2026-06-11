# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tools/lint_drift_markers.py`` — the self-collision lint.

TDD contract (written BEFORE the implementation), per the §6 action plan
(docs/superpowers/journal/2026-06-11-preflight-residual-triage-action-plan.md):
reject any ``upstream_drift_marker`` that is a substring of

  (a) the patcher's OWN replacement texts, or
  (b) its idempotency marker LINE — the Layer-6 prepend from
      ``sndr/kernel/text_patch.py``:
      ``marker_line = f"# [Genesis wiring marker: {self.marker}]\\n"``

EXCEPT markers starting with ``[Genesis`` (defended convention — custom
apply() wrappers skip them; remediation extends the skip to Layer 3).

Covered here: TRUE_RISK detection via replacement and via marker line,
the ``[Genesis`` exemption, allowlist parsing (justification-comment
required), violation filtering / stale-entry reporting, and exit-code
semantics (1 = violations, 0 = clean/allowlisted, 2 = nothing built).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "lint_drift_markers.py"


def _import_tool():
    spec = importlib.util.spec_from_file_location("lint_drift_markers",
                                                  TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lint_drift_markers"] = mod
    assert spec.loader is not None  # nosec
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ldm():
    return _import_tool()


@pytest.fixture()
def kernel():
    from sndr.kernel.text_patch import TextPatch, TextPatcher
    return TextPatch, TextPatcher


def _patcher(kernel, *, marker="GENESIS_TEST_MARKER_V1",
             replacement="    x = compute_safe(x)\n",
             drift_markers=(), extra_subs=()):
    """Fake patcher factory built on the REAL kernel dataclasses so the
    lint exercises the exact attribute surface TextPatcher exposes."""
    TextPatch, TextPatcher = kernel
    subs = [TextPatch(name="sub1",
                      anchor="    x = compute(x)\n",
                      replacement=replacement,
                      required=True)]
    subs.extend(extra_subs)
    return TextPatcher(
        patch_name="TEST fake patcher",
        target_file="/nonexistent/target.py",
        marker=marker,
        sub_patches=subs,
        upstream_drift_markers=list(drift_markers),
    )


# ─── core collision detection ─────────────────────────────────────────────


class TestCollisionCore:
    def test_replacement_collision_is_true_risk(self, ldm, kernel):
        """PN369 class: the drift marker is baked into the patch's own
        replacement → Layer 3 false-skips on every post-apply re-run."""
        p = _patcher(kernel,
                     replacement="    x = compute_safe(x)  # tq_guard_v2\n",
                     drift_markers=["tq_guard_v2"])
        findings = ldm.collisions_for_patcher(p)
        assert len(findings) == 1
        f = findings[0]
        assert f["marker"] == "tq_guard_v2"
        assert "replacement" in f["collides_with"]
        assert f["colliding_subs"] == ["sub1"]

    def test_marker_line_collision_is_true_risk(self, ldm, kernel):
        """PN54 class: the drift marker is a substring of the raw
        idempotency marker → present in the Layer-6 marker line."""
        p = _patcher(kernel, marker="GENESIS_PN54_TQ_SPLITS_V2",
                     drift_markers=["PN54_TQ_SPLITS"])
        findings = ldm.collisions_for_patcher(p)
        assert len(findings) == 1
        assert findings[0]["collides_with"] == ["idempotency_marker_line"]

    def test_marker_line_fixed_prefix_collision(self, ldm, kernel):
        """The Layer-6 line carries the constant prefix
        ``# [Genesis wiring marker: `` — a drift marker matching THAT
        text collides for every patch, so the lint must check the full
        LINE, not just the raw marker string."""
        p = _patcher(kernel, drift_markers=["Genesis wiring marker"])
        findings = ldm.collisions_for_patcher(p)
        assert len(findings) == 1
        assert findings[0]["collides_with"] == ["idempotency_marker_line"]

    def test_genesis_prefixed_marker_exempt(self, ldm, kernel):
        """Defended convention: ``[Genesis``-prefixed drift markers are
        the sanctioned self-referencing form (PN353A fix) — never flagged
        even when literally present in the replacement."""
        p = _patcher(kernel,
                     replacement="    pass  # [Genesis PN353A v2]\n",
                     drift_markers=["[Genesis PN353A"])
        assert ldm.collisions_for_patcher(p) == []

    def test_clean_patcher_no_findings(self, ldm, kernel):
        p = _patcher(kernel, drift_markers=["def upstream_native_fix("])
        assert ldm.collisions_for_patcher(p) == []

    def test_no_drift_markers_no_findings(self, ldm, kernel):
        assert ldm.collisions_for_patcher(_patcher(kernel)) == []

    def test_collision_in_second_sub_replacement(self, ldm, kernel):
        """ALL sub-patch replacements are scanned, not just the first."""
        TextPatch, _ = kernel
        extra = TextPatch(name="sub2", anchor="y = 1\n",
                          replacement="y = 2  # _genesis_extra_path\n")
        p = _patcher(kernel, drift_markers=["_genesis_extra_path"],
                     extra_subs=(extra,))
        findings = ldm.collisions_for_patcher(p)
        assert len(findings) == 1
        assert findings[0]["colliding_subs"] == ["sub2"]

    def test_both_sites_reported_together(self, ldm, kernel):
        p = _patcher(kernel, marker="GENESIS_DOUBLE_HIT",
                     replacement="    pass  # GENESIS_DOUBLE_HIT\n",
                     drift_markers=["GENESIS_DOUBLE_HIT"])
        findings = ldm.collisions_for_patcher(p)
        assert len(findings) == 1
        assert set(findings[0]["collides_with"]) == {
            "idempotency_marker_line", "replacement"}


# ─── allowlist parsing ────────────────────────────────────────────────────


class TestAllowlist:
    def test_parse_valid_entries(self, ldm, tmp_path):
        path = tmp_path / "allow.txt"
        path.write_text(
            "# Baseline 2026-06-11: pre-remediation (plan §6 item 3).\n"
            "tq_guard_v2\n"
            "\n"
            "# PN54: marker shared with anchor — retire queued (§3).\n"
            "PN54_TQ_SPLITS\n"
        )
        markers, errors = ldm.parse_allowlist(path)
        assert errors == []
        assert markers == ["tq_guard_v2", "PN54_TQ_SPLITS"]

    def test_missing_justification_is_error(self, ldm, tmp_path):
        path = tmp_path / "allow.txt"
        path.write_text("tq_guard_v2\n")
        markers, errors = ldm.parse_allowlist(path)
        assert markers == []
        assert len(errors) == 1
        assert "justification" in errors[0]

    def test_blank_line_resets_justification(self, ldm, tmp_path):
        """A blank line ends a justification block — the next marker
        needs a fresh comment (one block may cover several markers,
        but never across a blank-line separator)."""
        path = tmp_path / "allow.txt"
        path.write_text(
            "# Covers both PN369-batch markers below.\n"
            "marker_one\n"
            "marker_two\n"
            "\n"
            "orphan_marker\n"
        )
        markers, errors = ldm.parse_allowlist(path)
        assert markers == ["marker_one", "marker_two"]
        assert len(errors) == 1
        assert "orphan_marker" in errors[0]

    def test_comment_only_file_is_empty(self, ldm, tmp_path):
        path = tmp_path / "allow.txt"
        path.write_text("# format docs only, no entries yet\n")
        assert ldm.parse_allowlist(path) == ([], [])

    def test_missing_file_is_empty(self, ldm, tmp_path):
        assert ldm.parse_allowlist(tmp_path / "absent.txt") == ([], [])


# ─── report assembly + exit semantics ─────────────────────────────────────


def _entry(kernel, **kw):
    """One enumeration tuple as iter_buildable_patchers() yields them."""
    return ("tests.fake.module", "_make_patcher", ["PTEST"],
            _patcher(kernel, **kw))


class TestReportAndExit:
    def test_violation_reported_and_exit_1(self, ldm, kernel):
        entries = [_entry(kernel,
                          replacement="    pass  # tq_guard_v2\n",
                          drift_markers=["tq_guard_v2"])]
        report = ldm.run_lint(entries, allowlist=[])
        assert report["summary"]["violations"] == 1
        assert report["violations"][0]["patch_ids"] == ["PTEST"]
        assert ldm.decide_exit(report) == 1

    def test_allowlisted_violation_suppressed_exit_0(self, ldm, kernel):
        entries = [_entry(kernel,
                          replacement="    pass  # tq_guard_v2\n",
                          drift_markers=["tq_guard_v2"])]
        report = ldm.run_lint(entries, allowlist=["tq_guard_v2"])
        assert report["summary"]["violations"] == 0
        assert report["summary"]["allowlisted"] == 1
        assert ldm.decide_exit(report) == 0

    def test_stale_allowlist_entry_reported_not_fatal(self, ldm, kernel):
        entries = [_entry(kernel)]
        report = ldm.run_lint(entries, allowlist=["never_collides_now"])
        assert report["stale_allowlist"] == ["never_collides_now"]
        assert ldm.decide_exit(report) == 0

    def test_clean_run_exit_0(self, ldm, kernel):
        report = ldm.run_lint([_entry(kernel)], allowlist=[])
        assert report["summary"]["violations"] == 0
        assert ldm.decide_exit(report) == 0

    def test_zero_patchers_is_invocation_error(self, ldm):
        """An empty enumeration means the candidate root is wrong (or
        the registry import broke) — a lint that silently passes while
        checking nothing is the worst failure mode. Exit 2, not 0."""
        report = ldm.run_lint([], allowlist=[])
        assert ldm.decide_exit(report) == 2

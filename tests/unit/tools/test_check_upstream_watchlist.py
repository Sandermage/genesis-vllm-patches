# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tools/check_upstream_watchlist.py`` — the PR-sweep
``sweep:`` section validator of ``tools/upstream_watchlist.yaml``.

TDD contract (written BEFORE the implementation), per the 2026-06-11
PR-sweep roadmap (docs/superpowers/journal/
2026-06-11-pr-sweep-50-roadmap.md): the watchlist gained a second
top-level section ``sweep:`` (one row per studied upstream PR), with
schema distinct from the legacy ``watch:`` section that
``scripts/audit_upstream_watchlist.py`` owns:

  - ``pr``            int — upstream vllm PR number
  - ``genesis_patch`` str — existing patch id(s), ``planned: ...``,
                      or ``watch-only``
  - ``trigger``       one of retire-on-merge | reanchor-on-merge |
                      review-on-merge
  - ``note``          non-empty free text

Validator guarantees: YAML loads, required keys present, ``pr``
unique within the section, ``trigger`` in the allowed enum. Exit
codes: 0 = clean, 2 = schema error / missing file.

The live-file tests also pin the roadmap bookkeeping invariants:
all 50 swept PRs present, the G4_T1 racing cluster complete, and
the four duplicate relationships annotated.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "check_upstream_watchlist.py"
WATCHLIST = REPO_ROOT / "tools" / "upstream_watchlist.yaml"

# The 50 PRs deep-studied in the 2026-06-11 sweep (5 chunks x 10).
SWEPT_PRS = frozenset({
    # chunk 1
    45207, 45181, 45199, 45202, 45182, 45197, 45196, 45184, 45176, 45173,
    # chunk 2
    45100, 45146, 45144, 45068, 45126, 45151, 45120, 45053, 45109, 45130,
    # chunk 3
    45005, 45060, 44993, 45040, 45038, 44955, 45080, 45022, 45096, 45001,
    # chunk 4
    44877, 44844, 44943, 44837, 44880, 44868, 44850, 44912, 44784, 44932,
    # chunk 5
    44717, 44752, 44741, 45076, 44644, 44742, 44778, 44628, 44563, 44754,
})

# G4_T1 racing group — one annotated cluster (chunk-2 Theme B; #44844
# joined the enumeration when its v3 prep overlay vendored, 2026-06-11).
RACING_CLUSTER = frozenset({42006, 42237, 42300, 44741, 45068, 44844})


def _import_tool():
    spec = importlib.util.spec_from_file_location(
        "check_upstream_watchlist", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_upstream_watchlist"] = mod
    assert spec.loader is not None  # nosec
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _import_tool()


def _row(**over):
    base = {
        "pr": 44644,
        "genesis_patch": "PN348",
        "trigger": "retire-on-merge",
        "note": "vendored; retire on merge after deep-diff",
    }
    base.update(over)
    return base


class TestValidateRows:
    def test_clean_rows_no_errors(self, tool):
        assert tool.validate_rows([_row()]) == []

    def test_missing_required_key(self, tool):
        for key in ("pr", "genesis_patch", "trigger", "note"):
            row = _row()
            del row[key]
            errors = tool.validate_rows([row])
            assert any(key in e for e in errors), key

    def test_pr_must_be_int(self, tool):
        errors = tool.validate_rows([_row(pr="vllm#44644")])
        assert any("pr" in e for e in errors)

    def test_duplicate_pr_rejected(self, tool):
        errors = tool.validate_rows([_row(), _row()])
        assert any("duplicate" in e.lower() for e in errors)

    def test_invalid_trigger_rejected(self, tool):
        errors = tool.validate_rows([_row(trigger="yeet-on-merge")])
        assert any("trigger" in e for e in errors)

    def test_all_three_triggers_accepted(self, tool):
        rows = [
            _row(pr=1, trigger="retire-on-merge"),
            _row(pr=2, trigger="reanchor-on-merge"),
            _row(pr=3, trigger="review-on-merge"),
        ]
        assert tool.validate_rows(rows) == []

    def test_empty_note_rejected(self, tool):
        errors = tool.validate_rows([_row(note="")])
        assert any("note" in e for e in errors)

    def test_non_mapping_row_rejected(self, tool):
        errors = tool.validate_rows(["not-a-mapping"])
        assert errors


class TestLiveFile:
    """The committed watchlist satisfies the roadmap invariants."""

    def test_main_exit_zero_on_live_file(self, tool, capsys):
        rc = tool.main([])
        out = capsys.readouterr()
        assert rc == 0, out.err

    def test_live_sweep_covers_all_50_prs(self, tool):
        rows = tool.load_sweep(WATCHLIST)
        prs = {r["pr"] for r in rows}
        missing = SWEPT_PRS - prs
        assert not missing, f"swept PRs missing from sweep section: {missing}"

    def test_racing_cluster_present_and_annotated(self, tool):
        rows = tool.load_sweep(WATCHLIST)
        by_pr = {r["pr"]: r for r in rows}
        for pr in RACING_CLUSTER:
            assert pr in by_pr, f"racing-cluster PR {pr} missing"
            assert "racing" in by_pr[pr]["note"].lower(), pr

    def test_known_relationships(self, tool):
        rows = tool.load_sweep(WATCHLIST)
        by_pr = {r["pr"]: r for r in rows}
        # PN351 anchor-breaker -> reanchor-on-merge
        assert by_pr[45151]["trigger"] == "reanchor-on-merge"
        assert "PN351" in by_pr[45151]["genesis_patch"]
        # P67b drift-watch
        assert "P67b" in by_pr[45144]["genesis_patch"]
        # duplicates
        assert "44717" in by_pr[44752]["note"]
        assert "PN55" in by_pr[44778]["genesis_patch"]
        assert "PN348" in by_pr[44644]["genesis_patch"]
        assert "PN367" in by_pr[45076]["genesis_patch"]

    def test_legacy_watch_section_untouched_schema(self):
        """Adding ``sweep:`` must not break the legacy validator."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        try:
            import audit_upstream_watchlist as legacy
        finally:
            sys.path.pop(0)
        data = legacy._load_yaml()
        assert legacy._validate(data) == []


class TestExitCodes:
    def test_missing_file_exit_2(self, tool, tmp_path, capsys):
        rc = tool.main(["--watchlist", str(tmp_path / "nope.yaml")])
        capsys.readouterr()
        assert rc == 2

    def test_schema_error_exit_2(self, tool, tmp_path, capsys):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "sweep:\n  - pr: not-an-int\n    trigger: retire-on-merge\n",
            encoding="utf-8",
        )
        rc = tool.main(["--watchlist", str(bad)])
        capsys.readouterr()
        assert rc == 2

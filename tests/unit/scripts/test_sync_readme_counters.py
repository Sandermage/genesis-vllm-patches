# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/sync_readme_counters.py` — README counter drift gate.

Contract: every well-known counter line/badge in README.md must match
the authoritative count from PATCH_REGISTRY + V2 builtin registry. The
gate is idempotent: running the rewrite twice produces no further diff.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "sync_readme_counters.py"


def _import_script():
    name = "_sync_readme_counters_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Counts come from the live registry ───────────────────────────────


class TestCounts:
    def test_collect_counts_against_registry(self):
        mod = _import_script()
        counts = mod.collect_counts()
        # PATCH_REGISTRY sanity — at least 130 entries (recent baseline 136).
        assert counts.patches >= 130
        # Families is small + finite.
        assert 5 <= counts.families <= 50
        # V2 layered counts non-negative.
        assert counts.v2_models >= 0
        assert counts.v2_hardware >= 0
        assert counts.v2_profiles >= 0
        assert counts.v2_aliases >= 0


# ─── Rule pattern matching ────────────────────────────────────────────


class TestRulePatterns:
    def test_badge_pattern_matches(self):
        mod = _import_script()
        rule = next(r for r in mod.RULES if r.rule_id == "R-patch-badge")
        line = "[![Patches](https://img.shields.io/badge/patches-99-green.svg)](docs/PATCHES.md)"
        assert rule.pattern.search(line)
        # Capture group: the number itself.
        assert rule.pattern.search(line).group(1) == "99"

    def test_text_pattern_matches(self):
        mod = _import_script()
        rule = next(r for r in mod.RULES if r.rule_id == "R-text-N-community-patches")
        line = "Apache 2.0, **42 community patches** — the"
        assert rule.pattern.search(line)
        assert rule.pattern.search(line).group(1) == "42"

    def test_coverage_pattern_two_groups(self):
        mod = _import_script()
        rule = next(r for r in mod.RULES if r.rule_id == "R-coverage-line")
        line = "### Patch coverage — 134 patches across 19 categories"
        m = rule.pattern.search(line)
        assert m is not None
        assert m.group(1) == "134"
        assert m.group(2) == "19"

    def test_by_category_heading_pattern(self):
        mod = _import_script()
        rule = next(r for r in mod.RULES if r.rule_id == "R-by-category-heading")
        line = "## 📦 136 patches by category"
        assert rule.pattern.search(line).group(1) == "136"


# ─── End-to-end on a synthetic README ─────────────────────────────────


def _synth_readme(patches: int, families: int) -> str:
    return textwrap.dedent(f"""
        # Test repo

        [![Patches](https://img.shields.io/badge/patches-{patches}-green.svg)](docs/PATCHES.md)

        ## Why

        - **{patches} community patches** — the runtime layer
        - vllm + torch

        ## Layout

        | `sndr` | **{patches} community patches** + dispatcher |

        ### Patch coverage — {patches} patches across {families} categories

        ## 📦 {patches} patches by category

        - cf. [docs/PATCHES.md](docs/PATCHES.md) | All {patches} patches table |
    """).lstrip("\n")


class TestApplyRules:
    def test_already_correct_no_changes(self, tmp_path):
        mod = _import_script()
        counts = mod.collect_counts()
        text = _synth_readme(counts.patches, counts.families)
        new_text, hits = mod.apply_rules(text, counts)
        changed = [h for h in hits if h.changed]
        assert new_text == text
        assert changed == []

    def test_stale_counts_rewritten(self):
        mod = _import_script()
        counts = mod.collect_counts()
        # Synth README claims patches-1 / 2 categories — far off real.
        stale_text = _synth_readme(patches=1, families=2)
        new_text, hits = mod.apply_rules(stale_text, counts)
        changed = [h for h in hits if h.changed]
        # Every rule fires at least once.
        assert len(changed) >= 5
        # Real counts now in the text.
        assert str(counts.patches) in new_text
        assert str(counts.families) in new_text
        # Stale numbers removed (1 + 2 are gone — at least in our rules).
        # Use substring check; "1" might still occur in version strings,
        # but the badge/text shape we control replaces them.
        for r in mod.RULES:
            for line in new_text.splitlines():
                m = r.pattern.search(line)
                if not m:
                    continue
                if r.rule_id == "R-coverage-line":
                    assert m.group(1) == str(counts.patches)
                    assert m.group(2) == str(counts.families)
                else:
                    assert m.group(1) == str(counts.patches)

    def test_idempotent(self):
        """Apply twice — second pass produces no diff."""
        mod = _import_script()
        counts = mod.collect_counts()
        stale = _synth_readme(patches=99, families=99)
        first_text, _ = mod.apply_rules(stale, counts)
        second_text, second_hits = mod.apply_rules(first_text, counts)
        assert second_text == first_text
        assert all(not h.changed for h in second_hits)


# ─── Script CLI ────────────────────────────────────────────────────────


class TestScriptCLI:
    def test_check_passes_on_committed_readme(self):
        """After Entry 14 fix, --check must report clean (idempotent gate)."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--check", "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"committed README drift: {result.stdout[:1500]}"
        )
        payload = json.loads(result.stdout)
        assert payload["drift_count"] == 0
        assert payload["passed"] is True

    def test_check_detects_drift_on_synthetic(self, tmp_path):
        """Drop a stale synthetic file and confirm --check exits 1."""
        synth = _synth_readme(patches=5, families=5)
        fp = tmp_path / "FakeREADME.md"
        fp.write_text(synth, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--check", "--file", str(fp), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 1
        payload = json.loads(result.stdout)
        assert payload["drift_count"] > 0
        assert payload["passed"] is False

    def test_rewrite_mode_updates_file(self, tmp_path):
        synth = _synth_readme(patches=5, families=5)
        fp = tmp_path / "FakeREADME.md"
        fp.write_text(synth, encoding="utf-8")
        # Rewrite — exit 0 means "either no drift or we fixed it".
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--file", str(fp), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["drift_count"] > 0   # drift WAS present
        assert payload["wrote_file"] is True
        # Re-check now passes — file actually changed on disk.
        result2 = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--check", "--file", str(fp), "--json"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result2.returncode == 0
        payload2 = json.loads(result2.stdout)
        assert payload2["passed"] is True

    def test_missing_file_returns_two(self, tmp_path):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH),
             "--file", str(tmp_path / "nope.md"), "--check"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 2

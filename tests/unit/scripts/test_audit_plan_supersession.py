# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_plan_supersession.py`` —
AUDIT-CLOSURE.1.A.5 (2026-05-26).

Synthetic-corpus tests for every rule + edge case.

The audit's job is conservative supersession-target verification — it
only fires on **filename-resolvable** references (``X.md``), never on
narrative "superseded by Phase X work" prose. This matches the
audit's design goal of future-proofing the convention without noising
on the existing narrative-supersession corpus.

The live-corpus smoke confirms the operator's private planning tree
stays at 0 findings until the convention is adopted.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_plan_supersession.py"


def _import_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_plan_supersession", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_plan_supersession"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def audit_mod():
    return _import_audit_module()


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


class TestRuleSup1FilenameTargets:
    """R-SUP-1 — filename-target references must resolve."""

    def test_resolvable_target_silent(self, audit_mod, tmp_path):
        # Successor exists in same dir as source.
        _write(tmp_path, "NEW_R_2026-05-26_RU.md", "# Successor\n")
        _write(
            tmp_path,
            "OLD_2026-05-20_RU.md",
            "# Old doc\n\nSuperseded by NEW_R_2026-05-26_RU.md\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert findings == [], (
            f"resolvable supersession target should be silent, "
            f"got {findings}"
        )

    def test_missing_target_flagged(self, audit_mod, tmp_path):
        _write(
            tmp_path,
            "OLD_2026-05-20_RU.md",
            "# Old doc\n\nSuperseded by NONEXISTENT_2026-05-26_RU.md\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert len(findings) == 1
        assert findings[0].rule == "R-SUP-1"
        assert "NONEXISTENT_2026-05-26_RU.md" in findings[0].detail

    def test_supersedes_line_resolves(self, audit_mod, tmp_path):
        _write(tmp_path, "PARENT_2026-05-20_RU.md", "# Parent\n")
        _write(
            tmp_path,
            "CHILD_2026-05-25_RU.md",
            "# Child\n\nSupersedes: PARENT_2026-05-20_RU.md\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert findings == []

    def test_supersedes_missing_target_flagged(self, audit_mod, tmp_path):
        _write(
            tmp_path,
            "CHILD_2026-05-25_RU.md",
            "# Child\n\nSupersedes: MISSING_2026-05-20_RU.md\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert len(findings) == 1
        assert findings[0].rule == "R-SUP-1"

    def test_yaml_superseded_by_resolves(self, audit_mod, tmp_path):
        _write(tmp_path, "NEW_R_2026-05-26_RU.md", "# Successor\n")
        _write(
            tmp_path,
            "OLD_2026-05-20_RU.md",
            "---\n"
            "title: Old plan\n"
            "superseded_by: NEW_R_2026-05-26_RU.md\n"
            "---\n"
            "# Body\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert findings == []

    def test_subdir_target_resolves(self, audit_mod, tmp_path):
        """Target in a subdirectory (relative path with ``/``)."""
        sub = tmp_path / "_master_pack_2026-05-24"
        sub.mkdir()
        _write(sub, "INDEX.md", "# Index\n")
        _write(
            tmp_path,
            "OLD_2026-05-20_RU.md",
            "Superseded by `_master_pack_2026-05-24/INDEX.md`\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert findings == []

    def test_basename_fallback_resolves(self, audit_mod, tmp_path):
        """Target referenced by basename only resolves via rglob fallback."""
        sub = tmp_path / "deep" / "nested"
        sub.mkdir(parents=True)
        _write(sub, "DEEP_TARGET.md", "# Deep\n")
        _write(
            tmp_path,
            "OLD_2026-05-20_RU.md",
            "# Old\n\nSuperseded by DEEP_TARGET.md\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert findings == []

    def test_narrative_supersession_silent(self, audit_mod, tmp_path):
        """Narrative "superseded by PIN.R" without .md target must NOT
        fire — the audit deliberately ignores non-filename references."""
        _write(
            tmp_path,
            "OLD_2026-05-20_RU.md",
            "# Old\n\n"
            "- Block X (superseded by PIN.R + rig divergence audit)\n"
            "- Block Y (superseded by Phase 5-7 work)\n"
            "- Block Z superseded by upstream cudagraph refactor\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert findings == [], (
            f"narrative supersession refs must be silent, got {findings}"
        )


class TestRuleSup2StatusSupersededTarget:
    """R-SUP-2 — `status: superseded` requires successor reference."""

    def test_status_superseded_with_target_silent(self, audit_mod, tmp_path):
        _write(tmp_path, "NEW_2026-05-26_RU.md", "# New\n")
        _write(
            tmp_path,
            "OLD_2026-05-20_RU.md",
            "---\nstatus: superseded\nsuperseded_by: NEW_2026-05-26_RU.md\n---\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert findings == []

    def test_status_superseded_without_target_flagged(
        self, audit_mod, tmp_path,
    ):
        _write(
            tmp_path,
            "ORPHAN_2026-05-20_RU.md",
            "---\nstatus: superseded\n---\n\n# Body without successor reference\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert len(findings) == 1
        assert findings[0].rule == "R-SUP-2"
        assert "no `superseded_by:`" in findings[0].detail

    def test_status_superseded_with_body_superseded_by_satisfies(
        self, audit_mod, tmp_path,
    ):
        """Body-level `Superseded by NEW.md` counts toward R-SUP-2."""
        _write(tmp_path, "NEW.md", "# New\n")
        _write(
            tmp_path,
            "OLD.md",
            "---\nstatus: superseded\n---\n\nSuperseded by NEW.md\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert findings == [], (
            f"body Superseded-by line satisfies R-SUP-2, got {findings}"
        )


class TestInlineAllowMarker:
    """Same-line marker waives the finding."""

    def test_marker_waives_missing_target(self, audit_mod, tmp_path):
        _write(
            tmp_path,
            "QUOTE_2026-05-26_RU.md",
            "# Historical quote\n\n"
            "Older review wrote: \"Superseded by GHOST_2020.md\" "
            "<!-- audit-plan-supersession: allow -->\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert findings == [], (
            f"inline marker should waive the line, got {findings}"
        )

    def test_marker_on_different_line_does_not_waive(self, audit_mod, tmp_path):
        _write(
            tmp_path,
            "QUOTE.md",
            "<!-- audit-plan-supersession: allow -->\n"
            "Superseded by GHOST.md\n",
        )
        findings = audit_mod.audit_planning_tree(scan_root=tmp_path)
        assert len(findings) == 1


class TestResolveTarget:
    """Direct exercise of ``_resolve_target``."""

    def test_relative_to_source_dir(self, audit_mod, tmp_path):
        _write(tmp_path, "T.md", "# T\n")
        src = tmp_path / "S.md"
        src.write_text("placeholder", encoding="utf-8")
        assert audit_mod._resolve_target(
            "T.md", src, scan_root=tmp_path,
        ).name == "T.md"

    def test_missing_returns_none(self, audit_mod, tmp_path):
        src = tmp_path / "S.md"
        src.write_text("placeholder", encoding="utf-8")
        assert audit_mod._resolve_target(
            "GHOST.md", src, scan_root=tmp_path,
        ) is None


class TestLiveCorpus:
    """Smoke against the operator's live planning tree (gitignored)."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *args],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        )

    def test_live_default_exit_zero(self):
        """Live planning tree must remain clean (no filename-target
        supersession refs currently outstanding)."""
        result = self._run()
        assert result.returncode == 0, (
            f"live planning tree should be clean, got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_live_json_shape(self):
        result = self._run("--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "scan_root" in data
        assert "findings" in data
        assert "count" in data
        assert data["count"] == 0

    def test_missing_scan_root_returns_empty(self, tmp_path):
        """Non-existent scan root returns 0 findings (caller convention)."""
        ghost = tmp_path / "does-not-exist"
        result = self._run("--scan-root", str(ghost))
        assert result.returncode == 0

    def test_help_works(self):
        result = self._run("--help")
        assert result.returncode == 0
        assert "audit_plan_supersession" in result.stdout
        assert "--scan-root" in result.stdout

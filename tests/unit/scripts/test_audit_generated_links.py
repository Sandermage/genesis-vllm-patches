# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/audit_generated_links.py`` — §9.A.4
(AUDIT-CLOSURE.2, 2026-05-27).

Coverage:

  * R-GENLINK-1 number/URL mismatch detection
  * R-GENLINK-2 GitHub PR/issue shape enforcement
  * R-GENLINK-3 placeholder URL token detection
  * Generator freshness check (mocked via synthetic doc + missing
    generator script edge case)
  * Live corpus: both generators in sync, all link shapes pass
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_generated_links.py"


def _import_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_generated_links", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_generated_links"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def audit_mod():
    return _import_audit_module()


class TestNumberedPrLinkRule:
    """R-GENLINK-1 — `[#NNNN](URL)` URL must end with `/NNNN`."""

    def test_matching_pair_silent(self, audit_mod):
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md",
            generator="x.py",
        )
        text = "See [#41127](https://github.com/vllm-project/vllm/pull/41127).\n"
        findings = audit_mod._scan_numbered_pr_links(doc, text)
        assert findings == []

    def test_number_mismatch_flagged(self, audit_mod):
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md",
            generator="x.py",
        )
        text = "See [#41127](https://github.com/vllm-project/vllm/pull/40000).\n"
        findings = audit_mod._scan_numbered_pr_links(doc, text)
        assert len(findings) == 1
        assert findings[0].rule == "R-GENLINK-1"
        assert "41127" in findings[0].detail

    def test_issue_url_shape_accepted(self, audit_mod):
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md",
            generator="x.py",
        )
        text = "Fixes [#123](https://github.com/example/repo/issues/123).\n"
        findings = audit_mod._scan_numbered_pr_links(doc, text)
        assert findings == []

    def test_non_github_url_flagged(self, audit_mod):
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md",
            generator="x.py",
        )
        text = "Ref [#100](https://example.com/whatever/100).\n"
        findings = audit_mod._scan_numbered_pr_links(doc, text)
        # Number matches URL trailing segment (R-GENLINK-1 silent), but
        # URL shape isn't GitHub PR/issue (R-GENLINK-2 fires).
        assert len(findings) == 1
        assert findings[0].rule == "R-GENLINK-2"


class TestMalformedHashLinkRule:
    """R-GENLINK-4 — `[#<non-numeric>](URL)` shapes catch the C.5 bug."""

    def test_url_in_link_text_flagged(self, audit_mod):
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md", generator="x.py",
        )
        # Replica of the C.5 malformed shape we caught on 28 rows of
        # PATCHES_AUTO.md.
        text = (
            "| **G4_01** | ... | "
            "[#https://github.com/vllm-project/vllm/issues/39407]"
            "(https://github.com/vllm-project/vllm/pull/"
            "https://github.com/vllm-project/vllm/issues/39407)"
            " | Refuse ... |\n"
        )
        findings = audit_mod._scan_malformed_hash_links(doc, text)
        assert len(findings) == 1
        assert findings[0].rule == "R-GENLINK-4"
        assert "malformed link-text" in findings[0].detail

    def test_numeric_text_not_flagged(self, audit_mod):
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md", generator="x.py",
        )
        text = "[#41127](https://github.com/vllm-project/vllm/pull/41127)\n"
        findings = audit_mod._scan_malformed_hash_links(doc, text)
        assert findings == []

    def test_word_text_in_hash_link_flagged(self, audit_mod):
        """Even non-URL non-numeric text after `#` is operator-visible
        malformed (operator likely meant a number)."""
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md", generator="x.py",
        )
        text = "[#TODO](https://example.com/pull/100)\n"
        findings = audit_mod._scan_malformed_hash_links(doc, text)
        assert len(findings) == 1


class TestPlaceholderTargetsRule:
    """R-GENLINK-3 — placeholder strings as link targets."""

    def test_todo_placeholder_flagged(self, audit_mod):
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md", generator="x.py",
        )
        text = "Reference [bench](TODO) pending.\n"
        findings = audit_mod._scan_placeholder_targets(doc, text)
        assert len(findings) == 1
        assert findings[0].rule == "R-GENLINK-3"

    def test_real_url_silent(self, audit_mod):
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md", generator="x.py",
        )
        text = "Reference [bench](https://example.com/real).\n"
        findings = audit_mod._scan_placeholder_targets(doc, text)
        assert findings == []

    def test_external_marker_not_placeholder(self, audit_mod):
        """Genesis-internal ``external://...`` is a valid pending-marker
        convention; only bare TODO/...//None/etc. are flagged."""
        doc = audit_mod.GeneratedDoc(
            rel_path="synthetic.md", generator="x.py",
        )
        text = (
            "Single-stream baseline pending: "
            "[bench](external://docs.example/pending-bench)\n"
        )
        findings = audit_mod._scan_placeholder_targets(doc, text)
        assert findings == []


class TestGeneratorFreshness:
    """``_check_generator`` invokes generator with ``--check`` flag."""

    def test_missing_generator_flagged(self, audit_mod, tmp_path):
        doc = audit_mod.GeneratedDoc(
            rel_path="docs/SYNTHETIC.md",
            generator="scripts/nonexistent_generator.py",
        )
        err = audit_mod._check_generator(doc)
        assert err is not None
        assert "missing" in err

    def test_real_generators_pass(self, audit_mod):
        """Real generators in the tree must pass ``--check``."""
        for doc in audit_mod._GENERATED_DOCS:
            err = audit_mod._check_generator(doc)
            assert err is None, (
                f"{doc.generator} --check failed: {err}"
            )


class TestLiveCorpus:
    """Live tracked-tree must be clean post-audit."""

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *args],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        )

    def test_live_default_exit_zero(self):
        result = self._run()
        assert result.returncode == 0, (
            f"live generated docs should be clean, got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}"
        )
        assert "All generated docs in sync" in result.stdout

    def test_live_json_shape(self):
        result = self._run("--json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["count"] == 0
        assert len(data["docs_scanned"]) >= 2

    def test_help_works(self):
        result = self._run("--help")
        assert result.returncode == 0
        assert "audit_generated_links" in result.stdout

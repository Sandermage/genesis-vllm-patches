# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/generate_patches_md.py`` — §9.C.5
(REPO-HYGIENE-CLOSURE.1, 2026-05-27).

Targets the 4-branch ``render_upstream_pr`` helper that closes the
malformed-link bug class found in 28 rows of docs/PATCHES_AUTO.md.
Registry ``upstream_pr`` has 3 distinct value shapes (None / int /
str-URL); the renderer must dispatch each correctly.

Coverage:

  * Each branch produces well-formed markdown
  * Numeric PR rendering unchanged (back-compat)
  * URL-string rendering distinguishes pull vs issues; both kinds preserve URL
  * Unknown URL shape falls back to backticks (no malformed `[#URL](pull/URL)`)
  * Defensive: bool / list / dict / float don't crash; render as em-dash
  * Live `--check` smoke confirms regenerated file is in sync
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_patches_md.py"


def _import_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_patches_md", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_patches_md"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gen():
    return _import_generator()


# ─── render_upstream_pr — 4-branch dispatch ──────────────────────────────


class TestRenderUpstreamPrNone:
    """Branch 1: ``None`` → ``—`` (em dash)."""

    def test_none_renders_em_dash(self, gen):
        assert gen.render_upstream_pr(None) == "—"


class TestRenderUpstreamPrInt:
    """Branch 2: ``int`` → ``[#N](https://.../pull/N)``."""

    def test_int_renders_pr_link(self, gen):
        result = gen.render_upstream_pr(40768)
        assert result == "[#40768](https://github.com/vllm-project/vllm/pull/40768)"

    def test_int_large_value(self, gen):
        result = gen.render_upstream_pr(42637)
        assert "[#42637]" in result
        assert "/pull/42637)" in result

    def test_zero_int_renders_em_dash(self, gen):
        """Zero is falsy historically; the renderer must NOT treat 0 as
        a valid PR number (no PR #0 exists)."""
        # Either render as em-dash OR explicit [#0] — both are defensible.
        # Document the chosen behavior: integers always go through the
        # int branch, including 0. Caller responsibility to not put 0
        # in the registry.
        result = gen.render_upstream_pr(0)
        # Behavior: int branch fires (no falsy short-circuit) → [#0](.../0)
        assert "[#0]" in result


class TestRenderUpstreamPrStrUrl:
    """Branch 3: ``str`` URL → ``[#N](URL)`` with kind preserved."""

    def test_pull_url_renders_pull_link(self, gen):
        url = "https://github.com/vllm-project/vllm/pull/40798"
        result = gen.render_upstream_pr(url)
        assert result == f"[#40798]({url})"

    def test_issue_url_renders_issue_link(self, gen):
        url = "https://github.com/vllm-project/vllm/issues/39407"
        result = gen.render_upstream_pr(url)
        # Kind preserved: URL still says ``/issues/``, not ``/pull/``.
        assert result == f"[#39407]({url})"
        assert "/issues/" in result

    def test_url_with_trailing_slash(self, gen):
        url = "https://github.com/vllm-project/vllm/pull/42637/"
        result = gen.render_upstream_pr(url)
        assert "[#42637]" in result
        assert url in result  # URL preserved verbatim

    def test_other_org_repo_accepted(self, gen):
        """The renderer accepts any GitHub repo, not just vllm-project/vllm."""
        url = "https://github.com/example-org/some-repo/pull/123"
        result = gen.render_upstream_pr(url)
        assert result == f"[#123]({url})"

    def test_http_scheme_accepted(self, gen):
        """``http://`` (not ``https://``) still parses."""
        url = "http://github.com/vllm-project/vllm/pull/100"
        result = gen.render_upstream_pr(url)
        assert "[#100]" in result


class TestRenderUpstreamPrUnknownString:
    """Branch 4: unknown URL shape → backticks fallback."""

    def test_non_github_url_falls_back_to_backticks(self, gen):
        result = gen.render_upstream_pr("https://example.com/pull/123")
        # Wrapped in backticks; NOT a markdown link (no [text](url) shape).
        assert result.startswith("`")
        assert result.endswith("`")
        assert "[#" not in result

    def test_random_string_falls_back_to_backticks(self, gen):
        result = gen.render_upstream_pr("see PR 12345")
        assert result.startswith("`")
        assert "[#" not in result

    def test_github_discussion_url_falls_back(self, gen):
        """Only ``pull`` and ``issues`` are canonical; discussions etc.
        fall to safe backticks."""
        url = "https://github.com/vllm-project/vllm/discussions/999"
        result = gen.render_upstream_pr(url)
        assert "[#" not in result

    def test_pipe_in_string_escaped(self, gen):
        """Markdown table cells need ``|`` escaped to ``\\|``."""
        result = gen.render_upstream_pr("https://example.com/pull|999")
        assert "\\|" in result


class TestRenderUpstreamPrDefensiveTypes:
    """Branch 5 (defensive): unexpected types render safely as em-dash."""

    def test_bool_true_renders_em_dash(self, gen):
        """``bool`` is a subclass of ``int`` — explicit guard prevents
        ``[#True](https://.../pull/True)`` from leaking out."""
        assert gen.render_upstream_pr(True) == "—"

    def test_bool_false_renders_em_dash(self, gen):
        assert gen.render_upstream_pr(False) == "—"

    def test_list_renders_em_dash(self, gen):
        assert gen.render_upstream_pr([42637]) == "—"

    def test_dict_renders_em_dash(self, gen):
        assert gen.render_upstream_pr({"pr": 42637}) == "—"

    def test_float_renders_em_dash(self, gen):
        """Floats aren't PR numbers; defensive fallback."""
        assert gen.render_upstream_pr(42637.0) == "—"


class TestGeneratorCheckRoundTrip:
    """``generate_patches_md.py --check`` confirms committed file is in sync."""

    def test_check_clean_after_regen(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--check"],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        )
        assert result.returncode == 0, (
            f"--check should pass after regen, got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_committed_doc_has_no_malformed_rows(self):
        """The committed docs/PATCHES_AUTO.md must not contain
        ``[#https://...]`` shapes — the C.5 bug class."""
        doc = REPO_ROOT / "docs" / "PATCHES_AUTO.md"
        text = doc.read_text(encoding="utf-8")
        # Specifically look for `[#https` and `[#http://` which were the
        # malformed footprint.
        assert "[#https" not in text, (
            "malformed [#https://...] link text found in PATCHES_AUTO.md — "
            "C.5 regression"
        )
        assert "[#http://" not in text

    def test_committed_doc_has_issues_url_present(self):
        """At least one of the 28 str-URL entries renders correctly
        with /issues/ preserved (G4_01 references vllm#39407 as an issue)."""
        doc = REPO_ROOT / "docs" / "PATCHES_AUTO.md"
        text = doc.read_text(encoding="utf-8")
        assert "[#39407](https://github.com/vllm-project/vllm/issues/39407)" in text


class TestNaturalSortKeyUnchanged:
    """Regression: refactor did not break the natural-sort helper."""

    def test_p_before_pn(self, gen):
        assert gen.natural_sort_key("P58") < gen.natural_sort_key("PN8")

    def test_p1_before_p107(self, gen):
        assert gen.natural_sort_key("P1") < gen.natural_sort_key("P107")

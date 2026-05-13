# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/docs_stale_scan.py` — supplement §3 stale-token scan.

The script forbids tokens like `genesis doctor`, `./scripts/launch.sh`,
`vllm/sndr_core/wiring/`, etc. in public docs (everything except
`docs/_internal/`, `docs/upstream/`, `docs/reference/`, `_archive/`,
and a small migration-note allowlist).

Locking in the live committed corpus is the gating contract — the gate
was promoted from informational to gating on 2026-05-13.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "docs_stale_scan.py"


def _import():
    name = "_docs_stale_scan_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestForbiddenTokens:
    def test_known_set(self):
        mod = _import()
        assert "genesis doctor" in mod.FORBIDDEN_TOKENS
        assert "genesis verify" in mod.FORBIDDEN_TOKENS
        assert "genesis migrate" in mod.FORBIDDEN_TOKENS
        assert "./scripts/launch.sh" in mod.FORBIDDEN_TOKENS
        assert "vllm/sndr_core/wiring" in mod.FORBIDDEN_TOKENS
        assert "wiring/patch_" in mod.FORBIDDEN_TOKENS
        assert "vllm-server-mtp-test" in mod.FORBIDDEN_TOKENS

    def test_search_paths(self):
        mod = _import()
        assert "README.md" in mod.SEARCH_PATHS
        assert "docs" in mod.SEARCH_PATHS


class TestScanFile:
    def test_detects_forbidden_token(self, tmp_path):
        mod = _import()
        p = tmp_path / "x.md"
        p.write_text("run genesis doctor first\n")
        hits = mod._scan_file(p)
        assert hits
        assert hits[0][1] == "genesis doctor"

    def test_clean_file(self, tmp_path):
        mod = _import()
        p = tmp_path / "x.md"
        p.write_text("run sndr doctor first\n")
        assert mod._scan_file(p) == []

    def test_one_token_per_line(self, tmp_path):
        """When a line has two forbidden tokens, scanner records one
        hit per line (breaks after first match)."""
        mod = _import()
        p = tmp_path / "x.md"
        p.write_text("run genesis doctor and genesis verify\n")
        hits = mod._scan_file(p)
        assert len(hits) == 1


class TestAllowlist:
    def test_internal_allowlisted(self):
        mod = _import()
        assert mod._is_allowlisted(REPO_ROOT / "docs" / "_internal" / "x.md")

    def test_archive_allowlisted(self):
        mod = _import()
        assert mod._is_allowlisted(REPO_ROOT / "docs" / "archive" / "x.md")

    def test_public_doc_not_allowlisted(self):
        mod = _import()
        assert not mod._is_allowlisted(REPO_ROOT / "docs" / "PATCHES.md")


class TestLiveCorpus:
    def test_scan_clean_on_repo(self):
        """Live corpus must be clean — that's what gating enforces."""
        rc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert rc.returncode == 0, (
            f"docs_stale_scan failed on live corpus:\n"
            f"stdout:\n{rc.stdout}\n"
            f"stderr:\n{rc.stderr}"
        )

# SPDX-License-Identifier: Apache-2.0
"""Tests for `scripts/emit_paths_env.py` — canonical paths emit helper.

Contract:

  1. Default mode emits shell `export GENESIS_<KEY>=<value>` lines.
  2. --print emits pretty `<key> = <value>` lines.
  3. --prefix overrides the env-var prefix (default: GENESIS).
  4. Output is parseable by `bash -c "source"` (no shell syntax errors).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "emit_paths_env.py"


def _run(*extra) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )


class TestDefaultMode:
    def test_emits_export_lines(self):
        result = _run()
        assert result.returncode == 0
        # At least one line that looks like a shell export.
        assert re.search(r"^export GENESIS_\w+=", result.stdout, re.MULTILINE)

    def test_no_unbalanced_quotes(self):
        """Output must be sourcable — each export line balances quotes."""
        result = _run()
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line.startswith("export "):
                continue
            # Each line balances " and ' separately.
            assert line.count('"') % 2 == 0, f"unbalanced double-quote: {line}"
            assert line.count("'") % 2 == 0, f"unbalanced single-quote: {line}"

    def test_sourcable_by_bash(self):
        """`bash -c "source <(...)"` should accept output without error."""
        result = _run()
        # Pick the first GENESIS_<X>= var name and verify it resolves
        # after sourcing.
        first_var = None
        for line in result.stdout.splitlines():
            m = re.match(r"^export (GENESIS_\w+)=", line)
            if m:
                first_var = m.group(1)
                break
        assert first_var is not None, "no GENESIS_* exports emitted"

        check = subprocess.run(
            ["bash", "-c", f"{result.stdout}\necho \"${first_var}\""],
            capture_output=True, text=True, timeout=5,
        )
        assert check.returncode == 0, (
            f"bash source failed: {check.stderr}"
        )
        assert check.stdout.strip(), (
            f"{first_var} resolved empty after source"
        )


class TestPrintMode:
    def test_print_emits_kv_pairs(self):
        result = _run("--print")
        assert result.returncode == 0
        # Pretty format: `key = value` separated by " = "
        assert " = " in result.stdout
        # No shell export prefix in print mode
        assert not result.stdout.startswith("export ")


class TestCustomPrefix:
    def test_sndr_prefix_works(self):
        result = _run("--prefix", "SNDR")
        assert result.returncode == 0
        assert re.search(r"^export SNDR_\w+=", result.stdout, re.MULTILINE)

    def test_default_prefix_still_genesis(self):
        result = _run()
        # Should be no SNDR_ lines by default
        for line in result.stdout.splitlines():
            assert not line.startswith("export SNDR_"), line

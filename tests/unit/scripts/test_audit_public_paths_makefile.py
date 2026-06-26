# SPDX-License-Identifier: Apache-2.0
"""Tests for `make audit-public-paths` — the Makefile-only public-paths gate.

Context (commit 47b74cdd, 2026-05-30): the previous implementation used
``rg`` (ripgrep) which is not universally installed. On systems without
rg, the subshell silently returned empty stdout and the gate falsely
reported "clean" — meaning the gate had been broken on any developer
machine without ripgrep. Replaced with portable ``grep -rE``.

These tests prove the gate ACTUALLY detects violations. They invoke
``make audit-public-paths`` against a temp tree with an injected
violation, then assert non-zero exit + violation text in stdout.

Contract enforced:
  1. Clean tree → exit 0
  2. Tree with private IP (192.168.1.50) → exit 1 + line printed
  3. Tree with /home/sander → exit 1 + line printed
  4. Tree with a forbidden maintainer username (sander@host) → exit 1 + line printed
  5. Files inside sndr_private/ are NOT scanned (waiver)
  6. Hits in __pycache__ are excluded
"""
from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_gate(workdir: Path) -> subprocess.CompletedProcess:
    """Invoke `make audit-public-paths` in workdir."""
    # We use the same Makefile but copy a minimal subset into workdir so
    # `make` resolves the audit-public-paths target with the scanned
    # directories pointing at workdir.
    env = os.environ.copy()
    return subprocess.run(
        ["make", "-f", str(REPO_ROOT / "Makefile"), "audit-public-paths"],
        cwd=workdir,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def _build_tree(workdir: Path, files: dict[str, str]) -> None:
    """Create the scanned-directory layout under workdir matching the
    Makefile target's scope (README.md, docs/, scripts/, tools/,
    benchmarks/, vllm/)."""
    for rel, content in files.items():
        path = workdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def test_clean_tree_passes(tmp_path):
    """A tree with no forbidden paths returns exit 0."""
    _build_tree(tmp_path, {
        "README.md": "Genesis vLLM Patches\n\nNothing private here.\n",
        "scripts/example.sh": "#!/bin/bash\nset -e\necho 'safe'\n",
        "vllm/sample.py": "# safe code\nprint('hi')\n",
    })
    result = _run_gate(tmp_path)
    assert result.returncode == 0, (
        f"Expected exit 0 on clean tree, got {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "clean" in result.stdout


def test_private_ip_in_script_fails(tmp_path):
    """A script containing 192.168.1.50 must be flagged."""
    _build_tree(tmp_path, {
        "scripts/connect.sh": "#!/bin/bash\nssh user@192.168.1.50\n",
    })
    result = _run_gate(tmp_path)
    assert result.returncode != 0, (
        f"Expected non-zero exit on private-IP violation, got 0.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "192.168.1.50" in result.stdout
    assert "scripts/connect.sh" in result.stdout


def test_home_sander_in_yaml_fails(tmp_path):
    """A YAML with /home/sander path must be flagged."""
    _build_tree(tmp_path, {
        "vllm/config.yaml": "model: /home/sander/models/foo\n",
    })
    result = _run_gate(tmp_path)
    assert result.returncode != 0
    assert "/home/sander" in result.stdout
    assert "config.yaml" in result.stdout


def test_sander_at_host_in_doc_fails(tmp_path):
    """A doc with the forbidden maintainer username (sander@host) must be
    flagged. The gate carries the maintainer's own username in its
    forbidden-identifier list so it can catch operator leaks in public
    files; this fixture exercises that rule."""
    _build_tree(tmp_path, {
        "docs/RUNBOOK.md": "Run: `ssh sander@host`\n",
    })
    result = _run_gate(tmp_path)
    assert result.returncode != 0
    assert "sander@" in result.stdout


def test_sndr_private_dir_excluded(tmp_path):
    """Hits inside sndr_private/ are waived even if they contain forbidden
    paths (maintainer planning tree, gitignored)."""
    _build_tree(tmp_path, {
        "sndr_private/planning/notes.md": (
            "Internal planning. Reach via 192.168.1.50 / /home/sander.\n"
        ),
    })
    result = _run_gate(tmp_path)
    assert result.returncode == 0, (
        f"sndr_private should be excluded, got non-zero.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_pycache_excluded(tmp_path):
    """Hits inside __pycache__ are excluded — compiled .pyc artifacts
    routinely contain stale source paths."""
    _build_tree(tmp_path, {
        "vllm/__pycache__/binary.pyc": (
            "fake bytecode mentioning /home/sander/genesis-vllm-patches\n"
        ),
    })
    result = _run_gate(tmp_path)
    assert result.returncode == 0

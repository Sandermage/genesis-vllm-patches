# SPDX-License-Identifier: Apache-2.0
"""Tests for install.sh thin-bootstrap shim + plugin packaging metadata.

S-05 (2026-05-08): install.sh shrank from 783 lines to ~106 — all
operator logic (GPU detection, workload picker, pin resolution, clone,
plugin install, host paths, launch script generation, smoke test,
uninstall) moved into `vllm.sndr_core.cli.install`. The bootstrap's
only job is now: verify python+git, clone the repo, `pip install -e
.`, then exec `sndr install` with passed flags.

Behavior coverage of the canonical wizard lives in
`tests/installer/test_install_dry_run.py` and
`tests/unit/detection/test_*.py`. The tests here assert the BOOTSTRAP
layer remains valid bash that delegates to the canonical wizard.

Plus: tests for the `tools/genesis_vllm_plugin/pyproject.toml` (entry
points + console scripts) — unchanged from prior version.

Author: Sandermage (Sander) Barzov Aleksandr.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = REPO_ROOT / "install.sh"
PLUGIN_PYPROJECT = (
    REPO_ROOT / "tools" / "genesis_vllm_plugin" / "pyproject.toml"
)


# ─────────────────────────────────────────────────────────────────
# install.sh: shim file structure
# ─────────────────────────────────────────────────────────────────


def test_install_sh_exists():
    assert INSTALL_SH.is_file(), f"install.sh missing at {INSTALL_SH}"


def test_install_sh_has_bash_shebang():
    first_line = INSTALL_SH.read_text().splitlines()[0]
    assert first_line.startswith("#!"), f"missing shebang: {first_line}"
    assert "bash" in first_line


def test_install_sh_is_executable():
    mode = os.stat(INSTALL_SH).st_mode
    assert mode & stat.S_IXUSR, "install.sh not executable (chmod +x)"


def test_install_sh_bash_parses_clean():
    """`bash -n install.sh` reports zero syntax errors."""
    r = subprocess.run(
        ["bash", "-n", str(INSTALL_SH)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"bash -n failed: {r.stderr}"


def test_install_sh_uses_strict_mode():
    """All scripts must run under `set -euo pipefail`."""
    content = INSTALL_SH.read_text()
    assert "set -euo pipefail" in content


@pytest.mark.skip(
    reason="install.sh kept feature-complete (~700 LOC) — the S-05 "
           "thin-shim refactor target was reversed after the canonical "
           "`sndr install` wizard turned out to require a real Python "
           "interpreter resolution + repo-clone sequence that's cleaner "
           "in bash. New logic still belongs in vllm/sndr_core/cli/"
           "install.py for testability; this test stays as a guard "
           "against unbounded growth — re-enable with a higher ceiling "
           "once we settle on the long-term shape."
)
def test_install_sh_is_thin_shim_post_s05():
    """S-05 fix (2026-05-08): the bootstrap should be small. The old
    file was 783 lines of bash; the post-refactor target was ≤200."""
    n_lines = len(INSTALL_SH.read_text().splitlines())
    assert n_lines <= 200, (
        f"install.sh has grown to {n_lines} lines — keep it as a thin "
        "bootstrap and put new logic in vllm/sndr_core/cli/install.py"
    )


# ─────────────────────────────────────────────────────────────────
# install.sh: delegates to canonical wizard
# ─────────────────────────────────────────────────────────────────


@pytest.mark.skip(
    reason="install.sh handles installation directly (paired with the "
           "thin-shim reversal above); operator-facing logic is still "
           "duplicated in vllm/sndr_core/cli/install.py for IDE / unit "
           "testing. Re-enable when the bash bootstrap is fully retired."
)
def test_install_sh_delegates_to_sndr_install():
    """Bootstrap must hand off to `sndr install` (or python -m fallback)
    so all operator-facing logic lives in the canonical Python wizard."""
    content = INSTALL_SH.read_text()
    # Either `exec sndr install` (if on PATH) or python -m fallback
    has_sndr_exec = "exec sndr install" in content
    has_python_module_exec = (
        "vllm.sndr_core.cli install" in content
        or "vllm.sndr_core.cli" in content
    )
    assert has_sndr_exec and has_python_module_exec, (
        "bootstrap must exec `sndr install` AND fall back to "
        "`python -m vllm.sndr_core.cli install` for hosts where the "
        "console script isn't on PATH yet"
    )


def test_install_sh_python_minimum_version_check():
    """Bootstrap must verify Python ≥3.10 before delegating."""
    content = INSTALL_SH.read_text()
    # Look for the Python version comparison logic
    assert "PY_MAJOR" in content or "3.10" in content
    # Ensure the failure path exists
    assert "die" in content or "exit 1" in content


def test_install_sh_clones_into_sndr_home():
    """Bootstrap must default `SNDR_HOME` to `~/.sndr` (with `~/.genesis`
    legacy alias for back-compat with v7.x operators)."""
    content = INSTALL_SH.read_text()
    assert "SNDR_HOME" in content
    assert ".sndr" in content
    # Legacy alias preserved
    assert "GENESIS_HOME" in content


def test_install_sh_pip_install_uses_user_default():
    """Pip install defaults to `--user` (safer than system-wide); operator
    can override via `SNDR_PIP_FLAGS=`."""
    content = INSTALL_SH.read_text()
    assert "--user" in content
    assert "SNDR_PIP_FLAGS" in content


def test_install_sh_no_genesis_legacy_module_refs():
    """Bootstrap must not reference removed `vllm._genesis` paths."""
    content = INSTALL_SH.read_text()
    assert "vllm._genesis" not in content, (
        "install.sh still references the removed vllm._genesis package"
    )


# ─────────────────────────────────────────────────────────────────
# Plugin pyproject.toml — unchanged from prior version
# ─────────────────────────────────────────────────────────────────


def test_plugin_pyproject_exists():
    assert PLUGIN_PYPROJECT.is_file(), f"missing {PLUGIN_PYPROJECT}"


def test_plugin_pyproject_declares_genesis_console_script():
    """After `pip install`, `genesis` should be a top-level command."""
    content = PLUGIN_PYPROJECT.read_text()
    assert "[project.scripts]" in content, (
        "plugin pyproject.toml must declare [project.scripts] for the "
        "`genesis` console command"
    )
    assert "genesis = " in content
    assert "vllm.sndr_core.compat.cli:main" in content


def test_plugin_pyproject_declares_vllm_general_plugins_entry_point():
    content = PLUGIN_PYPROJECT.read_text()
    assert '[project.entry-points."vllm.general_plugins"]' in content
    assert "genesis_v7" in content


def test_plugin_pyproject_requires_python_3_10_plus():
    content = PLUGIN_PYPROJECT.read_text()
    assert 'requires-python = ">=3.10"' in content


def test_plugin_pyproject_apache_2_license():
    content = PLUGIN_PYPROJECT.read_text()
    assert "Apache-2.0" in content

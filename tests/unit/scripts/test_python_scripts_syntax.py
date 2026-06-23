# SPDX-License-Identifier: Apache-2.0
"""Minimal test coverage for Python scripts — TOOLING-HARDENING.2 L.6
(2026-05-26).

Counterpart to ``test_shell_scripts_syntax.py`` (L.7) but for Python:

  * ``py_compile`` for every active script under ``scripts/`` and
    ``tools/`` (auto-discovered, parametrized) — universal safe check.
  * ``--help`` smoke (subprocess) ONLY for an explicit allowlist of
    scripts where argparse / sub-parser parses ``--help`` and exits
    cleanly before any network / docker / rig side effect.

Excluded by design:
  * Files under ``_retired/`` or ``_archive/`` subtrees.
  * ``__init__.py`` files (plugin package init; covered transitively
    when their package gets imported).
  * Scripts with top-level side effects that prevent ``--help``:
      - ``audit_schema_sync.py`` / ``check_upstream_drift.py`` — no
        argparse, run their main work directly.
      - ``tools/external_probe/patch_*.py`` — apply text patches at
        import time.
      - ``tools/multi_conc_bench.py`` — top-level ``import aiohttp``
        (optional dep not present in the test venv).
      - ``tools/examples/.../plugin.py`` — plugin entry, no CLI.
    These still get ``py_compile`` coverage.

Scope discipline: this test file does **not** execute scripts beyond
``--help`` (no network, no docker, no SSH, no rig). A script that
launches a server / makes HTTP calls / opens GPU on ``--help`` would
be a bug worth fixing before adding it to the allowlist.

History:
  * 2026-05-26 — audit_patch_attribution.py line 237 had an unescaped
    ``%`` in argparse help text that crashed ``--help`` with
    ``TypeError: %i format: a real number is required, not dict``.
    Fixed (one-char ``%`` → ``%%``) in this commit so the script
    joins the HELP_SAFE allowlist.
"""
from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _discover_python_scripts() -> list[Path]:
    """Return active .py scripts under scripts/ + tools/, sorted.

    Excludes ``_retired/`` / ``_archive/`` / ``__pycache__/`` subtrees
    and ``__init__.py`` files (plugin package init has no CLI).
    """
    roots = [REPO_ROOT / "scripts", REPO_ROOT / "tools"]
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*.py"):
            parts = set(p.relative_to(REPO_ROOT).parts)
            if "_retired" in parts or "_archive" in parts or "__pycache__" in parts:
                continue
            if p.name == "__init__.py":
                continue
            out.append(p)
    return sorted(out)


PYTHON_SCRIPTS = _discover_python_scripts()
_PY_IDS = [str(p.relative_to(REPO_ROOT)) for p in PYTHON_SCRIPTS]


# Explicit allowlist for ``--help`` smoke. Add a script here only after
# manually verifying:
#   1. argparse parses ``--help`` and exits 0 before any side effect.
#   2. No top-level imports that require optional deps not in the test venv.
#   3. No top-level code that opens sockets / makes HTTP calls / spawns
#      processes / touches GPU.
HELP_SAFE_SCRIPTS: tuple[str, ...] = (
    # ── scripts/ — audit + utility ────────────────────────────────────
    "scripts/attach_bench_proof.py",
    "scripts/audit_artifacts.py",
    "scripts/audit_configs.py",
    "scripts/audit_license_anchor.py",
    "scripts/audit_no_new_v1.py",
    "scripts/audit_patch_attribution.py",
    "scripts/audit_patch_plan_resolves.py",
    "scripts/audit_private_namespace.py",
    "scripts/build_anchor_manifest.py",
    "scripts/check_dirty_state.py",
    "scripts/emit_paths_env.py",
    "scripts/generate_configs_md.py",
    "scripts/generate_patches_md.py",
    "scripts/generate_sbom.py",
    "scripts/security_scan.py",
    "scripts/stress/genesis_stress_v1.py",
    # ── tools/ — bench / probe / utility ──────────────────────────────
    "tools/bench_decode_tpot_clean_ab.py",
    "tools/genesis_bench_suite.py",
    "tools/kv_calc.py",
    # tools/license_keygen.py moved to sndr_private/tools/ (commercial license
    # tooling kept out of the public tree); no longer a public help-safe script.
    "tools/openai_smoke.py",
    "tools/progressive_context_probe.py",
)


def test_inventory_non_empty():
    """Guard rail — if the discovery rule changes, fail loudly rather
    than silently passing zero parametrized tests."""
    assert len(PYTHON_SCRIPTS) >= 20, (
        f"expected ≥20 Python scripts, found {len(PYTHON_SCRIPTS)}: "
        f"{[str(p) for p in PYTHON_SCRIPTS]}"
    )


def test_help_safe_paths_exist():
    """Every path in HELP_SAFE_SCRIPTS must resolve to an active file —
    catches typos and stale allowlist entries after script renames/moves."""
    missing = [
        rel for rel in HELP_SAFE_SCRIPTS
        if not (REPO_ROOT / rel).is_file()
    ]
    assert not missing, (
        f"HELP_SAFE_SCRIPTS references {len(missing)} missing path(s): "
        f"{missing}"
    )


@pytest.mark.parametrize("script", PYTHON_SCRIPTS, ids=_PY_IDS)
def test_py_compile_clean(script: Path, tmp_path):
    """Every active script parses cleanly. Catches:

      * SyntaxError (unbalanced brackets, bad indentation, stray operators)
      * Bad f-string contents (Python 3.12+ stricter f-string parser)
      * Encoding errors at module load
    """
    cfile = tmp_path / (script.stem + ".pyc")
    try:
        py_compile.compile(str(script), cfile=str(cfile), doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(
            f"{script.relative_to(REPO_ROOT)} failed py_compile:\n{e}"
        )


@pytest.mark.parametrize("rel_path", HELP_SAFE_SCRIPTS)
class TestHelpSmoke:
    """``python3 <script> --help`` must exit 0 and print a usage line.

    Catches:
      * argparse help-text format bugs (unescaped ``%`` is a real one
        we hit on audit_patch_attribution.py at L.6 read-only audit).
      * Top-level imports that fail in the test venv (missing optional
        deps that broke since the script was last manually run).
      * Missing main guard / wrong CLI entry point.
    """

    def _run_help(self, rel_path: str) -> subprocess.CompletedProcess:
        script_path = REPO_ROOT / rel_path
        return subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True, text=True, cwd=REPO_ROOT, check=False,
        )

    def test_help_exits_zero(self, rel_path: str):
        result = self._run_help(rel_path)
        assert result.returncode == 0, (
            f"{rel_path} --help should exit 0, got rc={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_help_prints_usage_line(self, rel_path: str):
        """argparse-generated help always starts with ``usage: ``."""
        result = self._run_help(rel_path)
        out = result.stdout + result.stderr
        assert "usage:" in out.lower(), (
            f"{rel_path} --help output lacks 'usage:' line.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

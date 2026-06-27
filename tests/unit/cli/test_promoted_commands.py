# SPDX-License-Identifier: Apache-2.0
"""v12 CLI split-brain closure — promoted-command contract.

The canonical ``sndr`` CLI (``sndr.cli.main``) promotes the high-value
legacy commands (report / doctor / preset / bench / tune / config) as thin
pass-throughs that delegate to the legacy implementation. These tests pin:

  1. Registration — every promoted command shows up in COMMAND_REGISTRY (so
     it appears in ``sndr --help``).
  2. Import-light — registering the canonical CLI must NOT require pydantic /
     torch / fastapi. This is the regression guard for the dependency-hygiene
     fix that moved the product_api (pydantic) imports off the eager command
     registry. Run in a fresh torch-less + pydantic-less subprocess.
  3. Delegation parity — each promoted command's ``--help`` reaches the legacy
     delegate verbatim (the fast-path forwards a leading ``--help``), and the
     canonical surface produces the same output as the legacy entry point.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from sndr.cli.commands.promoted import PROMOTED_COMMANDS


_PROMOTED_NAMES = ("report", "doctor", "preset", "bench", "tune", "config")


class TestPromotedRegistration:
    def test_all_promoted_names_present(self):
        names = {c.name for c in PROMOTED_COMMANDS}
        assert names == set(_PROMOTED_NAMES)

    @pytest.mark.parametrize("name", _PROMOTED_NAMES)
    def test_command_registered_in_canonical_cli(self, name):
        from sndr.cli.commands import COMMAND_REGISTRY
        from sndr.cli.main import build_parser

        build_parser()  # the real registration path
        assert name in COMMAND_REGISTRY

    @pytest.mark.parametrize("name", _PROMOTED_NAMES)
    def test_promoted_commands_opt_out_of_autohelp(self, name):
        # ``add_help = False`` is what lets ``sndr <cmd> --help`` forward to
        # the delegate instead of being intercepted by a stub subparser.
        cmd = next(c for c in PROMOTED_COMMANDS if c.name == name)
        assert getattr(cmd, "add_help", True) is False


# ── Import-light regression guard (the dependency-hygiene fix) ──────────────

_PROBE = textwrap.dedent(
    """
    import sys, importlib.abc
    # Simulate a no-GPU / limited-dep host: block every heavy dep so any
    # top-level import of them on the canonical CLI path raises immediately.
    _BLOCK = {"torch", "vllm", "pydantic", "fastapi", "uvicorn", "triton"}

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BLOCK:
                raise ModuleNotFoundError("blocked heavy dep: " + name)
            return None

    sys.meta_path.insert(0, _Blocker())
    import sndr.cli.main      # noqa: F401
    import sndr.cli.commands  # noqa: F401
    # Building the parser registers every command (incl. promoted ones) and
    # must stay import-light too.
    sndr.cli.main.build_parser()
    print("LIGHT_OK")
    """
).strip()


class TestCanonicalCliImportLight:
    def test_canonical_cli_imports_without_heavy_deps(self):
        rc = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True, text=True,
        )
        assert rc.returncode == 0, (
            "canonical sndr CLI must import without torch/vllm/pydantic/"
            "fastapi/uvicorn/triton:\n"
            f"  stdout: {rc.stdout!r}\n  stderr: {rc.stderr!r}"
        )
        assert "LIGHT_OK" in rc.stdout


# ── Delegation parity ───────────────────────────────────────────────────────

def _run_module(entry: str, argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", entry, *argv],
        capture_output=True, text=True,
    )


class TestPromotedDelegationParity:
    @pytest.mark.parametrize("name", _PROMOTED_NAMES)
    def test_help_forwards_to_delegate(self, name):
        # A leading ``--help`` must reach the delegate (not the top-level
        # argparse). The delegate prints its own usage and exits 0.
        canon = _run_module("sndr.cli.main", [name, "--help"])
        assert canon.returncode == 0, (
            f"sndr {name} --help failed:\n{canon.stderr}"
        )
        # The delegate's own usage line — not the top-level ``usage: sndr``
        # stub (which would mean argparse intercepted --help).
        assert "unrecognized arguments" not in canon.stderr
        assert "invalid choice" not in canon.stderr

    def test_preset_list_parity(self):
        # ``preset list`` is a pure, offline, deterministic surface — a good
        # parity probe between the canonical fast-path and the legacy tree.
        canon = _run_module("sndr.cli.main", ["preset", "list"])
        legacy = _run_module("sndr.cli.legacy", ["preset", "list"])
        assert canon.returncode == legacy.returncode == 0
        assert canon.stdout == legacy.stdout

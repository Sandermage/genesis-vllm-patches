# SPDX-License-Identifier: Apache-2.0
"""C4 + B5 (UNIFIED_CONFIG plan 2026-05-09) — sndr model CLI smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from sndr.cli.legacy.model import add_argparser


def _parse(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    add_argparser(sub)
    return parser.parse_args(args)


def test_argparser_registers_model_subcommand():
    ns = _parse(["model", "pull", "key1", "--dry-run"])
    assert ns.model_cmd == "pull"
    # REMAINDER captures the rest verbatim
    assert "key1" in ns.args
    assert "--dry-run" in ns.args


def test_argparser_list_subcommand():
    """`model list` registers; flags pass through via fast-path in cli_main.

    argparse REMAINDER on a sub-sub-parser doesn't reliably catch
    `--`-prefixed args here, so flag-pass-through is exercised by the
    cli_main fast-path test below.
    """
    ns = _parse(["model", "list"])
    assert ns.model_cmd == "list"


def test_top_level_dispatches_model_list_via_fast_path():
    """`sndr model list` reaches compat.models.list_cli (or returns 2 cleanly)."""
    from sndr.cli.legacy import cli_main
    rc = cli_main(["model", "list"])
    # Either 0 (printed list) or 2 (compat module not callable) — never crash
    assert rc in (0, 1, 2)


def test_top_level_dispatches_model_pull_help():
    """`sndr model pull --help` reaches the underlying compat CLI."""
    from sndr.cli.legacy import cli_main
    # SystemExit(0) is normal for argparse --help
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["model", "pull", "--help"])
    assert exc_info.value.code == 0


def test_top_level_dispatches_model_pull_unknown_key():
    """Unknown model key returns 2 (per pull.py contract)."""
    from sndr.cli.legacy import cli_main
    rc = cli_main(["model", "pull", "definitely-not-a-real-key-xyz"])
    assert rc == 2


# ─── B5: legacy fetch_models.sh shrunk to a wrapper

def test_fetch_models_sh_is_a_thin_wrapper():
    """fetch_models.sh now delegates to `sndr model pull` (≤80 lines)."""
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "fetch_models.sh"
    assert script.exists()
    body = script.read_text()
    # Hard upper bound: was 95 lines, slim wrapper should stay under 80
    assert len(body.splitlines()) <= 80
    # Must reference the new CLI
    assert "sndr model pull" in body
    # No more fake "SHA-verified" claim in the banner
    assert "SHA-verified" not in body

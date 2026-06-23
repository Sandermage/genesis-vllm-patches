# SPDX-License-Identifier: Apache-2.0
"""`sndr launch <preset> --dry-run -y` — stdout must carry ONLY the script.

Integrity-audit fix (2026-06-23): the dry-run path printed its advisory
output (the box-banner, the partial/placeholder patch warnings, and the
UNRESOLVED MOUNTS diagnostics) to STDOUT, interleaved with the rendered
vllm launch script. That broke the documented pipe contract:

    python3 -m sndr.cli.legacy launch <preset> --dry-run -y | bash

`bash -n` on the captured stdout returned rc=2 because the banner glyphs
and `⚠`-prefixed warnings parsed as garbage shell tokens. The fix routes
ALL non-script advisory output to STDERR on the dry-run path (via the new
`err=True` kwarg on _io.banner/info/warn), leaving stdout with the runnable
script alone. `_emit_unresolved_mount_diagnostics`'s docstring already
CLAIMED stderr ("so the preview output stays clean for piping") — this
makes the code match the comment.

These tests pin the contract so a future edit cannot silently regress the
banner/warnings back onto stdout.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


# One gemma preset + one qwen preset. Both ship enabled partial-status
# patches, so the `⚠ … implementation_status='partial'` warning fires —
# the exact line that previously contaminated stdout and broke `bash -n`.
_PRESETS = ["prod-gemma4-26b-default", "prod-qwen3.6-27b-dflash"]

# Advisory glyphs / strings that belong on STDERR, never on the
# script-carrying stdout.
_BANNER_GLYPHS = ("┌", "└", "│")
_FORBIDDEN_ON_STDOUT = ("SNDR Launch", "UNRESOLVED MOUNTS", "⚠")


def _run_dry_run(preset: str) -> subprocess.CompletedProcess[str]:
    """Run `sndr launch <preset> --dry-run -y`, capturing stdout/stderr
    SEPARATELY (the whole point of the test — they must not be merged)."""
    return subprocess.run(
        [sys.executable, "-m", "sndr.cli.legacy", "launch", preset,
         "--dry-run", "-y"],
        capture_output=True, text=True, timeout=120,
    )


@pytest.mark.parametrize("preset", _PRESETS)
def test_dry_run_stdout_is_runnable_bash(preset: str):
    """`bash -n` on the captured stdout ALONE must return 0 — stdout is
    the rendered script and nothing else."""
    proc = _run_dry_run(preset)
    assert proc.returncode == 0, (
        f"`sndr launch {preset} --dry-run -y` exited {proc.returncode}\n"
        f"stderr:\n{proc.stderr}"
    )
    # The script must actually look like a script, not be empty.
    assert proc.stdout.lstrip().startswith("#!/usr/bin/env bash"), (
        f"stdout does not start with a bash shebang:\n{proc.stdout[:200]}"
    )
    check = subprocess.run(
        ["bash", "-n"], input=proc.stdout, capture_output=True, text=True,
    )
    assert check.returncode == 0, (
        f"`bash -n` rejected the dry-run stdout for {preset} "
        f"(rc={check.returncode}) — advisory output is contaminating the "
        f"script:\n{check.stderr}"
    )


@pytest.mark.parametrize("preset", _PRESETS)
def test_dry_run_banner_and_warnings_absent_from_stdout(preset: str):
    """The banner glyphs, the 'SNDR Launch' title, the '⚠' warning marker,
    and the 'UNRESOLVED MOUNTS' diagnostic must NOT appear on stdout — they
    belong on stderr."""
    proc = _run_dry_run(preset)
    assert proc.returncode == 0, proc.stderr

    for glyph in _BANNER_GLYPHS:
        assert glyph not in proc.stdout, (
            f"banner glyph {glyph!r} leaked into stdout for {preset}"
        )
    for needle in _FORBIDDEN_ON_STDOUT:
        assert needle not in proc.stdout, (
            f"advisory string {needle!r} leaked into stdout for {preset}"
        )


@pytest.mark.parametrize("preset", _PRESETS)
def test_dry_run_banner_present_on_stderr(preset: str):
    """Sanity: the advisory output was not dropped — it moved to stderr."""
    proc = _run_dry_run(preset)
    assert proc.returncode == 0, proc.stderr
    assert "SNDR Launch" in proc.stderr, (
        "banner vanished entirely — it should be on stderr, not deleted"
    )

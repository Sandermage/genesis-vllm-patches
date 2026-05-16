# SPDX-License-Identifier: Apache-2.0
"""Phase D — `sndr launch --policy` opt-in filter on the live launch path.

Mirrors Phase C's compose-render flow: when --policy is set, the
patch_plan resolver runs before `to_launch_script()` and the rendered
shell script carries the filtered genesis_env instead of the raw
matrix. Default (no flag) keeps the legacy unfiltered path
byte-for-byte.

The tests run launch in dry-run mode and inspect the rendered script
content. dry-run is the only assertable surface that doesn't actually
exec the container — perfect for CI.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from vllm.sndr_core.cli import cli_main


def _run_cli(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = cli_main(argv)
    except SystemExit as e:
        rc = int(e.code) if isinstance(e.code, int) else 2
    return rc, buf.getvalue()


# ─── Default (no --policy) — backwards compat ────────────────────────────


class TestBackwardsCompat:
    def test_dry_run_without_policy_renders_full_env(self):
        rc, out = _run_cli([
            "launch", "prod-35b", "--dry-run", "-y",
        ])
        assert rc == 0, out[-500:]
        # The 35B PROD config has GENESIS_PN95_CONFIG_KEY (a parameter
        # that minimal would normally drop if treated as toggle — but
        # since this is no-policy mode, every env survives).
        assert "GENESIS_PN95_CONFIG_KEY" in out
        # No policy banner.
        assert "patch plan policy" not in out.lower()


# ─── --policy compat / safe / minimal ────────────────────────────────────


class TestPolicyFilters:
    def test_dry_run_with_compat_keeps_parameters(self):
        rc, out = _run_cli([
            "launch", "prod-35b", "--dry-run", "-y", "--policy", "compat",
        ])
        assert rc == 0, out[-500:]
        # Parameter must survive every policy.
        assert "GENESIS_PN95_CONFIG_KEY" in out

    def test_dry_run_with_minimal_drops_unknown_toggles(self):
        rc, out = _run_cli([
            "launch", "prod-35b", "--dry-run", "-y", "--policy", "minimal",
        ])
        assert rc == 0, out[-500:]
        # Parameter still survives minimal.
        assert "GENESIS_PN95_CONFIG_KEY" in out
        # An unknown-role toggle should be gone. Pick one we know is in
        # the 35B matrix but not in our backfilled attribution set —
        # e.g. GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE.
        assert "GENESIS_ENABLE_PN71_THINKING_TAG_NORMALIZE" not in out
        # A backfilled load_bearing toggle MUST stay (P67).
        assert "GENESIS_ENABLE_P67_TQ_MULTI_QUERY_KERNEL" in out


# ─── Invalid policy ──────────────────────────────────────────────────────


class TestInvalidPolicy:
    def test_bogus_policy_rejected(self):
        rc, _ = _run_cli([
            "launch", "prod-35b", "--dry-run", "-y", "--policy", "bogus",
        ])
        assert rc != 0

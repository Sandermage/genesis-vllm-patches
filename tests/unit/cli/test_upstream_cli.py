# SPDX-License-Identifier: Apache-2.0
"""C17 (UNIFIED_CONFIG plan 2026-05-09) — sndr upstream CLI tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sndr.cli.legacy.upstream import (
    add_argparser, run_check, run_show, run_list,
)


# Phase 10 (2026-06-01): V1 sunset — `sndr upstream` resolves preset via
# V1 registry only (no V2 alias dispatch yet). Mark V1-bound tests so
# they skip when the V1 file retires; non-config-bound tests (list, top-
# level dispatch, no-config check) continue to run.
_V1_DIR_UP = (Path(__file__).resolve().parents[3] / "vllm" / "sndr_core"
              / "model_configs" / "builtin")
_skip_if_no_v1_35b_up = pytest.mark.skipif(
    not (_V1_DIR_UP / "a5000-2x-35b-prod.yaml").is_file(),
    reason="V1 fixture a5000-2x-35b-prod.yaml retired (Phase 10 sunset)",
)
_skip_if_no_v1_27b_up = pytest.mark.skipif(
    not (_V1_DIR_UP / "a5000-2x-27b-int4-tq-k8v4.yaml").is_file(),
    reason="V1 fixture a5000-2x-27b-int4-tq-k8v4.yaml retired (Phase 10 sunset)",
)


def _parse(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    add_argparser(sub)
    return parser.parse_args(args)


# ─── argparser shape

def test_argparser_check_subcommand():
    ns = _parse(["upstream", "check"])
    assert ns.upstream_cmd == "check"
    assert ns.config is None
    assert ns.json is False


@_skip_if_no_v1_35b_up
def test_argparser_check_with_config():
    ns = _parse(["upstream", "check", "--config", "a5000-2x-35b-prod",
                 "--json", "--strict"])
    assert ns.config == "a5000-2x-35b-prod"
    assert ns.json is True
    assert ns.strict is True


def test_argparser_show_requires_positional():
    with pytest.raises(SystemExit):
        _parse(["upstream", "show"])


@_skip_if_no_v1_35b_up
def test_argparser_show_argument():
    ns = _parse(["upstream", "show", "a5000-2x-35b-prod"])
    assert ns.config == "a5000-2x-35b-prod"


# ─── live runs

def test_run_check_no_config(capsys):
    ns = _parse(["upstream", "check"])
    rc = run_check(ns)
    # On Mac dev (no vllm) → pin=None → not in allowlist
    # Without --strict, exit 0
    assert rc == 0
    out = capsys.readouterr().out
    assert "sndr upstream check" in out
    assert "Running pin:" in out


def test_run_check_strict_with_no_vllm_returns_1(capsys):
    """--strict + no vllm = pin missing from allowlist = exit 1."""
    ns = _parse(["upstream", "check", "--strict"])
    rc = run_check(ns)
    assert rc == 1


def test_run_check_unknown_config_returns_2(capsys):
    ns = _parse(["upstream", "check", "--config", "nonexistent-xyz"])
    rc = run_check(ns)
    assert rc == 2


def test_run_check_json_output_well_formed(capsys):
    ns = _parse(["upstream", "check", "--json"])
    rc = run_check(ns)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "running_pin" in data
    assert "in_known_good_allowlist" in data
    assert "known_good_count" in data
    assert data["known_good_count"] >= 1


@_skip_if_no_v1_35b_up
def test_run_check_with_config_json_includes_preset_keys(capsys):
    ns = _parse(["upstream", "check", "--config", "a5000-2x-35b-prod",
                 "--json"])
    rc = run_check(ns)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["preset"] == "a5000-2x-35b-prod"
    assert "preset_violation" in data


def test_run_show_unknown_config_returns_2(capsys):
    ns = _parse(["upstream", "show", "nonexistent-xyz"])
    rc = run_show(ns)
    assert rc == 2


@_skip_if_no_v1_35b_up
def test_run_show_35b_prod_has_y11_block(capsys):
    """35B PROD declares an upstream block with required_pin."""
    ns = _parse(["upstream", "show", "a5000-2x-35b-prod"])
    rc = run_show(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "required_pin:" in out
    assert "0.20.2rc1.dev93" in out


@_skip_if_no_v1_27b_up
def test_run_show_27b_no_y11_block_handles_cleanly(capsys):
    """27B PROD doesn't yet declare an upstream block — show that gracefully."""
    ns = _parse(["upstream", "show", "a5000-2x-27b-int4-tq-k8v4"])
    rc = run_show(ns)
    assert rc == 0
    out = capsys.readouterr().out
    # Either policy is None (graceful) OR a Y11 block was added later
    assert "policy" in out.lower() or "required_pin" in out.lower()


def test_run_list_human(capsys):
    ns = _parse(["upstream", "list"])
    rc = run_list(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "KNOWN_GOOD_VLLM_PINS" in out
    assert "0.20.2rc1.dev93" in out


def test_run_list_json(capsys):
    ns = _parse(["upstream", "list", "--json"])
    rc = run_list(ns)
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert "known_good_vllm_pins" in data
    assert len(data["known_good_vllm_pins"]) >= 1


def test_top_level_dispatches_upstream():
    from sndr.cli.legacy import cli_main
    rc = cli_main(["upstream", "list", "--json"])
    assert rc == 0

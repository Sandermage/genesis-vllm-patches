# SPDX-License-Identifier: Apache-2.0
"""C2 (UNIFIED_CONFIG plan 2026-05-09) — sndr deps CLI smoke tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from sndr.cli.legacy.deps import add_argparser, run_check, run_plan


# Phase 10 (2026-06-01): V1 sunset — `sndr deps` resolves --config via
# V1 registry only. Mark V1-bound tests for skip when V1 file retires;
# non-config-bound smoke tests (host inventory, JSON output) keep running.
_V1_DEPS_YAML = (Path(__file__).resolve().parents[3] / "vllm" / "sndr_core"
                 / "model_configs" / "builtin" / "a5000-2x-35b-prod.yaml")
_skip_if_no_v1_deps = pytest.mark.skipif(
    not _V1_DEPS_YAML.is_file(),
    reason="V1 fixture a5000-2x-35b-prod.yaml retired (Phase 10 sunset)",
)


def _parse(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    add_argparser(sub)
    return parser.parse_args(args)


def test_argparser_check_subcommand():
    ns = _parse(["deps", "check"])
    assert ns.deps_cmd == "check"
    assert ns.config is None
    assert ns.json is False


@_skip_if_no_v1_deps
def test_argparser_check_with_config_and_json():
    ns = _parse(["deps", "check", "--config", "a5000-2x-35b-prod", "--json"])
    assert ns.config == "a5000-2x-35b-prod"
    assert ns.json is True


def test_argparser_plan_requires_config():
    """--config is required for `plan`."""
    with pytest.raises(SystemExit):
        _parse(["deps", "plan"])


@_skip_if_no_v1_deps
def test_argparser_plan_with_strict():
    ns = _parse(["deps", "plan", "--config", "a5000-2x-35b-prod", "--strict"])
    assert ns.config == "a5000-2x-35b-prod"
    assert ns.strict is True


# ─── Live runs (depend on builtin registry but no host install)

def test_run_check_no_config_returns_zero(capsys):
    ns = _parse(["deps", "check"])
    rc = run_check(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Host inventory" in out
    assert "Python:" in out


def test_run_check_with_unknown_config_returns_2(capsys):
    ns = _parse(["deps", "check", "--config", "nonexistent-key-xyz"])
    rc = run_check(ns)
    assert rc == 2


@_skip_if_no_v1_deps
def test_run_check_with_known_config_emits_plan(capsys):
    """On a Mac with no docker/nvidia, the 35B config plan must surface
    blockers — and run_check returns 1.
    """
    ns = _parse(["deps", "check", "--config", "a5000-2x-35b-prod"])
    rc = run_check(ns)
    out = capsys.readouterr().out
    assert "Host inventory" in out
    # Plan section must show
    assert "Plan for config 'a5000-2x-35b-prod'" in out
    # On dev Mac (no docker, no NVIDIA), there must be blockers.
    # On the bench rig (full stack present), this returns 0. Either is fine.
    assert rc in (0, 1)


def test_run_check_json_output_is_valid(capsys):
    ns = _parse(["deps", "check", "--json"])
    rc = run_check(ns)
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "inventory" in parsed
    assert "os" in parsed["inventory"]


@_skip_if_no_v1_deps
def test_run_check_json_with_config_includes_plan(capsys):
    ns = _parse(["deps", "check", "--config", "a5000-2x-35b-prod", "--json"])
    rc = run_check(ns)
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "inventory" in parsed
    assert "plan" in parsed
    assert "items" in parsed["plan"]


def test_run_plan_unknown_config_returns_2(capsys):
    ns = _parse(["deps", "plan", "--config", "nonexistent-key-xyz"])
    rc = run_plan(ns)
    assert rc == 2


@_skip_if_no_v1_deps
def test_run_plan_strict_returns_1_on_blockers(capsys):
    """--strict flips a not-ready plan to exit 1; otherwise exit 0."""
    ns_strict = _parse(["deps", "plan", "--config", "a5000-2x-35b-prod",
                        "--strict"])
    rc_strict = run_plan(ns_strict)
    # On dev Mac: blockers → strict → 1.  On full server rig: 0.
    assert rc_strict in (0, 1)

    ns_nonstrict = _parse(["deps", "plan", "--config", "a5000-2x-35b-prod"])
    rc = run_plan(ns_nonstrict)
    assert rc == 0


@_skip_if_no_v1_deps
def test_run_check_writes_reports_to_dest(tmp_path, capsys):
    ns = _parse([
        "deps", "check",
        "--config", "a5000-2x-35b-prod",
        "--write-report",
        "--report-dir", str(tmp_path),
    ])
    run_check(ns)
    files = sorted(tmp_path.glob("*"))
    # Two artifacts each for inventory + plan = 4 files
    assert len(files) == 4
    assert any(f.name.startswith("inventory-") and f.suffix == ".json"
               for f in files)
    assert any(f.name.startswith("inventory-") and f.suffix == ".md"
               for f in files)
    assert any(f.name.startswith("deps-plan-") and f.suffix == ".json"
               for f in files)


def test_top_level_cli_dispatches_deps():
    """`python -m sndr.cli.legacy deps check --json` end-to-end."""
    from sndr.cli.legacy import cli_main
    rc = cli_main(["deps", "check", "--json"])
    assert rc == 0

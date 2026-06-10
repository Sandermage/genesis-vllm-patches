# SPDX-License-Identifier: Apache-2.0
"""C5 (UNIFIED_CONFIG plan 2026-05-09) — `sndr install --prepare` tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest


# Phase 10 (2026-06-01): V1 sunset — `sndr install --prepare` resolves
# --config via V1 registry. Mark V1-bound tests for skip when V1 file
# retires; argparse-shape + unknown/no-config tests keep running.
_V1_DIR_PREP = (Path(__file__).resolve().parents[2] / "vllm" / "sndr_core"
                / "model_configs" / "builtin")
_skip_if_no_v1_35b_prep = pytest.mark.skipif(
    not (_V1_DIR_PREP / "a5000-2x-35b-prod.yaml").is_file(),
    reason="V1 fixture a5000-2x-35b-prod.yaml retired (Phase 10 sunset)",
)
_skip_if_no_v1_27b_prep = pytest.mark.skipif(
    not (_V1_DIR_PREP / "a5000-2x-27b-int4-tq-k8v4.yaml").is_file(),
    reason="V1 fixture a5000-2x-27b-int4-tq-k8v4.yaml retired (Phase 10 sunset)",
)


def _make_opts(**kwargs) -> argparse.Namespace:
    defaults = dict(
        dry_run=True,
        non_interactive=True,
        workload="balanced",
        pin="stable",
        repo="https://github.com/Sandermage/genesis-vllm-patches.git",
        home=None,
        bare_metal=False,
        no_plugin=True,
        no_verify=True,
        system=False,
        uninstall=False,
        pro=False,
        license_key="",
        config=None,
        prepare=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_prepare_without_config_returns_2():
    """--prepare without --config returns exit 2 with friendly error."""
    from sndr.cli.legacy.install import run_install
    opts = _make_opts(prepare=True, config=None)
    rc = run_install(opts)
    assert rc == 2


def test_prepare_with_unknown_config_returns_2():
    from sndr.cli.legacy.install import run_install
    opts = _make_opts(prepare=True, config="nonexistent-key-xyz")
    rc = run_install(opts)
    assert rc == 2


@_skip_if_no_v1_35b_prep
def test_prepare_with_known_config_dry_run_succeeds():
    """--prepare --config <known-key> --dry-run completes cleanly."""
    from sndr.cli.legacy.install import run_install
    opts = _make_opts(prepare=True, config="a5000-2x-35b-prod")
    rc = run_install(opts)
    assert rc == 0


@_skip_if_no_v1_35b_prep
def test_prepare_pulls_workload_from_config():
    """When --workload is unset but --config has a workload_tag, prepare uses it."""
    from sndr.cli.legacy.install import run_install_prepare
    opts = _make_opts(prepare=True, config="a5000-2x-35b-prod", workload=None)
    # The function mutates opts to set workload from cfg
    rc = run_install_prepare(opts, "a5000-2x-35b-prod")
    assert rc == 0
    # 35B PROD has workload_tag=balanced
    assert opts.workload == "balanced"


@_skip_if_no_v1_35b_prep
def test_prepare_pulls_pin_from_config():
    """When --pin is 'stable' but --config has a vllm_pin_required,
    prepare overrides it to the exact pin."""
    from sndr.cli.legacy.install import run_install_prepare
    opts = _make_opts(prepare=True, config="a5000-2x-35b-prod")
    rc = run_install_prepare(opts, "a5000-2x-35b-prod")
    assert rc == 0
    # 35B PROD vllm_pin_required tracks the current PROD pin.
    # Reads from YAML so this stays accurate across pin bumps.
    from sndr.model_configs.registry import get as get_config
    expected_pin = get_config("a5000-2x-35b-prod").vllm_pin_required
    assert expected_pin in opts.pin


@_skip_if_no_v1_35b_prep
def test_argparser_registers_config_and_prepare_flags():
    from sndr.cli.legacy.install import add_argparser
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    add_argparser(sub)
    opts = p.parse_args([
        "install", "-y",
        "--config", "a5000-2x-35b-prod",
        "--prepare",
        "--dry-run", "--no-plugin", "--no-verify",
    ])
    assert opts.config == "a5000-2x-35b-prod"
    assert opts.prepare is True


@_skip_if_no_v1_27b_prep
def test_prepare_27b_works():
    from sndr.cli.legacy.install import run_install
    opts = _make_opts(prepare=True, config="a5000-2x-27b-int4-tq-k8v4")
    rc = run_install(opts)
    assert rc == 0

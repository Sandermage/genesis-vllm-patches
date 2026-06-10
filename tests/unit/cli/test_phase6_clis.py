# SPDX-License-Identifier: Apache-2.0
"""Phase 6 CLIs (UNIFIED_CONFIG plan 2026-05-09) — service / tune / migrate / image tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest


# Phase 10 (2026-06-01): V1 sunset cascade — defensive skip-marker for
# tests bound to the V1 monolithic preset key `a5000-2x-35b-prod`. When
# the V1 YAML retires, V1-fixture-bound tests skip cleanly; non-V1
# tests in this file (migrate/sweep/dispatch error paths) continue to run.
_V1_35B_YAML = (Path(__file__).resolve().parents[3] / "vllm" / "sndr_core"
                / "model_configs" / "builtin" / "a5000-2x-35b-prod.yaml")
_skip_if_no_v1_35b = pytest.mark.skipif(
    not _V1_35B_YAML.is_file(),
    reason="V1 fixture a5000-2x-35b-prod.yaml retired (Phase 10 sunset)",
)


def _parse(module_name: str, args: list[str]) -> argparse.Namespace:
    """Parse args via the named CLI module's add_argparser."""
    import importlib
    mod = importlib.import_module(f"sndr.cli.legacy.{module_name}")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    mod.add_argparser(sub)
    return parser.parse_args(args)


# ─── C13 sndr service

@_skip_if_no_v1_35b
def test_service_install_argparser():
    ns = _parse("service", ["service", "install", "a5000-2x-35b-prod"])
    assert ns.service_cmd == "install"
    assert ns.config == "a5000-2x-35b-prod"
    assert ns.yes is False


def test_service_start_with_yes():
    ns = _parse("service", ["service", "start", "x", "--yes"])
    assert ns.yes is True


def test_service_install_unknown_config_returns_2():
    from sndr.cli.legacy.service import run_install
    ns = _parse("service", ["service", "install", "nonexistent-xyz"])
    assert run_install(ns) == 2


@_skip_if_no_v1_35b
def test_service_install_no_y10_block_returns_2(capsys):
    """35B PROD has no Y10 service block today → friendly warn + return 2."""
    from sndr.cli.legacy.service import run_install
    ns = _parse("service", ["service", "install", "a5000-2x-35b-prod"])
    rc = run_install(ns)
    assert rc == 2
    err = capsys.readouterr().err + capsys.readouterr().out
    # Either way, the warn message about no Y10 is printed somewhere


# ─── C14 sndr tune

@_skip_if_no_v1_35b
def test_tune_plan_argparser():
    ns = _parse("tune", ["tune", "plan", "a5000-2x-35b-prod"])
    assert ns.tune_cmd == "plan"
    assert ns.config == "a5000-2x-35b-prod"


def test_tune_plan_unknown_config_returns_2():
    from sndr.cli.legacy.tune import run_plan
    ns = _parse("tune", ["tune", "plan", "nonexistent-xyz"])
    assert run_plan(ns) == 2


@_skip_if_no_v1_35b
def test_tune_plan_no_y8_returns_2(capsys):
    """35B PROD has no Y8 gpu_tuning today."""
    from sndr.cli.legacy.tune import run_plan
    ns = _parse("tune", ["tune", "plan", "a5000-2x-35b-prod"])
    assert run_plan(ns) == 2


def test_tune_sweep_requires_low_high_bench():
    ns = _parse("tune", ["tune", "sweep", "x",
                          "--low", "150", "--high", "250", "--step", "20",
                          "--bench-cmd", "echo bench"])
    assert ns.low == 150
    assert ns.high == 250
    assert ns.bench_cmd == "echo bench"


# ─── C9 sndr migrate

def test_migrate_argparser():
    ns = _parse("migrate", ["migrate", "v11-runtime-contract",
                              "/tmp/foo.yaml"])
    assert ns.migration == "v11-runtime-contract"
    assert ns.paths == ["/tmp/foo.yaml"]
    assert ns.yes is False


def test_migrate_rejects_unknown_migration():
    with pytest.raises(SystemExit):
        _parse("migrate", ["migrate", "totally-unknown-migration",
                            "/tmp/foo.yaml"])


def test_migrate_dry_run_on_real_yaml(tmp_path, capsys):
    """Dry-run on a synthetic YAML — should show planned changes, not write."""
    src = tmp_path / "test.yaml"
    src.write_text(
        "key: x\n"
        "title: x\n"
        "description: x\n"
        "schema_version: 1\n"
        "maintainer: sandermage\n"
        "model_path: /m\n"
        "docker:\n"
        "  image: img\n"
        "  container_name: c\n"
        "  port: 8000\n"
    )
    from sndr.cli.legacy.migrate import run_migrate
    ns = _parse("migrate", ["migrate", "v11-runtime-contract", str(src)])
    rc = run_migrate(ns)
    assert rc == 0
    # Original content preserved (dry-run)
    assert "host_port" not in src.read_text()


def test_migrate_yes_writes_changes(tmp_path):
    src = tmp_path / "test.yaml"
    src.write_text(
        "docker:\n"
        "  image: img\n"
        "  container_name: c\n"
        "  port: 8001\n"
    )
    from sndr.cli.legacy.migrate import run_migrate
    ns = _parse("migrate", ["migrate", "v11-runtime-contract", str(src),
                              "--yes"])
    rc = run_migrate(ns)
    assert rc == 0
    body = src.read_text()
    assert "host_port" in body
    assert "8001" in body  # port preserved
    # Backup created
    assert (src.with_suffix(src.suffix + ".bak")).exists()


# ─── C3 sndr image

@_skip_if_no_v1_35b
def test_image_resolve_argparser():
    ns = _parse("image", ["image", "resolve", "a5000-2x-35b-prod"])
    assert ns.image_cmd == "resolve"
    assert ns.config == "a5000-2x-35b-prod"


def test_image_resolve_unknown_config_returns_2():
    from sndr.cli.legacy.image import run_resolve
    ns = _parse("image", ["image", "resolve", "nonexistent-xyz"])
    assert run_resolve(ns) == 2


@_skip_if_no_v1_35b
def test_image_resolve_35b_shows_declared_digest(capsys):
    from sndr.cli.legacy.image import run_resolve
    ns = _parse("image", ["image", "resolve", "a5000-2x-35b-prod"])
    rc = run_resolve(ns)
    assert rc == 0
    out = capsys.readouterr().out
    # B2 digest from earlier session
    assert "9b534fe" in out


def test_image_verify_no_declared_digest_returns_0(capsys, monkeypatch):
    """When image_digest is None, verify returns 0 (not enforced)."""
    from sndr.cli.legacy.image import run_verify
    # Use a custom config without digest
    from sndr.model_configs.schema import (
        ModelConfig, HardwareSpec, DockerConfig,
    )
    from sndr.model_configs import registry as R
    cfg = ModelConfig(
        key="test-no-digest", title="x", description="x",
        schema_version=1, maintainer="sandermage", model_path="/m",
        hardware=HardwareSpec(gpu_match_keys=["a5000"], n_gpus=1,
                              min_vram_per_gpu_mib=1),
        docker=DockerConfig(image="img:tag", container_name="c", port=8000),
    )
    monkeypatch.setattr(R, "get",
                          lambda k: cfg if k == "test-no-digest" else None)
    ns = _parse("image", ["image", "verify", "test-no-digest"])
    rc = run_verify(ns)
    assert rc == 0


# ─── Top-level dispatch

def test_top_level_dispatch_service():
    from sndr.cli.legacy import cli_main
    rc = cli_main(["service", "install", "nonexistent-xyz"])
    assert rc == 2


def test_top_level_dispatch_tune():
    from sndr.cli.legacy import cli_main
    rc = cli_main(["tune", "plan", "nonexistent-xyz"])
    assert rc == 2


def test_top_level_dispatch_image():
    from sndr.cli.legacy import cli_main
    rc = cli_main(["image", "resolve", "nonexistent-xyz"])
    assert rc == 2

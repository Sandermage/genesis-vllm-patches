# SPDX-License-Identifier: Apache-2.0
"""S-05 (2026-05-08) — `sndr install` canonical wizard tests.

Replaces the Stage-11 8-step smoke tests after the install.sh →
install.py consolidation. The new wizard has 11 steps + uninstall:

  step_preflight, step_detect_hardware, step_detect_vllm,
  step_runtime_caveat, step_pick_workload, step_resolve_pin,
  step_clone_or_update, step_install_plugin, step_detect_host_paths,
  step_generate_launch, step_smoke_test

Live install ops (git clone, pip install, exec vllm serve) are skipped
under --dry-run. These tests exercise the wizard in dry-run only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest


# Phase 10 (2026-06-01): V1 sunset — the B3 regression test asserts that
# auto-match returns a key containing "a5000", which is V1-naming-bound.
# V2 composed-triplet keys may still contain "a5000" via the hardware
# layer; the explicit V1-key assertion stays V1-specific.
_V1_DIR_INS = (Path(__file__).resolve().parents[2] / "vllm" / "sndr_core"
               / "model_configs" / "builtin")
_skip_if_no_v1_35b_ins = pytest.mark.skipif(
    not (_V1_DIR_INS / "a5000-2x-35b-prod.yaml").is_file(),
    reason="V1 fixture a5000-2x-35b-prod.yaml retired (Phase 10 sunset)",
)


def _make_opts(**kwargs) -> argparse.Namespace:
    """argparse.Namespace mimicking `sndr install` flags after S-05 rewrite."""
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
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ─── Step-level coverage ──────────────────────────────────────────────────


def test_step_preflight_collects_python_and_disk():
    """Pre-flight reports OS/Python/git/disk info."""
    from sndr.cli.legacy.install import step_preflight
    res = step_preflight(_make_opts())
    assert "os" in res.data
    assert "python_version" in res.data
    assert "sndr_home" in res.data
    major, minor = res.data["python_version"].split(".")[:2]
    assert int(major) >= 3 and int(minor) >= 10


def test_step_detect_hardware_reports_gpu_count():
    """Hardware detection runs without crashing on CPU-only hosts."""
    from sndr.cli.legacy.install import step_detect_hardware
    res = step_detect_hardware(_make_opts())
    assert "n_gpus" in res.data
    assert res.data["n_gpus"] >= 0
    assert "gpu_class_hint" in res.data
    assert isinstance(res.data["gpu_class_hint"], str)


def test_step_detect_vllm_returns_installed_flag():
    """vllm detection sets `installed` boolean either way."""
    from sndr.cli.legacy.install import step_detect_vllm
    res = step_detect_vllm(_make_opts())
    assert "installed" in res.data
    assert isinstance(res.data["installed"], bool)


def test_step_runtime_caveat_returns_proxmox_flag():
    """Runtime caveat probe sets `proxmox_detected` boolean."""
    from sndr.cli.legacy.install import step_runtime_caveat
    res = step_runtime_caveat(_make_opts())
    assert "proxmox_detected" in res.data
    assert isinstance(res.data["proxmox_detected"], bool)


def test_step_pick_workload_explicit_flag_wins():
    """If `--workload tool_agent` is passed, it wins without prompting."""
    from sndr.cli.legacy.install import step_pick_workload
    res = step_pick_workload(_make_opts(workload="tool_agent"))
    assert res.data["workload"] == "tool_agent"


def test_step_pick_workload_invalid_value_fatal():
    """Invalid `--workload` aborts with SystemExit."""
    from sndr.cli.legacy.install import step_pick_workload
    with pytest.raises(SystemExit):
        step_pick_workload(_make_opts(workload="invalid_workload_xyz"))


def test_step_pick_workload_default_in_non_interactive():
    """Non-interactive without `--workload` defaults to balanced."""
    from sndr.cli.legacy.install import step_pick_workload
    res = step_pick_workload(_make_opts(workload=None))
    assert res.data["workload"] == "balanced"


def test_step_resolve_pin_dev_passes_through():
    """`--pin dev` resolves to literal `dev`."""
    from sndr.cli.legacy.install import step_resolve_pin
    res = step_resolve_pin(_make_opts(pin="dev"))
    assert res.data["pin"] == "dev"
    assert res.data["kind"] == "dev"


def test_step_resolve_pin_explicit_ref():
    """`--pin v7.69` returns the literal ref."""
    from sndr.cli.legacy.install import step_resolve_pin
    res = step_resolve_pin(_make_opts(pin="v7.69"))
    assert res.data["pin"] == "v7.69"
    assert res.data["kind"] == "explicit"


def test_step_resolve_pin_stable_handles_offline(monkeypatch):
    """`--pin stable` falls back to `main` when GitHub API is unreachable."""
    import sndr.cli.legacy.install as M
    monkeypatch.setattr(M, "_resolve_latest_tag", lambda: None)
    res = M.step_resolve_pin(_make_opts(pin="stable"))
    assert res.data["pin"] == "main"
    assert res.data["kind"] == "stable_fallback"


def test_step_clone_or_update_dry_run_returns_marker():
    """Dry-run clone returns `<dry-run>` head."""
    from sndr.cli.legacy.install import step_clone_or_update
    preflight = {"sndr_home": "/tmp/sndr-test-bogus"}
    pin = {"pin": "main"}
    res = step_clone_or_update(_make_opts(), preflight, pin)
    assert res.data["head"] == "<dry-run>"


def test_step_install_plugin_dry_run_skipped():
    """Dry-run plugin install reports skipped."""
    from sndr.cli.legacy.install import step_install_plugin
    res = step_install_plugin(_make_opts(), {"home": "/nonexistent"})
    assert res.data["installed"] is False
    assert res.data["reason"] in ("--no-plugin", "dry-run")


def test_step_smoke_test_skipped_with_no_verify():
    """`--no-verify` skips the smoke test."""
    from sndr.cli.legacy.install import step_smoke_test
    res = step_smoke_test(_make_opts(no_verify=True))
    assert res.data["ran"] is False


# ─── Top-level orchestrator ───────────────────────────────────────────────


def test_run_install_dry_run_full_flow():
    """End-to-end dry-run completes successfully on Mac dev (no GPU)."""
    from sndr.cli.legacy.install import run_install
    opts = _make_opts(workload="tool_agent")
    rc = run_install(opts)
    assert rc == 0


def test_run_install_uninstall_dispatches():
    """`--uninstall` flag dispatches to run_uninstall and returns 0."""
    from sndr.cli.legacy.install import run_install
    opts = _make_opts(uninstall=True)
    rc = run_install(opts)
    assert rc == 0


def test_argparser_registers_install_subcommand():
    """`add_argparser` wires the install subcommand correctly."""
    from sndr.cli.legacy.install import add_argparser
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    add_argparser(sub)
    opts = p.parse_args(["install", "-y", "--workload", "balanced",
                         "--dry-run", "--no-plugin", "--no-verify"])
    assert opts.workload == "balanced"
    assert opts.dry_run is True


# ─── B3 fix (UNIFIED_CONFIG plan 2026-05-09): _match_preset uses gpu_match_keys

def test_gpu_keys_match_substring_either_direction():
    """Helper matches needle⊂key OR key⊂needle, lowercased."""
    from sndr.cli.legacy.install import _gpu_keys_match
    # Detected longer than config key (typical nvidia-smi output)
    assert _gpu_keys_match("nvidia rtx a5000", ["rtx a5000"])
    # Detected shorter than config key (rare)
    assert _gpu_keys_match("rtx pro 6000", ["rtx pro 6000 blackwell"])
    # Multiple keys — any match wins
    assert _gpu_keys_match("a100", ["rtx a5000", "a100", "h100"])


def test_gpu_keys_match_no_match():
    """Different GPU class returns False."""
    from sndr.cli.legacy.install import _gpu_keys_match
    assert not _gpu_keys_match("rtx 4090", ["rtx a5000", "a100"])
    assert not _gpu_keys_match("", ["rtx a5000"])
    assert not _gpu_keys_match("rtx a5000", [])
    assert not _gpu_keys_match("rtx a5000", [""])


@_skip_if_no_v1_35b_ins
def test_match_preset_finds_a5000_2x_35b_prod():
    """B3 regression: 'NVIDIA RTX A5000' + 2 GPUs must match
    a5000-2x-35b-prod (which has gpu_match_keys=['rtx a5000'])."""
    from sndr.cli.legacy.install import _match_preset
    cfg, key = _match_preset("NVIDIA RTX A5000", 2, "balanced")
    assert cfg is not None, "preset match must not be None — schema field is gpu_match_keys"
    assert key is not None
    # At least one of the 2× A5000 builtin configs should win on n_gpus=2 + balanced
    assert "a5000" in key.lower()


def test_match_preset_returns_none_for_unknown_gpu():
    """Unknown GPU returns (None, None) cleanly (no crash on any builtin config)."""
    from sndr.cli.legacy.install import _match_preset
    cfg, key = _match_preset("Some Unknown GPU XYZ", 2, "balanced")
    assert cfg is None
    assert key is None

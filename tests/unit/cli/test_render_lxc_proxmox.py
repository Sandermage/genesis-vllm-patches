# SPDX-License-Identifier: Apache-2.0
"""`sndr model-config render --runtime lxc_proxmox` — runnable artifact.

Audit C4 closure (2026-05-16): the lxc_proxmox renderer used to emit a
manual-guide skeleton with commented-out `pct create` lines that operators
had to copy-paste into their shell. After this change the renderer emits
a single runnable bash script that drives the full Proxmox VE lifecycle
(create CT → wire GPU passthrough → start → bootstrap venv → write
launch.sh). These tests pin the runnable-artifact contract so the
surface cannot silently regress back to a skeleton.
"""
from __future__ import annotations

import subprocess
import tempfile
import warnings
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cfg():
    """Pick the first builtin config that has a docker block — the
    lxc_proxmox renderer requires one. We accept any V1 deprecation
    warning since several builtin configs are migrating to V2."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sndr.model_configs import get, list_keys
    for key in list_keys():
        c = get(key)
        if c is not None and c.docker is not None:
            return c
    pytest.skip("no builtin config with a docker block — cannot test")


def test_renders_runnable_bash_script(cfg):
    """Output must start with a bash shebang and parse via `bash -n`."""
    from sndr.compat.model_config_cli import _render_lxc_proxmox
    out = _render_lxc_proxmox(cfg)
    assert out.startswith("#!/usr/bin/env bash\n"), (
        "missing bash shebang — generated artifact must be directly executable"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
        f.write(out)
        path = Path(f.name)
    try:
        proc = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True, text=True,
        )
    finally:
        path.unlink(missing_ok=True)
    assert proc.returncode == 0, (
        f"bash -n rejected the generated script:\n{proc.stderr}"
    )


def test_includes_real_pct_lifecycle_commands(cfg):
    """The runnable renderer must invoke pct create / pct start /
    pct exec (not just emit them as comments like the old skeleton)."""
    from sndr.compat.model_config_cli import _render_lxc_proxmox
    out = _render_lxc_proxmox(cfg)
    # Look for executable lines (no leading "#" on the pct invocation).
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    assert any(ln.startswith("pct create ") for ln in lines), (
        "pct create is commented-out or missing — render is still skeleton"
    )
    assert any(ln.startswith("pct start ") for ln in lines), (
        "pct start is commented-out or missing"
    )
    assert any("pct exec " in ln for ln in lines), (
        "pct exec is missing — venv bootstrap step was dropped"
    )


def test_no_skeleton_markers(cfg):
    """The string 'SKELETON' / 'NOT a runnable artifact' / 'manual-guide'
    must not appear — those were the old skeleton's red flags."""
    from sndr.compat.model_config_cli import _render_lxc_proxmox
    out = _render_lxc_proxmox(cfg)
    forbidden = (
        "SKELETON",
        "NOT a runnable artifact",
        "manual-guide",
        "(DO NOT execute as-is",
    )
    for marker in forbidden:
        assert marker not in out, (
            f"forbidden skeleton marker {marker!r} present in renderer output"
        )


def test_overridable_env_vars_present(cfg):
    """Operator-overridable knobs (CTID, storage, bridge) must surface as
    env vars at the top so re-deploying on a different cluster topology
    doesn't require editing the script body."""
    from sndr.compat.model_config_cli import _render_lxc_proxmox
    out = _render_lxc_proxmox(cfg)
    required_overrides = (
        'SNDR_CTID="${SNDR_CTID:-',
        'SNDR_STORAGE="${SNDR_STORAGE:-',
        'SNDR_BRIDGE="${SNDR_BRIDGE:-',
        'SNDR_MEMORY_MIB="${SNDR_MEMORY_MIB:-',
        'SNDR_N_GPUS="${SNDR_N_GPUS:-',
    )
    for marker in required_overrides:
        assert marker in out, (
            f"missing operator-overridable env var: {marker}"
        )


def test_emits_idempotent_guard(cfg):
    """Re-running must reuse an existing CTID, not error out — check that
    the renderer wraps pct create with an existence guard."""
    from sndr.compat.model_config_cli import _render_lxc_proxmox
    out = _render_lxc_proxmox(cfg)
    assert 'pct status "${SNDR_CTID}"' in out, (
        "missing `pct status` idempotency guard around `pct create`"
    )


def test_emits_gpu_passthrough_wiring(cfg):
    """The script must inject the lxc.cgroup2.devices.allow + bind-mount
    entries into /etc/pve/lxc/<CTID>.conf for NVIDIA GPU passthrough."""
    from sndr.compat.model_config_cli import _render_lxc_proxmox
    out = _render_lxc_proxmox(cfg)
    assert "lxc.cgroup2.devices.allow" in out
    assert "lxc.mount.entry: /dev/nvidia" in out
    # And it must be applied idempotently via a marker grep.
    assert "GENESIS_MARKER" in out


def test_inner_launch_script_runs_vllm_serve(cfg):
    """The inline launch.sh that gets written into the container must
    actually invoke `vllm serve` with the captured flags."""
    from sndr.compat.model_config_cli import _render_lxc_proxmox
    out = _render_lxc_proxmox(cfg)
    assert "GENESIS_LAUNCH_EOF" in out, (
        "heredoc that writes the inner launch.sh is missing"
    )
    assert "exec vllm serve" in out, (
        "inner launch.sh does not exec `vllm serve` — would not boot"
    )
    # The captured ModelConfig's model path must propagate.
    assert cfg.model_path in out


def test_legacy_skeleton_alias_delegates(cfg):
    """The old `_render_lxc_proxmox_skeleton` symbol survives as a thin
    deprecation alias so external scripts that imported it directly do
    not break for one release. Output must match the new renderer."""
    from sndr.compat.model_config_cli import (
        _render_lxc_proxmox, _render_lxc_proxmox_skeleton,
    )
    assert _render_lxc_proxmox_skeleton(cfg) == _render_lxc_proxmox(cfg)


def test_render_without_docker_block_returns_clean_error():
    """Bare-metal-only configs must not crash; the renderer should
    return an operator-readable error string."""
    from sndr.compat.model_config_cli import _render_lxc_proxmox
    from sndr.model_configs.schema import HardwareSpec, ModelConfig
    cfg = ModelConfig(
        key="bare-only",
        title="bare-metal-only",
        description="no docker",
        schema_version=1,
        maintainer="test",
        model_path="/models/x",
        hardware=HardwareSpec(
            gpu_match_keys=["a5000"], n_gpus=1, min_vram_per_gpu_mib=24000,
        ),
        docker=None,
    )
    out = _render_lxc_proxmox(cfg)
    assert "ERROR" in out
    assert "docker" in out.lower()

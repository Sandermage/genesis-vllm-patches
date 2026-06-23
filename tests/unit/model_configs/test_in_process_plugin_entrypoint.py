# SPDX-License-Identifier: Apache-2.0
"""Regression: the in-process vLLM plugin must actually load in vllm serve.

UNIFIED ROOT BUG (diagnosed live on the rig 2026-06-22)
───────────────────────────────────────────────────────
The boot renders `python3 -m sndr.apply ; exec vllm serve` as TWO
processes. `sndr.apply` (subprocess) text-patches vLLM source files on
disk (those PERSIST), but RUNTIME monkey-patches (e.g. g4_85's
`method.apply = wrapper`) live only in that subprocess's memory and are
LOST when `exec vllm serve` starts a fresh process.

Runtime monkey-patches only persist if they run IN the vllm serve
process. vLLM does this via `vllm.plugins.load_general_plugins()`, which
calls every `vllm.general_plugins` setuptools entry-point at engine +
worker init. Our root `pyproject.toml` registers
    genesis_v7 = "sndr.plugin:register"
so when the `sndr` package is pip-installed WITH its entry-point metadata,
vllm serve auto-loads `sndr.plugin.register()` in-process and ALL runtime
monkey-patches (incl. g4_85) fire.

The bug had two parts, both asserted here:
  1. The render's dev-install branch pip-installed the EMPTY legacy
     `tools/genesis_vllm_plugin` subdir (whose pyproject registered the
     no-op `genesis_v7:register` shim) instead of the SNDR PACKAGE — so
     even when opted in, no Genesis runtime patch fired in the serving
     process.
  2. A bare bind-mount of `sndr/` makes sndr importable but registers NO
     entry-point (no dist-info), so `load_general_plugins()` never called
     register().

These tests prove a freshly-rendered launch WOULD let vllm serve load the
in-process plugin, and that both pyprojects register the canonical
`sndr.plugin:register` target.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import tomllib

from sndr.model_configs.emitters.docker_cmd import build_docker_cmd
from sndr.model_configs.emitters.vllm_cmd import build_vllm_cmd
from sndr.model_configs.schema import (
    DockerConfig,
    HardwareSpec,
    ModelConfig,
)
from sndr.plugin import register as _sndr_plugin_register

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ── pyproject entry-point contract ───────────────────────────────────────


def _entry_points(pyproject_path: Path) -> dict[str, str]:
    data = tomllib.loads(pyproject_path.read_text())
    return (
        data.get("project", {})
        .get("entry-points", {})
        .get("vllm.general_plugins", {})
    )


def test_root_pyproject_registers_sndr_plugin_register():
    """The ROOT pyproject must register the canonical in-process plugin
    entry-point pointing at `sndr.plugin:register`. This is what vLLM's
    `load_general_plugins()` loads at engine + worker init so runtime
    monkey-patches re-apply IN the serving process."""
    eps = _entry_points(_REPO_ROOT / "pyproject.toml")
    assert eps.get("genesis_v7") == "sndr.plugin:register", (
        "root pyproject must register genesis_v7 = 'sndr.plugin:register' "
        f"under [project.entry-points.\"vllm.general_plugins\"]; got {eps!r}"
    )


def test_legacy_plugin_subdir_pyproject_targets_canonical_register():
    """The legacy `tools/genesis_vllm_plugin` subdir pyproject must ALSO
    target `sndr.plugin:register` (not the empty `genesis_v7:register`
    shim). Installing EITHER the repo root OR the legacy subdir now
    registers the SAME in-process plugin — no divergent apply path."""
    legacy = _REPO_ROOT / "tools" / "genesis_vllm_plugin" / "pyproject.toml"
    eps = _entry_points(legacy)
    assert eps.get("genesis_v7") == "sndr.plugin:register", (
        "legacy subdir pyproject must target 'sndr.plugin:register'; "
        f"got {eps!r}. The empty 'genesis_v7:register' shim was the bug — "
        "it registered a plugin that applied no Genesis runtime patch."
    )


def test_sndr_plugin_register_is_importable_callable():
    """The entry-point target must actually resolve to a callable so vLLM
    can invoke it. A typo'd target silently no-ops at load time."""
    assert callable(_sndr_plugin_register)


# ── rendered docker bootstrap contract ───────────────────────────────────


def _make_cfg(mounts: list[str]) -> ModelConfig:
    return ModelConfig(
        key="ep-test",
        title="Entry-Point Test",
        description="d",
        schema_version=1,
        maintainer="x",
        model_path="/models/Test-7B",
        hardware=HardwareSpec(
            gpu_match_keys=["test"], n_gpus=2, min_vram_per_gpu_mib=24576
        ),
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        served_model_name="test-7b",
        docker=DockerConfig(
            image="vllm/vllm-openai:nightly",
            container_name="vllm-test",
            port=8000,
            mounts=mounts,
        ),
    )


def _bootstrap_body(cfg: ModelConfig) -> str:
    """Extract the bash `-c '...'` body, undoing the POSIX quote escaping."""
    script = build_docker_cmd(cfg, build_vllm_cmd(cfg), host_paths={})
    line = next(
        ln for ln in script.splitlines() if ln.strip().startswith("-c '")
    )
    return re.search(r"-c '(.*)'\s*$", line).group(1).replace("'\\''", "'")


def _dev_install_segment(body: str) -> str:
    """The body of the `if SNDR_DEV_INSTALL_PLUGIN=1; then ... fi` block."""
    assert "SNDR_DEV_INSTALL_PLUGIN" in body
    return body.split("SNDR_DEV_INSTALL_PLUGIN", 1)[1].split("fi", 1)[0]


def test_dev_install_installs_sndr_package_not_empty_subdir():
    """With a `/plugin` mount, the opt-in dev install must pip-install the
    SNDR PACKAGE (registers `sndr.plugin:register`), NOT the empty legacy
    `tools/genesis_vllm_plugin` subdir. The `-e <path>` install command
    must therefore target the copied repo root, never the subdir."""
    body = _bootstrap_body(
        _make_cfg(["/abs/models:/models:ro", "/abs/repo:/plugin:ro"])
    )
    seg = _dev_install_segment(body)
    # The install command targets the (writable copy of the) mounted repo.
    assert "-e /tmp/sndr_plugin_src" in seg, (
        "dev install must `pip install -e` the sndr repo root copy so its "
        f"entry-point registers; segment was: {seg!r}"
    )
    # It must NOT pip-install the empty legacy subdir.
    assert "-e /tmp/genesis_vllm_plugin" not in body, (
        "dev install must not target the empty tools/genesis_vllm_plugin "
        "subdir — that was the UNIFIED ROOT BUG (no runtime patch fired)."
    )


def test_dev_install_asserts_entrypoint_registered():
    """The dev install must VERIFY the `vllm.general_plugins` entry-point
    actually registered (`sndr.plugin` present), so a misconfigured
    `plugin_src` fails LOUDLY at boot instead of silently serving without
    the in-process plugin."""
    body = _bootstrap_body(
        _make_cfg(["/abs/models:/models:ro", "/abs/repo:/plugin:ro"])
    )
    seg = _dev_install_segment(body)
    assert "vllm.general_plugins" in seg
    assert "sndr.plugin" in seg
    # The check is a hard `assert ...` in the embedded python3 -c, so a
    # missing entry-point exits non-zero under `set -euo pipefail`.
    assert "assert any(" in seg


def test_text_patch_apply_step_unchanged():
    """No text-patch regression: the canonical `python3 -m sndr.apply`
    step and `exec vllm serve` must still be present and in order. The
    in-process plugin fix is ADDITIVE — it does not remove the disk-level
    text-patch apply path (which the plugin install gates in front of)."""
    body = _bootstrap_body(
        _make_cfg(["/abs/models:/models:ro", "/abs/repo:/plugin:ro"])
    )
    assert "python3 -m sndr.apply" in body
    assert "exec vllm serve" in body
    # Order: apply step runs before exec.
    assert body.index("python3 -m sndr.apply") < body.index("exec vllm serve")


def test_dev_install_is_opt_in_and_idempotent_guarded():
    """The dev install must stay behind the `SNDR_DEV_INSTALL_PLUGIN`
    gate (default off) so production (baked image) renders never run a
    boot-time pip install."""
    body = _bootstrap_body(
        _make_cfg(["/abs/models:/models:ro", "/abs/repo:/plugin:ro"])
    )
    # Gated on the env, defaulting to "0" (off).
    assert '"${SNDR_DEV_INSTALL_PLUGIN:-0}" = "1"' in body


def test_no_plugin_mount_renders_pure_apply_path():
    """Without a `/plugin` mount (PROD baked-image path), the render must
    NOT emit any dev-install block — just the canonical apply + exec. PROD
    relies on the entry-point being baked into the image."""
    body = _bootstrap_body(_make_cfg(["/abs/models:/models:ro"]))
    assert "SNDR_DEV_INSTALL_PLUGIN" not in body
    assert "python3 -m sndr.apply" in body
    assert "exec vllm serve" in body


def test_rendered_bootstrap_is_valid_bash():
    """The whole bootstrap (with the new entry-point assert) must parse as
    valid bash so it cannot break the boot."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    body = _bootstrap_body(
        _make_cfg(["/abs/models:/models:ro", "/abs/repo:/plugin:ro"])
    )
    r = subprocess.run(
        [bash, "-n", "-c", body], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, f"bash -n failed: {r.stderr}"

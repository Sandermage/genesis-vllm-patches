# SPDX-License-Identifier: Apache-2.0
"""``build_docker_cmd(cfg, vllm_parts, ...)`` — emit docker run cmd.

Pure function; takes a ``ModelConfig`` + pre-built vllm parts (from
:func:`.vllm_cmd.build_vllm_cmd`) + host paths + ``strict_mounts``
flag. Returns a single docker-run string ready for insertion into the
launch script.

Previously ``ModelConfig._build_docker_cmd`` in
``model_configs/schema.py``. Body unchanged.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..types import resolve_symbolic_mounts
from .shell import shell_quote

if TYPE_CHECKING:
    from ..schema import ModelConfig


def build_docker_cmd(
    cfg: "ModelConfig",
    vllm_parts: list[str],
    host_paths: Optional[dict[str, str]] = None,
    *,
    strict_mounts: bool = False,
) -> str:
    """Render docker run command embedding the vllm serve.

    Mounts containing ``${var}`` symbolic references are resolved
    through ``host_paths`` (or lazy-loaded ``host.yaml``) before being
    embedded in the docker ``-v`` flags. Mounts that are fully
    absolute paths pass through unchanged.

    Args:
        strict_mounts: when True, raises SchemaError on any
            unresolved ``${var}``. Set by ``render_launch_script`` for
            the live launch path (P0-8 audit 2026-05-08).
    """
    d = cfg.docker
    # Resolve symbolic mounts. Lazy-load host.yaml only if any mount
    # actually uses `${var}` — configs with fully absolute mounts
    # don't need a host config to render.
    #
    # Resolution order when host_paths is None:
    #   1. ~/.sndr/host.yaml (explicit operator config)
    #   2. host.detect_paths() (auto-probe common locations)
    #   3. unresolved → SchemaError with actionable message
    # Step 2 lets tests + dev machines render without setting up
    # host.yaml. detect_paths() probes _DEFAULT_*_CANDIDATES and
    # returns absolute paths for those that exist on this host.
    # Variables it can't find stay unresolved → SchemaError, which
    # is the correct outcome (operator must fix host.yaml).
    resolved_mounts = list(d.mounts)
    needs_resolution = any("${" in m for m in d.mounts)
    if needs_resolution:
        if host_paths is None:
            # Lazy import: host.py touches PyYAML, keep it off the
            # cold path for callers that pass host_paths explicitly.
            from ..host import load_host_config, detect_paths
            merged: dict[str, str] = {}
            try:
                merged.update(detect_paths())
            except Exception:
                pass
            try:
                merged.update(load_host_config().paths)
            except Exception:
                pass
            host_paths = merged
        # P0-8 (audit 2026-05-08): live launch passes strict_mounts=
        # True so unresolved vars raise SchemaError with a clear
        # "fix host.yaml" message. `--dry-run` paths use False to
        # preserve the preview-with-placeholders behavior.
        resolved_mounts = resolve_symbolic_mounts(
            d.mounts, host_paths, strict=strict_mounts,
        )

    lines = [
        f"docker rm -f {shell_quote(d.container_name)} 2>/dev/null || true",
        "",
        "docker run -d \\",
        f"  --name {shell_quote(d.container_name)} \\",
        "  --entrypoint /bin/bash \\",
        f"  --gpus {shell_quote(d.gpus)} \\",
        f"  --shm-size={shell_quote(d.shm_size)} \\",
    ]
    if d.memory_limit:
        lines.append(f"  --memory={shell_quote(d.memory_limit)} \\")
    if d.network:
        lines.append(f"  --network {shell_quote(d.network)} \\")
    # Y4: HOST:CONTAINER port mapping. Falls back to legacy
    # `port:port` when host_port/container_port are not split.
    lines.append(
        f"  -p {d.effective_host_port()}:{d.effective_container_port()} \\"
    )
    for m in resolved_mounts:
        lines.append(f"  -v {shell_quote(m)} \\")
    for f in d.extra_run_flags:
        lines.append(f"  {f} \\")
    # Env vars
    for k, v in sorted(cfg.system_env.items()):
        lines.append(f'  -e {k}={shell_quote(v)} \\')
    for k, v in sorted(cfg.genesis_env.items()):
        lines.append(f"  -e {k}={shell_quote(v)} \\")
    # Image + cmd
    lines.append(f"  {shell_quote(d.effective_image_ref())} \\")
    # Bash -c with canonical apply + exec vllm serve.
    # POSIX-escape single quotes inside the inner cmd so the outer
    # single-quoted -c '...' wrapper survives JSON args like
    # --speculative-config '{"method":"mtp",...}'.
    cmd = " ".join(vllm_parts)
    cmd_escaped = cmd.replace("'", "'\\''")
    # Build the bash bootstrap. If the operator mounts the genesis
    # plugin source at /plugin, install it in editable mode so its
    # `vllm.general_plugins` entry point auto-loads inside every
    # vllm worker process. Without this, patches only apply via the
    # explicit `apply` invocation — plugin-only paths (boot
    # banner, config detection) won't fire.
    has_plugin = any(
        ":/plugin" in m or m.endswith("/plugin")
        for m in d.mounts
    )
    # P0-8 (audit 2026-05-08): single canonical apply entrypoint.
    # The legacy apply-all fallback was a no-op
    # (module never existed in v10/v11) and silently masked any
    # apply failure as a successful sub-shell. Now the call is
    # direct — boot fails loudly if sndr_core is unimportable.
    apply_step = "python3 -m vllm.sndr_core.apply 2>&1 | tail -5"
    # P1-7 fix (audit 2026-05-08) + B6 (UNIFIED_CONFIG plan 2026-05-09):
    # runtime deps inside the container are pinned. Y1 introduced
    # `package_versions.python_packages` as the canonical source of
    # truth — when present it wins; otherwise the legacy hardcoded
    # baseline below is used. Operators can opt out via
    # `SNDR_DEV_INSTALL_RUNTIME_DEPS=1` for editable / dev workflows.
    runtime_deps = ""
    if cfg.package_versions is not None:
        runtime_deps = cfg.package_versions.to_pip_args()
    if not runtime_deps:
        runtime_deps = "pandas==2.2.3 scipy==1.14.1 xxhash==3.5.0"
    # DA-008 fix (audit 2026-05-08): production launch path NO LONGER
    # depends on the `/plugin` mount being present.
    #
    # Rationale: in production, `vllm-sndr-core` should already be
    # installed inside the container (via the wheel pip-installed at
    # image build time, or via a base image including it). Mounting
    # `/plugin` and pip-install'ing it at every container start is:
    #   - non-reproducible (whatever is in the operator's local repo wins);
    #   - slow (pip install adds ~10-30s to cold boot);
    #   - a supply-chain risk (operator's local edits become live).
    #
    # The new contract:
    #   - Production: the canonical apply step (`python3 -m
    #     vllm.sndr_core.apply`) is the ONLY thing run. If
    #     vllm-sndr-core isn't installed, the call fails loudly.
    #   - Dev: opt in to the legacy `/plugin` install via
    #     `SNDR_DEV_INSTALL_PLUGIN=1`. The original behavior is
    #     preserved verbatim under the env gate.
    #
    # `has_plugin` (presence of `/plugin` in mounts) used to
    # automatically TRIGGER the install. Now it just makes the dev
    # install POSSIBLE; the env flag must also be set.
    bootstrap_parts = ["set -euo pipefail"]
    # Optional dev-mode pinned runtime deps (P1-7).
    bootstrap_parts.append(
        'if [ "${SNDR_DEV_INSTALL_RUNTIME_DEPS:-0}" = "1" ]; then '
        f'pip install --quiet {runtime_deps} 2>&1 | tail -2; '
        'fi'
    )
    # Optional dev-mode plugin install (DA-008).
    if has_plugin:
        bootstrap_parts.append(
            'if [ "${SNDR_DEV_INSTALL_PLUGIN:-0}" = "1" ]; then '
            "cp -r /plugin /tmp/genesis_vllm_plugin && "
            "pip install --quiet --disable-pip-version-check "
            "--root-user-action=ignore --no-deps -e "
            "/tmp/genesis_vllm_plugin 2>&1 | tail -2; "
            'fi'
        )
    # Canonical apply step (always runs).
    bootstrap_parts.append(apply_step)
    bootstrap_parts.append(f"exec {cmd_escaped}")
    bootstrap = "; ".join(bootstrap_parts)
    lines.append(f"  -c '{bootstrap}'")
    return "\n".join(lines)

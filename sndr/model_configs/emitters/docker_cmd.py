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
    # Bash -c with canonical apply + exec vllm serve. The WHOLE bootstrap is
    # single-quote-escaped once at the end (see below) so the outer -c '...'
    # wrapper survives BOTH the JSON args (--speculative-config / --override-
    # generation-config '{...}') AND the plugin-assert's python string literals
    # ('vllm.general_plugins', 'sndr.plugin', ...). Pre-escaping only `cmd` left
    # the assert's quotes raw -> they terminated the outer -c '...' (bash syntax
    # error). 2026-06-23 fix.
    cmd = " ".join(vllm_parts)
    # ── TWO-PROCESS BOUNDARY (the whole reason this bootstrap is shaped ──
    # ── the way it is — diagnosed live on the rig 2026-06-22). ──────────
    #
    # The bootstrap renders as `python3 -m sndr.apply ; exec vllm serve`,
    # which is TWO processes:
    #
    #   1. `python3 -m sndr.apply` (subprocess) — applies the patch stack
    #      then exits. TEXT-patches (edits to vLLM source files on disk)
    #      PERSIST because the files stay changed. But RUNTIME monkey-
    #      patches (`SomeClass.method = wrapper`, e.g. g4_85) live ONLY in
    #      THAT subprocess's memory and are LOST the instant it exits.
    #
    #   2. `exec vllm serve` — replaces the shell with a brand-new Python
    #      process. It inherits the on-disk text-patches but NONE of the
    #      runtime monkey-patches from step 1.
    #
    # The ONLY supported way to re-apply runtime monkey-patches INSIDE the
    # serving process is vLLM's plugin system: `load_general_plugins()`
    # calls every `vllm.general_plugins` setuptools entry-point at engine +
    # worker init. Our root `pyproject.toml` registers
    #   genesis_v7 = "sndr.plugin:register"
    # so when the `sndr` package is pip-installed WITH its .dist-info /
    # .egg-info entry-point metadata, vllm serve auto-loads sndr.plugin and
    # ALL runtime monkey-patches (incl. g4_85) fire in-process.
    #
    # KEY: a bare bind-mount of sndr/ onto a site-packages dir makes sndr
    # IMPORTABLE but registers NO entry-point (no dist-info) — so the
    # in-process plugin never loads. The package must be pip-INSTALLED
    # (`pip install -e <sndr repo root>`) for the entry-point to exist.
    #
    # PROD path: bake the sndr wheel into the image at build time — the
    # entry-point is then already present and no boot-time install runs.
    # DEV path (below): mount the sndr REPO ROOT at /plugin and opt into
    # `SNDR_DEV_INSTALL_PLUGIN=1`, which pip-installs it editable so the
    # entry-point gets registered for this boot.
    has_plugin = any(
        ":/plugin" in m or m.endswith("/plugin")
        for m in d.mounts
    )
    # P0-8 (audit 2026-05-08): single canonical apply entrypoint.
    # The legacy apply-all fallback was a no-op
    # (module never existed in v10/v11) and silently masked any
    # apply failure as a successful sub-shell. Now the call is
    # direct — boot fails loudly if sndr_core is unimportable.
    # v12.0 retired the `vllm.sndr_core` shim — pyproject ships only `sndr*`.
    # Freshly-rendered launchers that drop the legacy mirror mount would fail
    # apply with ModuleNotFoundError on the old path, silently leaving Genesis
    # patches unapplied. The surviving entrypoint is `sndr/apply/__main__.py`.
    apply_step = "python3 -m sndr.apply 2>&1 | tail -5"
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
    # Rationale: in production, the `sndr` package should already be
    # installed inside the container (via the wheel pip-installed at
    # image build time, or via a base image including it). When baked,
    # its `vllm.general_plugins` entry-point is already on disk, so
    # vllm serve loads `sndr.plugin:register` IN-PROCESS with no boot-
    # time install — runtime monkey-patches fire automatically. Mounting
    # `/plugin` and pip-install'ing it at every container start is:
    #   - non-reproducible (whatever is in the operator's local repo wins);
    #   - slow (pip install adds ~10-30s to cold boot);
    #   - a supply-chain risk (operator's local edits become live).
    #
    # The new contract:
    #   - Production: the canonical apply step (`python3 -m
    #     sndr.apply`) is the ONLY thing run. If sndr isn't
    #     installed, the call fails loudly. The baked-in entry-point
    #     handles in-process runtime monkey-patches.
    #   - Dev: opt in to the `/plugin` editable install via
    #     `SNDR_DEV_INSTALL_PLUGIN=1`. UNIFIED ROOT BUG fix (2026-06-22):
    #     this installs the SNDR PACKAGE (the repo root mounted at
    #     `/plugin` — the dir whose `pyproject.toml` registers
    #     `genesis_v7 = "sndr.plugin:register"`), NOT the empty legacy
    #     `tools/genesis_vllm_plugin` subdir. The editable install writes
    #     the `.egg-info` entry-point metadata into the container's
    #     site-packages, so vllm serve's `load_general_plugins()` then
    #     loads `sndr.plugin:register` in-process and runtime monkey-
    #     patches (incl. g4_85) fire in the SERVING process — which the
    #     `python3 -m sndr.apply` subprocess cannot do (see the two-
    #     process boundary note above).
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
    # Optional dev-mode plugin install (DA-008 + UNIFIED ROOT BUG fix
    # 2026-06-22). Editable-install the SNDR PACKAGE so its
    # `vllm.general_plugins` entry-point (`genesis_v7 = sndr.plugin:register`)
    # is registered in the container's site-packages. This is what makes
    # vllm serve load the plugin IN-PROCESS and re-apply runtime monkey-
    # patches (incl. g4_85) — see the two-process boundary note above.
    #
    # `/plugin` MUST be the sndr repo ROOT (the dir holding the root
    # `pyproject.toml` whose `[project.entry-points."vllm.general_plugins"]`
    # points at `sndr.plugin:register`). Mounting the empty legacy
    # `tools/genesis_vllm_plugin` subdir here was the original bug: that
    # subdir's pyproject registered `genesis_v7:register` (an empty shim),
    # so no Genesis runtime patch ever fired in the serving process.
    #
    # We copy to a writable /tmp dir because `/plugin` is mounted `:ro`
    # and an editable install writes `*.egg-info` next to the source.
    # `--no-deps` keeps cold-boot fast (pyyaml/packaging are already in
    # the vllm image). After install we assert the entry-point actually
    # registered, so a misconfigured `plugin_src` fails LOUDLY instead of
    # silently booting without the in-process plugin.
    if has_plugin:
        bootstrap_parts.append(
            'if [ "${SNDR_DEV_INSTALL_PLUGIN:-0}" = "1" ]; then '
            "cp -r /plugin /tmp/sndr_plugin_src && "
            "pip install --quiet --disable-pip-version-check "
            "--root-user-action=ignore --no-deps -e "
            "/tmp/sndr_plugin_src 2>&1 | tail -2 && "
            "python3 -c \"import importlib.metadata as m; "
            "eps=m.entry_points(group='vllm.general_plugins'); "
            "names=[e.value for e in eps]; "
            "assert any('sndr.plugin' in n for n in names), "
            "'SNDR_DEV_INSTALL_PLUGIN=1 but vllm.general_plugins "
            "entry-point sndr.plugin:register NOT registered — is /plugin "
            "the sndr repo ROOT (with the root pyproject), not the empty "
            "tools/genesis_vllm_plugin subdir?'; "
            "print('[sndr] in-process plugin entry-point registered:', names)\"; "
            'fi'
        )
    # Canonical apply step (always runs).
    bootstrap_parts.append(apply_step)
    bootstrap_parts.append(f"exec {cmd}")
    bootstrap = "; ".join(bootstrap_parts)
    # POSIX-escape EVERY single quote in the assembled bootstrap (the vllm-arg
    # shell-quotes AND the plugin-assert's single-quoted python strings) so the
    # outer single-quoted -c '...' wrapper survives. Escaping the joined
    # bootstrap ONCE here (rather than pre-escaping only cmd) is what lets the
    # SNDR_DEV_INSTALL_PLUGIN assert coexist with the JSON args.
    bootstrap_escaped = bootstrap.replace("'", "'\\''")
    lines.append(f"  -c '{bootstrap_escaped}'")
    return "\n".join(lines)

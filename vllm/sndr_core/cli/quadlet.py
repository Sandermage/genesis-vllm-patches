# SPDX-License-Identifier: Apache-2.0
"""S3.2 (UNIFIED_CONFIG plan 2026-05-09; audit P3-2 closure 2026-05-12):
`sndr quadlet render` — Podman Quadlet (.container) renderer.

Why
---
Podman Quadlet — the recommended path for production-grade systemd-
managed containers without a root daemon (like Docker compose). The
operator places a `<name>.container` file into `~/.config/containers/systemd/`,
runs `systemctl --user daemon-reload`, and systemd itself starts
the container. Genesis did not have this path — the operator had to
manually translate the bash launch-script into Quadlet format.

Format
------
Quadlet is an INI-like format with `[Unit]`, `[Container]`,
`[Service]`, `[Install]` sections. Podman reads it and generates a
full-fledged `.service` file under the hood.

Minimal example:

    [Unit]
    Description=Genesis vLLM (preset X)
    After=network.target

    [Container]
    Image=vllm/vllm-openai:nightly
    ContainerName=vllm-genesis
    PublishPort=8000:8000
    Volume=/srv/models:/models:ro
    Environment=GENESIS_ENABLE_X=1
    Exec=vllm serve /models/...

    [Service]
    Restart=on-failure

    [Install]
    WantedBy=default.target

Test contract — `tests/unit/cli/test_quadlet_render.py`:
  • All required sections are present.
  • Environment lines render all env vars (one per line).
  • Volume lines render all mounts with host_paths substitution.
  • PublishPort is correct.
  • Exec line — vllm serve <args>.
  • Idempotence.
"""
from __future__ import annotations

import argparse
import re
import shlex
from pathlib import Path
from typing import Any, Optional

from . import _io
from .compose import _container_command, _load_host_paths, _resolve_mount, _resolve


__all__ = [
    "add_argparser", "render_quadlet",
    "run_quadlet_render",
]


# Etap 2.3 (audit 2026-05-12): systemd-safe escaping for Environment= /
# Exec= lines. Previously raw `f"Environment={k}={v}"` could be broken
# by newlines, quotes, or invalid env-key chars in values, producing a
# unit file that fails to load with cryptic systemd errors.
_SYSTEMD_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_env_key(key: str) -> None:
    """Reject env keys systemd cannot accept.

    Systemd `Environment=KEY=VALUE` requires KEY to match a POSIX
    shell variable name (letters, digits, underscore; not starting
    with a digit). Invalid keys silently break unit-file loading.
    """
    if not _SYSTEMD_ENV_KEY_RE.match(key):
        raise ValueError(
            f"systemd-invalid env key {key!r}: must match "
            r"^[A-Za-z_][A-Za-z0-9_]*$"
        )


def _escape_env_value(value: str) -> str:
    """Escape an env value for use inside `Environment=KEY=VALUE`.

    Systemd rules (systemd.exec(5)):
      - backslash → doubled
      - newline   → `\\n` literal
      - tab       → `\\t` literal
      - if the value contains whitespace, quotes, or `$`, wrap in
        double quotes (POSIX-style; systemd evaluates the escapes
        inside quotes).

    The empty string returns `""` so the assignment is still well-formed.
    """
    if not value:
        return '""'
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace("\n", "\\n").replace("\t", "\\t")
    needs_quote = any(c in value for c in ' \t\n"\'$')
    if needs_quote:
        escaped = escaped.replace('"', '\\"')
        return f'"{escaped}"'
    return escaped


def _argv_for_exec(argv: list[str]) -> str:
    """Build the single `Exec=` value from canonical argv.

    Systemd `Exec=` is one logical line — embedded newlines would
    truncate the command silently. We reject them explicitly; spaces
    and quotes are handled by `shlex.quote` (Podman/Quadlet honours
    POSIX-style word splitting).
    """
    for a in argv:
        if "\n" in a:
            raise ValueError(
                f"argv entry contains newline (not representable on "
                f"single-line systemd Exec=): {a!r}"
            )
    return " ".join(shlex.quote(a) for a in argv)


def add_argparser(subparsers: Any) -> None:
    p = subparsers.add_parser(
        "quadlet",
        help="Podman Quadlet (.container) renderer (audit P3-2).",
        description=(
            "Renders a preset as a Podman Quadlet `<name>.container` file "
            "for systemd-managed launch via `systemctl --user`. "
            "An alternative to docker compose for rootless production "
            "deployments."
        ),
    )
    sub = p.add_subparsers(dest="quadlet_cmd", required=True)

    p_render = sub.add_parser(
        "render", help="Render preset → <name>.container to stdout/file",
    )
    p_render.add_argument("config", help="preset key")
    p_render.add_argument(
        "-o", "--output", default=None,
        help=(
            "Write to file (recommended: "
            "~/.config/containers/systemd/<name>.container)."
        ),
    )
    p_render.set_defaults(func=run_quadlet_render)


def render_quadlet(cfg, host_paths: Optional[dict[str, str]] = None) -> str:
    """Renders ModelConfig into Podman Quadlet format."""
    if host_paths is None:
        host_paths = _load_host_paths()

    docker = cfg.docker
    if docker is None:
        raise ValueError(
            f"preset {cfg.key!r} has no docker block — quadlet "
            "requires a container image / name / port."
        )

    container_name = docker.container_name or f"sndr-{cfg.key}"
    image = docker.effective_image_ref()
    host_port = docker.effective_host_port()
    container_port = docker.effective_container_port()

    # Environment lines — one per line. Quadlet accepts repeated
    # Environment= entries.
    env_pairs: list[tuple[str, str]] = []
    for k, v in cfg.system_env.items():
        env_pairs.append((str(k), str(v)))
    for k, v in cfg.genesis_env.items():
        env_pairs.append((str(k), str(v)))
    if cfg.api_key and not any(k == "VLLM_API_KEY" for k, _ in env_pairs):
        env_pairs.append(("VLLM_API_KEY", str(cfg.api_key)))

    # Volume lines with host_paths substitution.
    volumes: list[str] = [
        _resolve_mount(m, host_paths)
        for m in (docker.mounts or [])
    ]

    # Etap 2.3: argv goes through `_argv_for_exec` which rejects
    # newlines (untenable on a single systemd Exec= line) and applies
    # shlex.quote per argument.
    exec_argv = _container_command(cfg)
    exec_line = _argv_for_exec(exec_argv)

    lines: list[str] = []
    lines.append(
        "# Generated by `sndr quadlet render` — DO NOT edit by hand."
    )
    lines.append(f"# Source preset: {cfg.key} ({cfg.title})")
    lines.append(f"# Maintainer: {cfg.maintainer}")
    lines.append("#")
    lines.append(
        "# Install: place this file at "
        "~/.config/containers/systemd/<name>.container"
    )
    lines.append(
        "# Reload: systemctl --user daemon-reload && "
        f"systemctl --user start {container_name}.service"
    )
    lines.append("")
    lines.append("[Unit]")
    lines.append(f"Description=Genesis vLLM (preset {cfg.key})")
    lines.append("After=network-online.target")
    lines.append("Wants=network-online.target")
    lines.append("")
    lines.append("[Container]")
    lines.append(f"Image={image}")
    lines.append(f"ContainerName={container_name}")
    lines.append(f"PublishPort={host_port}:{container_port}")
    if docker.shm_size:
        lines.append(f"ShmSize={docker.shm_size}")
    # GPU access — Podman Quadlet supports AddDevice= for nvidia.
    lines.append("AddDevice=nvidia.com/gpu=all")
    if docker.network:
        lines.append(f"Network={docker.network}")
    for vol in volumes:
        lines.append(f"Volume={vol}")
    # Etap 2.3: validate key + escape value per Environment= entry.
    # Invalid keys raise ValueError early; values are wrapped/escaped
    # so newlines / quotes / spaces don't break the unit file.
    for k, v in env_pairs:
        _validate_env_key(k)
        lines.append(f"Environment={k}={_escape_env_value(v)}")
    lines.append(f"Exec={exec_line}")
    lines.append("")
    lines.append("[Service]")
    lines.append("Restart=on-failure")
    lines.append("TimeoutStartSec=900")
    lines.append("")
    lines.append("[Install]")
    lines.append("WantedBy=default.target")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def run_quadlet_render(args: argparse.Namespace) -> int:
    cfg = _resolve(args.config)
    if cfg is None:
        return 2
    out = render_quadlet(cfg)
    if args.output:
        Path(args.output).write_text(out)
        _io.info(f"wrote {args.output} ({len(out)} bytes)")
    else:
        print(out)
    return 0

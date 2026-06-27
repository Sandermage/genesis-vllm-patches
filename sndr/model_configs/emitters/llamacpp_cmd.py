# SPDX-License-Identifier: Apache-2.0
"""``build_llamacpp_cmd(cfg)`` + ``build_llamacpp_docker_cmd(...)`` — emit the
llama.cpp (llama-server) launch for an ``engine == "llama-cpp"`` config.

Why a separate emitter (not the vLLM ``build_vllm_cmd`` / ``build_docker_cmd``):
the llama.cpp lane is structurally different from vLLM.

  - The command is ``llama-server -m <gguf> ...``, NOT ``vllm serve --model
    <dir> ...``.
  - There is NO Genesis patch stack to apply — Genesis is a vLLM/Qwen3-Next
    overlay; the official ``ghcr.io/ggml-org/llama.cpp`` image already carries
    native MTP (PR #22673). So the docker bootstrap is just ``exec
    llama-server ...`` — none of the two-process ``python3 -m sndr.apply`` +
    plugin-entry-point dance the vLLM bootstrap needs.
  - The image is the pinned llama.cpp CUDA server build, NOT the rig's vLLM
    image (the single-card hardware def the lane reuses pins a vLLM image for
    its vLLM presets; the llama.cpp lane overrides it here).

The argv itself is built ONCE in ``runtime_command.build_llamacpp_argv`` (the
canonical source of truth, also consumed by compose/quadlet/k8s). This emitter
shell-quotes that argv for the bash launch script, so the dry-run docker
command and the canonical argv can never drift.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..runtime_command import (
    LLAMACPP_SERVER_IMAGE,
    argv_to_shell,
    build_llamacpp_argv,
)
from ..types import resolve_symbolic_mounts
from .shell import shell_quote

if TYPE_CHECKING:
    from ..schema import ModelConfig

#: Docker label the daemon/GUI container-ops whitelist recognises as
#: SNDR-managed (kept in lock-step with
#: ``product_api.legacy.container_ops._MANAGED_LABEL``). Emitting it on the
#: llama.cpp container makes daemon ops (start/stop/restart/logs/stats) accept
#: it via the label path in addition to the "llamacpp" name-prefix path.
_MANAGED_LABEL = "sndr.managed"


def build_llamacpp_cmd(cfg: "ModelConfig") -> list[str]:
    """llama-server command parts (without exec/docker prefix), shell-quoted.

    Derived from the canonical ``build_llamacpp_argv`` argv so there is a
    single source of truth for the flag set / order.
    """
    return argv_to_shell(build_llamacpp_argv(cfg).argv)


def build_llamacpp_docker_cmd(
    cfg: "ModelConfig",
    cmd_parts: list[str],
    host_paths: Optional[dict[str, str]] = None,
    *,
    strict_mounts: bool = False,
) -> str:
    """Render the docker run command for the llama.cpp lane.

    Mirrors ``build_docker_cmd`` (mounts / ports / env / device flags) but:
      - uses the pinned ``LLAMACPP_SERVER_IMAGE`` instead of the rig's vLLM
        image,
      - the container bootstrap is a bare ``exec llama-server ...`` (no
        ``sndr.apply`` — there is no Genesis patch stack for llama.cpp).
    """
    d = cfg.docker
    if d is None:
        # Bare-metal llama.cpp launch — caller handles the non-docker branch.
        return "exec " + " \\\n  ".join(cmd_parts)

    # Resolve symbolic mounts exactly like the vLLM docker emitter.
    resolved_mounts = list(d.mounts)
    needs_resolution = any("${" in m for m in d.mounts)
    if needs_resolution:
        if host_paths is None:
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
        resolved_mounts = resolve_symbolic_mounts(
            d.mounts, host_paths, strict=strict_mounts,
        )

    lines = [
        f"docker rm -f {shell_quote(d.container_name)} 2>/dev/null || true",
        "",
        "docker run -d \\",
        f"  --name {shell_quote(d.container_name)} \\",
        # Override the entrypoint to a shell that execs the full command. The
        # image's default entrypoint IS llama-server, but routing through a
        # shell keeps the single-source-of-truth argv (which leads with the
        # "llama-server" token) intact and survives any future entrypoint
        # change in the upstream image.
        "  --entrypoint /bin/sh \\",
        # Mark the container SNDR-managed so the daemon/GUI container ops
        # (start/stop/restart/logs/stats) accept it. Without this label the
        # `llamacpp-...` container only passes the name-prefix whitelist; the
        # explicit label is the belt-and-suspenders path that survives an
        # operator SNDR_MANAGED_PREFIXES override that drops "llamacpp". Mirrors
        # the vLLM lane, whose `vllm-` name already prefix-matches.
        f"  --label {_MANAGED_LABEL}=true \\",
        f"  --gpus {shell_quote(d.gpus)} \\",
        f"  --shm-size={shell_quote(d.shm_size)} \\",
    ]
    if d.memory_limit:
        lines.append(f"  --memory={shell_quote(d.memory_limit)} \\")
    if d.network:
        lines.append(f"  --network {shell_quote(d.network)} \\")
    lines.append(
        f"  -p {d.effective_host_port()}:{d.effective_container_port()} \\"
    )
    for m in resolved_mounts:
        lines.append(f"  -v {shell_quote(m)} \\")
    for f in d.extra_run_flags:
        lines.append(f"  {f} \\")
    # Env vars — system_env only. The llama.cpp lane has NO GENESIS_* patch
    # env (those are vLLM-only); genesis_env is intentionally not emitted.
    for k, v in sorted(cfg.system_env.items()):
        lines.append(f'  -e {k}={shell_quote(v)} \\')
    # Pinned llama.cpp server image (NOT the rig's vLLM image).
    lines.append(f"  {shell_quote(LLAMACPP_SERVER_IMAGE)} \\")
    cmd = " ".join(cmd_parts)
    lines.append(f"  -c {shell_quote('exec ' + cmd)}")
    return "\n".join(lines)


__all__ = ["build_llamacpp_cmd", "build_llamacpp_docker_cmd"]

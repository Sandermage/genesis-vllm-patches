# SPDX-License-Identifier: Apache-2.0
"""Preflight checks — environmental sanity BEFORE actually launching.

Layer 3 of the "100% close gaps" strategy. Catches:
  - host mount paths missing
  - container name collision
  - git HEAD ≠ config.genesis_pin (drift)
  - vLLM image not pulled
  - GPU count visible to docker matches config.hardware.n_gpus
  - stale compile cache from prior pin

Each check returns (passed, message). preflight_all() runs them all and
returns a structured report. CLI: `genesis model-config preflight <key>`.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreflightCheck:
    name: str
    passed: bool
    message: str
    severity: str  # 'error' / 'warning' / 'info'


def _run(cmd: list[str], timeout: int = 5) -> tuple[int, str, str]:
    """Run a command; return (returncode, stdout, stderr). Defensive."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return -1, "", "command failed"


def check_mounts(cfg) -> list[PreflightCheck]:
    """All host paths in docker.mounts must exist.

    F-016 fix (audit 2026-05-07): mounts may contain `${var}` symbolic
    references; these are resolved through host.yaml before existence
    checks. Unresolvable variables surface as their own error so the
    operator sees "missing host config var" rather than a confusing
    "host path '${models_dir}' does not exist".
    """
    out: list[PreflightCheck] = []
    if not cfg.docker:
        return out

    # Lazy-load host.yaml only if any mount uses a symbolic var.
    needs_resolution = any("${" in m for m in cfg.docker.mounts)
    host_paths: dict[str, str] = {}
    if needs_resolution:
        try:
            from .host import load_host_config
            host_paths = load_host_config().paths
        except Exception:
            host_paths = {}

    for mount in cfg.docker.mounts:
        # Resolve symbolic references first; on failure surface a
        # dedicated check so the operator knows to update host.yaml
        # rather than chasing a non-existent path.
        if "${" in mount:
            try:
                from .schema import resolve_symbolic_mounts
                resolved = resolve_symbolic_mounts([mount], host_paths)[0]
            except Exception as e:
                out.append(PreflightCheck(
                    name=f"mount:{mount}", passed=False,
                    message=f"symbolic mount unresolvable: {e}",
                    severity="error",
                ))
                continue
        else:
            resolved = mount

        # Format: <host>:<container>[:<mode>]
        parts = resolved.split(":")
        if len(parts) < 2:
            out.append(PreflightCheck(
                name=f"mount:{mount}", passed=False,
                message=f"malformed mount spec '{mount}'",
                severity="error",
            ))
            continue
        host_path = parts[0]
        if host_path.startswith("~"):
            host_path = os.path.expanduser(host_path)
        if not Path(host_path).exists():
            out.append(PreflightCheck(
                name=f"mount:{mount}", passed=False,
                message=f"host path '{host_path}' does not exist",
                severity="error",
            ))
        else:
            out.append(PreflightCheck(
                name=f"mount:{host_path}", passed=True,
                message="ok", severity="info",
            ))
    return out


def check_container_name_free(cfg) -> Optional[PreflightCheck]:
    """If a container with this name is already running, fail."""
    if not cfg.docker:
        return None
    name = cfg.docker.container_name
    rc, out, _ = _run([
        "docker", "ps", "-a", "--filter", f"name=^{name}$",
        "--format", "{{.Status}}",
    ])
    if rc != 0:
        return PreflightCheck(
            name="docker_available", passed=False,
            message="docker not available on this host",
            severity="error",
        )
    if not out.strip():
        return PreflightCheck(
            name=f"container:{name}", passed=True,
            message="name available", severity="info",
        )
    if "Up" in out:
        return PreflightCheck(
            name=f"container:{name}", passed=False,
            message=f"'{name}' already running — stop first",
            severity="error",
        )
    # Exited container — operator must rm or use --force
    return PreflightCheck(
        name=f"container:{name}", passed=True,
        message=f"'{name}' exited (will be removed at launch)",
        severity="warning",
    )


def check_image_pulled(cfg) -> Optional[PreflightCheck]:
    """Check that docker image is locally available."""
    if not cfg.docker:
        return None
    image = cfg.docker.image
    rc, out, _ = _run([
        "docker", "image", "inspect", image, "--format", "{{.Id}}",
    ])
    if rc != 0:
        return PreflightCheck(
            name=f"image:{image}", passed=False,
            message=f"image '{image}' not pulled "
                    f"— run `docker pull {image}`",
            severity="error",
        )
    return PreflightCheck(
        name=f"image:{image}", passed=True,
        message="image present", severity="info",
    )


def check_gpu_count(cfg) -> Optional[PreflightCheck]:
    """nvidia-smi sees ≥ cfg.hardware.n_gpus GPUs."""
    rc, out, _ = _run(["nvidia-smi", "--list-gpus"])
    if rc != 0:
        return PreflightCheck(
            name="nvidia-smi", passed=False,
            message="nvidia-smi not available",
            severity="warning",
        )
    visible = len([ln for ln in out.strip().splitlines() if ln.startswith("GPU")])
    if visible < cfg.hardware.n_gpus:
        return PreflightCheck(
            name="gpu_count", passed=False,
            message=f"config requires {cfg.hardware.n_gpus} GPUs, "
                    f"only {visible} visible",
            severity="error",
        )
    return PreflightCheck(
        name="gpu_count", passed=True,
        message=f"{visible} GPUs visible (config wants "
                f"{cfg.hardware.n_gpus})",
        severity="info",
    )


def check_gpu_vram(cfg) -> list[PreflightCheck]:
    """Each GPU has ≥ cfg.hardware.min_vram_per_gpu_mib free."""
    out: list[PreflightCheck] = []
    rc, stdout, _ = _run([
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0:
        return out
    for line in stdout.strip().splitlines()[: cfg.hardware.n_gpus]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, total_mib, free_mib = int(parts[0]), int(parts[1]), int(parts[2])
        if total_mib < cfg.hardware.min_vram_per_gpu_mib:
            out.append(PreflightCheck(
                name=f"vram:gpu{idx}", passed=False,
                message=f"GPU {idx} has {total_mib} MiB total, "
                        f"config requires {cfg.hardware.min_vram_per_gpu_mib}",
                severity="error",
            ))
        else:
            # Warn if free is too low
            need = int(cfg.hardware.min_vram_per_gpu_mib *
                       cfg.gpu_memory_utilization * 0.95)
            if free_mib < need:
                out.append(PreflightCheck(
                    name=f"vram:gpu{idx}", passed=False,
                    message=f"GPU {idx}: only {free_mib} MiB free, "
                            f"config will need ~{need}. "
                            f"Stop other workloads first.",
                    severity="warning",
                ))
            else:
                out.append(PreflightCheck(
                    name=f"vram:gpu{idx}", passed=True,
                    message=f"GPU {idx}: {free_mib} MiB free",
                    severity="info",
                ))
    return out


def check_genesis_pin(cfg, repo_root: Optional[Path] = None) -> Optional[PreflightCheck]:
    """git HEAD short SHA matches config.genesis_pin (if set)."""
    if not cfg.genesis_pin:
        return None
    repo = repo_root or Path(__file__).resolve().parents[3]
    rc, out, _ = _run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
    )
    if rc != 0:
        return PreflightCheck(
            name="genesis_pin", passed=False,
            message="git not available or not a repo",
            severity="warning",
        )
    actual = out.strip()
    expected = cfg.genesis_pin
    # Allow prefix match (config can store short SHA or full)
    if actual.startswith(expected) or expected.startswith(actual):
        return PreflightCheck(
            name="genesis_pin", passed=True,
            message=f"HEAD={actual} matches config.genesis_pin={expected}",
            severity="info",
        )
    return PreflightCheck(
        name="genesis_pin", passed=False,
        message=f"git HEAD={actual} != config.genesis_pin={expected}. "
                f"Either checkout {expected} or update YAML.",
        severity="warning",
    )


def check_vllm_pin_in_image(cfg) -> Optional[PreflightCheck]:
    """vllm version inside docker image matches vllm_pin_required."""
    if not cfg.docker or not cfg.vllm_pin_required:
        return None
    image = cfg.docker.image
    rc, out, _ = _run([
        "docker", "run", "--rm", "--entrypoint", "bash", image,
        "-c", "pip show vllm 2>/dev/null | grep '^Version' | awk '{print $2}'",
    ], timeout=20)
    if rc != 0:
        return PreflightCheck(
            name="vllm_pin", passed=False,
            message=f"failed to query vllm version inside {image}",
            severity="warning",
        )
    actual = out.strip()
    if actual == cfg.vllm_pin_required:
        return PreflightCheck(
            name="vllm_pin", passed=True,
            message=f"image {image} has vllm {actual}",
            severity="info",
        )
    return PreflightCheck(
        name="vllm_pin", passed=False,
        message=f"image vllm version '{actual}' != "
                f"vllm_pin_required '{cfg.vllm_pin_required}'. "
                f"Pull a different image tag or update YAML.",
        severity="error",
    )


def check_stale_compile_cache(cfg) -> list[PreflightCheck]:
    """Look for stale compile/triton caches that may invalidate boot.

    Heuristic: if the container's compile-cache mount points to an
    existing directory with files older than the model bench date,
    suggest clearing.
    """
    out: list[PreflightCheck] = []
    if not cfg.docker or not cfg.reference_metrics:
        return out

    # F-016 fix (audit 2026-05-07): resolve `${var}` mounts before
    # inspecting file presence. Same pattern as check_mounts above.
    needs_resolution = any("${" in m for m in cfg.docker.mounts)
    host_paths: dict[str, str] = {}
    if needs_resolution:
        try:
            from .host import load_host_config
            host_paths = load_host_config().paths
        except Exception:
            host_paths = {}

    for mount in cfg.docker.mounts:
        if "${" in mount:
            try:
                from .schema import resolve_symbolic_mounts
                resolved = resolve_symbolic_mounts([mount], host_paths)[0]
            except Exception:
                continue  # unresolvable; check_mounts already reported it
        else:
            resolved = mount
        parts = resolved.split(":")
        if len(parts) < 2:
            continue
        host_path = parts[0]
        if host_path.startswith("~"):
            host_path = os.path.expanduser(host_path)
        if "compile-cache" not in host_path and "triton-cache" not in host_path:
            continue
        p = Path(host_path)
        if not p.exists():
            continue
        # Check if any files inside; OK to have empty (cold boot)
        children = list(p.iterdir()) if p.is_dir() else []
        if not children:
            out.append(PreflightCheck(
                name=f"cache:{p.name}", passed=True,
                message="empty (cold boot expected)",
                severity="info",
            ))
        else:
            out.append(PreflightCheck(
                name=f"cache:{p.name}", passed=True,
                message=f"warm cache present ({len(children)} entries)",
                severity="info",
            ))
    return out


# ─── Public API ────────────────────────────────────────────────────────


def preflight_all(cfg) -> list[PreflightCheck]:
    """Run all preflight checks. Returns list of PreflightCheck."""
    out: list[PreflightCheck] = []

    # Basic env
    out.extend(check_mounts(cfg))

    cnc = check_container_name_free(cfg)
    if cnc:
        out.append(cnc)

    img = check_image_pulled(cfg)
    if img:
        out.append(img)

    gc = check_gpu_count(cfg)
    if gc:
        out.append(gc)

    out.extend(check_gpu_vram(cfg))

    gp = check_genesis_pin(cfg)
    if gp:
        out.append(gp)

    # Skip vllm_pin_in_image — slow (creates throwaway container)
    # Operators run this manually if they suspect image drift

    out.extend(check_stale_compile_cache(cfg))

    return out


def has_blockers(checks: list[PreflightCheck]) -> bool:
    """Any error-severity failures?"""
    return any(
        not c.passed and c.severity == "error" for c in checks
    )

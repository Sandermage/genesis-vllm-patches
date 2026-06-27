# SPDX-License-Identifier: Apache-2.0
"""Launcher-side autodetect PREFLIGHT gate (A1-A9).

Runs BEFORE `docker run` / `vllm serve` and turns the common
"boots-then-dies-cryptically" failure classes into clear operator-facing
errors and warnings:

  A1  GPU count       nvidia-smi count vs cfg.hardware.n_gpus     → warn
  A2  vLLM pin        launch image pin vs cfg.vllm_pin_required   → warn
  A3  model path      <models_dir>/<model> exists                 → error
  A4  drafter model   spec_decode drafter checkpoint exists       → error
  A5  HF cache        huggingface cache mount present             → warn
  A6  max-model-len   max_model_len vs model max_position_embed.  → warn
  A7  served name     default served_model_name from model id     → mutate
  A8  port conflict   target host port is free                    → error
  A9  repo resolution SNDR_SRC/GENESIS_REPO → git toplevel / home → resolve

All host-touching operations live behind :class:`HostProbe` so the gate
is unit-testable with a fake probe (no GPU / docker / sockets / FS).

Severity contract:
  * errors  → caller MUST abort the launch (return non-zero).
  * warnings → surfaced, launch proceeds (operator may have a reason).
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ─── Result types ──────────────────────────────────────────────────────────


@dataclass
class PreflightFinding:
    """One preflight result line."""
    code: str          # "A1".."A9"
    message: str       # operator-facing English


@dataclass
class PreflightResult:
    errors: list[PreflightFinding] = field(default_factory=list)
    warnings: list[PreflightFinding] = field(default_factory=list)
    resolved_repo: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, code: str, message: str) -> None:
        self.errors.append(PreflightFinding(code, message))

    def warn(self, code: str, message: str) -> None:
        self.warnings.append(PreflightFinding(code, message))


# ─── Host probe (all side effects live here) ───────────────────────────────


class HostProbe:
    """Side-effecting host operations behind a swappable seam.

    The default implementation shells out to nvidia-smi / docker, opens
    sockets, and touches the filesystem. Tests subclass and override.
    """

    def gpu_count(self) -> Optional[int]:
        """Count GPUs via `nvidia-smi --query-gpu=name`. None if nvidia-smi
        is unavailable or errors (cannot assert → caller skips A1)."""
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if res.returncode != 0:
            return None
        lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
        return len(lines)

    def path_exists(self, path: str) -> bool:
        try:
            return Path(path).exists()
        except OSError:
            return False

    def port_in_use(self, port: int) -> bool:
        """True if a TCP listener already holds `port` on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
            except OSError:
                return True
            return False

    def image_pin(self, image_ref: str) -> Optional[str]:
        """Best-effort vLLM pin of a local docker image. Reads the
        `VLLM_VERSION` / pip-shown version via `docker inspect` labels;
        None when docker is unavailable or the label is absent."""
        try:
            res = subprocess.run(
                ["docker", "inspect", "--format",
                 "{{index .Config.Labels \"vllm.version\"}}", image_ref],
                capture_output=True, text=True, timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if res.returncode != 0:
            return None
        pin = res.stdout.strip()
        return pin or None

    def read_model_config_json(self, model_dir: str) -> Optional[dict]:
        """Parse `<model_dir>/config.json`. None when absent / unreadable."""
        cfg_path = Path(model_dir) / "config.json"
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def git_toplevel(self) -> Optional[str]:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if res.returncode != 0:
            return None
        top = res.stdout.strip()
        return top or None


# ─── Mount → host-path mapping ─────────────────────────────────────────────


def _parse_mount(m: str) -> Optional[tuple[str, str]]:
    """Split a docker `-v` mount spec into (host, container).

    Handles `host:container` and `host:container:ro|rw`. Returns None for
    unresolved (`${var}`) or malformed specs.
    """
    if "${" in m:
        return None
    parts = m.split(":")
    if len(parts) < 2:
        return None
    return parts[0], parts[1]


def container_to_host_path(
    container_path: str, mounts: list[str],
) -> Optional[str]:
    """Map a container-side path (e.g. ``/models/Foo``) to its host path
    by finding the bind-mount whose container target is a parent of it.

    Returns the host path, or None when no mount covers `container_path`.
    The longest matching container target wins (most specific mount).
    """
    best: Optional[tuple[int, str]] = None
    for m in mounts:
        pair = _parse_mount(m)
        if pair is None:
            continue
        host, cont = pair
        cont = cont.rstrip("/")
        if container_path == cont or container_path.startswith(cont + "/"):
            depth = len(cont)
            if best is None or depth > best[0]:
                rel = container_path[len(cont):].lstrip("/")
                resolved = str(Path(host) / rel) if rel else host
                best = (depth, resolved)
    return best[1] if best else None


# ─── Individual checks ─────────────────────────────────────────────────────


_HF_CONTAINER_TARGETS = ("/root/.cache/huggingface", "/huggingface")

# A6 (llama.cpp lane): safe single-card `-c` (context / KV-pool) ceiling.
# llama-server pre-allocates the FULL KV pool at `-c`, so the ceiling is a
# fits-on-load budget, not vLLM's clamp-at-runtime semantics. club-3090's
# single-card CLIFFS.md: 131072 fits (~22.5/24 GB, verify-stress 7/7), 200000
# already walls (OOM at load). 131072 is the validated ceiling; anything above
# warns (boots-then-OOMs is the exact failure this gate makes legible).
_LLAMACPP_SAFE_CTX_CEILING = 131072

# A6 (llama.cpp lane): the pinned llama-server CUDA image build the lane runs.
# Kept in lock-step with model_configs.runtime_command.LLAMACPP_SERVER_IMAGE.
_LLAMACPP_IMAGE_BUILD = "server-cuda-b9246"


def _check_a1_gpu_count(cfg, host, r) -> None:
    declared = int(getattr(getattr(cfg, "hardware", None), "n_gpus", 0) or 0)
    if declared <= 0:
        return
    detected = host.gpu_count()
    if detected is None:
        return  # nvidia-smi unavailable — cannot assert.
    if detected != declared:
        r.warn(
            "A1",
            f"GPU count mismatch: nvidia-smi detected {detected} GPU(s) but "
            f"the config requires {declared} (tensor-parallel-size). "
            f"vllm will fail at init or run degraded.",
        )


def _check_a2_pin(cfg, host, r) -> None:
    required = getattr(cfg, "vllm_pin_required", None)
    if not required:
        return
    docker = getattr(cfg, "docker", None)
    if docker is None:
        return
    image_ref = docker.effective_image_ref()
    observed = host.image_pin(image_ref)
    if observed is None:
        return  # cannot determine the image pin — skip.
    if observed != required:
        r.warn(
            "A2",
            f"vLLM pin mismatch: image {image_ref!r} reports pin "
            f"{observed!r} but the config requires {required!r}. "
            f"Re-tag the validated pin or update vllm_pin_required.",
        )


def _check_a3_model_path(cfg, host, host_model_path, r) -> None:
    if host_model_path is None:
        return  # unmapped (e.g. ${var}); A3 only fires when we know the path.
    if not host.path_exists(host_model_path):
        r.error(
            "A3",
            f"model path does not exist on host: {host_model_path} "
            f"(container path {getattr(cfg, 'model_path', '?')}). Fetch the "
            f"checkpoint or fix the models_dir mount before launching.",
        )


def _check_a4_drafter(cfg, host, r) -> None:
    spec = getattr(cfg, "spec_decode", None)
    if spec is None:
        return
    drafter = getattr(spec, "model", None)
    if not drafter:
        return  # mtp/ngram use the target's own head — no separate checkpoint.
    docker = getattr(cfg, "docker", None)
    mounts = list(getattr(docker, "mounts", []) or []) if docker else []
    host_drafter = container_to_host_path(drafter, mounts) or drafter
    if not host.path_exists(host_drafter):
        r.error(
            "A4",
            f"drafter model path does not exist: {host_drafter} "
            f"(spec_decode.method={spec.method}, container path {drafter}). "
            f"Speculative decoding (MTP) will fail cryptically at engine "
            f"init — fetch the drafter checkpoint first.",
        )


def _check_a5_hf_cache(cfg, host_paths, r) -> None:
    docker = getattr(cfg, "docker", None)
    if docker is None:
        return  # bare-metal: HF cache uses the host default, no mount needed.
    mounts = list(getattr(docker, "mounts", []) or [])
    has_hf = False
    for m in mounts:
        pair = _parse_mount(m)
        if pair is None:
            # An unresolved ${hf_cache} still signals operator intent.
            if "hf_cache" in m or "huggingface" in m:
                has_hf = True
            continue
        _, cont = pair
        if cont.rstrip("/") in _HF_CONTAINER_TARGETS:
            has_hf = True
            break
    if not has_hf:
        default = os.environ.get("HF_HOME") or str(
            Path.home() / ".cache" / "huggingface")
        r.warn(
            "A5",
            f"no HuggingFace cache mount found in the docker block; "
            f"on-the-fly downloads (tokenizer/config) inside the container "
            f"will not persist. Add a "
            f"'{default}:/root/.cache/huggingface' mount.",
        )


def _check_a6_max_model_len(cfg, host, host_model_path, r) -> None:
    if host_model_path is None:
        return
    cfg_json = host.read_model_config_json(host_model_path)
    if not cfg_json:
        return
    mpe = cfg_json.get("max_position_embeddings")
    if not isinstance(mpe, int) or mpe <= 0:
        return
    declared = int(getattr(cfg, "max_model_len", 0) or 0)
    if declared > mpe:
        r.warn(
            "A6",
            f"max_model_len={declared} exceeds the model's "
            f"max_position_embeddings={mpe} (from config.json). vllm will "
            f"clamp or error; lower max_model_len to <= {mpe}.",
        )


def _check_a6_gguf(cfg, host, host_model_path, r) -> None:
    """A6 for the llama.cpp (GGUF) lane — GGUF-aware checks.

    The vLLM A6 reads `<model_dir>/config.json`, which DOES NOT EXIST for a
    GGUF lane: `model_path` is a single `.gguf` FILE, so that check is dead
    here. This branch replaces it with three GGUF-specific checks:

      - PATH: `model_path` must name a single `.gguf` file (the llama.cpp
        file-not-dir contract). A misconfigured dir → error (boot fails with a
        cryptic "failed to load model").
      - IMAGE: the docker image must be the pinned llama-server build
        (`server-cuda-b9246`), NOT a vLLM image — a vLLM image here means the
        compose image-override regressed (the lane would exec llama-server in a
        vLLM container that has no such binary).
      - CTX: `-c` (max_model_len) within the safe single-card ceiling. Above it
        the full pre-allocated KV pool OOMs at load (club-3090 CLIFFS.md).
    """
    from sndr.model_configs.gguf_resolution import is_gguf_path

    # PATH — the .gguf file contract (works from the typed cfg even before the
    # weights land; the FS existence of the file is A3's job).
    model_path = getattr(cfg, "model_path", "") or ""
    if not is_gguf_path(model_path):
        r.error(
            "A6",
            f"llama.cpp lane requires model_path to be a single .gguf FILE "
            f"(e.g. /models/.../Qwen3.6-27B-Q4_K_M.gguf), got {model_path!r}. "
            f"The vLLM HF-directory convention does not load on llama-server.",
        )

    # IMAGE — the pinned llama-server build, not a vLLM image.
    docker = getattr(cfg, "docker", None)
    if docker is not None:
        image_ref = getattr(docker, "image", "") or ""
        if "vllm" in image_ref.lower():
            r.error(
                "A6",
                f"llama.cpp lane docker.image is a vLLM image ({image_ref!r}) "
                f"— it has no llama-server binary. The composer must override "
                f"it to the pinned llama.cpp image ({_LLAMACPP_IMAGE_BUILD}).",
            )
        elif _LLAMACPP_IMAGE_BUILD not in image_ref:
            r.warn(
                "A6",
                f"llama.cpp lane docker.image {image_ref!r} is not the "
                f"validated build ({_LLAMACPP_IMAGE_BUILD}); the rolling "
                f":server-cuda tag has regressed before (club-3090 #187). "
                f"Pin the validated build.",
            )

    # CTX — single-card pre-allocated KV-pool ceiling.
    declared = int(getattr(cfg, "max_model_len", 0) or 0)
    if declared > _LLAMACPP_SAFE_CTX_CEILING:
        r.warn(
            "A6",
            f"-c (max_model_len)={declared} exceeds the safe single-card "
            f"ceiling {_LLAMACPP_SAFE_CTX_CEILING}. llama-server pre-allocates "
            f"the full KV pool at load, so a higher -c OOMs at boot on a "
            f"24 GB card (club-3090 CLIFFS.md: 200K walls). Lower -c to "
            f"<= {_LLAMACPP_SAFE_CTX_CEILING}.",
        )


def _check_a7_served_name(cfg) -> None:
    if getattr(cfg, "served_model_name", None):
        return
    model_path = getattr(cfg, "model_path", "") or ""
    default = Path(model_path).name or model_path
    if default:
        # Mutate the (already deep-copied by the caller) config so the
        # rendered launcher carries a stable served-model-name.
        cfg.served_model_name = default


def _check_a8_port(cfg, host, r) -> None:
    docker = getattr(cfg, "docker", None)
    if docker is None:
        return
    port = docker.effective_host_port()
    if host.port_in_use(port):
        r.error(
            "A8",
            f"target host port {port} is already in use. Stop the process "
            f"holding it (or change docker.port / pass --port) before "
            f"launching.",
        )


def _resolve_a9_repo(host) -> Optional[str]:
    """Resolve the Genesis source repo for the apply phase.

    Order: SNDR_SRC / GENESIS_REPO env (legacy alias) → git toplevel →
    the sndr home (~/.sndr). Returns None only when nothing resolves.
    """
    for var in ("SNDR_SRC", "GENESIS_REPO"):
        val = os.environ.get(var)
        if val:
            return val
    top = host.git_toplevel()
    if top:
        return top
    home = Path.home() / ".sndr"
    if home.is_dir():
        return str(home)
    return None


# ─── Orchestrator ──────────────────────────────────────────────────────────


def run_autodetect_preflight(
    cfg: Any,
    host_paths: Optional[dict[str, str]] = None,
    *,
    host: Optional[HostProbe] = None,
) -> PreflightResult:
    """Run the A1-A9 autodetect gate and return a structured result.

    The caller (`sndr launch`) renders errors/warnings to the operator
    and aborts when ``result.ok`` is False. ``host_paths`` is the resolved
    host.yaml mapping (models_dir/hf_cache/...) used as a fallback when a
    container path is not covered by an explicit docker mount.
    """
    host = host or HostProbe()
    host_paths = host_paths or {}
    r = PreflightResult()

    docker = getattr(cfg, "docker", None)
    mounts = list(getattr(docker, "mounts", []) or []) if docker else []

    # Resolve the host-side model path. Prefer an explicit docker mount;
    # fall back to host_paths['models_dir'] + basename when the mount is
    # symbolic / unresolved. Defensive: a partial/minimal cfg may not
    # declare model_path — in that case path-dependent checks (A3/A6)
    # simply do not fire.
    model_path = getattr(cfg, "model_path", None)
    host_model_path = (
        container_to_host_path(model_path, mounts) if model_path else None
    )
    if (host_model_path is None and model_path
            and host_paths.get("models_dir")
            and model_path.startswith("/models/")):
        host_model_path = str(
            Path(host_paths["models_dir"]) / Path(model_path).name)

    engine = getattr(cfg, "engine", "vllm")

    _check_a1_gpu_count(cfg, host, r)
    _check_a2_pin(cfg, host, r)
    _check_a3_model_path(cfg, host, host_model_path, r)
    _check_a4_drafter(cfg, host, r)
    _check_a5_hf_cache(cfg, host_paths, r)
    # A6 is engine-specific: vLLM compares max_model_len against the HF
    # config.json (a directory); the llama.cpp lane has a single .gguf FILE
    # (no config.json), so it gets a GGUF-aware variant (path + image + ctx
    # ceiling) instead of the dead config.json read.
    if engine == "llama-cpp":
        _check_a6_gguf(cfg, host, host_model_path, r)
    else:
        _check_a6_max_model_len(cfg, host, host_model_path, r)
    _check_a7_served_name(cfg)
    _check_a8_port(cfg, host, r)
    r.resolved_repo = _resolve_a9_repo(host)

    return r


__all__ = [
    "HostProbe",
    "PreflightFinding",
    "PreflightResult",
    "container_to_host_path",
    "run_autodetect_preflight",
    "_check_a6_gguf",
]

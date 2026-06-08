# SPDX-License-Identifier: Apache-2.0
"""Docker-inspect-based captor for `sndr model-config new --from-running`.

Reads `docker inspect <container>` on a live vLLM container and reverse-
engineers a `ModelConfig` YAML that captures the running configuration:

  - image / container_name / port mapping / GPU access / mounts / shm_size
  - vllm serve CLI flags (parsed from container Cmd / Args) → canonical
    ModelConfig fields (model_path, max_model_len, gpu_memory_utilization,
    max_num_seqs, max_num_batched_tokens, dtype, quantization,
    kv_cache_dtype, tool_call_parser, reasoning_parser, ...)
  - environment variables → genesis_env (GENESIS_ENABLE_* and SNDR_ENABLE_*)
    and system_env (PYTORCH_*, NCCL_*, VLLM_*, OMP_*, CUDA_*, TRITON_*)
  - speculative-config JSON → SpecDecodeConfig
  - hardware spec derived from `--tensor-parallel-size` (gpu_match_keys
    left as a placeholder operators MUST verify, since docker inspect does
    not reveal the actual GPU model on the host)

The captor is read-only: no docker writes, no engine-level introspection.
It depends only on `docker inspect` being available on PATH and the
container existing (running or stopped). Works with podman too — the
output schema is compatible.

The output is a `ModelConfig` instance that round-trips through
`dump_yaml()` so it can be saved to `~/.sndr/configs/<key>.yaml`.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from typing import Any, Optional


# Environment-variable buckets used to route captured env vars into the
# right ModelConfig field. genesis_env collects patch toggles; system_env
# collects framework / hardware tuning knobs.
_GENESIS_PATCH_PREFIXES = ("GENESIS_ENABLE_", "SNDR_ENABLE_")
_SYSTEM_ENV_PREFIXES = (
    "PYTORCH_", "VLLM_", "NCCL_", "OMP_", "CUDA_", "TRITON_",
    "HF_", "TRANSFORMERS_", "TOKENIZERS_", "TORCHINDUCTOR_",
    "TORCH_", "NVIDIA_", "HUGGINGFACE_",
)

# Env vars that docker injects unconditionally; we strip them so they
# don't pollute the captured config.
_STRIP_ENV_KEYS = frozenset({
    "PATH", "HOME", "HOSTNAME", "PWD", "SHLVL", "TERM", "USER",
    "LD_LIBRARY_PATH",  # vendor-specific, varies between hosts
    "_",                # bash internal
})


class CaptureError(RuntimeError):
    """Raised when docker-inspect-based capture cannot proceed."""


def _docker_binary() -> str:
    """Return the container CLI to use. Prefers `docker`, falls back to
    `podman`. Operators on a Podman-only host get the same UX."""
    import shutil
    if shutil.which("docker"):
        return "docker"
    if shutil.which("podman"):
        return "podman"
    raise CaptureError(
        "neither `docker` nor `podman` found on PATH — install one of them, "
        "or run --from-running on the host that owns the container"
    )


def _run_inspect(container: str) -> dict[str, Any]:
    """Run `docker inspect <container>` and return the single JSON object.

    Raises CaptureError on any failure with a clean operator-facing message.
    """
    binary = _docker_binary()
    try:
        proc = subprocess.run(
            [binary, "inspect", container],
            capture_output=True, text=True, timeout=15,
        )
    except FileNotFoundError as exc:
        raise CaptureError(f"could not invoke {binary}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CaptureError(
            f"{binary} inspect {container} timed out after 15s"
        ) from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise CaptureError(
            f"{binary} inspect {container!r} failed (rc={proc.returncode}): "
            f"{stderr or '<no stderr>'}"
        )

    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise CaptureError(
            f"{binary} inspect returned non-JSON output: {exc}"
        ) from exc

    if not payload:
        raise CaptureError(
            f"no inspect record returned for container {container!r}"
        )
    return payload[0]


def _full_cmdline(record: dict[str, Any]) -> list[str]:
    """Return Entrypoint + Cmd concatenated as one argv list.

    Docker may put `vllm serve …` in Cmd OR pre-pend it via Entrypoint.
    The official vllm/vllm-openai image uses ENTRYPOINT=["python3","-m",
    "vllm.entrypoints.openai.api_server"] and exposes the serve flags via
    CMD. Some operators override entirely via `docker run … vllm serve …`
    which puts the whole command in Cmd. We concatenate both to handle
    both layouts uniformly.
    """
    cfg = record.get("Config", {}) or {}
    entry = cfg.get("Entrypoint") or []
    cmd = cfg.get("Cmd") or []
    if not isinstance(entry, list):
        entry = [str(entry)]
    if not isinstance(cmd, list):
        cmd = [str(cmd)]
    return [str(x) for x in entry] + [str(x) for x in cmd]


def _parse_serve_args(argv: list[str]) -> dict[str, Any]:
    """Walk `vllm serve …` flags and return a kwargs dict for ModelConfig.

    Unknown flags are collected into `vllm_extra_args` so the captured
    config still launches identically to the running container.
    """
    # Strip everything before the first vllm-serve token so we operate
    # purely on the flags. We look for "vllm" / "serve" / "api_server" as
    # entry markers — anything before is interpreter / module spec.
    i = 0
    while i < len(argv):
        if argv[i] == "serve" or argv[i].endswith("api_server"):
            i += 1
            break
        i += 1
    flags = argv[i:]

    out: dict[str, Any] = {
        "vllm_extra_args": [],
        "spec_decode_json": None,
    }
    j = 0
    while j < len(flags):
        token = flags[j]
        nxt = flags[j + 1] if j + 1 < len(flags) else None

        def consume_value() -> str:
            nonlocal j
            if nxt is None:
                raise CaptureError(f"flag {token!r} missing value")
            j += 1
            return nxt

        if token in ("--model",):
            out["model_path"] = consume_value()
        elif token == "--served-model-name":
            out["served_model_name"] = consume_value()
        elif token == "--tensor-parallel-size":
            out["tensor_parallel_size"] = int(consume_value())
        elif token == "--gpu-memory-utilization":
            out["gpu_memory_utilization"] = float(consume_value())
        elif token == "--max-model-len":
            out["max_model_len"] = int(consume_value())
        elif token == "--max-num-seqs":
            out["max_num_seqs"] = int(consume_value())
        elif token == "--max-num-batched-tokens":
            out["max_num_batched_tokens"] = int(consume_value())
        elif token == "--dtype":
            out["dtype"] = consume_value()
        elif token == "--kv-cache-dtype":
            out["kv_cache_dtype"] = consume_value()
        elif token == "--quantization":
            out["quantization"] = consume_value()
        elif token == "--tool-call-parser":
            out["tool_call_parser"] = consume_value()
        elif token == "--reasoning-parser":
            out["reasoning_parser"] = consume_value()
        elif token == "--port":
            out["container_port"] = int(consume_value())
        elif token == "--host":
            # captured but not used (host bind is always 0.0.0.0 inside docker)
            consume_value()
        elif token == "--api-key":
            out["api_key"] = consume_value()
        elif token == "--enable-chunked-prefill":
            out["enable_chunked_prefill"] = True
        elif token == "--enforce-eager":
            out["enforce_eager"] = True
        elif token == "--disable-custom-all-reduce":
            out["disable_custom_all_reduce"] = True
        elif token == "--language-model-only":
            out["language_model_only"] = True
        elif token == "--trust-remote-code":
            out["trust_remote_code"] = True
        elif token == "--enable-auto-tool-choice":
            out["enable_auto_tool_choice"] = True
        elif token == "--speculative-config":
            out["spec_decode_json"] = consume_value()
        elif token == "vllm" or token == "serve" or token.endswith("api_server"):
            # leftover marker, skip
            pass
        else:
            # unknown flag — preserve verbatim so launch is bit-exact
            out["vllm_extra_args"].append(token)
            # heuristic: if the next token does not look like a flag,
            # consume it too as the value half of a "--flag value" pair.
            if nxt is not None and not nxt.startswith("-"):
                out["vllm_extra_args"].append(nxt)
                j += 1
        j += 1
    return out


def _parse_env(env_list: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Split Config.Env (`["KEY=value", ...]`) into (genesis_env, system_env).

    Both dicts mirror ModelConfig's `genesis_env` / `system_env` fields,
    which store keys VERBATIM with their full canonical prefix
    (e.g. ``GENESIS_ENABLE_P67`` / ``GENESIS_BUFFER_MODE`` /
    ``NCCL_DEBUG``). The launch renderer emits ``export {key}={val}``
    without prefix manipulation, so the captor must preserve the exact
    key name to ensure round-trip exactness with the running container.
    """
    genesis: dict[str, str] = {}
    system: dict[str, str] = {}
    for entry in env_list or []:
        if "=" not in entry:
            continue
        key, _, val = entry.partition("=")
        if key in _STRIP_ENV_KEYS:
            continue
        # Genesis bucket: anything starting with GENESIS_ / SNDR_. Cover
        # both `GENESIS_ENABLE_*` (patch toggles) and `GENESIS_*` (extra
        # config knobs like GENESIS_BUFFER_MODE / GENESIS_TQ_MAX_MODEL_LEN
        # that the schema also stores verbatim).
        if key.startswith(("GENESIS_", "SNDR_")):
            genesis[key] = val
        elif key.startswith(_SYSTEM_ENV_PREFIXES):
            system[key] = val
        # else: dropped (unrecognised non-Genesis env)
    return genesis, system


def _parse_mounts(record: dict[str, Any]) -> list[str]:
    """Build `-v <src>:<dst>[:ro]` mount strings from docker inspect Mounts.

    The captured mounts are LITERAL host paths (not symbolic-mount vars).
    Operators porting the captured config to a different host should
    review and substitute `${var}` references back in.
    """
    mounts: list[str] = []
    for m in record.get("Mounts", []) or []:
        if not isinstance(m, dict):
            continue
        src = m.get("Source") or ""
        dst = m.get("Destination") or ""
        mode = m.get("Mode") or ""
        if not src or not dst:
            continue
        spec = f"{src}:{dst}"
        if mode and "ro" in mode.split(","):
            spec += ":ro"
        mounts.append(spec)
    return mounts


def _parse_ports(record: dict[str, Any], container_port: int) -> tuple[int, int]:
    """Return (host_port, container_port) from inspect output.

    Docker stores port bindings as
        NetworkSettings.Ports = {"8000/tcp": [{"HostPort": "8101", ...}]}
    """
    net = record.get("NetworkSettings", {}) or {}
    ports = net.get("Ports", {}) or {}
    host_port = container_port
    key = f"{container_port}/tcp"
    bindings = ports.get(key) or []
    if bindings and isinstance(bindings, list):
        first = bindings[0]
        if isinstance(first, dict):
            try:
                host_port = int(first.get("HostPort") or container_port)
            except (TypeError, ValueError):
                host_port = container_port
    return host_port, container_port


def _parse_shm_size(record: dict[str, Any]) -> str:
    """Render HostConfig.ShmSize (bytes) as a docker-compatible string."""
    raw = record.get("HostConfig", {}).get("ShmSize")
    if not isinstance(raw, int) or raw <= 0:
        return "8g"  # ModelConfig default
    # Round to nearest GiB/MiB for readable YAML.
    gib = raw / (1024 ** 3)
    if gib >= 1 and abs(gib - round(gib)) < 0.05:
        return f"{int(round(gib))}g"
    mib = raw / (1024 ** 2)
    return f"{int(round(mib))}m"


def _parse_gpus(record: dict[str, Any]) -> str:
    """Map docker GPU access to ModelConfig.docker.gpus string.

    Handles three common forms:
      1. --gpus all       → DeviceRequests=[{Driver:nvidia, Count:-1}]  → "all"
      2. --gpus 2         → DeviceRequests=[{Count:2}]                  → "2"
      3. --gpus 'device=0,1' → DeviceIDs=["0","1"]                      → '"device=0,1"'

    Falls back to "all" when no DeviceRequests are present but
    NVIDIA_VISIBLE_DEVICES env var is set.
    """
    host = record.get("HostConfig", {}) or {}
    requests = host.get("DeviceRequests") or []
    for req in requests:
        if not isinstance(req, dict):
            continue
        caps = req.get("Capabilities") or []
        wants_gpu = any("gpu" in (c or []) for c in caps if isinstance(c, list))
        if not wants_gpu and (req.get("Driver") or "").lower() != "nvidia":
            continue
        ids = req.get("DeviceIDs") or []
        if ids:
            return "device=" + ",".join(str(i) for i in ids)
        count = req.get("Count")
        if isinstance(count, int):
            return "all" if count == -1 else str(count)
    # Env-var fallback
    for entry in record.get("Config", {}).get("Env", []) or []:
        if entry.startswith("NVIDIA_VISIBLE_DEVICES="):
            val = entry.split("=", 1)[1]
            if val and val != "all":
                return "device=" + val
            return "all"
    return "all"


def _image_digest(record: dict[str, Any]) -> Optional[str]:
    """Extract a pinned image@sha256 digest if one is recorded.

    docker inspect shows the *image* field as the resolved sha256 (e.g.
    "sha256:..."), but ModelConfig.image_digest expects the canonical
    `image@sha256:...` form pulled from the image's RepoDigests. We
    reconstruct it by combining Config.Image (the user-supplied tag) with
    Image (the resolved sha256) when the tag is `*@sha256:*` already.
    Otherwise we leave it None — the operator can fill in via
    `docker inspect -f '{{index .RepoDigests 0}}' <image>` and edit the
    YAML manually.
    """
    img = (record.get("Config", {}) or {}).get("Image", "") or ""
    if "@sha256:" in img:
        return img
    return None


def capture_from_running(
    container: str, *, key: str, maintainer: str = "<your-username>",
) -> "ModelConfig":  # noqa: F821 — forward ref to schema.ModelConfig
    """Public entry point. Returns a fully-populated `ModelConfig`.

    Args:
        container: Docker / Podman container name or ID.
        key: kebab-case key to assign to the new captured config.
        maintainer: GitHub-style user attribution for the YAML header.

    Raises:
        CaptureError: when docker is unavailable, the container can't be
            inspected, or the running command is not a recognisable
            `vllm serve …` invocation.
    """
    # Lazy imports so the captor module stays cheap to import in CLI
    # arg-parse paths (e.g. --help) that never hit the captor codepath.
    from sndr.model_configs.schema import (
        DockerConfig, HardwareSpec, ModelConfig, SpecDecodeConfig,
    )

    record = _run_inspect(container)
    argv = _full_cmdline(record)
    if not argv:
        raise CaptureError(
            f"container {container!r} has empty Entrypoint+Cmd — is it a "
            "vLLM container? --from-running needs a `vllm serve …` argv"
        )
    flat = " ".join(argv)
    if "vllm" not in flat and "api_server" not in flat:
        raise CaptureError(
            f"container {container!r} does not appear to run vLLM "
            f"(argv head: {' '.join(argv[:6])!r}). --from-running only "
            "supports vllm/vllm-openai derivatives."
        )

    parsed = _parse_serve_args(argv)
    if "model_path" not in parsed:
        raise CaptureError(
            "captured argv is missing --model — cannot reverse-engineer "
            "ModelConfig without a model path"
        )

    container_port = parsed.get("container_port", 8000)
    host_port, container_port = _parse_ports(record, container_port)
    image = (record.get("Config", {}) or {}).get("Image", "") or ""
    digest = _image_digest(record)
    docker_cfg = DockerConfig(
        image=image,
        container_name=container.lstrip("/"),
        port=container_port,
        host_port=host_port if host_port != container_port else None,
        container_port=container_port,
        shm_size=_parse_shm_size(record),
        gpus=_parse_gpus(record),
        mounts=_parse_mounts(record),
        image_digest=digest,
    )

    n_gpus = parsed.get("tensor_parallel_size", 1)
    # gpu_match_keys + min_vram_per_gpu_mib cannot be derived from docker
    # inspect — the operator MUST review both and replace the placeholders
    # with the actual host GPU id (matches `detection/gpu.py` keys:
    # a5000, a100-40gb, h100, rtx-3090, ...) and minimum VRAM. The schema
    # rejects min_vram_per_gpu_mib<=0, so we seed it with 1 (the smallest
    # valid placeholder) — see the "review checklist" printed by the CLI
    # handler so the operator knows to bump it before validate/launch.
    hardware = HardwareSpec(
        gpu_match_keys=["__REPLACE_WITH_HOST_GPU_KEY__"],
        n_gpus=n_gpus,
        min_vram_per_gpu_mib=1,
    )

    genesis_env, system_env = _parse_env(
        (record.get("Config", {}) or {}).get("Env", []) or []
    )

    spec_decode = None
    if parsed.get("spec_decode_json"):
        try:
            spec_obj = json.loads(parsed["spec_decode_json"])
            if isinstance(spec_obj, dict):
                spec_decode = SpecDecodeConfig(
                    method=str(spec_obj.get("method", "ngram")),
                    num_speculative_tokens=int(
                        spec_obj.get("num_speculative_tokens", 1)
                    ),
                    model=spec_obj.get("model"),
                )
        except (json.JSONDecodeError, ValueError, TypeError):
            # Keep the raw flag in vllm_extra_args so launch stays
            # bit-exact even if we can't introspect the spec config.
            parsed.setdefault("vllm_extra_args", []).extend(
                ["--speculative-config", parsed["spec_decode_json"]]
            )

    cfg = ModelConfig(
        key=key,
        title=f"Captured from running container {container}",
        description=(
            f"Auto-captured by `sndr model-config new {key} "
            f"--from-running {container}`. Review GPU match key + image "
            "digest + mounts before promoting to production."
        ),
        schema_version=1,
        maintainer=maintainer,
        model_path=parsed["model_path"],
        hardware=hardware,
        served_model_name=parsed.get("served_model_name"),
        quantization=parsed.get("quantization"),
        kv_cache_dtype=parsed.get("kv_cache_dtype"),
        max_model_len=parsed.get("max_model_len", 32768),
        gpu_memory_utilization=parsed.get("gpu_memory_utilization", 0.90),
        max_num_seqs=parsed.get("max_num_seqs", 2),
        max_num_batched_tokens=parsed.get("max_num_batched_tokens", 4096),
        enable_chunked_prefill=parsed.get("enable_chunked_prefill", True),
        dtype=parsed.get("dtype", "float16"),
        enforce_eager=parsed.get("enforce_eager", False),
        disable_custom_all_reduce=parsed.get("disable_custom_all_reduce", True),
        language_model_only=parsed.get("language_model_only", True),
        trust_remote_code=parsed.get("trust_remote_code", True),
        enable_auto_tool_choice=parsed.get("enable_auto_tool_choice", True),
        tool_call_parser=parsed.get("tool_call_parser"),
        reasoning_parser=parsed.get("reasoning_parser"),
        spec_decode=spec_decode,
        genesis_env=genesis_env,
        system_env=system_env,
        vllm_extra_args=parsed.get("vllm_extra_args", []),
        docker=docker_cfg,
    )
    return cfg


__all__ = ["CaptureError", "capture_from_running"]

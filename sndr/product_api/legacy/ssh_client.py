# SPDX-License-Identifier: Apache-2.0
"""Paramiko-backed SSH/SFTP connectivity for the GUI's remote-host features.

Security boundary — two distinct tiers, deliberately gated differently:

* **Read-only introspection (ungated).** :func:`check_connectivity`,
  :func:`discover_host`, :func:`read_model_config` and :func:`discover_api_key`
  open SSH and run a *fixed, hard-coded* set of read commands (``uname``,
  ``nvidia-smi``, ``docker inspect``/``exec … cat config.json``/``du``). They
  take **no operator-supplied command** — every interpolated value (host,
  container, model path) is validated against a strict allow-list regex
  (:data:`_HOST_RE`, :data:`_CONTAINER_RE`, :data:`_MODEL_PATH_RE`) before it
  reaches a shell, and reads are single-quoted. These are intentionally
  available without ``SNDR_ENABLE_APPLY`` so the GUI can *inspect* a host
  (reachability, GPU topology, the running model's real architecture) while the
  daemon stays read-only. They mutate nothing.
* **Mutation / arbitrary execution (gated).** The PTY terminal, the artifact
  push + command execution (remote install), and the updater run only when
  ``SNDR_ENABLE_APPLY`` is set. Those are the only paths that can change remote
  state or run an operator-chosen command.

Password auth reads the password from the encrypted :mod:`secrets_store` (by
``secret_id``) so it is never round-tripped through the GUI or persisted in
``hosts.json``. The whole module degrades gracefully: without ``paramiko`` the
remote features report unavailable instead of crashing the daemon.
"""
from __future__ import annotations

import base64
import os
import re
import time
from typing import Any, Optional

from . import secrets_store

_HOST_RE = re.compile(r"^[A-Za-z0-9._-]{1,253}$")


def _load_paramiko():
    """Return the paramiko module, or None if the optional dep is absent."""
    try:
        import paramiko

        return paramiko
    except Exception:
        return None


def available() -> bool:
    return _load_paramiko() is not None


def _safe_host(host: str) -> Optional[str]:
    host = (host or "").strip()
    return host if _HOST_RE.match(host) else None


def _resolve_password(target: dict[str, Any]) -> Optional[str]:
    """Explicit password wins; else pull the stored one by secret_id."""
    explicit = target.get("password")
    if explicit:
        return str(explicit)
    secret_id = target.get("secret_id")
    if secret_id:
        try:
            return secrets_store.get_secret(str(secret_id))
        except Exception:
            return None
    return None


def _connect_kwargs(target: dict[str, Any], timeout: float) -> dict[str, Any]:
    host = _safe_host(str(target.get("host", "")))
    try:
        port = int(target.get("port") or 22)
    except (TypeError, ValueError):
        port = 22
    if not (1 <= port <= 65535):
        port = 22
    kwargs: dict[str, Any] = {
        "hostname": host,
        "port": port,
        "username": str(target.get("user") or os.environ.get("USER") or "root"),
        "timeout": timeout,
        "banner_timeout": timeout,
        "auth_timeout": timeout,
        "allow_agent": False,
        "look_for_keys": False,
    }
    method = str(target.get("auth_method") or "agent").lower()
    if method == "password":
        kwargs["password"] = _resolve_password(target) or ""
    elif method == "key":
        key_path = str(target.get("key_path") or "").strip()
        if key_path:
            kwargs["key_filename"] = os.path.expanduser(key_path)
        kwargs["allow_agent"] = True
    else:  # "agent" — rely on the ssh-agent / default keys
        kwargs["allow_agent"] = True
        kwargs["look_for_keys"] = True
    return kwargs


def _open_client(target: dict[str, Any], timeout: float):
    """Connect and return a live paramiko client. Raises on failure."""
    paramiko = _load_paramiko()
    if paramiko is None:
        raise RuntimeError("paramiko not installed")
    client = paramiko.SSHClient()
    try:
        client.load_system_host_keys()  # ~/.ssh/known_hosts + system known_hosts
    except Exception:
        pass
    # Host-key policy. Default is TOFU (AutoAddPolicy): an unknown host is trusted
    # on first connect, but paramiko still REJECTS a CHANGED key for a known host
    # (the real MITM signal). Enterprise deployments that pre-provision
    # known_hosts can set SNDR_SSH_STRICT_HOST_KEYS=1 to reject unknown hosts too.
    _strict = (os.environ.get("SNDR_SSH_STRICT_HOST_KEYS") or "").strip().lower() in ("1", "true", "yes", "on")
    client.set_missing_host_key_policy(
        paramiko.RejectPolicy() if _strict else paramiko.AutoAddPolicy()
    )
    client.connect(**_connect_kwargs(target, timeout))
    return client


def _exec(client, command: str, timeout: float) -> tuple[int, str, str]:
    """Run a (fixed, read-only) command and return (rc, stdout, stderr)."""
    _stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = 0
    try:
        rc = stdout.channel.recv_exit_status()
    except Exception:
        pass
    return rc, out, err


_CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_KEY_RE = re.compile(r"(?:VLLM_API_KEY=|--api[-_]key[= ])['\"]?([A-Za-z0-9._\-]+)")


def discover_api_key(target: dict[str, Any], *, containers: tuple[str, ...] = (), timeout: float = 10.0) -> dict[str, Any]:
    """Read the engine's API key off the host over SSH (read-only discovery).

    Looks first in the running vLLM container's ``VLLM_API_KEY`` env, then falls
    back to grepping the operator's ``start_*.sh`` launch scripts. Runs only
    these fixed, read-only commands — no operator-supplied command, no mutation.
    """
    result: dict[str, Any] = {"available": True, "found": False, "key": None, "source": None, "error": None}
    if _load_paramiko() is None:
        result["available"] = False
        result["error"] = "paramiko not installed — pip install 'vllm-sndr-core[gui-remote]'"
        return result
    if not _safe_host(str(target.get("host", ""))):
        result["error"] = "invalid host"
        return result
    try:
        client = _open_client(target, timeout)
    except Exception as exc:
        result["error"] = _describe(exc)
        return result
    try:
        names = [c for c in containers if _CONTAINER_RE.match(c)]
        if not names:
            rc, out, _ = _exec(client, "docker ps --format '{{.Names}}'", timeout)
            if rc == 0:
                names = [n for n in out.split() if _CONTAINER_RE.match(n) and "vllm" in n.lower()]
        for name in names:
            rc, out, _ = _exec(client, f"docker inspect {name} --format '{{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}'", timeout)
            if rc != 0:
                continue
            for line in out.splitlines():
                if line.startswith("VLLM_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip("'\"")
                    if val:
                        result.update(found=True, key=val, source=f"container:{name}")
                        return result
        # Fallback: launch scripts in the login home.
        rc, out, _ = _exec(client, "grep -rhoE -- '(VLLM_API_KEY=|--api[-_]key[ =])[^\"'\"'\"' ]+' ~/*.sh 2>/dev/null | head -5", timeout)
        if rc == 0 and out.strip():
            for line in out.splitlines():
                m = _KEY_RE.search(line)
                if m:
                    result.update(found=True, key=m.group(1), source="start-script")
                    return result
        result["error"] = "no VLLM_API_KEY found in vLLM containers or ~/*.sh"
        return result
    finally:
        try:
            client.close()
        except Exception:
            pass


_PORT_RE = re.compile(r"(?:\d{1,3}(?:\.\d{1,3}){3}|:::):(\d+)->(\d+)/tcp")


def _engine_port_candidates(ports: str) -> list[int]:
    """Published host ports from a ``docker ps`` Ports string, ordered API-first.

    A vLLM container may publish several ports (OpenAI API, the metrics server,
    custom layouts like ``-p 8102:8102``). The old logic kept a single guess and
    broke on the metrics port (container 8001), so a probe could hit the wrong
    port and report the engine as down. Here we return *every* distinct host
    port, ranked so the OpenAI endpoint is tried first:

      0. mapping whose CONTAINER port is 8000 (the canonical vLLM port)
      1. any other mapping (e.g. a custom ``host:host`` layout)
      2. the metrics mapping (container 8001) last

    The caller probes them in order and keeps the first that answers ``/health``.
    IPv4/IPv6 duplicates of the same mapping collapse to one.
    """
    maps: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for m in _PORT_RE.finditer(ports or ""):
        pair = (int(m.group(1)), int(m.group(2)))
        if pair in seen:
            continue
        seen.add(pair)
        maps.append(pair)

    def _rank(pair: tuple[int, int]) -> tuple[int, int]:
        host_p, container_p = pair
        if container_p == 8000:
            return (0, host_p)
        if container_p == 8001:
            return (2, host_p)
        return (1, host_p)

    ordered: list[int] = []
    hp_seen: set[int] = set()
    for host_p, _container_p in sorted(maps, key=_rank):
        if host_p in hp_seen:
            continue
        hp_seen.add(host_p)
        ordered.append(host_p)
    return ordered


def discover_host(target: dict[str, Any], *, timeout: float = 12.0) -> dict[str, Any]:
    """Auto-discover what's running on a host over SSH (read-only).

    Finds vLLM docker containers and their published host ports, the image tag,
    and GPUs — so the operator doesn't have to know which port the engine is on.
    Runs only fixed, read-only discovery commands.
    """
    result: dict[str, Any] = {"available": True, "docker": False, "engines": [], "gpus": [], "error": None}
    if _load_paramiko() is None:
        result["available"] = False
        result["error"] = "paramiko not installed — pip install 'vllm-sndr-core[gui-remote]'"
        return result
    if not _safe_host(str(target.get("host", ""))):
        result["error"] = "invalid host"
        return result
    try:
        client = _open_client(target, timeout)
    except Exception as exc:
        result["error"] = _describe(exc)
        return result
    try:
        # Containers: name | ports | image | status. Tab-separated, robust to spaces.
        rc, out, _ = _exec(client, "docker ps --format '{{.Names}}\t{{.Ports}}\t{{.Image}}\t{{.Status}}'", timeout)
        if rc == 0:
            result["docker"] = True
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                name, ports, image, status = parts[0], parts[1], parts[2], parts[3]
                if "vllm" not in (name + " " + image).lower():
                    continue
                # Every published host port, API-first; the caller probes each so
                # a custom/metrics mapping can't hide a running engine.
                candidates = _engine_port_candidates(ports)
                entry = {"container": name, "host_port": candidates[0] if candidates else None,
                         "host_ports": candidates, "image": image,
                         "status": status, "ports": ports}
                # Active Genesis patch flags on the running container — the real
                # runtime patch state of this host ("what's actually enabled").
                if _CONTAINER_RE.match(name):
                    ev_rc, ev_out, _ = _exec(
                        client, f"docker inspect {name} --format '{{{{range .Config.Env}}}}{{{{println .}}}}{{{{end}}}}'", timeout)
                    if ev_rc == 0:
                        entry["genesis_flags"] = [
                            ln.split("=", 1)[0] for ln in ev_out.splitlines()
                            if ln.startswith("GENESIS_") and ln.rstrip().endswith("=1")
                        ][:60]
                result["engines"].append(entry)
        # GPUs (best-effort) — name, VRAM, utilisation, compute capability.
        rc, out, _ = _exec(client, "nvidia-smi --query-gpu=name,memory.total,utilization.gpu,compute_cap --format=csv,noheader,nounits", timeout)
        if rc == 0 and out.strip():
            from . import gpu_arch

            for line in out.splitlines():
                cells = [c.strip() for c in line.split(",")]
                if len(cells) >= 2:
                    cap = cells[3] if len(cells) > 3 else None
                    cls = gpu_arch.classify(name=cells[0], compute_cap=cap)
                    result["gpus"].append({
                        "name": cells[0], "memory_total_mib": cells[1],
                        "utilization": cells[2] if len(cells) > 2 else None,
                        "compute_cap": cls["compute_cap"], "arch": cls["arch"],
                    })
            # One arch advisory for the rig (GPUs are homogeneous in practice).
            if result["gpus"]:
                first = result["gpus"][0]
                result["arch_advice"] = gpu_arch.classify(name=first["name"], compute_cap=first.get("compute_cap"))
        # Interconnect topology (NVLink vs PCIe) — decisive for dual-card perf.
        if len(result["gpus"]) > 1:
            rc, out, _ = _exec(client, "nvidia-smi topo -m", timeout)
            if rc == 0 and out:
                result["interconnect"] = _summarize_topology(out)
                result["topology_raw"] = out.strip()[:2000]
        if not result["docker"] and not result["gpus"]:
            result["error"] = "no docker / nvidia-smi on host (or not permitted)"
        return result
    finally:
        try:
            client.close()
        except Exception:
            pass


def _summarize_topology(topo: str) -> dict[str, Any]:
    """Reduce ``nvidia-smi topo -m`` to a link verdict (NVLink / PCIe / mixed).

    The matrix encodes the *best* path between GPU pairs: ``NV#`` = NVLink,
    ``PIX``/``PXB``/``PHB`` = PCIe (closer→farther), ``NODE``/``SYS`` = cross
    NUMA/socket. The slowest GPU↔GPU link bounds tensor-parallel bandwidth.
    """
    links: list[str] = []
    for line in topo.splitlines():
        if not line.startswith("GPU"):
            continue
        for tok in line.split()[1:]:
            if re.fullmatch(r"NV\d+", tok):
                links.append("NVLink")
            elif tok in ("PIX", "PXB", "PHB"):
                links.append("PCIe")
            elif tok in ("NODE", "SYS"):
                links.append("cross-NUMA")
    nvlink = any(l == "NVLink" for l in links)
    worst = "cross-NUMA" if "cross-NUMA" in links else ("PCIe" if "PCIe" in links else ("NVLink" if nvlink else "single"))
    return {
        "has_nvlink": nvlink,
        "worst_link": worst,
        "note": (
            "NVLink present — tensor-parallel bandwidth is good." if nvlink
            else "No NVLink — GPU↔GPU runs over PCIe; link width (x4/x8/x16) bounds TP throughput." if worst == "PCIe"
            else "GPUs span NUMA nodes/sockets — expect the weakest cross-card bandwidth." if worst == "cross-NUMA"
            else "Single interconnect."
        ),
    }


_MODEL_PATH_RE = re.compile(r"^(?:/|~)[\w./\-]+$")


def _extract_model_path(joined: str) -> Optional[str]:
    """Pull the served model path from a container's space-joined command tokens.

    Handles ``--model <path>``, ``vllm serve <path>`` and a bare ``/models/...``
    directory token.
    """
    tokens = joined.split()
    for i, t in enumerate(tokens):
        if t in ("--model", "--model-path", "serve") and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt.startswith(("/", "~")):
                return nxt
    for t in tokens:
        if t.startswith("/") and "/" in t[1:] and not t.endswith((".py", ".sh", ".json", ".yaml")):
            return t
    return None


def _arch_from_hf_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract transformer dims from an HF ``config.json`` (handles the nested
    ``text_config`` used by multimodal models like Gemma)."""
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}

    def g(key: str, default: Any = None) -> Any:
        return cfg.get(key, tc.get(key, default))

    n_heads = g("num_attention_heads") or 0
    hidden = g("hidden_size") or 0
    head_dim = g("head_dim") or (int(hidden // n_heads) if n_heads else 128)
    experts = g("num_experts") or g("num_local_experts") or g("num_routed_experts") or g("n_routed_experts")
    quant = cfg.get("quantization_config", {})
    num_layers = int(g("num_hidden_layers") or 0)

    # Sliding-window attention: count the *global* (full-attention) layers so the
    # calculator knows long-context KV only grows on those.
    sliding_window = g("sliding_window") or 0
    global_layers = None
    layer_types = g("layer_types")
    pattern = g("sliding_window_pattern")
    if isinstance(layer_types, list) and layer_types:
        global_layers = sum(1 for t in layer_types if "full" in str(t).lower() or "global" in str(t).lower())
    elif isinstance(pattern, int) and pattern > 1 and num_layers:
        global_layers = max(1, num_layers // pattern)  # 1 global per `pattern` layers

    return {
        "num_layers": num_layers,
        "num_attention_heads": int(n_heads),
        "num_kv_heads": int(g("num_key_value_heads") or n_heads or 8),
        "head_dim": int(head_dim or 128),
        "hidden_size": int(hidden),
        "is_moe": bool(experts),
        "num_experts": int(experts) if experts else 0,
        "max_context": int(g("max_position_embeddings") or 0),
        "sliding_window": int(sliding_window) if sliding_window else 0,
        "global_layers": int(global_layers) if global_layers is not None else None,
        "quant_method": str(quant.get("quant_method") or quant.get("quantization") or "") if isinstance(quant, dict) else "",
        "model_type": str(g("model_type") or ""),
    }


def read_model_config(target: dict[str, Any], *, container: str, timeout: float = 15.0) -> dict[str, Any]:
    """Read the running model's real architecture from the host over SSH.

    Pulls ``config.json`` (exact KV/GQA/MoE dims + native context window) and the
    on-disk weight size via ``du`` (the exact resident weight bytes). This makes
    the fit calculator size from reality instead of curated guesses.
    """
    result: dict[str, Any] = {"ok": False, "error": None}
    if _load_paramiko() is None:
        result["error"] = "paramiko not installed"
        return result
    if not _CONTAINER_RE.match(container or ""):
        result["error"] = "invalid container name"
        return result
    try:
        client = _open_client(target, timeout)
    except Exception as exc:
        result["error"] = _describe(exc)
        return result
    try:
        rc, out, _ = _exec(client, f"docker inspect {container} --format '{{{{join .Config.Entrypoint \" \"}}}} {{{{join .Config.Cmd \" \"}}}} {{{{join .Args \" \"}}}}'", timeout)
        path = _extract_model_path(out) if rc == 0 else None
        if not path:
            # The container often runs a launcher script — the real --model lives
            # in the vLLM process. Scan every process command line.
            rc2, procs, _ = _exec(client, f"docker exec {container} sh -c 'for f in /proc/[0-9]*/cmdline; do tr \"\\0\" \" \" < \"$f\" 2>/dev/null; echo; done'", timeout)
            if rc2 == 0:
                for line in procs.splitlines():
                    p = _extract_model_path(line)
                    if p:
                        path = p
                        break
        if not path or not _MODEL_PATH_RE.match(path):
            result["error"] = "could not locate the model path (container command or process args)"
            return result
        rc, cfg_txt, _ = _exec(client, f"docker exec {container} cat '{path}/config.json'", timeout)
        if rc != 0 or not cfg_txt.strip():
            result["error"] = f"config.json not readable at {path}"
            return result
        import json as _json

        try:
            cfg = _json.loads(cfg_txt)
        except Exception:
            result["error"] = "config.json is not valid JSON"
            return result
        arch = _arch_from_hf_config(cfg)
        # Exact resident weight size from the model files.
        rc, du, _ = _exec(client, f"docker exec {container} du -sb '{path}' 2>/dev/null", timeout)
        weights_bytes = None
        if rc == 0 and du.split():
            try:
                weights_bytes = int(du.split()[0])
            except (ValueError, IndexError):
                weights_bytes = None
        result.update(ok=True, model_path=path, weights_bytes=weights_bytes, **arch)
        return result
    finally:
        try:
            client.close()
        except Exception:
            pass


# In-container introspection of the server's own sndr_core install — patcher
# version, vLLM build, builtin config count and patch-registry size. Base64'd so
# the script crosses SSH + docker exec + the shell without any quoting hazard.
_SNDR_PROBE = b"""
import json
o = {}
try:
    import vllm
    o["vllm"] = vllm.__version__
except Exception:
    o["vllm"] = None
try:
    from importlib.metadata import version
    o["sndr"] = version("vllm-sndr-core")
except Exception:
    try:
        import vllm.sndr_core as s
        o["sndr"] = getattr(s, "__version__", None)
    except Exception:
        o["sndr"] = None
try:
    import os, glob
    import vllm.sndr_core as s2
    d = os.path.join(os.path.dirname(s2.__file__), "model_configs", "builtin")
    o["configs"] = len(glob.glob(d + "/**/*.yaml", recursive=True))
except Exception:
    o["configs"] = None
try:
    from vllm.sndr_core.dispatcher.spec import iter_patch_specs
    o["patches"] = sum(1 for _ in iter_patch_specs())
except Exception:
    o["patches"] = None
print(json.dumps(o))
"""


def read_sndr_state(target: dict[str, Any], *, container: Optional[str] = None, timeout: float = 15.0) -> dict[str, Any]:
    """Read the server's own sndr_core state from inside the running container
    over SSH (read-only): patcher version, vLLM build, builtin config count and
    patch-registry size. This is the 'light Path B' — the operator sees a host's
    management identity without standing up a daemon on it."""
    result: dict[str, Any] = {"ok": False, "container": None, "vllm_version": None,
                              "sndr_version": None, "configs": None, "patches": None, "error": None}
    if _load_paramiko() is None:
        result["error"] = "paramiko not installed"
        return result
    if container is not None and not _CONTAINER_RE.match(container or ""):
        result["error"] = "invalid container name"
        return result
    try:
        client = _open_client(target, timeout)
    except Exception as exc:
        result["error"] = _describe(exc)
        return result
    try:
        name = container
        if not name:
            rc, out, _ = _exec(client, "docker ps --format '{{.Names}}'", timeout)
            if rc == 0:
                for n in out.split():
                    if _CONTAINER_RE.match(n) and "vllm" in n.lower():
                        name = n
                        break
        if not name:
            result["error"] = "no vLLM container found"
            return result
        result["container"] = name
        b64 = base64.b64encode(_SNDR_PROBE).decode("ascii")
        rc, out, _ = _exec(client, f"docker exec {name} sh -c 'echo {b64} | base64 -d | python3 -'", timeout)
        if rc != 0 or not out.strip():
            result["error"] = "sndr_core introspection failed in container"
            return result
        import json as _json

        data = None
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = _json.loads(line)
                    break
                except Exception:
                    continue
        if data is None:
            result["error"] = "unparseable introspection output"
            return result
        result.update(ok=True, vllm_version=data.get("vllm"), sndr_version=data.get("sndr"),
                      configs=data.get("configs"), patches=data.get("patches"))
        return result
    finally:
        try:
            client.close()
        except Exception:
            pass


_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_WORKDIR_RE = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")


def run_apply(
    target: dict[str, Any],
    *,
    artifact_name: str,
    artifact_content: str,
    commands: list[str],
    extra_files: Optional[list[tuple[str, bytes]]] = None,
    workdir: str = "sndr-install",
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Execute an install plan on a host over SSH (MUTATING — gate before call).

    SFTPs ``artifact_content`` to ``workdir/artifact_name`` (plus any binary
    ``extra_files`` — e.g. a code bundle), then runs each command from that
    directory, capturing rc + output per step. Stops at the first failing
    command. The HTTP layer must check ``SNDR_ENABLE_APPLY`` and an explicit
    confirm before invoking this — it changes remote state.
    """
    import io

    result: dict[str, Any] = {"ok": False, "steps": [], "error": None}
    if _load_paramiko() is None:
        result["error"] = "paramiko not installed"
        return result
    if not _safe_host(str(target.get("host", ""))):
        result["error"] = "invalid host"
        return result
    if not _ARTIFACT_NAME_RE.match(artifact_name or ""):
        result["error"] = "invalid artifact name"
        return result
    if not _WORKDIR_RE.match(workdir or ""):
        result["error"] = "invalid workdir"
        return result
    try:
        client = _open_client(target, timeout)
    except Exception as exc:
        result["error"] = _describe(exc)
        return result
    try:
        _exec(client, f"mkdir -p {workdir}", timeout)
        try:
            sftp = client.open_sftp()
            sftp.putfo(io.BytesIO(artifact_content.encode("utf-8")), f"{workdir}/{artifact_name}")
            uploaded = [artifact_name]
            for name, content in (extra_files or []):
                if not _ARTIFACT_NAME_RE.match(name or ""):
                    sftp.close()
                    result["error"] = f"invalid extra-file name: {name}"
                    return result
                data = content if isinstance(content, (bytes, bytearray)) else str(content).encode("utf-8")
                sftp.putfo(io.BytesIO(data), f"{workdir}/{name}")
                uploaded.append(name)
            sftp.close()
            result["steps"].append({"cmd": f"upload {', '.join(uploaded)}", "rc": 0,
                                    "output": f"sent {len(uploaded)} file(s) to {workdir}/"})
        except Exception as exc:
            result["error"] = f"SFTP upload failed: {_describe(exc)}"
            return result
        for cmd in commands or []:
            rc, out, err = _exec(client, f"cd {workdir} && {cmd}", timeout)
            combined = (out + ("\n" + err if err.strip() else "")).strip()
            result["steps"].append({"cmd": cmd, "rc": rc, "output": combined[:6000]})
            if rc != 0:
                result["error"] = f"command failed (exit {rc}): {cmd}"
                return result
        result["ok"] = True
        return result
    finally:
        try:
            client.close()
        except Exception:
            pass


def open_shell(target: dict[str, Any], *, term: str = "xterm-256color", cols: int = 120, rows: int = 32, timeout: float = 15.0):
    """Open an interactive PTY shell on the host. Returns (client, channel).

    Caller owns both and must close them. This is full remote shell access —
    the websocket route that uses it is gated behind ``SNDR_ENABLE_APPLY``.
    """
    client = _open_client(target, timeout)
    channel = client.invoke_shell(term=term, width=int(cols) or 120, height=int(rows) or 32)
    channel.settimeout(0.0)
    return client, channel


def read_nonblocking(channel) -> Optional[bytes]:
    """Drain ready PTY output without blocking. ``None`` = nothing yet, ``b""`` = closed."""
    try:
        if channel.recv_ready():
            return channel.recv(65536)
        if getattr(channel, "closed", False) or channel.eof_received:
            return b""
        return None
    except Exception:
        return b""


def check_connectivity(target: dict[str, Any], *, timeout: float = 8.0) -> dict[str, Any]:
    """Probe SSH auth + SFTP for a host. Read-only: no operator command runs."""
    result: dict[str, Any] = {
        "available": True,
        "ssh_ok": False,
        "sftp_ok": False,
        "latency_ms": None,
        "banner": None,
        "uname": None,
        "error": None,
    }
    paramiko = _load_paramiko()
    if paramiko is None:
        result["available"] = False
        result["error"] = "paramiko not installed — pip install 'vllm-sndr-core[gui-remote]'"
        return result

    host = _safe_host(str(target.get("host", "")))
    if not host:
        result["error"] = "invalid host"
        return result

    started = time.monotonic()
    try:
        client = _open_client(target, timeout)
        result["ssh_ok"] = True
        result["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
    except Exception as exc:  # auth / network / timeout
        result["error"] = _describe(exc)
        return result

    # Best-effort enrichment — failures here don't flip ssh_ok.
    try:
        transport = client.get_transport()
        banner = transport.get_banner() if transport else None
        if banner:
            result["banner"] = banner.decode("utf-8", "replace").strip() if isinstance(banner, bytes) else str(banner)
    except Exception:
        pass
    try:
        _stdin, stdout, _stderr = client.exec_command("uname -sr", timeout=timeout)
        result["uname"] = stdout.read().decode("utf-8", "replace").strip() or None
    except Exception:
        pass
    try:
        sftp = client.open_sftp()
        sftp.normalize(".")
        result["sftp_ok"] = True
        try:
            sftp.close()
        except Exception:
            pass
    except Exception:
        result["sftp_ok"] = False

    try:
        client.close()
    except Exception:
        pass
    return result


def _describe(exc: Exception) -> str:
    msg = str(exc) or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {msg}" if msg == exc.__class__.__name__ else msg


__all__ = ["available", "check_connectivity", "discover_api_key", "discover_host", "open_shell", "read_model_config", "read_nonblocking"]

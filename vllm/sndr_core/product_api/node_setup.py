# SPDX-License-Identifier: Apache-2.0
"""One-button node setup — turn a GPU/engine host into a managed cluster node.

The fleet/cluster model (Path B) needs each node to run the SNDR management
daemon (the Product API) serving THAT node's native catalog/patches/configs, so
the central GUI can switch to it. Doing that by hand is fragile (ship the daemon
code, refresh bytecode, run a sidecar from the vLLM image with the right
mounts/env, bind the LAN, enable auth + CORS). This module collapses all of it
into a single gated SSH apply:

  1. Bundle the ``product_api`` package (.py only) on the central daemon.
  2. SFTP the bundle + a self-contained ``setup-node.sh`` to the node.
  3. The script unpacks the daemon code into the node's mounted ``sndr_core``,
     clears stale bytecode, and runs the management daemon as a sidecar of the
     running vLLM engine (same image → has sndr_core deps; LAN-bound; auth on;
     CORS open for the central GUI), then health-checks it.

Double-gated like every apply: ``SNDR_ENABLE_APPLY`` AND an explicit confirm.
``run_apply`` is injected so the orchestration is unit-testable without SSH.
"""
from __future__ import annotations

import io
import os
import tarfile
from typing import Any, Callable, Optional

_BUNDLE_NAME = "sndr-daemon-bundle.tar.gz"


def node_bundle() -> bytes:
    """tar.gz of the daemon code + its data, shipped to a node so its management
    daemon runs a CONSISTENT set: the ``product_api`` package (.py — the API) AND
    ``model_configs`` (.py + .yaml — the preset/config corpus the API reads). They
    must match: fresh API code against a node's stale corpus crashes the catalog
    (the 500 we hit). Arcnames are relative to ``sndr_core`` so the node script
    unpacks both into place. Excludes ``web_static`` (the central GUI is the UI)
    and ``__pycache__`` (stale bytecode must not travel)."""
    product_api_root = os.path.dirname(os.path.abspath(__file__))  # .../sndr_core/product_api
    sndr_core_root = os.path.dirname(product_api_root)              # .../sndr_core
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("product_api", "model_configs"):
            base = os.path.join(sndr_core_root, sub)
            if not os.path.isdir(base):
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "web_static")]
                for name in sorted(filenames):
                    if not name.endswith((".py", ".yaml", ".yml")):
                        continue
                    full = os.path.join(dirpath, name)
                    arc = os.path.relpath(full, sndr_core_root)  # e.g. product_api/http_app.py
                    tar.add(full, arcname=arc)
    return buf.getvalue()


# Back-compat alias.
product_api_bundle = node_bundle


def _sh_squote(value: str) -> str:
    """Safely single-quote a value for embedding in a bash script."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def setup_node_script(*, port: int = 8765, engine_port: int = 8102,
                      admin_password: str = "", allow_all_origins: bool = True) -> str:
    """Self-contained node bootstrap: deploy the daemon code into the node's
    sndr_core, refresh bytecode, run the management daemon sidecar, health-check."""
    allow = "1" if allow_all_origins else "0"
    pw = _sh_squote(admin_password)
    return f"""#!/usr/bin/env bash
# One-button SNDR node setup — deploy the management daemon onto this engine host.
set -euo pipefail
PORT={int(port)}; ENGINE_PORT={int(engine_port)}; NAME=sndr-daemon
ENGINE=$(docker ps --filter name=vllm --format '{{{{.Names}}}}' | head -1)
[ -z "$ENGINE" ] && {{ echo "[setup] no running vLLM container found"; exit 1; }}
IMAGE=$(docker inspect "$ENGINE" --format '{{{{.Config.Image}}}}')
# Source (host) + Destination (container) of the engine's sndr_core mount.
SRC=$(docker inspect "$ENGINE" --format '{{{{range .Mounts}}}}{{{{.Source}}}} {{{{.Destination}}}}{{{{println}}}}{{{{end}}}}' | awk '$2 ~ /\\/vllm\\/sndr_core$/ {{print $1}}' | head -1)
DST=$(docker inspect "$ENGINE" --format '{{{{range .Mounts}}}}{{{{.Source}}}} {{{{.Destination}}}}{{{{println}}}}{{{{end}}}}' | awk '$2 ~ /\\/vllm\\/sndr_core$/ {{print $2}}' | head -1)
[ -z "$SRC" ] && {{ echo "[setup] could not find the sndr_core mount on $ENGINE"; exit 1; }}
echo "[setup] engine=$ENGINE image=$IMAGE"
echo "[setup] sndr_core: $SRC -> $DST"
# 1) Deploy the daemon code + corpus into the node's sndr_core (consistent set)
#    and drop stale bytecode so the fresh .py is what runs (mounted read-only).
tar -xzf {_BUNDLE_NAME} -C "$SRC"
find "$SRC/product_api" "$SRC/model_configs" -name '*.pyc' -delete 2>/dev/null || true
find "$SRC/product_api" "$SRC/model_configs" -name '__pycache__' -type d -prune -exec rm -rf {{}} + 2>/dev/null || true
echo "[setup] daemon code + corpus deployed"
# 2) Run the management daemon as a sidecar from the engine image (it has the
#    sndr_core deps). LAN-bound so the central GUI can switch straight to it.
#    We mount /var/run/docker.sock so the daemon can report the host's docker and
#    manage the engine containers (scoped to a whitelist). SNDR_ENABLE_EXEC is
#    deliberately NOT set — in-container exec stays off until the operator opts in.
docker rm -f "$NAME" >/dev/null 2>&1 || true
# --gpus gives the daemon READ access to nvidia-smi (real card data in the
# inventory, fast) — it only queries, never allocates CUDA, so it does NOT
# reserve GPU memory. We TRY --gpus all and fall back to a CPU-only run if the
# nvidia container runtime isn't wired the way we guessed: detecting GPU support
# by parsing `docker info` is unreliable (the toolkit often works via a prestart
# hook on the default runtime without registering a named "nvidia" runtime), so
# the only robust signal is whether `docker run --gpus all` actually succeeds.
run_daemon() {{  # $1 = extra run args (e.g. "--gpus all" or "")
  docker run -d --name "$NAME" --restart unless-stopped --network host $1 \\
    --entrypoint python3 \\
    -e SNDR_ADMIN_PASSWORD={pw} \\
    -e SNDR_ENABLE_APPLY=1 \\
    -e SNDR_ALLOW_ALL_ORIGINS={allow} \\
    -e SNDR_RUNTIME_HOST=127.0.0.1 \\
    -e SNDR_OPENAI_BASE_URL=http://127.0.0.1:$ENGINE_PORT/v1 \\
    -e SNDR_METRICS_URL=http://127.0.0.1:$ENGINE_PORT/metrics \\
    -e PYTHONDONTWRITEBYTECODE=1 \\
    -v "$SRC":"$DST":ro \\
    -v /var/run/docker.sock:/var/run/docker.sock \\
    "$IMAGE" \\
    -c "from vllm.sndr_core.product_api.http_app import run_server; run_server(host='0.0.0.0', port=$PORT)"
}}
if run_daemon "--gpus all" 2>/tmp/sndr_run_err; then
  echo "[setup] daemon started with GPU access (--gpus all) — inventory will show real cards"
else
  echo "[setup] --gpus all not supported here, retrying CPU-only (inventory GPU data unavailable):"
  sed 's/^/[setup]   /' /tmp/sndr_run_err 2>/dev/null | tail -3
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  run_daemon ""
fi
# 3) Health-check.
sleep 6
docker ps --filter name="$NAME" --format 'STATUS: {{{{.Status}}}}'
if curl -sf "http://127.0.0.1:$PORT/api/v1/health" >/dev/null 2>&1; then
  echo "[setup] OK — daemon healthy on 0.0.0.0:$PORT (login: root)"
else
  echo "[setup] daemon not healthy yet — recent logs:"; docker logs --tail 25 "$NAME" 2>&1 | tail -25
fi
"""


def setup_node(
    *,
    ssh_target: dict[str, Any],
    run_apply: Callable[..., dict[str, Any]],
    apply_enabled: bool,
    confirm: bool,
    admin_password: str,
    port: int = 8765,
    engine_port: int = 8102,
    allow_all_origins: bool = True,
) -> dict[str, Any]:
    """Deploy + run the management daemon on a node in one gated SSH apply."""
    if not apply_enabled:
        return {"ok": False, "applied": False,
                "error": "apply is disabled — start the daemon with SNDR_ENABLE_APPLY=1"}
    if not confirm:
        return {"ok": False, "applied": False, "error": "explicit confirm is required"}
    if not admin_password or len(admin_password) < 4:
        return {"ok": False, "applied": False, "error": "an admin password (min 4 chars) is required"}

    script = setup_node_script(port=port, engine_port=engine_port,
                               admin_password=admin_password, allow_all_origins=allow_all_origins)
    result = run_apply(
        ssh_target,
        artifact_name="setup-node.sh",
        artifact_content=script,
        commands=["chmod +x setup-node.sh", "./setup-node.sh"],
        extra_files=[(_BUNDLE_NAME, node_bundle())],
    )
    return {
        "ok": bool(result.get("ok")),
        "applied": True,
        "port": port,
        "steps": result.get("steps", []),
        "error": result.get("error"),
    }


__all__ = ["node_bundle", "product_api_bundle", "setup_node_script", "setup_node"]

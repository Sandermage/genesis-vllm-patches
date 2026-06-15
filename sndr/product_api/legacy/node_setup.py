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
  3. The script unpacks the daemon code into the node's mounted ``sndr``
     package, clears stale bytecode, and runs the management daemon as a sidecar
     of the running vLLM engine (same image → has the sndr deps; LAN-bound; auth
     on; CORS open for the central GUI), then health-checks it.

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
    """tar.gz of the canonical ``sndr`` package, shipped to a node so its
    management daemon runs a CONSISTENT set with the central one.

    Post-v12 the daemon imports the canonical top-level ``sndr`` package
    (``sndr.product_api.legacy.http_app`` + its corpus under ``sndr.model_configs``),
    mounted into the engine's site-packages at ``dist-packages/sndr`` at runtime.
    So the node needs that exact ``sndr/`` code (.py + .yaml/.yml), else fresh API
    code runs against a node's stale corpus and the catalog 500s. Arcnames are
    relative to the repo root (``sndr/...``) so the node script unpacks it into
    the host repo root that backs the ``dist-packages/sndr`` mount. Excludes
    ``web_static`` (the central GUI is the UI) and ``__pycache__`` (stale
    bytecode must not travel)."""
    # node_setup.py = <repo>/sndr/product_api/legacy/node_setup.py
    here = os.path.dirname(os.path.abspath(__file__))   # .../sndr/product_api/legacy
    sndr_pkg = os.path.dirname(os.path.dirname(here))   # .../sndr  (canonical package root)
    repo_root = os.path.dirname(sndr_pkg)               # repo root (contains sndr/ + vllm/)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(sndr_pkg):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "web_static")]
            for name in sorted(filenames):
                if not name.endswith((".py", ".yaml", ".yml")):
                    continue
                full = os.path.join(dirpath, name)
                arc = os.path.relpath(full, repo_root)  # e.g. sndr/product_api/legacy/http_app.py
                tar.add(full, arcname=arc)
    return buf.getvalue()


# Back-compat alias.
product_api_bundle = node_bundle


def _sh_squote(value: str) -> str:
    """Safely single-quote a value for embedding in a bash script."""
    return "'" + str(value).replace("'", "'\\''") + "'"


# The daemon launcher, shipped base64 (avoids shell-quoting hell now that the
# entrypoint is `sh -c` — which lets us conditionally install the k8s client when
# a kubeconfig is mounted). Reads bind/port/apply from env.
_DAEMON_LAUNCHER = (
    b"import os\n"
    b"from sndr.product_api.legacy.http_app import run_server\n"
    b"run_server(host=os.environ.get('SNDR_BIND', '0.0.0.0'), "
    b"port=int(os.environ.get('SNDR_GUI_PORT') or 8765), "
    b"enable_apply=bool(os.environ.get('SNDR_ENABLE_APPLY')))\n"
)


def setup_node_script(*, port: int = 8765, engine_port: int = 8102,
                      admin_password: str = "", allow_all_origins: bool = True) -> str:
    """Self-contained node bootstrap: deploy the daemon code into the node's
    sndr package, refresh bytecode, run the management daemon sidecar, health-check."""
    import base64
    allow = "1" if allow_all_origins else "0"
    pw = _sh_squote(admin_password)
    launcher_b64 = base64.b64encode(_DAEMON_LAUNCHER).decode("ascii")
    return f"""#!/usr/bin/env bash
# One-button SNDR node setup — deploy the management daemon onto this engine host.
set -euo pipefail
PORT={int(port)}; ENGINE_PORT={int(engine_port)}; NAME=sndr-daemon
ENGINE=$(docker ps --filter name=vllm --format '{{{{.Names}}}}' | head -1)
[ -z "$ENGINE" ] && {{ echo "[setup] no running vLLM container found"; exit 1; }}
IMAGE=$(docker inspect "$ENGINE" --format '{{{{.Config.Image}}}}')
# Source (host) + Destination (container) of the engine's sndr mount.
SRC=$(docker inspect "$ENGINE" --format '{{{{range .Mounts}}}}{{{{.Source}}}} {{{{.Destination}}}}{{{{println}}}}{{{{end}}}}' | awk '$2 ~ /\\/dist-packages\\/sndr$/ {{print $1}}' | head -1)
DST=$(docker inspect "$ENGINE" --format '{{{{range .Mounts}}}}{{{{.Source}}}} {{{{.Destination}}}}{{{{println}}}}{{{{end}}}}' | awk '$2 ~ /\\/dist-packages\\/sndr$/ {{print $2}}' | head -1)
[ -z "$SRC" ] && {{ echo "[setup] could not find the sndr mount on $ENGINE"; exit 1; }}
echo "[setup] engine=$ENGINE image=$IMAGE"
echo "[setup] sndr: $SRC -> $DST"
# v12: the canonical package is the top-level sndr/, mounted into the engine's
# site-packages at dist-packages/sndr at runtime. Derive the host repo root +
# container site-packages from that mount and (re)deploy the canonical sndr/
# package so `import sndr` resolves — else the daemon dies on
# `from sndr.version import ...` (No module named 'sndr').
REPO_ROOT=$(dirname "$SRC")
SITE_PKGS=$(dirname "$DST")
SNDR_SRC="$REPO_ROOT/sndr"
SNDR_DST="$SITE_PKGS/sndr"
# 1) Deploy the canonical sndr/ package (consistent set) next to vllm/ and drop
#    stale bytecode so the fresh .py is what runs (mounted read-only).
tar -xzf {_BUNDLE_NAME} -C "$REPO_ROOT"
[ -d "$SNDR_SRC" ] || {{ echo "[setup] sndr/ package missing at $SNDR_SRC after unpack"; exit 1; }}
find "$SNDR_SRC" -name '*.pyc' -delete 2>/dev/null || true
find "$SNDR_SRC" -name '__pycache__' -type d -prune -exec rm -rf {{}} + 2>/dev/null || true
echo "[setup] canonical sndr/ deployed: $SNDR_SRC -> $SNDR_DST"
# 2) Run the management daemon as a sidecar from the engine image (it has the
#    sndr deps). LAN-bound so the central GUI can switch straight to it.
#    We mount /var/run/docker.sock so the daemon can report the host's docker and
#    manage the engine containers (scoped to a whitelist). SNDR_ENABLE_EXEC is
#    deliberately NOT set — in-container exec stays off until the operator opts in.
docker rm -f "$NAME" >/dev/null 2>&1 || true
# Kubernetes mode: if the host has a kubeconfig, mount it so the daemon can serve
# the GUI's read-only k8s view (nodes/pods/events) + deploy tab. The launcher
# then pip-installs the k8s client on start (only when the kubeconfig is present,
# so non-k8s hosts pay nothing). This makes k8s a persistent part of the panel —
# it survives container recreate, not just restart.
KUBE_OPT=""
KUBECFG=$(ls /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config" 2>/dev/null | head -1 || true)
[ -n "$KUBECFG" ] && {{ KUBE_OPT="-v $KUBECFG:/root/.kube/config:ro"; echo "[setup] kubeconfig found ($KUBECFG) — k8s mode will be enabled"; }}
# --gpus gives the daemon READ access to nvidia-smi (real card data in the
# inventory, fast) — it only queries, never allocates CUDA, so it does NOT
# reserve GPU memory. We TRY --gpus all and fall back to a CPU-only run if the
# nvidia container runtime isn't wired the way we guessed.
run_daemon() {{  # $1 = extra run args (e.g. "--gpus all" or "")
  docker run -d --name "$NAME" --restart unless-stopped --network host $1 $KUBE_OPT \\
    --entrypoint sh \\
    -e SNDR_ADMIN_PASSWORD={pw} \\
    -e SNDR_ENABLE_APPLY=1 \\
    -e SNDR_ALLOW_ALL_ORIGINS={allow} \\
    -e SNDR_RUNTIME_HOST=127.0.0.1 \\
    -e SNDR_GUI_PORT=$PORT \\
    -e SNDR_OPENAI_BASE_URL=http://127.0.0.1:$ENGINE_PORT/v1 \\
    -e SNDR_METRICS_URL=http://127.0.0.1:$ENGINE_PORT/metrics \\
    -e PYTHONDONTWRITEBYTECODE=1 \\
    -v "$SRC":"$DST":ro \\
    -v "$SNDR_SRC":"$SNDR_DST":ro \\
    -v /var/run/docker.sock:/var/run/docker.sock \\
    "$IMAGE" \\
    -c "[ -f /root/.kube/config ] && pip install -q kubernetes 2>/dev/null || true; echo {launcher_b64} | base64 -d | python3 -"
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

# SNDR Control Center (GUI)

The SNDR Control Center is a read-only operator dashboard over the SNDR Product
API. It turns the CLI/runtime/patch/config/evidence surface into a browsable
control plane: catalog, preset recommendation, launch planning, diagnostics,
patches, benchmarks, evidence, clients, and reports.

The GUI is a **client of a stable API**, never a wrapper that parses CLI stdout.
The same Product API backs the CLI and the GUI.

```text
React + TypeScript UI  ->  sndr gui-api (FastAPI daemon)  ->  Product API  ->  sndr_core
```

## What it does today

- The catalog/patch/config/evidence surface is **read-only by design**: it never
  writes V2 YAML, the patch registry, or runtime artifacts.
- Service-action and launch flows are **plan-before-apply**: applies produce
  dry-run jobs with the exact commands an operator would run unless apply-mode is
  explicitly enabled.
- **Container & engine management** (the Containers screen) is a separate,
  opt-in capability: with the docker socket mounted it can list, inspect, and
  run lifecycle actions (start/stop/restart/recreate) on engine containers, read
  live vLLM metrics, and install the daemon onto a node. Destructive actions
  (recreate, node setup) are gated by **apply-mode + an explicit confirm**, and
  the daemon refuses to recreate itself.
- **Kubernetes** is read-only by default (nodes/pods/events); deploying a preset
  renders a manifest you apply yourself (`kubectl apply`).
- Operator-local writes are limited to your install dir (`$SNDR_HOME`, default
  `~/.sndr`): host profiles, GUI settings, auth store, container update-mode
  preferences, and generated report bundles.

## Install

The GUI daemon needs the optional `gui-api` extra (FastAPI + Uvicorn):

```bash
pip install "vllm-sndr-core[gui-api]"
```

The web assets are built from `gui/web` with Node:

```bash
cd gui/web
npm ci
npm run build      # production assets in gui/web/dist
```

## Run — integrated single-process mode (recommended)

Build the UI into the package once, then the daemon serves **both the UI and the
API on one port** (no separate Node server, same-origin, no CORS hop):

```bash
make gui-build     # npm ci && build, copies dist → vllm/sndr_core/product_api/web_static
python3 -m vllm.sndr_core.cli gui-api --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765
```

The built assets are bundled into the wheel, so a `pip install` of the package
serves the UI directly. Override the served assets dir with `SNDR_GUI_STATIC`.

## CLI flags (`sndr gui-api`)

```bash
python3 -m vllm.sndr_core.cli gui-api [--host H] [--port P] [--log-level L] [--enable-apply] [--open]
```

| Flag | Default | Purpose |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Bind address. Use `0.0.0.0` only behind a trusted boundary **with auth on**. |
| `--port` | `8765` | Bind port. |
| `--log-level` | `info` | uvicorn log level (`critical`/`error`/`warning`/`info`/`debug`/`trace`). |
| `--enable-apply` | off | Opt into **real** service-action/launch execution (otherwise dry-run). Also set by `SNDR_ENABLE_APPLY=1`. |
| `--open` | off | Open the served UI in a browser shortly after start. |

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `SNDR_HOME` | `~/.sndr` | Operator-local state: host profiles, settings, reports, **auth store** (`$SNDR_HOME/auth`). Mount as a volume in a container. |
| `SNDR_GUI_STATIC` | packaged `web_static` → repo `gui/web/dist` | Override the directory of built UI assets the daemon serves. |
| `SNDR_ENABLE_APPLY` | `0` | Enable real execution of mutating actions (same as `--enable-apply`). |
| `SNDR_GUI_TOKEN` | _(unset)_ | Legacy shared bearer token. When set, `/api/v1/*` requires it (still works alongside user auth). |
| `SNDR_AUTH` | `auto` | User authentication: `on` / `off` / `auto` (auto = on when non-loopback, accounts exist, or a token is set). |
| `SNDR_ADMIN_PASSWORD` | _(generated once)_ | Initial admin password on first bootstrap; otherwise auto-generated and printed once to the log. |
| `SNDR_AUTH_SESSION_TTL` | `86400` | Session lifetime in seconds. |
| `SNDR_AUTH_LOCK_THRESHOLD` / `_WINDOW` / `_SECONDS` | `8` / `300` / `900` | Brute-force throttle: failures before a temporary lockout, the counting window, and the lockout duration (seconds). |
| `SNDR_PUBLIC_URL` | `http://<host>:8765` | Public base URL — used for OAuth redirect URIs and `secure` cookies on HTTPS. |
| `SNDR_AUTH_PAM` | `0` | Enable system-account login via PAM (host deployments; needs the `gui-auth-pam` extra). |
| `SNDR_OAUTH_GOOGLE_CLIENT_ID` / `_SECRET` | _(unset)_ | Enable Google sign-in when both are set. |
| `SNDR_OAUTH_APPLE_CLIENT_ID` / `_SECRET` | _(unset)_ | Enable Apple sign-in when both are set. |
| `SNDR_IN_CONTAINER` | auto-detected | Force container-context detection (otherwise inferred from `/.dockerenv`, cgroups). |
| `SNDR_PROXMOX_HOST` / `_TOKEN_ID` / `_TOKEN_SECRET` | _(unset)_ | Connect the **Virtualization** panel to a Proxmox VE cluster (API token; e.g. `https://pve:8006`, `root@pam!sndr`, secret). Read-only. |
| `SNDR_PROXMOX_VERIFY_SSL` | `1` | Set `0` to accept a self-signed Proxmox certificate. |

The full authentication model, OAuth setup, container persistence and endpoint
list live in [`docs/GUI_SECURITY.md`](GUI_SECURITY.md).

## Run — dev mode (hot reload)

During frontend development, run the Vite dev server alongside the daemon:

```bash
python3 -m vllm.sndr_core.cli gui-api --host 127.0.0.1 --port 8765
cd gui/web && npm run dev        # http://127.0.0.1:5173
```

In dev the UI calls the API at `http://127.0.0.1:8765` (CORS-allowed). When the
UI is served by the daemon it defaults to the **same origin** automatically.
Override via the top-bar input, the `VITE_SNDR_API_BASE` build var, or the
`sndr.gui.apiBase` localStorage key.

## Run — remote desktop over SSH tunnel (recommended remote mode)

The GPU host is usually Linux; your laptop may be macOS or Windows. Do **not**
expose the daemon on the LAN. Keep it bound to `127.0.0.1` on the host and
forward a port over SSH:

```bash
# on your laptop
ssh -L 8765:127.0.0.1:8765 user@gpu-host
# then open the UI pointed at http://127.0.0.1:8765
```

The Hosts screen stores SSH host profiles (operator-local) and shows a copyable
tunnel command. It never opens the tunnel for you — that stays an explicit step.

## Authentication (server deployments)

For a shared/server install, enable user authentication. The simplest path:

```bash
SNDR_AUTH=on SNDR_ADMIN_PASSWORD="change-me" \
  python3 -m vllm.sndr_core.cli gui-api --host 0.0.0.0 --port 8765
```

On first start an **admin account is bootstrapped** (username = the system user
running the daemon; password from `SNDR_ADMIN_PASSWORD`, else generated and
printed once to the log). The UI then shows a sign-in screen. From **Advanced →
Admin** an admin can create/remove users and everyone can change their password
and enrol **TOTP two-factor**. Optional system login (PAM) and Google/Apple
sign-in are configured via environment variables.

Auth state (accounts, 2FA, session key) persists under `$SNDR_HOME/auth` — mount
that path as a volume so it survives container restarts. Full details, the OAuth
setup, the PAM-in-container caveat and the endpoint list:
[`docs/GUI_SECURITY.md`](GUI_SECURITY.md).

## Screens

| Screen | Purpose |
| --- | --- |
| Overview | Mission control: platform, catalog health, runtime environment/version, coverage |
| Setup | First-run host bootstrap wizard (detect → mode → preset → validate → plan) |
| Hosts | Local/remote host profiles and runtime target matrix |
| Containers | Live container management over the local docker socket (or a registered SSH host): cards/table list, **per-container detail** with live CPU/mem + GPU telemetry, a merged **Inference** panel (live vLLM `/metrics`), logs, exec, file browser, filesystem changes, processes, stats; start/stop/restart/recreate; **update modes** (manual/semi/auto) with pin-policy gating; install the SNDR daemon onto a node |
| Kubernetes | Read-only cluster view — nodes/pods/events with **GPU free/allocatable per node** and why a pod is `Pending` — plus **deploy a preset**: renders an SNDR-identity-stamped manifest (`sndr.io/preset`, `sndr.io/pin`, `sndr.io/patches` labels) so pods map back to the preset that defined them |
| Virtualization | One pane over compute — **Proxmox VE** hosts & guests (VMs + LXC), **KubeVirt** VMs, and **Kubernetes nodes** — each linked back to the SNDR preset it runs (Proxmox tag `sndr-preset-<id>`). Read-only; degrades to a connect/not-installed card per source. Bilingual (EN/RU) |
| Models | Catalog **summary strip + per-model key-facts**, identity/provenance, **capabilities + generation/sampling config**, requirements, hardware fit, patch matrix, cache + Hugging Face search/download |
| Configs | Graphical V2 composition editor (model/hardware/profile/preset) + per-element editor |
| Presets | Catalog, selected card, workload rules, policy graph |
| Launch Plan | Recommendation builder, backend-owned plan, gates, artifacts, CLI mirror |
| Services | Lifecycle planner (read-only/dry-run), **live engine status + live Prometheus metrics with trend sparklines**, engine & dependencies, capability contracts |
| Doctor | Aggregated diagnostics, readiness gates, registry coverage |
| Patches | Registry summary, inventory, lifecycle, bundles, upstream diff, per-patch explain |
| Benchmarks | Baseline, **live benchmark + A/B (real TTFT/TPOT/throughput from the engine)**, capability status, proof coverage, run plan/commands |
| Evidence | Evidence refs, proof artifact status, coverage breakdown, bundle commands |
| Clients | **Live engine status**, **Playground (streaming or one-shot prompt)**, OpenAI-compatible endpoints, copy-paste clients, auth |
| Reports | Generate redacted snapshot bundles into `$SNDR_HOME/reports/` |
| Advanced | Appearance, API/schema explorer, admin matrix, **account & security (password + 2FA)**, **user management (admin)**, feature contracts, config draft, CLI mirror |

## Live updates

The daemon exposes a Server-Sent Events stream at `GET /api/v1/events` plus a
pollable feed at `GET /api/v1/events/recent`. The UI consumes the SSE stream
when the daemon is open and falls back to authenticated polling when a token is
configured (browsers cannot attach an Authorization header to `EventSource`).

## Reports

The Reports screen generates a redacted snapshot bundle (`snapshot.json` +
`summary.md`) into `$SNDR_HOME/reports/<id>/`. Redaction strips your home path
and the GUI token by default. Nothing is written to the repo or a remote host.

## Container & engine management

The **Containers** screen manages Docker containers through the local docker
socket (mount `/var/run/docker.sock` into the daemon) or a registered SSH host.

**Capabilities**

- List as cards or a dense table; filter and sort; bulk-select with rolling
  start/stop/restart.
- Per-container detail tabs: **Overview, Config, Processes, Files, Changes,
  Logs, Stats, Exec**.
- Overview is a fluid dashboard: live CPU/mem sparklines, uptime/restarts/health,
  compact **GPU telemetry** (util, VRAM, temp, power, clocks, PCIe) for engine
  containers, a merged **Inference** panel, container facts, a Runtime card, an
  Engine & SNDR card, and Mounts/Env/Labels.
- Lifecycle: start / stop / restart / **recreate** (pull + recreate preserving
  config, recording the previous image for rollback). The daemon refuses to
  recreate **itself**.
- **Update modes** per container — `manual` / `semi` / `auto`. Engine containers
  are blocked from `auto` by the vLLM pin policy (critical-gating).
- **Exec** runs commands inside the container; disabled unless
  `SNDR_ENABLE_EXEC=1`.
- **Install SNDR daemon on this node** ships the canonical package and launches
  the sidecar daemon (double-gated: apply-mode **and** an explicit confirm).

**Source linking & drift** — `sndr profile render-launchers` stamps every engine
container with identity labels (`sndr.preset`, `sndr.pin`, `sndr.served-model`,
`sndr.patch-count`, `sndr.role`). The detail page then links the running engine
back to its preset/profile, shows the served model + pin (no engine api-key
needed), lists live Genesis patches, and **diffs the live runtime against the
YAML** (image + `GENESIS_*` flags) — surfacing config drift. Containers launched
by hand without the label fall back to a container-name match, or show "no
linked preset".

**Inference panel** (engine containers) scrapes the engine's Prometheus
`/metrics` (the daemon's configured engine): running/waiting requests, KV-cache
utilization, generation tokens/s, TTFT/TPOT/E2E latency, success/preemptions,
token totals, and MTP spec-decode acceptance. It requires the engine's vLLM
**stat logger to be ON** — launchers ship `--disable-log-stats` by default
(small overhead); enable it per rig/profile (see "Files & config locations").

**Requirements:** the docker socket mounted (or an SSH host profile); for live
metrics, the engine's `/metrics` reachable from the daemon.

**Where it writes:** per-container update-mode and the recorded previous image
live under `$SNDR_HOME/state/`. Lifecycle actions hit docker/SSH directly (not
dry-run); only recreate and node-setup require the apply+confirm gate.

## Kubernetes

Read-only by default; honours your kubeconfig + RBAC and degrades to an
"unavailable" state when no cluster is reachable.

- **Monitor** — Nodes (GPU capacity/allocatable/free, conditions, taints, GPU
  labels), Pods (phase, ready, restarts, GPU request, **+ SNDR identity chips**),
  Events (e.g. `FailedScheduling: Insufficient nvidia.com/gpu`).
- **Deploy** — pick a preset → render a manifest set (ConfigMap + Service + PVC +
  Deployment) with GPU limits and `/health` probes. Every Deployment is stamped
  with `sndr.io/preset`, `sndr.io/pin`, `sndr.io/patches` and
  `app.kubernetes.io/managed-by: sndr`, so the Pods tab maps each pod back to the
  preset/pin/patches that produced it.

```bash
sndr k8s render <preset>            # print manifests (also the GUI Deploy tab)
kubectl apply -f sndr-<preset>.yaml
kubectl rollout status deploy/sndr-<preset>
```

**Requirements:** a reachable cluster (kubeconfig at `/etc/rancher/k3s/k3s.yaml`
or `~/.kube/config`), the python `kubernetes` client (auto-installed by node
setup when a kubeconfig is mounted), and the NVIDIA device plugin / gpu-operator
so nodes advertise `nvidia.com/gpu`.

## Files & config locations

Backend lives in the canonical `sndr/` package (re-exported as `vllm/sndr_core/*`
for back-compat). Key files:

| File | Role |
| --- | --- |
| `sndr/product_api/legacy/http_app.py` | FastAPI daemon (`run_server`) + every `/api/v1/*` route |
| `sndr/product_api/legacy/container_ops.py` | Docker-socket / SSH container ops (inspect, stats, logs, exec, recreate) |
| `sndr/product_api/legacy/update_prefs.py` | Per-container update-mode + previous-image store |
| `sndr/product_api/legacy/engine_client.py` | vLLM `/metrics` scrape + chat proxy (Inference panel, Clients) |
| `sndr/product_api/legacy/k8s_client.py` | Read-only k8s shaping (nodes/pods/events, GPU, SNDR identity) |
| `sndr/product_api/legacy/node_setup.py` | Node bundle + gated SSH daemon install |
| `sndr/cli/legacy/k8s.py` | k8s manifest renderer (identity labels) |
| `gui/web/` | React/TS UI; build output is copied to `sndr/product_api/legacy/web_static/` |

Config files and where they are written:

| Path | What |
| --- | --- |
| `$SNDR_HOME` (default `~/.sndr`) | `auth/` (accounts/2FA/session), `state/` (host profiles, GUI settings, container update prefs), `reports/` |
| `sndr/model_configs/builtin/{models,hardware,profile,presets}/*.yaml` | V2 catalog (model, hardware/rig, profile, preset definitions) |
| `…/hardware/<rig>.yaml` → `sizing.disable_log_stats` | Engine stat-logger toggle. `false` exposes live vLLM metrics to the Inference panel; also settable as a profile `sizing_override`. After changing it, re-render + restart the engine |
| host `start_*.sh` (generated) | Engine launch scripts produced by `sndr profile render-launchers <profile>` |
| `/etc/rancher/k3s/k3s.yaml` or `~/.kube/config` | kubeconfig the daemon reads for Kubernetes / KubeVirt / Virtualization |
| `SNDR_PROXMOX_*` (env) | Proxmox VE connection for the Virtualization panel (host + API token) |

The UI is bilingual (English / Russian) — toggle with the **EN/RU** button in the
top bar; the choice persists in `localStorage` (`sndr.gui.lang`). New surfaces
(Virtualization, the sidebar nav) are fully translated; other screens adopt
translations incrementally and fall back to English.

## Verifying it works

```bash
curl http://127.0.0.1:8765/api/v1/health        # -> {"status":"ok",...,"read_only":true}
curl http://127.0.0.1:8765/openapi.json | head  # OpenAPI document
```

Frontend checks:

```bash
cd gui/web
npm run typecheck
npm run test           # vitest unit/component suite (263 tests)
npm run check:api      # fails if the client/daemon API contract drifts
npm run test:e2e       # Playwright smoke (daemon + dev server must be running)
```

## See also

- `docs/PRODUCT_API.md` — the read-only API contract and full route map.
- `docs/GUI_SECURITY.md` — bind, auth (accounts/2FA/OAuth/PAM), persistence,
  redaction, and dangerous-action policy.
- `docs/GUI_ROADMAP.md` — what is done and what remains (prioritized).

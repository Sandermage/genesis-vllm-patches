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
make gui-build     # npm ci && build, copies dist → sndr/product_api/legacy/web_static
python3 -m sndr.cli gui-api --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765
```

The built assets are bundled into the wheel, so a `pip install` of the package
serves the UI directly. Override the served assets dir with `SNDR_GUI_STATIC`.

## CLI flags (`sndr gui-api`)

```bash
python3 -m sndr.cli gui-api [--host H] [--port P] [--log-level L] [--enable-apply] [--open]
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
python3 -m sndr.cli gui-api --host 127.0.0.1 --port 8765
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
  python3 -m sndr.cli gui-api --host 0.0.0.0 --port 8765
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
| Fleet | Every registered GPU/engine host at a glance — one concurrent SSH sweep shows status, running model, vLLM version, GPUs and live patch count per server |
| Hosts | Local/remote host profiles and runtime target matrix |
| Containers | Live container management over the local docker socket (or a registered SSH host): cards/table list, **per-container detail** with live CPU/mem + GPU telemetry, a merged **Inference** panel (live vLLM `/metrics`), logs, exec, file browser, filesystem changes, processes, stats; start/stop/restart/recreate; **update modes** (manual/semi/auto) with pin-policy gating; install the SNDR daemon onto a node |
| Virtualization | One control plane over compute, two providers. **Proxmox VE**: host nodes (CPU/mem/disk meters) + guests (VMs + LXC) with resources & uptime. **Kubernetes**: node cards with GPU free/allocatable, pods (phase/ready/restarts/GPU + SNDR identity), events (why a pod is `Pending`), KubeVirt VMs, and **deploy a preset** (SNDR-identity-stamped manifest — `sndr.io/preset`/`pin`/`patches`). Every guest/pod links back to the SNDR preset it runs (Proxmox tag `sndr-preset-<id>`). Read-only; degrades to a connect/not-installed card per source. Bilingual (EN/RU). _The standalone Kubernetes screen was folded in here; the old `#kubernetes` link still lands here._ |
| Models | Catalog **summary strip + per-model key-facts**, identity/provenance, **capabilities + generation/sampling config**, requirements, hardware fit, patch matrix, cache + Hugging Face search/download |
| Configs | Graphical V2 composition editor (model/hardware/profile/preset) + per-element editor |
| Presets | Catalog, selected card, workload rules, policy graph |
| Planner | KV-cache / VRAM fit calculator (GQA / MoE / tensor-parallel aware, calibratable) + quality-baseline regression diff |
| Launch Plan | Recommendation builder, backend-owned plan, gates, artifacts, CLI mirror |
| Services | Lifecycle planner (read-only/dry-run), **live engine status + live Prometheus metrics with trend sparklines**, engine & dependencies, capability contracts |
| Routing | Deterministic spec-decode workload router — per bench-validated profile, which workloads are allowed/denied + their measured TPS delta, plus a live classifier that predicts how a request's signals resolve to a profile (one source of truth, shared with the gateway) |
| Doctor | Aggregated diagnostics, readiness gates, registry coverage |
| Patches | Registry summary, inventory, lifecycle, bundles, upstream diff, per-patch explain |
| Flags | Searchable / filterable matrix of every `GENESIS_ENABLE_*` flag with its default; overlay a running engine's live ON/OFF state and flag drift |
| Benchmarks | Baseline, **live benchmark + A/B (real TTFT/TPOT/throughput from the engine)**, capability status, proof coverage, run plan/commands |
| Evidence | Evidence refs, proof artifact status, coverage breakdown, bundle commands |
| Chat | Multi-turn **streaming** chat with any running vLLM model — model/host picker, sampling, system prompt + **prompt library**, **reasoning/thinking** trace, and live **web search** + **project-RAG** grounding with cited sources (see [Chat & Ops Copilot](#chat--ops-copilot)) |
| Copilot | Read-only **tool-calling** assistant over the Product API + market/news tools — answers from real catalog/doctor/preset/patch/capacity data and proposes changes you review & apply |
| Clients | **Live engine status**, **Playground (streaming or one-shot prompt)**, OpenAI-compatible endpoints, copy-paste clients, auth |
| Reports | Generate redacted snapshot bundles into `$SNDR_HOME/reports/` |
| Operations | Run sndr_core's canonical maintenance / audit / proof workflows as live-monitored jobs — the CLI surface, integrated |
| Advanced | Appearance, API/schema explorer, admin matrix, **account & security (password + 2FA)**, **user management (admin)**, feature contracts, config draft, CLI mirror, audit log, updates |

## Chat & Ops Copilot

The **Engine** surface gives two assistants over your running stack. Both pick the
engine host/port (and any registered SSH host) from the top of the panel.

**Chat** (the _Chat_ screen) — multi-turn **streaming** chat with any running vLLM
model:

- **Model & sampling.** Pick the model in the _Settings_ tab; tune temperature,
  top-p / top-k / min-p, presence / frequency / repetition penalty, seed, stop
  sequences and max tokens. Max tokens defaults high enough for reasoning models
  (which spend tokens in `reasoning_content` before the visible answer).
- **Recommended sampling.** When the running model is matched to the catalog,
  the sampling section shows the model's **validated defaults** (e.g. Qwen 3.6 →
  `temperature 0.6 · top_p 0.95 · top_k 20`, cross-referenced against the
  community club-3090 recipes) with a one-click **Apply** — so you get the right
  settings for the model instead of the generic 0.7.
- **Thinking mode** (`enable_thinking`). Reasoning models (Qwen3, …) emit a
  `<think>` trace that the daemon — launched with `--reasoning-parser` — routes to
  `delta.reasoning_content`. The GUI renders it in a collapsible block above the
  answer, so a reply that is "all reasoning, empty content" is never shown as
  blank. (With a `coder` model + tool-calls, the panel warns that the
  `qwen3_coder` streaming parser drops `delta.tool_calls` — serve with
  `--tool-call-parser qwen3_xml` for reliable streaming tool calls.)
- **Web search.** A toggle grounds the answer in live web results via your
  **self-hosted SearXNG** — no external API key. Retrieved pages are cited as
  source chips under the answer.
- **Project RAG.** A toggle grounds answers in _your_ knowledge — project
  patches/presets/configs plus connected Obsidian / notes folders (add folders in
  _Params_). Web and project grounding can be combined; each source chip shows
  which kind it is.
- **Prompt library.** Choose a reusable system prompt, or manage your own — see
  [Prompt & tool library](#prompt--tool-library).

**Ops Copilot** (the _Copilot_ screen) — a **read-only, tool-calling** assistant.
It answers operational questions by calling real Product API tools (catalog,
doctor, presets, patches, capacity) plus market/news-analysis tools, then proposes
concrete changes you review and apply — it never mutates anything itself. Tools
are grouped by category (read / plan / search / analysis / observability /
custom), and you can register your own (below).

Backend: `POST /api/v1/engine/chat/stream` (chat, Server-Sent tokens — reasoning,
content, sources and a final usage/timing frame arrive as distinct events),
`POST /api/v1/copilot/chat` (copilot), `POST /api/v1/chat/retrieve` +
`/api/v1/chat/rag/preview` (project RAG).

## Prompt & tool library

Open from Chat or Copilot — an OpenWebUI-style store of reusable **prompts** and
**tools**, persisted under `$SNDR_HOME` and shared by both assistants through one
cache (edit a prompt once, the chat selector and the library both update).

- **Prompts** — named system prompts (title + body). Built-ins are seeded
  (including a full crypto-analyst prompt); create your own under any name and
  pick one as the chat system prompt.
- **Tools** — declarative HTTP tools the copilot can call. Each fixes a host in a
  URL template and URL-encodes named arguments into `{param}` placeholders, so it
  is **SSRF-safe** (arguments can never move the host). Custom names appear in the
  copilot's `custom` category.
- **Native analysis tools** ship in-box: a crypto **market overview** (top coins,
  global stats, Fear & Greed, futures, macro), per-coin data, and a **news /
  info-field** classifier (ETF / liquidations / geopolitics / regulation) over web
  search — the building blocks for market + news-field analysis.

CRUD endpoints: `/api/v1/prompts` and `/api/v1/tools/managed`.

## Command palette & keyboard shortcuts

| Key | Action |
| --- | --- |
| **⌘K** / **Ctrl-K** | Open/close the command palette — fuzzy-jump to any section, preset or action |
| **?** | Toggle the keyboard-shortcuts help overlay |
| **Esc** | Close the palette or any open overlay/modal |
| **`g` then a key** | GitHub/Linear-style navigation chord (second key within 1.2 s) |

Navigation chords (`g` then):

| Chord | Section | Chord | Section |
| --- | --- | --- | --- |
| `g o` | Overview | `g c` | Configs |
| `g s` | Setup | `g p` | Presets |
| `g f` | Fleet | `g n` | Containers |
| `g h` | Hosts | `g d` | Doctor |
| `g m` | Models | `g l` | Launch Plan |
| | | `g b` | Benchmarks |

Single keys and chords are suppressed while typing in an input/textarea, so they
never eat real text.

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

Backend lives in the canonical `sndr/` package. Key files:

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

Frontend checks (from the repo root):

```bash
make gui-lint            # tsc -b (strict) + eslint, with jsx-a11y enforced as errors
make audit-i18n          # every tr() string has a Russian translation (ratchet gate, baseline 0)
make test-gui-contract   # every api.ts path resolves to a live daemon route (drift gate)
make gui-build           # production build + copy into the served web_static/
```

Or directly inside `gui/web`: `npm run typecheck`, `npm run lint`, `npm run build`.
All four `make` checks are part of the repo's CI `gates` target.

## Glossary

Plain-language definitions for the metrics and terms the GUI surfaces (chat
stats, the Planner, benchmarks, the Inference panel):

| Term | What it means |
| --- | --- |
| **TPS** | Tokens per second — output generation speed. ~70 TPS reads as conversational. |
| **Wall TPS** | `completion_tokens / wall_time` — user-perceived total speed (includes prefill). |
| **Decode TPS** | `completion_tokens / (wall_time − TTFT)` — pure decode rate, excludes prefill. |
| **TTFT** | Time to first token — dominated by prefill on long prompts. |
| **CV** | Coefficient of variation across bench runs; lower = more predictable (target < 5%). |
| **Prefill** | Processing the whole input before the first output token (slow first request, fast on follow-ups via prefix cache). |
| **Decode** | Generating output tokens one at a time after prefill. |
| **KV cache** | The model's working memory of the conversation; larger context = more KV-cache VRAM. |
| **Prefix cache** | Reusing a shared leading prompt across requests (skip re-prefill) — big win for long-document/agent loops. |
| **Context window** | Tokens the model holds at once (`--max-model-len`). The Planner predicts the largest that fits. |
| **Activation memory** | Transient forward-pass memory; activation peaks during prefill cause most OOMs. |
| **Acceptance length (AL)** | Mean tokens accepted per speculative step (MTP / spec-decode); higher = more speedup. |
| **MTP / spec-decode** | Multi-Token Prediction / speculative decoding — a draft head proposes tokens the target verifies in one pass. |
| **KV dtype** | Precision the KV cache is stored in (fp16 → int8 → TQ3); lower halves VRAM at a small quality cost. The Planner + Hardware/Fleet arch notes recommend the best one your GPU accelerates. |
| **Quant (AWQ / GPTQ / AutoRound / FP8 / TQ)** | Weight/KV compression schemes; what your GPU runs _natively_ vs _emulated_ depends on its arch (the Planner's "Arch-aware notes" and the Fleet card's "Arch-aware flags" show this per-GPU). |

## Further reading (community recipes)

For deeper reference on the exact models and consumer/Ada/Blackwell GPUs this
stack targets, the community **club-3090** project keeps high-quality recipes
(our catalog is cross-referenced against it — see the per-model recommended
sampling in chat, and the arch-aware dtype notes in the Planner/Fleet):

- KV-cache VRAM math (behind the Planner) — [`club-3090/docs/KV_MATH.md`](https://github.com/noonghunna/club-3090/blob/master/docs/KV_MATH.md)
- GPU dtype/quant capability matrix (Ampere/Ada/Hopper/Blackwell) — [`DTYPE_MATRIX.md`](https://github.com/noonghunna/club-3090/blob/master/docs/DTYPE_MATRIX.md), [`QUANTIZATION.md`](https://github.com/noonghunna/club-3090/blob/master/docs/QUANTIZATION.md)
- Performance cliffs (long-context / sustained-agent OOM) — [`CLIFFS.md`](https://github.com/noonghunna/club-3090/blob/master/docs/CLIFFS.md)
- Terminology primer — [`GLOSSARY.md`](https://github.com/noonghunna/club-3090/blob/master/docs/GLOSSARY.md), [`LOCAL_AI_PRIMER.md`](https://github.com/noonghunna/club-3090/blob/master/docs/LOCAL_AI_PRIMER.md)

## See also

- `docs/PRODUCT_API.md` — the read-only API contract and full route map.
- `docs/GUI_SECURITY.md` — bind, auth (accounts/2FA/OAuth/PAM), persistence,
  redaction, and dangerous-action policy.
- `gui/web/README.md` — frontend architecture (tech stack, layout, i18n, dev/build).

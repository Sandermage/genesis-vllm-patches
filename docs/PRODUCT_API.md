# SNDR Product API

The Product API is a stable, typed, read-only data layer inside the `sndr`
package (`sndr/product_api/`). It is consumed by the CLI, the GUI daemon, and
future SDK/desktop clients. The HTTP daemon (`sndr gui-api`) is a thin transport
over it — it adds no business logic.

## Principles

- **One backend, multiple shells.** GUI and CLI call the same Product API. No
  client parses human CLI stdout to infer state.
- **Import-safe.** The package imports without FastAPI, torch, or vLLM at module
  top; heavy dependencies are imported lazily inside functions.
- **Read-only.** No endpoint writes V2 YAML, the patch registry, or runtime
  artifacts, and none runs a subprocess against a host. Apply endpoints produce
  dry-run jobs. The only writes are operator-local (`$SNDR_HOME`): host
  profiles, GUI settings, report bundles, the **auth store** under `$SNDR_HOME/auth`, and the **job/event
  store** under `$SNDR_HOME/state`. The `/api/v1/auth/*` routes are the one
  intentional, isolated mutating surface (login, account + 2FA management).
- **JSON-safe.** Responses are frozen dataclasses serialized with
  `dataclasses.asdict`; the daemon coerces them to JSON.

## Running the daemon

```bash
python3 -m sndr.cli gui-api --host 127.0.0.1 --port 8765 --log-level info
```

`GET /openapi.json` exposes the full OpenAPI document. The frontend types are
generated from it (`gui/web` → `npm run gen:api`), and a contract guard
(`npm run check:api` + a compile-time route assertion) catches drift.

## Route map

### Status & platform

- `GET /api/v1/health` — liveness; reports `read_only: true`.
- `GET /api/v1/auth/status` — auth required?, enabled backends, OAuth providers,
  deployment context (container/system-user/PAM), and the current user.
- `GET /api/v1/capabilities` — runtime targets + feature inventory with statuses
  (`available` / `partial` / `render_only` / `deferred` / `missing`).
- `GET /api/v1/overview` — combined dashboard snapshot.
- `GET /api/v1/catalog/summary` — V2 catalog counts and distributions.
- `GET /api/v1/environment` — project/engine versions and dependency stack.
- `GET /api/v1/doctor` — aggregated diagnostics report.

### Authentication & sessions

Active only when auth is enabled (`SNDR_AUTH=on`/`auto`); see
[`docs/GUI_SECURITY.md`](GUI_SECURITY.md) for the full model and env vars.

- `POST /api/v1/auth/login` — `{username,password}` → session cookie + token, or
  `{needs_2fa:true}`.
- `POST /api/v1/auth/login/2fa` — `{username,code}` → session.
- `POST /api/v1/auth/logout` — clear the session.
- `GET /api/v1/auth/me` — the current account.
- `POST /api/v1/auth/password` — `{current,new}` change own password.
- `GET /api/v1/auth/users` · `POST /api/v1/auth/users` · `DELETE /api/v1/auth/users/{username}`
  — admin user directory (create/list/remove).
- `POST /api/v1/auth/2fa/enroll` → `{secret, otpauth_uri}`;
  `POST /api/v1/auth/2fa/activate` `{code}`; `POST /api/v1/auth/2fa/disable`.
- `GET /api/v1/auth/oauth/{provider}/login` → 307 to Google/Apple;
  `GET|POST /api/v1/auth/oauth/{provider}/callback` → session.

### Catalog & presets

- `GET /api/v1/presets` — preset catalog (filters: family, workload, hardware,
  mode, status).
- `GET /api/v1/presets/recommend` — ranked recommendations by workload.
- `GET /api/v1/presets/{preset_id}` — preset record.
- `GET /api/v1/presets/{preset_id}/explain` — card + composed runtime + fallback.

### V2 config editor

- `GET /api/v1/configs/v2/catalog` — model/hardware/profile/preset layers.
- `GET /api/v1/configs/v2/preview` — compose a draft from layer ids.
- `POST /api/v1/configs/v2/plan` — write-safe plan (diff + draft YAML, no write).
- `POST /api/v1/configs/v2/apply` — write an operator-local preset (atomic +
  backup + lock, under `$SNDR_HOME`).
- `GET /api/v1/configs/v2/user-presets` — operator-local presets.
- `GET /api/v1/configs/v2/layer/{kind}/{layer_id}` — full layer definition.
- `POST /api/v1/configs/v2/layer/apply` — write an edited layer to the user dir.

### Launch, services, jobs, events

- `GET /api/v1/launch/plan` — backend-owned launch plan (gated, `actionable` flag).
- `GET /api/v1/services/plan` — read-only lifecycle action plan.
- `POST /api/v1/services/apply` — create a dry-run lifecycle job.
- `GET /api/v1/jobs`, `GET /api/v1/jobs/{job_id}` — job store (dry-run + executed
  + live background jobs). **Persisted** under `$SNDR_HOME/state` — survives restart.
- `GET /api/v1/events` — SSE event stream (snapshot + incremental + heartbeat).
- `GET /api/v1/events/recent` — pollable JSON event feed (cursor via `since_seq`).

### Patches & evidence

- `GET /api/v1/patches` — registry inventory with filters and summary counts.
- `GET /api/v1/patches/doctor` — apply-module coverage.
- `GET /api/v1/patches/overrides` · `POST /api/v1/patches/overrides` — operator
  force a patch on/off/default; persisted under `$SNDR_HOME` and emitted as
  `GENESIS_ENABLE_<flag>=1|0` into the launch env (validated, no shell injection).
- `GET /api/v1/patches/{patch_id}/explain` — per-patch meta/spec/live decision.
- `GET /api/v1/patches/bundles`, `GET /api/v1/patches/bundles/{name}`.
- `GET /api/v1/patches/diff-upstream` — upstream relationship.
- `GET /api/v1/proof/status` — proof artifact status.

### Live engine bridge

Unlike the rest of the API (static project state), these reach the **running**
vLLM OpenAI server. The engine is addressed by a validated host (or the
operator-set `SNDR_OPENAI_BASE_URL` / `SNDR_METRICS_URL`) with fixed
ports/paths — never an arbitrary client URL (anti-SSRF). An unreachable engine
yields a structured "down" payload, not a 500.

- `GET /api/v1/engine/status?host=` — `/health` + `/version` + `/v1/models`
  (reachable, loaded models, engine version).
- `GET /api/v1/engine/metrics?host=` — scrape + distill the engine's Prometheus
  `/metrics` into KPIs (queue, KV-cache, throughput delta, TTFT/TPOT, spec-decode
  acceptance) plus a rolling `history` ring buffer for trend sparklines.
- `POST /api/v1/engine/chat` — proxy a non-streaming chat completion (a GUI
  smoke test); returns reply, token usage and latency. `502` on engine error,
  `503` when unreachable.
- `POST /api/v1/engine/chat/stream` — same, **streamed** as ND-JSON
  (`{"delta"}` chunks then `{"done", ttft_ms, latency_ms, tokens}`).
- `POST /api/v1/engine/bench` — run a **real** micro-benchmark against the engine
  (streamed completions): throughput (tok/s), TTFT p50/p90, TPOT, CV and
  request counts. A quick GUI bench (params echoed) for A/B deltas — not the
  canonical Wave suite.

### Models, memory, reports, hosts

- `GET /api/v1/models/cache` — checkpoint presence on the daemon host.
- `GET /api/v1/models/hub/search?query=&limit=` — search the Hugging Face Hub
  (id, downloads, likes, pipeline, gated). TLS verified via certifi.
- `POST /api/v1/models/download` — pull weights for a **catalog** model (`model_id`)
  or any **Hugging Face** repo (`repo_id`, `org/name`). Both ids strictly validated.
  With `--enable-apply` it runs the pull as a **live background job** (status,
  streaming log, `progress`); otherwise a dry-run job.
- `GET /api/v1/memory/fit` — model × hardware compatibility (GPU/CUDA/blocklist)
  with informational VRAM.
- `POST /api/v1/reports/bundle` — generate a redacted bundle into
  `$SNDR_HOME/reports/`. `GET /api/v1/reports/types` lists supported types.
- `GET /api/v1/hosts`, `POST /api/v1/hosts`, `DELETE /api/v1/hosts/{host_id}` —
  operator-local host profiles.

## Error mapping

- Missing/invalid session or token (auth on) → `401`; insufficient role → `403`.
- Unknown workload → `400`.
- Unknown preset / model / hardware / job / layer → `404`.
- Config plan/apply conflict → `409`; validation failure → `422`.
- Compose / report generation failure → `500`.

## Tests

```bash
python3 -m pytest tests/unit/product_api            # API + routes
python3 -m pytest tests/unit/product_api/auth       # auth subsystem
```

Covers dataclass shapes, deterministic statuses under a faked `which`, catalog
counts vs the registry, the HTTP routes via FastAPI `TestClient`, and the
write-safe boundary. The `auth/` suite covers the crypto core (scrypt, RFC-6238
TOTP, signed sessions), the persistent user store, the login/2FA/admin state
machine, OAuth URL/claim handling, and the HTTP auth flow end-to-end.

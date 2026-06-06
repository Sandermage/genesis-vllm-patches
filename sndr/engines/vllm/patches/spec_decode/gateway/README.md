# spec_decode gateway — D2a

OpenAI-compatible reverse proxy that routes requests between a
**default** vLLM upstream (MTP OFF, production-safe) and a
**structured-k4** vLLM upstream (validated MTP profile) based on
explicit request signals + bench artifact + safety guard.

This is the **deployment layer** sitting above the spec-decode
contract / planner / artifact / guard / router stack. Gateway is a
pure-Python FastAPI service. It does NOT touch vLLM internals.

## Quick start (local smoke)

```bash
# Terminal 1 — default vLLM (MTP OFF), port 8101
bash ~/start_g4_baseline_nomtp.sh

# Terminal 2 — structured-k4 vLLM (β′-A K=4), port 8102
bash ~/start_g4_betaA_k1.sh 4

# Terminal 3 — gateway
export GENESIS_GATEWAY_DEFAULT_URL=http://localhost:8101
export GENESIS_GATEWAY_STRUCTURED_URL=http://localhost:8102
export GENESIS_GATEWAY_PROFILE=gemma4-31b-tq-mtp-structured-k4
python -m vllm.sndr_core.integrations.spec_decode.gateway

# Terminal 4 — verify routing
curl -X POST http://localhost:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer genesis-local' \
  -d '{"model":"gemma-4-31b","messages":[{"role":"user","content":"json please"}],
       "response_format":{"type":"json_object"}}'
# → routed to structured upstream

curl -X POST http://localhost:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-31b","messages":[{"role":"user","content":"hello"}]}'
# → routed to default upstream
```

## Routing rules (D2a)

| signal | decision | upstream |
|---|---|---|
| `response_format: {type: json_object/json_schema}` | tool_json | structured |
| `tool_choice: "required"` | tool_json | structured |
| `tool_choice: {type: "function", ...}` | tool_json | structured |
| `extra_body.workload_class` in artifact's `allowed_workloads` | that class | structured |
| `tool_choice: "auto"` alone | no signal | default |
| free chat (no signal) | no signal | default |
| `stream=true` (D2a only) | streaming | default |
| artifact missing / router exception | uncertainty | default |
| force-default admin flag on | override | default |
| structured upstream `down` | health-driven | default |

**Default-first** is the safety invariant. Any uncertainty → default.

## Env vars

All env names below have a canonical `SNDR_*` form and a `GENESIS_*`
legacy alias. SNDR wins if both are set; GENESIS emits a deprecation
warning on each first read.

| canonical name | default | meaning |
|---|---|---|
| `SNDR_GATEWAY_DEFAULT_URL` | `http://localhost:8101` | default upstream |
| `SNDR_GATEWAY_STRUCTURED_URL` | `http://localhost:8102` | structured upstream |
| `SNDR_GATEWAY_PROFILE` | `gemma4-31b-tq-mtp-structured-k4` | which artifact to load |
| `SNDR_GATEWAY_BIND_HOST` | `0.0.0.0` | gateway bind host |
| `SNDR_GATEWAY_BIND_PORT` | `8100` | gateway bind port |
| `SNDR_GATEWAY_HEALTH_INTERVAL` | `5` | upstream probe interval (s) |
| `SNDR_GATEWAY_TIMEOUT` | `120` | upstream request timeout (s) |
| `SNDR_GATEWAY_LOG_LEVEL` | `INFO` | logging |
| `SNDR_GATEWAY_ADMIN_ALLOW_REMOTE` | unset | `1` to permit non-localhost admin calls |
| `SNDR_SPEC_DECODE_ARTIFACTS_DIR` | unset | extra artifact search dir |
| `SNDR_ALLOW_SPEC_DECODE_KV_ADAPTER` | unset | safety guard: structural opt-in |
| `SNDR_ALLOW_SPEC_DECODE_FUNCTIONAL_UNKNOWN` | unset | safety guard: unverified opt-in |
| `SNDR_DISABLE_SPEC_DECODE_SAFETY_GUARD` | unset | `1` to disable PN274 guard entirely |

## Public endpoints

| method | path | purpose |
|---|---|---|
| GET | `/healthz` | gateway liveness |
| GET | `/readyz` | 200 iff default upstream is `up` |
| GET | `/v1/models` | passthrough from default |
| POST | `/v1/chat/completions` | router-decided proxy |
| POST | `/v1/completions` | router-decided proxy |
| GET | `/metrics` | Prometheus exposition |

## Admin endpoints (localhost-only)

| method | path | purpose |
|---|---|---|
| POST | `/admin/force-default` | route all traffic to default |
| POST | `/admin/clear-force-default` | clear the flag |
| POST | `/admin/reload-artifacts` | re-read artifact JSON |
| GET | `/admin/state` | full state dump |

SIGHUP also triggers artifact reload (POSIX only).

## Metrics (Prometheus catalog)

D2a/D2b shipped under `genesis_*` namespace. D2c adds canonical
`sndr_*` metrics alongside (existing names are NOT renamed —
dashboards built on D2a/D2b expositions continue to work).

### D2a/D2b (existing, unchanged)

| metric | labels | type |
|---|---|---|
| `genesis_routed_default_total` | – | counter |
| `genesis_routed_structured_total` | – | counter |
| `genesis_fallback_total` | `reason` | counter |
| `genesis_upstream_error_total` | `upstream, kind` | counter |
| `genesis_router_decision_total` | `profile, accepted` | counter |
| `genesis_request_latency_seconds` | `upstream` | histogram |
| `genesis_force_default_active` | – | gauge |
| `genesis_upstream_health` | `upstream` | gauge (1=up, 0.5=degraded, 0=down) |

### D2c (new SNDR-namespaced)

| metric | labels | type |
|---|---|---|
| `sndr_route_latency_seconds` | `upstream, profile, stream` | histogram |
| `sndr_streaming_request_total` | `upstream` | counter |
| `sndr_streaming_error_total` | `upstream, reason` | counter |
| `sndr_upstream_probe_failures_total` | `upstream` | counter |
| `sndr_upstream_health_state` | `upstream` | gauge (mirrors `genesis_upstream_health`) |

Fallback reasons emitted: `force`, `streaming`, `no_artifact`,
`router_error`, `router_denied`, `structured_down`, `upstream_error`.
Streaming error reasons: `open_failed`, `mid_stream`,
`client_disconnect` (last two emitted by future patches).

### Dashboards + alerts

| artifact | location |
|---|---|
| Grafana dashboard JSON | [deploy/dashboards/sndr-gateway-overview.json](deploy/dashboards/sndr-gateway-overview.json) |
| Prometheus alert rules | [deploy/alerts/sndr-gateway-alerts.yaml](deploy/alerts/sndr-gateway-alerts.yaml) |

Six panels: traffic split, fallback reasons, p50/p95 latency,
upstream health timeline, structured acceptance proxy (placeholder),
errors. Six alerts: gateway down, structured down >2m, default down,
fallback rate >5%, p95 latency >60s, probe failures sustained, VRAM
low (placeholder).

## D2a out of scope (documented exclusions)

- TLS termination
- Rate limiting
- Multi-model routing
- Prompt-text classifier
- Per-request engine MTP toggle
- Streaming through structured upstream (D2b will add)
- Artifact mutation via API

## Smoke gate (acceptance criterion for D2a)

Six cases must all pass:

1. tool_json (`response_format`) + structured up → structured
2. free chat → default (router_denied)
3. tool_json + structured down → default (structured_down)
4. force-default flag set + structured up → default (force)
5. artifact not loaded → default (no_artifact)
6. Qwen-style model (no Gemma artifact) → default (no_artifact)

Plus `/admin/reload-artifacts` must restore artifact after manual clear.

All 6 cases verified locally on 2026-05-20 with mock upstreams. See
the smoke runner in commit message of `2497c371`'s successor.

## Rollback

| level | action | recovery time |
|---|---|---|
| 1 | `POST /admin/force-default` | 1 sec |
| 2 | stop structured container | 5-15 sec (health probe) |
| 3 | stop gateway container | LB cutover (~30s) |
| 4 | remove gateway from LB rotation | LB cutover |

See `docs/_internal/DEPLOYMENT_PROFILE_ROUTING_PLAN_2026-05-20.md`
section 6 for full rollback hierarchy.

## File map

```
gateway/
├── __init__.py
├── __main__.py     # python -m … entry point
├── app.py          # FastAPI app, routes, startup/shutdown
├── upstream.py     # UpstreamState + health-check loop
├── proxy.py        # request forwarding + route decision
├── admin.py        # localhost-only operator endpoints
├── metrics.py      # Prometheus catalog
└── README.md       # this file
```

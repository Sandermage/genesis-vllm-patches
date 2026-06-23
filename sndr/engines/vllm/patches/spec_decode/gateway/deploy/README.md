# SNDR gateway — deployment

D2b deliverable. This directory carries the deployment templates
(Dockerfile, docker-compose, dashboards, alerts) that an operator
uses to run the spec-decode gateway in production alongside one or
two vLLM upstreams.

## File map

```
deploy/
├── Dockerfile                          gateway image (no torch, no vllm)
├── docker-compose.yml                  three-service stack
├── alerts/                             Prometheus alerting rules
├── dashboards/                         Grafana dashboard JSON
└── README.md                           this file
```

The legacy `server/` directory of host-mode launcher scripts
(`start_gateway.sh`, `start_gemma4_default.sh`,
`start_gemma4_structured_k4.sh`) is intentionally absent from the
public tree as of Phase 3 bucket 5 (2026-05-21). Per the
architectural invariant in
``sndr_private/planning/audits/RELOCATION_DESIGN_2026-05-21_RU.md``
§0.5 Rule 4: launcher / docker-run wrapper scripts must not live
under ``integrations/`` (which is for patch code only).

## How to launch vLLM upstreams

**Canonical (preferred)**: use the V2 profile renderer:

```bash
# vLLM default (MTP OFF, TQ-only, port 8101 by convention):
python3 -m sndr.cli.legacy profile render-launchers gemma4-31b-tq-default \
    > /tmp/start_gemma4_default.sh
bash /tmp/start_gemma4_default.sh

# vLLM structured (β′-A K=4, port 8102):
python3 -m sndr.cli.legacy profile render-launchers gemma4-31b-tq-mtp-structured-k4 \
    > /tmp/start_gemma4_structured_k4.sh
bash /tmp/start_gemma4_structured_k4.sh
```

The rendered launcher is byte-equivalent to the validated
hand-written ``start_g4_betaA_k1.sh`` (proved by the P1.7d
byte-equivalence test in
``tests/unit/cli/test_profile_render_launchers.py``).

**Operator-internal reference**: the maintainer's host scripts live
under ``sndr_private/scripts/launch/gateway/`` (gitignored —
operator-private maintainer tree). These are kept for legacy
compatibility and bench-rehearsal needs; new operators should use
``sndr profile render-launchers`` instead.

The gateway itself does not require a host script — bring it up with
``docker compose up -d gateway`` from the directory below.

## docker-compose alternative

```bash
cd deploy
docker compose up -d default                       # default-only (production)
docker compose --profile gateway --profile structured up -d  # full stack
docker compose down structured                     # instant rollback level 2
```

The compose file uses a docker bridge with named service hostnames
(`default`, `structured`) so the gateway resolves them by DNS.

## G4 container smoke (verified 2026-05-20)

```
✓ /healthz             status=200
✓ /readyz              status=200, default=up, structured=up
✓ /v1/models           passthrough, model id returned
✓ /metrics             prometheus exposition, genesis_routed_* present
✓ /admin/state         localhost-only enforced; full state JSON returned
✓ tool_json POST       routed to structured upstream
✓ free_chat POST       routed to default upstream
✓ tool_json stream=true SSE chunks pass through (3 chunks + [DONE])
```

## G5 — real-upstream integration plan

After G4 passes (which it has), G5 verifies the gateway routes
correctly when both upstreams are REAL vLLM containers, not mocks.

### G5 acceptance criteria

1. Boot vLLM default via ``sndr profile render-launchers
   gemma4-31b-tq-default`` → wait for ``/v1/models``
2. Boot vLLM structured via ``sndr profile render-launchers
   gemma4-31b-tq-mtp-structured-k4`` → wait for ``/v1/models`` AND
   guard log line ``verdict=FUNCTIONALLY_VALIDATED allowed=True``
3. Bring up gateway container (``docker compose up -d gateway``
   from this directory) with default+structured URLs pointing at
   the two real containers
4. Gateway smoke (same 6 cases as D2a):
   - C1 tool_json → structured (real β′-A acceptance trace fires)
   - C2 free chat → default
   - C3 stop structured → fallback to default
   - C4 force-default → default
   - C5 stash artifact → default
   - C6 Qwen-style → default
5. Verify metrics show real upstream latencies (no artifact mocking)
6. Acceptance trace (PN248 log inside structured container) shows
   non-zero accepted_per_req on tool_json requests routed via gateway

### G5 expected runtime

~10 minutes:
- default boot: ~3 min
- structured boot: ~3 min
- gateway boot: ~30 s
- smoke run: ~3 min
- cleanup: ~30 s

### G5 NOT in scope yet

- Multi-concurrency stress (1 client at a time)
- Long-context (default `max_model_len=4096`)
- TLS / auth-token rotation
- Production cutover (D2f)
- Bench (already done in artifact)

### G5 run trigger

Operator-driven, separate session. Requires server GPU availability.

## Rollback levels

| level | action | time |
|---|---|---|
| 1 | `curl -X POST :8100/admin/force-default` | 1 sec |
| 2 | `docker stop sndr-gemma4-structured-k4` | 5-15 sec (probe driven) |
| 3 | `docker stop sndr-gateway` | LB cutover |
| 4 | remove gateway from LB rotation; LB → :8101 directly | LB cutover |

## Env name reference

All envs use SNDR_* canonical. GENESIS_* aliases work with
deprecation warning. See gateway/README.md for the full env table.

## Failure modes & mitigations

| failure | mitigation |
|---|---|
| gateway can't reach upstream | docker network mode mismatch (compose vs host) — use one consistently |
| guard says KERNEL_STORAGE_DTYPE_MISMATCH | structured container env matrix doesn't match artifact's kv_plan — check skip-list and G4_71b/G4_75 flags |
| artifact not matched | model basename differs OR kv_plan changed — regenerate artifact or rename `model` |
| streaming chunks not flowing | check that upstream sets `content-type: text/event-stream`; gateway preserves it |
| admin endpoints return 403 from outside | expected behavior; set `SNDR_GATEWAY_ADMIN_ALLOW_REMOTE=1` if you need remote access |

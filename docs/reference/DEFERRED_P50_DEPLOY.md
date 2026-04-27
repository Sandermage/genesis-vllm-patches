# DEFERRED — P50 ResponseCacheMiddleware deployment

**Status:** RESEARCH COMPLETE, IMPLEMENTATION DEFERRED (per Sander 2026-04-27)
**Owner:** Sander
**Pickup trigger:** when we get back to homelab response-caching work

---

## TL;DR

P50 (`vllm/_genesis/middleware/response_cache_middleware.py`) is an ASGI
response cache designed to live in front of a FastAPI service. The
homelab already has the right place to mount it: **`genesis-proxy`
(port 8318)**, a Python FastAPI/uvicorn wrapper around `cliproxyapi`.

`cliproxyapi` itself is a Go binary (`eceasy/cli-proxy-api-plus:latest`)
with NO plugin/middleware surface and NO response-cache feature in its
config — so P50 cannot live there directly.

---

## Recommended deployment path (when we resume)

### Path A-prime — drop P50 into `genesis-proxy`

1. Copy
   `vllm/_genesis/middleware/response_cache_middleware.py`
   → `/home/sander/Genesis_Project/genesis_proxy/genesis_proxy/response_cache_middleware.py`
2. Edit `/home/sander/Genesis_Project/genesis_proxy/genesis_proxy/app.py`
   ~line 60, after `app = FastAPI(...)`:
   ```python
   from .response_cache_middleware import ResponseCacheMiddleware
   app.add_middleware(ResponseCacheMiddleware, ttl=300)
   ```
3. Add any new deps to
   `/home/sander/Genesis_Project/genesis_proxy/requirements.txt`
4. Add a unit test under
   `/home/sander/Genesis_Project/genesis_proxy/tests/`
   (TDD: cache hit returns 200 with `X-Cache: HIT`)
5. Rebuild + redeploy:
   `cd /home/sander/Genesis_Project && docker compose up -d --build genesis-proxy`
6. Verify: `curl http://192.168.1.10:8318/health` then send two
   identical `/v1/chat/completions` requests; confirm 2nd hits cache
   (response header `X-Cache: HIT`)

### Why NOT cliproxyapi (Path A — rejected)

- cliproxyapi is a Go binary, no Python plugin surface.
- `config.example.yaml` (22 KB) has ZERO response-cache options.
- Only cache-related keys are `cache-user-id` (Codex `user_id`) and
  `antigravity-signature-cache-enabled` (Antigravity thinking-block
  signature cache). Neither is a response cache.

### Why NOT a separate proxy in front (Path C — redundant)

- `genesis-proxy` (port 8318) already wraps `cliproxyapi:8317`.
- Already uses cache-pattern (`DedupCache` in `dedup_cache.py`,
  `sha256(model+messages)`).
- Mounting another proxy = pointless extra hop.

---

## Caveat to verify when we resume

P50 was designed as ASGI middleware for vLLM's FastAPI server.
`genesis-proxy` is also FastAPI/ASGI, so the contract matches — but
**verify the middleware does not assume vLLM-specific request shapes**
(e.g., `/v1/completions` vs `/v1/chat/completions`, `model` field
handling). If it does, adapt or wrap before mounting.

---

## File paths reference

- `/home/sander/Genesis_Project/genesis_proxy/genesis_proxy/app.py`
  (target for `add_middleware`)
- `/home/sander/Genesis_Project/genesis_proxy/genesis_proxy/dedup_cache.py`
  (reference pattern for keying)
- `/home/sander/Genesis_Project/genesis_proxy/Dockerfile`
  (no edits needed — middleware is just a Python file)
- `/home/sander/Genesis_Project/docker-compose.yml` lines 296–325
  (cliproxyapi + genesis-proxy services)
- Source to copy:
  `vllm/_genesis/middleware/response_cache_middleware.py`

---

## Pickup checklist (5 min when ready)

- [ ] cp middleware → genesis-proxy/genesis_proxy/
- [ ] Verify request-shape compatibility (chat/completions vs completions)
- [ ] Edit app.py to register middleware
- [ ] Add TTL/cache-key configuration to genesis-proxy `.env`
- [ ] Write TDD test (cache HIT returns 200 + X-Cache: HIT)
- [ ] Rebuild genesis-proxy container
- [ ] Curl-test golden path (two identical requests → 2nd is HIT)
- [ ] Add observability: log cache hit/miss count to dashboards

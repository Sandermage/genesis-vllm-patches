# SNDR backend handoff — chat reasoning fix + web-search integration

Second handoff (see `SNDR_BACKEND_HANDOFF.md` for the release-hardening one). This
one covers the chat **"(empty) response"** bug and the **internet-search** feature.
The GUI agent owns `gui/web/`; everything below is `sndr/` work for the backend
agent. Decisions locked with the operator: **search backend = self-hosted
SearXNG**, **trigger = manual 🌐 toggle + auto native tool-calling (model-aware)**.

---

> **STATUS UPDATE (implemented, not just specced):** §1 below is now DONE in
> `engine_client.py` (operator lifted the gui-only rule for this fix):
> `_iter_chat_events` captures `reasoning_content` → `{"reasoning":…}` events and
> adds `finish_reason` + `had_reasoning` to the `done` event; `engine_chat`
> returns `reasoning` too. Unit-tested (`test_engine_client.py`, 2 new tests) and
> live on the local daemon. **For prod it needs the host `sndr-daemon` rebuild**
> (same as the hardening in `SNDR_BACKEND_HANDOFF.md`). The GUI side (reasoning
> trace render, truncation note, history trim) is also shipped. **Only §2 (web
> search / SearXNG) remains for the backend agent.**

## 1. BUG — chat returns `(empty)` with a non-zero token count

### Root cause (confirmed in code + live config)

1. Every prod engine launches with `--reasoning-parser qwen3`
   (`compose/prod-35b.yml:168`, `prod-27b-tq.yml:166`, …). vLLM therefore routes
   the model's `<think>` span into `delta.reasoning_content` and the final answer
   into `delta.content` **only after** `</think>`.
2. The chat proxy reads **content only** —
   `sndr/product_api/legacy/engine_client.py:390`:
   `delta = (choice.get("delta") or {}).get("content")`. All `reasoning_content`
   is silently discarded.
3. The token figure shown in the GUI is `usage.completion_tokens`
   (`engine_client.py:404`) — i.e. **all** generated tokens, reasoning included.
4. For an unanswerable / real-time question (e.g. "crypto market today, 11 Jun
   2026") the model spends the whole `max_tokens` budget inside
   `reasoning_content` and never emits a `content` token. Zero `{delta}` lines
   reach the GUI → empty bubble, while the stat shows the full token count
   (`1024 tok`). Deterministic, not flaky — happens for any prompt that triggers
   long reasoning under a finite budget.

### GUI mitigation already shipped (no backend dependency)

`gui/web/src/Engine.tsx`: default `maxTokens` 512 → **2048** (reasoning headroom);
when `done` arrives with `tokens > 0` and empty content, the bubble now renders an
actionable advisory ("the model used all its tokens thinking… raise Max tokens or
enable web search") instead of a bare `(empty)`. This makes the failure legible
but does **not** recover the dropped reasoning — that needs the backend fix below.

### Backend fix required (in `engine_client.py`)

`stream_chat` (line ~344) and `engine_chat` / `chat_raw`:

1. **Capture reasoning.** In the streaming loop, also read
   `(choice.get("delta") or {}).get("reasoning_content")` and emit it as a
   distinct ND-JSON event so the GUI can render a collapsible "Thinking…" trace:

   ```python
   d = choice.get("delta") or {}
   if (rc := d.get("reasoning_content")):
       yield json.dumps({"reasoning": rc})
   if (content := d.get("content")):
       ...                                   # unchanged: yield {"delta": content}
   ```

2. **Surface truncation.** Capture `choice.get("finish_reason")` and include it in
   the final `done` event so the GUI can distinguish "ran out of budget mid-think"
   (`finish_reason == "length"` with no content) from an intentional empty answer:

   ```python
   yield json.dumps({"done": True, "finish_reason": finish_reason,
                     "had_reasoning": saw_reasoning, ...existing fields...})
   ```

3. **Non-stream parity.** `engine_chat` (line ~435) returns
   `message.content` only — also pass through `message.reasoning_content` and
   `finish_reason` so the non-streaming path isn't silently empty either.

   Keep the existing `tokens`/`usage` accounting; just stop throwing reasoning
   away. The GUI already tolerates unknown event keys, so adding `{"reasoning":…}`
   and new `done` fields is backward-compatible.

### Tests (extend `tests/unit/product_api/`)

- Feed a fake SSE stream whose deltas carry only `reasoning_content` then a final
  `content`; assert `stream_chat` yields `{"reasoning":…}` lines AND the final
  `{"delta":…}`, and that `done.finish_reason` propagates.
- Feed a stream that ends with `finish_reason="length"` and no `content`; assert
  `done` reports it (so the GUI advisory has ground truth instead of a heuristic).

---

## 2. FEATURE — internet search in chat (SearXNG, model-aware)

### Topology

```
GUI chat ──► daemon /api/v1/engine/chat/stream (+ web_search flag / tool)
                 │
                 ├─ manual toggle: search(query) → inject context → generate
                 └─ auto: expose web_search tool → model calls it → search → loop
                          (reuse run_copilot tool-loop, http_app.py:2491)
                 │
                 ▼
        legacy/web_search.py  ──►  SearXNG  (self-hosted, JSON API)
                 │                   http://<homelab-host>:8888/search?format=json
                 └─ fetch + extract top-N result pages → clean snippets + citations
```

### 2a. Stand up SearXNG (homelab)

- Deploy SearXNG via docker on the homelab (compose snippet to add under
  `Genesis_homelab_scripts`). Enable the **JSON** format in `settings.yml`
  (`search.formats: [html, json]`) and set a bind the daemon can reach.
- Config the daemon with `SNDR_SEARXNG_URL` (e.g. `http://127.0.0.1:8888`).
  No API key, no third-party egress beyond SearXNG's own upstream metasearch.

### 2b. `legacy/web_search.py` (new module — pure + cached)

- `search(query, *, k=5, timeout=8.0) -> list[SearchResult]` — GET
  `${SNDR_SEARXNG_URL}/search?q=…&format=json&safesearch=1`, take top-k
  `{title, url, content}`. Pure HTTP, mirror the read-only/timeout discipline of
  `engine_client`.
- `fetch_extract(url) -> str` — optional: pull the page, strip to main text
  (cap length). Start without it (SearXNG `content` snippets are usually enough);
  add behind a flag if answers need fuller context.
- `build_search_context(results) -> str` — mirror `chat_rag.buildRagContext`
  exactly: a numbered `[n] (url) snippet` block the model can cite. Reuse the
  GUI's existing `sources?: RagDoc[]` rendering — return results in the same
  `{ref, snippet}` shape so the 🌐 path lights up `SourcesRow` for free.
- Cache identical queries briefly (e.g. 60–120 s) to avoid hammering SearXNG on
  regenerate.

### 2c. Manual toggle path (deterministic, streams cleanly)

- Extend the chat payload with `web_search: bool` (+ optional `web_k`). When set,
  `stream_chat` (or a thin wrapper) runs `web_search.search(last_user_msg)` first,
  prepends `build_search_context(...)` as a system message, then proceeds exactly
  like the current RAG injection (`http_app.py` chat path already supports a
  retrieved-context system block). Return the citations to the GUI via a
  `{"sources": [...]}` ND-JSON event before the deltas.
- This path is independent of tool-calling, so it works on **every** model and
  streams without the `qwen3_coder` caveat below.

### 2d. Auto path — native tool-calling (the "model-aware" part)

- Engines already run `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
  (`compose/prod-35b.yml:162-167`), so the model can decide to search.
- Expose one OpenAI tool, `web_search(query: string)`. Reuse the existing copilot
  tool-loop (`copilot.run_copilot` with `chat_fn(msgs, tools=…)`,
  `http_app.py:2491`) so you don't reinvent the loop: model emits a `web_search`
  tool_call → backend runs `web_search.search` → feeds the result back as a
  `tool` message → model answers with citations.
- **Model-aware gating:** only advertise the tool when the resolved engine
  actually has a tool-call parser. `config_editor.py:644` already reads engine
  capabilities (`reasoning_parser`, and the tool parser is in the same caps) —
  expose a `supports_tools` / `supports_reasoning` flag on
  `/api/v1/capabilities` (or the engine resolve) so the GUI can enable "Auto
  search" only where it works, and fall back to the manual toggle elsewhere.
- **Known caveat (do not skip):** `qwen3_coder` drops `delta.tool_calls` in
  streaming (the GUI already warns about this at `Engine.tsx:930`, and patch
  `p59_qwen3_reasoning_tool_call_recovery.py` documents the streaming gap). So the
  auto path must run the **tool-decision round non-streamed** (collect the full
  tool_call), execute the search, then **stream the final answer** — OR serve with
  `--tool-call-parser qwen3_xml`. Pick one and note it; don't assume streamed
  tool_calls work on the current pin.

### 2e. Contract the GUI will depend on (so we build in lockstep)

The GUI agent will, at the 2-hour mark, build against these. Please keep the
shapes stable:

- `POST /api/v1/engine/chat/stream` accepts `web_search: bool` (manual path) and
  emits, in addition to `{"delta":…}` / `{"done":…}`:
  - `{"reasoning": "…"}` — thinking trace chunks (from §1 fix)
  - `{"sources": [{"ref": "<url>", "snippet": "…", "title": "…"}]}` — citations
  - `{"done": {..., "finish_reason": "...", "had_reasoning": bool}}`
- `GET /api/v1/capabilities` (or engine resolve) exposes `supports_tools: bool`
  and `supports_reasoning: bool` per engine.
- A standalone `POST /api/v1/web/search {query, k}` → `{results:[…]}` is useful for
  the GUI to preview/debug search independently (optional but cheap).

### 2f. Security / safety notes

- SearXNG egress is the only new outbound path; keep it server-side (the GUI never
  calls SearXNG directly — it goes through the daemon, preserving the existing
  same-origin + auth posture).
- Bound result count, snippet length, fetch timeout, and total tool-loop steps
  (`run_copilot` already has `max_steps`). Treat fetched page text as untrusted —
  it lands in the model context, not in any eval/exec path.
- Redact the SearXNG URL/key from any report bundle (follow the existing
  redaction in the report generator).

---

## 3. Split of work

| Layer | Owner | Items |
| --- | --- | --- |
| Chat reasoning capture + finish_reason | sndr agent | §1 backend fix + tests |
| `web_search.py` + SearXNG + tool-loop + caps flags | sndr agent | §2a–2d |
| SearXNG container / homelab compose | sndr / homelab | §2a |
| Empty-state advisory + maxTokens default | **GUI (done)** | shipped this turn |
| 🌐 toggle UI, reasoning trace render, sources row, auto-search gating, settings (SearXNG url) | **GUI (in 2h)** | builds against §2e contract |

#!/usr/bin/env python3
"""Diagnostic dual-backend proxy for isolating spec-decode regressions.

Listens on :9000 and forwards each incoming OpenAI-style chat completion
to TWO backends in parallel (typically prod and a spec-decode test
container on different ports), captures both responses byte-for-byte,
runs degenerate-pattern detection, and returns a structured diagnostic
record. The proxy itself returns whichever backend the operator selects
as primary; diagnostics are written to a JSONL log for offline review.

Why dual-forward instead of just logging one call: subtle output
corruption (e.g. extra XML wrapping, slight token-count differences,
different completion tail) only shows up when you compare the same
prompt across two backends. Single-stream logs miss it.

Usage:
    pip install fastapi uvicorn httpx
    BACKEND_A=http://localhost:8000 \\
    BACKEND_B=http://localhost:8001 \\
    PRIMARY=A \\
    LOG_FILE=/tmp/dual_proxy.jsonl \\
    python3 dual_backend_diagnostic_proxy.py

Then point your client at http://this-host:9000/v1/chat/completions and
each request gets fanned out to both backends. Read the JSONL log for
diagnostics.

Defaults:
  BACKEND_A=http://localhost:8000
  BACKEND_B=http://localhost:8001
  PRIMARY=A    (which backend's response to return to client)
  LOG_FILE=/tmp/dual_proxy.jsonl
  TIMEOUT_S=300
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


BACKEND_A = os.environ.get("BACKEND_A", "http://localhost:8000")
BACKEND_B = os.environ.get("BACKEND_B", "http://localhost:8001")
PRIMARY = os.environ.get("PRIMARY", "A").upper()
LOG_FILE = os.environ.get("LOG_FILE", "/tmp/dual_proxy.jsonl")
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", "300"))


app = FastAPI()


# ──────────────────────────────────────────────────────────────────────
# Degenerate-pattern detectors
# ──────────────────────────────────────────────────────────────────────

DEGENERATE_PATTERNS = [
    # Same word repeated 4+ times in a row, e.g. "amber amber amber amber"
    (re.compile(r"\b(\w+)\b(\s+\1\b){3,}", re.IGNORECASE), "word_loop_4plus"),
    # Same XML tag closed 3+ times in a row, e.g. </parameter></parameter></parameter>
    (re.compile(r"(</\w+>)\1{2,}"), "xml_close_loop_3plus"),
    # Same XML opening tag in a row
    (re.compile(r"(<\w+>)\1{2,}"), "xml_open_loop_3plus"),
    # <tool_call><tool_call><tool_call> (the noonghunna canonical loop)
    (re.compile(r"<tool_call>\s*<tool_call>\s*<tool_call>"), "tool_call_loop"),
    # Long repetition of the same character (>=10), e.g. "aaaaaaaaaa"
    (re.compile(r"(\S)\1{9,}"), "char_loop_10plus"),
    # Corrupted attribute syntax like <parameter=parameter=foo>
    (re.compile(r"<\w+=\w+=\w+"), "double_attr_corruption"),
]


def detect_degenerate(text: str) -> list[dict]:
    if not text:
        return []
    hits: list[dict] = []
    for rx, name in DEGENERATE_PATTERNS:
        m = rx.search(text)
        if m:
            hits.append(
                {"pattern": name, "match": m.group(0)[:100], "pos": m.start()}
            )
    return hits


def normalize_for_diff(s: str) -> str:
    """Drop whitespace + lowercase to compare structural shape across runs."""
    return re.sub(r"\s+", "", (s or "").lower())


# ──────────────────────────────────────────────────────────────────────
# Forward + capture
# ──────────────────────────────────────────────────────────────────────

async def forward_one(
    client: httpx.AsyncClient,
    base_url: str,
    body_bytes: bytes,
    headers: dict[str, str],
) -> dict:
    """Forward one request, capture full response + timing."""
    t0 = time.perf_counter()
    out: dict[str, Any] = {
        "url": base_url,
        "ttft_s": None,
        "total_s": None,
        "http_status": None,
        "error": None,
        "body": None,
        "headers": None,
    }
    try:
        resp = await client.post(
            f"{base_url}/v1/chat/completions",
            content=body_bytes,
            headers=headers,
            timeout=TIMEOUT_S,
        )
        out["http_status"] = resp.status_code
        out["headers"] = dict(resp.headers)
        out["body"] = resp.content.decode("utf-8", errors="replace")
        out["total_s"] = time.perf_counter() - t0
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["total_s"] = time.perf_counter() - t0
    return out


def extract_assistant_message(resp_body: str) -> dict | None:
    """Pull message from one OpenAI-style chat completion response."""
    try:
        d = json.loads(resp_body)
    except Exception:
        return None
    try:
        msg = d["choices"][0]["message"]
    except (KeyError, IndexError):
        return None
    return {
        "content": msg.get("content"),
        "reasoning": msg.get("reasoning"),
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": d["choices"][0].get("finish_reason"),
        "usage": d.get("usage"),
        "model": d.get("model"),
    }


def diff_messages(a: dict, b: dict) -> dict:
    """Compute structured diff of two assistant messages."""
    out: dict[str, Any] = {}
    for k in ("content", "reasoning", "finish_reason", "model"):
        va, vb = a.get(k), b.get(k)
        if va != vb:
            out[k] = {
                "A": (va[:200] + ("…" if va and len(va) > 200 else "")) if isinstance(va, str) else va,
                "B": (vb[:200] + ("…" if vb and len(vb) > 200 else "")) if isinstance(vb, str) else vb,
            }
    # Tool call comparison: counts + first-call function name + arguments shape
    tca, tcb = a.get("tool_calls", []), b.get("tool_calls", [])
    if len(tca) != len(tcb):
        out["tool_calls_count"] = {"A": len(tca), "B": len(tcb)}
    elif tca:
        for i, (ca, cb) in enumerate(zip(tca, tcb)):
            fa = ca.get("function", {}) if isinstance(ca, dict) else {}
            fb = cb.get("function", {}) if isinstance(cb, dict) else {}
            if fa.get("name") != fb.get("name"):
                out[f"tool_call_{i}_name"] = {"A": fa.get("name"), "B": fb.get("name")}
            if fa.get("arguments") != fb.get("arguments"):
                out[f"tool_call_{i}_args"] = {
                    "A": (fa.get("arguments") or "")[:200],
                    "B": (fb.get("arguments") or "")[:200],
                }
    # Usage delta
    ua, ub = a.get("usage") or {}, b.get("usage") or {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if ua.get(k) != ub.get(k):
            out[f"usage_{k}"] = {"A": ua.get(k), "B": ub.get(k)}
    # Structural normalised content shape
    na = normalize_for_diff(a.get("content") or "")
    nb = normalize_for_diff(b.get("content") or "")
    if na != nb:
        out["content_normalised_differs"] = True
    return out


# ──────────────────────────────────────────────────────────────────────
# HTTP entry
# ──────────────────────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.body()
    headers = {k: v for k, v in req.headers.items() if k.lower() in (
        "content-type", "authorization", "accept",
    )}
    req_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    try:
        body_json = json.loads(body)
    except Exception:
        body_json = {"_unparseable": True}

    async with httpx.AsyncClient() as client:
        a_task = forward_one(client, BACKEND_A, body, headers)
        b_task = forward_one(client, BACKEND_B, body, headers)
        a, b = await asyncio.gather(a_task, b_task)

    # Extract structured messages
    msg_a = extract_assistant_message(a.get("body") or "") if a.get("http_status") == 200 else None
    msg_b = extract_assistant_message(b.get("body") or "") if b.get("http_status") == 200 else None

    # Detect degenerate patterns in each
    deg_a = detect_degenerate((msg_a or {}).get("content") or "")
    deg_b = detect_degenerate((msg_b or {}).get("content") or "")

    # Diff if both succeeded
    diff = diff_messages(msg_a, msg_b) if (msg_a and msg_b) else None

    record = {
        "req_id": req_id,
        "t_start": t0,
        "model": body_json.get("model"),
        "messages_preview": [
            {"role": m.get("role"), "content_len": len(m.get("content") or "")}
            for m in body_json.get("messages", [])[-3:]
        ],
        "max_tokens": body_json.get("max_tokens"),
        "tools_count": len(body_json.get("tools") or []),
        "spec_decode_in_request": "speculative_config" in body_json or "speculative" in body_json,
        "A": {
            "url": BACKEND_A,
            "http_status": a.get("http_status"),
            "ttft_s": a.get("ttft_s"),
            "total_s": a.get("total_s"),
            "error": a.get("error"),
            "msg": msg_a,
            "degenerate": deg_a,
        },
        "B": {
            "url": BACKEND_B,
            "http_status": b.get("http_status"),
            "ttft_s": b.get("ttft_s"),
            "total_s": b.get("total_s"),
            "error": b.get("error"),
            "msg": msg_b,
            "degenerate": deg_b,
        },
        "diff": diff,
    }

    # Append to log
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass

    # Return primary backend's response to client
    primary = a if PRIMARY == "A" else b
    if primary.get("http_status"):
        return JSONResponse(
            content=json.loads(primary["body"]) if primary.get("body") else {},
            status_code=primary["http_status"],
            headers={
                "x-proxy-req-id": req_id,
                "x-proxy-primary": PRIMARY,
                "x-proxy-degenerate-A": ",".join(d["pattern"] for d in deg_a) or "none",
                "x-proxy-degenerate-B": ",".join(d["pattern"] for d in deg_b) or "none",
                "x-proxy-diff-keys": ",".join((diff or {}).keys()) or "none",
            },
        )
    else:
        return JSONResponse(
            content={"error": primary.get("error") or "backend failed"},
            status_code=502,
            headers={"x-proxy-req-id": req_id},
        )


@app.get("/health")
async def health():
    return {"status": "ok", "backend_a": BACKEND_A, "backend_b": BACKEND_B}


@app.get("/")
async def root():
    return {
        "service": "dual-backend diagnostic proxy",
        "backend_a": BACKEND_A,
        "backend_b": BACKEND_B,
        "primary": PRIMARY,
        "log_file": LOG_FILE,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="warning")

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Streaming tool-call drop probe (the club-3090 #145 residual, vllm#39598).

club-3090's issue #145 close-out: on stock vLLM >= v0.22, tool_choice=auto +
streaming is fixed natively, BUT `tool_choice="required"` + thinking + STREAMING
still drops a large fraction of tool calls when MTP speculative decoding is on
(they measured 13/20 dropped with MTP n=3; 0/20 without MTP; parser-independent).
Genesis P64 was the historical mitigation and is version-capped off on pins
where the engine-native parser owns streaming — so our 35B (MTP K=5 + qwen3_xml
+ tool agents) is directly exposed IF the residual reproduces on our stack.

Nothing in our gates catches this: the fleet boot-smoke tool-call check is
non-streaming with tool_choice=auto. This probe runs N streaming requests with
tool_choice="required" and counts turns that finish WITHOUT a tool_call delta —
the drop signature (content-channel leak or empty finish).

Exit 0 = no drops; exit 1 = drops detected (report the ratio to the operator).

Usage:
  python3 scripts/stream_toolcall_probe.py --n 20 \
      --base-url http://127.0.0.1:8102 --api-key genesis-local
"""
from __future__ import annotations

import argparse
import json
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

PROMPTS = [
    "What is the weather in Berlin right now?",
    "Check the current weather in Tokyo for me.",
    "I need today's weather in Paris.",
    "How is the weather in New York at the moment?",
]


def _one_stream(base_url: str, api_key: str, model: str, prompt: str,
                timeout: float) -> dict:
    """One streaming turn; returns {tool_call, content_leak, finish, error}."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "tool_choice": "required",
        "stream": True,
        "max_tokens": 300,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {api_key}", "content-type": "application/json"},
    )
    saw_tool = False
    content_buf: list[str] = []
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("tool_calls"):
                        saw_tool = True
                    if delta.get("content"):
                        content_buf.append(delta["content"])
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
    except Exception as e:  # noqa: BLE001
        return {"tool_call": False, "content_leak": "", "finish": None, "error": str(e)[:120]}
    leak = "".join(content_buf).strip()
    return {"tool_call": saw_tool, "content_leak": leak[:80], "finish": finish, "error": None}


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming required-tool-call drop probe.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8102")
    ap.add_argument("--api-key", default="genesis-local")
    ap.add_argument("--model", default="", help="served model name (auto-detected when empty)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=120.0)
    a = ap.parse_args()

    model = a.model
    if not model:
        req = urllib.request.Request(
            f"{a.base_url.rstrip('/')}/v1/models",
            headers={"Authorization": f"Bearer {a.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            model = json.load(r)["data"][0]["id"]

    drops = 0
    leaks = 0
    errors = 0
    for i in range(a.n):
        res = _one_stream(a.base_url, a.api_key, model, PROMPTS[i % len(PROMPTS)], a.timeout)
        status = "OK  " if res["tool_call"] else "DROP"
        if not res["tool_call"]:
            drops += 1
        if res["content_leak"] and "<tool_call" in res["content_leak"]:
            leaks += 1
        if res["error"]:
            errors += 1
        detail = res["error"] or (f"leak={res['content_leak']!r}" if res["content_leak"] else f"finish={res['finish']}")
        print(f"[{i + 1:>2}/{a.n}] {status}  {detail}")

    print(f"\nresult: {drops}/{a.n} dropped, {leaks} content-channel XML leaks, {errors} errors")
    if drops:
        print("DROPS DETECTED — the vllm#39598 residual (club-3090 #145 close-out) "
              "reproduces on this stack: tool_choice=required + streaming + MTP. "
              "Mitigations: re-scope P64's version cap, or route required-mode "
              "agents through non-streaming until the upstream fix lands.")
    return 1 if drops else 0


if __name__ == "__main__":
    raise SystemExit(main())

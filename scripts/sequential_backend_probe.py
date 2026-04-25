#!/usr/bin/env python3
"""Sequential backend probe — runs the same test set against the currently
active vLLM backend at http://localhost:8000, captures every byte of the
response, writes one JSONL line per probe.

Run it twice with different `--label` (e.g. `prod` vs `specdec_p56`) and
diff the outputs offline to surface subtle output corruption.

Usage:
    python3 sequential_backend_probe.py \\
        --host http://localhost:8000 \\
        --api-key genesis-local \\
        --model qwen3.6-35b-a3b \\
        --label prod \\
        --out /tmp/probe_prod.jsonl

    # ... switch backend (down prod, up spec-decode test) ...

    python3 sequential_backend_probe.py \\
        --model qwen3.6-35b-a3b-specdec \\
        --label specdec_p56 \\
        --out /tmp/probe_specdec_p56.jsonl

    python3 sequential_backend_probe.py --diff \\
        /tmp/probe_prod.jsonl /tmp/probe_specdec_p56.jsonl

Probe set covers spec-decode-sensitive shapes:
- bare narrative
- structured tool call (no-thinking)
- structured tool call (thinking on, default Qwen3)
- needle recall short (1k filler)
- needle recall long (10k filler)
- repetitive forced-token
- streaming (TODO if useful)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from typing import Any


PROBES: list[dict[str, Any]] = [
    {
        "name": "smoke_hello",
        "messages": [{"role": "user", "content": "Reply with the single word HELLO and nothing else."}],
        "max_tokens": 16,
        "thinking": False,
    },
    {
        "name": "narrative_short",
        "messages": [{"role": "user", "content": "Write exactly two sentences about a cat hunting a mouse. Plain prose only."}],
        "max_tokens": 100,
        "thinking": False,
    },
    {
        "name": "tool_call_simple",
        "messages": [{"role": "user", "content": "What is the weather in Paris in Celsius? Use the get_weather function."}],
        "max_tokens": 256,
        "thinking": False,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}, "unit": {"type": "string"}
                }}
            }
        }],
    },
    {
        "name": "tool_call_thinking_on",
        "messages": [{"role": "user", "content": "What is the weather in Berlin in Celsius? Use the get_weather function."}],
        "max_tokens": 1024,
        "thinking": True,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}, "unit": {"type": "string"}
                }}
            }
        }],
    },
    {
        "name": "json_structured",
        "messages": [{
            "role": "user",
            "content": "Reply with a single JSON object with exactly these keys: name (string), age (int), city (string). Use values: name=Alice, age=30, city=Paris. Reply ONLY with the JSON, no markdown."
        }],
        "max_tokens": 64,
        "thinking": False,
    },
    {
        "name": "needle_short_1k",
        "messages": [{
            "role": "user",
            "content": "The secret password is RUBY-PHOENIX-77.\n\n" +
                       ("the quick brown fox jumps over the lazy dog. " * 100) +
                       "\n\nQuestion: what was the secret password mentioned at the very start? Reply with only the password code."
        }],
        "max_tokens": 32,
        "thinking": False,
    },
    {
        "name": "needle_medium_5k",
        "messages": [{
            "role": "user",
            "content": "The secret password is RUBY-PHOENIX-77.\n\n" +
                       ("the quick brown fox jumps over the lazy dog. " * 500) +
                       "\n\nQuestion: what was the secret password mentioned at the very start? Reply with only the password code."
        }],
        "max_tokens": 32,
        "thinking": False,
    },
    {
        "name": "code_quicksort",
        "messages": [{
            "role": "user",
            "content": "Write a Python function that sorts a list using quicksort. Reply with only the function definition, no explanation."
        }],
        "max_tokens": 200,
        "thinking": False,
    },
    {
        "name": "structured_xml",
        "messages": [{
            "role": "user",
            "content": "Output an XML element <person name=\"Alice\" age=\"30\"/> on a single line, nothing else."
        }],
        "max_tokens": 32,
        "thinking": False,
    },
]


DEGENERATE_PATTERNS = [
    (re.compile(r"\b(\w+)\b(\s+\1\b){3,}", re.IGNORECASE), "word_loop_4plus"),
    (re.compile(r"(</\w+>)\1{2,}"), "xml_close_loop_3plus"),
    (re.compile(r"(<\w+>)\1{2,}"), "xml_open_loop_3plus"),
    (re.compile(r"<tool_call>\s*<tool_call>\s*<tool_call>"), "tool_call_loop"),
    (re.compile(r"(\S)\1{9,}"), "char_loop_10plus"),
    (re.compile(r"<\w+=\w+=\w+"), "double_attr_corruption"),
]


def detect_degenerate(text: str) -> list[dict]:
    if not text:
        return []
    return [
        {"pattern": name, "match": m.group(0)[:100], "pos": m.start()}
        for rx, name in DEGENERATE_PATTERNS
        for m in [rx.search(text)] if m
    ]


def call_probe(host: str, api_key: str, model: str, probe: dict) -> dict:
    """Issue one probe, return full structured record."""
    body = {
        "model": model,
        "messages": probe["messages"],
        "max_tokens": probe.get("max_tokens", 256),
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": probe.get("thinking", False)},
    }
    if probe.get("tools"):
        body["tools"] = probe["tools"]

    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    t0 = time.perf_counter()
    record: dict[str, Any] = {
        "name": probe["name"],
        "model_requested": model,
        "thinking": probe.get("thinking", False),
        "max_tokens": probe.get("max_tokens", 256),
        "tools_count": len(probe.get("tools", []) or []),
        "ts": time.time(),
        "http_status": None,
        "error": None,
        "total_s": None,
        "response": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            record["http_status"] = resp.status
            data = json.loads(resp.read().decode())
            record["response"] = data
        record["total_s"] = round(time.perf_counter() - t0, 3)
    except urllib.error.HTTPError as e:
        record["http_status"] = e.code
        record["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
        record["total_s"] = round(time.perf_counter() - t0, 3)
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        record["total_s"] = round(time.perf_counter() - t0, 3)

    # Extract message + degenerate patterns
    msg = None
    if record.get("response") and isinstance(record["response"], dict):
        try:
            msg = record["response"]["choices"][0]["message"]
        except (KeyError, IndexError):
            msg = None
    if msg:
        record["msg_summary"] = {
            "content_len": len(msg.get("content") or ""),
            "reasoning_len": len(msg.get("reasoning") or ""),
            "tool_calls_count": len(msg.get("tool_calls") or []),
            "finish_reason": record["response"]["choices"][0].get("finish_reason"),
        }
        record["degenerate"] = detect_degenerate(msg.get("content") or "")
        record["tool_calls"] = msg.get("tool_calls") or []
    return record


def cmd_run(args):
    out_path = args.out
    label = args.label
    print(f"[{label}] running {len(PROBES)} probes against {args.host}, model={args.model}")
    with open(out_path, "w") as f:
        for probe in PROBES:
            print(f"  → {probe['name']}", end=" ", flush=True)
            rec = call_probe(args.host, args.api_key, args.model, probe)
            rec["label"] = label
            f.write(json.dumps(rec, default=str) + "\n")
            ms = rec.get("msg_summary") or {}
            deg = rec.get("degenerate") or []
            print(
                f"{rec['http_status']} {rec.get('total_s')}s "
                f"content={ms.get('content_len')}c "
                f"tcs={ms.get('tool_calls_count')} "
                f"deg={len(deg)}"
            )
    print(f"[{label}] wrote {out_path}")


def cmd_diff(args):
    a_path, b_path = args.left, args.right
    a_recs = {r["name"]: r for r in (json.loads(line) for line in open(a_path) if line.strip())}
    b_recs = {r["name"]: r for r in (json.loads(line) for line in open(b_path) if line.strip())}

    print(f"=== Diff {a_path} vs {b_path} ===")
    print(f"{'probe':<28} {'A.deg':<8} {'B.deg':<8} {'A.tcs':<6} {'B.tcs':<6} {'A.fin':<14} {'B.fin':<14} {'shape':<8}")
    print("-" * 110)

    for name in sorted(set(a_recs) | set(b_recs)):
        a, b = a_recs.get(name), b_recs.get(name)
        if not a or not b:
            print(f"{name:<28} {'MISS' if not a else 'OK':<8} {'MISS' if not b else 'OK':<8}")
            continue
        ams, bms = a.get("msg_summary") or {}, b.get("msg_summary") or {}
        ad, bd = a.get("degenerate") or [], b.get("degenerate") or []

        a_content = (a.get("response") or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
        b_content = (b.get("response") or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
        norm_a = re.sub(r"\s+", "", a_content.lower())
        norm_b = re.sub(r"\s+", "", b_content.lower())
        shape = "same" if norm_a == norm_b else "DIFF"

        print(
            f"{name:<28} "
            f"{len(ad):<8} {len(bd):<8} "
            f"{ams.get('tool_calls_count', '?'):<6} {bms.get('tool_calls_count', '?'):<6} "
            f"{(ams.get('finish_reason') or '-'):<14} {(bms.get('finish_reason') or '-'):<14} "
            f"{shape:<8}"
        )
    print()
    # Detailed diff for divergent probes
    print("=== Detailed divergent probes ===")
    for name in sorted(set(a_recs) | set(b_recs)):
        a, b = a_recs.get(name), b_recs.get(name)
        if not (a and b):
            continue
        a_content = (a.get("response") or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
        b_content = (b.get("response") or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
        a_tcs = (a.get("response") or {}).get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
        b_tcs = (b.get("response") or {}).get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
        norm_a = re.sub(r"\s+", "", a_content.lower())
        norm_b = re.sub(r"\s+", "", b_content.lower())
        if norm_a == norm_b and len(a_tcs) == len(b_tcs):
            continue
        print(f"\n--- {name} ---")
        print(f"  A.content: {repr(a_content[:300])}")
        print(f"  B.content: {repr(b_content[:300])}")
        if a_tcs or b_tcs:
            print(f"  A.tool_calls[0]: {json.dumps(a_tcs[0] if a_tcs else None)[:200]}")
            print(f"  B.tool_calls[0]: {json.dumps(b_tcs[0] if b_tcs else None)[:200]}")
        if a.get("degenerate") or b.get("degenerate"):
            print(f"  A.degenerate: {a.get('degenerate')}")
            print(f"  B.degenerate: {b.get('degenerate')}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--host", default="http://localhost:8000")
    p_run.add_argument("--api-key", default="genesis-local")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--label", required=True)
    p_run.add_argument("--out", required=True)
    p_run.set_defaults(func=cmd_run)

    p_diff = sub.add_parser("diff")
    p_diff.add_argument("left")
    p_diff.add_argument("right")
    p_diff.set_defaults(func=cmd_diff)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

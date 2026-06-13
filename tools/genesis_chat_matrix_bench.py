#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Chat-type speed matrix — measure TTFT / TPOT / wall_TPS per chat variant.

Answers the operator question "which speed do I get for which kind of
chat" on a live Genesis vLLM deployment. Variants covered:

  short_chat      short prompt -> ~200 tok answer (typical chat turn)
  long_gen        short prompt -> 1024 tok answer (storytelling / codegen)
  code_gen        code prompt  -> 512 tok code answer
  long_ctx_8k     ~8K-token prompt -> 256 tok answer (doc QA)
  long_ctx_32k    ~32K-token prompt -> 256 tok answer (long-doc QA)
  multi_turn      5-turn accumulated history -> 256 tok (prefix-cache path)
  thinking_on     reasoning enabled  -> 512 tok
  thinking_off    reasoning disabled -> 512 tok
  tool_call       2-tool schema, forced tool answer

Each variant runs `--runs` streaming requests (default 5) and reports
mean/median TTFT, decode TPOT, wall TPS, output tokens, and — when P89
(vendor of vllm#45471) is enabled on the server — the mean reasoning
tokens per request, read from
`usage.completion_tokens_details.reasoning_tokens`. The reasoning column
is the reasoning-vs-answer TPOT-attribution denominator and the
MTP/TurboQuant tuning lever; it shows "-" when P89 is off or the model
has no reasoning parser.

Usage:
  python3 tools/genesis_chat_matrix_bench.py \
      --url http://localhost:8102/v1 --api-key genesis-local \
      --model qwen3.6-35b-a3b --runs 5 --md-out /tmp/chat_matrix.md
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
import urllib.request


def _sse_stream(url: str, api_key: str, payload: dict, timeout: float = 600.0):
    """POST payload, yield (event_time, data_dict) per SSE data line."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            try:
                yield time.perf_counter(), json.loads(body)
            except json.JSONDecodeError:
                continue


def measure_chat(base_url: str, api_key: str, payload: dict):
    """One streaming chat request -> dict(ttft_ms, tpot_ms, wall_tps, tokens).

    Token counts come from the final `usage` chunk (stream_options), NOT
    from counting SSE deltas — with MTP spec-decode one delta can carry
    several accepted tokens, so chunk-count understates tokens ~3x.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = dict(payload)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    first = None
    last = None
    n_chunks = 0
    finish = None
    completion_tokens = None
    # [Genesis P89] reasoning-token attribution from the final usage chunk.
    # P89 (vendor of vllm#45471) surfaces
    # usage.completion_tokens_details.reasoning_tokens on the chat path for
    # reasoning models (qwen3 on all 4 PROD models). It is None when P89 is
    # disabled or the model has no reasoning parser, so the bench degrades
    # cleanly to "no attribution" rather than failing.
    reasoning_tokens = None
    for t, data in _sse_stream(url, api_key, payload):
        usage = data.get("usage")
        if usage and usage.get("completion_tokens"):
            completion_tokens = usage["completion_tokens"]
        if usage:
            details = usage.get("completion_tokens_details")
            if details and details.get("reasoning_tokens") is not None:
                reasoning_tokens = details["reasoning_tokens"]
        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        # vllm 0.22.x renamed the reasoning-parser SSE field
        # reasoning_content -> reasoning; accept both.
        if (delta.get("content") or delta.get("reasoning_content")
                or delta.get("reasoning") or delta.get("tool_calls")):
            if first is None:
                first = t
            last = t
            n_chunks += 1
        if choices[0].get("finish_reason"):
            finish = choices[0]["finish_reason"]
    if first is None or last is None or n_chunks < 2:
        return None
    ttft = first - t0
    decode_s = last - first
    wall = last - t0
    tokens = completion_tokens or n_chunks
    return {
        "ttft_ms": ttft * 1e3,
        "tpot_ms": (decode_s / max(tokens - 1, 1)) * 1e3,
        "wall_tps": tokens / wall,
        "tokens": tokens,
        # [Genesis P89] reasoning vs answer split. -1 means "unavailable"
        # (P89 off / non-reasoning model); a real count enables the
        # reasoning/answer TPOT-attribution column in the report.
        "reasoning_tokens": reasoning_tokens if reasoning_tokens is not None else -1,
        "finish": finish or "?",
    }


def synth_text(n_words: int, seed: int) -> str:
    """Deterministic word salad — avoids prefix-cache collisions between runs."""
    rng = random.Random(seed)
    vocab = (
        "system kernel tensor stream graph batch decode prefill cache page "
        "block latency throughput memory bandwidth schedule rotate quantize "
        "attention recurrent gated delta hybrid expert router projection"
    ).split()
    return " ".join(rng.choice(vocab) for _ in range(n_words))


def build_variants(model: str, runs: int):
    base = {"model": model, "temperature": 0.7}
    tools = [
        {
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
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]

    def chat(msgs, max_tokens, **extra):
        p = dict(base)
        p["messages"] = msgs
        p["max_tokens"] = max_tokens
        p.update(extra)
        return p

    variants = {}
    variants["short_chat"] = [
        chat([{"role": "user", "content": f"Explain in a few sentences why the sky is blue. (run {i})"}], 200)
        for i in range(runs)
    ]
    variants["long_gen"] = [
        chat([{"role": "user", "content": f"Write a detailed story about a robot learning to paint. (run {i})"}], 1024)
        for i in range(runs)
    ]
    variants["code_gen"] = [
        chat([{"role": "user", "content": f"Write a Python class implementing an LRU cache with O(1) ops, with tests. (variant {i})"}], 512)
        for i in range(runs)
    ]
    variants["long_ctx_8k"] = [
        chat([{"role": "user", "content": synth_text(6000, seed=100 + i) + "\n\nSummarize the main repeated themes of the text above in 5 bullet points."}], 256)
        for i in range(runs)
    ]
    variants["long_ctx_32k"] = [
        chat([{"role": "user", "content": synth_text(24000, seed=200 + i) + "\n\nSummarize the main repeated themes of the text above in 5 bullet points."}], 256)
        for i in range(runs)
    ]
    history = []
    for turn in range(4):
        history.append({"role": "user", "content": f"Fact {turn}: the {turn}-th component is called part-{turn}. Acknowledge briefly."})
        history.append({"role": "assistant", "content": f"Noted: part-{turn} is component {turn}."})
    variants["multi_turn"] = [
        chat(history + [{"role": "user", "content": f"List all parts mentioned so far and their numbers. (run {i})"}], 256)
        for i in range(runs)
    ]
    variants["thinking_on"] = [
        chat([{"role": "user", "content": f"If a train leaves at 14:05 averaging 73 km/h and another at 14:35 at 91 km/h on the same 200 km route, which arrives first? (run {i})"}], 512,
             chat_template_kwargs={"enable_thinking": True})
        for i in range(runs)
    ]
    variants["thinking_off"] = [
        chat([{"role": "user", "content": f"If a train leaves at 14:05 averaging 73 km/h and another at 14:35 at 91 km/h on the same 200 km route, which arrives first? (run {i})"}], 512,
             chat_template_kwargs={"enable_thinking": False})
        for i in range(runs)
    ]
    variants["tool_call"] = [
        chat([{"role": "user", "content": f"What's the weather in Odessa right now? (run {i})"}], 256,
             tools=tools)
        for i in range(runs)
    ]
    return variants


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8102/v1")
    ap.add_argument("--api-key", default="genesis-local")
    ap.add_argument("--model", default="qwen3.6-35b-a3b")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tag", default="chat_matrix")
    ap.add_argument("--md-out", default="/tmp/chat_matrix.md")
    args = ap.parse_args()

    variants = build_variants(args.model, args.runs)
    rows = []
    for name, payloads in variants.items():
        results = []
        for p in payloads:
            try:
                r = measure_chat(args.url, args.api_key, p)
            except Exception as e:  # noqa: BLE001
                print(f"[{name}] request error: {e}", file=sys.stderr)
                r = None
            if r:
                results.append(r)
        if not results:
            rows.append((name, None))
            print(f"[{name}] ALL FAILED")
            continue
        # [Genesis P89] mean reasoning tokens over the runs that reported it
        # (>= 0). None when no run had attribution available (P89 off or a
        # non-reasoning model) -> the report shows "-".
        reasoning_vals = [x["reasoning_tokens"] for x in results if x["reasoning_tokens"] >= 0]
        reasoning_mean = st.mean(reasoning_vals) if reasoning_vals else None
        agg = {
            "n": len(results),
            "ttft_ms": st.mean(x["ttft_ms"] for x in results),
            "ttft_med": st.median(x["ttft_ms"] for x in results),
            "tpot_ms": st.mean(x["tpot_ms"] for x in results),
            "wall_tps": st.mean(x["wall_tps"] for x in results),
            "wall_med": st.median(x["wall_tps"] for x in results),
            "tokens": st.mean(x["tokens"] for x in results),
            "reasoning": reasoning_mean,
            "finish": results[0]["finish"],
        }
        rows.append((name, agg))
        reason_str = f" reason={agg['reasoning']:.0f}" if agg["reasoning"] is not None else ""
        print(f"[{name}] n={agg['n']} TTFT={agg['ttft_ms']:.1f}ms TPOT={agg['tpot_ms']:.3f}ms wall={agg['wall_tps']:.1f}TPS tok={agg['tokens']:.0f}{reason_str}")

    lines = [
        f"# Chat-type speed matrix — {args.tag}",
        "",
        f"Model `{args.model}` · runs/variant = {args.runs} · streaming chat completions",
        "",
        # The `reason tok` column keys on P89's
        # completion_tokens_details.reasoning_tokens (vendor of vllm#45471):
        # mean reasoning tokens per request, "-" when unavailable. It is the
        # denominator for reasoning-vs-answer TPOT attribution and the
        # MTP/TurboQuant tuning lever the P89 spec calls for.
        "| variant | n | TTFT ms (med) | TPOT ms | wall TPS (med) | tokens | reason tok | finish |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, agg in rows:
        if agg is None:
            lines.append(f"| {name} | 0 | FAIL | - | - | - | - | - |")
        else:
            reason_cell = f"{agg['reasoning']:.0f}" if agg["reasoning"] is not None else "-"
            lines.append(
                f"| {name} | {agg['n']} | {agg['ttft_ms']:.1f} ({agg['ttft_med']:.1f}) "
                f"| {agg['tpot_ms']:.3f} | {agg['wall_tps']:.1f} ({agg['wall_med']:.1f}) "
                f"| {agg['tokens']:.0f} | {reason_cell} | {agg['finish']} |"
            )
    md = "\n".join(lines) + "\n"
    with open(args.md_out, "w") as f:
        f.write(md)
    print(f"\nWritten: {args.md_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

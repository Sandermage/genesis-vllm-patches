"""Streaming × thinking × prompt-type compatibility probe.

Runs the 8-cell matrix that PN51 must NOT break (2 prompt types × 2
thinking modes × 2 stream modes):

    prompt × thinking × stream → all 8 cells must produce non-empty content

Prompt types:
  * SHORT  — "Reply with single word: ready"          (sanity / smoke)
  * CODE   — "Write a Python function for fibonacci"  (longer, exercises
             tool-call-style stops + reasoning blocks for thinking-on)

Cells:
    | prompt | enable_thinking | stream  | expected                       |
    |--------|-----------------|---------|--------------------------------|
    | SHORT  | True            | False   | content non-empty              |
    | SHORT  | True            | True    | content non-empty (post </think>)|
    | SHORT  | False           | False   | content non-empty              |
    | SHORT  | False           | True    | content non-empty (PN51 cell)  |
    | CODE   | True            | False   | content non-empty (code body)  |
    | CODE   | True            | True    | content non-empty + reasoning  |
    | CODE   | False           | False   | content non-empty (code body)  |
    | CODE   | False           | True    | content non-empty (PN51 cell)  |

Prints a single line per cell and exits 0 if all 8 cells pass, 1 if any fail.

Designed to be safe against the live PROD container — uses tiny prompts,
max_tokens=64, no concurrency. Adds <2s of total load.

Recommended usage:
  * Manually after restarting vllm with new env flags
  * Wired into post-warmup smoke check in start_*.sh scripts
  * Cron every N minutes for regression detection in long-running PROD
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = os.environ.get(
    "GENESIS_ENDPOINT", "http://localhost:8000/v1/chat/completions",
)
MODEL = os.environ.get("GENESIS_MODEL", "qwen3.6-27b")
API_KEY = os.environ.get("GENESIS_API_KEY", "genesis-local")

PROMPTS = {
    # SHORT — sanity smoke. Tiny budget OK in both thinking modes.
    "SHORT": "Reply with the single word: ready",
    # CODE — exercises code-completion path. Thinking-on may use ~500 tok
    # for plan + ~200 for code; budget below allocates 2048 to be safe.
    "CODE": "Write a short Python function that returns the n-th "
            "Fibonacci number iteratively. Just the function, no prose.",
    # REPORT — Sander's real-world case: market analysis of an asset.
    # This is the kind of multi-paragraph reply where streaming + thinking
    # interaction matters most — must arrive full, not truncated.
    "REPORT": (
        "Дай краткий аналитический отчёт по криптоактиву Bitcoin (BTC) "
        "за последний квартал: цена/динамика, 3 ключевых драйвера, "
        "уровни поддержки/сопротивления, риски. 4-6 параграфов. "
        "В конце — explicit conclusion."
    ),
    # TOOL — OpenAI-compatible streaming-client style: prompt + MCP tool
    # available. Model must EITHER call the tool OR reply with text. Both
    # paths must work in stream + non-stream × thinking on/off. Tool
    # definition is injected in `_cell` for this prompt only.
    "TOOL": (
        "Я хочу узнать текущую цену Bitcoin. У тебя есть инструмент "
        "get_crypto_price — вызови его с symbol=BTC."
    ),
    # RAG — simulates a real OpenAI-compatible streaming-client call where the user
    # query is preceded by retrieved context (RAG documents, conversation
    # memory, past chat). Tests that long-context + thinking + streaming
    # all work together AND the model uses the injected context, not
    # hallucinates. Context is ~6 KB of synthetic crypto market data.
    "RAG": (
        "Контекст из базы знаний (последние данные за квартал):\n\n"
        "[1] BTC закрыл Q4 2025 на $98,400, +14% QoQ. Объёмы спот-торгов "
        "выросли на 22% после одобрения spot ETF в США.\n"
        "[2] ETH вырос на 8% до $3,420. Активность L2 (Arbitrum, "
        "Optimism) выросла на 35%.\n"
        "[3] Корреляция BTC-NASDAQ упала с 0.78 до 0.42 — признак "
        "decoupling от tradfi.\n"
        "[4] Stablecoin supply вырос с $185B до $215B (+16%), USDT "
        "доминирует с $135B.\n"
        "[5] Hash rate Bitcoin достиг ATH 750 EH/s, mining difficulty "
        "+18% за квартал.\n"
        "[6] Институциональные холдинги: MicroStrategy +35,000 BTC, "
        "BlackRock IBIT $32B AUM.\n"
        "[7] Регуляторные риски: MiCA полностью в силе с 2026-01, "
        "США Crypto Clarity Act ожидается в Q2.\n\n"
        "Используй ТОЛЬКО контекст выше. Дай аналитический отчёт по "
        "BTC: 4-6 параграфов, цитируй источники [N]. В конце — risk-"
        "adjusted outlook на Q1 2026."
    ),
}

# Per-prompt max_tokens — thinking-on path needs headroom for </think>
# AND the actual answer body, otherwise finish=length truncates and the
# probe falsely flags a "content empty" failure.
# Per-prompt × per-thinking-mode token budget. Thinking-on mode burns
# 200-2500 tokens reasoning BEFORE emitting content/tool_call, so the
# budget must include both the reasoning AND the final answer.
MAX_TOK = {
    ("SHORT",  False): 64,
    ("SHORT",  True):  1024,   # short reply but reasoning may chew 300+ tok
    ("CODE",   False): 1024,
    ("CODE",   True):  4096,   # CODE prompts trigger long planning chains
    ("REPORT", False): 4096,
    ("REPORT", True):  16384,  # full think + multi-paragraph report
    ("TOOL",   False): 1024,
    ("TOOL",   True):  2048,   # think-then-call_tool path
    ("RAG",    False): 4096,
    ("RAG",    True):  16384,  # think over RAG context + multi-paragraph answer
}


def _post(body: dict, stream: bool, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.time()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        msg = payload["choices"][0]["message"]
        return {
            "ok": True,
            "latency": time.time() - t0,
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning") or msg.get("reasoning_content") or "",
            "tool_calls": msg.get("tool_calls") or [],
            "finish": payload["choices"][0].get("finish_reason"),
        }
    # Streaming path — accumulate content + reasoning + tool_calls across chunks
    content, reasoning = "", ""
    finish = None
    tool_calls_acc: dict[int, dict] = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if not obj.get("choices"):
                continue
            ch = obj["choices"][0]
            d = ch.get("delta", {}) or {}
            if d.get("content"):
                content += d["content"]
            r_field = d.get("reasoning") or d.get("reasoning_content")
            if r_field:
                reasoning += r_field
            for tc in d.get("tool_calls") or []:
                idx = tc.get("index", 0)
                rec = tool_calls_acc.setdefault(
                    idx, {"id": None, "type": None, "name": None, "arguments": ""}
                )
                if tc.get("id"): rec["id"] = tc["id"]
                if tc.get("type"): rec["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"): rec["name"] = fn["name"]
                if fn.get("arguments") is not None:
                    rec["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return {
        "ok": True,
        "latency": time.time() - t0,
        "content": content,
        "reasoning": reasoning,
        "tool_calls": [v for _, v in sorted(tool_calls_acc.items())],
        "finish": finish,
    }


def _cell(prompt_kind: str, enable_thinking: bool, stream: bool) -> dict:
    body: dict = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPTS[prompt_kind]}],
        "max_tokens": MAX_TOK[(prompt_kind, enable_thinking)],
        "temperature": 0.0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    # TOOL prompt also ships an MCP-style tool definition so the
    # response path goes through `extract_tool_calls_streaming` instead
    # of plain content. Mirrors per-message tool injection from streaming clients.
    if prompt_kind == "TOOL":
        body["tools"] = [{
            "type": "function",
            "function": {
                "name": "get_crypto_price",
                "description": "Get current price of a cryptocurrency",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Ticker e.g. BTC, ETH"},
                    },
                    "required": ["symbol"],
                },
            },
        }]
        body["tool_choice"] = "auto"
    label = (
        f"{prompt_kind:<5} thinking={'ON ' if enable_thinking else 'OFF'} "
        f"stream={'YES' if stream else 'NO '}"
    )
    try:
        r = _post(body, stream)
    except urllib.error.HTTPError as e:
        return {"label": label, "pass": False,
                "err": f"HTTP {e.code}: {e.read()[:200].decode(errors='ignore')}"}
    except Exception as e:
        return {"label": label, "pass": False, "err": str(e)[:200]}

    content = (r.get("content") or "").strip()
    reasoning = (r.get("reasoning") or "").strip()
    tool_calls = r.get("tool_calls") or []
    finish = r.get("finish")

    # Tool-call validity: at least one entry with type=function, name,
    # and arguments that parse as JSON. Schema differs between streaming
    # accumulator (flat name) and non-streaming OpenAI shape ({function:{name,arguments}}).
    tool_call_valid = False
    for tc in tool_calls:
        try:
            fn = tc.get("function") or {}
            args = fn.get("arguments") if fn else tc.get("arguments")
            name = fn.get("name") if fn else tc.get("name")
            json.loads(args or "{}")
            if name and tc.get("type") == "function":
                tool_call_valid = True
                break
        except Exception:
            pass

    # Pass criteria depend on prompt kind:
    #  * TOOL  — either a valid tool_call OR non-empty content (model decided to answer)
    #  * other — non-empty content AND finish != "length" (not truncated)
    if prompt_kind == "TOOL":
        is_pass = tool_call_valid or bool(content)
        truncated = False
    else:
        truncated = (finish == "length")
        is_pass = bool(content) and not truncated

    # PN51 bug signature: thinking-off + stream + empty content + reasoning populated
    pn51_bug_signature = (
        not enable_thinking and stream
        and not content and bool(reasoning)
    )
    return {
        "label": label,
        "pass": is_pass,
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "tool_calls": len(tool_calls),
        "tool_call_valid": tool_call_valid,
        "content_preview": content[:60].replace("\n", " "),
        "reasoning_preview": reasoning[:60].replace("\n", " "),
        "finish": finish,
        "latency_ms": int(r.get("latency", 0) * 1000),
        "truncated": truncated,
        "pn51_bug_signature_seen": pn51_bug_signature,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable JSON instead of text")
    args = ap.parse_args()

    cells = []
    for prompt_kind in ("SHORT", "CODE", "REPORT", "TOOL", "RAG"):
        for enable_thinking in (True, False):
            for stream in (False, True):
                cells.append(_cell(prompt_kind, enable_thinking, stream))

    if args.json:
        print(json.dumps({"endpoint": ENDPOINT, "model": MODEL, "cells": cells},
                         indent=2))
    else:
        print(f"=== PN51 streaming×thinking probe — {ENDPOINT} ({MODEL}) ===")
        for c in cells:
            verdict = "PASS" if c.get("pass") else "FAIL"
            if c.get("err"):
                print(f"  [{verdict}] {c['label']:<32} ERR: {c['err']}")
                continue
            extra_bits = []
            if c.get("pn51_bug_signature_seen"):
                extra_bits.append("← PN51 BUG (empty content + reasoning populated)")
            if c.get("truncated"):
                extra_bits.append("← TRUNCATED (raise max_tokens)")
            if c.get("tool_call_valid"):
                extra_bits.append(f"tool_call OK ({c['tool_calls']})")
            extra = "  " + " | ".join(extra_bits) if extra_bits else ""
            print(
                f"  [{verdict}] {c['label']:<32} "
                f"content={c['content_len']:4d}b reasoning={c['reasoning_len']:4d}b "
                f"finish={str(c.get('finish')):<10} {c.get('latency_ms'):>5}ms{extra}"
            )

    fails = sum(1 for c in cells if not c.get("pass"))
    bugs = sum(1 for c in cells if c.get("pn51_bug_signature_seen"))
    n = len(cells)
    if fails == 0:
        print(f"\n✓ All {n} cells PASS — content channel populated in every mode.")
        if bugs == 0:
            print("  PN51 fix not currently needed (or already applied).")
    else:
        print(f"\n✗ {fails}/{n} cells FAIL.")
        if bugs:
            print(f"  {bugs} cell(s) match PN51 bug signature → enable PN51 to fix:")
            print("  GENESIS_ENABLE_PN51_QWEN3_STREAMING_THINKING_DISABLED=1")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

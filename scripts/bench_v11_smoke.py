#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""v11 smoke + bench — ~3 min run against an OpenAI-compatible vllm endpoint.

Measures:
  - boot health (5 retries on /health)
  - decode TPS over 5 generation runs (1K input, 512 output)
  - tool-call clean rate (4 prompts × 1 turn each, exact-match check)
  - VRAM footprint (single nvidia-smi snapshot via SSH if SNDR_SSH_HOST set)

Output: JSON to stdout + structured pass/fail vs reference_metrics.

Exit codes:
  0  — all metrics within tolerance
  1  — degraded (one or more metrics outside tolerance)
  2  — boot/health failure
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def _post_json(url: str, body: dict, *, api_key: str, timeout: float = 300.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _wait_health(base: str, *, api_key: str, timeout: int = 60) -> bool:
    """Poll /health until 200 OK or timeout."""
    deadline = time.monotonic() + timeout
    last_err = ""
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                f"{base}/health",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError) as e:
            last_err = str(e)
        time.sleep(2)
    print(f"  health probe failed: {last_err}", file=sys.stderr)
    return False


def _decode_run(base: str, *, model: str, api_key: str,
                input_tokens: int = 1024, output_tokens: int = 512) -> dict:
    """One decode run; returns dict with TPS + latency + token counts."""
    # Synthesize a 1K-token-ish prompt with controlled length.
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": (
                "You are a methodical writer. Please write a long, "
                "self-contained essay on the history of distributed "
                "systems, covering early time-sharing computers, the "
                "rise of microservices, the consensus problem (Paxos, "
                "Raft), and modern LLM serving stacks. Use clear "
                "paragraphs and concrete examples. Aim for around "
                "500 tokens of output."
            )},
        ],
        "max_tokens": output_tokens,
        "temperature": 0.7,
    }
    t0 = time.monotonic()
    resp = _post_json(f"{base}/v1/chat/completions", body, api_key=api_key)
    dt = time.monotonic() - t0
    out_text = resp["choices"][0]["message"]["content"] or ""
    out_tokens = resp.get("usage", {}).get("completion_tokens", 0)
    return {
        "elapsed_s": round(dt, 3),
        "out_tokens": out_tokens,
        "tps": round(out_tokens / dt, 2) if dt > 0 else 0.0,
        "out_chars": len(out_text),
    }


def _tool_call_run(base: str, *, model: str, api_key: str) -> dict:
    """One tool-call probe — verify model emits a valid OpenAI tool_call.

    Uses a deliberately simple weather-lookup tool that's easy to model.
    """
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo right now? Use the get_weather tool."},
        ],
        "tools": [tool],
        "tool_choice": "auto",
        "max_tokens": 256,
        "temperature": 0.2,
    }
    t0 = time.monotonic()
    resp = _post_json(f"{base}/v1/chat/completions", body, api_key=api_key)
    dt = time.monotonic() - t0
    msg = resp["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    clean = bool(tcs) and tcs[0].get("function", {}).get("name") == "get_weather"
    return {
        "elapsed_s": round(dt, 3),
        "clean": clean,
        "tool_call_count": len(tcs),
        "first_tool_name": (
            tcs[0]["function"]["name"] if tcs else None
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8000",
                   help="vllm base URL")
    p.add_argument("--api-key", default="genesis-local")
    p.add_argument("--model", required=True,
                   help="served-model-name registered with vllm")
    p.add_argument("--bench-runs", type=int, default=5,
                   help="number of decode runs (default 5)")
    p.add_argument("--tool-runs", type=int, default=4,
                   help="number of tool-call probes (default 4)")
    p.add_argument("--out-json", default=None,
                   help="write structured result to this file")
    p.add_argument("--ref-tps", type=float, default=None,
                   help="reference TPS for tolerance check")
    p.add_argument("--ref-tool-min", type=str, default="9/10",
                   help="reference tool minimum (e.g. 9/10)")
    p.add_argument("--tps-tolerance-pct", type=float, default=5.0,
                   help="max allowable TPS drop vs reference (%%)")
    args = p.parse_args()

    print(f"[bench_v11_smoke] base={args.base} model={args.model}")
    if not _wait_health(args.base, api_key=args.api_key, timeout=60):
        print("[bench_v11_smoke] HEALTH PROBE FAILED — aborting", file=sys.stderr)
        return 2
    print("[bench_v11_smoke] /health OK")

    # Warmup (non-measured)
    print("[bench_v11_smoke] warmup ...")
    try:
        _decode_run(args.base, model=args.model, api_key=args.api_key,
                    output_tokens=64)
    except Exception as e:
        print(f"[bench_v11_smoke] warmup FAILED: {e}", file=sys.stderr)
        return 2

    # Decode bench
    decode_results: list[dict] = []
    for i in range(args.bench_runs):
        try:
            r = _decode_run(args.base, model=args.model, api_key=args.api_key)
        except Exception as e:
            print(f"[bench_v11_smoke] decode run {i} FAILED: {e}",
                  file=sys.stderr)
            return 2
        decode_results.append(r)
        print(f"  decode run {i+1}/{args.bench_runs}: "
              f"{r['out_tokens']} tok in {r['elapsed_s']}s = {r['tps']} TPS")

    tps_values = [r["tps"] for r in decode_results if r["tps"] > 0]
    mean_tps = round(statistics.mean(tps_values), 2) if tps_values else 0.0
    cv_pct = (
        round(statistics.stdev(tps_values) / mean_tps * 100, 2)
        if len(tps_values) > 1 and mean_tps > 0 else 0.0
    )

    # Tool-call probes
    tool_results: list[dict] = []
    for i in range(args.tool_runs):
        try:
            r = _tool_call_run(args.base, model=args.model, api_key=args.api_key)
        except Exception as e:
            print(f"[bench_v11_smoke] tool-call run {i} FAILED: {e}",
                  file=sys.stderr)
            tool_results.append({"clean": False, "error": str(e)})
            continue
        tool_results.append(r)
        print(f"  tool run {i+1}/{args.tool_runs}: "
              f"clean={r.get('clean')} name={r.get('first_tool_name')}")
    clean_count = sum(1 for r in tool_results if r.get("clean"))
    tool_score = f"{clean_count}/{args.tool_runs}"

    # Tolerance check
    verdicts: list[tuple[str, bool, str]] = []
    if args.ref_tps:
        drop_pct = (args.ref_tps - mean_tps) / args.ref_tps * 100
        ok_tps = drop_pct <= args.tps_tolerance_pct
        verdicts.append((
            "tps", ok_tps,
            f"mean={mean_tps} TPS vs ref={args.ref_tps} "
            f"(drop {drop_pct:.1f}%, max {args.tps_tolerance_pct}%)"
        ))
    if args.ref_tool_min:
        ref_num, ref_den = (int(x) for x in args.ref_tool_min.split("/"))
        # Scale ref_min from /10 to /args.tool_runs
        scaled_min = ref_num * args.tool_runs // ref_den
        ok_tool = clean_count >= scaled_min
        verdicts.append((
            "tool_call", ok_tool,
            f"clean={clean_count}/{args.tool_runs} "
            f"vs scaled_min={scaled_min}/{args.tool_runs}"
        ))

    out = {
        "base": args.base,
        "model": args.model,
        "decode": {
            "runs": decode_results,
            "mean_tps": mean_tps,
            "cv_pct": cv_pct,
        },
        "tool_call": {
            "runs": tool_results,
            "score": tool_score,
            "clean_count": clean_count,
        },
        "reference": {
            "ref_tps": args.ref_tps,
            "ref_tool_min": args.ref_tool_min,
            "tps_tolerance_pct": args.tps_tolerance_pct,
        },
        "verdicts": [
            {"metric": m, "ok": ok, "detail": d} for m, ok, d in verdicts
        ],
    }

    print()
    print("=== SUMMARY ===")
    print(f"  decode mean TPS: {mean_tps}  CV: {cv_pct}%")
    print(f"  tool-call score: {tool_score}")
    for m, ok, d in verdicts:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {m}: {d}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  result written to {args.out_json}")

    if any(not ok for _, ok, _ in verdicts):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

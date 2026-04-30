#!/usr/bin/env python3
"""Progressive context-window probe.

Sends increasingly long prompts to find the model's max stable context.
Stops at first failure (HTTP 5xx, OOM, timeout, connection refused).
Records elapsed time + tokens-per-second for each successful step.

Usage:
    python3 tools/progressive_context_probe.py --host localhost --model qwen3.6-27b
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.error, urllib.request


def make_payload(model: str, target_tokens: int, max_completion: int = 200) -> bytes:
    # Approximate tokens with words; vllm tokenizer averages ~2.0 tokens per word for "hello"
    word_count = max(1, (target_tokens - 100) // 2)
    p = "hello " * word_count
    payload = {"model": model, "messages": [{"role": "user", "content": p}],
               "max_tokens": max_completion, "stream": False}
    return json.dumps(payload).encode()


def probe(host: str, port: int, api_key: str, model: str, target: int, timeout: int):
    url = f"http://{host}:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    payload = make_payload(model, target)
    req = urllib.request.Request(url, data=payload, headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        elapsed = time.perf_counter() - t0
        usage = data.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        # decode TPS = ct / (elapsed - prompt_processing_time). prompt_processing_time
        # we don't know exactly, but we can estimate: bench shows ~5ms/decode-token
        # so decode_part ≈ ct * 5e-3 → prefill_part ≈ elapsed - decode_part
        decode_est = ct * 0.005
        prefill_est = max(0.001, elapsed - decode_est)
        prefill_tps = pt / prefill_est if prefill_est > 0 else 0
        decode_tps = ct / decode_est if decode_est > 0 else 0
        return {"target": target, "verdict": "PASS", "prompt_tokens": pt,
                "completion_tokens": ct, "elapsed_s": round(elapsed, 2),
                "prefill_est_s": round(prefill_est, 2),
                "prefill_tps_est": round(prefill_tps, 1),
                "decode_tps_est": round(decode_tps, 1)}
    except urllib.error.HTTPError as e:
        return {"target": target, "verdict": f"FAIL_HTTP_{e.code}",
                "error": str(e)[:200]}
    except (urllib.error.URLError, TimeoutError, ConnectionRefusedError, OSError) as e:
        return {"target": target, "verdict": "FAIL_CONNECTION",
                "error": str(e)[:200]}
    except Exception as e:
        return {"target": target, "verdict": "FAIL_OTHER",
                "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default="genesis-local")
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--steps", default="16384,32768,65536,98304,131072,163840,196608,262144")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--out", default="docs/_internal/runs/progressive_ctx_probe.json")
    ap.add_argument("--stop-on-fail", action="store_true", default=True)
    args = ap.parse_args()

    targets = [int(s) for s in args.steps.split(",")]
    results = []
    print(f"Progressive context probe — model={args.model} host={args.host} timeout={args.timeout}s")
    print(f"Steps: {targets}")
    print()

    for t in targets:
        print(f"  → probing ctx ~{t} ({t/1024:.0f}K)... ", end="", flush=True)
        r = probe(args.host, args.port, args.api_key, args.model, t, args.timeout)
        results.append(r)
        if r["verdict"] == "PASS":
            print(f"PASS  prompt={r['prompt_tokens']} elapsed={r['elapsed_s']}s "
                  f"prefill≈{r['prefill_tps_est']} t/s")
        else:
            print(f"{r['verdict']}: {r.get('error', '')[:80]}")
            if args.stop_on_fail:
                print(f"\nStopping on first failure at {t/1024:.0f}K.")
                break

    print()
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"Results saved: {args.out}")
    # Summary
    passed = [r for r in results if r["verdict"] == "PASS"]
    if passed:
        max_pass = max(r["target"] for r in passed)
        print(f"Max stable context: {max_pass} tokens ({max_pass/1024:.0f}K)")


if __name__ == "__main__":
    main()

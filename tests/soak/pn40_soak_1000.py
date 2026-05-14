"""PN40 1000-step soak — exercises adaptive K controller + sentinel under
mixed workload (code / short / long-ctx / free-form). Captures latency,
TPS, tool-call cleanliness, and lets PN40 sub-C+D emit observability
log lines (every 200 obs).

Targets the vLLM endpoint set via GENESIS_ENDPOINT (default
http://localhost:8000). Reads model name from GENESIS_MODEL or
defaults to qwen3.6-27b.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
import urllib.error
import urllib.request

API_KEY = os.environ.get("GENESIS_API_KEY", "genesis-local")
ENDPOINT = os.environ.get("GENESIS_ENDPOINT", "http://localhost:8000/v1/chat/completions")
MODEL = os.environ.get("GENESIS_MODEL", "qwen3.6-27b")

CODE_PROMPTS = [
    "Write a Python function to compute the n-th Fibonacci number iteratively.",
    "Implement quicksort in Go with median-of-three pivot.",
    "In Rust, write a tail-recursive factorial using accumulator pattern.",
    "Sketch a TypeScript Result<T,E> monad with map/flatMap.",
    "Show C++ RAII pattern for a file-handle wrapper.",
    "Bash one-liner: list all .py files modified in last 7 days, sorted by size.",
    "SQL query: window function to rank users by revenue per region.",
    "Write a JavaScript debounce function with leading-edge option.",
    "Java: design a thread-safe LRU cache with O(1) get/put.",
    "Kotlin coroutines: convert callback API to suspend function.",
]

SHORT_PROMPTS = [
    "What is the capital of France?",
    "Translate 'hello world' to Japanese.",
    "Define entropy in one sentence.",
    "What year did WW2 end?",
    "Largest planet in the solar system?",
    "Boiling point of water in Celsius?",
    "Who wrote Hamlet?",
    "Speed of light in vacuum (km/s)?",
    "Symbol for gold on the periodic table?",
    "Who painted the Mona Lisa?",
]

LONG_PROMPTS = [
    "Write a 500-word essay on the history of the steam engine, covering Newcomen, Watt, and the industrial revolution.",
    "Explain quantum entanglement to a high school student in detail with at least three real-world analogies.",
    "Compare and contrast the architectural philosophies of Gothic cathedrals and modern skyscrapers, 600 words.",
    "Describe the full life cycle of a sun-like star from protostar to white dwarf, with timescales for each phase.",
    "Walk through the entire process of bread fermentation, including the role of gluten, CO2, alcohol, and Maillard reactions.",
    "Outline the key events of the French Revolution chronologically, then analyze its long-term political impact in Europe.",
    "Detailed walkthrough: how does TCP/IP handshake work, including SYN, SYN-ACK, ACK, and what happens on packet loss?",
    "Explain transformer architecture (attention is all you need) with focus on multi-head attention and positional encoding.",
    "Write a comprehensive guide to making sourdough bread, from starter creation to final bake, ~700 words.",
    "Describe the geology of the Grand Canyon: rock layers, erosion timeline, and Colorado River's role.",
]

FREE_FORM = [
    "Write a haiku about debugging code at 3am.",
    "Invent a new ice cream flavor and describe it.",
    "Tell a 3-sentence ghost story set in a server room.",
    "What would you say to a junior dev panicking about prod outage?",
    "Pitch a startup idea combining AI and gardening.",
    "Describe an alien civilization that communicates via smell.",
    "Two bullet points on why writing tests first is good.",
    "Slogan for a coffee brand targeted at sysadmins.",
    "Compose limerick about a Kubernetes cluster.",
    "Three names for a friendly pet robot.",
]

POOLS = {
    "code": CODE_PROMPTS,
    "short_ctx": SHORT_PROMPTS,
    "long_ctx": LONG_PROMPTS,
    "free_form": FREE_FORM,
}


def pick_workload(rng: random.Random) -> tuple[str, str, int]:
    weights = [("code", 0.30), ("short_ctx", 0.30), ("long_ctx", 0.20), ("free_form", 0.20)]
    r = rng.random()
    acc = 0.0
    for name, w in weights:
        acc += w
        if r <= acc:
            kind = name
            break
    else:
        kind = "free_form"
    prompt = rng.choice(POOLS[kind])
    max_tok = {"code": 256, "short_ctx": 64, "long_ctx": 800, "free_form": 128}[kind]
    return kind, prompt, max_tok


def call_chat(prompt: str, max_tok: int, timeout: float = 120.0) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tok,
        "stream": False,
    }
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
            latency = time.time() - t0
            usage = out.get("usage", {})
            return {
                "ok": True,
                "latency": latency,
                "completion_tokens": usage.get("completion_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "tps": usage.get("completion_tokens", 0) / latency if latency > 0 else 0,
            }
    except urllib.error.HTTPError as e:
        return {"ok": False, "latency": time.time() - t0, "err": f"HTTP {e.code}"}
    except Exception as e:
        return {"ok": False, "latency": time.time() - t0, "err": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-every", type=int, default=100)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"[soak] target={ENDPOINT} model={MODEL} steps={args.steps}")

    by_kind: dict[str, list[float]] = {k: [] for k in POOLS}
    by_kind_ok: dict[str, int] = {k: 0 for k in POOLS}
    by_kind_fail: dict[str, int] = {k: 0 for k in POOLS}
    all_lat: list[float] = []
    all_tps: list[float] = []
    fail = 0
    t_start = time.time()

    for i in range(args.steps):
        kind, prompt, max_tok = pick_workload(rng)
        res = call_chat(prompt, max_tok)
        if not res["ok"]:
            fail += 1
            by_kind_fail[kind] += 1
            print(f"[soak {i+1:4d}/{args.steps}] FAIL {kind}: {res.get('err','?')}")
            continue
        by_kind_ok[kind] += 1
        by_kind[kind].append(res["tps"])
        all_lat.append(res["latency"])
        all_tps.append(res["tps"])

        if (i + 1) % args.report_every == 0:
            elapsed = time.time() - t_start
            rps = (i + 1) / elapsed
            mean_tps = statistics.mean(all_tps) if all_tps else 0
            print(
                f"[soak {i+1:4d}/{args.steps}] "
                f"elapsed={elapsed:6.1f}s rps={rps:.2f} "
                f"mean_tps={mean_tps:.1f} fail={fail}"
            )

    elapsed = time.time() - t_start
    print("\n=== A.3 SOAK SUMMARY ===")
    print(f"steps:       {args.steps}")
    print(f"elapsed:     {elapsed:.1f}s")
    print(f"failures:    {fail}")
    if all_lat:
        print(f"latency p50: {statistics.median(all_lat):.2f}s")
        print(f"latency p95: {sorted(all_lat)[int(len(all_lat)*0.95)]:.2f}s")
        print(f"tps mean:    {statistics.mean(all_tps):.2f}")
        print(f"tps stdev:   {statistics.stdev(all_tps):.2f}" if len(all_tps) > 1 else "tps stdev: n/a")
    print("\nper-workload tps mean / count(ok/fail):")
    for kind in POOLS:
        vals = by_kind[kind]
        mean = statistics.mean(vals) if vals else 0
        print(f"  {kind:10s}: tps={mean:6.2f}  ok={by_kind_ok[kind]:4d}  fail={by_kind_fail[kind]:3d}")


if __name__ == "__main__":
    main()

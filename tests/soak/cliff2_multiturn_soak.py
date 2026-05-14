"""Cliff 2b multi-turn soak harness — Variant D Phase 3 GPU integration probe.

Reproduces noonghunna's Cliff 2 OOM scenario in a Genesis-internal
harness. Used to validate PN59 streaming-GDN solves the OOM:

  Baseline (PN59 OFF): 6/6 single-card configs FAIL after 4-5 turns
  Target (PN59 ON):    survive 20+ turns continuous

Workflow
--------
1. Send hermes-style chat (system + user) → response
2. Append assistant + new user → next turn
3. Repeat N turns, monitoring per-turn allocator delta
4. Report: max-turns-survived, peak VRAM, per-turn delta trajectory

Eligibility
-----------
- Server URL via GENESIS_ENDPOINT (default http://localhost:8000)
- Model: 27B Lorbus hybrid (Cliff 2b reproducer model)
- WSL2 / single-3090 / 24GB headroom = traditional Cliff 2 trigger config

Comparison harness with noonghunna's `SOAK_MODE=continuous` style.
Coordinates with Genesis Issue #19 thread for cross-rig validation.

Usage
-----
  # Baseline (vanilla, no PN59)
  GENESIS_ENABLE_PN59_STREAMING_GDN=0 python3 tests/soak/cliff2_multiturn_soak.py \\
      --model qwen3.6-27b --turns 20 --report-every 1

  # With PN59
  GENESIS_ENABLE_PN59_STREAMING_GDN=1 \\
      GENESIS_VARIANT_D_WINDOW_NT=4 \\
      python3 tests/soak/cliff2_multiturn_soak.py \\
      --model qwen3.6-27b --turns 20 --report-every 1

Author: Sandermage 2026-05-05, Variant D Phase 3.
Coordinated with noonghunna issue #20 thread.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


API_KEY = os.environ.get("GENESIS_API_KEY", "genesis-local")
ENDPOINT = os.environ.get(
    "GENESIS_ENDPOINT", "http://localhost:8000/v1/chat/completions"
)
MODEL = os.environ.get("GENESIS_MODEL", "qwen3.6-27b")
# Empty default = local-only run (no SSH telemetry). Operator wires
# GENESIS_SSH_HOST=<user>@<host> when running this probe against a
# remote rig and wanting nvidia-smi memory snapshots in the report.
SSH_HOST = os.environ.get("GENESIS_SSH_HOST", "")


SYSTEM_PROMPT = (
    "You are a helpful assistant focused on detailed technical analysis. "
    "Respond thoroughly with explanations."
)


# Hermes-style turn prompts: gradually accumulating context
TURN_PROMPTS = [
    "Explain the differences between Mamba2 and Gated Delta Rule attention.",
    "Continue with: how does the Triton chunk kernel maintain recurrent state?",
    "Now explain the trade-offs of materializing full hidden state vs streaming.",
    "Compare with Flash Attention v3 — what's structurally different?",
    "What about FA4 on Blackwell? Will it support GLA family?",
    "Sketch a per-token latency budget for a 27B model on 2× 24GB.",
    "How would you parallelize this across 4 GPUs?",
    "What about TP=8 with NVLink? Worth it?",
    "Discuss spec-decode interactions — MTP K=3 vs ngram fallback.",
    "Now tie it all together: what's the highest-impact optimization left?",
    "Cite 3 academic papers that influenced these designs.",
    "And 3 production system papers (e.g. Anthropic, Google, Meta).",
    "What's the 12-month outlook for this stack?",
    "What changes would Blackwell B200 enable?",
    "Imagine cost-per-token at scale. Where's the bottleneck?",
    "Final thoughts: is there a fundamentally better approach?",
    "Could quantum hardware help any of this in the next decade?",
    "Energy-efficiency angle: where do we waste the most joules?",
    "Last technical deep-dive: explain online quant for activations.",
    "Wrap-up: top 5 lessons from this conversation.",
]


def _vram_snapshot() -> dict | None:
    """Capture per-GPU VRAM via ssh + nvidia-smi."""
    try:
        result = subprocess.run(
            ["ssh", SSH_HOST, "nvidia-smi", "--query-gpu=index,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "index": int(parts[0]),
                    "used_mib": int(parts[1]),
                    "free_mib": int(parts[2]),
                })
        return {"gpus": gpus}
    except Exception:
        return None


def _send_turn(messages: list[dict], max_tokens: int = 256, timeout: float = 90.0) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
        msg = payload["choices"][0]["message"]
        return {
            "ok": True,
            "latency_s": time.time() - t0,
            "content": msg.get("content") or "",
            "completion_tokens": payload.get("usage", {}).get("completion_tokens", 0),
            "prompt_tokens": payload.get("usage", {}).get("prompt_tokens", 0),
        }
    except urllib.error.HTTPError as e:
        return {"ok": False, "err": f"HTTP {e.code}: {e.read()[:300].decode(errors='ignore')}"}
    except Exception as e:
        return {"ok": False, "err": f"{type(e).__name__}: {str(e)[:200]}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=20,
                    help="Max turns to attempt before giving up")
    ap.add_argument("--report-every", type=int, default=1,
                    help="Print VRAM snapshot every N turns")
    ap.add_argument("--max-tokens-per-turn", type=int, default=256)
    ap.add_argument("--prompt-set", choices=["default", "code-heavy"], default="default")
    args = ap.parse_args()

    print(f"=== Cliff 2b multi-turn soak — Variant D Phase 3 GPU probe ===")
    print(f"Endpoint:     {ENDPOINT}")
    print(f"Model:        {MODEL}")
    print(f"Max turns:    {args.turns}")
    print(f"Max tok/turn: {args.max_tokens_per_turn}")
    print()

    vram_pre = _vram_snapshot()
    if vram_pre:
        for g in vram_pre["gpus"]:
            print(f"  Pre-soak GPU{g['index']}: {g['used_mib']} MiB used, "
                  f"{g['free_mib']} MiB free")
    print()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    per_turn_metrics = []

    for turn in range(1, args.turns + 1):
        if turn - 1 >= len(TURN_PROMPTS):
            # Cycle if we run out of curated prompts
            user_prompt = TURN_PROMPTS[(turn - 1) % len(TURN_PROMPTS)]
        else:
            user_prompt = TURN_PROMPTS[turn - 1]

        messages.append({"role": "user", "content": user_prompt})

        result = _send_turn(messages, max_tokens=args.max_tokens_per_turn)

        if not result["ok"]:
            print(f"[turn {turn:3d}] ✗ FAILURE: {result['err'][:120]}")
            print(f"\n=== SOAK FAILED AT TURN {turn} ===")
            print(f"Survived: {turn - 1} turns")
            if "OutOfMemoryError" in result["err"] or "out of memory" in result["err"].lower():
                print("Class:    OOM (Cliff 2b signature)")
            return 1

        messages.append({"role": "assistant", "content": result["content"]})

        per_turn_metrics.append({
            "turn": turn,
            "latency_s": result["latency_s"],
            "completion": result["completion_tokens"],
            "prompt": result["prompt_tokens"],
        })

        if turn % args.report_every == 0:
            vram = _vram_snapshot()
            vram_str = ""
            if vram:
                used = sum(g["used_mib"] for g in vram["gpus"])
                free_min = min(g["free_mib"] for g in vram["gpus"])
                vram_str = f"VRAM total_used={used}MiB min_free={free_min}MiB"
            print(f"[turn {turn:3d}] ✓ "
                  f"prompt={result['prompt_tokens']:5d}t "
                  f"completion={result['completion_tokens']:4d}t "
                  f"latency={result['latency_s']:5.1f}s "
                  f"{vram_str}")

    print(f"\n=== SOAK SURVIVED {args.turns} TURNS ===")
    if per_turn_metrics:
        max_prompt = max(m["prompt"] for m in per_turn_metrics)
        avg_latency = sum(m["latency_s"] for m in per_turn_metrics) / len(per_turn_metrics)
        print(f"Final prompt size: {max_prompt} tokens")
        print(f"Avg latency:       {avg_latency:.1f}s")

    vram_post = _vram_snapshot()
    if vram_post and vram_pre:
        for i, (pre, post) in enumerate(zip(vram_pre["gpus"], vram_post["gpus"])):
            delta = post["used_mib"] - pre["used_mib"]
            print(f"  GPU{i} VRAM delta over soak: {delta:+d} MiB "
                  f"(pre={pre['used_mib']} → post={post['used_mib']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

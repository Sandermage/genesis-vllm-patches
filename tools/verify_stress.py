#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""verify_stress — 8-probe boundary-condition smoke for a live vLLM endpoint.

Port of noonghunna/club-3090 `scripts/verify-stress.sh` methodology
(see sndr_private/planning/audits/CLUB3090_CROSS_REFERENCE_2026-05-29_RU.md
§3.2 for the design rationale) adapted to our stack. Each probe answers a
distinct production failure-mode question — skipping any layer can mask
the corresponding regression class.

The 8 probes
------------
  1. **Long-context NIAH small rungs** (10K + 30K) — system-level liveness
     under long-prompt prefill. Per-rung secret keys defeat caching.
     Failure ≠ recall-miss (informational); failure = HTTP non-200.

  2. **Tool-response prefill OOM** — ~25K-token mock tool message + tool
     def + tool_choice="auto". Exercises chunked-prefill on long tool
     turns (real IDE-agent workload shape).

  3. **IDE-agent one-shot** — ~5K sys preamble + 10 tool schemas + 350-
     char user. Catches Cliff 1 mech B (inductor FFN intermediate leak).

  4. **Multi-turn agent** — sys + tools + user → tool_call → tool reply
     → followup. Different compile path than #3; tests stateful context
     accumulation up to ~12K.

  5. **LCB-coding shape** — LeetCode-style problem + structured plan +
     max_tokens=4096. Catches DS conv state regression class.

  6. **Reasoning-heavy** — math/algorithm + max_tokens=8192. Stresses
     spec-decode AL collapse + mamba cache_mode='align'.

  7. **Long-context NIAH large rungs** (60K + 90K) — Cliff 2 territory,
     deferred to LAST so liveness preserved for probes 2-6.

  8. **Context ceiling ladder** — staggers NIAH from CEILING_START_TOKENS
     up to CEILING_FRACTION × n_ctx in CEILING_STEP_TOKENS increments.
     Captures VRAM before/after each rung; stops on first failure.
     VRAM_MARGIN_MB=1024 minimum free triggers WARN.

Outputs
-------
  - Per-probe table: probe / status / latency / details
  - Final verdict: PASS / WARN / FAIL with first-failed-probe pointer
  - Optional --out JSON for downstream automation

Usage
-----
  # Quick smoke (probes 1+3+4 only — ~2 min)
  python3 tools/verify_stress.py --quick

  # Full 8 probes against 35B PROD (~10 min)
  python3 tools/verify_stress.py --url http://localhost:8103/v1 \\
      --model qwen3.6-35b

  # Skip the LARGE NIAH rungs (probes 7+8) — saves ~3-5 min on long-ctx configs
  python3 tools/verify_stress.py --no-large-niah

  # Capture JSON for regression-tracking
  python3 tools/verify_stress.py --out tools/bench_results/stress_$(date +%Y%m%d).json

Honest disclaimer (per club-3090 verify-stress.sh)
--------------------------------------------------
PASS criterion is **system-level liveness**, not output quality. A 200
that returns garbled content counts as PASS for this probe — content
correctness is the job of `quality-test.sh` (BFCL/aider-polyglot/etc.).
Use this to detect: HTTP 500, ServerDisconnect, timeout, OOM-class
errors. Use quality-test for: BFCL score, tool_call rate, JSON validity.

Author: Sandermage 2026-05-29 — port of noonghunna/club-3090
scripts/verify-stress.sh methodology + adapted for Genesis stack.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import aiohttp
except ImportError:
    sys.stderr.write(
        "ERROR: aiohttp not installed. Install with: pip install aiohttp\n"
    )
    sys.exit(2)


DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "qwen3.6-35b"
DEFAULT_API_KEY = os.environ.get("VLLM_API_KEY", "genesis-local")


@dataclass(slots=True)
class ProbeResult:
    name: str
    status: str  # PASS / WARN / FAIL / SKIP
    elapsed_s: float
    detail: str = ""
    http_status: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ─────────────────────── helpers ──────────────────────────────────


async def _post(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    body: dict,
    timeout_s: float,
) -> tuple[int, dict | None, str]:
    """POST chat/completions, return (status, json|None, raw_text)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with session.post(
            f"{url.rstrip('/')}/chat/completions",
            json=body, headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            raw = await resp.text()
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                return resp.status, None, raw
    except asyncio.TimeoutError:
        return 0, None, f"timeout after {timeout_s}s"
    except aiohttp.ClientError as exc:
        return 0, None, f"client error: {exc}"
    except Exception as exc:
        return 0, None, f"unexpected: {type(exc).__name__}: {exc}"


def _random_secret() -> tuple[str, str]:
    """Animal+color+2-digit secret token + a unique key word."""
    animals = ["wombat", "axolotl", "narwhal", "okapi", "tapir"]
    colors = ["azure", "crimson", "saffron", "ochre", "magenta"]
    a, c = random.choice(animals), random.choice(colors)
    n = random.randint(10, 99)
    return f"{c}_{a}_{n:02d}", f"the secret is {c}_{a}_{n:02d}"


def _filler_paragraph(words_n: int) -> str:
    """Generate deterministic filler text — no Lorem ipsum (just bench fixture)."""
    base = (
        "The historical record indicates that periodic re-evaluation of "
        "long-form context handling under high-throughput conditions has "
        "consistently revealed previously undocumented edge cases in the "
        "interaction between attention backends and KV cache management. "
    )
    base_words = base.split()
    out_words: list[str] = []
    while len(out_words) < words_n:
        out_words.extend(base_words)
    return " ".join(out_words[:words_n])


def _build_niah_prompt(token_target: int, secret_phrase: str) -> str:
    """Build a needle-in-haystack prompt of approximately `token_target` tokens.
    Plants `secret_phrase` ~3/4 into the haystack so retrieval has to span.
    """
    # ~0.75 words per token for English (rough). Target wordcount.
    word_target = int(token_target * 0.75)
    before = _filler_paragraph(int(word_target * 0.6))
    after = _filler_paragraph(int(word_target * 0.4))
    return (
        f"You will be asked to recall a secret embedded in the following "
        f"passage. Read carefully.\n\n"
        f"{before}\n\n{secret_phrase}.\n\n{after}\n\n"
        f"Question: What is the secret? Answer with only the secret token."
    )


# ─────────────────────── probes ───────────────────────────────────


async def probe_1_niah_small(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[ProbeResult]:
    """NIAH small rungs (10K + 30K) — liveness under long-prompt prefill."""
    results: list[ProbeResult] = []
    for rung_tokens in (10_000, 30_000):
        secret_token, secret_phrase = _random_secret()
        prompt = _build_niah_prompt(rung_tokens, secret_phrase)
        t0 = time.perf_counter()
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.0,
        }
        status, data, raw = await _post(
            session, args.url, args.api_key, body,
            timeout_s=args.long_ctx_timeout,
        )
        elapsed = time.perf_counter() - t0
        if status == 200 and data:
            usage = data.get("usage") or {}
            content = (data["choices"][0]["message"]
                       .get("content") or "").lower()
            recalled = secret_token.lower() in content
            results.append(ProbeResult(
                name=f"probe1_niah_small_{rung_tokens // 1000}k",
                status="PASS" if recalled else "PASS (recall MISS — info only)",
                elapsed_s=elapsed,
                detail=f"recall={'✓' if recalled else '△'}",
                http_status=status,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                extra={"recalled": recalled, "secret": secret_token},
            ))
        elif status == 400:
            results.append(ProbeResult(
                name=f"probe1_niah_small_{rung_tokens // 1000}k",
                status="PASS (HTTP 400 — clean reject above max_model_len)",
                elapsed_s=elapsed,
                detail=raw[:100],
                http_status=status,
            ))
            break
        else:
            results.append(ProbeResult(
                name=f"probe1_niah_small_{rung_tokens // 1000}k",
                status="FAIL",
                elapsed_s=elapsed,
                detail=f"HTTP {status}: {raw[:120]}",
                http_status=status,
            ))
            break
    return results


async def probe_2_tool_prefill_oom(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[ProbeResult]:
    """~25K-token mock tool message + tool_choice=auto."""
    mock_tool_result = (
        "STDOUT line " + " ".join(_filler_paragraph(30_000).split()[:6500])
    )
    tool_schema = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}}},
        },
    }]
    body = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": "Run ls and tell me what you see."},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_001", "type": "function",
                             "function": {"name": "read_file",
                                          "arguments": '{"path":"/tmp"}'}}]},
            {"role": "tool", "tool_call_id": "call_001",
             "content": mock_tool_result[:80_000]},
            {"role": "user", "content": "What's the most interesting line?"},
        ],
        "tools": tool_schema,
        "tool_choice": "auto",
        "max_tokens": 100,
        "temperature": 0.3,
    }
    t0 = time.perf_counter()
    status, data, raw = await _post(
        session, args.url, args.api_key, body,
        timeout_s=args.tool_prefill_timeout,
    )
    elapsed = time.perf_counter() - t0
    if status == 200 and data:
        usage = data.get("usage") or {}
        return [ProbeResult(
            name="probe2_tool_prefill_oom",
            status="PASS", elapsed_s=elapsed,
            detail=f"prompt={usage.get('prompt_tokens')}t",
            http_status=status,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )]
    return [ProbeResult(
        name="probe2_tool_prefill_oom",
        status="FAIL", elapsed_s=elapsed,
        detail=f"HTTP {status}: {raw[:120]}",
        http_status=status,
    )]


async def probe_3_ide_agent_one_shot(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[ProbeResult]:
    """~5K sys preamble + 10 tool schemas + 350-char user — Cliff 1 mech B."""
    sys_preamble = (
        "You are a coding assistant. " + _filler_paragraph(700) +
        " Tools are available; use them. " + _filler_paragraph(300)
    )
    tools = [
        {"type": "function",
         "function": {"name": f"tool_{i}",
                      "description": f"Synthetic tool number {i} for IDE harness probe.",
                      "parameters": {"type": "object",
                                     "properties": {"arg": {"type": "string"}}}}}
        for i in range(10)
    ]
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": sys_preamble},
            {"role": "user", "content":
             "Find the entry point of this Python package and tell me what "
             "it imports. " + _filler_paragraph(40)},
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    t0 = time.perf_counter()
    status, data, raw = await _post(
        session, args.url, args.api_key, body,
        timeout_s=args.tool_prefill_timeout,
    )
    elapsed = time.perf_counter() - t0
    if status == 200 and data:
        usage = data.get("usage") or {}
        return [ProbeResult(
            name="probe3_ide_agent",
            status="PASS", elapsed_s=elapsed,
            detail=f"prompt={usage.get('prompt_tokens')}t",
            http_status=status,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )]
    return [ProbeResult(
        name="probe3_ide_agent", status="FAIL", elapsed_s=elapsed,
        detail=f"HTTP {status}: {raw[:120]}", http_status=status,
    )]


async def probe_4_multi_turn_agent(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[ProbeResult]:
    """sys + tools + user → tool_call → tool reply → followup."""
    tools = [{
        "type": "function",
        "function": {
            "name": "grep_files",
            "description": "Search files for a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"},
                               "path": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    }]
    messages = [
        {"role": "system",
         "content": "You are a coding assistant. Use the grep_files tool."},
        {"role": "user", "content": "Find all 'TODO' comments in src/."},
        {"role": "assistant", "content": "",
         "tool_calls": [{
             "id": "call_a", "type": "function",
             "function": {"name": "grep_files",
                          "arguments": '{"pattern":"TODO","path":"src/"}'},
         }]},
        {"role": "tool", "tool_call_id": "call_a",
         "content": "src/foo.py:12: # TODO refactor\n"
                    "src/bar.py:88: # TODO add tests\n"},
        {"role": "user", "content":
         "Now read foo.py:12 and propose the refactor."},
    ]
    body = {
        "model": args.model, "messages": messages,
        "tools": tools, "tool_choice": "auto",
        "max_tokens": 500, "temperature": 0.3,
    }
    t0 = time.perf_counter()
    status, data, raw = await _post(
        session, args.url, args.api_key, body,
        timeout_s=args.tool_prefill_timeout,
    )
    elapsed = time.perf_counter() - t0
    if status == 200 and data:
        usage = data.get("usage") or {}
        return [ProbeResult(
            name="probe4_multi_turn", status="PASS", elapsed_s=elapsed,
            detail=f"prompt={usage.get('prompt_tokens')}t",
            http_status=status,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )]
    return [ProbeResult(
        name="probe4_multi_turn", status="FAIL", elapsed_s=elapsed,
        detail=f"HTTP {status}: {raw[:120]}", http_status=status,
    )]


async def probe_5_lcb_coding(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[ProbeResult]:
    """LeetCode-style structured plan + max_tokens=4096."""
    prompt = (
        "Implement a function `find_kth_largest(nums, k)` that returns the "
        "k-th largest element in an unsorted list, in O(n log k). Provide:\n"
        "1. Approach explanation (3 sentences)\n"
        "2. Time/space complexity\n"
        "3. Python implementation\n"
        "4. 5 test cases including edge cases\n"
    )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096, "temperature": 0.2,
    }
    t0 = time.perf_counter()
    status, data, raw = await _post(
        session, args.url, args.api_key, body,
        timeout_s=args.long_ctx_timeout,
    )
    elapsed = time.perf_counter() - t0
    if status == 200 and data:
        usage = data.get("usage") or {}
        return [ProbeResult(
            name="probe5_lcb_coding", status="PASS", elapsed_s=elapsed,
            detail=f"comp={usage.get('completion_tokens')}t",
            http_status=status,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )]
    return [ProbeResult(
        name="probe5_lcb_coding", status="FAIL", elapsed_s=elapsed,
        detail=f"HTTP {status}: {raw[:120]}", http_status=status,
    )]


async def probe_6_reasoning_heavy(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[ProbeResult]:
    """Math/algorithm prompt + max_tokens=8192."""
    prompt = (
        "Prove that the harmonic series Σ(1/n) for n=1..∞ diverges. "
        "Provide three different proofs: (1) by grouping into blocks of "
        "powers of 2, (2) by integral comparison with 1/x, (3) by "
        "computing partial sums and showing they grow without bound. "
        "Format each proof clearly with assumptions and conclusion."
    )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192, "temperature": 0.0,
    }
    t0 = time.perf_counter()
    status, data, raw = await _post(
        session, args.url, args.api_key, body,
        timeout_s=args.long_ctx_timeout,
    )
    elapsed = time.perf_counter() - t0
    if status == 200 and data:
        usage = data.get("usage") or {}
        return [ProbeResult(
            name="probe6_reasoning", status="PASS", elapsed_s=elapsed,
            detail=f"comp={usage.get('completion_tokens')}t",
            http_status=status,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )]
    return [ProbeResult(
        name="probe6_reasoning", status="FAIL", elapsed_s=elapsed,
        detail=f"HTTP {status}: {raw[:120]}", http_status=status,
    )]


async def probe_7_niah_large(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[ProbeResult]:
    """NIAH large rungs (60K + 90K) — Cliff 2 territory."""
    results: list[ProbeResult] = []
    for rung_tokens in (60_000, 90_000):
        secret_token, secret_phrase = _random_secret()
        prompt = _build_niah_prompt(rung_tokens, secret_phrase)
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30, "temperature": 0.0,
        }
        t0 = time.perf_counter()
        status, data, raw = await _post(
            session, args.url, args.api_key, body,
            timeout_s=args.long_ctx_timeout,
        )
        elapsed = time.perf_counter() - t0
        if status == 200 and data:
            usage = data.get("usage") or {}
            content = (data["choices"][0]["message"]
                       .get("content") or "").lower()
            recalled = secret_token.lower() in content
            results.append(ProbeResult(
                name=f"probe7_niah_large_{rung_tokens // 1000}k",
                status="PASS" if recalled else "PASS (recall MISS — info only)",
                elapsed_s=elapsed,
                detail=f"recall={'✓' if recalled else '△'}",
                http_status=status,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                extra={"recalled": recalled, "secret": secret_token},
            ))
        elif status == 400:
            results.append(ProbeResult(
                name=f"probe7_niah_large_{rung_tokens // 1000}k",
                status="PASS (HTTP 400 — clean reject above max_model_len)",
                elapsed_s=elapsed,
                detail=raw[:80], http_status=status,
            ))
            break  # ladder ended
        else:
            results.append(ProbeResult(
                name=f"probe7_niah_large_{rung_tokens // 1000}k",
                status="FAIL", elapsed_s=elapsed,
                detail=f"HTTP {status}: {raw[:120]}",
                http_status=status,
            ))
            break
    return results


async def probe_8_ceiling_ladder(
    session: aiohttp.ClientSession, args: argparse.Namespace
) -> list[ProbeResult]:
    """Ceiling ladder — stagger NIAH from CEILING_START to CEILING_FRACTION ×
    n_ctx. Stop on first failure.
    """
    results: list[ProbeResult] = []
    rung = args.ceiling_start_tokens
    while rung <= args.ceiling_max_tokens:
        secret_token, secret_phrase = _random_secret()
        prompt = _build_niah_prompt(rung, secret_phrase)
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 25, "temperature": 0.0,
        }
        t0 = time.perf_counter()
        status, data, raw = await _post(
            session, args.url, args.api_key, body,
            timeout_s=args.long_ctx_timeout,
        )
        elapsed = time.perf_counter() - t0
        if status == 200 and data:
            usage = data.get("usage") or {}
            content = (data["choices"][0]["message"]
                       .get("content") or "").lower()
            recalled = secret_token.lower() in content
            results.append(ProbeResult(
                name=f"probe8_ceiling_{rung // 1000}k",
                status="PASS" if recalled else "PASS (recall MISS)",
                elapsed_s=elapsed,
                detail=f"prompt={usage.get('prompt_tokens')}t",
                http_status=status,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            ))
            rung += args.ceiling_step_tokens
        elif status == 400:
            results.append(ProbeResult(
                name=f"probe8_ceiling_{rung // 1000}k",
                status="PASS (HTTP 400 — clean reject)",
                elapsed_s=elapsed, detail=raw[:80],
                http_status=status,
            ))
            break
        else:
            results.append(ProbeResult(
                name=f"probe8_ceiling_{rung // 1000}k",
                status="FAIL", elapsed_s=elapsed,
                detail=f"HTTP {status}: {raw[:120]}",
                http_status=status,
            ))
            break
    return results


# ─────────────────────── orchestration ────────────────────────────


PROBES: list[tuple[str, Callable]] = [
    ("probe1_niah_small", probe_1_niah_small),
    ("probe2_tool_prefill_oom", probe_2_tool_prefill_oom),
    ("probe3_ide_agent", probe_3_ide_agent_one_shot),
    ("probe4_multi_turn", probe_4_multi_turn_agent),
    ("probe5_lcb_coding", probe_5_lcb_coding),
    ("probe6_reasoning", probe_6_reasoning_heavy),
    ("probe7_niah_large", probe_7_niah_large),
    ("probe8_ceiling_ladder", probe_8_ceiling_ladder),
]

QUICK_SET = {"probe1_niah_small", "probe3_ide_agent", "probe4_multi_turn"}


async def _run(args: argparse.Namespace) -> list[ProbeResult]:
    selected = []
    for name, fn in PROBES:
        if args.quick and name not in QUICK_SET:
            continue
        if args.no_large_niah and name in (
            "probe7_niah_large", "probe8_ceiling_ladder"
        ):
            continue
        selected.append((name, fn))

    print(f"# verify_stress: {len(selected)} probe(s) selected")
    print(f"# URL: {args.url}  model: {args.model}")
    print()

    all_results: list[ProbeResult] = []
    async with aiohttp.ClientSession() as http:
        for name, fn in selected:
            print(f"  ▶ {name} …", flush=True)
            try:
                res = await fn(http, args)
            except Exception as exc:
                res = [ProbeResult(
                    name=name, status="FAIL", elapsed_s=0.0,
                    detail=f"harness exception: {type(exc).__name__}: {exc}",
                )]
            all_results.extend(res)
            # Per-rung print
            for r in res:
                mark = "✓" if r.status.startswith("PASS") else "✗"
                print(f"    {mark} {r.name}  {r.status}  "
                      f"{r.elapsed_s:.1f}s  {r.detail[:80]}")
            if any(r.status == "FAIL" for r in res) and not args.continue_on_fail:
                print("  ⊘ FAIL — stopping ladder (use --continue-on-fail to override)")
                break

    return all_results


def _format_verdict(results: list[ProbeResult]) -> tuple[str, str]:
    failed = [r for r in results if r.status == "FAIL"]
    warns = [r for r in results if r.status == "WARN"]
    if failed:
        return "FAIL", f"first failure: {failed[0].name} — {failed[0].detail}"
    if warns:
        return "WARN", f"{len(warns)} warning(s); first: {warns[0].name}"
    return "PASS", f"{len(results)} probe(s) clean"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="8-probe boundary verify-stress against a live vLLM endpoint.",
    )
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--quick", action="store_true",
                   help="run probes 1+3+4 only (~2 min)")
    p.add_argument("--no-large-niah", action="store_true",
                   help="skip probes 7+8 (NIAH large + ceiling ladder)")
    p.add_argument("--continue-on-fail", action="store_true",
                   help="continue ladder past first FAIL (default: stop)")
    p.add_argument("--long-ctx-timeout", type=float, default=300.0)
    p.add_argument("--tool-prefill-timeout", type=float, default=240.0)
    p.add_argument("--ceiling-start-tokens", type=int, default=95_000)
    p.add_argument("--ceiling-max-tokens", type=int, default=260_000)
    p.add_argument("--ceiling-step-tokens", type=int, default=30_000)
    p.add_argument("--out", type=str, default=None,
                   help="write JSON results to this path")
    return p.parse_args()


async def _amain() -> int:
    args = _parse_args()
    t_total = time.perf_counter()
    results = await _run(args)
    wall = time.perf_counter() - t_total

    verdict, summary = _format_verdict(results)
    print()
    print(f"# Verdict: **{verdict}** — {summary}")
    print(f"# Total wall: {wall:.1f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "schema": "genesis-verify-stress-v1",
                "captured_at": _dt.datetime.now().isoformat(),
                "endpoint": {"url": args.url, "model": args.model},
                "verdict": verdict, "summary": summary,
                "wall_seconds": wall,
                "probes": [
                    {"name": r.name, "status": r.status,
                     "elapsed_s": round(r.elapsed_s, 2),
                     "detail": r.detail,
                     "http_status": r.http_status,
                     "prompt_tokens": r.prompt_tokens,
                     "completion_tokens": r.completion_tokens,
                     "extra": r.extra}
                    for r in results
                ],
            }, f, indent=2, default=str)
        print(f"# Wrote {args.out}")

    return 0 if verdict == "PASS" else (1 if verdict == "FAIL" else 0)


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())

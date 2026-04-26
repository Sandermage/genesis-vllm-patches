#!/usr/bin/env python3
"""
Genesis Ultimate Benchmark v3.0
— Comprehensive speed, stability, reliability, stress, and context sweep tests
   for vLLM + Genesis Patch (Qwen3.6 thinking models)

Tests:
  1. Speed Test          — throughput at various max_tokens (64..2048)
  2. Context Window      — standard steps (4k..160k)
  3. Context Sweep       — fine-grained (148k..160k, step 2k)
  4. Stability Test      — 30 sequential requests, error tracking
  5. Stress Test         — rapid-fire bursts, measuring degradation
  6. Long Generation     — 1024/2048 token outputs
  7. Server Metrics      — GPU cache, throughput counters

Usage:
    python genesis_bench_v3.py [--host HOST] [--port PORT] [--label LABEL]
                               [--speed-runs N] [--stability-n N]
                               [--stress-n N] [--max-context-k K]
                               [--sweep-from K] [--sweep-to K] [--sweep-step K]
"""

import argparse
import json
import time
import sys
import statistics
import traceback
from datetime import datetime

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# ── Config ──────────────────────────────────────────────────────────
API_KEY = "genesis-local"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
SEP = "─" * 58


# ── SSE streaming helper ───────────────────────────────────────────
def stream_chat(base_url, model_id, prompt, max_tokens=256, temperature=0.7, timeout=600):
    """
    Stream a chat completion.  Handles Qwen3.6 thinking model:
      delta.reasoning / delta.reasoning_content  → thinking tokens
      delta.content                              → answer tokens
    Returns dict with TTFT, speed, token counts, etc.
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    first_token_time = None
    reasoning_tokens = 0
    content_tokens = 0
    reasoning_text = ""
    content_text = ""
    finish_reason = None
    usage = None

    try:
        with requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers=HEADERS,
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                r_content = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if r_content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    reasoning_tokens += 1
                    reasoning_text += r_content

                c_content = delta.get("content") or ""
                if c_content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    content_tokens += 1
                    content_text += c_content

                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr

    except requests.exceptions.Timeout:
        return {"error": "timeout", "prompt_chars": len(prompt), "max_tokens": max_tokens}
    except requests.exceptions.HTTPError as e:
        return {"error": str(e)[:300], "prompt_chars": len(prompt), "max_tokens": max_tokens}
    except Exception as e:
        return {"error": str(e)[:300], "prompt_chars": len(prompt), "max_tokens": max_tokens}

    t_end = time.perf_counter()
    chunk_count = reasoning_tokens + content_tokens  # chunks received, NOT tokens
    total_time = t_end - t_start
    ttft = (first_token_time - t_start) if first_token_time else None
    gen_time = (t_end - first_token_time) if first_token_time else total_time

    server_completion = None
    server_prompt = None
    if usage:
        server_completion = usage.get("completion_tokens")
        server_prompt = usage.get("prompt_tokens")

    # v3.1 fix: vLLM nightly batches stream deltas (3-5 tokens per chunk),
    # so chunk_count !=  token_count. Use server-side usage.completion_tokens
    # when available (always set when stream_options.include_usage=true).
    # Fallback to chunk_count only if server didn't report usage.
    token_count = server_completion if server_completion is not None else chunk_count
    tps = token_count / gen_time if gen_time > 0 and token_count > 0 else 0

    return {
        "ttft_s": round(ttft, 4) if ttft else None,
        "total_time_s": round(total_time, 3),
        "gen_time_s": round(gen_time, 3),
        "tokens_generated": token_count,
        "chunks_received": chunk_count,
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": content_tokens,
        "tokens_per_sec": round(tps, 2),
        "finish_reason": finish_reason,
        "content_preview": content_text[:200],
        "reasoning_preview": reasoning_text[:100],
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "server_completion_tokens": server_completion,
        "server_prompt_tokens": server_prompt,
    }


# ── Helpers ─────────────────────────────────────────────────────────
def get_model_info(base_url):
    r = requests.get(f"{base_url}/v1/models", headers=HEADERS, timeout=10)
    r.raise_for_status()
    models = r.json()["data"]
    if not models:
        raise RuntimeError("No models loaded")
    m = models[0]
    return {"id": m["id"], "owned_by": m.get("owned_by", "unknown")}


def avg_or_none(values, decimals=2):
    if not values:
        return None
    return round(statistics.mean(values), decimals)


def stats_dict(values, label=""):
    if not values:
        return {"avg": None, "min": None, "max": None, "stdev": None, "count": 0}
    return {
        "avg": round(statistics.mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0,
        "count": len(values),
    }


def print_section(title):
    print(f"\n── {title} {SEP[len(title)+4:]}")


def get_server_metrics(base_url):
    """Fetch Prometheus metrics from vLLM."""
    print_section("Server Metrics")
    try:
        r = requests.get(f"{base_url}/metrics", headers=HEADERS, timeout=10)
        r.raise_for_status()
        text = r.text
        interesting = {}
        keys = [
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:gpu_cache_usage_perc",
            "vllm:cpu_cache_usage_perc",
            "vllm:avg_generation_throughput_toks_per_s",
            "vllm:avg_prompt_throughput_toks_per_s",
            "vllm:num_preemptions_total",
            "vllm:num_requests_total",
        ]
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            for key in keys:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        interesting[key] = parts[-1]
        for k, v in interesting.items():
            short = k.replace("vllm:", "")
            print(f"  {short}: {v}")
        return interesting
    except Exception as e:
        print(f"  Failed: {e}")
        return {"error": str(e)}


# ── Test 1: Speed ──────────────────────────────────────────────────
def test_speed(base_url, model_id, max_tokens_list=(64, 128, 256, 512, 1024, 2048), runs=3):
    """Throughput at different generation lengths."""
    print_section("Speed Test")
    prompt = ("Write a detailed technical analysis of modern GPU architectures, "
              "covering memory hierarchy, compute units, tensor cores, and their "
              "impact on machine learning inference performance. Include specific "
              "examples from NVIDIA Ampere and Hopper architectures.")
    results = []

    for mt in max_tokens_list:
        run_results = []
        print(f"  max_tokens={mt}:", end=" ", flush=True)
        for _ in range(runs):
            r = stream_chat(base_url, model_id, prompt, max_tokens=mt, temperature=0.7)
            if r.get("error"):
                print(f"ERR ", end="", flush=True)
                run_results.append(r)
                continue
            run_results.append(r)
            print(f"{r['tokens_per_sec']}t/s", end=" ", flush=True)
        print()

        valid = [x for x in run_results if not x.get("error")]
        if valid:
            avg_tps = avg_or_none([x["tokens_per_sec"] for x in valid])
            avg_ttft = avg_or_none([x["ttft_s"] for x in valid if x["ttft_s"]], 4)
            avg_total = avg_or_none([x["total_time_s"] for x in valid], 3)
            avg_tokens = avg_or_none([x["tokens_generated"] for x in valid], 1)
            tps_vals = [x["tokens_per_sec"] for x in valid]
            min_tps = round(min(tps_vals), 2) if tps_vals else None
            max_tps = round(max(tps_vals), 2) if tps_vals else None
            print(f"    AVG: {avg_tps} tok/s (min={min_tps}, max={max_tps}), "
                  f"TTFT={avg_ttft}s, {avg_tokens} tok, {avg_total}s total")
        else:
            avg_tps = avg_ttft = avg_total = avg_tokens = min_tps = max_tps = None
            print(f"    ALL FAILED")

        results.append({
            "max_tokens": mt,
            "runs": run_results,
            "avg_tokens_per_sec": avg_tps,
            "min_tokens_per_sec": min_tps,
            "max_tokens_per_sec": max_tps,
            "avg_ttft_s": avg_ttft,
            "avg_total_time_s": avg_total,
            "avg_tokens": avg_tokens,
            "successes": len(valid),
            "failures": len(run_results) - len(valid),
        })

    return results


# ── Test 2: Context Window (standard) ──────────────────────────────
def test_context_windows(base_url, model_id, sizes_k, runs=2):
    """TTFT and speed at different input sizes."""
    print_section("Context Window Test")
    word = "hello "
    results = []
    max_working_k = 0

    for size_k in sizes_k:
        n_words = int(size_k * 1000 / 1.3)
        prompt = word * n_words
        run_results = []
        print(f"  ~{size_k}k context:", end=" ", flush=True)

        for _ in range(runs):
            r = stream_chat(base_url, model_id, prompt, max_tokens=32, temperature=0.7, timeout=900)
            if r.get("error"):
                err_short = r["error"][:50]
                print(f"FAIL({err_short})", end=" ", flush=True)
                run_results.append(r)
                continue
            run_results.append(r)
            print(f"TTFT={r['ttft_s']}s/{r['tokens_per_sec']}t/s", end=" ", flush=True)

        print()
        valid = [x for x in run_results if not x.get("error")]
        if valid:
            max_working_k = size_k
            avg_ttft = avg_or_none([x["ttft_s"] for x in valid if x["ttft_s"]], 4)
            avg_tps = avg_or_none([x["tokens_per_sec"] for x in valid])
            print(f"    AVG: TTFT={avg_ttft}s, {avg_tps} tok/s")
        else:
            avg_ttft = avg_tps = None
            print(f"    ALL FAILED — max working: ~{max_working_k}k")
            results.append({"target_k": size_k, "status": "failed", "runs": run_results})
            break

        results.append({
            "target_k": size_k,
            "status": "ok",
            "avg_ttft_s": avg_ttft,
            "avg_tokens_per_sec": avg_tps,
            "runs": run_results,
        })

    print(f"\n  Max working context: ~{max_working_k}k tokens")
    return {"max_working_k": max_working_k, "details": results}


# ── Test 3: Context Sweep (fine-grained) ───────────────────────────
def test_context_sweep(base_url, model_id, from_k, to_k, step_k, runs=2):
    """Fine-grained context window sweep to find exact limit."""
    print_section(f"Context Sweep ({from_k}k → {to_k}k, step {step_k}k)")
    word = "hello "
    results = []
    max_stable_k = 0
    all_ok_k = 0

    sizes = list(range(from_k, to_k + 1, step_k))
    for size_k in sizes:
        n_words = int(size_k * 1000 / 1.3)
        prompt = word * n_words
        run_results = []
        successes = 0
        print(f"  ~{size_k}k:", end=" ", flush=True)

        for _ in range(runs):
            r = stream_chat(base_url, model_id, prompt, max_tokens=32, temperature=0.7, timeout=900)
            if r.get("error"):
                err_short = r["error"][:40]
                print(f"FAIL({err_short})", end=" ", flush=True)
                run_results.append(r)
                continue
            run_results.append(r)
            successes += 1
            print(f"OK(TTFT={r['ttft_s']}s/{r['tokens_per_sec']}t/s)", end=" ", flush=True)

        print()
        valid = [x for x in run_results if not x.get("error")]
        status = "ok" if successes == runs else ("partial" if successes > 0 else "failed")

        if successes > 0:
            max_stable_k = size_k
        if successes == runs:
            all_ok_k = size_k

        avg_ttft = avg_or_none([x["ttft_s"] for x in valid if x.get("ttft_s")], 4) if valid else None
        avg_tps = avg_or_none([x["tokens_per_sec"] for x in valid]) if valid else None

        print(f"    {status.upper()} — {successes}/{runs} passed" +
              (f", TTFT={avg_ttft}s, {avg_tps} tok/s" if valid else ""))

        results.append({
            "target_k": size_k,
            "status": status,
            "successes": successes,
            "total_runs": runs,
            "avg_ttft_s": avg_ttft,
            "avg_tokens_per_sec": avg_tps,
            "runs": run_results,
        })

        # If completely failed, check one more and then stop
        if successes == 0 and len(results) >= 2 and results[-2].get("successes", 0) == 0:
            print(f"  ↳ Two consecutive failures, stopping sweep at {size_k}k")
            break

    print(f"\n  Max fully stable: ~{all_ok_k}k tokens (all runs OK)")
    print(f"  Max partial:      ~{max_stable_k}k tokens (at least 1 run OK)")
    return {
        "from_k": from_k,
        "to_k": to_k,
        "step_k": step_k,
        "max_fully_stable_k": all_ok_k,
        "max_partial_k": max_stable_k,
        "details": results,
    }


# ── Test 4: Stability ─────────────────────────────────────────────
def test_stability(base_url, model_id, num_requests=30):
    """Sequential requests with different prompts and complexities."""
    print_section(f"Stability Test ({num_requests} requests)")
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list using merge sort.",
        "Расскажи про архитектуру трансформеров подробно.",
        "What are the key differences between TCP and UDP?",
        "Напиши подробный анализ производительности GPU NVIDIA.",
        "Explain the concept of attention mechanisms in neural networks.",
        "Создай детальный план развертывания ML системы.",
        "What is the time complexity of quicksort? Prove it.",
        "Опиши принцип работы KV-кэша в больших языковых моделях.",
        "Write a brief comparison of REST and GraphQL APIs.",
        "Напиши функцию на Python для бинарного поиска с тестами.",
        "Explain gradient descent optimization techniques.",
        "Какие есть методы оптимизации вывода нейросетей?",
        "Write a recursive Fibonacci with memoization in Python.",
        "Объясни разницу между FP16, BF16 и FP8 форматами.",
        "What is speculative decoding and how does it work?",
        "Расскажи о принципах SOLID на примерах Python.",
        "Explain the CAP theorem with real-world examples.",
        "Как работает Flash Attention и почему она быстрее?",
        "Write a Python async HTTP client with retry logic.",
        "Что такое MoE (Mixture of Experts) архитектура?",
        "Explain Docker networking: bridge, host, overlay modes.",
        "Напиши SQL запрос для аналитики продаж по месяцам.",
        "What are CUDA cores vs Tensor cores?",
        "Объясни как работает prefix caching в vLLM.",
        "Write a Python decorator for rate limiting.",
        "Расскажи про chunked prefill и его преимущества.",
        "Explain the difference between greedy and beam search.",
        "Как работает paged attention в vLLM?",
        "Write a concise summary of the Transformer paper.",
    ]

    results = []
    errors = 0
    tps_list = []
    ttft_list = []
    error_positions = []

    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        print(f"  [{i+1:2d}/{num_requests}]", end=" ", flush=True)
        r = stream_chat(base_url, model_id, prompt, max_tokens=128, temperature=0.7)
        if r.get("error"):
            print(f"ERROR: {r['error'][:60]}")
            errors += 1
            error_positions.append(i + 1)
        else:
            tps_list.append(r["tokens_per_sec"])
            if r["ttft_s"]:
                ttft_list.append(r["ttft_s"])
            print(f"OK — {r['tokens_per_sec']:5.1f} t/s, TTFT={r['ttft_s']}s, "
                  f"{r['tokens_generated']} tok ({r['reasoning_tokens']}r+{r['content_tokens']}c)")
        results.append(r)

    success_count = num_requests - errors
    print(f"\n  Result: {success_count}/{num_requests} OK, {errors} errors")
    if error_positions:
        print(f"  Errors at positions: {error_positions}")
    if tps_list:
        tps_stats = stats_dict(tps_list)
        ttft_stats = stats_dict(ttft_list)
        print(f"  Speed: avg={tps_stats['avg']} tok/s, min={tps_stats['min']}, "
              f"max={tps_stats['max']}, stdev={tps_stats['stdev']}")
        print(f"  TTFT:  avg={ttft_stats['avg']}s, min={ttft_stats['min']}, max={ttft_stats['max']}")

        # Check for degradation: compare first half vs second half
        mid = len(tps_list) // 2
        if mid > 1:
            first_half = statistics.mean(tps_list[:mid])
            second_half = statistics.mean(tps_list[mid:])
            drift_pct = round((second_half - first_half) / first_half * 100, 1)
            print(f"  Drift: first_half={round(first_half, 1)} → second_half={round(second_half, 1)} "
                  f"({'+' if drift_pct >= 0 else ''}{drift_pct}%)")
    else:
        tps_stats = ttft_stats = {"avg": None}

    return {
        "total_requests": num_requests,
        "successes": success_count,
        "errors": errors,
        "error_positions": error_positions,
        "tps_stats": stats_dict(tps_list) if tps_list else None,
        "ttft_stats": stats_dict(ttft_list) if ttft_list else None,
        "details": results,
    }


# ── Test 5: Stress Test ───────────────────────────────────────────
def test_stress(base_url, model_id, num_bursts=5, requests_per_burst=5):
    """Rapid-fire sequential bursts with minimal delay, checking for errors and degradation."""
    total_requests = num_bursts * requests_per_burst
    print_section(f"Stress Test ({num_bursts} bursts × {requests_per_burst} = {total_requests} requests)")

    prompt = "Quick: what is 2+2? Answer in one word."
    all_results = []
    burst_stats = []
    total_errors = 0

    for burst in range(num_bursts):
        print(f"  Burst {burst+1}/{num_bursts}:", end=" ", flush=True)
        burst_results = []
        burst_tps = []
        burst_errors = 0

        for req in range(requests_per_burst):
            r = stream_chat(base_url, model_id, prompt, max_tokens=64, temperature=0.1)
            if r.get("error"):
                print(f"ERR", end=" ", flush=True)
                burst_errors += 1
                total_errors += 1
            else:
                burst_tps.append(r["tokens_per_sec"])
                print(f"{r['tokens_per_sec']:.0f}", end=" ", flush=True)
            burst_results.append(r)
            all_results.append(r)

        avg_tps = avg_or_none(burst_tps)
        print(f" → avg={avg_tps} t/s, {burst_errors} err")

        burst_stats.append({
            "burst_num": burst + 1,
            "avg_tps": avg_tps,
            "errors": burst_errors,
            "results": burst_results,
        })

    # Overall stats
    all_tps = [r["tokens_per_sec"] for r in all_results if not r.get("error")]
    print(f"\n  Total: {total_requests - total_errors}/{total_requests} OK, {total_errors} errors")
    if all_tps:
        tps_stats = stats_dict(all_tps)
        print(f"  Speed: avg={tps_stats['avg']} tok/s, min={tps_stats['min']}, "
              f"max={tps_stats['max']}, stdev={tps_stats['stdev']}")

        # Compare first vs last burst
        first_burst_tps = [r["tokens_per_sec"] for r in burst_stats[0]["results"] if not r.get("error")]
        last_burst_tps = [r["tokens_per_sec"] for r in burst_stats[-1]["results"] if not r.get("error")]
        if first_burst_tps and last_burst_tps:
            f_avg = statistics.mean(first_burst_tps)
            l_avg = statistics.mean(last_burst_tps)
            drift = round((l_avg - f_avg) / f_avg * 100, 1)
            print(f"  Burst drift: first={round(f_avg, 1)} → last={round(l_avg, 1)} "
                  f"({'+' if drift >= 0 else ''}{drift}%)")

    return {
        "total_requests": total_requests,
        "total_errors": total_errors,
        "burst_stats": burst_stats,
        "overall_tps": stats_dict(all_tps) if all_tps else None,
    }


# ── Test 6: Long Generation ───────────────────────────────────────
def test_long_generation(base_url, model_id, token_counts=(1024, 2048)):
    """Test long generation outputs for stability and sustained throughput."""
    print_section("Long Generation Test")
    prompt = ("Write an extremely detailed and comprehensive analysis of the history "
              "of artificial intelligence, from its origins in the 1950s to modern "
              "large language models. Cover key milestones, breakthroughs, setbacks, "
              "and the people behind them. Include technical details about architectures, "
              "training methods, and hardware evolution.")
    results = []

    for mt in token_counts:
        print(f"  max_tokens={mt}:", end=" ", flush=True)
        r = stream_chat(base_url, model_id, prompt, max_tokens=mt, temperature=0.7, timeout=300)
        if r.get("error"):
            print(f"FAIL({r['error'][:50]})")
        else:
            print(f"{r['tokens_per_sec']} t/s, {r['tokens_generated']} tok, "
                  f"TTFT={r['ttft_s']}s, total={r['total_time_s']}s")
        results.append({"max_tokens": mt, "result": r})

    return results


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genesis Ultimate Benchmark v3.0")
    parser.add_argument("--host", default="192.168.1.10")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--label", default="genesis-v3")
    parser.add_argument("--speed-runs", type=int, default=3, help="Runs per speed test point")
    parser.add_argument("--max-context-k", type=int, default=160, help="Max context for standard test")
    parser.add_argument("--stability-n", type=int, default=30, help="Number of stability requests")
    parser.add_argument("--stress-bursts", type=int, default=5, help="Number of stress bursts")
    parser.add_argument("--stress-per-burst", type=int, default=5, help="Requests per stress burst")
    parser.add_argument("--sweep-from", type=int, default=148, help="Context sweep start (k)")
    parser.add_argument("--sweep-to", type=int, default=160, help="Context sweep end (k)")
    parser.add_argument("--sweep-step", type=int, default=2, help="Context sweep step (k)")
    parser.add_argument("--sweep-runs", type=int, default=3, help="Runs per sweep point")
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--skip-context", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--skip-long", action="store_true")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    iso_time = datetime.now().isoformat()

    # Build context sizes (standard)
    standard_ctx = [k for k in [4, 8, 16, 32, 64, 96, 128, 148, 160] if k <= args.max_context_k]

    print("=" * 60)
    print(f"Genesis Ultimate Benchmark v3.0 — {args.label}")
    print(f"Server: {base_url}")
    print(f"Time: {iso_time}")
    print(f"Config: speed_runs={args.speed_runs}, stability_n={args.stability_n}")
    print(f"        stress={args.stress_bursts}×{args.stress_per_burst}")
    print(f"        sweep={args.sweep_from}k→{args.sweep_to}k step {args.sweep_step}k (×{args.sweep_runs})")
    print("=" * 60)

    # ─── Model info ───
    print_section("Model Info")
    try:
        model = get_model_info(base_url)
        print(f"  Model: {model['id']}")
        print(f"  Owner: {model['owned_by']}")
    except Exception as e:
        print(f"  FATAL: {e}")
        sys.exit(1)

    all_results = {
        "benchmark_version": "3.0",
        "label": args.label,
        "timestamp": timestamp,
        "iso_time": iso_time,
        "server": base_url,
        "model": model,
        "config": {
            "speed_runs": args.speed_runs,
            "stability_n": args.stability_n,
            "stress_bursts": args.stress_bursts,
            "stress_per_burst": args.stress_per_burst,
            "sweep_from_k": args.sweep_from,
            "sweep_to_k": args.sweep_to,
            "sweep_step_k": args.sweep_step,
            "sweep_runs": args.sweep_runs,
            "max_context_k": args.max_context_k,
        },
    }

    # ─── Sanity check ───
    print_section("Sanity Check")
    r = stream_chat(base_url, model["id"], "Say 'hello world'", max_tokens=64, temperature=0.1)
    if r.get("error"):
        print(f"  FATAL: {r['error']}")
        sys.exit(1)
    print(f"  OK — {r['tokens_generated']} tokens ({r['reasoning_tokens']}r + {r['content_tokens']}c)")
    print(f"  TTFT: {r['ttft_s']}s, Speed: {r['tokens_per_sec']} tok/s")
    all_results["sanity"] = r

    # ─── Speed test ───
    if not args.skip_speed:
        all_results["speed"] = test_speed(
            base_url, model["id"],
            max_tokens_list=[64, 128, 256, 512, 1024, 2048],
            runs=args.speed_runs,
        )

    # ─── Context window test (standard) ───
    if not args.skip_context:
        all_results["context"] = test_context_windows(
            base_url, model["id"], standard_ctx, runs=2,
        )

    # ─── Context sweep (fine-grained) ───
    if not args.skip_sweep:
        all_results["context_sweep"] = test_context_sweep(
            base_url, model["id"],
            from_k=args.sweep_from, to_k=args.sweep_to, step_k=args.sweep_step,
            runs=args.sweep_runs,
        )

    # ─── Stability test ───
    if not args.skip_stability:
        all_results["stability"] = test_stability(
            base_url, model["id"], num_requests=args.stability_n,
        )

    # ─── Stress test ───
    if not args.skip_stress:
        all_results["stress"] = test_stress(
            base_url, model["id"],
            num_bursts=args.stress_bursts, requests_per_burst=args.stress_per_burst,
        )

    # ─── Long generation ───
    if not args.skip_long:
        all_results["long_generation"] = test_long_generation(
            base_url, model["id"], token_counts=[1024, 2048],
        )

    # ─── Server metrics ───
    all_results["metrics"] = get_server_metrics(base_url)

    # ─── Save ───
    filename = f"genesis_bench_{args.label}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {filename}")

    # ─── SUMMARY ───
    print("\n" + "=" * 60)
    print("FULL REPORT")
    print("=" * 60)
    print(f"Model:        {model['id']}")
    print(f"Benchmark:    Genesis Ultimate v3.0")
    print(f"Label:        {args.label}")
    print(f"Time:         {iso_time}")
    print()

    # Speed summary
    if "speed" in all_results:
        print("◆ SPEED TEST:")
        print(f"  {'max_tok':>8}  {'avg t/s':>8}  {'min':>6}  {'max':>6}  {'TTFT':>7}  {'tokens':>6}  {'time':>7}")
        print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*7}")
        for sp in all_results["speed"]:
            mt = sp["max_tokens"]
            tps = sp["avg_tokens_per_sec"] or 0
            mn = sp.get("min_tokens_per_sec") or 0
            mx = sp.get("max_tokens_per_sec") or 0
            ttft = sp["avg_ttft_s"] or 0
            tok = sp.get("avg_tokens") or 0
            tt = sp.get("avg_total_time_s") or 0
            print(f"  {mt:>8}  {tps:>8.1f}  {mn:>6.1f}  {mx:>6.1f}  {ttft:>6.4f}s  {tok:>6.0f}  {tt:>6.1f}s")
        print()

    # Context window summary
    if "context" in all_results:
        ctx = all_results["context"]
        print(f"◆ CONTEXT WINDOW (standard):")
        print(f"  Max working: ~{ctx['max_working_k']}k tokens")
        print(f"  {'size':>6}  {'TTFT':>8}  {'speed':>8}  {'status':>8}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}")
        for d in ctx["details"]:
            k = d["target_k"]
            st = d["status"]
            if st == "ok":
                ttft = d.get("avg_ttft_s") or 0
                tps = d.get("avg_tokens_per_sec") or 0
                print(f"  {k:>5}k  {ttft:>7.4f}s  {tps:>7.1f}  {'OK':>8}")
            else:
                print(f"  {k:>5}k  {'─':>8}  {'─':>8}  {'FAIL':>8}")
        print()

    # Context sweep summary
    if "context_sweep" in all_results:
        sw = all_results["context_sweep"]
        print(f"◆ CONTEXT SWEEP ({sw['from_k']}k → {sw['to_k']}k, step {sw['step_k']}k):")
        print(f"  Max fully stable: ~{sw['max_fully_stable_k']}k")
        print(f"  Max partial:      ~{sw['max_partial_k']}k")
        print(f"  {'size':>6}  {'TTFT':>8}  {'speed':>8}  {'pass':>6}  {'status':>8}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}")
        for d in sw["details"]:
            k = d["target_k"]
            st = d["status"]
            succ = d.get("successes", 0)
            total = d.get("total_runs", 0)
            ttft = d.get("avg_ttft_s")
            tps = d.get("avg_tokens_per_sec")
            ttft_s = f"{ttft:.4f}s" if ttft else "─"
            tps_s = f"{tps:.1f}" if tps else "─"
            print(f"  {k:>5}k  {ttft_s:>8}  {tps_s:>8}  {succ}/{total:>3}  {st.upper():>8}")
        print()

    # Stability summary
    if "stability" in all_results:
        st = all_results["stability"]
        print(f"◆ STABILITY TEST ({st['total_requests']} requests):")
        print(f"  Success rate: {st['successes']}/{st['total_requests']} "
              f"({round(st['successes']/st['total_requests']*100, 1)}%)")
        if st.get("tps_stats"):
            ts = st["tps_stats"]
            print(f"  Speed: avg={ts['avg']} tok/s, min={ts['min']}, max={ts['max']}, stdev={ts['stdev']}")
        if st.get("ttft_stats"):
            tf = st["ttft_stats"]
            print(f"  TTFT:  avg={tf['avg']}s, min={tf['min']}, max={tf['max']}")
        if st.get("error_positions"):
            print(f"  Errors at: {st['error_positions']}")
        print()

    # Stress summary
    if "stress" in all_results:
        sr = all_results["stress"]
        tr = sr["total_requests"]
        te = sr["total_errors"]
        print(f"◆ STRESS TEST ({tr} rapid-fire requests):")
        print(f"  Success rate: {tr - te}/{tr} ({round((tr-te)/tr*100, 1)}%)")
        if sr.get("overall_tps"):
            ot = sr["overall_tps"]
            print(f"  Speed: avg={ot['avg']} tok/s, min={ot['min']}, max={ot['max']}, stdev={ot['stdev']}")
        print()

    # Long generation summary
    if "long_generation" in all_results:
        print(f"◆ LONG GENERATION:")
        for item in all_results["long_generation"]:
            mt = item["max_tokens"]
            r = item["result"]
            if r.get("error"):
                print(f"  {mt} tokens: FAIL — {r['error'][:60]}")
            else:
                print(f"  {mt} tokens: {r['tokens_per_sec']} t/s, {r['tokens_generated']} tok, "
                      f"TTFT={r['ttft_s']}s, total={r['total_time_s']}s")
        print()

    # Verdict
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    issues = []
    if "stability" in all_results and all_results["stability"]["errors"] > 0:
        issues.append(f"Stability: {all_results['stability']['errors']} errors")
    if "stress" in all_results and all_results["stress"]["total_errors"] > 0:
        issues.append(f"Stress: {all_results['stress']['total_errors']} errors")
    if "context_sweep" in all_results:
        sw = all_results["context_sweep"]
        if sw["max_fully_stable_k"] < args.sweep_to:
            issues.append(f"Context: max stable={sw['max_fully_stable_k']}k < target {args.sweep_to}k")

    if not issues:
        print("  ✅ ALL TESTS PASSED — build is STABLE")
    else:
        print("  ⚠️  Issues detected:")
        for iss in issues:
            print(f"    • {iss}")

    print("=" * 60)


if __name__ == "__main__":
    main()

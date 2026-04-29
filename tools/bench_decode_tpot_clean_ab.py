#!/usr/bin/env python3
"""bench_decode_tpot_clean_ab.py — decode-only TPOT clean A/B harness.

Methodology adopted from thc1006's bench_v3_clean_ab.py:
    https://github.com/thc1006/qwen3.6-vllm-2x3090/blob/master/scripts/bench_v3_clean_ab.py

Why decode-only TPOT: wall_TPS conflates TTFT + queue + scheduler with decode
speed. For MTP / spec-decode A/B the fair primary metric is
decode_TPOT_ms = (elapsed - TTFT) / max(ct - 1, 1) * 1000.

Per-arm contract: SHA1 content audit, N=25 trials, per-prompt holds,
streaming with usage for authoritative completion_tokens. Process isolation
between arms is the operator's job: bounce server, re-invoke with a new
--arm-name, then --compare A.json B.json for Welch's t-test.

Style mirrors scripts/genesis_bench_v4.py (argparse, requests, SSE reader).
"""

import argparse, hashlib, json, math, statistics, sys, time
from datetime import datetime
try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

API_KEY_DEFAULT = "genesis-local"

PROMPTS_STANDARD = [
    "Write a detailed technical analysis of modern GPU architectures, covering memory hierarchy, compute units, tensor cores, and their impact on ML inference. /no_think",
    "Explain the difference between TCP and UDP in concrete operational terms: congestion control, ordering, head-of-line blocking, NAT traversal. /no_think",
    "Walk through how speculative decoding works in vLLM: draft model, verifier, acceptance sampling, and why MTP draft heads can outperform a separate draft model. /no_think",
    "Describe a deployment plan for a 35B LLM on two RTX A5000 GPUs: tensor parallelism, KV cache layout, prefix caching, and failure modes you would monitor. /no_think",
    "Write a Python async HTTP client that posts JSON with retry and exponential backoff, supports per-request deadlines, and emits structured logs. /no_think",
]

PROMPTS_SHORT = [
    "Why does the sky look blue? Two sentences. /no_think",
    "List five differences between Python and Go in bullets. /no_think",
    "Explain what KV cache is in transformers. /no_think",
    "Write a one-paragraph summary of attention. /no_think",
    "Give a haiku about debugging at 2am. /no_think",
]


def stream_chat(base_url, model_id, prompt, max_tokens, api_key,
                temperature=0.5, seed=42, timeout=600):
    """Stream a chat completion. Returns dict with TTFT/decode/TPOT split."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    server_completion = server_prompt = finish_reason = None
    try:
        with requests.post(f"{base_url}/v1/chat/completions", json=payload,
                           headers=headers, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    server_completion = chunk["usage"].get("completion_tokens")
                    server_prompt = chunk["usage"].get("prompt_tokens")
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                # /no_think → content; either field promotes TTFT.
                piece = (delta.get("content") or delta.get("reasoning")
                         or delta.get("reasoning_content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(piece)
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:300]}

    elapsed = time.perf_counter() - t0
    text = "".join(chunks)
    ct = server_completion if server_completion is not None else 0
    decode_s = max(elapsed - (ttft or 0.0), 0.0)
    return {
        "ttft_s": round(ttft, 4) if ttft else None,
        "elapsed_s": round(elapsed, 4),
        "decode_s": round(decode_s, 4),
        "completion_tokens": ct,
        "prompt_tokens": server_prompt,
        "decode_tpot_ms": round((decode_s / max(ct - 1, 1)) * 1000.0, 4) if ct > 1 else 0.0,
        "wall_tps": round((ct / elapsed) if elapsed > 0 and ct > 0 else 0.0, 2),
        "decode_tps": round(((ct - 1) / decode_s) if decode_s > 0 and ct > 1 else 0.0, 2),
        "finish_reason": finish_reason,
        "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()[:16],
        "text_len": len(text),
        "text_preview": text[:200],
    }


def fetch_accept_rate(base_url, api_key):
    """Best-effort vllm:spec_decode_* prom scrape. None if --disable-log-stats."""
    try:
        r = requests.get(f"{base_url}/metrics",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
        r.raise_for_status()
    except Exception:
        return None
    accepted = emitted = None
    for line in r.text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            if line.startswith("vllm:spec_decode_acceptance_rate"):
                return float(line.split()[-1])
            if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                accepted = float(line.split()[-1])
            elif line.startswith("vllm:spec_decode_num_emitted_tokens_total"):
                emitted = float(line.split()[-1])
        except Exception:
            continue
    return (accepted / emitted) if (accepted is not None and emitted) else None


def stats_dict(values):
    if not values:
        return {"mean": None, "std": None, "cv": None, "min": None, "max": None, "n": 0}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": round(mean, 4), "std": round(std, 4),
            "cv": round((std / mean) if mean else 0.0, 4),
            "min": round(min(values), 4), "max": round(max(values), 4), "n": len(values)}


def _student_t_two_sided_p(t, df):
    """Simpson's-rule integration of Student-t pdf for two-sided p (stdlib only)."""
    if df <= 0:
        return None
    abs_t, upper, n = abs(t), max(abs(t) + 30.0, 50.0), 4096
    h = (upper - abs_t) / n
    coef = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    pdf = lambda x: coef * (1.0 + (x * x) / df) ** (-(df + 1) / 2.0)
    s = pdf(abs_t) + pdf(upper)
    for i in range(1, n):
        s += (4.0 if i % 2 else 2.0) * pdf(abs_t + i * h)
    return min(max(2.0 * (h / 3.0) * s, 0.0), 1.0)


def welch_ttest(a, b):
    if len(a) < 2 or len(b) < 2:
        return None, None, None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    na, nb = len(a), len(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, na + nb - 2, 1.0
    t = (ma - mb) / se
    num = (va / na + vb / nb) ** 2
    den = ((va / na) ** 2) / max(na - 1, 1) + ((vb / nb) ** 2) / max(nb - 1, 1)
    df = num / den if den > 0 else (na + nb - 2)
    return round(t, 4), round(df, 2), round(_student_t_two_sided_p(t, df), 6)


def run_arm(base_url, model_id, api_key, prompts, runs, max_tokens,
            arm_name, quiet=False):
    def vp(*a, **kw):
        if not quiet:
            print(*a, **kw, flush=True)

    vp(f"\n=== arm '{arm_name}' — {runs} runs x {len(prompts)} prompts x {max_tokens} tokens ===")
    accept_pre = fetch_accept_rate(base_url, api_key)

    vp("  [warmup] full prompt set @ 64 tokens")
    for i, p in enumerate(prompts, 1):
        w = stream_chat(base_url, model_id, p, max_tokens=64, api_key=api_key)
        if w.get("error"):
            vp(f"    p{i}: WARMUP ERROR — {w['error']}")
        else:
            vp(f"    p{i}: ct={w['completion_tokens']:>4} ttft={1000*(w['ttft_s'] or 0):.0f}ms "
               f"decode_tpot={w['decode_tpot_ms']:.2f}ms sha1={w['text_sha1'][:10]}")

    per_prompt = {f"p{i+1}": [] for i in range(len(prompts))}
    flat_results = []
    for trial in range(1, runs + 1):
        vp(f"\n  trial {trial}/{runs}")
        for i, p in enumerate(prompts, 1):
            r = stream_chat(base_url, model_id, p, max_tokens=max_tokens, api_key=api_key)
            if r.get("error"):
                vp(f"    p{i}: ERROR — {r['error']}")
                per_prompt[f"p{i}"].append(None)
                flat_results.append({"trial": trial, "prompt_idx": i, "error": r["error"]})
                continue
            vp(f"    p{i}: ct={r['completion_tokens']:>4} ttft={1000*(r['ttft_s'] or 0):>5.0f}ms "
               f"decode={r['decode_s']:>5.2f}s decode_tpot={r['decode_tpot_ms']:>6.2f}ms "
               f"wall_tps={r['wall_tps']:>6.1f} sha1={r['text_sha1'][:10]}")
            per_prompt[f"p{i}"].append(r)
            flat_results.append({"trial": trial, "prompt_idx": i, **r})

    accept_post = fetch_accept_rate(base_url, api_key)

    per_prompt_stats = {}
    for k, results in per_prompt.items():
        good = [x for x in results if x and not x.get("error") and x["completion_tokens"] > 1]
        per_prompt_stats[k] = {
            "decode_tpot_ms": stats_dict([x["decode_tpot_ms"] for x in good]),
            "ttft_ms": stats_dict([1000 * x["ttft_s"] for x in good if x.get("ttft_s")]),
            "wall_tps": stats_dict([x["wall_tps"] for x in good]),
            "completion_tokens_mean": (round(statistics.mean(
                [x["completion_tokens"] for x in good]), 1) if good else None),
            "n_good": len(good), "n_total": len(results),
            "sha1s": sorted({x["text_sha1"] for x in good}),
        }

    good_flat = [x for x in flat_results if not x.get("error") and x.get("completion_tokens", 0) > 1]
    all_tpot = [x["decode_tpot_ms"] for x in good_flat]
    all_ttft = [1000 * x["ttft_s"] for x in good_flat if x.get("ttft_s")]
    all_wall = [x["wall_tps"] for x in good_flat]

    accept_mean = (round((accept_post + accept_pre) / 2, 4)
                   if accept_pre is not None and accept_post is not None
                   else (round(accept_post, 4) if accept_post is not None else None))
    summary = {
        "arm_name": arm_name,
        "decode_TPOT_ms": stats_dict(all_tpot),
        "TTFT_ms": stats_dict(all_ttft),
        "wall_TPS": stats_dict(all_wall),
        "accept_rate_pre": accept_pre, "accept_rate_post": accept_post,
        "accept_rate_mean": accept_mean,
        "n_runs": runs, "n_prompts": len(prompts), "max_tokens": max_tokens,
    }
    s = summary
    vp(f"\n  ── arm '{arm_name}' summary ──")
    vp(f"    decode_TPOT_ms : mean={s['decode_TPOT_ms']['mean']} std={s['decode_TPOT_ms']['std']} "
       f"cv={s['decode_TPOT_ms']['cv']} n={s['decode_TPOT_ms']['n']}")
    vp(f"    TTFT_ms        : mean={s['TTFT_ms']['mean']} std={s['TTFT_ms']['std']}")
    vp(f"    wall_TPS       : mean={s['wall_TPS']['mean']} std={s['wall_TPS']['std']}")
    vp(f"    accept_rate    : pre={accept_pre} post={accept_post} mean={accept_mean}")

    return {"summary": summary, "per_prompt": per_prompt_stats, "flat_results": flat_results}


def _verdict(p, delta_pct):
    if p is None:
        return "INCONCLUSIVE (insufficient samples)"
    if p >= 0.05:
        return f"NOT SIGNIFICANT (p={p:.4f})"
    if delta_pct is None:
        return f"SIGNIFICANT (p={p:.4f})"
    if delta_pct < 0:
        return f"B FASTER by {-delta_pct}% (p={p:.4f})"
    return f"B SLOWER by {delta_pct}% (p={p:.4f})"


def compare_arms(path_a, path_b, quiet=False):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    def _good_tpot(arm):
        return [x["decode_tpot_ms"] for x in arm["flat_results"]
                if not x.get("error") and x.get("completion_tokens", 0) > 1]

    a_tpot, b_tpot = _good_tpot(a), _good_tpot(b)
    t, df, p = welch_ttest(a_tpot, b_tpot)
    a_mean = a["summary"]["decode_TPOT_ms"]["mean"]
    b_mean = b["summary"]["decode_TPOT_ms"]["mean"]
    delta_pct = (round(100.0 * (b_mean - a_mean) / a_mean, 2) if a_mean and b_mean else None)

    def _arm_block(arm, path):
        s = arm["summary"]["decode_TPOT_ms"]
        return {"name": arm["arm_name"], "json": path,
                "decode_TPOT_ms_mean": s["mean"],
                "decode_TPOT_ms_std": s["std"], "n": s["n"]}

    out = {
        "arm_A": _arm_block(a, path_a),
        "arm_B": _arm_block(b, path_b),
        "delta_decode_TPOT_ms": (round(b_mean - a_mean, 4)
                                  if a_mean is not None and b_mean is not None else None),
        "delta_pct_vs_A": delta_pct,
        "welch_t": t, "welch_df": df, "welch_p_two_sided": p,
        "verdict": _verdict(p, delta_pct),
    }
    if not quiet:
        print("\n=== A/B comparison ===", flush=True)
        for tag, blk in (("A", out["arm_A"]), ("B", out["arm_B"])):
            print(f"  {tag} '{blk['name']}': mean={blk['decode_TPOT_ms_mean']}ms "
                  f"std={blk['decode_TPOT_ms_std']} n={blk['n']}")
        print(f"  delta = {out['delta_decode_TPOT_ms']} ms ({delta_pct}%)")
        print(f"  Welch t={t} df={df} p={p}")
        print(f"  verdict: {out['verdict']}")
    return out


def get_model_id(base_url, api_key):
    r = requests.get(f"{base_url}/v1/models",
                     headers={"Authorization": f"Bearer {api_key}"}, timeout=10)
    r.raise_for_status()
    data = r.json()["data"]
    if not data:
        raise RuntimeError("server reports no models loaded")
    return data[0]["id"]


def main():
    ap = argparse.ArgumentParser(
        description="Decode-only TPOT clean A/B bench (thc1006 methodology, ported).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--api-key", default=API_KEY_DEFAULT)
    ap.add_argument("--model", default=None, help="Defaults to first /v1/models entry.")
    ap.add_argument("--runs", type=int, default=25, help="Trials per prompt (thc1006: 25).")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Decode tokens per request.")
    ap.add_argument("--prompts", choices=("standard", "short"), default="standard")
    ap.add_argument("--arm-name", default="A", help="Arm label (e.g. 'baseline_v759').")
    ap.add_argument("--out", default=None, help="Output JSON. Default: bench_decode_tpot_<arm>_<ts>.json")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"),
                    help="Skip benching; compare two arm JSONs.")
    ap.add_argument("--compare-out", default=None, help="Write comparison JSON here.")
    args = ap.parse_args()

    if args.compare:
        result = compare_arms(args.compare[0], args.compare[1], quiet=args.quiet)
        if args.compare_out:
            with open(args.compare_out, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            if not args.quiet:
                print(f"\n  comparison written: {args.compare_out}", flush=True)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    base_url = f"http://{args.host}:{args.port}"
    prompts = PROMPTS_STANDARD if args.prompts == "standard" else PROMPTS_SHORT
    if not args.quiet:
        print("=" * 60)
        print(f"bench_decode_tpot_clean_ab.py — arm '{args.arm_name}'")
        print(f"  server={base_url}  prompts={args.prompts}({len(prompts)})  "
              f"runs={args.runs}  max_tokens={args.max_tokens}")
        print("=" * 60)
    model_id = args.model or get_model_id(base_url, args.api_key)
    if not args.quiet:
        print(f"  model      : {model_id}")

    arm_data = run_arm(base_url, model_id, args.api_key, prompts,
                       runs=args.runs, max_tokens=args.max_tokens,
                       arm_name=args.arm_name, quiet=args.quiet)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or f"bench_decode_tpot_{args.arm_name}_{ts}.json"
    payload = {
        "tool": "bench_decode_tpot_clean_ab",
        "methodology_source": "thc1006/qwen3.6-vllm-2x3090/scripts/bench_v3_clean_ab.py",
        "iso_time": datetime.now().isoformat(),
        "server": base_url, "model": model_id, "arm_name": args.arm_name,
        "config": {
            "runs": args.runs, "max_tokens": args.max_tokens,
            "prompts_set": args.prompts, "prompts": prompts,
            "temperature": 0.5, "seed": 42, "streaming": True,
            "include_usage": True, "no_think": True,
        },
        **arm_data,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if not args.quiet:
        print(f"\nResults saved to: {out}")
        print("Next: bounce server, re-run with different --arm-name, then "
              "--compare A.json B.json")


if __name__ == "__main__":
    main()

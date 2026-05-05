#!/usr/bin/env python3
"""Context-window sweep: prefix prompt with filler tokens to hit target N,
then generate 64 tokens; measure TTFT, decode tok/s, and a needle recall pass.

For each target context size (tokens):
  - Build prompt = <needle paragraph at start> + <filler> + <needle question>
  - Target total input tokens ~= N (within 2% of N)
  - Record: http_status, prompt_tokens, output_tokens, ttft, decode_tok_s,
            needle_recall_pass (bool), total_time_s
"""
import argparse, json, time
import urllib.request
import urllib.error

NEEDLE_CANARY = "CANARY_BRAVO_SIERRA_42"
NEEDLE_PARA = (
    f"IMPORTANT NOTE: Remember this exact code: {NEEDLE_CANARY}. "
    f"You will be asked to recall it at the end. The code is {NEEDLE_CANARY}.\n\n"
)
QUESTION = (
    f"\n\nWhat was the special code mentioned at the very start of this text? "
    f"Reply with ONLY the code itself and nothing else."
)

FILLER_WORD = "the quick brown fox jumps over the lazy dog. "
# rough 1 token per 3 chars English

def build_prompt(target_tokens, avg_chars_per_token=3.2):
    target_chars = int(target_tokens * avg_chars_per_token)
    # subtract needle_para + question overhead
    overhead = len(NEEDLE_PARA) + len(QUESTION)
    filler_chars = max(100, target_chars - overhead)
    filler = (FILLER_WORD * (filler_chars // len(FILLER_WORD) + 1))[:filler_chars]
    return NEEDLE_PARA + filler + QUESTION

def stream_chat(host, api_key, model, prompt, max_tokens=64, timeout=600):
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    text = ""
    usage = None
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                if not raw or not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except Exception:
                    continue
                if "usage" in j and j["usage"]:
                    usage = j["usage"]
                ch = j.get("choices") or []
                if not ch:
                    continue
                d = ch[0].get("delta") or {}
                chunk = d.get("content") or ""
                if chunk:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    tokens += 1
                    text += chunk
    except Exception as e:
        err = str(e)
    t_total = time.perf_counter() - t0
    comp = (usage and usage.get("completion_tokens")) or tokens
    prompt_tok = usage and usage.get("prompt_tokens")
    decode_t = t_total - (ttft or 0)
    tps = comp / decode_t if decode_t > 0 else 0
    return {
        "err": err,
        "ttft": ttft, "total": t_total, "comp_tokens": comp,
        "prompt_tokens": prompt_tok,
        "decode_tok_s": tps, "text": text,
    }

def needle_pass(text):
    return NEEDLE_CANARY in (text or "")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--api-key", default="genesis-local")
    p.add_argument("--model", default="qwen3.6-35b-a3b")
    p.add_argument("--from-k", type=int, default=160, help="start context in k tokens")
    p.add_argument("--to-k", type=int, default=512, help="end context (inclusive, k tokens)")
    p.add_argument("--step-k", type=int, default=4)
    p.add_argument("--runs", type=int, default=1, help="repeats per context (for stability)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--label", default="sweep")
    p.add_argument("--out", default="/tmp/context_sweep.jsonl")
    args = p.parse_args()

    outfh = open(args.out, "w")
    sizes = list(range(args.from_k, args.to_k + 1, args.step_k))
    total = len(sizes) * args.runs
    print(f"# sweep {args.label}: {len(sizes)} points {args.from_k}k..{args.to_k}k step {args.step_k}k, {args.runs} runs each = {total} trials")
    print(f"# ctx_k\tprompt_tok\tttft\tdecode_tok_s\tcomp_tok\tneedle\terror")
    for k in sizes:
        for r in range(args.runs):
            prompt = build_prompt(k * 1024)
            res = stream_chat(args.host, args.api_key, args.model, prompt, max_tokens=args.max_tokens)
            needle = needle_pass(res["text"])
            prompt_tok = res["prompt_tokens"] or -1
            ttft = res["ttft"] or 0
            tps = res["decode_tok_s"] or 0
            err = (res["err"] or "")[:80]
            status = "OK" if needle and not err else ("FAIL_NEEDLE" if not err else "ERR")
            print(f"  {k}k\tprompt={prompt_tok}\tttft={ttft:.2f}s\tdecode={tps:.1f}t/s\tcomp={res['comp_tokens']}\tneedle={needle}\t{status}\t{err}")
            outfh.write(json.dumps({
                "label": args.label, "ctx_k": k, "run": r, "timestamp": time.time(),
                "prompt_tokens": prompt_tok, "ttft_s": ttft, "decode_tok_s": tps,
                "comp_tokens": res["comp_tokens"], "total_s": res["total"],
                "needle_pass": needle, "error": err,
                "first_80_chars": (res["text"] or "")[:80],
            }) + "\n")
            outfh.flush()
    outfh.close()
    print(f"# done, results at {args.out}")

if __name__ == "__main__":
    main()

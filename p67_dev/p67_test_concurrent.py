"""P67 concurrent (B=2) test — verify kernel works with concurrent requests.

Production has --max-num-seqs 2, meaning up to 2 concurrent requests can
batch together. P67's kernel is grid (B, num_head_groups, 1) — needs to
handle B=2 cases. So far only B=1 has been observed in real workload.

This test fires 4 concurrent requests via threading. With max-num-seqs 2,
the engine will batch pairs of decode steps together → P67 sees B=2.

Usage: python3 p67_test_concurrent.py
"""
import json, urllib.request, time, threading, sys

URL = "http://localhost:8000/v1/chat/completions"
HDR = {"Authorization": "Bearer genesis-local", "Content-Type": "application/json"}
MODEL = "qwen3.6-35b-a3b"


def call(prompt: str, max_tokens: int = 80) -> tuple[float, int, str]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
    }
    t0 = time.time()
    r = urllib.request.urlopen(
        urllib.request.Request(URL, data=json.dumps(body).encode(), headers=HDR),
        timeout=60,
    )
    elapsed = time.time() - t0
    j = json.loads(r.read())
    n_tok = j["usage"]["completion_tokens"]
    content = j["choices"][0]["message"]["content"]
    return elapsed, n_tok, content


def worker(prompt: str, idx: int, results: list, lock: threading.Lock):
    try:
        elapsed, n_tok, content = call(prompt)
        with lock:
            results.append({"idx": idx, "elapsed": elapsed, "n_tok": n_tok,
                            "content": content[:80], "ok": True})
    except Exception as e:
        with lock:
            results.append({"idx": idx, "elapsed": 0, "n_tok": 0,
                            "content": f"ERROR: {e}", "ok": False})


def main():
    prompts = [
        "Write a haiku about mountains.",
        "Name 5 colors of the rainbow.",
        "What is 17 times 23? Answer with just the number.",
        "Explain gravity in 2 sentences.",
    ]
    print(f"=== P67 concurrent test: {len(prompts)} parallel requests ===")
    results = []
    lock = threading.Lock()
    threads = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        t = threading.Thread(target=worker, args=(p, i, results, lock))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    total = time.time() - t0
    results.sort(key=lambda x: x["idx"])
    all_ok = True
    total_tok = 0
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        all_ok = all_ok and r["ok"]
        total_tok += r["n_tok"]
        print(f"  {r['idx']} [{status}] {r['n_tok']:3d}tok in {r['elapsed']:.2f}s — {r['content']!r}")
    print(f"\n--- Aggregate: {total_tok} toks / {total:.2f}s wall = "
          f"{total_tok/total:.1f} tok/s combined throughput")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

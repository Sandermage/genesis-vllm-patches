"""P67 long-context test — needle-in-haystack at 16K, 64K, 128K context.

Verifies that P67 doesn't degrade accuracy on long context where Phase 1
loop runs many iterations (potentially exposes accumulator overflow,
sanitization-induced losses, or block_table indexing bugs).

Generates a haystack of fluff text with a unique "needle" sentence at a
specific depth. Asks the model to retrieve the needle. Compares P67 ON
vs OFF accuracy.

Usage (from server):
  python3 p67_test_long_context.py [16k|64k|128k]
"""
import json, urllib.request, time, sys, os

URL = "http://localhost:8000/v1/chat/completions"
HDR = {"Authorization": "Bearer genesis-local", "Content-Type": "application/json"}
MODEL = "qwen3.6-35b-a3b"


def make_haystack(target_tokens: int, needle_at_depth: float = 0.5) -> tuple[str, str, str]:
    """Build a haystack with a needle at given depth (0.0=start, 1.0=end).

    Returns: (haystack, needle_sentence, retrieval_question)
    """
    fluff_paragraph = (
        "The Roman Empire was characterized by its vast military, "
        "intricate political structures, and sophisticated engineering. "
        "Aqueducts brought water to cities, roads connected provinces, "
        "and the legions enforced Pax Romana. Trade flourished across "
        "the Mediterranean, with grain from Egypt, wine from Gaul, and "
        "marble from Greece. The empire's longevity depended on a delicate "
        "balance between central authority and regional autonomy. "
    )
    # ~80 tokens per paragraph with this prompt
    needed_paragraphs = target_tokens // 80
    needle = (
        "The secret pass-phrase chosen by Marcus Aurelius for the year "
        "180 AD was 'Olympic Phoenix Velvet Cascade Nebula'."
    )
    insert_at = int(needed_paragraphs * needle_at_depth)
    paragraphs = [fluff_paragraph] * needed_paragraphs
    paragraphs.insert(insert_at, needle)
    haystack = "".join(paragraphs)
    question = (
        "What was the secret pass-phrase chosen by Marcus Aurelius for the year 180 AD?"
    )
    return haystack, needle, question


def call(prompt: str, max_tokens: int = 100) -> tuple[float, int, str]:
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
        timeout=180,
    )
    elapsed = time.time() - t0
    j = json.loads(r.read())
    n_tok = j["usage"]["completion_tokens"]
    content = j["choices"][0]["message"]["content"]
    return elapsed, n_tok, content


def main():
    sizes = {
        "8k": 8000, "16k": 16000, "32k": 32000, "64k": 64000, "128k": 128000,
    }
    selected = sys.argv[1] if len(sys.argv) > 1 else "16k"
    if selected not in sizes:
        print(f"Unknown size {selected}. Choices: {list(sizes.keys())}")
        return 1
    target = sizes[selected]

    haystack, needle, question = make_haystack(target, needle_at_depth=0.5)
    full_prompt = (
        f"You are reading a long document. Find a specific fact at the end.\n\n"
        f"--- DOCUMENT START ---\n{haystack}\n--- DOCUMENT END ---\n\n"
        f"Question: {question}\nAnswer concisely with only the pass-phrase."
    )
    print(f"=== Needle-in-haystack: {selected} (~{target} tokens) ===")
    print(f"Needle inserted at depth=0.5")
    print(f"Prompt length: ~{len(full_prompt) // 4} tokens (rough)")

    try:
        elapsed, n_tok, content = call(full_prompt)
    except Exception as e:
        print(f"FAIL: request raised {type(e).__name__}: {e}")
        return 1

    expected_phrase = "Olympic Phoenix Velvet Cascade Nebula"
    found = expected_phrase.lower() in content.lower()
    print(f"  Time: {elapsed:.1f}s   tokens out: {n_tok}   tok/s: {n_tok/elapsed:.1f}")
    print(f"  Expected: {expected_phrase!r}")
    print(f"  Got:      {content[:200]!r}")
    print(f"  {'✓ PASS — needle retrieved' if found else '✗ FAIL — needle missing'}")
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main())

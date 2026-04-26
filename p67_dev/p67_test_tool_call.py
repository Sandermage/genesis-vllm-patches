"""P67 tool-call regression test — verify P67 doesn't break tool call quality.

Tool-call clean rate is our PRIMARY quality metric (per memory:
v7.13_strict_ngram_breakthrough). With strict ngram (prompt_lookup_min=8)
we get 100% clean on single-query / 96% on multi-query diverse.

P67 changes the attention path under spec-decode. We need to verify
tool-call quality is preserved.

Tests:
1. Single tool-call (noonghunna's reproducer) × 10
2. Multi-tool selection × 5
3. Long-context tool-call (P68/P69 territory)

Usage: python3 p67_test_tool_call.py
"""
import json, urllib.request, time, sys

URL = "http://localhost:8000/v1/chat/completions"
HDR = {"Authorization": "Bearer genesis-local", "Content-Type": "application/json"}
MODEL = "qwen3.6-35b-a3b"


def call(body, timeout=60):
    r = urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers=HDR), timeout=timeout)
    return json.loads(r.read())


def test_1_single_tool(n=10):
    print(f"\n=== Test 1: Single tool-call (noonghunna repro) × {n} ===")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the weather in Paris in Celsius? Use the tool."}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
                "required": ["city", "unit"]}}}],
        "tool_choice": "auto",
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
    }
    clean = 0
    times = []
    for i in range(1, n + 1):
        try:
            t0 = time.time()
            r = call(body)
            ms = (time.time() - t0) * 1000
            times.append(ms)
            msg = r["choices"][0]["message"]
            tc = msg.get("tool_calls") or []
            ok = bool(tc and tc[0].get("function", {}).get("name") == "get_weather")
            if ok:
                clean += 1
            content = (msg.get("content") or "")[:60]
            print(f"  {i:2d} {'OK' if ok else 'FAIL':4} {ms:6.0f}ms tc={len(tc)} content={content!r}")
        except Exception as e:
            print(f"  {i:2d} ERR {e}")
    print(f"--- Test 1: {clean}/{n} clean (avg {sum(times)/len(times):.0f}ms)")
    return clean, n


def test_2_multi_tool(n=5):
    print(f"\n=== Test 2: Multi-tool selection × {n} ===")
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "First check stock price for AAPL, then get weather for NYC."}],
        "tools": [
            {"type": "function", "function": {"name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
            {"type": "function", "function": {"name": "get_stock",
                "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}},
        ],
        "tool_choice": "auto",
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
    }
    clean = 0
    for i in range(1, n + 1):
        try:
            r = call(body)
            tc = r["choices"][0]["message"].get("tool_calls") or []
            ok = bool(tc and tc[0].get("function", {}).get("name") in ("get_stock", "get_weather"))
            if ok:
                clean += 1
            print(f"  {i:2d} {'OK' if ok else 'FAIL':4} tc={len(tc)} first={(tc[0].get('function',{}).get('name') if tc else None)!r}")
        except Exception as e:
            print(f"  {i:2d} ERR {e}")
    print(f"--- Test 2: {clean}/{n} clean")
    return clean, n


def test_3_long_ctx_tool(n=3):
    print(f"\n=== Test 3: Long-ctx tool-call (P68/P69 territory) × {n} ===")
    fluff = "Fact about history: Romans built aqueducts. " * 200  # ~10K chars
    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": f"{fluff}\n\nNow check the weather in Tokyo using the tool."},
        ],
        "tools": [{"type": "function", "function": {
            "name": "get_weather",
            "parameters": {"type": "object",
                "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
        "tool_choice": "auto",
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.0,
    }
    clean = 0
    for i in range(1, n + 1):
        try:
            r = call(body)
            tc = r["choices"][0]["message"].get("tool_calls") or []
            ok = bool(tc and tc[0].get("function", {}).get("name") == "get_weather")
            if ok:
                clean += 1
            print(f"  {i:2d} {'OK' if ok else 'FAIL':4} tc={len(tc)}")
        except Exception as e:
            print(f"  {i:2d} ERR {e}")
    print(f"--- Test 3: {clean}/{n} clean")
    return clean, n


def main():
    c1, n1 = test_1_single_tool()
    c2, n2 = test_2_multi_tool()
    c3, n3 = test_3_long_ctx_tool()
    total_c = c1 + c2 + c3
    total_n = n1 + n2 + n3
    print(f"\n=== AGGREGATE: {total_c}/{total_n} clean ({total_c*100//total_n}%) ===")
    return 0 if total_c == total_n else 1


if __name__ == "__main__":
    sys.exit(main())

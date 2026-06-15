# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the prompt library + managed declarative tools + copilot wiring."""
from __future__ import annotations

import pytest


@pytest.fixture
def store_home(tmp_path, monkeypatch):
    # Redirect the operator-local store dir into a temp dir.
    monkeypatch.setattr("sndr.engines.vllm.locations.project_paths.install_root", lambda: tmp_path)
    return tmp_path


def test_prompts_builtins_plus_crud(store_home):
    from sndr.product_api.legacy import prompts_store as ps
    ids = {p["id"] for p in ps.list_prompts()}
    assert "crypto-analyst" in ids and "general" in ids        # seeds present
    rec = ps.create_prompt("My News", "Analyse the news.", title="t")
    assert rec["id"] == "my-news" and rec["builtin"] is False
    assert any(p["id"] == "my-news" for p in ps.list_prompts())
    ps.update_prompt("my-news", content="Updated content.")
    assert ps.get_prompt("my-news")["content"] == "Updated content."
    assert ps.delete_prompt("my-news") is True
    assert ps.delete_prompt("crypto-analyst") is False         # builtins are read-only


def test_prompts_require_name_and_content(store_home):
    from sndr.product_api.legacy import prompts_store as ps
    with pytest.raises(ValueError):
        ps.create_prompt("", "x")
    with pytest.raises(ValueError):
        ps.create_prompt("name", "   ")


def test_tools_validation(store_home):
    from sndr.product_api.legacy import tools_store as ts
    with pytest.raises(ValueError):
        ts.create_tool("Bad Name", "https://x.com/a")          # not a lowercase identifier
    with pytest.raises(ValueError):
        ts.create_tool("t", "ftp://x.com/a")                   # scheme must be http(s)
    with pytest.raises(ValueError):
        ts.create_tool("t", "https://x.com/{p}", params=[])    # placeholder without a parameter


def test_tools_spec_and_ssrf_safe_executor(store_home, monkeypatch):
    from sndr.product_api.legacy import tools_store as ts
    ts.create_tool("price", "https://api.ex.com/price/{coin}?d={days}", description="coin price",
                   params=[{"name": "coin", "type": "string", "required": True},
                           {"name": "days", "type": "integer"}])
    spec = ts.enabled_tool_specs()[0]
    assert spec["name"] == "price"
    assert "coin" in spec["parameters"]["properties"] and spec["parameters"]["required"] == ["coin"]

    captured = {}

    class _Resp:
        def read(self, n=-1):
            return b'{"ok": 1}'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ts.urllib.request, "urlopen", lambda req, timeout=15.0: (captured.update(url=req.full_url) or _Resp()))
    out = ts.run_tool("price", {"coin": "btc/../secret", "days": 7})
    assert out["result"] == {"ok": 1}
    assert "api.ex.com" in captured["url"]                     # host is fixed by the template
    assert "btc%2F..%2Fsecret" in captured["url"]              # arg URL-encoded — can't break the path/host
    with pytest.raises(ValueError):
        ts.run_tool("price", {"days": 7})                      # missing required 'coin'


def test_disabled_tool_not_exposed_or_run(store_home):
    from sndr.product_api.legacy import tools_store as ts
    ts.create_tool("hidden", "https://api.ex.com/x", enabled=False)
    assert ts.enabled_tool_specs() == []
    with pytest.raises(ValueError):
        ts.run_tool("hidden", {})


def test_copilot_exposes_and_runs_managed_tools(store_home, monkeypatch):
    from sndr.product_api.legacy import tools_store as ts, copilot
    ts.create_tool("weather", "https://api.w.com/{city}",
                   params=[{"name": "city", "type": "string", "required": True}])
    cat = {t["name"]: t["category"] for t in copilot.tool_catalog()}
    assert cat.get("weather") == "custom"
    assert "weather" in {s["function"]["name"] for s in copilot.tool_specs()}
    monkeypatch.setattr(ts, "run_tool", lambda tid, args: {"tool": tid, "result": "X", "args": args})
    out = copilot.execute_tool("weather", {"city": "kyiv"})
    assert out["ok"] is True and out["result"]["result"] == "X" and out["result"]["args"]["city"] == "kyiv"


def test_native_crypto_tools_registered():
    from sndr.product_api.legacy import copilot
    names = {t["name"] for t in copilot.tool_catalog()}
    assert {"crypto_market_overview", "coin_data", "news_analysis"} <= names

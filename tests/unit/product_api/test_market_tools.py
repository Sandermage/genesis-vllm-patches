# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the native market & news tools (ported from OpenWebUI). No network."""
from __future__ import annotations

import pytest

from sndr.product_api.legacy import market_tools as mt


def _fake_get(url, timeout=12.0):
    if "coins/markets" in url and "symbols=" not in url:
        return [{"symbol": "btc", "current_price": 69000, "price_change_percentage_24h": 2.5, "market_cap": 1.3e12}]
    if url.endswith("/global"):
        return {"data": {"total_market_cap": {"usd": 2.4e12}, "market_cap_change_percentage_24h_usd": 1.1,
                         "market_cap_percentage": {"btc": 54.2, "eth": 17.1}}}
    if "alternative.me" in url:
        return {"data": [{"value": "72", "value_classification": "Greed"}]}
    if "premiumIndex" in url:
        return {"lastFundingRate": "0.0001", "markPrice": "69010"}
    if "openInterest" in url:
        return {"openInterest": "85000.5"}
    if "finance/spark" in url:  # batched macro request, keyed by raw symbol
        return {"DX-Y.NYB": {"close": [100, 102]}, "^GSPC": {"close": [100, 102]},
                "GC=F": {"close": [4200, 4335]}, "^VIX": {"close": [18, 17, None]}}
    return {}


def test_crypto_overview_shape(monkeypatch):
    monkeypatch.setattr(mt, "_get_json", _fake_get)
    out = mt.crypto_market_overview()
    assert out["top_coins"][0]["symbol"] == "BTC" and out["top_coins"][0]["price"] == 69000
    assert out["global"]["btc_dominance_pct"] == 54.2
    assert out["fear_greed"]["value"] == 72
    # 0.0001 * 100 = 0.01 -> not > 0.01 -> neutral
    assert out["btc_derivatives"]["btc_funding_state"] == "neutral"
    assert out["btc_derivatives"]["btc_open_interest"] == 85000.5
    assert out["macro"]["dxy"]["change_pct"] == 2.0  # (102-100)/100*100
    assert out["data_quality"]["top_coins"] == "ok"


def test_crypto_overview_degrades_per_section(monkeypatch):
    def partial(url, timeout=12.0):
        if url.endswith("/global"):
            raise RuntimeError("source down")
        return _fake_get(url, timeout)
    monkeypatch.setattr(mt, "_get_json", partial)
    out = mt.crypto_market_overview()
    assert "unavailable" in out["data_quality"]["global"]  # flagged, not crashed
    assert out["data_quality"]["fear_greed"] == "ok"        # others still work


def test_coin_data(monkeypatch):
    monkeypatch.setattr(mt, "_get_json", lambda url, timeout=12.0: [
        {"symbol": "eth", "current_price": 3500, "price_change_percentage_24h_in_currency": 1.2,
         "price_change_percentage_7d_in_currency": 5.0, "market_cap": 4e11, "total_volume": 2e10, "market_cap_rank": 2}])
    out = mt.coin_data("ETH,DOGE")
    assert out["coins"][0]["symbol"] == "ETH" and out["coins"][0]["price"] == 3500
    assert out["coins"][1]["symbol"] == "DOGE" and "error" in out["coins"][1]


def test_coin_data_requires_symbols():
    with pytest.raises(ValueError):
        mt.coin_data("  ")


def test_news_analysis_classes(monkeypatch):
    from sndr.product_api.legacy import external_clients
    monkeypatch.setattr(external_clients, "web_search",
                        lambda q, **k: {"results": [{"title": "T", "url": "http://u", "snippet": "s"}]})
    out = mt.news_analysis()
    assert {"etf_flows", "liquidations", "geopolitics", "regulation"} <= set(out["classes"])
    assert out["total_results"] >= 4
    assert out["by_class"]["etf_flows"][0]["url"] == "http://u"


def test_news_analysis_one_class_down(monkeypatch):
    from sndr.product_api.legacy import external_clients

    def flaky(q, **k):
        if "liquidations" in q:
            raise external_clients.ServiceError("search down")
        return {"results": [{"title": "T", "url": "http://u", "snippet": "s"}]}
    monkeypatch.setattr(external_clients, "web_search", flaky)
    out = mt.news_analysis()
    assert out["by_class"]["liquidations"] == []      # degraded
    assert out["by_class"]["etf_flows"]                # others fine
    assert "liquidations" in out.get("errors", {})


def test_extract_crypto_symbols_case_insensitive_majors_only():
    assert mt._extract_crypto_symbols("дай анализ актива wld на сегодня") == ["WLD"]
    assert set(mt._extract_crypto_symbols("compare BTC and eth please")) == {"BTC", "ETH"}
    assert mt._extract_crypto_symbols("how are you today") == []   # no false positives
    assert mt._extract_crypto_symbols("") == []


def test_market_grounding_formats_live_figures(monkeypatch):
    monkeypatch.setattr(mt, "coin_data", lambda s: {"coins": [
        {"symbol": "WLD", "price": 0.684, "change_24h_pct": 13.7, "change_7d_pct": 34.0,
         "mcap": 2_331_000_000, "volume_24h": 1_341_000_000, "rank": 42}]})
    g = mt.market_grounding("analyze wld today")
    assert "WLD" in g and "0.684" in g and "13.7" in g and "#42" in g
    assert "EXACT figures" in g


def test_market_grounding_empty_without_crypto():
    assert mt.market_grounding("hello there friend") == ""


def test_date_grounding_includes_current_date_and_anti_refusal():
    from datetime import datetime, timezone

    dt = datetime(2026, 6, 17, 9, 30, tzinfo=timezone.utc)
    out = mt.date_grounding(dt)
    assert "2026-06-17" in out          # the actual current date
    assert "Wednesday" in out           # weekday so the model can reason about "today"
    assert "UTC" in out
    assert "never refuse" in out.lower()  # anti-refusal: don't bail on "future" dates


def test_date_grounding_defaults_to_now():
    out = mt.date_grounding()
    assert out.startswith("Current date:")
    assert "UTC" in out


def test_needs_web_search_positive_signals():
    pos = [
        "дай анализ актива WLD на сегодня",
        "what's the latest news on BTC",
        "who won the election today",
        "what happened this week in crypto",
        "search for the current ETH price",
        "что нового сейчас на рынке",
        "последние новости по worldcoin",
        "release of the new model in 2026",
        "загугли это",
        "what is the current weather",
    ]
    for q in pos:
        assert mt.needs_web_search(q), f"should trigger: {q!r}"


def test_needs_web_search_negative_signals():
    neg = [
        "hello",
        "explain how proof of stake works",
        "write a python function to sort a list",
        "что такое блокчейн",
        "how does a transformer model work",
        "summarize this text",
    ]
    for q in neg:
        assert not mt.needs_web_search(q), f"should NOT trigger: {q!r}"


def test_market_grounding_tolerates_null_mcap_volume(monkeypatch):
    # CoinGecko returns null mcap/volume for thin/new coins; the formatter must
    # not crash (f"{None:,}" -> TypeError), which previously dropped ALL grounding.
    monkeypatch.setattr(mt, "coin_data", lambda s: {"coins": [
        {"symbol": "WLD", "price": 0.01, "change_24h_pct": 1.0, "change_7d_pct": 2.0,
         "mcap": None, "volume_24h": None, "rank": None}]})
    g = mt.market_grounding("analyze WLD")
    assert "WLD" in g and "0.01" in g          # row still produced
    assert "n/a" in g                          # null money fields degrade gracefully

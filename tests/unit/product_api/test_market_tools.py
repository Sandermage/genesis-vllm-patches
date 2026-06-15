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
    if "finance/chart" in url:
        return {"chart": {"result": [{"indicators": {"quote": [{"close": [100, 102]}]}}]}}
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

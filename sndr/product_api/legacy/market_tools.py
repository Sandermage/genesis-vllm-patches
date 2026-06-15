# SPDX-License-Identifier: Apache-2.0
"""Native market & news intelligence tools for the SNDR copilot.

A faithful core port of the operator's OpenWebUI ``crypto_market_data`` tool:
live crypto prices + global + Fear&Greed + BTC-derivatives + macro, plus a
news/info-field analysis tool that classifies the news field (ETF flows,
liquidations, geopolitics, regulation) over the web-search backend.

Same discipline as :mod:`engine_client` / :mod:`external_clients`: stdlib
``urllib`` only, a fixed allow-list of public no-auth API hosts (anti-SSRF),
short timeouts, and partial results on a flaky source (each section is
best-effort and flagged in ``data_quality``) rather than failing the whole call.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

# Fixed public hosts the tool may reach — no auth, free. A crafted symbol/arg can
# never redirect a request elsewhere (scheme + host + path are built here).
_COINGECKO = "https://api.coingecko.com/api/v3"
_FNG = "https://api.alternative.me/fng/?limit=1"
_BINANCE_FUT = "https://fapi.binance.com"
_BINANCE_SPOT = "https://api.binance.com"
_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart"
_UA = "sndr-market-tools/1.0 (+https://sndr.local)"


def _get_json(url: str, *, timeout: float = 12.0) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed public hosts
        return json.loads(resp.read().decode("utf-8", "replace"))


def _try(section: str, fn, quality: dict[str, str]) -> Any:
    """Run a best-effort section; record its status in ``quality`` instead of
    raising, so one dead source never sinks the whole overview."""
    try:
        out = fn()
        quality[section] = "ok"
        return out
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, flag it
        quality[section] = f"unavailable ({type(exc).__name__})"
        return None


def _yahoo_last(symbol: str) -> Optional[dict[str, Any]]:
    enc = urllib.parse.quote(symbol, safe="")
    data = _get_json(f"{_YAHOO}/{enc}?interval=1d&range=5d", timeout=10.0)
    res = (((data or {}).get("chart") or {}).get("result") or [{}])[0]
    closes = [c for c in ((((res.get("indicators") or {}).get("quote") or [{}])[0]).get("close") or []) if c is not None]
    if len(closes) < 2:
        return None
    last, prev = closes[-1], closes[-2]
    return {"value": round(last, 2), "change_pct": round((last - prev) / prev * 100, 2) if prev else None}


# ── crypto market overview (the headline tool) ───────────────────────────────


def crypto_market_overview() -> dict[str, Any]:
    """Live crypto market snapshot — top coins, global cap + BTC dominance,
    Fear&Greed, BTC derivatives (funding + open interest), and the macro
    backdrop (DXY, S&P 500, Gold, VIX). Numbers are live from public APIs; the
    model must treat them as ground truth and never invent figures."""
    q: dict[str, str] = {}

    def _top() -> list[dict[str, Any]]:
        rows = _get_json(f"{_COINGECKO}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=12&page=1&price_change_percentage=24h")
        return [{"symbol": (r.get("symbol") or "").upper(), "price": r.get("current_price"),
                 "change_24h_pct": round(r.get("price_change_percentage_24h") or 0, 2),
                 "mcap": r.get("market_cap")} for r in (rows or [])[:12]]

    def _global() -> dict[str, Any]:
        g = (_get_json(f"{_COINGECKO}/global") or {}).get("data") or {}
        return {"total_mcap_usd": (g.get("total_market_cap") or {}).get("usd"),
                "mcap_change_24h_pct": round(g.get("market_cap_change_percentage_24h_usd") or 0, 2),
                "btc_dominance_pct": round((g.get("market_cap_percentage") or {}).get("btc") or 0, 2),
                "eth_dominance_pct": round((g.get("market_cap_percentage") or {}).get("eth") or 0, 2)}

    def _fng() -> dict[str, Any]:
        d = (_get_json(_FNG) or {}).get("data") or [{}]
        return {"value": int(d[0].get("value")), "classification": d[0].get("value_classification")}

    def _derivs() -> dict[str, Any]:
        prem = _get_json(f"{_BINANCE_FUT}/fapi/v1/premiumIndex?symbol=BTCUSDT")
        oi = _get_json(f"{_BINANCE_FUT}/fapi/v1/openInterest?symbol=BTCUSDT")
        funding = float(prem.get("lastFundingRate") or 0) * 100
        return {"btc_funding_pct": round(funding, 4),
                "btc_funding_state": "longs hot" if funding > 0.01 else "shorts hot" if funding < -0.01 else "neutral",
                "btc_open_interest": round(float(oi.get("openInterest") or 0), 1),
                "btc_mark_price": round(float(prem.get("markPrice") or 0), 2)}

    def _macro() -> dict[str, Any]:
        return {"dxy": _yahoo_last("DX-Y.NYB"), "sp500": _yahoo_last("%5EGSPC"),
                "gold": _yahoo_last("GC%3DF"), "vix": _yahoo_last("%5EVIX")}

    out = {
        "top_coins": _try("top_coins", _top, q),
        "global": _try("global", _global, q),
        "fear_greed": _try("fear_greed", _fng, q),
        "btc_derivatives": _try("btc_derivatives", _derivs, q),
        "macro": _try("macro", _macro, q),
        "data_quality": q,
    }
    return out


def coin_data(symbols: str) -> dict[str, Any]:
    """Detailed live data for specific coins (comma-separated symbols, e.g.
    "BTC,ETH,SOL"): price, 24h %, market cap and volume from CoinGecko. Call this
    after the overview for coins not in the top list."""
    syms = [s.strip().lower() for s in str(symbols or "").split(",") if s.strip()][:10]
    if not syms:
        raise ValueError("symbols is required, e.g. 'BTC,ETH'")
    ids = ",".join(syms)
    rows = _get_json(f"{_COINGECKO}/coins/markets?vs_currency=usd&symbols={urllib.parse.quote(ids)}&price_change_percentage=24h,7d")
    found = {(r.get("symbol") or "").lower(): r for r in (rows or [])}
    out = []
    for s in syms:
        r = found.get(s)
        if not r:
            out.append({"symbol": s.upper(), "error": "not found on CoinGecko"})
            continue
        out.append({"symbol": s.upper(), "price": r.get("current_price"),
                    "change_24h_pct": round(r.get("price_change_percentage_24h_in_currency") or 0, 2),
                    "change_7d_pct": round(r.get("price_change_percentage_7d_in_currency") or 0, 2),
                    "mcap": r.get("market_cap"), "volume_24h": r.get("total_volume"),
                    "rank": r.get("market_cap_rank")})
    return {"count": len(out), "coins": out}


# ── news / information-field analysis ────────────────────────────────────────

# The news classes the operator's analyst framework tracks. Each is a focused
# web query; results are returned grouped by class so the model can reason over
# the information field instead of a flat search dump.
_NEWS_CLASSES: dict[str, str] = {
    "etf_flows": "bitcoin ETF net flow inflow outflow today",
    "liquidations": "crypto liquidations 24h bitcoin long short",
    "geopolitics": "crypto geopolitical risk tariffs sanctions macro today",
    "regulation": "crypto regulation SEC policy enforcement this week",
}


def news_analysis(focus: Optional[str] = None, *, per_class: int = 4) -> dict[str, Any]:
    """Analyse the crypto news / information field, grouped into classes (ETF
    flows, liquidations, geopolitics, regulation) via the web-search backend (no
    external paid API). Pass ``focus`` to add a custom class. Returns titled,
    cited results per class for the model to weigh — not a flat search dump."""
    from . import external_clients as ext

    classes = dict(_NEWS_CLASSES)
    if focus and str(focus).strip():
        classes["focus"] = str(focus).strip()
    per_class = max(1, min(8, int(per_class)))
    by_class: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, query in classes.items():
        try:
            res = ext.web_search(query, limit=per_class)
            by_class[name] = [{"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet")}
                              for r in (res.get("results") or [])[:per_class]]
        except Exception as exc:  # noqa: BLE001 - one class failing shouldn't sink the rest
            by_class[name] = []
            errors[name] = f"{type(exc).__name__}"
    total = sum(len(v) for v in by_class.values())
    return {"classes": list(classes), "total_results": total, "by_class": by_class,
            **({"errors": errors} if errors else {})}


__all__ = ["crypto_market_overview", "coin_data", "news_analysis"]

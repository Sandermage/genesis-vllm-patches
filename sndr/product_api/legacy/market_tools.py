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
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Optional

# Fixed public hosts the tool may reach — no auth, free. A crafted symbol/arg can
# never redirect a request elsewhere (scheme + host + path are built here).
_COINGECKO = "https://api.coingecko.com/api/v3"
_FNG = "https://api.alternative.me/fng/?limit=1"
_BINANCE_FUT = "https://fapi.binance.com"
_BINANCE_SPOT = "https://api.binance.com"
# Batched spark endpoint: ONE request for all macro symbols. Four rapid per-symbol
# chart calls trip Yahoo's rate limit (429) from a datacenter IP; the batch does not.
_YAHOO_SPARK = "https://query1.finance.yahoo.com/v8/finance/spark"
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


def _macro_batch(symbols: dict[str, str]) -> dict[str, Any]:
    """Last close + 24h % for several Yahoo symbols in ONE batched spark request
    (``{label: yahoo_symbol}``). Today's bar can be unclosed (null) — we take the
    last two non-null closes."""
    enc = ",".join(urllib.parse.quote(sym, safe="") for sym in symbols.values())
    data = _get_json(f"{_YAHOO_SPARK}?symbols={enc}&range=5d&interval=1d", timeout=10.0)
    out: dict[str, Any] = {}
    for label, sym in symbols.items():
        node = (data or {}).get(sym) if isinstance(data, dict) else None
        closes = [c for c in ((node or {}).get("close") or []) if c is not None]
        if len(closes) >= 2:
            last, prev = closes[-1], closes[-2]
            out[label] = {"value": round(last, 2), "change_pct": round((last - prev) / prev * 100, 2) if prev else None}
        else:
            out[label] = None
    return out


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
        return {"value": int(d[0].get("value") or 0), "classification": d[0].get("value_classification")}

    def _derivs() -> dict[str, Any]:
        prem = _get_json(f"{_BINANCE_FUT}/fapi/v1/premiumIndex?symbol=BTCUSDT")
        oi = _get_json(f"{_BINANCE_FUT}/fapi/v1/openInterest?symbol=BTCUSDT")
        funding = float(prem.get("lastFundingRate") or 0) * 100
        return {"btc_funding_pct": round(funding, 4),
                "btc_funding_state": "longs hot" if funding > 0.01 else "shorts hot" if funding < -0.01 else "neutral",
                "btc_open_interest": round(float(oi.get("openInterest") or 0), 1),
                "btc_mark_price": round(float(prem.get("markPrice") or 0), 2)}

    def _macro() -> dict[str, Any]:
        return _macro_batch({"dxy": "DX-Y.NYB", "sp500": "^GSPC", "gold": "GC=F", "vix": "^VIX"})

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


# Major tickers we recognise in a free-text chat query, so the chat can ground a
# crypto question in live CoinGecko figures instead of the model's stale memory.
# Curated (majors only) to avoid false positives on common words; less-common
# coins are still served by the copilot's coin_data tool.
_MAJOR_SYMBOLS = {
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "MATIC", "TON", "TRX", "LTC", "BCH", "NEAR", "UNI", "APT", "ARB", "OP",
    "ATOM", "WLD", "SUI", "SEI", "INJ", "TIA", "RNDR", "FET", "PEPE", "SHIB",
    "FIL", "ICP", "HBAR", "STX", "IMX", "AAVE", "MKR", "LDO", "CRV", "GRT",
    "ALGO", "XLM", "VET", "ETC", "FTM", "JUP", "ENA", "ONDO", "KAS", "TAO",
}


def _extract_crypto_symbols(text: str) -> list[str]:
    """Major crypto tickers in a free-text query (case-insensitive, de-duplicated,
    order-preserving). Curated allow-list → no false positives on common words."""
    out: list[str] = []
    for word in re.findall(r"[A-Za-z]{2,5}", text or ""):
        sym = word.upper()
        if sym in _MAJOR_SYMBOLS and sym not in out:
            out.append(sym)
    return out[:6]


def market_grounding(query: str) -> str:
    """Real-time market-data grounding block for any major crypto tickers in the
    query (empty when none / on error). Lets the chat answer price/analysis
    questions with live CoinGecko figures instead of stale model memory."""
    syms = _extract_crypto_symbols(query)
    if not syms:
        return ""
    try:
        coins = coin_data(",".join(syms)).get("coins") or []
    except Exception:  # noqa: BLE001 - grounding is best-effort
        return ""
    def _money(v: Any) -> str:
        # CoinGecko returns null mcap/volume for thin/new coins; f"{None:,}"
        # raises TypeError and would drop the whole grounding block for that turn.
        return f"${v:,}" if isinstance(v, (int, float)) else "n/a"

    rows = []
    for c in coins:
        if c.get("error") or c.get("price") is None:
            continue
        rows.append(
            f"{c['symbol']}: ${c.get('price')} (24h {c.get('change_24h_pct')}%, "
            f"7d {c.get('change_7d_pct')}%, mcap {_money(c.get('mcap'))}, "
            f"vol {_money(c.get('volume_24h'))}, rank #{c.get('rank')})")
    if not rows:
        return ""
    return ("Live market data (CoinGecko, real-time — use these EXACT figures for "
            "prices / market caps; your training data is stale, do not quote prices "
            "from memory):\n" + "\n".join(rows))


# Signals that a question needs LIVE web data (vs the model's static training).
# Conservative + multilingual (EN + RU): only temporal cues ("today/now/latest"),
# news/events, an explicit search ask, or a recent year trigger it — timeless
# "explain X" / "how does Y work" questions deliberately do NOT, so simple chats
# are not slowed by a needless search. Crypto tickers are handled separately
# (market_grounding is always injected).
_WEB_SIGNAL_RE = re.compile(
    r"\b(?:today|now|current(?:ly)?|right\s+now|latest|recent(?:ly)?|nowadays|"
    r"as\s+of|up[-\s]?to[-\s]?date|this\s+(?:week|month|year)|"
    r"breaking|live|news|headlines?)\b"
    r"|\b(?:search|google|look\s?up|browse\s+online|find\s+online|web\s?search)\b"
    r"|\b20(?:2[4-9]|[3-9]\d)\b"
    r"|сегодня|сейчас|текущ|последн(?:ие|их|яя|ее|ий)|свеж|актуальн|"
    r"на\s+данный\s+момент|новост|загугли|найди\s+в\s+(?:интернете|сети|вебе)",
    re.IGNORECASE,
)


def needs_web_search(query: str) -> bool:
    """True when a question likely needs live web data (so the chat can auto-enable
    web search even with the toggle off). Heuristic, deterministic, offline."""
    return bool(_WEB_SIGNAL_RE.search(query or ""))


def date_grounding(now: Optional[datetime] = None) -> str:
    """Current-date grounding line for the chat. A language model has no clock, so
    without this it refuses "future"-dated questions or silently reasons from its
    stale training cutoff. Pass ``now`` for deterministic tests; defaults to UTC.

    The anti-refusal clause is deliberate: users routinely ask "analyse X today"
    or name an approximate date — the model should answer from the live data with
    an as-of note, not decline because a date looks like it is in the future."""
    now = now or datetime.now(timezone.utc)
    return (
        f"Current date: {now:%Y-%m-%d} ({now:%A}), {now:%H:%M} UTC. Treat this as the "
        "present moment. Any live data below is real-time as of now — answer the user's "
        "question using it. If the user names a date, give the current analysis from this "
        "live data and note the as-of date; never refuse to answer merely because a date "
        "is mentioned or appears to be in the future.")


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

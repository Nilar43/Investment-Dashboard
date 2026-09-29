"""Ticker resolution (ISIN -> Yahoo Finance) and EUR price history."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).with_name("tickers.json")


def load_map() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_map(m: dict) -> None:
    CACHE.write_text(json.dumps(m, indent=2, sort_keys=True), encoding="utf-8")


def search_ticker(key: str, name: str) -> str:
    if key.startswith("CRYPTO:"):
        return f"{key.split(':', 1)[1]}-EUR"
    query = key if not key.startswith("NAME:") else name
    try:
        quotes = yf.Search(query, max_results=10).quotes
    except Exception:
        quotes = []
    symbols = [q["symbol"] for q in quotes if q.get("symbol")]
    # prefer a primary listing over the Stuttgart ISIN-ticker fallback
    primary = [s for s in symbols if not s.startswith(key)]
    return (primary or symbols or [""])[0]


def resolve(items: dict[str, str], overrides: dict | None = None) -> dict[str, str]:
    """items: key -> name. Returns key -> ticker (cached on disk)."""
    m = load_map()
    changed = False
    for key, name in items.items():
        if overrides and overrides.get(key):
            if m.get(key) != overrides[key]:
                m[key] = overrides[key]
                changed = True
        elif key not in m or not m[key]:
            m[key] = search_ticker(key, name)
            changed = True
    if changed:
        save_map(m)
    return {k: m.get(k, "") for k in items}


def _currency(ticker: str) -> str:
    try:
        return yf.Ticker(ticker).fast_info["currency"] or "EUR"
    except Exception:
        return "EUR"


def history_eur(tickers: list[str], start: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """Daily close prices converted to EUR. Returns (prices df, currency per ticker)."""
    tickers = sorted({t for t in tickers if t})
    if not tickers:
        return pd.DataFrame(), {}
    start = (pd.Timestamp(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False,
                      group_by="column", threads=True)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()

    ccys = {t: _currency(t) for t in tickers}
    need_fx = {("GBP" if c in ("GBp", "GBX") else c.upper()) for c in ccys.values()} - {"EUR"}
    fx = {}
    if need_fx:
        pairs = [f"{c}EUR=X" for c in need_fx]
        fxraw = yf.download(pairs, start=start, auto_adjust=False, progress=False, group_by="column")
        fxc = fxraw["Close"] if isinstance(fxraw.columns, pd.MultiIndex) else fxraw[["Close"]].rename(columns={"Close": pairs[0]})
        fxc.index = pd.to_datetime(fxc.index).tz_localize(None).normalize()
        fx = {c: fxc[f"{c}EUR=X"] for c in need_fx}

    idx = pd.date_range(close.index.min(), pd.Timestamp.today().normalize(), freq="D")
    out = pd.DataFrame(index=idx)
    for t in tickers:
        if t not in close.columns:
            continue
        s = close[t].reindex(idx).ffill()
        c = ccys[t]
        if c in ("GBp", "GBX"):
            s = s / 100 * fx["GBP"].reindex(idx).ffill().bfill()
        elif c.upper() != "EUR":
            s = s * fx[c.upper()].reindex(idx).ffill().bfill()
        out[t] = s
    return out, ccys

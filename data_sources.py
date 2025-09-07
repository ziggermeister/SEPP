#!/usr/bin/env python3
# data_sources.py
# Unified price+dividend fetchers with multi-source fallback/consensus.
# Outputs a MultiIndex DataFrame with columns:
#   (Ticker, 'Adj Close'), (Ticker, 'Dividends')

from __future__ import annotations
import os, io, time, datetime as dt
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ========= Utility helpers =========

def _to_date(s: str | dt.date | None) -> str:
    if s is None:
        return dt.date.today().isoformat()
    if isinstance(s, dt.date):
        return s.isoformat()
    return str(s)

def _clean_series(s: pd.Series, dtype=float) -> pd.Series:
    if s is None or not isinstance(s, pd.Series):
        return pd.Series(dtype=dtype)
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(dtype)

def _multi_from_maps(prices: Dict[str, pd.Series],
                     divs: Dict[str, pd.Series]) -> pd.DataFrame:
    frames = []
    for sym in sorted(prices.keys()):
        px = _clean_series(prices.get(sym))
        if px.empty:
            # ensure column pair exists (all-NaN)
            px = pd.Series(np.nan, index=pd.DatetimeIndex([], name="Date"))
        dv = _clean_series(divs.get(sym), dtype=float).reindex(px.index).fillna(0.0)
        frames.append(pd.concat({
            (sym, "Adj Close"): px.astype(float),
            (sym, "Dividends"): dv.astype(float),
        }, axis=1))
    if frames:
        out = pd.concat(frames, axis=1)
    else:
        out = _empty_multi_df()
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["Ticker", "Field"])
    return out

def _empty_multi_df() -> pd.DataFrame:
    cols = pd.MultiIndex.from_tuples([], names=["Ticker", "Field"])
    return pd.DataFrame(columns=cols, dtype=float)

def _coalesce_series(idx: pd.DatetimeIndex,
                     *candidates: Optional[pd.Series],
                     fill: float | None = 0.0) -> pd.Series:
    for s in candidates:
        s = _clean_series(s)
        if not s.empty:
            return s.reindex(idx)
    if fill is None:
        return pd.Series(np.nan, index=idx, dtype=float)
    return pd.Series(fill, index=idx, dtype=float)

def _union_index(*maps: Dict[str, pd.Series]) -> Optional[pd.DatetimeIndex]:
    idx: Optional[pd.DatetimeIndex] = None
    for mp in maps:
        if not mp:
            continue
        for s in mp.values():
            s = _clean_series(s)
            if not s.empty:
                idx = s.index if idx is None else idx.union(s.index)
    return None if idx is None else idx.sort_values()


# ========= Source 1: Yahoo (yfinance) =========

def fetch_yahoo(symbols: List[str], start, end) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    start, end = _to_date(start), _to_date(end)
    # actions=True pulls dividends into the frame
    raw = yf.download(
        tickers=" ".join(symbols),
        start=start, end=end,
        auto_adjust=False, group_by="ticker",
        threads=True, progress=False, actions=True,
    )

    px: Dict[str, pd.Series] = {}
    dv: Dict[str, pd.Series] = {}
    is_multi = isinstance(raw.columns, pd.MultiIndex)

    for sym in symbols:
        if is_multi:
            if (sym, "Adj Close") not in raw.columns:
                # symbol may be unknown at Yahoo
                continue
            adj = raw[(sym, "Adj Close")]
            div = raw[(sym, "Dividends")] if (sym, "Dividends") in raw.columns else pd.Series(0.0, index=adj.index)
        else:
            # degenerate single-ticker layout (rare)
            if "Adj Close" not in raw.columns:
                continue
            adj = raw["Adj Close"]
            div = raw["Dividends"] if "Dividends" in raw.columns else pd.Series(0.0, index=adj.index)
        px[sym] = _clean_series(adj)
        dv[sym] = _clean_series(div).fillna(0.0)

    return px, dv


# ========= Source 2: Stooq (free CSV) =========
# US tickers typically require ".us" suffix at Stooq. If a ticker already
# contains a dot (e.g., "EURUSD"), we leave it as-is.

def _stooq_symbol(sym: str) -> str:
    return sym.lower() if "." in sym else f"{sym.lower()}.us"

def _stooq_fetch_one(sym: str) -> pd.Series:
    url = f"https://stooq.com/q/d/l/?s={_stooq_symbol(sym)}&i=d"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # expected columns: Date, Open, High, Low, Close, Volume
    if "Date" not in df.columns or "Close" not in df.columns:
        raise RuntimeError(f"Stooq payload unexpected for {sym}: {df.head(3)}")
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.set_index("Date").sort_index()
    # Stooq "Close" is not adjusted; acceptable for fallback/fill/consensus
    return df["Close"].astype(float)

def fetch_stooq(symbols: List[str], start, end) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    start = pd.to_datetime(_to_date(start))
    end   = pd.to_datetime(_to_date(end))
    px: Dict[str, pd.Series] = {}
    dv: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            s = _stooq_fetch_one(sym)
            s = s.loc[(s.index >= start) & (s.index <= end)]
            s = _clean_series(s)
            if s.empty:
                continue
            px[sym] = s
            dv[sym] = pd.Series(0.0, index=s.index)  # Stooq has no dividends here
        except Exception:
            # Missing symbol or transient error; leave empty and let merge handle it
            pass
    return px, dv


# ========= Source 3: Alpha Vantage (best-effort, free JSON) =========
# Free tier + JSON TIME_SERIES_DAILY_ADJUSTED is rate-limited.
# We:
#   - skip entirely if no key,
#   - return empty if rate-limited / bad payload,
#   - sleep lightly between calls to be kind to the API.

def _alpha_series_daily_adjusted_json(sym: str, api_key: str) -> Tuple[pd.Series, pd.Series]:
    """Returns (Adj Close, Dividends) series or raises on hard failure."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": sym,
        "outputsize": "full",
        "datatype": "json",
        "apikey": api_key,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()

    # Handle throttling / errors gracefully by raising (caller will catch)
    if not isinstance(js, dict) or "Time Series (Daily)" not in js:
        # common keys on throttling: "Note", "Information", "Error Message"
        raise RuntimeError(f"Alpha JSON missing 'Time Series (Daily)' for {sym}: keys={list(js.keys())[:3]}")

    ts = js["Time Series (Daily)"]
    if not isinstance(ts, dict) or not ts:
        raise RuntimeError(f"Alpha JSON empty time series for {sym}")

    # Build series
    idx = pd.to_datetime(list(ts.keys())).tz_localize(None)
    df = pd.DataFrame(ts).T
    df.index = idx
    # expected fields: '5. adjusted close', '7. dividend amount'
    adj = df.get("5. adjusted close")
    dvd = df.get("7. dividend amount")
    if adj is None:
        raise RuntimeError(f"Alpha JSON missing adjusted close for {sym}")
    adj = pd.to_numeric(adj, errors="coerce")
    dvd = pd.to_numeric(dvd, errors="coerce") if dvd is not None else pd.Series(0.0, index=adj.index)

    adj = _clean_series(adj)
    dvd = _clean_series(dvd).reindex(adj.index).fillna(0.0)
    return adj, dvd

def fetch_alpha(symbols: List[str], start, end, api_key: Optional[str]) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    if not api_key:
        return {}, {}  # silently skip
    start = pd.to_datetime(_to_date(start))
    end   = pd.to_datetime(_to_date(end))
    px: Dict[str, pd.Series] = {}
    dv: Dict[str, pd.Series] = {}
    for i, sym in enumerate(symbols):
        try:
            adj, dvd = _alpha_series_daily_adjusted_json(sym, api_key)
            # clip to window
            adj = adj.loc[(adj.index >= start) & (adj.index <= end)]
            dvd = dvd.reindex(adj.index).fillna(0.0)
            if adj.empty:
                continue
            px[sym] = adj
            dv[sym] = dvd
        except Exception:
            # throttle / symbol not covered / misc error → skip
            pass
        # gentle pause to reduce throttle
        time.sleep(1.1)
    return px, dv


# ========= Merge strategies =========

def _prefer_yahoo_fill_gaps(
    symbols: List[str],
    y_px: Dict[str, pd.Series], y_dv: Dict[str, pd.Series],
    a_px: Dict[str, pd.Series], a_dv: Dict[str, pd.Series],
    s_px: Dict[str, pd.Series], s_dv: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    For each symbol:
      - Start with Yahoo price.
      - Fill NaNs / missing days from Alpha, then Stooq.
      - Dividends: Yahoo → Alpha → Stooq(zeros) → 0.0
    """
    all_idx = _union_index(y_px, a_px, s_px)
    if all_idx is None:
        return _empty_multi_df()

    merged_px: Dict[str, pd.Series] = {}
    merged_dv: Dict[str, pd.Series] = {}

    for sym in symbols:
        # Price
        base = _clean_series(y_px.get(sym)).reindex(all_idx)
        if base.empty:
            base = pd.Series(np.nan, index=all_idx, dtype=float)
        a = _clean_series(a_px.get(sym)).reindex(all_idx)
        s = _clean_series(s_px.get(sym)).reindex(all_idx)

        px = base.copy()
        if not a.empty:
            px = px.where(~px.isna(), a)
        if not s.empty:
            px = px.where(~px.isna(), s)
        px = px.ffill()

        # Dividends
        dv = _coalesce_series(all_idx, y_dv.get(sym), a_dv.get(sym), s_dv.get(sym), fill=0.0).fillna(0.0)

        merged_px[sym] = px.astype(float)
        merged_dv[sym] = dv.astype(float)

    return _multi_from_maps(merged_px, merged_dv)

def _median_of_available(
    symbols: List[str],
    y_px: Dict[str, pd.Series], a_px: Dict[str, pd.Series], s_px: Dict[str, pd.Series],
    y_dv: Dict[str, pd.Series], a_dv: Dict[str, pd.Series], s_dv: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    For each symbol:
      - Price: per-day median across available sources (Yahoo/Alpha/Stooq), ffilled.
      - Dividends: Yahoo → Alpha → Stooq(zeros) → 0.0
    """
    all_idx = _union_index(y_px, a_px, s_px)
    if all_idx is None:
        return _empty_multi_df()

    merged_px: Dict[str, pd.Series] = {}
    merged_dv: Dict[str, pd.Series] = {}

    for sym in symbols:
        cands = []
        for mp in (y_px, a_px, s_px):
            s = _clean_series(mp.get(sym))
            if not s.empty:
                cands.append(s.reindex(all_idx))
        if cands:
            mat = pd.concat(cands, axis=1)
            px = mat.median(axis=1, skipna=True).astype(float).ffill()
        else:
            px = pd.Series(np.nan, index=all_idx, dtype=float)

        dv = _coalesce_series(all_idx, y_dv.get(sym), a_dv.get(sym), s_dv.get(sym), fill=0.0).fillna(0.0)

        merged_px[sym] = px
        merged_dv[sym] = dv

    return _multi_from_maps(merged_px, merged_dv)


# ========= Public entry point =========

def fetch_prices_multi(
    symbols: List[str],
    start, end=None,
    sources: List[str] = ("yahoo", "stooq"),     # default free + robust
    consensus: str = "prefer-yahoo-fill",        # or "median"
    alpha_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch prices/dividends from one or more sources and merge by policy.

    Returns:
      DataFrame with MultiIndex columns: (Ticker, 'Adj Close'), (Ticker, 'Dividends').
      Index is a DatetimeIndex (sorted).
    """
    start, end = _to_date(start), _to_date(end)
    use_y = "yahoo" in sources
    use_s = "stooq" in sources
    use_a = "alpha" in sources

    y_px = y_dv = a_px = a_dv = s_px = s_dv = {}

    if use_y:
        y_px, y_dv = fetch_yahoo(symbols, start, end)
    if use_s:
        s_px, s_dv = fetch_stooq(symbols, start, end)
    if use_a:
        key = alpha_key or os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_KEY")
        a_px, a_dv = fetch_alpha(symbols, start, end, key)

    if consensus == "median":
        return _median_of_available(symbols, y_px, a_px, s_px, y_dv, a_dv, s_dv)
    else:
        return _prefer_yahoo_fill_gaps(symbols, y_px, y_dv, a_px, a_dv, s_px, s_dv)


# ========= Convenience wrappers (optional) =========

def fetch_prices_yahoo(symbols, start, end=None):
    return fetch_prices_multi(symbols, start, end, sources=["yahoo"], consensus="prefer-yahoo-fill")

def fetch_prices_stooq_only(symbols, start, end=None):
    return fetch_prices_multi(symbols, start, end, sources=["stooq"], consensus="median")

def fetch_prices_alpha_only(symbols, start, end=None, alpha_key=None):
    return fetch_prices_multi(symbols, start, end, sources=["alpha"], consensus="median", alpha_key=alpha_key)
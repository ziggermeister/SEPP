#!/usr/bin/env python3
"""
Build a frozen parameter pack from live data (yfinance), so the engine can be
validated offline & reproducibly.

Usage:
  python tests/make_param_pack.py \
    --symbols SGOV VGIT BND VWOB SCHD CDC VIG GLD VTI IEFA VWO QQQ CHAT IBIT DGIN \
    --start 2016-01-01 --end 2024-12-31 \
    --out tests/param_packs/pack_2024-12-31.json
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def _annualize_daily(logret_daily: pd.Series) -> float:
    return float(logret_daily.mean() * 252.0)


def _annualize_daily_vol(ret_daily: pd.Series) -> float:
    return float(ret_daily.std(ddof=0) * math.sqrt(252.0))


def fetch_prices(symbols, start, end):
    raw = yf.download(
        " ".join(symbols),
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
        actions=True,
    )
    frames = []
    for sym in symbols:
        if isinstance(raw.columns, pd.MultiIndex):
            adj = raw[(sym, "Adj Close")].rename((sym, "Adj Close"))
            div = (
                raw[(sym, "Dividends")].rename((sym, "Dividends"))
                if (sym, "Dividends") in raw.columns
                else None
            )
        else:
            adj = raw["Adj Close"].rename((sym, "Adj Close"))
            div = (
                raw["Dividends"].rename((sym, "Dividends")) if "Dividends" in raw.columns else None
            )
        if div is None or (isinstance(div, pd.Series) and div.dropna().sum() == 0.0):
            try:
                td = yf.Ticker(sym).dividends
                if td is None or td.empty:
                    div = pd.Series(0.0, index=adj.index)
                else:
                    div = td.resample("D").sum().reindex(adj.index, fill_value=0.0)
                div = div.rename((sym, "Dividends"))
            except Exception:
                div = pd.Series(0.0, index=adj.index, name=(sym, "Dividends"))
        frames.append(pd.concat([adj, div], axis=1))
    prices = pd.concat(frames, axis=1)
    prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["Ticker", "Field"])
    prices = prices.dropna(how="all")
    for sym in symbols:
        prices[(sym, "Dividends")] = prices[(sym, "Dividends")].fillna(0.0)
    return prices


def compute_params(prices: pd.DataFrame, symbols: list[str]):
    adj = prices.xs("Adj Close", axis=1, level="Field").ffill().dropna()
    adj = adj[[s for s in symbols if s in adj.columns]]
    ret = adj.pct_change().dropna(how="any")
    logret = np.log(adj).diff().dropna(how="any")
    mu = {s: _annualize_daily(logret[s]) for s in symbols}
    sig = {s: _annualize_daily_vol(ret[s]) for s in symbols}
    rho_df: pd.DataFrame = ret.corr().reindex(index=symbols, columns=symbols)
    div_df: pd.DataFrame = prices.xs("Dividends", axis=1, level="Field")
    div_df = div_df.reindex_like(adj).fillna(0.0)
    div = div_df
    div_12m = div.rolling(252, min_periods=20).sum()
    last_price = adj.iloc[-1]
    last_div = div_12m.iloc[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        yld = (last_div / last_price).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 0.12)

    return mu, sig, rho_df, yld


def stable_hash(obj) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    end = args.end or dt.date.today().isoformat()
    prices = fetch_prices(args.symbols, args.start, end)
    mu, sig, rho_df, yld = compute_params(prices, args.symbols)

    pack = {
        "version": "v6.4.3",
        "as_of": end,
        "universe": args.symbols,
        "mu": mu,
        "sigma": sig,
        "yield_rate": yld.to_dict(),
        "rho": rho_df.to_dict(),
        "sources": {"prices": "yfinance"},
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    pack["stable_hash"] = stable_hash(
        {
            "universe": args.symbols,
            "as_of": end,
            "mu": mu,
            "sigma": sig,
            "rho": rho_df.to_dict(),
            "yield_rate": yld.to_dict(),
        }
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pack, indent=2))
    print(f"Wrote param pack: {args.out}")
    print(f"  universe: {', '.join(args.symbols)}")
    print(f"  as_of:    {end}")
    print(f"  hash:     {pack['stable_hash']}")


if __name__ == "__main__":
    main()

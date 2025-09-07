#!/usr/bin/env python3
# run_portfolios.py
# Read one CSV with multiple portfolios (tickers + quantity or value),
# build weights from value, fetch live inputs via Yahoo, and score each portfolio.

import argparse, math, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

import sepp_engine as eng  # your v6_4_3 engine (fast bootstrap, etc.)


# ---------- utilities reused from wire_live_to_engine ----------
def _annualize_daily(logret_daily: pd.Series) -> float:
    return float(logret_daily.mean() * 252.0)

def _annualize_daily_vol(ret_daily: pd.Series) -> float:
    return float(ret_daily.std(ddof=0) * math.sqrt(252.0))

def _safe_indices(symbols):
    safe_candidates = {"SGOV", "VGIT", "BND", "VWOB", "SHY", "IEF", "AGG"}
    idx = [i for i, s in enumerate(symbols) if s in safe_candidates]
    return idx if idx else [0]

def fetch_prices(symbols, start, end=None):
    end = end or dt.date.today().isoformat()
    raw = yf.download(
        tickers=" ".join(symbols),
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
            if (sym, "Adj Close") not in raw.columns:
                raise ValueError(f"Missing Adj Close for {sym}")
            adj = raw[(sym, "Adj Close")].rename((sym, "Adj Close"))
            div = raw[(sym, "Dividends")].rename((sym, "Dividends")) if (sym, "Dividends") in raw.columns else None
        else:
            if "Adj Close" not in raw.columns:
                raise ValueError(f"Missing Adj Close for {sym}")
            adj = raw["Adj Close"].rename((sym, "Adj Close"))
            div = raw["Dividends"].rename((sym, "Dividends")) if "Dividends" in raw.columns else None

        # fallback for dividends
        need_fallback = (div is None) or (isinstance(div, pd.Series) and (div.dropna().sum() == 0.0))
        if need_fallback:
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

def compute_inputs(prices: pd.DataFrame, symbols: list[str]):
    adj = prices.xs("Adj Close", axis=1, level="Field").ffill().dropna()
    adj = adj[[s for s in symbols if s in adj.columns]]

    ret = adj.pct_change().dropna(how="any")
    logret = np.log(adj).diff().dropna(how="any")

    mu = np.array([_annualize_daily(logret[s]) for s in adj.columns], dtype=float)
    sig = np.array([_annualize_daily_vol(ret[s]) for s in adj.columns], dtype=float)
    rho = ret.corr().to_numpy(dtype=float)

    div = prices.xs("Dividends", axis=1, level="Field").reindex_like(adj).fillna(0.0)
    div_12m = div.rolling(252, min_periods=20).sum()
    last_price = adj.iloc[-1]
    last_div = div_12m.iloc[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        yld = (last_div / last_price).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yld = np.clip(yld.to_numpy(dtype=float), 0.0, 0.12)
    return mu, sig, rho, yld


# ---------- CSV ingestion + weight building ----------
def load_portfolios(csv_path: str):
    """
    Expect columns: Portfolio, Ticker, Quantity, Value
    - If Value is NaN, compute Value = Quantity * latest price (we’ll do this later).
    - Returns: dict[name] -> dict[ticker] -> {'qty': q, 'val': v or None}
    """
    df = pd.read_csv(csv_path)
    req = {"Portfolio", "Ticker"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {req}")

    if "Quantity" not in df.columns and "Value" not in df.columns:
        raise ValueError("CSV must include at least one of: Quantity or Value")

    portfolios = {}
    for _, row in df.iterrows():
        name = str(row["Portfolio"])
        t = str(row["Ticker"]).strip().upper()
        q = float(row["Quantity"]) if "Quantity" in df.columns and not pd.isna(row["Quantity"]) else None
        v = float(row["Value"]) if "Value" in df.columns and not pd.isna(row["Value"]) else None
        portfolios.setdefault(name, {})
        portfolios[name][t] = {"qty": q, "val": v}
    return portfolios

def union_symbols(portfolios_dict):
    syms = set()
    for pmap in portfolios_dict.values():
        syms.update(pmap.keys())
    return sorted(syms)


def build_weights_for_portfolio(name, pos_map, symbols, last_prices):
    """
    Convert positions (qty/val) into a weight vector aligned with 'symbols'.
    last_prices: pd.Series of latest Adj Close indexed by ticker.
    """
    values = []
    for s in symbols:
        entry = pos_map.get(s, {"qty": None, "val": None})
        v = entry["val"]
        if v is None:
            q = entry["qty"] or 0.0
            px = float(last_prices.get(s, np.nan))
            v = q * (px if np.isfinite(px) else 0.0)
        values.append(v or 0.0)

    values = np.array(values, dtype=float)
    total = values.sum()
    if total <= 0:
        raise ValueError(f"Portfolio {name}: total $value is zero; cannot form weights.")
    w = values / total
    return w


# ---------- CLI + main ----------
def parse_args():
    p = argparse.ArgumentParser(description="Score multiple real portfolios (CSV) with live inputs.")
    p.add_argument("--csv", required=True, help="CSV with columns: Portfolio,Ticker,Quantity,Value")
    p.add_argument("--start", required=True, help="Lookback start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default today)")
    p.add_argument("--params", default="data/params.json", help="sepp engine param json (optional)")
    return p.parse_args()

def apply_params(path):
    if not path or not Path(path).exists():
        return
    import json
    cfg = json.loads(Path(path).read_text())
    eng.YEARS = int(cfg.get("years", eng.YEARS))
    eng.ANNUAL_WITHDRAWAL = float(cfg.get("annual_withdrawal", eng.ANNUAL_WITHDRAWAL))
    eng.INITIAL_PORTFOLIO_VALUE = float(cfg.get("initial_portfolio_value", eng.INITIAL_PORTFOLIO_VALUE))
    eng.N_SIM = int(cfg.get("n_sim", eng.N_SIM))
    eng.SEED = int(cfg.get("seed", eng.SEED))
    eng.BOOTSTRAP_RESAMPLES = int(cfg.get("bootstrap_resamples", eng.BOOTSTRAP_RESAMPLES))
    eng.STRESS_BLEND = cfg.get("stress_blend", eng.STRESS_BLEND)
    eng.LIQ_METHOD = cfg.get("liq_method", eng.LIQ_METHOD)
    eng.LIQ_CAP = int(cfg.get("liq_cap", eng.LIQ_CAP))
    eng.DEBUG_PRINT_ENV = bool(cfg.get("debug_print_env", eng.DEBUG_PRINT_ENV))
    eng.DEBUG_LIQ_STATS = bool(cfg.get("debug_liq_stats", eng.DEBUG_LIQ_STATS))
    eng.DEBUG_SAMPLE_PATHS = bool(cfg.get("debug_sample_paths", eng.DEBUG_SAMPLE_PATHS))

def main():
    args = parse_args()
    apply_params(args.params)

    # 1) load portfolios
    portfolios = load_portfolios(args.csv)
    symbols = union_symbols(portfolios)

    # 2) fetch prices and compute market inputs on the UNION universe
    prices = fetch_prices(symbols, args.start, args.end)
    mu, sig, rho, yld = compute_inputs(prices, symbols)

    # 3) set engine globals for this run
    eng.ASSETS = symbols[:]  # ordered
    eng.MU = mu
    eng.SIG = sig
    eng.RHO = rho
    eng.YIELD_RATE = yld
    eng.SAFE_IDX = np.array(_safe_indices(symbols), dtype=int)
    eng.GROWTH_IDX = np.array([i for i in range(len(symbols)) if i not in eng.SAFE_IDX], dtype=int)
    eng.safe_asset_tickers = [symbols[i] for i in eng.SAFE_IDX]
    eng.growth_asset_tickers = [symbols[i] for i in eng.GROWTH_IDX]

    last_adj = prices.xs("Adj Close", axis=1, level="Field").ffill().iloc[-1]
    print("=== LIVE INPUTS (union universe) ===")
    print("Symbols:", symbols)
    print("mu  :", np.round(mu, 4))
    print("sig :", np.round(sig, 4))
    print("yld :", np.round(yld, 4))
    print("SAFE_IDX:", list(eng.SAFE_IDX), "->", [symbols[i] for i in eng.SAFE_IDX])

    # 4) score each portfolio
    for name, pos_map in portfolios.items():
        w = build_weights_for_portfolio(name, pos_map, symbols, last_adj)
        eng.score_portfolio(  # uses your existing engine printer
            name, w, symbols, mu, sig, rho, yld, eng.SAFE_IDX, eng.GROWTH_IDX
        )

if __name__ == "__main__":
    main()
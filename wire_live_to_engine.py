#!/usr/bin/env python3
# wire_live_to_engine.py
# Glue: fetch live data, compute inputs, patch sepp_engine globals, run engine.

import argparse, json, math, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# import your v6_4_3 engine renamed to sepp_engine.py
import sepp_engine as eng


# ---------- utility ----------
def _annualize_daily(logret_daily: pd.Series) -> float:
    """Annualized geometric mean from daily log-returns."""
    return float(logret_daily.mean() * 252.0)

def _annualize_daily_vol(ret_daily: pd.Series) -> float:
    """Annualized stdev from daily *arithmetic* returns."""
    return float(ret_daily.std(ddof=0) * math.sqrt(252.0))

def _safe_indices(symbols):
    """Choose safe sleeve by known tickers; fallback to first if none."""
    safe_candidates = {"SGOV", "VGIT", "BND", "VWOB", "SHY", "IEF", "AGG"}
    idx = [i for i, s in enumerate(symbols) if s in safe_candidates]
    return idx if idx else [0]


# ---------- robust Yahoo fetch (with per-ticker dividend fallback) ----------
def compute_prices(symbols, start, end=None):
    """
    Download daily Adj Close and Dividends for symbols.
    Uses bulk download for prices; if dividends absent, fetch per-ticker.
    Returns a DataFrame with MultiIndex columns (Ticker, Field).
    """
    end = end or dt.datetime.utcnow().date().isoformat()

    raw = yf.download(
        tickers=" ".join(symbols),
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
        actions=True,  # request corp actions (dividends/splits)
    )

    frames = []
    for sym in symbols:
        # --- Adj Close ---
        if isinstance(raw.columns, pd.MultiIndex):
            if (sym, "Adj Close") not in raw.columns:
                raise ValueError(f"Missing Adj Close for {sym}")
            adj = raw[(sym, "Adj Close")].rename((sym, "Adj Close"))
            div = raw[(sym, "Dividends")].rename((sym, "Dividends")) if (sym, "Dividends") in raw.columns else None
        else:
            # single ticker layout
            if "Adj Close" not in raw.columns:
                raise ValueError(f"Missing Adj Close for {sym}")
            adj = raw["Adj Close"].rename((sym, "Adj Close"))
            div = raw["Dividends"].rename((sym, "Dividends")) if "Dividends" in raw.columns else None

        # --- Per-ticker dividend fallback if missing/empty ---
        need_fallback = (div is None) or (isinstance(div, pd.Series) and (div.dropna().sum() == 0.0))
        if need_fallback:
            try:
                td = yf.Ticker(sym).dividends  # Series indexed by date
                if td is None or td.empty:
                    div = pd.Series(0.0, index=adj.index)
                else:
                    td_daily = td.resample("D").sum().reindex(adj.index, fill_value=0.0)
                    div = td_daily
                div = div.rename((sym, "Dividends"))
            except Exception:
                div = pd.Series(0.0, index=adj.index, name=(sym, "Dividends"))

        frames.append(pd.concat([adj, div], axis=1))

    prices = pd.concat(frames, axis=1)
    prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["Ticker", "Field"])
    prices = prices.dropna(how="all")

    # Ensure dividends zeros not NaN
    for sym in symbols:
        prices[(sym, "Dividends")] = prices[(sym, "Dividends")].fillna(0.0)

    return prices


def compute_inputs(prices: pd.DataFrame, symbols: list[str]):
    """
    From prices (Adj Close, Dividends), compute annualized mu/sig/rho and a simple yield proxy.
    """
    # Keep common dates & order columns by requested symbols
    adj = prices.xs("Adj Close", axis=1, level="Field").ffill().dropna()
    adj = adj[[s for s in symbols if s in adj.columns]]

    # Daily arithmetic & log returns
    ret = adj.pct_change().dropna(how="any")
    logret = np.log(adj).diff().dropna(how="any")

    # Annualized moments
    mu = np.array([_annualize_daily(logret[s]) for s in adj.columns], dtype=float)
    sig = np.array([_annualize_daily_vol(ret[s]) for s in adj.columns], dtype=float)
    rho = ret.corr().to_numpy(dtype=float)

    # Yield proxy: trailing 12m dividends / last price; clamp to [0, 12%]
    div = prices.xs("Dividends", axis=1, level="Field").reindex_like(adj).fillna(0.0)
    div_12m = div.rolling(252, min_periods=20).sum()
    last_price = adj.iloc[-1]
    last_div = div_12m.iloc[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        yld = (last_div / last_price).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yld = np.clip(yld.to_numpy(dtype=float), 0.0, 0.12)

    return mu, sig, rho, yld


# ---------- args / run ----------
def parse_args():
    p = argparse.ArgumentParser(description="Fetch live data, compute inputs, and run SEPP engine.")
    p.add_argument("--symbols", nargs="+", required=True, help="Tickers in portfolio universe, order matters.")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today UTC)")
    p.add_argument("--portfolio_json", required=False,
                   help='JSON string: {"name":"Live","weights":{"SGOV":0.3,"VTI":0.7}}')
    p.add_argument("--params", type=str, required=False, help="Path to params.json (optional)")
    return p.parse_args()


def apply_params(params_path: str | None):
    """Patch engine configuration from a JSON file if provided."""
    if not params_path:
        return
    cfg = json.loads(Path(params_path).read_text())
    # Core engine knobs (present names in your v6_4_3 file)
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


def run():
    args = parse_args()
    apply_params(args.params)

    symbols = args.symbols
    prices = compute_prices(symbols, args.start, args.end)
    mu, sig, rho, yld = compute_inputs(prices, symbols)

    # Patch engine market inputs & universe
    eng.ASSETS = symbols[:]                              # order matters downstream
    eng.MU = mu
    eng.SIG = sig
    eng.RHO = rho
    eng.YIELD_RATE = yld

    # Safe/growth split
    eng.SAFE_IDX = np.array(_safe_indices(symbols), dtype=int)
    eng.GROWTH_IDX = np.array([i for i in range(len(symbols)) if i not in eng.SAFE_IDX], dtype=int)

    # If engine references these ticker lists, keep them synced:
    eng.safe_asset_tickers = [symbols[i] for i in eng.SAFE_IDX]
    eng.growth_asset_tickers = [symbols[i] for i in eng.GROWTH_IDX]

    # Build weights vector from portfolio_json (default: equal weight)
    if args.portfolio_json:
        pj = json.loads(args.portfolio_json)
        w_map = pj.get("weights", {})
        name = pj.get("name", "Live")
    else:
        w_map = {s: 1.0 / len(symbols) for s in symbols}
        name = "Live"

    w = np.array([float(w_map.get(s, 0.0)) for s in symbols], dtype=float)
    s = w.sum()
    if s <= 0:
        raise ValueError("Portfolio weights sum to 0; nothing to do.")
    w = w / s

    # Pretty header
    asof = dt.datetime.now(dt.timezone.utc).date()
    print(f"=== LIVE INPUTS ===")
    print(f"Symbols: {symbols}")
    np.set_printoptions(precision=4, suppress=True)
    print("mu  :", np.round(mu, 4))
    print("sig :", np.round(sig, 4))
    print("yld :", np.round(yld, 4))
    print(f"SAFE_IDX: {list(eng.SAFE_IDX)} -> {[symbols[i] for i in eng.SAFE_IDX]}")

    # Run the engine's scorer for this single portfolio
    eng.score_portfolio(
        name,
        w,
        symbols,          # assets (list of tickers, order matters)
        mu,
        sig,
        rho,
        yld,
        eng.SAFE_IDX,     # safe_idx
        eng.GROWTH_IDX    # growth_idx
    )

if __name__ == "__main__":
    run()


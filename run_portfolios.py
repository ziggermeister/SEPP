#!/usr/bin/env python3
# run_portfolios.py
# Read one CSV with multiple portfolios (tickers + quantity or value),
# build weights from value, fetch live inputs (multi-source), and score each portfolio.

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

import sepp_engine as eng  # your v6_4_3 engine (fast bootstrap, etc.)
from data_sources import fetch_prices_multi  # multi-source price/div loader


# ---------- utilities reused from wire_live_to_engine ----------
def _annualize_daily(logret_daily: pd.Series) -> float:
    return float(logret_daily.mean() * 252.0)


def _annualize_daily_vol(ret_daily: pd.Series) -> float:
    return float(ret_daily.std(ddof=0) * math.sqrt(252.0))


def _safe_indices(symbols: List[str]) -> List[int]:
    safe_candidates = {"SGOV", "VGIT", "BND", "VWOB", "SHY", "IEF", "AGG"}
    idx = [i for i, s in enumerate(symbols) if s in safe_candidates]
    return idx if idx else [0]  # always have at least one safe to avoid edge cases


def fetch_prices_yahoo(symbols: List[str], start: str, end: str | None = None) -> pd.DataFrame:
    """
    Legacy Yahoo-only fetcher (kept for reference/fallback). Multi-source path is now default via fetch_prices_multi.
    """
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
            div = (
                raw[(sym, "Dividends")].rename((sym, "Dividends"))
                if (sym, "Dividends") in raw.columns
                else None
            )
        else:
            if "Adj Close" not in raw.columns:
                raise ValueError(f"Missing Adj Close for {sym}")
            adj = raw["Adj Close"].rename((sym, "Adj Close"))
            div = (
                raw["Dividends"].rename((sym, "Dividends")) if "Dividends" in raw.columns else None
            )

        # fallback for dividends
        need_fallback = (div is None) or (
            isinstance(div, pd.Series) and (div.dropna().sum() == 0.0)
        )
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


def compute_inputs(prices: pd.DataFrame, symbols: List[str]):
    """
    Given a prices dataframe with MultiIndex columns (Ticker, Field) and daily index,
    compute mu (annualized log return), sig (annualized vol), rho (corr), and forward-looking yld (12m trailing dividends/price).
    """
    adj = prices.xs("Adj Close", axis=1, level="Field").ffill().dropna()
    adj = adj[[s for s in symbols if s in adj.columns]]

    ret = adj.pct_change().dropna(how="any")
    logret = np.log(adj).diff().dropna(how="any")
    mu = np.array([_annualize_daily(logret[s]) for s in adj.columns], dtype=float)
    sig = np.array([_annualize_daily_vol(ret[s]) for s in adj.columns], dtype=float)
    rho = ret.corr().to_numpy(dtype=float)
    div_df: pd.DataFrame = prices.xs("Dividends", axis=1, level="Field")
    div_df = div_df.reindex_like(adj).fillna(0.0)
    div = div_df
    div_12m = div.rolling(252, min_periods=20).sum()
    last_price = adj.iloc[-1]
    last_div = div_12m.iloc[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        yld = (last_div / last_price).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yld = np.clip(yld.to_numpy(dtype=float), 0.0, 0.12)
    return mu, sig, rho, yld


# ---------- CSV ingestion + weight building ----------
def load_portfolios(csv_path: str) -> Dict[str, Dict[str, Dict[str, float | None]]]:
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

    portfolios: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    for _, row in df.iterrows():
        name = str(row["Portfolio"])
        t = str(row["Ticker"]).strip().upper()
        q = (
            float(row["Quantity"])
            if "Quantity" in df.columns and not pd.isna(row["Quantity"])
            else None
        )
        v = float(row["Value"]) if "Value" in df.columns and not pd.isna(row["Value"]) else None
        portfolios.setdefault(name, {})
        portfolios[name][t] = {"qty": q, "val": v}
    return portfolios


def union_symbols(
    portfolios_dict: Dict[str, Dict[str, Dict[str, float | None]]],
) -> List[str]:
    syms: set[str] = set()
    for pmap in portfolios_dict.values():
        syms.update(pmap.keys())
    return sorted(syms)


def build_weights_for_portfolio(
    name: str,
    pos_map: Dict[str, Dict[str, float | None]],
    symbols: List[str],
    last_prices: pd.Series,
) -> np.ndarray:
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
    p = argparse.ArgumentParser(
        description="Score multiple real portfolios (CSV) with live inputs."
    )
    p.add_argument("--csv", required=True, help="CSV with columns: Portfolio,Ticker,Quantity,Value")
    p.add_argument("--start", required=True, help="Lookback start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD, default today)")
    p.add_argument("--params", default="data/params.json", help="sepp engine param json (optional)")

    # CMA controls
    p.add_argument(
        "--cma_alpha",
        type=float,
        default=None,
        help="Blend weight for CMA anchor (0–1). If set, overrides default alpha.",
    )
    p.add_argument(
        "--no_cma",
        action="store_true",
        help="Disable CMA anchoring; use trailing-only mu.",
    )

    # Safe asset override
    p.add_argument(
        "--safe_tickers",
        type=str,
        default=None,
        help="Comma-separated list of tickers to force as 'safe' (overrides auto-detect).",
    )

    # Data sources
    p.add_argument(
        "--sources",
        type=str,
        default="yahoo",
        help="Comma-separated list of sources to use in a single run (yahoo,alpha,stooq).",
    )
    p.add_argument(
        "--consensus",
        type=str,
        choices=["prefer-yahoo-fill", "median"],
        default="prefer-yahoo-fill",
        help="Consensus method when multiple sources are given in --sources.",
    )
    p.add_argument(
        "--alpha_key",
        type=str,
        default=None,
        help="Alpha Vantage API key (else read ALPHAVANTAGE_API_KEY env).",
    )

    # Repeat-per-source mode (runs full scoring once per source)
    p.add_argument(
        "--repeat_sources",
        type=str,
        default=None,
        help="Comma-separated list to run full scoring once per source (e.g. 'yahoo,alpha,stooq').",
    )

    return p.parse_args()


def apply_params(path: str | None):
    if not path or not Path(path).exists():
        return
    import json

    cfg = json.loads(Path(path).read_text())
    eng.YEARS = int(cfg.get("years", eng.YEARS))
    eng.ANNUAL_WITHDRAWAL = float(cfg.get("annual_withdrawal", eng.ANNUAL_WITHDRAWAL))
    eng.INITIAL_PORTFOLIO_VALUE = float(
        cfg.get("initial_portfolio_value", eng.INITIAL_PORTFOLIO_VALUE)
    )
    eng.N_SIM = int(cfg.get("n_sim", eng.N_SIM))
    eng.SEED = int(cfg.get("seed", eng.SEED))
    eng.BOOTSTRAP_RESAMPLES = int(cfg.get("bootstrap_resamples", eng.BOOTSTRAP_RESAMPLES))
    eng.STRESS_BLEND = cfg.get("stress_blend", eng.STRESS_BLEND)
    eng.LIQ_METHOD = cfg.get("liq_method", eng.LIQ_METHOD)
    eng.LIQ_CAP = int(cfg.get("liq_cap", eng.LIQ_CAP))
    eng.DEBUG_PRINT_ENV = bool(cfg.get("debug_print_env", eng.DEBUG_PRINT_ENV))
    eng.DEBUG_LIQ_STATS = bool(cfg.get("debug_liq_stats", eng.DEBUG_LIQ_STATS))
    eng.DEBUG_SAMPLE_PATHS = bool(cfg.get("debug_sample_paths", eng.DEBUG_SAMPLE_PATHS))


def _run_once_for_sources(args, source_list: List[str], label: str):
    """
    Executes one full scoring pass using the given source_list (e.g., ['yahoo'] or ['alpha'] or ['yahoo','alpha']).
    """
    # 1) load portfolios & symbols
    portfolios = load_portfolios(args.csv)
    symbols = union_symbols(portfolios)

    # 2) fetch prices for THIS source_list
    # removed unused: alpha_key (legacy multi-source)
    # removed unused: chosen_consensus (legacy multi-source)
    prices = fetch_prices_multi(symbols, args.start, args.end)

    # 3) compute inputs from prices
    mu, sig, rho, yld = compute_inputs(prices, symbols)

    # 4) apply CMA (optional)
    MU_TRAIL = mu.copy()
    if not args.no_cma:
        cma = []
        for s, m, y in zip(symbols, mu, yld):
            if s in {"SGOV", "VGIT", "BND", "VWOB", "SHY", "IEF", "AGG"}:
                cma.append(float(min(max(y, 0.0), 0.06)))  # bonds ≈ yield (cap 6%)
            elif s in {"GLD"}:
                cma.append(0.03)  # gold long-run anchor
            elif s in {"QQQ", "VTI", "IEFA", "VWO", "SCHD", "VIG", "CDC", "CHAT"}:
                cma.append(0.07)  # equities long-run
            else:
                cma.append(0.10)  # satellites/crypto conservative
        MU_CMA = np.array(cma, dtype=float)
        alpha = args.cma_alpha if args.cma_alpha is not None else 0.7
        mu = alpha * MU_CMA + (1 - alpha) * MU_TRAIL
        print(
            f"[CMA] alpha={alpha} trail→CMA sample: {np.round(MU_TRAIL[:4],4)} -> {np.round(mu[:4],4)}"
        )
    else:
        print("[CMA] disabled; using trailing-only mu")

    # 5) set engine globals for this run
    eng.ASSETS = symbols[:]
    eng.MU = mu
    eng.SIG = sig
    eng.RHO = rho
    eng.YIELD_RATE = yld

    # safe/growth classification (respect override if provided)
    if args.safe_tickers:
        safe_set = {t.strip().upper() for t in args.safe_tickers.split(",") if t.strip()}
        idx = [i for i, s in enumerate(symbols) if s in safe_set]
        if not idx:
            print(
                "[warn] Override provided no safe assets in the universe; falling back to defaults."
            )
            idx = _safe_indices(symbols)
    else:
        idx = _safe_indices(symbols)
    eng.SAFE_IDX = np.array(idx, dtype=int)
    eng.GROWTH_IDX = np.array([i for i in range(len(symbols)) if i not in eng.SAFE_IDX], dtype=int)
    eng.safe_asset_tickers = [symbols[i] for i in eng.SAFE_IDX]
    eng.growth_asset_tickers = [symbols[i] for i in eng.GROWTH_IDX]

    # 6) banner + basic inputs snapshot
    last_adj = prices.xs("Adj Close", axis=1, level="Field").ffill().iloc[-1]
    print("\n" + "=" * 12, f"Source: {label}", "=" * 12)
    print("=== LIVE INPUTS (Yahoo-only) (union universe) ===")
    print("Symbols:", symbols)
    print("mu  :", np.round(mu, 4))
    print("sig :", np.round(sig, 4))
    print("yld :", np.round(yld, 4))
    print("SAFE_IDX:", list(eng.SAFE_IDX), "->", [symbols[i] for i in eng.SAFE_IDX])

    # 7) score each portfolio
    for name, pos_map in portfolios.items():
        w = build_weights_for_portfolio(name, pos_map, symbols, last_adj)
        eng.score_portfolio(name, w, symbols, mu, sig, rho, yld, eng.SAFE_IDX, eng.GROWTH_IDX)


def main():
    args = parse_args()
    apply_params(args.params)

    if args.repeat_sources:
        sources_to_run = [s.strip().lower() for s in args.repeat_sources.split(",") if s.strip()]
        for s in sources_to_run:
            _run_once_for_sources(args, [s], label=s.upper())
    else:
        # existing single-run behavior (can still pass --sources 'yahoo,alpha,stooq' and --consensus)
        source_list = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
        label = ",".join([x.upper() for x in source_list])
        _run_once_for_sources(args, source_list, label=label)


if __name__ == "__main__":
    main()

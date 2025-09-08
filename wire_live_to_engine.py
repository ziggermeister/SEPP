import argparse
import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# import your v6_4_3 engine renamed to sepp_engine.py
import sepp_engine as eng

#!/usr/bin/env python3
# wire_live_to_engine.py
# Glue: fetch live data, compute inputs, patch sepp_engine globals, run engine.


# ---------- utility ----------


def load_assets_config(path: str):
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        # basic shape defaults
        cfg.setdefault("categories", {})
        cfg.setdefault("cma_anchors", {})
        cfg["cma_anchors"].setdefault("default", 0.10)
        return cfg
    except FileNotFoundError:
        # fall back to sensible defaults if file missing
        return {
            "categories": {
                "safe": ["SGOV", "VGIT", "BND", "VWOB", "SHY", "IEF", "AGG"],
                "gold": ["GLD"],
                "equity": ["QQQ", "VTI", "IEFA", "VWO", "SCHD", "VIG", "CDC", "CHAT"],
                "crypto": [],
            },
            "cma_anchors": {
                "safe": "yield_cap_0.06",
                "gold": 0.03,
                "equity": 0.07,
                "crypto": 0.10,
                "default": 0.10,
            },
        }


def build_category_map(cfg):
    catmap = {}
    for cat, tickers in cfg.get("categories", {}).items():
        for t in tickers:
            catmap[t.upper()] = cat
    return catmap


def cma_for_symbol(sym, trail_mu, live_yield, catmap, anchors):
    cat = catmap.get(sym.upper(), None)
    anchor = anchors.get(cat, anchors.get("default", 0.10))

    # rule: yield-cap string for safe sleeve
    if isinstance(anchor, str) and anchor.startswith("yield_cap_"):
        try:
            cap = float(anchor.split("_")[-1])
        except Exception:
            cap = 0.06
        return float(min(max(live_yield, 0.0), cap))

    # numeric anchors
    return float(anchor)


def _annualize_daily(logret_daily: pd.Series) -> float:
    """Annualized geometric mean from daily log-returns."""
    return float(logret_daily.mean() * 252.0)


def _annualize_daily_vol(ret_daily: pd.Series) -> float:
    """Annualized stdev from daily *arithmetic* returns."""
    return float(ret_daily.std(ddof=0) * math.sqrt(252.0))


def _safe_indices(symbols):
    safe_candidates = {"SGOV", "VGIT", "BND", "VWOB", "SHY", "IEF", "AGG"}
    idx = [i for i, s in enumerate(symbols) if s in safe_candidates]
    # no fallback: if none found, return an empty list — engine handles empty safe sleeve
    return idx


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
            div = (
                raw[(sym, "Dividends")].rename((sym, "Dividends"))
                if (sym, "Dividends") in raw.columns
                else None
            )
        else:
            # single ticker layout
            if "Adj Close" not in raw.columns:
                raise ValueError(f"Missing Adj Close for {sym}")
            adj = raw["Adj Close"].rename((sym, "Adj Close"))
            div = (
                raw["Dividends"].rename((sym, "Dividends"))
                if "Dividends" in raw.columns
                else None
            )

        # --- Per-ticker dividend fallback if missing/empty ---
        need_fallback = (div is None) or (
            isinstance(div, pd.Series) and (div.dropna().sum() == 0.0)
        )
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
    prices.columns = pd.MultiIndex.from_tuples(
        prices.columns, names=["Ticker", "Field"]
    )
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
        yld = (
            (last_div / last_price)
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    yld = np.clip(yld.to_numpy(dtype=float), 0.0, 0.12)

    return mu, sig, rho, yld


# ---------- args / run ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch live data, compute inputs, and run SEPP engine."
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Tickers in portfolio universe, order matters.",
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument(
        "--end", default=None, help="End date YYYY-MM-DD (default: today UTC)"
    )
    p.add_argument(
        "--portfolio_json",
        required=False,
        help='JSON string: {"name":"Live","weights":{"SGOV":0.3,"VTI":0.7}}',
    )
    p.add_argument(
        "--params", type=str, required=False, help="Path to params.json (optional)"
    )

    p.add_argument(
        "--sources", default="yahoo", help="Comma-separated sources: yahoo,alpha,stooq"
    )
    p.add_argument(
        "--consensus",
        default="prefer-yahoo-fill",
        choices=["prefer-yahoo-fill", "median"],
        help="How to combine sources",
    )
    p.add_argument(
        "--alpha_key",
        default=None,
        help="Alpha Vantage API key (else env ALPHAVANTAGE_API_KEY)",
    )

    p.add_argument(
        "--sources", default="yahoo", help="Comma-separated sources: yahoo,alpha,stooq"
    )
    p.add_argument(
        "--consensus",
        default="prefer-yahoo-fill",
        choices=["prefer-yahoo-fill", "median"],
        help="How to combine sources",
    )
    p.add_argument(
        "--alpha_key",
        default=None,
        help="Alpha Vantage API key (else env ALPHAVANTAGE_API_KEY)",
    )

    p.add_argument(
        "--sources", default="yahoo", help="Comma-separated sources: yahoo,alpha,stooq"
    )
    p.add_argument(
        "--consensus",
        default="prefer-yahoo-fill",
        choices=["prefer-yahoo-fill", "median"],
        help="How to combine sources",
    )
    p.add_argument(
        "--alpha_key",
        default=None,
        help="Alpha Vantage API key (else env ALPHAVANTAGE_API_KEY)",
    )
    p.add_argument(
        "--precheck",
        type=str,
        required=False,
        help="Param pack path to run golden test before live run",
    )
    p.add_argument(
        "--cma_alpha",
        type=float,
        default=0.7,
        help="Weight on CMA (0..1). 0 = pure trailing; 1 = pure CMA",
    )
    p.add_argument(
        "--no_cma", action="store_true", help="Disable CMA blend and use trailing mu"
    )
    p.add_argument(
        "--assets_config",
        default="config/assets.json",
        help="Path to assets config (categories + CMA anchors).",
    )
    return p.parse_args()


# in run(), right after: mu, sig, rho, yld = compute_inputs(...)


def apply_params(params_path: str | None):
    """Patch engine configuration from a JSON file if provided."""
    if not params_path:
        return
    cfg = json.loads(Path(params_path).read_text())
    # Core engine knobs (present names in your v6_4_3 file)
    eng.YEARS = int(cfg.get("years", eng.YEARS))
    eng.ANNUAL_WITHDRAWAL = float(cfg.get("annual_withdrawal", eng.ANNUAL_WITHDRAWAL))
    eng.INITIAL_PORTFOLIO_VALUE = float(
        cfg.get("initial_portfolio_value", eng.INITIAL_PORTFOLIO_VALUE)
    )
    eng.N_SIM = int(cfg.get("n_sim", eng.N_SIM))
    eng.SEED = int(cfg.get("seed", eng.SEED))
    eng.BOOTSTRAP_RESAMPLES = int(
        cfg.get("bootstrap_resamples", eng.BOOTSTRAP_RESAMPLES)
    )
    eng.STRESS_BLEND = cfg.get("stress_blend", eng.STRESS_BLEND)
    eng.LIQ_METHOD = cfg.get("liq_method", eng.LIQ_METHOD)
    eng.LIQ_CAP = int(cfg.get("liq_cap", eng.LIQ_CAP))
    eng.DEBUG_PRINT_ENV = bool(cfg.get("debug_print_env", eng.DEBUG_PRINT_ENV))
    eng.DEBUG_LIQ_STATS = bool(cfg.get("debug_liq_stats", eng.DEBUG_LIQ_STATS))
    eng.DEBUG_SAMPLE_PATHS = bool(cfg.get("debug_sample_paths", eng.DEBUG_SAMPLE_PATHS))


def run():
    """
    CLI runner for single-portfolio live scoring.
    Supports multi-source fetching and CMA anchoring (shared with run_portfolios.py).
    """
    args = parse_args()  # must define parse_args() like you already have
    # Apply engine params (same helper you use in run_portfolios.py)
    try:
        from run_portfolios import apply_params

        apply_params(args.params)
    except Exception:
        pass

    # -------- Optional golden precheck (idempotent) --------
    if getattr(args, "precheck", None):
        try:
            test_script = (Path(__file__).parent / "tests" / "run_golden.py").resolve()
            params_path = Path(args.precheck).resolve() if args.precheck else None
            cmd = [sys.executable, str(test_script)]
            if params_path:
                cmd += ["--params", str(params_path)]
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
            print((res.stdout or "").rstrip())
            if "GOLDEN PASS" not in (res.stdout or ""):
                print("Precheck failed; aborting live run.")
                return
        except subprocess.CalledProcessError as e:
            print((e.stdout or "").rstrip())
            print("Precheck failed; aborting live run.")
            return

    symbols = args.symbols  # list from argparse

    # -------- Fetch prices (prefer multi-source; fallback to Yahoo-only) --------
    prices = None
    used_multi = False
    try:
        from data_sources import fetch_prices_multi  # your consensus fetcher

        used_multi = True
        # removed unused: sources
        if getattr(args, "sources", None):
            sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
            # removed unused: consensus
            # removed unused: alpha_key
            "ALPHAVANTAGE_API_KEY", ""
        prices = fetch_prices_multi(symbols, args.start, args.end)
    except Exception as e:
        if used_multi:
            print(f"[multi-source] falling back to Yahoo-only due to: {e}")
        # reuse your Yahoo-only helper from run_portfolios.py
        from run_portfolios import fetch_prices

        prices = fetch_prices(symbols, args.start, args.end)

    # -------- Compute inputs --------
    from run_portfolios import compute_inputs

    mu, sig, rho, yld = compute_inputs(prices, symbols)

    # -------- CMA anchoring via shared config --------
    MU_TRAIL = mu.copy()
    try:
        from run_portfolios import (
            build_category_map,
            cma_for_symbol,
            load_assets_config,
        )

        assets_cfg = load_assets_config(
            getattr(args, "assets_config", "config/assets.json")
        )
        catmap = build_category_map(assets_cfg)
        anchors = assets_cfg.get("cma_anchors", {})
        if not getattr(args, "no_cma", False):
            alpha = args.cma_alpha if args.cma_alpha is not None else 0.7
            MU_CMA = np.array(
                [
                    cma_for_symbol(s, m, y, catmap, anchors)
                    for s, m, y in zip(symbols, MU_TRAIL, yld)
                ],
                dtype=float,
            )
            mu = alpha * MU_CMA + (1 - alpha) * MU_TRAIL
            print(
                f"[CMA] alpha={alpha}  trail→CMA sample: {np.round(MU_TRAIL[:4],4)} -> {np.round(mu[:4],4)}"
            )
        else:
            print("[CMA] disabled; using trailing-only mu")
    except Exception:
        # If config helpers aren’t available, keep trailing mu
        if not getattr(args, "no_cma", False):
            print("[CMA] config not found; using trailing-only mu")
        else:
            print("[CMA] disabled; using trailing-only mu")

    # -------- Safe/Growth indices (override > config > fallback) --------
    safe_idx = []
    try:
        from run_portfolios import build_category_map, load_assets_config

        assets_cfg = load_assets_config(
            getattr(args, "assets_config", "config/assets.json")
        )
        safe_cfg = set(
            t.upper() for t in assets_cfg.get("categories", {}).get("safe", [])
        )
        if getattr(args, "safe_tickers", None):
            safe_override = [t.strip().upper() for t in args.safe_tickers.split(",")]
            safe_idx = [i for i, s in enumerate(symbols) if s.upper() in safe_override]
            if not safe_idx:
                print(
                    "[warn] safe_tickers override provided no matches; falling back to config categories."
                )
                safe_idx = [i for i, s in enumerate(symbols) if s.upper() in safe_cfg]
        else:
            safe_idx = [i for i, s in enumerate(symbols) if s.upper() in safe_cfg]
    except Exception:
        # fallback to hardcoded list used earlier
        from run_portfolios import _safe_indices as _fallback_safe

        safe_idx = _fallback_safe(symbols)

    if not safe_idx:
        print("[warn] No safe assets found in the universe; Liquidity will be ~0y.")
    growth_idx = [i for i in range(len(symbols)) if i not in safe_idx]

    # -------- Engine globals --------
    eng.ASSETS = symbols[:]
    eng.MU = mu
    eng.SIG = sig
    eng.RHO = rho
    eng.YIELD_RATE = yld
    eng.SAFE_IDX = np.array(safe_idx, dtype=int)
    eng.GROWTH_IDX = np.array(growth_idx, dtype=int)
    eng.safe_asset_tickers = [symbols[i] for i in eng.SAFE_IDX]
    eng.growth_asset_tickers = [symbols[i] for i in eng.GROWTH_IDX]

    # -------- Build weights from --portfolio_json --------
    pj = json.loads(args.portfolio_json)
    name = pj.get("name", "Live")
    wmap = {k.upper(): float(v) for k, v in pj.get("weights", {}).items()}
    w = np.array([wmap.get(s.upper(), 0.0) for s in symbols], dtype=float)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Provided portfolio_json has zero or negative total weight.")
    w = w / total

    # -------- Print inputs & score --------
    # removed unused: last_adj
    print("=== LIVE INPUTS (Yahoo-only) ===")
    print("Symbols:", symbols)
    print("mu  :", np.round(mu, 4))
    print("sig :", np.round(sig, 4))
    print("yld :", np.round(yld, 4))
    print("SAFE_IDX:", list(eng.SAFE_IDX), "->", [symbols[i] for i in eng.SAFE_IDX])

    eng.score_portfolio(
        name, w, symbols, mu, sig, rho, yld, eng.SAFE_IDX, eng.GROWTH_IDX
    )


if __name__ == "__main__":
    run()

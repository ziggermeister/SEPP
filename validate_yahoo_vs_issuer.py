#!/usr/bin/env python3
# Validate Yahoo Adj Close vs Issuer Total-Return/NAV
# Usage example:
#   python validate_yahoo_vs_issuer.py \
#     --start 2016-01-01 --end 2024-12-31 \
#     --issuer_config config/issuer_nav_map.json \
#     --out_prefix runs/yahoo_vs_issuer

from __future__ import annotations
import argparse, json, os, io, math, datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def _to_date(s):
    if s is None: return dt.date.today().isoformat()
    return str(s)

def load_issuer_series(csv_path: str, date_col: str, nav_col: str,
                       start: str, end: str,
                       parse_dates: bool=True, tz_aware: bool=False) -> pd.Series:
    """Load issuer Total Return / NAV series from a CSV you provide."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Issuer CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col])
        if not tz_aware:
            df[date_col] = df[date_col].dt.tz_localize(None)
    df = df[[date_col, nav_col]].dropna()
    df = df.set_index(date_col).sort_index()
    start = pd.to_datetime(_to_date(start)); end = pd.to_datetime(_to_date(end))
    df = df.loc[(df.index >= start) & (df.index <= end)]
    s = df[nav_col].astype(float)
    # if values look like returns (0.123), transform to index
    if s.dropna().abs().max() < 5 and s.dropna().abs().mean() < 1.0:
        # treat as periodic simple return; make index
        idx = (1 + s.fillna(0)).cumprod()
        # rebase to 100
        s = idx / idx.iloc[0] * 100.0
    return s

def load_yahoo_adjclose(ticker: str, start: str, end: str) -> pd.Series:
    """Yahoo Adjusted Close, actions-aware (yfinance)."""
    df = yf.download(
        tickers=ticker,
        start=_to_date(start), end=_to_date(end),
        auto_adjust=False, progress=False, actions=True, group_by="ticker",
    )
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if (ticker, "Adj Close") not in df.columns:
            # single-ticker sometimes returns flat columns
            if "Adj Close" in df.columns:
                s = df["Adj Close"].copy()
            else:
                return pd.Series(dtype=float)
        else:
            s = df[(ticker, "Adj Close")].copy()
    else:
        s = df.get("Adj Close", pd.Series(dtype=float))
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.astype(float)

def normalize_to_base(a: pd.Series, base: float=100.0) -> pd.Series:
    a = a.dropna()
    if a.empty: return a
    return a / a.iloc[0] * base

def compare_series(yahoo: pd.Series, issuer: pd.Series) -> Dict[str, float]:
    """Compute alignment stats after normalizing both to 100 at first common date."""
    idx = yahoo.dropna().index.intersection(issuer.dropna().index)
    if len(idx) < 10:
        return {
            "overlap_days": float(len(idx)),
            "end_drift_pct": float("nan"),
            "mean_abs_diff_bps": float("nan"),
            "median_abs_diff_bps": float("nan"),
            "p95_abs_diff_bps": float("nan"),
            "max_abs_diff_bps": float("nan"),
            "tracking_error_bps": float("nan"),
            "corr": float("nan"),
        }
    y = normalize_to_base(yahoo.loc[idx].astype(float), 100.0)
    i = normalize_to_base(issuer.loc[idx].astype(float), 100.0)
    diff_pct = (y - i) / i.replace(0, np.nan)
    abs_diff_bps = (diff_pct.abs() * 1e4).replace([np.inf, -np.inf], np.nan).dropna()
    # daily return tracking error (std of daily differences)
    y_ret = y.pct_change().dropna()
    i_ret = i.pct_change().dropna()
    ret_idx = y_ret.index.intersection(i_ret.index)
    te = ((y_ret.loc[ret_idx] - i_ret.loc[ret_idx]).std() * 1e4) if len(ret_idx) > 5 else np.nan
    return {
        "overlap_days": float(len(idx)),
        "end_drift_pct": float(((y.iloc[-1] - i.iloc[-1]) / i.iloc[-1]) * 100.0),
        "mean_abs_diff_bps": float(abs_diff_bps.mean()),
        "median_abs_diff_bps": float(abs_diff_bps.median()),
        "p95_abs_diff_bps": float(abs_diff_bps.quantile(0.95)),
        "max_abs_diff_bps": float(abs_diff_bps.max()),
        "tracking_error_bps": float(te),
        "corr": float(pd.concat([y, i], axis=1).corr().iloc[0,1]),
    }

def make_plot(ticker: str, yahoo: pd.Series, issuer: pd.Series, out_png: str):
    plt.figure(figsize=(8,4.5))
    idx = yahoo.dropna().index.intersection(issuer.dropna().index)
    if len(idx) >= 2:
        y = normalize_to_base(yahoo.loc[idx], 100.0)
        i = normalize_to_base(issuer.loc[idx], 100.0)
        plt.plot(y.index, y.values, label=f"{ticker} Yahoo AdjClose (rebased=100)")
        plt.plot(i.index, i.values, label=f"{ticker} Issuer TR/NAV (rebased=100)")
    else:
        plt.plot(yahoo.index, yahoo.values, label=f"{ticker} Yahoo AdjClose")
        plt.plot(issuer.index, issuer.values, label=f"{ticker} Issuer")
    plt.title(f"{ticker}: Yahoo vs Issuer (rebased)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=140)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--issuer_config", required=True,
                    help="JSON file mapping tickers → {csv, date_col, nav_col}")
    ap.add_argument("--out_prefix", default="runs/yahoo_vs_issuer",
                    help="Prefix for outputs: <prefix>.csv and <prefix>_plots/*.png")
    args = ap.parse_args()

    with open(args.issuer_config, "r") as f:
        cfg = json.load(f)

    rows = []
    plot_dir = f"{args.out_prefix}_plots"
    os.makedirs("runs", exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # If user didn’t restrict tickers, use all in config
    tickers = list(cfg.keys())

    for t in tickers:
        meta = cfg[t]
        csv_path = meta["csv"]
        date_col = meta.get("date_col", "Date")
        nav_col  = meta.get("nav_col",  "TotalReturn")

        yh = load_yahoo_adjclose(t, args.start, args.end)
        try:
            iss = load_issuer_series(csv_path, date_col, nav_col, args.start, args.end)
        except Exception as e:
            print(f"[WARN] {t}: issuer CSV load failed → {e}")
            iss = pd.Series(dtype=float)

        stats = compare_series(yh, iss)
        yhn = yh.dropna()
        issn = iss.dropna()

        rows.append({
            "Ticker": t,
            "CoverageYahoo": float(0 if yhn.empty else len(yhn)),
            "CoverageIssuer": float(0 if issn.empty else len(issn)),
            **stats
        })

        # BEFORE (lines around 160s in your file)
"CoverageYahoo": float((~yh.dropna().empty) and len(yh.dropna())),
"CoverageIssuer": float((~iss.dropna().empty) and len(iss.dropna())),

# AFTER


        # plot
        try:
            out_png = os.path.join(plot_dir, f"{t}.png")
            make_plot(t, yh, iss, out_png)
        except Exception as e:
            print(f"[WARN] plot failed for {t}: {e}")

    df = pd.DataFrame(rows)
    out_csv = f"{args.out_prefix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote report → {out_csv}")
    print(f"[OK] Plots      → {plot_dir}/<ticker>.png")

if __name__ == "__main__":
    main()
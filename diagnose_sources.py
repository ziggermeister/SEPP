#!/usr/bin/env python3
# diagnose_sources.py
#
# Coverage + pairwise source consistency diagnostics.
# - Coverage per source (pct of days between --start/--end with data)
# - Yahoo vs Stooq consistency where both have data (overlap)
#   * mean / median / p95 / max absolute pct diff (bps)
#   * count of outlier days (> 1% abs diff)

from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

from data_sources import fetch_prices_multi

def _read_symbols_from_csv(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if "ticker" not in cols:
        raise ValueError(f"{csv_path} must have a 'Ticker' column")
    return sorted(df[cols["ticker"]].dropna().astype(str).unique().tolist())

def _coverage(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Percent of business days with a price for each (source,ticker)."""
    # Expect MultiIndex columns: (Ticker, Field)
    # We care about Adj Close rows only
    tickers = sorted(set([t for (t, f) in df.columns if f == "Adj Close"]))
    # Use the actual index as the date range baseline (already clipped in fetchers)
    denom = len(df.index)
    rows = []
    for t in tickers:
        ser = df[(t, "Adj Close")]
        have = ser.notna().sum()
        rows.append((t, 100.0 * have / max(denom, 1)))
    out = pd.DataFrame(rows, columns=["Ticker", "Coverage%"])
    return out

def _pair_consistency(df_y: pd.DataFrame, df_s: pd.DataFrame) -> pd.DataFrame:
    """Compare Yahoo vs Stooq Adj Close where both present. Returns bps stats by ticker."""
    tickers = sorted(set([t for (t, f) in df_y.columns if f == "Adj Close"]) |
                     set([t for (t, f) in df_s.columns if f == "Adj Close"]))
    stats = []
    for t in tickers:
        if (t, "Adj Close") not in df_y.columns or (t, "Adj Close") not in df_s.columns:
            stats.append((t, 0, np.nan, np.nan, np.nan, np.nan, 0))
            continue
        y = df_y[(t, "Adj Close")].copy()
        s = df_s[(t, "Adj Close")].copy()
        # align on overlapping, non-null days only
        both = y.dropna().index.intersection(s.dropna().index)
        if len(both) == 0:
            stats.append((t, 0, np.nan, np.nan, np.nan, np.nan, 0))
            continue
        yv = y.reindex(both)
        sv = s.reindex(both)
        # abs pct diff vs Yahoo (you could also use mid-price; Yahoo anchor is fine here)
        apd = (yv - sv).abs() / yv.replace(0, np.nan)
        apd = apd.dropna()
        if apd.empty:
            stats.append((t, len(both), np.nan, np.nan, np.nan, np.nan, 0))
            continue
        apd_bps = (apd * 1e4).astype(float)  # 1% = 100 bps
        mean_bps = float(apd_bps.mean())
        med_bps  = float(apd_bps.median())
        p95_bps  = float(apd_bps.quantile(0.95))
        max_bps  = float(apd_bps.max())
        outliers = int((apd > 0.01).sum())   # > 1% absolute diff
        stats.append((t, len(both), mean_bps, med_bps, p95_bps, max_bps, outliers))
    return pd.DataFrame(stats, columns=[
        "Ticker", "OverlapDays", "MeanAbsDiff_bps", "MedianAbsDiff_bps",
        "P95AbsDiff_bps", "MaxAbsDiff_bps", "OutlierDays_gt1pct"
    ])

def main():
    ap = argparse.ArgumentParser(description="Diagnose data source coverage & consistency")
    ap.add_argument("--csv", default="data/portfolios_live.csv",
                    help="Portfolio CSV to infer tickers (default: data/portfolios_live.csv)")
    ap.add_argument("--symbols", nargs="+", help="Explicit list of symbols (overrides --csv)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out_csv", default="runs/consistency_yahoo_vs_stooq.csv")
    args = ap.parse_args()

    os.makedirs("runs", exist_ok=True)

    symbols = args.symbols or _read_symbols_from_csv(args.csv)

    # Fetch Yahoo-only and Stooq-only frames
    df_y = fetch_prices_multi(symbols, args.start, args.end,
                              sources=["yahoo"], consensus="prefer-yahoo-fill")
    df_s = fetch_prices_multi(symbols, args.start, args.end,
                              sources=["stooq"], consensus="prefer-yahoo-fill")

    # Coverage by source
    cov_y = _coverage(df_y, args.start, args.end); cov_y["Source"] = "yahoo"
    cov_s = _coverage(df_s, args.start, args.end); cov_s["Source"] = "stooq"
    cov = pd.concat([cov_y, cov_s], ignore_index=True)

    # Pairwise consistency on overlapping days
    pair = _pair_consistency(df_y, df_s)

    # Pretty print
    with pd.option_context("display.max_rows", None, "display.width", 140, "display.precision", 2):
        print("== Coverage by Source ==")
        print(cov.sort_values(["Ticker", "Source"]).to_string(index=False))
        print("\n== Yahoo vs Stooq (overlap only; absolute pct diff, in bps) ==")
        print(pair.sort_values("Ticker").to_string(index=False))

    # Save CSV
    pair_out = pair.merge(cov, on="Ticker", how="left", suffixes=("", "_yahoo_stooq_cov"))
    pair_out.to_csv(args.out_csv, index=False)
    print(f"\n[OK] Wrote pairwise consistency report → {args.out_csv}")

if __name__ == "__main__":
    main()
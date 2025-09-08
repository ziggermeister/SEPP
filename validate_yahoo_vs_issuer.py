#!/usr/bin/env python3
"""
Validate Yahoo Adjusted Close against Issuer NAV/Total Return CSVs.

Usage:
  python validate_yahoo_vs_issuer.py \
    --start 2016-01-01 --end 2024-12-31 \
    --issuer_config config/issuer_nav_map.json \
    --out_prefix runs/yahoo_vs_issuer

If --issuer_config is omitted, common defaults are tried:
  vendor/vanguard/<TICKER>_TR_or_NAV.csv
  vendor/spdr/<TICKER>_TR_or_NAV.csv
  vendor/ishares/<TICKER>_TR_or_NAV.csv
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def _to_date(s):
    import datetime as dt

    if s is None:
        return dt.date.today().isoformat()
    if isinstance(s, dt.date) or isinstance(s, dt.datetime):
        return s.strftime("%Y-%m-%d")
    return str(s)


def load_yahoo_close(ticker: str, start: str | None, end: str | None) -> pd.Series:
    yfobj = yf.Ticker(ticker)
    df = yfobj.history(start=_to_date(start), end=_to_date(end), auto_adjust=False)
    # yfinance sometimes returns single-level columns; otherwise MultiIndex
    if not isinstance(df.columns, pd.MultiIndex) and "Adj Close" in df.columns:
        s = df["Adj Close"].astype(float)
    else:
        s = df.xs("Adj Close", axis=1, level=0).squeeze().astype(float)
    return s.dropna().sort_index()


def load_issuer_csv(
    path: str,
    date_col: str = "Date",
    nav_col: str = "NAV",
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    df = pd.read_csv(path)
    if date_col not in df.columns or nav_col not in df.columns:
        # Fallbacks commonly seen
        if "Close" in df.columns:
            nav_col = "Close"
        elif "Adj Close" in df.columns:
            nav_col = "Adj Close"
        else:
            raise ValueError(f"{path}: missing NAV/Close column")

    df = df[[date_col, nav_col]].dropna()
    # Parse dates strictly
    idx = pd.to_datetime(df[date_col], errors="raise")
    df = df.set_index(idx).drop(columns=[date_col]).sort_index()

    start_ts = pd.to_datetime(_to_date(start))
    end_ts = pd.to_datetime(_to_date(end))
    df = df.loc[(pd.to_datetime(df.index) >= start_ts) & (pd.to_datetime(df.index) <= end_ts)]

    return df[nav_col].astype(float)


def normalize_to_base(a: pd.Series, base: float = 100.0) -> pd.Series:
    a = a.dropna()
    if a.empty:
        return a
    return a / a.iloc[0] * base


def compare_series(yh: pd.Series, iss: pd.Series) -> dict:
    both = yh.dropna().to_frame("YH").join(iss.dropna().to_frame("ISS"), how="inner")
    if both.empty:
        return {
            "OverlapDays": 0,
            "MeanAbsDiff_bps": np.nan,
            "MedianAbsDiff_bps": np.nan,
            "P95AbsDiff_bps": np.nan,
            "MaxAbsDiff_bps": np.nan,
            "OutlierDays_gt1pct": 0,
        }

    rel = (both["YH"] / both["ISS"] - 1.0).abs()
    bps = rel * 1e4  # basis points
    return {
        "OverlapDays": int(len(both)),
        "MeanAbsDiff_bps": float(bps.mean()),
        "MedianAbsDiff_bps": float(bps.median()),
        "P95AbsDiff_bps": float(bps.quantile(0.95)),
        "MaxAbsDiff_bps": float(bps.max()),
        "OutlierDays_gt1pct": int((rel > 0.01).sum()),
    }


def make_plot(ticker: str, yh: pd.Series, iss: pd.Series, out_png: str) -> None:
    plt.figure(figsize=(8, 4))
    n_yh = normalize_to_base(yh)
    n_iss = normalize_to_base(iss)

    if not n_yh.empty:
        plt.plot(pd.to_datetime(n_yh.index), n_yh.to_numpy(dtype=float), label="Yahoo (Adj Close)")
    if not n_iss.empty:
        plt.plot(pd.to_datetime(n_iss.index), n_iss.to_numpy(dtype=float), label="Issuer (NAV/TR)")

    plt.title(f"{ticker}: Yahoo vs Issuer (normalized)")
    plt.xlabel("Date")
    plt.ylabel("Index (base=100)")
    plt.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--issuer_config",
        default=None,
        help="JSON map {TICKER:{path,date_col,nav_col}}",
    )
    ap.add_argument("--out_prefix", default="runs/yahoo_vs_issuer")
    args = ap.parse_args()

    start = args.start
    end = args.end

    # Union universe you’ve been using
    tickers = [
        "BND",
        "CDC",
        "CHAT",
        "DGIN",
        "GLD",
        "IBIT",
        "IEFA",
        "QQQ",
        "SCHD",
        "SGOV",
        "VGIT",
        "VIG",
        "VTI",
        "VWO",
        "VWOB",
    ]

    cfg: dict = {}
    if args.issuer_config and Path(args.issuer_config).exists():
        with open(args.issuer_config, "r") as f:
            cfg = json.load(f)

    rows: list[dict] = []
    plot_dir = f"{args.out_prefix}_plots"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    for t in tickers:
        # Yahoo
        try:
            yh = load_yahoo_close(t, start, end)
        except Exception as e:
            print(f"[WARN] {t}: yahoo load failed → {e}")
            yh = pd.Series(dtype=float)

        # Issuer CSV
        path = None
        date_col = "Date"
        nav_col = "NAV"

        if t in cfg:
            ent = cfg[t]
            path = ent.get("path")
            date_col = ent.get("date_col", date_col)
            nav_col = ent.get("nav_col", nav_col)
        if path is None or not Path(path).exists():
            print(f"[WARN] {t}: issuer CSV load failed → not found")
            iss = pd.Series(dtype=float)
        else:
            try:
                iss = load_issuer_csv(
                    path, date_col=date_col, nav_col=nav_col, start=start, end=end
                )
            except Exception as e:
                print(f"[WARN] {t}: issuer CSV load failed → {e}")
                iss = pd.Series(dtype=float)

        stats = compare_series(yh, iss)

        # Coverage counts
        yhn = yh.dropna()
        issn = iss.dropna()
        cov_y = float(0 if yhn.empty else len(yhn))
        cov_i = float(0 if issn.empty else len(issn))

        rows.append(
            {
                "Ticker": t,
                "CoverageYahoo": cov_y,
                "CoverageIssuer": cov_i,
                **stats,
            }
        )

        # Plot (best-effort)
        try:
            out_png = os.path.join(plot_dir, f"{t}.png")
            make_plot(t, yh, iss, out_png)
        except Exception as e:
            print(f"[WARN] plot failed for {t}: {e}")

    df = pd.DataFrame(rows)
    out_csv = f"{args.out_prefix}.csv"
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote report → {out_csv}")
    print(f"[OK] Plots      → {plot_dir}/<ticker>.png")


if __name__ == "__main__":
    main()

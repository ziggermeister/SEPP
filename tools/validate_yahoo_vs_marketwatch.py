#!/usr/bin/env python3
"""
Validate Yahoo vs. MarketWatch CSVs (lightweight helper).

Reads a MarketWatch-exported CSV and returns a clean tz-naive
DatetimeIndex series for comparison/plotting. Also includes a
tiny compare + plot helper.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _tz_naive(idx: pd.Index) -> pd.DatetimeIndex:
    """Return a tz-naive DatetimeIndex from any datetime-like index."""
    di = pd.to_datetime(idx)
    try:
        return di.tz_localize(None)  # type: ignore[attr-defined]
    except Exception:
        # If tz-aware, convert; if not, just ensure datetime64[ns]
        if getattr(di, "tz", None) is not None:  # type: ignore[attr-defined]
            return di.tz_convert(None)  # type: ignore[attr-defined]
        return pd.to_datetime(di)


def _pick_price_column(cols: Iterable[str]) -> Optional[str]:
    """Pick the most likely price/NAV column from a CSV header."""
    candidates = ["Adj Close", "Close", "NAV", "Price", "Value", "Total Return"]
    normalized = {c.strip().lower(): c for c in cols}
    for want in (c.lower() for c in candidates):
        if want in normalized:
            return normalized[want]
    # Fallback: return the second column if it exists
    cols_list = list(cols)
    if len(cols_list) >= 2:
        return cols_list[1]
    return None


def load_marketwatch_close(csv_path: str) -> pd.Series:
    """
    Accepts a MarketWatch CSV exported from the UI and returns a pd.Series[float]
    indexed by tz-naive DatetimeIndex.
    """
    df = pd.read_csv(csv_path)
    # Normalize/strip header names
    df.columns = pd.Index([str(c).strip() for c in df.columns])

    # Date column: prefer "Date" else first column
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    px_col = _pick_price_column(df.columns)
    if px_col is None:
        raise ValueError(f"Could not infer price/NAV column in {csv_path}")

    # Build series
    s = pd.Series(df[px_col].to_numpy(dtype=float), index=pd.to_datetime(df[date_col]))
    s = s.dropna().sort_index()
    s.index = _tz_naive(s.index)
    s.name = os.path.basename(csv_path)
    return s


def compare_series(ticker: str, yh: pd.Series, mw: pd.Series) -> Dict[str, float]:
    """Basic comparison stats across the intersection of dates."""
    idx = yh.index.intersection(mw.index)
    yh_c = yh.reindex(idx).astype(float)
    mw_c = mw.reindex(idx).astype(float)

    # Normalized (base=100) for rough visual parity
    def _norm(a: pd.Series, base: float = 100.0) -> pd.Series:
        a = a.dropna()
        return (a / a.iloc[0] * base) if not a.empty else a

    n_yh = _norm(yh_c)
    n_mw = _norm(mw_c)

    # Simple stats
    out: Dict[str, float] = {}
    out["CoverageIntersect"] = float(len(idx))
    out["N_Yahoo"] = float(yh_c.notna().sum())
    out["N_MW"] = float(mw_c.notna().sum())
    out["AbsDiffMed"] = (
        float(np.nanmedian(np.abs(n_yh.values - n_mw.values))) if len(idx) else float("nan")
    )
    out["AbsDiffP95"] = (
        float(np.nanpercentile(np.abs(n_yh.values - n_mw.values), 95)) if len(idx) else float("nan")
    )
    out["Corr"] = (
        float(pd.concat([yh_c, mw_c], axis=1).corr().iloc[0, 1]) if len(idx) else float("nan")
    )
    return out


def make_plot(ticker: str, yh: pd.Series, mw: pd.Series, out_png: str) -> None:
    """Save a small comparison chart."""
    idx = yh.index.intersection(mw.index)
    yh_c = yh.reindex(idx).astype(float)
    mw_c = mw.reindex(idx).astype(float)

    def _norm(a: pd.Series, base: float = 100.0) -> pd.Series:
        a = a.dropna()
        return (a / a.iloc[0] * base) if not a.empty else a

    n_yh = _norm(yh_c)
    n_mw = _norm(mw_c)

    plt.figure(figsize=(8, 4.2))
    plt.plot(n_yh.index, n_yh.to_numpy(dtype=float), label="Yahoo (Adj Close)")
    plt.plot(n_mw.index, n_mw.to_numpy(dtype=float), label="MarketWatch (Close/NAV)", alpha=0.85)
    plt.title(f"{ticker} – Normalized (base=100)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=144)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--mw_csv", required=True, help="Path to MarketWatch CSV")
    ap.add_argument("--out_png", default="runs/marketwatch/<ticker>.png")
    args = ap.parse_args()

    s_mw = load_marketwatch_close(args.mw_csv)

    # If you later add a Yahoo loader, compare the two here; for now just plot MW.
    # Example usage (with a Yahoo series `s_yh`):
    # stats = compare_series(args.ticker, s_yh, s_mw)
    out_png = args.out_png.replace("<ticker>", args.ticker)
    make_plot(args.ticker, s_mw, s_mw, out_png)  # plot itself to ensure pipeline works
    print(f"[OK] Plot → {out_png}")


if __name__ == "__main__":
    main()

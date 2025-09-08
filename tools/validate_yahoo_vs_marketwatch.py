#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _tz_naive(idx: pd.Index) -> pd.DatetimeIndex:
    di = pd.to_datetime(idx)
    try:
        return di.tz_localize(None)
    except Exception:
        if getattr(di, 'tz', None) is not None:
            return di.tz_convert(None)
        return pd.to_datetime(di)

import yfinance as yf


def load_yahoo_close(ticker: str, start: str, end: str) -> pd.Series:
    """
    Robustly load an adjusted-close-like series from Yahoo:
      - prefer ('Adj Close'), else fall back to ('Close')
      - handles single- and multi-index columns
      - returns tz-naive index
    """
    y = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,  # keep Adj Close present when possible
        progress=False,
        actions=True,
        group_by="ticker",
        threads=True,
    )
    if y is None or y.empty:
        return pd.Series(dtype=float, name=ticker)

    def pick(df: pd.DataFrame) -> pd.Series | None:
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance sometimes returns (TICKER, FIELD)
            if (ticker, "Adj Close") in df.columns:
                return df[(ticker, "Adj Close")]
            if (ticker, "Close") in df.columns:
                return df[(ticker, "Close")]
            # some variants omit (ticker, ...) and just have single level even with group_by
            try:
                return df["Adj Close"]
            except Exception:
                try:
                    return df["Close"]
                except Exception:
                    return None
        else:
            if "Adj Close" in df.columns:
                return df["Adj Close"]
            if "Close" in df.columns:
                return df["Close"]
            return None

    s = pick(y)
    if s is None or s.empty:
        return pd.Series(dtype=float, name=ticker)

    s = pd.to_numeric(s, errors="coerce").dropna()
    # ensure tz-naive DatetimeIndex
    try:
    s.index = _tz_naive(s.index)
    except Exception:
        s.index = s.index
        if getattr(s.index, "tz", None) is not None:
    s.index = _tz_naive(s.index)
    s = s.sort_index()
    s.name = ticker
    return s


def load_marketwatch_close(csv_path: str) -> pd.Series:
    """
    Accepts MarketWatch CSV exported from the UI.
    Expected columns (order may vary): Date, Open, High, Low, Close, Volume
    - Dates like '09/05/2025' or '2025-09-05'
    - Numbers may contain commas
    - Typically reverse-chronological; we sort ascending
    """
    df = pd.read_csv(csv_path)
    # normalize col names (strip, title-case issues)
    df.columns = pd.Index([c.strip() for c in df.columns])
    if "Date" not in df.columns:
        raise ValueError(f"Missing 'Date' in {csv_path}")
    if "Close" not in df.columns:
        # sometimes MarketWatch uses 'Close*' or similar — try a loose pick
        close_col = next((c for c in df.columns if c.lower().startswith("close")), None)
        if not close_col:
            raise ValueError(f"Missing 'Close' in {csv_path}")
        df["Close"] = df[close_col]

    # parse dates robustly
    try:
        idx = pd.to_datetime(df["Date"], errors="raise")
    except Exception:
        # fallback: explicit formats
        try:
            idx = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        except Exception:
            idx = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    vals = pd.Series(df["Close"]).astype(str).str.replace(",", "", regex=False)
    vals = pd.to_numeric(vals, errors="coerce")

    s = pd.Series(vals.to_numpy(dtype=float), index=idx)
    s = s.dropna().sort_index()
    # ensure tz-naive DatetimeIndex for plotting/comparison
    try:
    s.index = _tz_naive(s.index)
    except Exception:
        s.index = s.index
        if getattr(s.index, "tz", None) is not None:
    s.index = _tz_naive(s.index)
    s.index = _tz_naive(s.index)
    s.name = os.path.basename(csv_path)
    return s


def compare_series(ticker: str, yh: pd.Series, mw: pd.Series) -> dict:
    # intersect dates only
    idx = yh.index.intersection(mw.index)
    if len(idx) == 0:
        return {
            "Ticker": ticker,
            "OverlapDays": 0,
            "MeanAbsDiff_bps": np.nan,
            "MedianAbsDiff_bps": np.nan,
            "P95AbsDiff_bps": np.nan,
            "MaxAbsDiff_bps": np.nan,
            "OutlierDays_gt1pct": 0,
            "CoverageYahoo": float(len(yh)),
            "CoverageMW": float(len(mw)),
        }

    ya = yh.reindex(idx)
    ma = mw.reindex(idx)

    # difference in basis points vs Yahoo’s level
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = (ma / ya) - 1.0
    bps = (rel * 1e4).abs()

    return {
        "Ticker": ticker,
        "OverlapDays": int(len(idx)),
        "MeanAbsDiff_bps": float(np.nanmean(bps)),
        "MedianAbsDiff_bps": float(np.nanmedian(bps)),
        "P95AbsDiff_bps": float(np.nanpercentile(bps, 95)),
        "MaxAbsDiff_bps": float(np.nanmax(bps)),
        "OutlierDays_gt1pct": int(np.nansum((bps >= 100).astype(int))),
        "CoverageYahoo": float(len(yh)),
        "CoverageMW": float(len(mw)),
    }


def plot_series(ticker: str, yh: pd.Series, mw: pd.Series, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(9, 4.5))
    # align on union to visualize gaps too
    idx = yh.index.union(mw.index)
    ya = yh.reindex(idx)
    ma = mw.reindex(idx)
    plt.plot(ya.index, ya.to_numpy(dtype=float), label="Yahoo (Adj/Close)")
    plt.plot(ma.index, ma.to_numpy(dtype=float), label="MarketWatch (Close)", alpha=0.8)
    plt.title(f"{ticker} — Yahoo vs MarketWatch")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{ticker}.png"), dpi=144)
    plt.close()


def infer_ticker_from_filename(fn: str) -> str:
    # Filenames look like: "Download Data - FUND_US_ARCX_VTI.csv" → take last _XXXX before ".csv"
    base = os.path.basename(fn)
    core = base.rsplit(".", 1)[0]
    if "_" in core:
        cand = core.split("_")[-1]
        # sanitize (upper)
        return cand.upper()
    # fallback: strip non-letters and uppercase
    return "".join([c for c in core if c.isalnum()]).upper()


def main(vendor_dir: str, out_prefix: str):
    # Collect all candidate CSVs
    files = sorted(
        [os.path.join(vendor_dir, f) for f in os.listdir(vendor_dir) if f.lower().endswith(".csv")]
    )

    results = []
    plots_dir = f"{out_prefix}_plots"
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    for path in files:
        ticker = infer_ticker_from_filename(path)
        print(f"[CHECK] {ticker} from {path}")
        try:
            mw = load_marketwatch_close(path)
        except Exception as e:
            print(f"  !! MarketWatch parse failed for {ticker}: {e}")
            results.append(
                {
                    "Ticker": ticker,
                    "OverlapDays": 0,
                    "MeanAbsDiff_bps": np.nan,
                    "MedianAbsDiff_bps": np.nan,
                    "P95AbsDiff_bps": np.nan,
                    "MaxAbsDiff_bps": np.nan,
                    "OutlierDays_gt1pct": 0,
                    "CoverageYahoo": 0.0,
                    "CoverageMW": 0.0,
                    "Note": f"MW parse error: {e}",
                }
            )
            continue

        # Match Yahoo to MW's date span (+ a tiny buffer)
        start = mw.index.min().strftime("%Y-%m-%d")
        end = mw.index.max().strftime("%Y-%m-%d")
        yh = load_yahoo_close(ticker, start, end)

        if yh.empty or mw.empty or yh.index.intersection(mw.index).empty:
            print(f"  !! No overlap for {ticker}")
            results.append(
                {
                    "Ticker": ticker,
                    "OverlapDays": 0,
                    "MeanAbsDiff_bps": np.nan,
                    "MedianAbsDiff_bps": np.nan,
                    "P95AbsDiff_bps": np.nan,
                    "MaxAbsDiff_bps": np.nan,
                    "OutlierDays_gt1pct": 0,
                    "CoverageYahoo": float(len(yh)),
                    "CoverageMW": float(len(mw)),
                    "Note": "no overlap",
                }
            )
            continue

        stats = compare_series(ticker, yh, mw)
        results.append(stats)
        try:
            plot_series(ticker, yh, mw, plots_dir)
        except Exception as e:
            print(f"  (plot skipped for {ticker}: {e})")

    out_csv = f"{out_prefix}_summary.csv"
    pd.DataFrame(results).sort_values(["Ticker"]).to_csv(out_csv, index=False)
    print(f"[DONE] Wrote summary → {out_csv}")
    print(f"[DONE] Plots folder  → {plots_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor_dir", required=True)
    ap.add_argument("--out_prefix", required=True)
    args = ap.parse_args()
    main(args.vendor_dir, args.out_prefix)

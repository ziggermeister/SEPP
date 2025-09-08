from __future__ import annotations

import argparse
import hashlib
import json
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- helpers ----------


_PRICE_CANDIDATES: tuple[str, ...] = ("Adj Close", "Close", "NAV", "Price", "Value")
_DIV_CANDIDATES: tuple[str, ...] = (
    "Dividends",
    "Dividend",
    "Cash Dividends",
    "Cash Dividend",
    "Distributions",
    "Distribution",
)


def _detect_field_level(cols: pd.MultiIndex, candidates: Iterable[str]) -> int | None:
    """
    Return the level index that contains field labels (e.g., 'Adj Close', 'Dividends').
    Try by name first ('Field'), then by membership of any candidate in each level.
    """
    names = list(cols.names or [])
    if "Field" in names:
        return names.index("Field")

    # Try level membership
    lvl0 = set(cols.get_level_values(0))
    lvl1 = set(cols.get_level_values(1))
    cand = set(candidates)

    if lvl0 & cand:
        return 0
    if lvl1 & cand:
        return 1
    return None


def _select_field_frame(
    prices: pd.DataFrame,
    candidates: Iterable[str],
    as_float: bool = True,
) -> pd.DataFrame | None:
    """
    Try candidates in order along the detected Field level and return a 2D frame.
    Returns None if none found.
    """
    if not isinstance(prices.columns, pd.MultiIndex) or prices.columns.nlevels != 2:
        raise ValueError(
            "Expected a 2-level MultiIndex columns frame, got: "
            f"{type(prices.columns)}, nlevels={getattr(prices.columns, 'nlevels', None)}"
        )

    fld = _detect_field_level(prices.columns, candidates)
    if fld is None:
        return None

    available = set(prices.columns.get_level_values(fld))
    for c in candidates:
        if c in available:
            out = prices.xs(c, axis=1, level=fld)
            return pd.DataFrame(out, dtype=float if as_float else None)
    return None


def select_prices_or_raise(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Select a usable price frame; raise with a helpful message if not found.
    """
    frame = _select_field_frame(prices, _PRICE_CANDIDATES, as_float=True)
    if frame is not None:
        return frame

    # Build a readable message of what fields are actually present on each level
    if isinstance(prices.columns, pd.MultiIndex) and prices.columns.nlevels == 2:
        lvl0_vals = sorted(set(prices.columns.get_level_values(0)))
        lvl1_vals = sorted(set(prices.columns.get_level_values(1)))
        msg = (
            f"None of price candidates {list(_PRICE_CANDIDATES)} found.\n"
            f"Level 0 values: {lvl0_vals[:20]}{' ...' if len(lvl0_vals)>20 else ''}\n"
            f"Level 1 values: {lvl1_vals[:20]}{' ...' if len(lvl1_vals)>20 else ''}"
        )
    else:
        msg = "Columns are not a 2-level MultiIndex."

    raise ValueError(msg)


def select_dividends(prices: pd.DataFrame, like: pd.DataFrame) -> pd.DataFrame:
    """
    Select a dividends frame; if none present, return a zero DataFrame aligned to `like`.
    """
    div = _select_field_frame(prices, _DIV_CANDIDATES, as_float=True)
    if div is None:
        return pd.DataFrame(0.0, index=like.index, columns=like.columns)
    return pd.DataFrame(div, dtype=float).reindex_like(like).fillna(0.0)


def _annualize_daily(logret_daily: pd.Series) -> float:
    """Annualized arithmetic mean of log-returns (≈ drift)."""
    mu_d = float(pd.to_numeric(logret_daily, errors="coerce").mean())
    return mu_d * 252.0


def _annualize_daily_vol(ret_daily: pd.Series) -> float:
    """Annualized stdev of daily returns."""
    sd = float(pd.to_numeric(ret_daily, errors="coerce").std(ddof=0))
    return sd * float(np.sqrt(252.0))


def _fetch_one(sym: str, start: str, end: str | None) -> pd.DataFrame:
    """Fetch one symbol with Adj Close + Dividends as a 2-col DataFrame."""
    df = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data for {sym}")
    needed = []
    if "Adj Close" in df.columns:
        needed.append("Adj Close")
    elif "Adj Close" not in df.columns and "Close" in df.columns:
        df = df.rename(columns={"Close": "Adj Close"})
        needed.append("Adj Close")
    else:
        raise ValueError(f"{sym}: missing Adj Close/Close")

    if "Dividends" in df.columns:
        needed.append("Dividends")
    # Some tickers have no dividends — create a zero series aligned to index
    if "Dividends" not in df.columns:
        df["Dividends"] = 0.0
        needed.append("Dividends")

    out = df[needed].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.columns = pd.MultiIndex.from_tuples(
        [(sym, c) for c in out.columns], names=["Ticker", "Field"]
    )
    return out


def fetch_prices(symbols: List[str], start: str, end: str | None) -> pd.DataFrame:
    """Return wide DataFrame with MultiIndex columns (Ticker, Field)."""
    frames: list[pd.DataFrame] = []
    for s in symbols:
        frames.append(_fetch_one(s, start, end))
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="first")]
    return prices


def _adj_and_div(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split prices into adj-close and dividends DataFrames (both wide, float)."""
    adj = prices.xs("Adj Close", axis=1, level="Field")
    div = prices.xs("Dividends", axis=1, level="Field")
    # Make sure both are float frames
    adj = pd.DataFrame(adj, dtype=float)
    div = pd.DataFrame(div, dtype=float)
    return adj, div


def compute_params(prices: pd.DataFrame, symbols: list[str]) -> dict:
    """Compute annualized μ/σ/ρ and a simple trailing dividend yield proxy."""
    adj, div = _adj_and_div(prices)

    # Daily log-returns
    logret = np.log(adj).diff().dropna(how="any")
    ret_df: pd.DataFrame = pd.DataFrame(logret, dtype=float)

    # μ and σ
    mu = ret_df.apply(_annualize_daily, axis=0)
    sig = ret_df.apply(_annualize_daily_vol, axis=0)

    # ρ (use explicit DataFrame path for mypy)
    corr_df: pd.DataFrame = pd.DataFrame(ret_df).corr(method="pearson")
    rho_df: pd.DataFrame = corr_df.reindex(index=symbols, columns=symbols)

    # Align dividends to Adj Close shape for any downstream reindex_like usage
    div_df: pd.DataFrame = (
        pd.DataFrame(div, dtype=float).reindex_like(pd.DataFrame(adj)).fillna(0.0)
    )

    # Very simple trailing yield proxy (sum last 1y dividends / last price)
    # Guard against missing last row:
    if len(adj.index) == 0:
        yld = pd.Series(0.0, index=adj.columns)
    else:
        last_px = adj.ffill().iloc[-1]
        one_year_ago = adj.index[-1] - pd.Timedelta(days=365)
        trailing_div = div_df.loc[div_df.index >= one_year_ago].sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            yld = (trailing_div / last_px).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return {
        "mu": mu.astype(float).reindex(symbols).to_dict(),
        "sig": sig.astype(float).reindex(symbols).to_dict(),
        "rho": rho_df.astype(float).to_dict(),
        "yld": yld.astype(float).reindex(symbols).to_dict(),
    }


def stable_hash(obj) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:16]


# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None, help="Write JSON param pack here (optional)")
    args = ap.parse_args()

    prices = fetch_prices(args.symbols, args.start, args.end)
    params = compute_params(prices, args.symbols)
    params["hash"] = stable_hash(params)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(params, f, indent=2, sort_keys=True)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(params, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

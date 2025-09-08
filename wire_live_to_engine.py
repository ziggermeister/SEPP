from __future__ import annotations

import argparse
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- basic live fetch + compute (minimal, typed) ----------


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


def compute_prices(
    symbols: List[str], start: str, end: str | None = None
) -> pd.DataFrame:
    """Download daily Adj Close and Dividends for symbols into MultiIndex columns."""
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, auto_adjust=False, progress=False)
        if df.empty:
            raise ValueError(f"No data for {sym}")
        if "Adj Close" not in df.columns:
            if "Close" in df.columns:
                df = df.rename(columns={"Close": "Adj Close"})
            else:
                raise ValueError(f"{sym}: missing Adj Close/Close")
        if "Dividends" not in df.columns:
            df["Dividends"] = 0.0
        out = df[["Adj Close", "Dividends"]].copy()
        out.index = pd.to_datetime(out.index).tz_localize(None)
        out.columns = pd.MultiIndex.from_tuples(
            [(sym, c) for c in out.columns], names=["Ticker", "Field"]
        )
        frames.append(out)
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="first")]
    return prices


def _annualize_daily(logret_daily: pd.Series) -> float:
    mu_d = float(pd.to_numeric(logret_daily, errors="coerce").mean())
    return mu_d * 252.0


def _annualize_daily_vol(ret_daily: pd.Series) -> float:
    sd = float(pd.to_numeric(ret_daily, errors="coerce").std(ddof=0))
    return sd * float(np.sqrt(252.0))


def compute_inputs(
    prices: pd.DataFrame, symbols: list[str]
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Return (mu, sig, rho, yld) from prices (Adj Close, Dividends)."""
    adj = pd.DataFrame(prices.xs("Adj Close", axis=1, level="Field"), dtype=float)
    div = pd.DataFrame(prices.xs("Dividends", axis=1, level="Field"), dtype=float)

    logret = np.log(adj).diff().dropna(how="any")
    ret_df: pd.DataFrame = pd.DataFrame(logret, dtype=float)

    mu = ret_df.apply(_annualize_daily, axis=0).reindex(symbols)
    sig = ret_df.apply(_annualize_daily_vol, axis=0).reindex(symbols)
    rho = (
        pd.DataFrame(ret_df)
        .corr(method="pearson")
        .reindex(index=symbols, columns=symbols)
    )

    one_year_ago = (
        adj.index[-1] - pd.Timedelta(days=365)
        if not adj.empty
        else pd.Timestamp("1970-01-01")
    )
    last_px = adj.ffill().iloc[-1] if not adj.empty else pd.Series(1.0, index=symbols)
    trailing_div = (
        pd.DataFrame(div, dtype=float)
        .reindex_like(pd.DataFrame(adj))
        .loc[lambda d: d.index >= one_year_ago]
        .sum()
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        yld = (
            (trailing_div / last_px)
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
            .reindex(symbols)
        )

    return mu.astype(float), sig.astype(float), rho.astype(float), yld.astype(float)


# ---------- simple CLI to exercise pipeline ----------


def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch live data, compute inputs, and show summary."
    )
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    return p.parse_args()


def run():
    args = parse_args()
    syms = [s.upper() for s in args.symbols]
    prices = compute_prices(syms, args.start, args.end)
    mu, sig, rho, yld = compute_inputs(prices, syms)
    last_adj = prices.xs("Adj Close", axis=1, level="Field").ffill().iloc[-1]

    print("\n" + "=" * 12, "Live Inputs", "=" * 12)
    print("Symbols:", syms)
    print("mu  :", np.round(mu, 4).to_dict())
    print("sig :", np.round(sig, 4).to_dict())
    print("yld :", np.round(yld, 4).to_dict())
    print("rho :\n", np.round(rho, 3))
    print("\nLast Adj Close:", np.round(last_adj, 2).to_dict())


if __name__ == "__main__":
    run()

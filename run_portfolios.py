from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- small helpers (kept minimal but typed) ----------


def _annualize_daily(logret_daily: pd.Series) -> float:
    mu_d = float(pd.to_numeric(logret_daily, errors="coerce").mean())
    return mu_d * 252.0


def _annualize_daily_vol(ret_daily: pd.Series) -> float:
    sd = float(pd.to_numeric(ret_daily, errors="coerce").std(ddof=0))
    return sd * float(np.sqrt(252.0))


def _fetch_one(sym: str, start: str, end: str | None) -> pd.DataFrame:
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
    return out


def fetch_prices_yahoo(
    symbols: List[str], start: str, end: str | None = None
) -> pd.DataFrame:
    frames = [_fetch_one(s, start, end) for s in symbols]
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="first")]
    return prices


def compute_inputs(
    prices: pd.DataFrame, symbols: List[str]
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Return (mu, sig, rho, yld) for given symbols from prices (Adj Close, Dividends)."""
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


# ---------- portfolio helpers (super minimal) ----------


def load_portfolios(csv_path: str) -> Dict[str, Dict[str, Dict[str, float | None]]]:
    """
    Expect CSV columns: Portfolio, Ticker, Quantity, Value
    Returns {portfolio: {ticker: {"quantity": q, "value": v}}}
    """
    df = pd.read_csv(csv_path)
    need = {"Portfolio", "Ticker", "Quantity", "Value"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"CSV must include columns: {sorted(need)}")

    out: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    for _, row in df.iterrows():
        port = str(row["Portfolio"])
        tic = str(row["Ticker"])
        qty = float(row["Quantity"]) if not pd.isna(row["Quantity"]) else None
        val = float(row["Value"]) if not pd.isna(row["Value"]) else None
        out.setdefault(port, {})[tic] = {"quantity": qty, "value": val}
    return out


def build_weights_for_portfolio(
    name: str,
    pos_map: Dict[str, Dict[str, float | None]],
    symbols: List[str],
    last_adj: pd.Series,
) -> pd.Series:
    """Make simple value weights; if only quantities present, value = qty*price."""
    vals = {}
    for s in symbols:
        meta = pos_map.get(s, {})
        v = meta.get("value")
        q = meta.get("quantity")
        if v is None and q is not None:
            v = float(q) * float(last_adj.get(s, np.nan))
        vals[s] = float(v) if v is not None else 0.0
    w = pd.Series(vals, index=symbols, dtype=float)
    tot = float(w.sum())
    return (w / tot).fillna(0.0) if tot > 0 else w


# ---------- CLI ----------


def parse_args():
    p = argparse.ArgumentParser(
        description="Score portfolios with live inputs (Yahoo)."
    )
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", default=None)
    p.add_argument(
        "--port_csv",
        required=False,
        help="Portfolio CSV (Portfolio,Ticker,Quantity,Value)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    symbols = [s.upper() for s in args.symbols]
    prices = fetch_prices_yahoo(symbols, args.start, args.end)
    mu, sig, rho, yld = compute_inputs(prices, symbols)

    last_adj = prices.xs("Adj Close", axis=1, level="Field").ffill().iloc[-1]

    print("\n" + "=" * 12, "Source: Yahoo", "=" * 12)
    print("=== LIVE INPUTS (Yahoo-only) (union universe) ===")
    print("Symbols:", symbols)
    print("mu  :", np.round(mu, 4).to_dict())
    print("sig :", np.round(sig, 4).to_dict())
    print("yld :", np.round(yld, 4).to_dict())

    if args.port_csv:
        ports = load_portfolios(args.port_csv)
        for name, pos_map in ports.items():
            w = build_weights_for_portfolio(name, pos_map, symbols, last_adj)
            print(f"\nPortfolio: {name}")
            print("weights:", np.round(w, 4).to_dict())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

# local engine
import sepp_engine as eng

# ---------- CSV ingestion + weight building ----------


def infer_symbols_from_port_csv(csv_path: str) -> List[str]:
    """Read the portfolio CSV and return the unique tickers present."""
    df = pd.read_csv(csv_path)
    need = {"Portfolio", "Ticker"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {sorted(need)}")
    syms = sorted({str(t).strip().upper() for t in df["Ticker"].dropna()})
    if not syms:
        raise ValueError("No tickers found in CSV.")
    return syms


def load_portfolios(csv_path: str) -> Dict[str, Dict[str, Dict[str, float | None]]]:
    """
    Return: {portfolio_name: {ticker: {qty,val}}}
    CSV columns required: Portfolio, Ticker, Quantity, Value
    """
    df = pd.read_csv(csv_path)
    need = {"Portfolio", "Ticker", "Quantity", "Value"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {sorted(need)}")

    out: Dict[str, Dict[str, Dict[str, float | None]]] = {}
    for _, row in df.iterrows():
        name = str(row["Portfolio"]).strip()
        sym = str(row["Ticker"]).strip().upper()
        qty = float(row["Quantity"]) if not pd.isna(row["Quantity"]) else None
        val = float(row["Value"]) if not pd.isna(row["Value"]) else None
        out.setdefault(name, {})[sym] = {"qty": qty, "val": val}
    return out


def build_weights_for_portfolio(
    name: str,
    pos_map: Dict[str, Dict[str, float | None]],
    symbols: List[str],
    last_prices: pd.Series,
) -> pd.Series:
    """
    Make value-weights; if only quantities present, use qty*price.
    Returns a pd.Series indexed by symbols (missing tickers weight=0).
    """
    vals: Dict[str, float] = {}
    for sym, d in pos_map.items():
        qty, val = d.get("qty"), d.get("val")
        if val is None and qty is None:
            continue
        if val is None:
            px = float(last_prices.get(sym, np.nan))
            if not np.isfinite(px):
                raise ValueError(f"{name}:{sym} has qty but no live price.")
            val = qty * px  # type: ignore[operator]
        vals[sym] = vals.get(sym, 0.0) + float(val)

    total = sum(vals.values())
    w_map = {s: (vals.get(s, 0.0) / total if total > 0 else 0.0) for s in symbols}
    return pd.Series(w_map, index=symbols, dtype=float)


# ---------- Yahoo fetch (simple & robust for this CLI) ----------


def fetch_prices_yahoo(
    symbols: List[str], start: str, end: str | None = None
) -> pd.DataFrame:
    """
    Download daily Adj Close and Dividends for symbols via yfinance.
    Returns a DataFrame with MultiIndex columns (Ticker, Field).
    """
    tickers = " ".join(symbols)
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    frames = []
    for sym in symbols:
        raw = (
            data.get(sym)
            if isinstance(data, pd.DataFrame)
            and sym in data.columns.get_level_values(0)
            else data
        )
        if raw is None or not isinstance(raw, pd.DataFrame):
            raise ValueError(f"No data for {sym}")

        # Try multi-asset block first (yfinance style)
        if {"Adj Close", "Close"}.issubset(raw.columns):
            adj = raw["Adj Close"].rename((sym, "Adj Close"))
            div = (
                raw["Dividends"].rename((sym, "Dividends"))
                if "Dividends" in raw.columns
                else None
            )
        else:
            # Single-asset fallback shape
            if "Adj Close" not in raw.columns and "Close" not in raw.columns:
                raise ValueError(f"{sym}: missing Adj Close/Close")
            adj = (
                raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
            ).rename((sym, "Adj Close"))
            div = (
                raw["Dividends"].rename((sym, "Dividends"))
                if "Dividends" in raw.columns
                else None
            )

        if div is None:
            # conservative zero-dividends fallback on same index
            div = pd.Series(0.0, index=adj.index, name=(sym, "Dividends"))

        frames.append(pd.concat([adj, div], axis=1))

    prices = pd.concat(frames, axis=1)
    prices.columns = pd.MultiIndex.from_tuples(
        prices.columns, names=["Ticker", "Field"]
    )
    prices = prices.sort_index(axis=1)
    return prices


# ---------- Column normalization & field selection ----------


def normalize_prices_columns(prices: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """
    Ensure columns are MultiIndex ('Ticker','Field'); swap if ('Field','Ticker').
    """
    if not isinstance(prices.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns (Ticker, Field).")

    lvl0 = list(prices.columns.get_level_values(0))
    lvl1 = list(prices.columns.get_level_values(1))
    names = [n or "" for n in (prices.columns.names or ["", ""])]

    candidate_fields = {"Adj Close", "Close", "NAV", "Price", "Value", "Dividends"}
    looks_like_field_ticker = any(x in candidate_fields for x in set(lvl0)) and all(
        s in set(lvl1) for s in symbols
    )
    names_say_reversed = [n.lower() for n in names] == ["field", "ticker"]

    if looks_like_field_ticker or names_say_reversed:
        prices = prices.swaplevel(0, 1, axis=1)

    prices.columns = prices.columns.set_names(["Ticker", "Field"])
    prices = prices.sort_index(axis=1)
    return prices


def select_prices_or_raise(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Choose the best price panel among common field names.
    Returns a float DataFrame with columns=tickers.
    """
    candidates = ["Adj Close", "Close", "NAV", "Price", "Value"]
    if not isinstance(prices.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns (Ticker, Field).")
    names = prices.columns.names or ["", ""]
    field_level = names.index("Field") if "Field" in names else -1

    last_err: Exception | None = None
    for fld in candidates:
        try:
            sub = prices.xs(fld, axis=1, level=field_level)
            if isinstance(sub, pd.Series):
                sub = sub.to_frame()
            sub = sub.astype(float)
            if sub.notna().any().any():
                return sub
        except Exception as e:
            last_err = e
            continue

    lvl0_vals = sorted(set(prices.columns.get_level_values(0)))[:20]
    lvl1_vals = sorted(set(prices.columns.get_level_values(1)))[:20]
    msg = (
        f"None of price candidates {candidates} found.\n"
        f"Level 0 values: {lvl0_vals}\n"
        f"Level 1 values: {lvl1_vals}"
    )
    if last_err:
        msg += f"\n(last error: {last_err})"
    raise ValueError(msg)


# ---------- Inputs (mu/sig/rho/yld) ----------


def compute_inputs(prices: pd.DataFrame, symbols: List[str]):
    """
    From prices (Adj Close, Dividends), compute annualized mu/sig/rho and a simple yield proxy.
    """
    prices = normalize_prices_columns(prices, symbols)
    adj = select_prices_or_raise(prices).ffill()

    # Returns
    logret = np.log(adj).diff().dropna(how="any")
    mu = (1.0 + logret.mean()) ** 252 - 1.0
    sig = logret.std() * np.sqrt(252.0)
    rho = logret.corr()

    # Yield (Dividends sum over 252d / last price); if missing, zeros
    yld = pd.Series(0.0, index=adj.columns)
    try:
        div = (
            prices.xs("Dividends", axis=1, level="Field")
            .ffill()
            .reindex_like(adj)
            .fillna(0.0)
        )
        yld = div.rolling(252, min_periods=20).sum().iloc[-1] / adj.iloc[-1]
    except Exception:
        pass

    return mu, sig, rho, yld


def safe_indices_for(
    symbols: list[str], config_path: Path = Path("config/assets.json")
) -> list[int]:
    """
    Resolve 'safe' indices for the given symbols, using config/assets.json.
    Falls back to a sensible default set if the config is missing.
    """
    # default fallback set if config is absent/malformed
    fallback_safe = {"SGOV", "SHY", "VGIT", "IEF", "BND", "AGG", "VWOB", "TLT"}
    safe_set: set[str] = set()

    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        listed = cfg.get("safe", [])
        if not isinstance(listed, list) or not all(isinstance(x, str) for x in listed):
            raise ValueError("config/assets.json 'safe' must be a list[str]")
        safe_set = {s.upper() for s in listed}
    except Exception as e:
        print(f"Warning: {config_path} not usable ({e}); using fallback safe set.")
        safe_set = fallback_safe

    # Map to indices in the *current* symbol order
    idx = [i for i, s in enumerate(symbols) if s.upper() in safe_set]
    return idx


# ---------- CLI ----------


def parse_args():
    p = argparse.ArgumentParser(
        description="Score portfolios with live inputs (Yahoo)."
    )
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--port_csv",
        required=True,
    )
    return p.parse_args()


# --- pretty banner ---
def _banner(source: str):
    print("\n" + "=" * 12, f"Source: {source}", "=" * 12)
    print("=== LIVE INPUTS (Yahoo-only) ===")


def main():
    args = parse_args()

    # 1) Always infer symbols from the provided portfolio CSV
    if not args.port_csv:
        raise SystemExit("--port_csv is required (symbols are inferred from it)")
    symbols = infer_symbols_from_port_csv(args.port_csv)
    print("Symbols inferred from CSV:", symbols)

    # 2) Fetch prices (MultiIndex columns: ['Ticker', 'Field'])
    prices = fetch_prices_yahoo(symbols, args.start, args.end)

    # 3) Compute inputs (mu, sig, rho, yld) as pandas objects keyed by ticker
    mu_s, sig_s, rho_df, yld_s = compute_inputs(prices, symbols)

    _banner("Yahoo")
    print("mu  :", np.round(mu_s, 4).to_dict())
    print("sig :", np.round(sig_s, 4).to_dict())
    print("yld :", np.round(yld_s, 4).to_dict())

    # 4) Load portfolios and build weights (value weights if 'Value' exists; else qty * last price)
    ports = load_portfolios(args.port_csv)
    adj = select_prices_or_raise(
        prices
    )  # chooses from ["Adj Close","Close","NAV","Price","Value"]
    last_adj = adj.ffill().iloc[-1]

    # 5) Safe / Growth indices from engine config (supports config/config.json or config/assets.json)
    cfg = eng.load_assets_config()
    safe_tickers = set(cfg["safe"])
    safe_idx = [i for i, s in enumerate(symbols) if s in safe_tickers]
    growth_idx = [i for i in range(len(symbols)) if i not in safe_idx]

    if not safe_idx:
        print("ERROR: No 'safe' assets found for this symbol set.")
        print("Symbols:", symbols)
        print("config safe list:", sorted(safe_tickers))
        raise SystemExit(
            "Update config/assets.json 'safe' to include at least one of your symbols "
            "(e.g., SGOV/VGIT/BND/VWOB/SHY/IEF/AGG), or adjust your CSV."
        )
    else:
        print(f"Safe assets found: {[symbols[i] for i in safe_idx]}")

    # 6) Convert mu/sig/rho/yld into numpy arrays/matrix in *exact* symbols order
    def series_to_array(s: pd.Series) -> np.ndarray:
        return np.asarray([float(s[sym]) for sym in symbols], dtype=float)

    mu = series_to_array(mu_s)
    sig = series_to_array(sig_s)
    yld = series_to_array(yld_s)
    rho = rho_df.reindex(index=symbols, columns=symbols).to_numpy(dtype=float)

    # 7) Score each portfolio
    for name, pos_map in ports.items():
        w = build_weights_for_portfolio(name, pos_map, symbols, last_adj)
        print(f"\nPortfolio: {name}")
        print("weights:", np.round(w, 4).to_dict())

        # Call the engine scorer (numpy inputs; explicit indices)
        blended, headline, subs, se, liq_per_path, t0_liq = eng.score_portfolio(
            name, w, symbols, mu, sig, rho, yld, safe_idx, growth_idx
        )

        score = float(headline.get("score", blended.get("score", np.nan)))
        ruin = float(se.get("ruin_prob", np.nan))
        print(f"score={score:.1f}, liq={t0_liq:.2f}y, ruin={ruin:.4f}")


if __name__ == "__main__":
    main()

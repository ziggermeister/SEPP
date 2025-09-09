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

# ---------------- Scoring presets loader (case-insensitive) ----------------
from pathlib import Path

def load_scoring_config(path: str = "config/scoring.json") -> dict:
    """
    Load config/scoring.json and return the parsed dict.
    Exits with a clear message if the file is missing or invalid.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing scoring config: {path}")
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise SystemExit(f"Failed to parse {path}: {e}")

def select_preset_or_die(requested: str | None, scoring_cfg: dict) -> tuple[dict, str]:
    """
    Return (weights_dict, canonical_name) for the requested preset.
    Lookup is case-insensitive and trims whitespace.
    If not found, exits and prints available presets.
    """
    presets = scoring_cfg.get("presets", {})
    if not isinstance(presets, dict) or not presets:
        raise SystemExit("No presets found under 'presets' in config/scoring.json.")

    # Case-insensitive map from lowercased key to canonical key
    ci_map = {k.casefold(): k for k in presets.keys()}

    # Default to "SEPP" if nothing specified
    wanted = (requested or "SEPP").strip()
    canon = ci_map.get(wanted.casefold())
    if not canon:
        available = ", ".join(sorted(presets.keys()))
        raise SystemExit(
            f"Preset '{wanted}' not found in config/scoring.json. "
            f"Available: {available}"
        )

    weights = presets[canon]
    if not isinstance(weights, dict) or not weights:
        raise SystemExit(f"Preset '{canon}' exists but has no weights object.")

    # Light sanity: ensure the 3 period keys exist
    for period in ("Yrs1-4", "Yrs5-8", "Yrs9-12"):
        if period not in weights or not isinstance(weights[period], dict):
            raise SystemExit(f"Preset '{canon}' is missing period '{period}'.")

    return weights, canon

def apply_preset_to_engine(eng, weights: dict, name: str) -> None:
    """
    Override engine weights for this run and print which preset was applied.
    """
    eng.WEIGHTS = weights
    print(f"Using preset: {name}")

def _normalize_presets_object(raw: dict) -> dict:
    """
    Accept either:
      { "presets": { "Name": {..}, ... } }  or  { "Name": {..}, ... }
    Return the inner map { preset_name: weights } or {} on failure.
    """
    if not isinstance(raw, dict):
        return {}
    if "presets" in raw and isinstance(raw["presets"], dict):
        return raw["presets"]
    return raw


def load_scoring_presets(path: str = "config/scoring.json") -> dict:
    """
    Load scoring presets from JSON. Returns a dict: {preset_name: weights_dict}.
    Handles both wrapped and flat JSON forms.
    """
    try:
        with open(path, "r") as f:
            raw = json.load(f)
        return _normalize_presets_object(raw)
    except FileNotFoundError:
        print(f"Warning: {path} not found; using engine default weights.")
        return {}
    except Exception as e:
        print(f"Warning: could not read {path}: {e}; using engine default weights.")
        return {}


def apply_ruin_constraint(blended: dict, preset: dict) -> bool:
    """Return True if blended passes optional 'Constraint' in preset, else True if none."""
    cons = preset.get("Constraint", {}) if isinstance(preset, dict) else {}
    rmax = cons.get("RuinMax")
    if rmax is not None and blended.get("Ruin") is not None:
        return float(blended["Ruin"]) <= float(rmax)
    return True  # no constraint or not applicable


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

def fetch_prices_yahoo(symbols: List[str], start: str, end: str | None = None) -> pd.DataFrame:
    """
    Download daily Adj Close and Dividends for symbols via yfinance.
    Returns a DataFrame with MultiIndex columns (Ticker, Field).
    """
    tickers = " ".join(symbols)
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, group_by="ticker", progress=False)

    frames = []
    for sym in symbols:
        raw = data.get(sym) if isinstance(data, pd.DataFrame) and sym in data.columns.get_level_values(0) else data
        if raw is None or not isinstance(raw, pd.DataFrame):
            raise ValueError(f"No data for {sym}")

        # Try multi-asset block first (yfinance style)
        if {"Adj Close", "Close"}.issubset(raw.columns):
            adj = raw["Adj Close"].rename((sym, "Adj Close"))
            div = raw["Dividends"].rename((sym, "Dividends")) if "Dividends" in raw.columns else None
        else:
            # Single-asset fallback shape
            if "Adj Close" not in raw.columns and "Close" not in raw.columns:
                raise ValueError(f"{sym}: missing Adj Close/Close")
            adj = (raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]).rename((sym, "Adj Close"))
            div = raw["Dividends"].rename((sym, "Dividends")) if "Dividends" in raw.columns else None

        if div is None:
            # conservative zero-dividends fallback on same index
            div = pd.Series(0.0, index=adj.index, name=(sym, "Dividends"))

        frames.append(pd.concat([adj, div], axis=1))

    prices = pd.concat(frames, axis=1)
    prices.columns = pd.MultiIndex.from_tuples(prices.columns, names=["Ticker", "Field"])
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
        div = prices.xs("Dividends", axis=1, level="Field").ffill().reindex_like(adj).fillna(0.0)
        yld = div.rolling(252, min_periods=20).sum().iloc[-1] / adj.iloc[-1]
    except Exception:
        pass

    return mu, sig, rho, yld


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Score portfolios with live inputs (Yahoo).")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--port_csv", required=True)
    p.add_argument("--preset", default=None, help="Name of scoring preset in config/scoring.json")
    p.add_argument("--all_presets", action="store_true", help="Run all presets from config/scoring.json")
    return p.parse_args()


# --- pretty banner ---
def _banner(source: str):
    print("\n" + "=" * 12, f"Source: {source}", "=" * 12)
    print("=== LIVE INPUTS (Yahoo-only) ===")


def main():
    args = parse_args()
    # --- Load scoring presets and pick one (case-insensitive) ---
    import json
    from pathlib import Path

    def load_scoring_config(path="config/scoring.json"):
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Missing scoring config: {path}")
        try:
            return json.loads(p.read_text())
        except Exception as e:
            raise SystemExit(f"Failed to parse {path}: {e}")

    scoring_cfg = load_scoring_config()
    presets = scoring_cfg.get("presets", {})
    if not presets:
        raise SystemExit("No presets found under 'presets' in config/scoring.json.")

    ci_map = {k.casefold(): k for k in presets.keys()}
    requested = (args.preset or "SEPP").strip()
    canon_key = ci_map.get(requested.casefold())
    if not canon_key:
        avail = ", ".join(sorted(presets.keys()))
        raise SystemExit(f"Preset '{requested}' not found. Available: {avail}")

    import sepp_engine as eng
    eng.WEIGHTS = presets[canon_key]
    print(f"Using preset: {canon_key}")


    # 1) Infer symbols from the provided portfolio CSV
    if not args.port_csv:
        raise SystemExit("--port_csv is required (symbols are inferred from it)")
    symbols = infer_symbols_from_port_csv(args.port_csv)
    print("Symbols inferred from CSV:", symbols)

    # 2) Fetch prices (MultiIndex columns: ['Ticker','Field'])
    prices = fetch_prices_yahoo(symbols, args.start, args.end)

    # 3) Compute inputs (mu, sig, rho, yld) keyed by ticker
    mu_s, sig_s, rho_df, yld_s = compute_inputs(prices, symbols)

    _banner("Yahoo")
    print("mu  :", np.round(mu_s, 4).to_dict())
    print("sig :", np.round(sig_s, 4).to_dict())
    print("yld :", np.round(yld_s, 4).to_dict())

    # 4) Load portfolios; build weights at last available adjusted price
    ports = load_portfolios(args.port_csv)
    adj = select_prices_or_raise(prices)  # chooses from ["Adj Close","Close","NAV","Price","Value"]
    last_adj = adj.ffill().iloc[-1]

    # 5) Safe/Growth indices from engine config (config/assets.json or config/config.json)
    cfg = eng.load_assets_config()
    safe_tickers = set(cfg.get("safe", []))
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

    # 6) Convert mu/sig/rho/yld to numpy arrays/matrix in the exact symbols order
    def series_to_array(s: pd.Series) -> np.ndarray:
        return np.asarray([float(s[sym]) for sym in symbols], dtype=float)

    mu = series_to_array(mu_s)
    sig = series_to_array(sig_s)
    yld = series_to_array(yld_s)
    rho = rho_df.reindex(index=symbols, columns=symbols).to_numpy(dtype=float)

    # 7) Load scoring presets and resolve which to use (case-insensitive)
    presets_path = "config/scoring.json"
    presets_map = load_scoring_presets(presets_path)  # returns dict[name]->weights or {}
    _engine_default = getattr(eng, "WEIGHTS", {})

    if args.all_presets:
        preset_map = presets_map or {"_EngineDefault": _engine_default}
    elif args.preset:
        requested = args.preset.strip()
        ci = {k.casefold(): k for k in presets_map.keys()}
        canon = ci.get(requested.casefold())
        if not canon:
            available = ", ".join(sorted(presets_map.keys())) or "_EngineDefault"
            raise SystemExit(
                f"Preset '{requested}' not found in {presets_path}. Available: {available}"
            )
        preset_map = {canon: presets_map[canon]}
    else:
        preset_map = {"_EngineDefault": _engine_default}

    # 8) Score each portfolio under the chosen preset(s)
    for name, pos_map in ports.items():
        w = build_weights_for_portfolio(name, pos_map, symbols, last_adj)
        print(f"\nPortfolio: {name}")
        print("weights:", np.round(w, 4).to_dict())

        for preset_name, preset_weights in preset_map.items():
            # Swap the engine lens for this evaluation
            eng.WEIGHTS = preset_weights
            if preset_name != "_EngineDefault":
                print(f"Using preset: {preset_name}")

            blended, headline, subs, se, liq_per_path, t0_liq = eng.score_portfolio(
                name, w, symbols, mu, sig, rho, yld, safe_idx, growth_idx
            )

            # Optional constraint: fail fast if preset enforces a ruin ceiling
            if not apply_ruin_constraint(blended, preset_weights):
                print(
                    f"[{preset_name}] constraint FAIL → ruin={blended.get('Ruin', float('nan')):.4f}"
                )
                continue

            # Headline is a float (new API); still guard against legacy dict shape
            score = (
                float(headline)
                if isinstance(headline, (int, float))
                else float(headline.get("score", np.nan))
            )
            ruin = float(blended.get("Ruin", np.nan))
            liq = float(t0_liq)
            print(f"[{preset_name}] score={score:.1f}, liq={liq:.2f}y, ruin={ruin:.4f}")

    # 9) Restore the engine’s default weights
    eng.WEIGHTS = _engine_default


if __name__ == "__main__":
    main()
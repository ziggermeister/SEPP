from __future__ import annotations
import os, hashlib, pickle
from pathlib import Path
from typing import Iterable, Optional, List
import pandas as pd

from adapters.provider_yf import YahooProvider
from adapters.provider_tiingo import TiingoProvider
from adapters.provider_alpha import AlphaVantageProvider
from adapters.provider_fmp import FMPProvider

CACHE_DIR = Path(os.getenv("SEPP_CACHE_DIR", "data/cache"))

def _cache_key(symbols: Iterable[str], start: str, end: Optional[str], interval: str) -> Path:
    key = f"{sorted(list(symbols))}|{start}|{end}|{interval}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"prices_{h}.pkl"

def _save_cache(path: Path, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(df, f)

def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def get_prices_multi(symbols: Iterable[str], start: str, end: Optional[str]=None,
                     interval: str="1d", prefer: Optional[List[str]]=None) -> pd.DataFrame:
    """
    Try multiple providers until we get data for all requested symbols.
    prefer sets the order, e.g. ['tiingo','yahoo','alphavantage','fmp'].
    """
    prefer = prefer or ["tiingo","yahoo","alphavantage","fmp"]
    providers = {
        "yahoo": YahooProvider(),
        "tiingo": TiingoProvider(),
        "alphavantage": AlphaVantageProvider(),
        "fmp": FMPProvider(),
    }

    cache_path = _cache_key(symbols, start, end, interval)
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    want = set(symbols)
    collected = []
    for key in prefer:
        df = providers[key].get_prices(want, start, end, interval)
        if df is None or df.empty:
            continue
        # Identify which symbols we got
        got = set([c[0] for c in df.columns.unique(level=0)])
        collected.append(df)
        want = want - got
        if not want:
            break

    if not collected:
        return pd.DataFrame()

    merged = pd.concat(collected, axis=1)
    # Drop duplicated (symbol, field) columns keeping first
    merged = merged.loc[:,~merged.columns.duplicated()]
    merged = merged.sort_index()
    _save_cache(cache_path, merged)
    return merged

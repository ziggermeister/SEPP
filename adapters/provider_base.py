from abc import ABC, abstractmethod
from typing import Iterable, Optional

import pandas as pd


class PriceProvider(ABC):
    """Abstract provider API: returns a tidy DataFrame with index=Datetime, columns MultiIndex(levels=[symbol, field])."""

    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def get_prices(
        self,
        symbols: Iterable[str],
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame: ...


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts wide or tidy, returns tidy MultiIndex columns: (symbol, field) with fields among:
    ['Open','High','Low','Close','Adj Close','Volume'] and DatetimeIndex tz-naive.
    """
    if df.empty:
        return df
    # If it's already multiindex with 2 levels, assume good.
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
        out = df.copy()
    else:
        # Try to detect yfinance-style columns
        if "Adj Close" in df.columns or "Close" in df.columns:
            # single symbol inferred
            sym = df.attrs.get("symbol") or "SYMBOL"
            out = pd.concat(
                {
                    sym: df[
                        [
                            c
                            for c in df.columns
                            if c
                            in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                        ]
                    ]
                },
                axis=1,
            )
        else:
            # already symbol-level columns (yfinance multi-Ticker download)
            out = df.copy()
    out = out.sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out

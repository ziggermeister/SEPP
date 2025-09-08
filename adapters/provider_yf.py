from typing import Iterable, Optional

import pandas as pd
import yfinance as yf

from .provider_base import PriceProvider, standardize


class YahooProvider(PriceProvider):
    def name(self) -> str:
        return "yahoo"

    def get_prices(
        self,
        symbols: Iterable[str],
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        syms = list(symbols)
        if not syms:
            return pd.DataFrame()
        data = yf.download(
            tickers=" ".join(syms),
            start=start,
            end=end,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        # yfinance returns MultiIndex (field, symbol) or (symbol, field) depending on version; normalize
        if isinstance(data.columns, pd.MultiIndex):
            # sometimes fields level first; ensure (symbol, field)
            levels = list(data.columns.names)
            if levels[0] in ["Adj Close", "Close", "Open", "High", "Low", "Volume"]:
                data = data.swaplevel(0, 1, axis=1)
        else:
            # single symbol returns flat columns
            sym = syms[0]
            data.attrs["symbol"] = sym
        return standardize(data)

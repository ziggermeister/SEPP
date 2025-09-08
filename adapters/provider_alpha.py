import os
import time
from typing import Iterable, Optional

import pandas as pd
import requests

from .provider_base import PriceProvider, standardize

ALPHA_URL = "https://www.alphavantage.co/query"


class AlphaVantageProvider(PriceProvider):
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("ALPHAVANTAGE_API_KEY")

    def name(self) -> str:
        return "alphavantage"

    def get_prices(
        self,
        symbols: Iterable[str],
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if not self.token:
            return pd.DataFrame()
        out = {}
        for sym in symbols:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": sym,
                "apikey": self.token,
                "outputsize": "full",
            }
            r = requests.get(ALPHA_URL, params=params, timeout=30)
            if r.status_code != 200:
                continue
            j = r.json()
            key = "Time Series (Daily)"
            if key not in j:
                continue
            df = pd.DataFrame(j[key]).T
            df.index = pd.to_datetime(df.index)
            df = (
                df.rename(
                    columns={
                        "1. open": "Open",
                        "2. high": "High",
                        "3. low": "Low",
                        "4. close": "Close",
                        "5. adjusted close": "Adj Close",
                        "6. volume": "Volume",
                    }
                )[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                .astype(float)
                .sort_index()
            )
            df.attrs["symbol"] = sym
            out[sym] = df
            time.sleep(12)  # Alpha Vantage rate limits
        if not out:
            return pd.DataFrame()
        wide = pd.concat(out, axis=1)
        return standardize(wide)

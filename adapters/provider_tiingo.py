import os
import time
from typing import Iterable, Optional

import pandas as pd
import requests

from .provider_base import PriceProvider, standardize

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{sym}/prices"


class TiingoProvider(PriceProvider):
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("TIINGO_API_KEY")

    def name(self) -> str:
        return "tiingo"

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
            params = {"startDate": start.split("T")[0], "token": self.token}
            if end:
                params["endDate"] = end.split("T")[0]
            r = requests.get(TIINGO_URL.format(sym=sym), params=params, timeout=30)
            if r.status_code != 200:
                continue
            j = r.json()
            if not j:
                continue
            df = pd.DataFrame(j)
            if "adjClose" in df.columns:
                df.rename(
                    columns={
                        "adjClose": "Adj Close",
                        "close": "Close",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "volume": "Volume",
                        "date": "Date",
                    },
                    inplace=True,
                )
            elif "close" in df.columns:
                df.rename(
                    columns={
                        "close": "Close",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "volume": "Volume",
                        "date": "Date",
                    },
                    inplace=True,
                )
                df["Adj Close"] = df["Close"]
            df = df.set_index(pd.to_datetime(df["Date"])).drop(columns=["Date"], errors="ignore")
            df.attrs["symbol"] = sym
            out[sym] = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
            time.sleep(0.05)  # polite
        if not out:
            return pd.DataFrame()
        wide = pd.concat(out, axis=1)
        return standardize(wide)

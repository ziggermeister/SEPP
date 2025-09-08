import os
import time
from typing import Iterable, Optional

import pandas as pd
import requests

from .provider_base import PriceProvider, standardize

FMP_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/{sym}"


class FMPProvider(PriceProvider):
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("FMP_API_KEY")

    def name(self) -> str:
        return "fmp"

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
            params = {"apikey": self.token, "from": start.split("T")[0]}
            if end:
                params["to"] = end.split("T")[0]
            r = requests.get(FMP_URL.format(sym=sym), params=params, timeout=30)
            if r.status_code != 200:
                continue
            j = r.json()
            hist = j.get("historical", [])
            if not hist:
                continue
            df = pd.DataFrame(hist)
            df.rename(
                columns={
                    "date": "Date",
                    "adjClose": "Adj Close",
                    "close": "Close",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "volume": "Volume",
                },
                inplace=True,
            )
            df = df.set_index(pd.to_datetime(df["Date"])).drop(
                columns=["Date"], errors="ignore"
            )
            df = df[
                ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            ].sort_index()
            df.attrs["symbol"] = sym
            out[sym] = df
            time.sleep(0.25)
        if not out:
            return pd.DataFrame()
        wide = pd.concat(out, axis=1)
        return standardize(wide)

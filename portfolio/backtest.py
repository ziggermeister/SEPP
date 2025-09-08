from __future__ import annotations

import numpy as np
import pandas as pd


def realized_cagr(prices: pd.DataFrame, weights: np.ndarray, start: str, end: str) -> float:
    """
    Compute realized CAGR using Adj Close over [start,end] for a static-weight portfolio.
    """
    adj = prices.loc[:, pd.IndexSlice[:, "Adj Close"]]
    symbols = [c[0] for c in adj.columns]
    px = adj.copy()
    px.columns = pd.Index(symbols)
    px.index = pd.to_datetime(px.index)
    idx = pd.to_datetime(px.index)
    sub = px.loc[start:end].dropna()
    sub.index = idx.loc[sub.index]
    if sub.empty:
        return float("nan")
    start_v = (sub.iloc[0] * weights).sum()
    end_v = (sub.iloc[-1] * weights).sum()
    i = pd.to_datetime(sub.index)
    years = (i[-1] - i[0]).total_seconds() / (365.25 * 86400)
    if years <= 0:
        return float("nan")
    return (end_v / start_v) ** (1 / years) - 1

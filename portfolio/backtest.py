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
    px.columns = symbols
    sub = px.loc[start:end].dropna()
    if sub.empty:
        return float("nan")
    start_v = (sub.iloc[0] * weights).sum()
    end_v = (sub.iloc[-1] * weights).sum()
    years = (sub.index[-1] - sub.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return (end_v / start_v) ** (1/years) - 1

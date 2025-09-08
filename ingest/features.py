import numpy as np
import pandas as pd


def annualized_mu_sigma(
    prices: pd.DataFrame, trading_days: int = 252
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    prices: tidy MultiIndex columns (symbol, field) with 'Adj Close'.
    Returns (mu, sigma, rho) estimated from daily log returns, annualized.
    """
    # Pivot to symbols x time
    adj = prices.loc[:, pd.IndexSlice[:, "Adj Close"]]
    adj.columns = pd.Index([c[0] for c in adj.columns])
    adj = adj.dropna(axis=0, how="any").sort_index()

    rets = np.log(adj / adj.shift(1)).dropna()
    mu_daily = rets.mean().values
    cov_daily = np.cov(rets.values, rowvar=False)
    sig_daily = np.sqrt(np.diag(cov_daily))

    mu_ann = (1 + mu_daily) ** trading_days - 1  # approx
    sig_ann = sig_daily * np.sqrt(trading_days)
    rho = np.corrcoef(rets.values, rowvar=False)
    return mu_ann, sig_ann, rho


def dividend_yield_from_prices(prices: pd.DataFrame) -> np.ndarray:
    """
    Placeholder: if 'Adj Close' vs 'Close' differs, derive rough yield proxy.
    For ETFs you may want to replace with actual trailing twelve month yield via an API.
    """
    adj = prices.loc[:, pd.IndexSlice[:, "Adj Close"]]
    close = prices.loc[:, pd.IndexSlice[:, "Close"]]
    adj.columns = pd.Index([c[0] for c in adj.columns])
    close.columns = pd.Index([c[0] for c in close.columns])
    align = adj.join(close, lsuffix="_adj", rsuffix="_px", how="inner")
    # crude proxy: long-run average (close - adj) / close, clipped
    diff = (align.filter(like="_px") - align.filter(like="_adj")).abs()
    denom = align.filter(like="_px").replace(0, np.nan)
    y = (diff / denom).mean().clip(0, 0.07)  # cap 7%
    return y.to_numpy(dtype=float)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_portfolio_prices(
    prices: pd.DataFrame, weights: dict, title: str = "Portfolio (Adj Close)"
) -> None:
    adj = prices.loc[:, pd.IndexSlice[:, "Adj Close"]]
    adj.columns = pd.Index([c[0] for c in adj.columns])
    wvec = np.array([weights.get(s, 0.0) for s in adj.columns], dtype=float)
    wvec = wvec / wvec.sum() if wvec.sum() > 0 else wvec
    nav = (adj / adj.iloc[0]) @ wvec
    plt.figure(figsize=(9, 4))
    plt.plot(nav.index, nav.values)
    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

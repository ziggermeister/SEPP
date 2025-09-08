#!/usr/bin/env python3
import argparse
import json

import numpy as np

import sepp_engine as eng  # <-- single source of truth


def load_param_pack(path):
    import pandas as pd

    P = json.loads(open(path).read())
    tickers = P["universe"]
    mu = np.array([P["mu"][t] for t in tickers], float)
    sig = np.array([P["sigma"][t] for t in tickers], float)
    yld = np.array([P["yield_rate"][t] for t in tickers], float)
    rho = pd.DataFrame(P["rho"]).loc[tickers, tickers].values.astype(float)
    return tickers, mu, sig, rho, yld, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to param pack JSON")
    args = ap.parse_args()

    # Optional environment banner
    try:
        import numpy
        import scipy

        print(f"NumPy {numpy.__version__} | SciPy {scipy.__version__}")
    except Exception:
        pass

    assets, MU, SIG, RHO, YIELD, PACK = load_param_pack(args.params)
    print(
        f"PARAM PACK: as_of={PACK.get('as_of')} hash={PACK.get('stable_hash')} source_prices={PACK.get('sources', {}).get('prices')}"
    )
    print(f"UNIVERSE: {', '.join(assets)}")

    # safe/growth split (same rule used in wire_live_to_engine)
    safe_names = {"SGOV", "VGIT", "BND", "VWOB", "SHY", "IEF", "AGG"}
    SAFE_IDX = np.array([i for i, t in enumerate(assets) if t in safe_names], dtype=int)
    GROWTH_IDX = np.array(
        [i for i in range(len(assets)) if i not in SAFE_IDX], dtype=int
    )

    # Use the portfolios and weights defined in sepp_engine
    for name, w in eng.PORTFOLIOS.items():
        eng.score_portfolio(name, w, assets, MU, SIG, RHO, YIELD, SAFE_IDX, GROWTH_IDX)


if __name__ == "__main__":
    main()

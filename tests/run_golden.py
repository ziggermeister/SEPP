#!/usr/bin/env python3
# tests/run_golden.py
# Golden smoke test against a known param pack using the current engine.
# Assumes the pack universe order matches the arrays below.

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse, json, sys
from pathlib import Path
import numpy as np

import sepp_engine as eng  # uses your current engine

# --- Legacy portfolios in PACK universe order:
# Universe we've used to build the pack:
# SGOV, VGIT, BND, VWOB, SCHD, CDC, VIG, GLD, VTI, IEFA, VWO, QQQ, CHAT, IBIT, DGIN
PORTFOLIOS = {
    "1_Sliding_Window": np.array([
        0.100, 0.054, 0.090, 0.036, 0.133, 0.035, 0.019, 0.040, 0.256, 0.035,
        0.025, 0.054, 0.054, 0.030, 0.039
    ], dtype=float),
    "2_Suggested_Barbell": np.array([
        0.134, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.076, 0.453, 0.069,
        0.084, 0.053, 0.055, 0.053, 0.023
    ], dtype=float),
    "3_Current_Diversified": np.array([
        0.034, 0.075, 0.126, 0.050, 0.167, 0.048, 0.024, 0.042, 0.249, 0.038,
        0.046, 0.029, 0.030, 0.029, 0.013
    ], dtype=float),
}

# Loose golden expectations based on your pack_2024-12-31 run
GOLDEN = {
    "1_Sliding_Window": {"score": (94.0, 96.5), "liq": (1.4, 2.6), "ruin_max": 0.0025},
    "2_Suggested_Barbell": {"score": (81.0, 84.5), "liq": (0.0, 0.3), "ruin_max": 0.005},
    "3_Current_Diversified": {"score": (98.5, 100.5), "liq": (2.2, 3.1), "ruin_max": 0.005},
}

def load_pack(p):
    import pandas as pd
    P = json.loads(Path(p).read_text())
    tickers = P["universe"]
    mu  = np.array([P["mu"][t] for t in tickers], float)
    sig = np.array([P["sigma"][t] for t in tickers], float)
    yld = np.array([P["yield_rate"][t] for t in tickers], float)
    rho = pd.DataFrame(P["rho"]).loc[tickers, tickers].values.astype(float)
    return tickers, mu, sig, rho, yld, P

def safe_growth_idx(universe):
    safe = {"SGOV", "VGIT", "BND", "VWOB"}
    sidx = np.array([i for i, t in enumerate(universe) if t in safe], dtype=int)
    gidx = np.array([i for i in range(len(universe)) if i not in sidx], dtype=int)
    return sidx, gidx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to param pack json")
    args = ap.parse_args()

    # Load pack + set engine globals used by helpers
    assets, MU, SIG, RHO, YIELD, PACK = load_pack(args.params)
    sidx, gidx = safe_growth_idx(assets)

    # Apply engine knobs from your defaults (or override here if needed)
    # (We keep engine’s defaults; golden numbers were produced with them.)

    all_ok = True
    results = {}

    for name, w in PORTFOLIOS.items():
        # Run blended regimes and compute blended metrics via engine
        blended = None
        base_totals = base_safe = base_yld = base_egsp = None

        for regime, weight in eng.STRESS_BLEND.items():
            totals, safe_totals, safe_yld_eff, egsp_flags = eng.simulate_paths(
                eng.INITIAL_PORTFOLIO_VALUE, w, MU, SIG, RHO, YIELD,
                eng.YEARS, eng.ANNUAL_WITHDRAWAL, sidx, gidx, assets,
                n_sims=eng.N_SIM, seed=eng.SEED, regime=regime
            )
            if regime == "Base":
                base_totals, base_safe, base_yld, base_egsp = totals, safe_totals, safe_yld_eff, egsp_flags

            m = eng.compute_metrics(totals, safe_totals, safe_yld_eff, egsp_flags,
                                    eng.ANNUAL_WITHDRAWAL, eng.MIN_ACCEPTABLE_RETURN, eng.YEARS)
            if blended is None:
                blended = {k: 0.0 for k in m.keys()}
            for k, v in m.items():
                blended[k] += weight * v

        # Use engine’s internal weights for headline (same as live)
        WEIGHTS = {
            "Yrs1-4":  {"Ruin":0.20,"Liquidity":0.25,"Median_Return":0.125,"Upside":0.125,"CVaR":0.10,"Sortino":0.075,"Drawdown_Recovery":0.10,"Early_Sale":0.05,"Complexity":0.025},
            "Yrs5-8":  {"Ruin":0.175,"Liquidity":0.225,"Median_Return":0.15, "Upside":0.15, "CVaR":0.10,"Sortino":0.075,"Drawdown_Recovery":0.10,"Early_Sale":0.05,"Complexity":0.025},
            "Yrs9-12": {"Ruin":0.15, "Liquidity":0.20, "Median_Return":0.175,"Upside":0.20, "CVaR":0.10,"Sortino":0.075,"Drawdown_Recovery":0.10,"Early_Sale":0.05,"Complexity":0.025},
        }
        headline, _ = eng.composite_score(blended, eng.count_holdings(w), WEIGHTS)

        results[name] = {
            "score": float(headline),
            "liq": float(blended["Liquidity"]),
            "ruin": float(blended["Ruin"]),
        }

        # Compare to golden tolerances
        g = GOLDEN[name]
        ok = (g["score"][0] <= headline <= g["score"][1]) \
             and (g["liq"][0] <= blended["Liquidity"] <= g["liq"][1]) \
             and (blended["Ruin"] <= g["ruin_max"])
        all_ok = all_ok and ok

    # Report
    for k, v in results.items():
        print(f"{k:>20s}: score={v['score']:.1f}, liq={v['liq']:.2f}y, ruin={v['ruin']:.4f}")

    if all_ok:
        print("GOLDEN PASS")
        sys.exit(0)
    else:
        print("GOLDEN FAIL")
        sys.exit(2)

if __name__ == "__main__":
    main()
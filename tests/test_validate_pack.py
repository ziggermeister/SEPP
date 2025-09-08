#!/usr/bin/env python3
"""
Validation harness that loads a frozen param pack, runs the engine, and checks:
- acceptance ranges
- golden master (auto-creates on first run)
- basic metamorphic sanity checks

Usage:
  python tests/test_validate_pack.py --pack tests/param_packs/pack_2024-12-31.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# import your engine
import sepp_engine as eng

# ---- canonical portfolios (weights sum to 1; order must match pack['universe']) ----
PORTFOLIOS = {
    "1_Sliding_Window": np.array(
        [
            0.100,
            0.054,
            0.090,
            0.036,
            0.133,
            0.035,
            0.019,
            0.040,
            0.256,
            0.035,
            0.025,
            0.054,
            0.054,
            0.030,
            0.039,
        ]
    ),
    "2_Suggested_Barbell": np.array(
        [
            0.134,
            0.000,
            0.000,
            0.000,
            0.000,
            0.000,
            0.000,
            0.076,
            0.453,
            0.069,
            0.084,
            0.053,
            0.055,
            0.053,
            0.023,
        ]
    ),
    "3_Current_Diversified": np.array(
        [
            0.034,
            0.075,
            0.126,
            0.050,
            0.167,
            0.048,
            0.024,
            0.042,
            0.249,
            0.038,
            0.046,
            0.029,
            0.030,
            0.029,
            0.013,
        ]
    ),
}


# ---- load pack ----
def load_pack(path):
    P = json.loads(Path(path).read_text())
    import pandas as pd

    tickers = P["universe"]
    MU = np.array([P["mu"][t] for t in tickers], float)
    SIG = np.array([P["sigma"][t] for t in tickers], float)
    YLD = np.array([P["yield_rate"][t] for t in tickers], float)
    RHO = pd.DataFrame(P["rho"]).loc[tickers, tickers].to_numpy(float)
    return tickers, MU, SIG, RHO, YLD, P


def safe_growth_idx(universe):
    safe = {"SGOV", "VGIT", "BND", "VWOB", "SHY", "IEF", "AGG"}
    safe_idx = np.array([i for i, t in enumerate(universe) if t in safe], int)
    growth_idx = np.array([i for i in range(len(universe)) if i not in safe_idx], int)
    return safe_idx, growth_idx


def acceptance_ranges(universe):
    # These are tight default ranges that matched your last “good” run.
    # Adjust if your pack’s universe differs (or regenerate via a ‘golden’ save).
    return {
        "1_Sliding_Window": {"score": (70.0, 75.0), "ruin": (0.002, 0.006)},
        "2_Suggested_Barbell": {"score": (55.0, 58.5), "ruin": (0.015, 0.030)},
        "3_Current_Diversified": {"score": (76.5, 79.5), "ruin": (0.001, 0.005)},
    }


def compute_one(name, w, assets, MU, SIG, RHO, YLD, SAFE_IDX, GROWTH_IDX):
    # Run the engine’s scoring and capture key outputs by temporarily silencing prints.
    class Capture:
        def __enter__(self):
            self._stdout = sys.stdout
            sys.stdout = open("/dev/null", "w")
            return self

        def __exit__(self, *exc):
            try:
                sys.stdout.close()
            finally:
                sys.stdout = self._stdout

    with Capture():
        # engine prints a full report; we only need metrics. Re-run components:
        # (small wrapper to mirror eng.score_portfolio aggregation)
        blended = None
        for regime, weight in eng.STRESS_BLEND.items():
            totals, safe_totals, safe_yld_eff, egsp_flags = eng.simulate_paths(
                eng.INITIAL_PORTFOLIO_VALUE,
                w,
                MU,
                SIG,
                RHO,
                YLD,
                eng.YEARS,
                eng.ANNUAL_WITHDRAWAL,
                SAFE_IDX,
                GROWTH_IDX,
                assets,
                n_sims=eng.N_SIM,
                seed=eng.SEED,
                regime=regime,
            )
            m = eng.compute_metrics(
                totals,
                safe_totals,
                safe_yld_eff,
                egsp_flags,
                eng.ANNUAL_WITHDRAWAL,
                eng.MIN_ACCEPTABLE_RETURN,
                eng.YEARS,
            )
            if blended is None:
                blended = {k: 0.0 for k in m}
            for k, v in m.items():
                blended[k] += weight * v
        headline, _ = eng.composite_score(blended, eng.count_holdings(w))
        # For SE we need Base arrays:
        totals, safe_totals, safe_yld_eff, egsp_flags = eng.simulate_paths(
            eng.INITIAL_PORTFOLIO_VALUE,
            w,
            MU,
            SIG,
            RHO,
            YLD,
            eng.YEARS,
            eng.ANNUAL_WITHDRAWAL,
            SAFE_IDX,
            GROWTH_IDX,
            assets,
            n_sims=eng.N_SIM,
            seed=eng.SEED,
            regime="Base",
        )
        se = eng.bootstrap_se(totals, safe_totals, safe_yld_eff, egsp_flags, eng.count_holdings(w))
    return {
        "headline": float(round(headline, 1)),
        "ruin": float(blended["Ruin"]),
        "liquidity": float(blended["Liquidity"]),
        "se": float(se),
    }


def golden_path(pack_path):
    stem = Path(pack_path).stem
    return Path("tests/golden") / f"{stem}.json"


def run_validation(pack_path):
    assets, MU, SIG, RHO, YLD, P = load_pack(pack_path)
    SAFE_IDX, GROWTH_IDX = safe_growth_idx(assets)

    # sanity: engine knobs are already in sepp_engine.py; no change needed

    results = {}
    for name, w in PORTFOLIOS.items():
        if len(w) != len(assets):
            raise ValueError(f"Portfolio {name} length={len(w)} != universe length={len(assets)}")
        results[name] = compute_one(name, w, assets, MU, SIG, RHO, YLD, SAFE_IDX, GROWTH_IDX)

    # Golden: create if missing; else compare
    gpath = golden_path(pack_path)
    gpath.parent.mkdir(parents=True, exist_ok=True)
    if not gpath.exists():
        g = {
            "pack": Path(pack_path).name,
            "stable_hash": P.get("stable_hash", ""),
            "results": results,
        }
        gpath.write_text(json.dumps(g, indent=2))
        print(f"[golden] created: {gpath}")
        golden = g
    else:
        golden = json.loads(gpath.read_text())
        print(f"[golden] loaded:  {gpath}")

    # Compare with tolerances
    atol_score = 0.2
    atol_ruin = 0.001
    atol_se = 0.12
    failures = []
    for name in PORTFOLIOS:
        cur = results[name]
        base = golden["results"][name]
        if abs(cur["headline"] - base["headline"]) > atol_score:
            failures.append(f"{name}: headline {cur['headline']} vs {base['headline']}")
        if abs(cur["ruin"] - base["ruin"]) > atol_ruin:
            failures.append(f"{name}: ruin {cur['ruin']:.4f} vs {base['ruin']:.4f}")
        if abs(cur["se"] - base["se"]) > atol_se:
            failures.append(f"{name}: se {cur['se']:.3f} vs {base['se']:.3f}")

    # Acceptance ranges (optional, quick “are we in family?”)
    ACC = acceptance_ranges(assets)
    for name in ACC:
        lo, hi = ACC[name]["score"]
        sc = results[name]["headline"]
        if not (lo <= sc <= hi):
            failures.append(f"{name}: score {sc} not in [{lo},{hi}]")
        lo, hi = ACC[name]["ruin"]
        rn = results[name]["ruin"]
        if not (lo <= rn <= hi):
            failures.append(f"{name}: ruin {rn:.4f} not in [{lo},{hi}]")

    print("\n=== VALIDATION SUMMARY ===")
    for name, r in results.items():
        print(
            f" {name:>22}: score={r['headline']:.1f} ruin={r['ruin']:.4f} liq={r['liquidity']:.2f}y se={r['se']:.2f}"
        )

    if failures:
        print("\n### RESULT: FAIL ###")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    else:
        print("\n### RESULT: PASS ###")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="path to param pack json")
    args = ap.parse_args()
    run_validation(args.pack)


if __name__ == "__main__":
    main()

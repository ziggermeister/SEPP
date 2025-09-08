# validation_harness_pack.py
# Validation harness that:
# - Loads a params pack and prints metadata (as_of, stable_hash, sources, monthly_obs).
# - Runs acceptance checks, golden master, metamorphic invariants, bootstrap equivalence (tiny N).
# - Adds DRIFT GATES: if headline score deltas vs golden > threshold, flag.
#
# Usage:
#   python validation_harness_pack.py --params params/params_YYYY-MM-DD_<hash>.json

import argparse

# Import the engine (expects sepp_v6_4_3_pack.py in path)
import importlib.util
import sys
from datetime import datetime
from time import perf_counter

import numpy as np


def import_engine(path="sepp_v6_4_3_pack.py"):
    spec = importlib.util.spec_from_file_location("sepp", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sepp"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------- GOLDEN MASTER (based on your last good run) ---------------------
GOLDEN = {
    "1_Sliding_Window": {
        "blended": {
            "Ruin": 0.0030,
            "Liquidity": 1.918,
            "Median_Return": 0.060,
            "Upside": 0.158,
            "CVaR": -0.096,
            "Sortino": 0.92,
            "MDD": -0.253,
            "Calmar": 0.25,
            "Early_Sale": 0.000,
        },
        "headline": 72.3,
        "se": 0.72,
    },
    "2_Suggested_Barbell": {
        "blended": {
            "Ruin": 0.0223,
            "Liquidity": 0.000,
            "Median_Return": 0.061,
            "Upside": 0.181,
            "CVaR": -0.153,
            "Sortino": 0.56,
            "MDD": -0.328,
            "Calmar": 0.20,
            "Early_Sale": 0.002,
        },
        "headline": 56.8,
        "se": 0.93,
    },
    "3_Current_Diversified": {
        "blended": {
            "Ruin": 0.0021,
            "Liquidity": 2.702,
            "Median_Return": 0.060,
            "Upside": 0.153,
            "CVaR": -0.081,
            "Sortino": 1.02,
            "MDD": -0.232,
            "Calmar": 0.28,
            "Early_Sale": 0.000,
        },
        "headline": 78.1,
        "se": 0.90,
    },
}

ACCEPT = {
    "1_Sliding_Window": {
        "t0_liq_years": (5.4, 5.7),
        "liq_p10_med": (1.6, 2.6),
        "blended_liq": (1.7, 2.2),
        "ruin": (0.002, 0.006),
        "score": (70.0, 75.0),
        "se_y1_4": (0.5, 1.2),
    },
    "2_Suggested_Barbell": {
        "t0_liq_years": (2.3, 2.7),
        "liq_p10_med": (0.00, 0.10),
        "blended_liq": (-0.1, 0.1),
        "ruin": (0.015, 0.030),
        "score": (55.0, 58.5),
        "se_y1_4": (0.6, 1.3),
    },
    "3_Current_Diversified": {
        "t0_liq_years": (5.4, 6.0),
        "liq_p10_med": (2.6, 3.3),
        "blended_liq": (2.4, 3.0),
        "ruin": (0.001, 0.005),
        "score": (76.5, 79.5),
        "se_y1_4": (0.6, 1.3),
    },
}

# DRIFT gates (relative to GOLDEN)
DRIFT_THRESH = {
    "headline_abs": 1.5,  # headline score Δ must be ≤ 1.5 pts
    "ruin_abs": 0.003,  # ruin Δ ≤ 0.003
    "liq_abs": 0.30,  # blended liquidity Δ ≤ 0.30 years
}


# --------------------- Harness helpers ---------------------
def compute_blended_for_weights(
    sepp, w, assets, MU, SIG, RHO, YIELD_RATE, SAFE_IDX, GROWTH_IDX
):
    blended = None
    base_totals = base_safe = base_yld = base_egsp = None

    for regime, weight in sepp.STRESS_BLEND.items():
        totals, safe_totals, safe_yld_eff, egsp_flags = sepp.simulate_paths(
            sepp.INITIAL_PORTFOLIO_VALUE,
            w,
            MU,
            SIG,
            RHO,
            YIELD_RATE,
            sepp.YEARS,
            sepp.ANNUAL_WITHDRAWAL,
            SAFE_IDX,
            GROWTH_IDX,
            assets,
            n_sims=sepp.N_SIM,
            seed=sepp.SEED,
            regime=regime,
        )
        metrics = sepp.compute_metrics(
            totals,
            safe_totals,
            safe_yld_eff,
            egsp_flags,
            sepp.ANNUAL_WITHDRAWAL,
            sepp.MIN_ACCEPTABLE_RETURN,
            sepp.YEARS,
        )
        if blended is None:
            blended = {k: 0.0 for k in metrics.keys()}
        for k, v in metrics.items():
            blended[k] += weight * v
        if regime == "Base":
            base_totals, base_safe, base_yld, base_egsp = (
                totals,
                safe_totals,
                safe_yld_eff,
                egsp_flags,
            )

    headline, subs = sepp.composite_score(blended, sepp.count_holdings(w))
    se = sepp.bootstrap_se(
        base_totals, base_safe, base_yld, base_egsp, sepp.count_holdings(w)
    )

    # per-path liquidity stat for LIQ_METHOD on Base regime
    n_sims = base_safe.shape[0]
    liq_per_path = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        liq_per_path[i] = sepp.path_liquidity_stat(
            base_safe[i],
            base_yld[i],
            draw=sepp.ANNUAL_WITHDRAWAL,
            method=sepp.LIQ_METHOD,
            cap=sepp.LIQ_CAP,
        )
    liq_per_path = np.where(np.isfinite(liq_per_path), liq_per_path, 0.0)

    # t0 snapshot liquidity
    safe_w = float(np.sum(w[SAFE_IDX]))
    safe_init = sepp.INITIAL_PORTFOLIO_VALUE * safe_w
    safe_eff_yield = (
        float(np.sum((w[SAFE_IDX] / max(safe_w, 1e-12)) * YIELD_RATE[SAFE_IDX]))
        if safe_w > 0
        else 0.0
    )
    t0_liq = sepp.years_covered_forward(
        safe_init, safe_eff_yield, draw=sepp.ANNUAL_WITHDRAWAL, cap=50
    )

    return (
        blended,
        float(round(headline, 1)),
        subs,
        float(se),
        liq_per_path,
        float(t0_liq),
    )


def check_range(label, val, lo, hi, failures, pfx=""):
    ok = (val >= lo) and (val <= hi)
    print(
        f"  {pfx}{label:<18} {val:>7.3f}  in [{lo:.3f}, {hi:.3f}]  -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append((pfx + label, val, lo, hi))


def golden_master_check(result_by_name):
    failures = []
    for name, res in result_by_name.items():
        G = GOLDEN[name]
        # blended metrics
        for k, gv in G["blended"].items():
            v = res["blended"][k]
            ok = np.isclose(v, gv, atol=0.005, rtol=0.01)
            if not ok:
                failures.append(f"{name}: blended[{k}] {v:.6f} != {gv:.6f}")
        # headline
        if not np.isclose(res["headline"], G["headline"], atol=0.1, rtol=0.0):
            failures.append(
                f"{name}: headline {res['headline']:.3f} != {G['headline']:.3f}"
            )
        # se
        if not np.isclose(res["se"], G["se"], atol=0.15, rtol=0.1):
            failures.append(f"{name}: se {res['se']:.3f} != {G['se']:.3f}")
    print("\n=== GOLDEN MASTER CHECK ===")
    if failures:
        for f in failures:
            print("  -", f)
        print("### GOLDEN MASTER: FAIL ###")
    else:
        print("### GOLDEN MASTER: PASS ###")
    return not failures


def drift_gates_check(result_by_name):
    issues = []
    for name, res in result_by_name.items():
        G = GOLDEN[name]
        d_headline = abs(res["headline"] - G["headline"])
        d_ruin = abs(res["blended"]["Ruin"] - G["blended"]["Ruin"])
        d_liq = abs(res["blended"]["Liquidity"] - G["blended"]["Liquidity"])
        if d_headline > DRIFT_THRESH["headline_abs"]:
            issues.append(
                f"{name}: |Δheadline|={d_headline:.2f} > {DRIFT_THRESH['headline_abs']}"
            )
        if d_ruin > DRIFT_THRESH["ruin_abs"]:
            issues.append(f"{name}: |Δruin|={d_ruin:.4f} > {DRIFT_THRESH['ruin_abs']}")
        if d_liq > DRIFT_THRESH["liq_abs"]:
            issues.append(
                f"{name}: |Δliquidity|={d_liq:.2f} > {DRIFT_THRESH['liq_abs']}"
            )
    print("\n=== DRIFT GATES ===")
    if issues:
        for f in issues:
            print("  -", f)
        print("### DRIFT: FAIL (manual review) ###")
    else:
        print("### DRIFT: PASS ###")
    return not issues


def metamorphic_checks(
    sepp, assets, MU, SIG, RHO, YIELD_RATE, SAFE_IDX, GROWTH_IDX, w_baseline, n_sims=800
):
    # Run with baseline vs zero-withdraw vs doubled safe yield; check monotonicity
    def blended_for(draw, yld_override=None):
        yld = YIELD_RATE.copy()
        if yld_override is not None:
            yld = yld_override
        blended = None
        for regime, weight in sepp.STRESS_BLEND.items():
            totals, safe_totals, safe_yld_eff, egsp_flags = sepp.simulate_paths(
                sepp.INITIAL_PORTFOLIO_VALUE,
                w_baseline,
                MU,
                SIG,
                RHO,
                yld,
                sepp.YEARS,
                draw,
                SAFE_IDX,
                GROWTH_IDX,
                assets,
                n_sims=n_sims,
                seed=sepp.SEED,
                regime=regime,
            )
            m = sepp.compute_metrics(
                totals,
                safe_totals,
                safe_yld_eff,
                egsp_flags,
                draw,
                sepp.MIN_ACCEPTABLE_RETURN,
                sepp.YEARS,
            )
            if blended is None:
                blended = {k: 0.0 for k in m.keys()}
            for k, v in m.items():
                blended[k] += weight * v
        return blended

    base = blended_for(sepp.ANNUAL_WITHDRAWAL)
    zero = blended_for(0.0)
    yld2 = YIELD_RATE.copy()
    yld2[SAFE_IDX] = yld2[SAFE_IDX] * 2.0
    dy = blended_for(sepp.ANNUAL_WITHDRAWAL, yld2)

    ok1 = zero["Ruin"] <= base["Ruin"] + 1e-9
    ok2 = zero["Liquidity"] >= base["Liquidity"] - 1e-6
    ok3 = dy["Liquidity"] >= base["Liquidity"] - 1e-6
    # median_all >= p10_yr1_8 (on average)
    sepp.LIQ_METHOD = "median_all"
    totals, safe_totals, safe_yld_eff, egsp_flags = sepp.simulate_paths(
        sepp.INITIAL_PORTFOLIO_VALUE,
        w_baseline,
        MU,
        SIG,
        RHO,
        YIELD_RATE,
        sepp.YEARS,
        sepp.ANNUAL_WITHDRAWAL,
        SAFE_IDX,
        GROWTH_IDX,
        assets,
        n_sims=n_sims,
        seed=sepp.SEED,
        regime="Base",
    )
    n_s = safe_totals.shape[0]
    med_vals = np.array(
        [
            sepp.path_liquidity_stat(
                safe_totals[i],
                safe_yld_eff[i],
                draw=sepp.ANNUAL_WITHDRAWAL,
                method="median_all",
                cap=sepp.LIQ_CAP,
            )
            for i in range(n_s)
        ]
    )
    p10_vals = np.array(
        [
            sepp.path_liquidity_stat(
                safe_totals[i],
                safe_yld_eff[i],
                draw=sepp.ANNUAL_WITHDRAWAL,
                method="p10_yr1_8",
                cap=sepp.LIQ_CAP,
            )
            for i in range(n_s)
        ]
    )
    ok4 = np.mean(med_vals - p10_vals) >= -1e-9
    sepp.LIQ_METHOD = "p10_yr1_8"

    print("\n=== METAMORPHIC CHECKS ===")
    print(
        f"  Ruin baseline={base['Ruin']:.4f} vs zero-draw={zero['Ruin']:.4f}  (must decrease)"
    )
    print(
        f"  Liqu baseline={base['Liquidity']:.2f} vs zero-draw={zero['Liquidity']:.2f} (must increase)"
    )
    print(
        f"  Liqu baseline={base['Liquidity']:.2f} vs doubled-safe-yield={dy['Liquidity']:.2f} (must not decrease)"
    )
    print(
        f"  median_all - p10_yr1_8 (mean diff) = {np.mean(med_vals - p10_vals):.3f} (must be >= 0)"
    )
    print(
        "### METAMORPHIC:", "PASS ###" if (ok1 and ok2 and ok3 and ok4) else "FAIL ###"
    )
    return ok1 and ok2 and ok3 and ok4


def compare_bootstrap_methods(
    sepp, assets, MU, SIG, RHO, YIELD_RATE, SAFE_IDX, GROWTH_IDX, portfolios
):
    # Tiny N for speed
    _N_SIM = 300
    _B = 250
    out = {}
    print("\n=== BOOTSTRAP EQUIVALENCE (tiny N) ===")
    for name, w in portfolios.items():
        totals, safe_totals, safe_yld_eff, egsp_flags = sepp.simulate_paths(
            sepp.INITIAL_PORTFOLIO_VALUE,
            w,
            MU,
            SIG,
            RHO,
            YIELD_RATE,
            sepp.YEARS,
            sepp.ANNUAL_WITHDRAWAL,
            SAFE_IDX,
            GROWTH_IDX,
            assets,
            n_sims=_N_SIM,
            seed=sepp.SEED,
            regime="Base",
        )
        # fast
        se_fast = sepp.bootstrap_se(
            totals,
            safe_totals,
            safe_yld_eff,
            egsp_flags,
            sepp.count_holdings(w),
            n_resample=_B,
        )

        # slow
        def slow_bootstrap_se():
            rs = np.random.RandomState(sepp.SEED + 777)
            scores = []
            for _ in range(_B):
                idx = rs.choice(_N_SIM, size=_N_SIM, replace=True)
                m = sepp.compute_metrics(
                    totals[idx],
                    safe_totals[idx],
                    safe_yld_eff[idx],
                    egsp_flags[idx],
                    sepp.ANNUAL_WITHDRAWAL,
                    sepp.MIN_ACCEPTABLE_RETURN,
                    sepp.YEARS,
                )
                sc, _ = sepp.period_score(
                    m, sepp.count_holdings(w), sepp.WEIGHTS["Yrs1-4"]
                )
                scores.append(sc)
            return float(np.std(scores))

        se_slow = slow_bootstrap_se()
        diff = abs(se_fast - se_slow)
        out[name] = (se_fast, se_slow, diff)
        print(f"  {name}: fast={se_fast:.4f} slow={se_slow:.4f} | diff={diff:.4f}")
    ok = all(diff < 0.12 for (_, _, diff) in out.values())
    print("### BOOTSTRAP EQUIVALENCE:", "PASS ###" if ok else "FAIL ###")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to params_*.json")
    ap.add_argument("--engine", default="sepp_v6_4_3_pack.py")
    args = ap.parse_args()

    sepp = import_engine(args.engine)

    t0 = perf_counter()

    # Load params
    assets, MU, SIG, RHO, YIELD_RATE, PACK = sepp.load_param_pack(args.params)
    SAFE_IDX = np.array(
        [i for i, t in enumerate(assets) if t in ("SGOV", "VGIT", "BND", "VWOB")]
    )
    GROWTH_IDX = np.array([i for i in range(len(assets)) if i not in SAFE_IDX])

    # Print pack metadata
    print(
        f"PARAM PACK: as_of={PACK['as_of']} hash={PACK['stable_hash']} monthly_obs={PACK.get('monthly_obs')}"
    )
    print("SOURCES:", PACK.get("sources"))
    print("UNIVERSE:", ", ".join(assets))

    # Reproduce portfolio printouts (including liquidity summaries)
    for name, w in sepp.PORTFOLIOS.items():
        sepp.score_portfolio(
            name, w, assets, MU, SIG, RHO, YIELD_RATE, SAFE_IDX, GROWTH_IDX
        )

    # Programmatic validation on top
    print("\n=== VALIDATION RUN ===")
    all_failures = []
    result_by_name = {}
    for name, w in sepp.PORTFOLIOS.items():
        blended, headline, subs, se, liq_per_path, t0_liq = compute_blended_for_weights(
            sepp, w, assets, MU, SIG, RHO, YIELD_RATE, SAFE_IDX, GROWTH_IDX
        )
        result_by_name[name] = {"blended": blended, "headline": headline, "se": se}

        print(f"\n-- Portfolio: {name} --")
        print(f"  Snapshot t=0 Liquidity (years): {t0_liq:.2f}")
        print(
            f"  Base LIQ_METHOD median (per-path): {float(np.median(liq_per_path)):.2f}"
        )
        print(f"  Blended Liquidity (years): {blended['Liquidity']:.2f}")
        print(f"  Blended Ruin: {blended['Ruin']:.4f}")
        print(f"  Headline Score: {headline:.1f}")
        print(f"  Bootstrap SE (Yrs1-4): {se:.2f}")

        R = ACCEPT[name]
        check_range("t0_liq_years", t0_liq, *R["t0_liq_years"], all_failures)
        check_range(
            "liq_p10_med",
            float(np.median(liq_per_path)),
            *R["liq_p10_med"],
            all_failures,
        )
        check_range(
            "blended_liq", blended["Liquidity"], *R["blended_liq"], all_failures
        )
        check_range("ruin", blended["Ruin"], *R["ruin"], all_failures)
        check_range("score", headline, *R["score"], all_failures)
        check_range("se_y1_4", se, *R["se_y1_4"], all_failures)

    end = datetime.now()
    t1 = perf_counter()
    print(f"\nEnd:   {end}")
    print(f"Total: {(t1 - t0):.2f} sec")
    print("\n### VALIDATION RESULT:", "PASS ###" if not all_failures else "FAIL ###")
    if all_failures:
        for lab, val, lo, hi in all_failures:
            print(f"  - {lab} = {val:.4f} not in [{lo:.4f}, {hi:.4f}]")

    # Golden master + drift gates + metamorphic + bootstrap equivalence
    golden_master_check(result_by_name)
    drift_gates_check(result_by_name)
    metamorphic_checks(
        sepp,
        assets,
        MU,
        SIG,
        RHO,
        YIELD_RATE,
        SAFE_IDX,
        GROWTH_IDX,
        list(sepp.PORTFOLIOS.values())[2],
        n_sims=800,
    )
    compare_bootstrap_methods(
        sepp, assets, MU, SIG, RHO, YIELD_RATE, SAFE_IDX, GROWTH_IDX, sepp.PORTFOLIOS
    )


if __name__ == "__main__":
    main()

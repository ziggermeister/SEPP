# sepp_v6_4_3_pack.py
# SEPP Composite Score Framework - Version 6.4.3 (Fast Bootstrap, Param-Pack)
# - Loads MU/SIG/RHO/YIELD_RATE from a versioned parameter pack JSON.
# - Logs pack as_of, stable_hash, and sources.
# - Fix: liquidity_distribution_stats prints the correct LIQ_METHOD label.
# - Minor modularity in simulate_paths (overlay builder factored).

import json, argparse
from datetime import datetime
from time import perf_counter

import numpy as np
from numpy.linalg import eigh
from scipy.stats import t as student_t, norm

np.seterr(all='ignore')

# --------------------------- Flags -------------------------------------
DEBUG_PRINT_ENV = True
DEBUG_LIQ_STATS = True
DEBUG_SAMPLE_PATHS = False

YEARS = 12
ANNUAL_WITHDRAWAL = 42000.0
INITIAL_PORTFOLIO_VALUE = 726399.79
MIN_ACCEPTABLE_RETURN = 0.02
N_SIM = 2000
SEED = 42
BOOTSTRAP_RESAMPLES = 800
STRESS_BLEND = {"Base": 0.6, "Front": 0.2, "Prolonged": 0.2}

# Liquidity method
LIQ_METHOD = "p10_yr1_8"   # "min_all", "median_all", "p10_yr1_8"
LIQ_CAP = 50

# --------------------------- Portfolios (weights sum to 1) ---------------------
PORTFOLIOS = {
    "1_Sliding_Window": np.array([0.100, 0.054, 0.090, 0.036, 0.133, 0.035, 0.019, 0.040,
                                  0.256, 0.035, 0.025, 0.054, 0.054, 0.030, 0.039]),
    "2_Suggested_Barbell": np.array([0.134, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.076,
                                     0.453, 0.069, 0.084, 0.053, 0.055, 0.053, 0.023]),
    "3_Current_Diversified": np.array([0.034, 0.075, 0.126, 0.050, 0.167, 0.048, 0.024, 0.042,
                                       0.249, 0.038, 0.046, 0.029, 0.030, 0.029, 0.013])
}

# Period weights
WEIGHTS = {
    "Yrs1-4": {"Ruin": 0.20, "Liquidity": 0.25, "Median_Return": 0.125, "Upside": 0.125,
               "CVaR": 0.10, "Sortino": 0.075, "Drawdown_Recovery": 0.10, "Early_Sale": 0.05,
               "Complexity": 0.025},
    "Yrs5-8": {"Ruin": 0.175, "Liquidity": 0.225, "Median_Return": 0.15, "Upside": 0.15,
               "CVaR": 0.10, "Sortino": 0.075, "Drawdown_Recovery": 0.10, "Early_Sale": 0.05,
               "Complexity": 0.025},
    "Yrs9-12": {"Ruin": 0.15, "Liquidity": 0.20, "Median_Return": 0.175, "Upside": 0.20,
                "CVaR": 0.10, "Sortino": 0.075, "Drawdown_Recovery": 0.10, "Early_Sale": 0.05,
                "Complexity": 0.025}
}

# --------------------------- Utilities -----------------------------------------
def load_param_pack(path):
    import pandas as pd
    with open(path) as f:
        P = json.load(f)
    tickers = P["universe"]
    mu = np.array([P["mu"][t] for t in tickers], float)
    sig = np.array([P["sigma"][t] for t in tickers], float)
    yld = np.array([P["yield_rate"][t] for t in tickers], float)
    rho = pd.DataFrame(P["rho"]).loc[tickers, tickers].values.astype(float)
    return tickers, mu, sig, rho, yld, P

def higham_psd(A):
    A = (A + A.T) / 2.0
    w, V = eigh(A)
    w = np.maximum(w, 0.0)
    return V @ np.diag(w) @ V.T

def gaussian_copula_t_draws(n_sims, n_years, mu, sig, rho, df=5, seed=SEED):
    rs = np.random.RandomState(seed)
    n_assets = len(mu)
    eps_scale = 0.25 / np.sqrt(max(n_sims, 1))

    chol_list = []
    for _ in range(n_sims):
        eps = rs.normal(0, eps_scale, (n_assets, n_assets))
        eps = (eps + eps.T) / 2.0
        rho_i = np.clip(rho + eps, -1.0, 1.0)
        rho_i = higham_psd(rho_i)
        try:
            L = np.linalg.cholesky(rho_i + 1e-12 * np.eye(n_assets))
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(higham_psd(rho_i) + 1e-10 * np.eye(n_assets))
        chol_list.append(L)

    Z = rs.normal(size=(n_sims, n_years, n_assets))
    shocks = np.empty_like(Z)
    for i in range(n_sims):
        Zc = Z[i] @ chol_list[i].T
        U = norm.cdf(Zc)
        U = np.clip(U, 1e-12, 1.0 - 1e-12)
        Tq = student_t.ppf(U, df=df)
        Tq = np.where(np.isfinite(Tq), Tq, 0.0)
        t_std = Tq / np.sqrt(df / (df - 2))
        shocks[i] = mu + t_std * sig
    return np.where(np.isfinite(shocks), shocks, 0.0)

def years_covered_forward(safe_balance, yield_rate, draw=ANNUAL_WITHDRAWAL, cap=50):
    bal = max(float(safe_balance), 0.0)
    y = max(float(yield_rate), 0.0)
    if not np.isfinite(bal) or not np.isfinite(y): return np.nan
    years = 0.0
    for _ in range(cap):
        income = bal * y
        if not np.isfinite(income): return np.nan
        if income >= draw: return float(cap)
        if bal + income <= draw:
            return years + (bal + income) / draw
        bal = bal + income - draw
        years += 1.0
    return years

def count_holdings(w, tol=1e-6): return int(np.sum(w > tol))

# --------------------------- Simulation -----------------------------------------
def build_overlay(years, n_assets, growth_idx, regime):
    overlay = np.zeros((years, n_assets), dtype=float)
    if regime == "Front":
        overlay[0:2, growth_idx] += -0.20
    elif regime == "Prolonged":
        seq = np.array([-0.05, 0.00, -0.03, 0.00], dtype=float)
        T = min(years, len(seq))
        overlay[0:T, growth_idx] += seq[:T, None]
    return overlay

def simulate_paths(initial_value, weights, mu, sig, rho, yld, years,
                   annual_withdrawal, safe_idx, growth_idx, assets,
                   n_sims=N_SIM, seed=SEED, regime="Base"):
    rs = np.random.RandomState(seed)
    n_assets = len(weights)
    values = np.zeros((n_sims, years + 1, n_assets), dtype=float)
    values[:, 0, :] = initial_value * weights
    safe_totals = np.zeros((n_sims, years + 1), dtype=float)
    safe_yld_eff = np.zeros((n_sims, years + 1), dtype=float)
    egsp_flags = np.zeros(n_sims, dtype=bool)

    # initial safe sleeve state
    safe_prev_by_asset = values[:, 0, :][:, safe_idx]
    safe_prev_total = np.sum(safe_prev_by_asset, axis=1)
    safe_totals[:, 0] = np.where(np.isfinite(safe_prev_total), safe_prev_total, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        wts0 = np.where(safe_prev_total[:, None] > 0,
                        safe_prev_by_asset / np.maximum(safe_prev_total[:, None], 1e-12),
                        0.0)
        eff0 = (wts0 * yld[safe_idx]).sum(axis=1)
        safe_yld_eff[:, 0] = np.where(np.isfinite(eff0), eff0, 0.0)

    shocks = gaussian_copula_t_draws(n_sims, years, mu, sig, rho, df=5, seed=seed)
    overlay = build_overlay(years, n_assets, growth_idx, regime)

    # withdrawal order: all safe first, then growth (by given asset order)
    cascade_idx = list(safe_idx) + list(growth_idx)

    for t in range(1, years + 1):
        prev = values[:, t - 1, :]
        total_r = shocks[:, t - 1, :]
        price_r = total_r - yld
        if regime in ("Front", "Prolonged"):
            price_r = price_r + overlay[t - 1, :]

        price_r = np.where(np.isfinite(price_r), price_r, 0.0)
        after_price = np.where(np.isfinite(prev * (1.0 + price_r)), prev * (1.0 + price_r), 0.0)
        income = np.where(np.isfinite(prev * yld), prev * yld, 0.0)
        values[:, t, :] = after_price + income

        income_total = np.where(np.isfinite(income.sum(axis=1)), income.sum(axis=1), 0.0)
        principal_to_withdraw = np.maximum(0.0, annual_withdrawal - income_total)

        # cascade
        for aidx in cascade_idx:
            bal = np.where(np.isfinite(values[:, t, aidx]), values[:, t, aidx], 0.0)
            draw_amt = np.minimum(principal_to_withdraw, bal)
            values[:, t, aidx] = bal - draw_amt
            principal_to_withdraw = principal_to_withdraw - draw_amt
            if not np.any(principal_to_withdraw > 1e-9):
                break

        values[:, t, :] = np.maximum(np.where(np.isfinite(values[:, t, :]), values[:, t, :], 0.0), 0.0)

        # track safe sleeve
        safe_now = values[:, t, :][:, safe_idx]
        safe_sum = np.where(np.isfinite(np.sum(safe_now, axis=1)), np.sum(safe_now, axis=1), 0.0)
        safe_totals[:, t] = safe_sum
        with np.errstate(divide='ignore', invalid='ignore'):
            wts = np.where(safe_sum[:, None] > 0, safe_now / np.maximum(safe_sum[:, None], 1e-12), 0.0)
            eff = (wts * yld[safe_idx]).sum(axis=1)
            safe_yld_eff[:, t] = np.where(np.isfinite(eff), eff, 0.0)

        if t <= 3:
            egsp_flags = egsp_flags | (safe_sum <= 1e-9)

    totals = values.sum(axis=2)
    if DEBUG_SAMPLE_PATHS:
        print("DEBUG totals[0,:6] ->", totals[0, :6])
    return totals, safe_totals, safe_yld_eff, egsp_flags

# --------------------------- Liquidity helpers ----------------------------------
def path_liquidity_stat(safe_path, yld_path, draw, method=LIQ_METHOD, cap=LIQ_CAP):
    T = safe_path.shape[0]
    liq_series = np.empty(T-1, dtype=float)
    for t in range(1, T):
        liq_series[t-1] = years_covered_forward(safe_path[t], yld_path[t], draw=draw, cap=cap)
    if method == "min_all":
        return float(np.min(liq_series))
    elif method == "median_all":
        return float(np.median(liq_series))
    elif method == "p10_yr1_8":
        hi = min(T-1, 8)
        window = liq_series[:hi]
        return float(np.percentile(window, 10))
    return float(np.min(liq_series))

def liquidity_distribution_stats(safe_totals, safe_yld_eff, annual_withdrawal=ANNUAL_WITHDRAWAL, cap=50, method=LIQ_METHOD):
    n_sims, T = safe_totals.shape
    liq_per_path = np.zeros(n_sims, dtype=float)
    for i in range(n_sims):
        liq_min = np.inf
        for t in range(1, T):
            liq_t = years_covered_forward(safe_totals[i, t], safe_yld_eff[i, t], draw=annual_withdrawal, cap=cap)
            if liq_t < liq_min:
                liq_min = liq_t
        liq_per_path[i] = 0.0 if not np.isfinite(liq_min) else liq_min
    print(
        f"  Liquidity (method={method}) summary: "
        f"med={np.median(liq_per_path):.2f}y "
        f"p25={np.percentile(liq_per_path,25):.2f}y "
        f"p75={np.percentile(liq_per_path,75):.2f}y "
        f"min={np.min(liq_per_path):.2f}y "
        f"max={np.max(liq_per_path):.2f}y "
        f"zeros={np.sum(liq_per_path==0)}"
    )
    return liq_per_path

# --------------------------- Metrics & Scoring ----------------------------------
def per_path_max_drawdown(series):
    peak = np.maximum.accumulate(series)
    dd = (series - peak) / np.maximum(peak, 1e-12)
    return np.min(dd)

def compute_metrics(total_values, safe_totals, safe_yld_eff, egsp_flags,
                    annual_withdrawal=ANNUAL_WITHDRAWAL, min_acceptable_return=MIN_ACCEPTABLE_RETURN,
                    years=YEARS):
    n_sims, _ = total_values.shape
    ruin = float(np.mean(np.any(total_values[:, 1:] <= 1e-6, axis=1)))

    liq_per_path = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        liq_per_path[i] = path_liquidity_stat(safe_totals[i], safe_yld_eff[i], draw=annual_withdrawal, method=LIQ_METHOD, cap=LIQ_CAP)
    liq_per_path = np.where(np.isfinite(liq_per_path), liq_per_path, 0.0)
    liquidity = float(np.median(liq_per_path))

    ok = total_values[:, -1] > 1e-6
    if np.sum(ok) < max(1, int(0.05 * n_sims)):
        return {"Ruin": ruin, "Liquidity": max(0.0, liquidity),
                "Median_Return": -1.0, "Upside": -1.0, "CVaR": -1.0, "Sortino": 0.0,
                "MDD": -1.0, "Calmar": 0.0, "Early_Sale": float(np.mean(egsp_flags))}

    tv_ok = total_values[ok]
    ann_log_ret = np.log(tv_ok[:, -1] / tv_ok[:, 0]) / years
    median_ret = float(np.median(ann_log_ret))
    upside_p95 = float(np.percentile(ann_log_ret, 95))
    k = max(1, int(0.05 * len(ann_log_ret)))
    cvar = float(np.mean(np.sort(ann_log_ret)[:k]))
    MAR = min_acceptable_return
    downside = ann_log_ret[ann_log_ret < MAR]
    downside_dev = float(np.std(downside)) if downside.size > 0 else 0.0
    sortino = float((np.mean(ann_log_ret) - MAR) / downside_dev) if downside_dev > 0 else 2.0

    mdd_vals = np.array([per_path_max_drawdown(tv_ok[i, :]) for i in range(tv_ok.shape[0])])
    mdd_avg = float(np.mean(mdd_vals))
    calmar = float(median_ret / -mdd_avg) if mdd_avg < 0 else 0.0

    return {"Ruin": ruin, "Liquidity": max(0.0, liquidity),
            "Median_Return": median_ret, "Upside": upside_p95,
            "CVaR": cvar, "Sortino": sortino, "MDD": mdd_avg, "Calmar": calmar,
            "Early_Sale": float(np.mean(egsp_flags))}

def norm_ruin(r):
    r = min(max(r, 0.0), 0.20); return 100.0 * (1.0 - r / 0.20)
def norm_liquidity(years):
    y = max(0.0, years)
    return 90.0 * (y / 3.0) if y <= 3.0 else 90.0 + 10.0 * min((y - 3.0) / 2.0, 1.0)
def norm_linear_floor_ceiling(x, lo, hi):
    if x <= lo: return 0.0
    if x >= hi: return 100.0
    return 100.0 * (x - lo) / (hi - lo)
def norm_median(ret): return norm_linear_floor_ceiling(ret, 0.03, 0.10)
def norm_upside(ret): return 100.0 * min(np.log(max(ret,0.06)/0.06)/np.log(20.0/6.0), 1.0)
def norm_cvar(es):    return 100.0 * min((0.50 + es) / 0.40, 1.0)
def norm_sortino(s):  return 100.0 * min(max(s, 0.0) / 2.0, 1.0)
def norm_mdd(mdd):    return 100.0 * min((0.50 + mdd) / 0.40, 1.0)
def norm_calmar(c):   return 100.0 * min(max(c, 0.0) / 2.0, 1.0)
def norm_egsp(p):     return 100.0 * (1.0 - min(max(p, 0.0) / 0.50, 1.0))
def norm_complexity(n_holdings):
    h = max(int(n_holdings), 5)
    score = 100.0 * (1.0 - np.log(h / 5.0) / np.log(25.0 / 5.0))
    return float(np.clip(score, 0.0, 100.0))

def period_score(metrics, n_holdings, weights):
    mdd_s = norm_mdd(metrics["MDD"])
    cal_s = norm_calmar(metrics["Calmar"])
    if metrics["MDD"] > -0.10: cal_s = min(cal_s, mdd_s)
    draw_rec = 0.5 * (mdd_s + cal_s)
    parts = {
        "Ruin": norm_ruin(metrics["Ruin"]),
        "Liquidity": norm_liquidity(metrics["Liquidity"]),
        "Median_Return": norm_median(metrics["Median_Return"]),
        "Upside": norm_upside(metrics["Upside"]),
        "CVaR": norm_cvar(metrics["CVaR"]),
        "Sortino": norm_sortino(metrics["Sortino"]),
        "Drawdown_Recovery": draw_rec,
        "Early_Sale": norm_egsp(metrics["Early_Sale"]),
        "Complexity": norm_complexity(n_holdings),
    }
    score = sum(weights[k] * parts[k] for k in weights.keys())
    if metrics["Ruin"] > 0.25: score = min(score, 50.0)
    return score, parts

def composite_score(metrics, n_holdings):
    s1, _ = period_score(metrics, n_holdings, WEIGHTS["Yrs1-4"])
    s2, _ = period_score(metrics, n_holdings, WEIGHTS["Yrs5-8"])
    s3, _ = period_score(metrics, n_holdings, WEIGHTS["Yrs9-12"])
    headline = (s1 + s2 + s3) / 3.0
    return headline, {"Yrs1-4": round(s1, 1), "Yrs5-8": round(s2, 1), "Yrs9-12": round(s3, 1)}

# --------- Fast bootstrap (precompute per-path stats) ---------------------------
def _precompute_path_stats(total_values, safe_totals, safe_yld_eff, egsp_flags,
                           annual_withdrawal=ANNUAL_WITHDRAWAL, years=YEARS,
                           min_acceptable_return=MIN_ACCEPTABLE_RETURN):
    n_sims, _ = total_values.shape
    ruin_i = np.any(total_values[:, 1:] <= 1e-6, axis=1).astype(float)
    liq_i = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        liq_i[i] = path_liquidity_stat(safe_totals[i], safe_yld_eff[i], draw=annual_withdrawal, method=LIQ_METHOD, cap=LIQ_CAP)
    liq_i = np.where(np.isfinite(liq_i), liq_i, 0.0)
    survive = (total_values[:, -1] > 1e-6)
    with np.errstate(divide='ignore', invalid='ignore'):
        ann_log_ret_i = np.log(total_values[:, -1] / total_values[:, 0]) / years
    mdd_i = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        mdd_i[i] = per_path_max_drawdown(total_values[i, :]) if survive[i] else np.nan
    return {"ruin":ruin_i, "liq":liq_i, "egsp":egsp_flags.astype(float),
            "survive":survive, "ann_log_ret":ann_log_ret_i, "mdd":mdd_i, "MAR":min_acceptable_return}

def bootstrap_se(total_values, safe_totals, safe_yld_eff, egsp_flags,
                 n_holdings, n_resample=BOOTSTRAP_RESAMPLES):
    rs = np.random.RandomState(SEED + 123)
    s = _precompute_path_stats(total_values, safe_totals, safe_yld_eff, egsp_flags)
    ruin_i, liq_i, egsp_i, survive, ann_log_ret, mdd_i, MAR = s["ruin"], s["liq"], s["egsp"], s["survive"], s["ann_log_ret"], s["mdd"], s["MAR"]
    n = len(ruin_i)
    scores = np.empty(n_resample, dtype=float)
    for b in range(n_resample):
        idx = rs.choice(n, size=n, replace=True)
        ruin_b = float(np.mean(ruin_i[idx]))
        liq_b  = float(np.median(liq_i[idx]))
        egsp_b = float(np.mean(egsp_i[idx]))
        surv_idx = idx[survive[idx]]
        if surv_idx.size < max(1, int(0.05 * n)):
            metrics_b = {"Ruin": ruin_b, "Liquidity": liq_b, "Median_Return": -1.0, "Upside": -1.0,
                         "CVaR": -1.0, "Sortino": 0.0, "MDD": -1.0, "Calmar": 0.0, "Early_Sale": egsp_b}
        else:
            r = ann_log_ret[surv_idx]
            med_r = float(np.median(r))
            up95 = float(np.percentile(r, 95))
            k = max(1, int(0.05 * r.size))
            cvar = float(np.mean(np.sort(r)[:k]))
            downside = r[r < MAR]
            downside_dev = float(np.std(downside)) if downside.size > 0 else 0.0
            sortino = float((np.mean(r) - MAR) / downside_dev) if downside_dev > 0 else 2.0
            mdd_b = float(np.nanmean(mdd_i[surv_idx]))
            calmar = float(med_r / -mdd_b) if mdd_b < 0 else 0.0
            metrics_b = {"Ruin": ruin_b, "Liquidity": liq_b, "Median_Return": med_r, "Upside": up95,
                         "CVaR": cvar, "Sortino": sortino, "MDD": mdd_b, "Calmar": calmar, "Early_Sale": egsp_b}
        sc_b, _ = period_score(metrics_b, n_holdings, WEIGHTS["Yrs1-4"])
        scores[b] = sc_b
    return float(np.std(scores))

# --------------------------- Scoring Wrapper ------------------------------------
def score_portfolio(name, w, assets, mu, sig, rho, yld, safe_idx, growth_idx):
    n_hold = count_holdings(w)
    # t=0 sanity
    safe_w = float(np.sum(w[safe_idx]))
    safe_init_balance = INITIAL_PORTFOLIO_VALUE * safe_w
    safe_eff_yield = float(np.sum((w[safe_idx] / max(safe_w, 1e-12)) * yld[safe_idx])) if safe_w > 0 else 0.0
    liq_t0 = years_covered_forward(safe_init_balance, safe_eff_yield, draw=ANNUAL_WITHDRAWAL, cap=50)
    print(f"  SNAPSHOT Liquidity(t=0): safe_w={safe_w:.3f}, safe_init=${safe_init_balance:,.0f}, yld_eff={safe_eff_yield:.2%}, years≈{liq_t0:.2f}")

    blended = None
    base_totals = base_safe = base_yld = base_egsp = None

    for regime, weight in STRESS_BLEND.items():
        totals, safe_totals, safe_yld_eff, egsp_flags = simulate_paths(
            INITIAL_PORTFOLIO_VALUE, w, mu, sig, rho, yld, YEARS,
            ANNUAL_WITHDRAWAL, safe_idx, growth_idx, assets, n_sims=N_SIM, seed=SEED, regime=regime
        )
        if regime == "Base" and DEBUG_LIQ_STATS:
            base_totals, base_safe, base_yld, base_egsp = totals, safe_totals, safe_yld_eff, egsp_flags
            _ = liquidity_distribution_stats(base_safe, base_yld, ANNUAL_WITHDRAWAL, cap=50, method=LIQ_METHOD)
        m = compute_metrics(totals, safe_totals, safe_yld_eff, egsp_flags, ANNUAL_WITHDRAWAL, MIN_ACCEPTABLE_RETURN, YEARS)
        if blended is None: blended = {k: 0.0 for k in m.keys()}
        for k, v in m.items(): blended[k] += weight * v

    headline, subs = composite_score(blended, n_hold)
    se = bootstrap_se(base_totals, base_safe, base_yld, base_egsp, n_hold)

    print(f"\n=== {name} ===")
    print(f"Holdings: {n_hold} | Headline Score: {round(headline,1)} ±4 (SE≈{se:.2f})")
    print(f"Period sub-scores: {{ Yrs1-4: {round(subs['Yrs1-4'],1)}, Yrs5-8: {round(subs['Yrs5-8'],1)}, Yrs9-12: {round(subs['Yrs9-12'],1)} }}")
    print("Blended Metrics (Base 60%, Front 20%, Prol 20%):")
    print(f" Ruin={blended['Ruin']:.3f} Liquidity={blended['Liquidity']:.2f}y MedRet={blended['Median_Return']:.3f} "
          f"Upside={blended['Upside']:.3f} CVaR={blended['CVaR']:.3f} Sortino={blended['Sortino']:.2f} "
          f"MDD={blended['MDD']:.3f} Calmar={blended['Calmar']:.2f} EGSP={blended['Early_Sale']:.3f}")

# --------------------------- Unit Tests ----------------------------------------
def test_normalization():
    assert abs(norm_ruin(0.02) - 90.0) < 1e-6
    assert abs(norm_liquidity(5) - 100.0) < 1e-6
    assert abs(norm_upside(0.20) - 100.0) < 1e-6
    assert abs(norm_cvar(-0.30) - 50.0) < 1e-6
    assert abs(norm_egsp(0.50) - 0.0) < 1e-6

# --------------------------- Main ----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="Path to params_*.json")
    args = ap.parse_args()

    start_ts = datetime.now(); t0 = perf_counter()
    if DEBUG_PRINT_ENV:
        import numpy, scipy
        print(f"NumPy {numpy.__version__} | SciPy {scipy.__version__}")

    assets, MU, SIG, RHO, YIELD_RATE, PACK = load_param_pack(args.params)
    SAFE_IDX = np.array([i for i, t in enumerate(assets) if t in ("SGOV","VGIT","BND","VWOB")])
    GROWTH_IDX = np.array([i for i in range(len(assets)) if i not in SAFE_IDX])

    print(f"PARAM PACK: as_of={PACK['as_of']} hash={PACK['stable_hash']} source_prices={PACK['sources']['prices']}")
    print(f"UNIVERSE: {', '.join(assets)}")

    test_normalization()
    # Ensure portfolio weight vectors match current universe order
    # (Assumes the 15 assets are the same and in the same order used to define the portfolios.)
    for name, w in PORTFOLIOS.items():
        score_portfolio(name, w, assets, MU, SIG, RHO, YIELD_RATE, SAFE_IDX, GROWTH_IDX)

    t1 = perf_counter(); end_ts = datetime.now()
    print("\n--- Execution Time ---")
    print(f"Start: {start_ts}")
    print(f"End:   {end_ts}")
    print(f"Total: {t1 - t0:.2f} seconds")

if __name__ == "__main__":
    main()

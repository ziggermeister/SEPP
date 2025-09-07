#!/usr/bin/env python3
# sepp_engine.py
# SEPP Composite Score Framework - Version 6.4.3 (Fast Bootstrap; clamped norms)
# - Correct LIQ_METHOD label in liquidity_distribution_stats print
# - build_overlay factored out of simulate_paths
# - All normalization functions hard-clamped to [0, 100]
# - Period and Headline scores hard-clamped to [0, 100]

import numpy as np
from numpy.linalg import eigh
from scipy.stats import t as student_t, norm

# --------------------------- Flags / Params (can be patched by harness) --------
DEBUG_PRINT_ENV   = False
DEBUG_LIQ_STATS   = True
DEBUG_SAMPLE_PATHS = False

YEARS                   = 12
ANNUAL_WITHDRAWAL       = 42000.0
INITIAL_PORTFOLIO_VALUE = 726_399.79
MIN_ACCEPTABLE_RETURN   = 0.02
N_SIM                   = 2000
SEED                    = 42
BOOTSTRAP_RESAMPLES     = 800
STRESS_BLEND            = {"Base": 0.6, "Front": 0.2, "Prolonged": 0.2}

# Liquidity method & cap
LIQ_METHOD = "p10_yr1_8"  # one of: "min_all", "median_all", "p10_yr1_8"
LIQ_CAP    = 50

# Will be set by harness / live wire
ASSETS      = []
MU          = np.array([])
SIG         = np.array([])
RHO         = np.array([[]])
YIELD_RATE  = np.array([])
SAFE_IDX    = np.array([], dtype=int)
GROWTH_IDX  = np.array([], dtype=int)

np.seterr(all="ignore")


# --------------------------- Utilities -----------------------------------------
def higham_psd(A: np.ndarray) -> np.ndarray:
    """Project a (nearly) SPD matrix to the nearest PSD."""
    A = (A + A.T) / 2.0
    w, V = eigh(A)
    w = np.maximum(w, 0.0)
    return V @ np.diag(w) @ V.T


def gaussian_copula_t_draws(n_sims, n_years, mu, sig, rho, df=5, seed=SEED):
    """Student-t marginals with Gaussian copula and slight per-sim rho jitter."""
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


def years_covered_forward(safe_balance, yield_rate, draw=ANNUAL_WITHDRAWAL, cap=LIQ_CAP):
    """Forward-drain years covered by the safe sleeve using only its own yield."""
    bal = max(float(safe_balance), 0.0)
    y   = max(float(yield_rate), 0.0)
    if not np.isfinite(bal) or not np.isfinite(y): return np.nan
    years = 0.0
    for _ in range(cap):
        income = bal * y
        if not np.isfinite(income): return np.nan
        if income >= draw:  # sustained by yield alone
            return float(cap)
        if bal + income <= draw:
            return years + (bal + income) / draw
        bal = bal + income - draw
        years += 1.0
    return years


def count_holdings(w, tol=1e-6):
    return int(np.sum(w > tol))


# --------------------------- Simulation ----------------------------------------
def build_overlay(years, n_assets, growth_idx, regime):
    """Stress overlays: Front=early large shock; Prolonged=drip bear."""
    overlay = np.zeros((years, n_assets), dtype=float)
    if regime == "Front":
        overlay[0:2, growth_idx] += -0.20
    elif regime == "Prolonged":
        seq = np.array([-0.05, 0.00, -0.03, 0.00], dtype=float)
        T = min(years, len(seq))
        overlay[0:T, growth_idx] += seq[:T, None]
    return overlay


def simulate_paths(initial_value,
                   weights,
                   mu,
                   sig,
                   rho,
                   yld,
                   years,
                   annual_withdrawal,
                   safe_idx,
                   growth_idx,
                   assets,
                   n_sims=N_SIM,
                   seed=SEED,
                   regime="Base"):
    """Path simulator with safe-first withdrawal cascade."""
    rs = np.random.RandomState(seed)
    n_assets = len(weights)

    values       = np.zeros((n_sims, years + 1, n_assets), dtype=float)
    values[:, 0, :] = initial_value * weights
    safe_totals  = np.zeros((n_sims, years + 1), dtype=float)
    safe_yld_eff = np.zeros((n_sims, years + 1), dtype=float)
    egsp_flags   = np.zeros(n_sims, dtype=bool)

    # t=0 safe sleeve
    safe_prev_by_asset = values[:, 0, :][:, safe_idx]
    safe_prev_total = np.sum(safe_prev_by_asset, axis=1)
    safe_totals[:, 0] = np.where(np.isfinite(safe_prev_total), safe_prev_total, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        wts0 = np.where(
            safe_prev_total[:, None] > 0,
            safe_prev_by_asset / np.maximum(safe_prev_total[:, None], 1e-12),
            0.0
        )
        eff0 = (wts0 * yld[safe_idx]).sum(axis=1)
        safe_yld_eff[:, 0] = np.where(np.isfinite(eff0), eff0, 0.0)

    shocks  = gaussian_copula_t_draws(n_sims, years, mu, sig, rho, df=5, seed=seed)
    overlay = build_overlay(years, n_assets, growth_idx, regime)

    cascade_idx = list(safe_idx) + list(growth_idx)  # drain safe first

    for t in range(1, years + 1):
        prev    = values[:, t - 1, :]
        total_r = shocks[:, t - 1, :]
        price_r = total_r - yld  # separate income vs price
        if regime in ("Front", "Prolonged"):
            price_r = price_r + overlay[t - 1, :]

        price_r     = np.where(np.isfinite(price_r), price_r, 0.0)
        after_price = np.where(np.isfinite(prev * (1.0 + price_r)), prev * (1.0 + price_r), 0.0)
        income      = np.where(np.isfinite(prev * yld), prev * yld, 0.0)

        values[:, t, :] = after_price + income

        income_total = np.where(np.isfinite(income.sum(axis=1)), income.sum(axis=1), 0.0)
        principal_to_withdraw = np.maximum(0.0, annual_withdrawal - income_total)

        # cascade withdrawal
        for aidx in cascade_idx:
            bal     = np.where(np.isfinite(values[:, t, aidx]), values[:, t, aidx], 0.0)
            drawamt = np.minimum(principal_to_withdraw, bal)
            values[:, t, aidx] = bal - drawamt
            principal_to_withdraw = principal_to_withdraw - drawamt
            if not np.any(principal_to_withdraw > 1e-9):
                break

        values[:, t, :] = np.maximum(np.where(np.isfinite(values[:, t, :]), values[:, t, :], 0.0), 0.0)

        # track safe sleeve state
        safe_now = values[:, t, :][:, safe_idx]
        safe_sum = np.where(np.isfinite(np.sum(safe_now, axis=1)), np.sum(safe_now, axis=1), 0.0)
        safe_totals[:, t] = safe_sum
        with np.errstate(divide='ignore', invalid='ignore'):
            wts = np.where(safe_sum[:, None] > 0,
                           safe_now / np.maximum(safe_sum[:, None], 1e-12),
                           0.0)
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
    """Single-path liquidity statistic according to LIQ_METHOD."""
    T = safe_path.shape[0]
    liq_series = np.empty(T - 1, dtype=float)
    for t in range(1, T):
        liq_series[t - 1] = years_covered_forward(safe_path[t], yld_path[t], draw=draw, cap=cap)

    if method == "min_all":
        return float(np.min(liq_series))
    elif method == "median_all":
        return float(np.median(liq_series))
    elif method == "p10_yr1_8":
        hi = min(T - 1, 8)
        window = liq_series[:hi]
        return float(np.percentile(window, 10))
    # default fallback
    return float(np.min(liq_series))


def liquidity_distribution_stats(safe_totals,
                                 safe_yld_eff,
                                 annual_withdrawal=ANNUAL_WITHDRAWAL,
                                 cap=LIQ_CAP,
                                 method=LIQ_METHOD):
    """Compute per-path liquidity (using LIQ_METHOD) and print summary label correctly."""
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
def per_path_max_drawdown(series: np.ndarray) -> float:
    peak = np.maximum.accumulate(series)
    dd = (series - peak) / np.maximum(peak, 1e-12)
    return np.min(dd)


def compute_metrics(total_values,
                    safe_totals,
                    safe_yld_eff,
                    egsp_flags,
                    annual_withdrawal=ANNUAL_WITHDRAWAL,
                    min_acceptable_return=MIN_ACCEPTABLE_RETURN,
                    years=YEARS):
    """Compute blended-period metrics given simulated paths."""
    n_sims, _ = total_values.shape
    ruin = float(np.mean(np.any(total_values[:, 1:] <= 1e-6, axis=1)))

    # Liquidity per path via LIQ_METHOD
    liq_per_path = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        liq_per_path[i] = path_liquidity_stat(
            safe_totals[i], safe_yld_eff[i], draw=annual_withdrawal, method=LIQ_METHOD, cap=LIQ_CAP
        )
    liq_per_path = np.where(np.isfinite(liq_per_path), liq_per_path, 0.0)
    liquidity = float(np.median(liq_per_path))

    ok = total_values[:, -1] > 1e-6
    if np.sum(ok) < max(1, int(0.05 * n_sims)):
        # too many ruins => degrade returns components
        return {
            "Ruin": ruin,
            "Liquidity": max(0.0, liquidity),
            "Median_Return": -1.0,
            "Upside": -1.0,
            "CVaR": -1.0,
            "Sortino": 0.0,
            "MDD": -1.0,
            "Calmar": 0.0,
            "Early_Sale": float(np.mean(egsp_flags)),
        }

    tv_ok = total_values[ok]
    ann_log_ret = np.log(tv_ok[:, -1] / tv_ok[:, 0]) / years
    median_ret  = float(np.median(ann_log_ret))
    upside_p95  = float(np.percentile(ann_log_ret, 95))
    k           = max(1, int(0.05 * len(ann_log_ret)))
    cvar        = float(np.mean(np.sort(ann_log_ret)[:k]))
    MAR         = min_acceptable_return
    downside    = ann_log_ret[ann_log_ret < MAR]
    downside_dev = float(np.std(downside)) if downside.size > 0 else 0.0
    sortino     = float((np.mean(ann_log_ret) - MAR) / downside_dev) if downside_dev > 0 else 2.0

    mdd_vals = np.array([per_path_max_drawdown(tv_ok[i, :]) for i in range(tv_ok.shape[0])])
    mdd_avg  = float(np.mean(mdd_vals))
    calmar   = float(median_ret / -mdd_avg) if mdd_avg < 0 else 0.0

    return {
        "Ruin": ruin,
        "Liquidity": max(0.0, liquidity),
        "Median_Return": median_ret,
        "Upside": upside_p95,
        "CVaR": cvar,
        "Sortino": sortino,
        "MDD": mdd_avg,
        "Calmar": calmar,
        "Early_Sale": float(np.mean(egsp_flags)),
    }


# === Normalizations (clamped to [0, 100]) ======================================
def norm_ruin(r):
    r = float(np.clip(r, 0.0, 1.0))
    score = 100.0 * (1.0 - r / 0.20)
    return float(np.clip(score, 0.0, 100.0))

def norm_liquidity(years):
    y = float(max(0.0, years))
    if y <= 3.0:
        score = 90.0 * (y / 3.0)
    else:
        score = 90.0 + 10.0 * min((y - 3.0) / 2.0, 1.0)
    return float(np.clip(score, 0.0, 100.0))

def norm_linear_floor_ceiling(x, lo, hi):
    if x <= lo: return 0.0
    if x >= hi: return 100.0
    return 100.0 * (x - lo) / (hi - lo)

def norm_median(ret):
    return float(np.clip(norm_linear_floor_ceiling(ret, 0.03, 0.10), 0.0, 100.0))

def norm_upside(ret):
    r = float(max(ret, 1e-9))
    base, top = 0.06, 0.20
    score = 100.0 * min(max(np.log(max(r, base) / base) / np.log(top / base), 0.0), 1.0)
    return float(np.clip(score, 0.0, 100.0))

def norm_cvar(es):
    score = 100.0 * min(max((0.50 + es) / 0.40, 0.0), 1.0)
    return float(np.clip(score, 0.0, 100.0))

def norm_sortino(s):
    s_eff = float(max(s, 0.0))
    score = 100.0 * min(s_eff / 2.0, 1.0)
    return float(np.clip(score, 0.0, 100.0))

def norm_mdd(mdd):
    score = 100.0 * min(max((0.50 + mdd) / 0.40, 0.0), 1.0)
    return float(np.clip(score, 0.0, 100.0))

def norm_calmar(c):
    c_eff = float(max(c, 0.0))
    score = 100.0 * min(c_eff / 2.0, 1.0)
    return float(np.clip(score, 0.0, 100.0))

def norm_egsp(p):
    p_eff = float(np.clip(p, 0.0, 1.0))
    score = 100.0 * (1.0 - min(p_eff / 0.50, 1.0))
    return float(np.clip(score, 0.0, 100.0))

def norm_complexity(n_holdings):
    h = max(int(n_holdings), 5)
    score = 100.0 * (1.0 - np.log(h / 5.0) / np.log(25.0 / 5.0))
    return float(np.clip(score, 0.0, 100.0))


def period_score(metrics, n_holdings, weights):
    """Compute period score with drawdown-recovery coupling and clamps."""
    mdd_s = norm_mdd(metrics["MDD"])
    cal_s = norm_calmar(metrics["Calmar"])
    if metrics["MDD"] > -0.10:
        cal_s = min(cal_s, mdd_s)

    draw_rec = 0.5 * (mdd_s + cal_s)

    parts = {
        "Ruin":               norm_ruin(metrics["Ruin"]),
        "Liquidity":          norm_liquidity(metrics["Liquidity"]),
        "Median_Return":      norm_median(metrics["Median_Return"]),
        "Upside":             norm_upside(metrics["Upside"]),
        "CVaR":               norm_cvar(metrics["CVaR"]),
        "Sortino":            norm_sortino(metrics["Sortino"]),
        "Drawdown_Recovery":  draw_rec,
        "Early_Sale":         norm_egsp(metrics["Early_Sale"]),
        "Complexity":         norm_complexity(n_holdings),
    }

    score = sum(weights[k] * parts[k] for k in weights.keys())
    if metrics["Ruin"] > 0.25:
        score = min(score, 50.0)  # catastrophic cap

    score = float(np.clip(score, 0.0, 100.0))  # hard clamp
    return score, parts


def composite_score(metrics, n_holdings, WEIGHTS):
    s1, _ = period_score(metrics, n_holdings, WEIGHTS["Yrs1-4"])
    s2, _ = period_score(metrics, n_holdings, WEIGHTS["Yrs5-8"])
    s3, _ = period_score(metrics, n_holdings, WEIGHTS["Yrs9-12"])
    headline = float(np.clip((s1 + s2 + s3) / 3.0, 0.0, 100.0))
    return headline, {"Yrs1-4": round(s1, 1), "Yrs5-8": round(s2, 1), "Yrs9-12": round(s3, 1)}


# --------- Fast bootstrap (precompute per-path stats) ---------------------------
def _precompute_path_stats(total_values,
                           safe_totals,
                           safe_yld_eff,
                           egsp_flags,
                           annual_withdrawal=ANNUAL_WITHDRAWAL,
                           years=YEARS,
                           min_acceptable_return=MIN_ACCEPTABLE_RETURN):
    n_sims, _ = total_values.shape
    ruin_i = np.any(total_values[:, 1:] <= 1e-6, axis=1).astype(float)

    liq_i = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        liq_i[i] = path_liquidity_stat(
            safe_totals[i], safe_yld_eff[i],
            draw=annual_withdrawal, method=LIQ_METHOD, cap=LIQ_CAP
        )
    liq_i = np.where(np.isfinite(liq_i), liq_i, 0.0)

    survive = (total_values[:, -1] > 1e-6)
    with np.errstate(divide='ignore', invalid='ignore'):
        ann_log_ret_i = np.log(total_values[:, -1] / total_values[:, 0]) / years

    mdd_i = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        mdd_i[i] = per_path_max_drawdown(total_values[i, :]) if survive[i] else np.nan

    return {
        "ruin": ruin_i,
        "liq": liq_i,
        "egsp": egsp_flags.astype(float),
        "survive": survive,
        "ann_log_ret": ann_log_ret_i,
        "mdd": mdd_i,
        "MAR": min_acceptable_return,
    }


def bootstrap_se(total_values,
                 safe_totals,
                 safe_yld_eff,
                 egsp_flags,
                 n_holdings,
                 WEIGHTS,
                 n_resample=BOOTSTRAP_RESAMPLES):
    """SE of the Yrs1-4 period score via fast bootstrap."""
    rs = np.random.RandomState(SEED + 123)
    s = _precompute_path_stats(total_values, safe_totals, safe_yld_eff, egsp_flags)
    ruin_i, liq_i, egsp_i, survive, ann_log_ret, mdd_i, MAR = (
        s["ruin"], s["liq"], s["egsp"], s["survive"], s["ann_log_ret"], s["mdd"], s["MAR"]
    )
    n = len(ruin_i)
    scores = np.empty(n_resample, dtype=float)

    for b in range(n_resample):
        idx = rs.choice(n, size=n, replace=True)
        ruin_b = float(np.mean(ruin_i[idx]))
        liq_b  = float(np.median(liq_i[idx]))
        egsp_b = float(np.mean(egsp_i[idx]))

        surv_idx = idx[survive[idx]]
        if surv_idx.size < max(1, int(0.05 * n)):
            metrics_b = {
                "Ruin": ruin_b, "Liquidity": liq_b, "Median_Return": -1.0, "Upside": -1.0,
                "CVaR": -1.0, "Sortino": 0.0, "MDD": -1.0, "Calmar": 0.0, "Early_Sale": egsp_b
            }
        else:
            r      = ann_log_ret[surv_idx]
            med_r  = float(np.median(r))
            up95   = float(np.percentile(r, 95))
            k      = max(1, int(0.05 * r.size))
            cvar   = float(np.mean(np.sort(r)[:k]))
            dn     = r[r < MAR]
            dn_dev = float(np.std(dn)) if dn.size > 0 else 0.0
            sortino = float((np.mean(r) - MAR) / dn_dev) if dn_dev > 0 else 2.0
            mdd_b   = float(np.nanmean(mdd_i[surv_idx]))
            calmar  = float(med_r / -mdd_b) if mdd_b < 0 else 0.0

            metrics_b = {
                "Ruin": ruin_b, "Liquidity": liq_b, "Median_Return": med_r, "Upside": up95,
                "CVaR": cvar, "Sortino": sortino, "MDD": mdd_b, "Calmar": calmar, "Early_Sale": egsp_b
            }

        # SE based on the *front* period weights by spec
        s_front, _ = period_score(metrics_b, n_holdings, WEIGHTS["Yrs1-4"])
        scores[b] = s_front

    return float(np.std(scores))


# --------------------------- Public Scoring API --------------------------------
def score_portfolio(name, w, assets, mu, sig, rho, yld, safe_idx, growth_idx):
    """
    Entry point called by live wires. Prints:
      - t=0 safe-liquidity snapshot
      - Base-regime liquidity distribution summary with correct LIQ_METHOD label
      - Headline score + subs + blended metrics
    """
    # period weights (left here to keep engine self-contained)
    WEIGHTS = {
        "Yrs1-4":  {"Ruin":0.20, "Liquidity":0.25, "Median_Return":0.125, "Upside":0.125,
                    "CVaR":0.10, "Sortino":0.075, "Drawdown_Recovery":0.10, "Early_Sale":0.05, "Complexity":0.025},
        "Yrs5-8":  {"Ruin":0.175,"Liquidity":0.225,"Median_Return":0.15,  "Upside":0.15,
                    "CVaR":0.10, "Sortino":0.075, "Drawdown_Recovery":0.10, "Early_Sale":0.05, "Complexity":0.025},
        "Yrs9-12": {"Ruin":0.15, "Liquidity":0.20, "Median_Return":0.175, "Upside":0.20,
                    "CVaR":0.10, "Sortino":0.075, "Drawdown_Recovery":0.10, "Early_Sale":0.05, "Complexity":0.025},
    }

    n_hold = count_holdings(w)

    # t=0 snapshot (safe-only forward drain)
    safe_w = float(np.sum(w[safe_idx]))
    safe_init_balance = INITIAL_PORTFOLIO_VALUE * safe_w
    safe_eff_yield = float(np.sum((w[safe_idx] / max(safe_w, 1e-12)) * yld[safe_idx])) if safe_w > 0 else 0.0
    liq_t0 = years_covered_forward(safe_init_balance, safe_eff_yield, draw=ANNUAL_WITHDRAWAL, cap=LIQ_CAP)
    print(f"  SNAPSHOT Liquidity(t=0): safe_w={safe_w:.3f}, safe_init=${safe_init_balance:,.0f}, "
          f"yld_eff={safe_eff_yield:.2%}, years≈{liq_t0:.2f}")

    blended = None
    base_totals = base_safe = base_yld = base_egsp = None

    # Run regimes & blend metrics
    for regime, weight in STRESS_BLEND.items():
        totals, safe_totals, safe_yld_eff, egsp_flags = simulate_paths(
            INITIAL_PORTFOLIO_VALUE, w, mu, sig, rho, yld, YEARS, ANNUAL_WITHDRAWAL,
            safe_idx, growth_idx, assets, n_sims=N_SIM, seed=SEED, regime=regime
        )
        if regime == "Base" and DEBUG_LIQ_STATS:
            base_totals, base_safe, base_yld, base_egsp = totals, safe_totals, safe_yld_eff, egsp_flags
            _ = liquidity_distribution_stats(base_safe, base_yld, ANNUAL_WITHDRAWAL, cap=LIQ_CAP, method=LIQ_METHOD)

        m = compute_metrics(totals, safe_totals, safe_yld_eff, egsp_flags,
                            ANNUAL_WITHDRAWAL, MIN_ACCEPTABLE_RETURN, YEARS)
        if blended is None:
            blended = {k: 0.0 for k in m.keys()}
        for k, v in m.items():
            blended[k] += weight * v

    # Headline / subs
    headline, subs = composite_score(blended, n_hold, WEIGHTS)

    # Bootstrap SE on front period
    se = bootstrap_se(base_totals, base_safe, base_yld, base_egsp, n_holdings=n_hold, WEIGHTS=WEIGHTS)

    # Output
    print(f"\n=== {name} ===")
    print(f"Holdings: {n_hold} | Headline Score: {round(headline,1)} ±4 (SE≈{se:.2f})")
    print(f"Period sub-scores: {{ Yrs1-4: {round(subs['Yrs1-4'],1)}, "
          f"Yrs5-8: {round(subs['Yrs5-8'],1)}, Yrs9-12: {round(subs['Yrs9-12'],1)} }}")
    print("Blended Metrics (Base 60%, Front 20%, Prol 20%):")
    print(
        f" Ruin={blended['Ruin']:.3f} Liquidity={blended['Liquidity']:.2f}y "
        f"MedRet={blended['Median_Return']:.3f} Upside={blended['Upside']:.3f} "
        f"CVaR={blended['CVaR']:.3f} Sortino={blended['Sortino']:.2f} "
        f"MDD={blended['MDD']:.3f} Calmar={blended['Calmar']:.2f} EGSP={blended['Early_Sale']:.3f}"
    )